//! Local-LLM cleanup worker.
//!
//! Loads a GGUF instruction model once, then serves one request per line of stdin:
//! `{"system": "...", "user": "..."}` → `{"text": "..."}` on stdout. The WhimprFlow
//! app spawns this and keeps it warm so cleanup is fast and fully offline.
//!
//! Usage: `whimpr-llm-worker <model.gguf>` (or WHIMPR_LLM_MODEL env var).

use std::io::{BufRead, Write};
use std::num::NonZeroU32;

use anyhow::Context as _;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::TokenToStringError;
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Msg {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Request {
    /// Full multi-turn message list (system + few-shot + user). Preferred.
    #[serde(default)]
    messages: Vec<Msg>,
    /// Back-compat single-turn form, used only when `messages` is empty.
    #[serde(default)]
    system: String,
    #[serde(default)]
    user: String,
    #[serde(default = "default_max")]
    max_tokens: i32,
}
fn default_max() -> i32 {
    400
}

#[derive(Serialize)]
struct Response {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("WHIMPR_LLM_MODEL").ok())
        .context("model path required (argv[1] or WHIMPR_LLM_MODEL)")?;

    let backend = LlamaBackend::init()?;
    // Offload everything to the Apple GPU (Metal) — capped by what fits.
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model {model_path}"))?;
    // Build the context ONCE and reuse it for every request. Creating it per
    // request meant each cleanup allocated a ~300 MB Metal compute buffer, compiled
    // the Metal pipelines, then freed it all again — pure overhead repeated on
    // every single dictation, and the dominant source of perceived latency.
    // The KV cache is cleared per request instead, which is what actually needs
    // resetting between independent utterances.
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(4096));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("failed to create llama context")?;
    eprintln!("[llm-worker] model loaded, ready");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Request>(&line) {
            Ok(req) => match generate(&mut ctx, &model, &req) {
                Ok(text) => Response { text, error: None },
                Err(e) => Response {
                    text: String::new(),
                    error: Some(e.to_string()),
                },
            },
            Err(e) => Response {
                text: String::new(),
                error: Some(format!("bad request: {e}")),
            },
        };
        serde_json::to_writer(&mut stdout, &resp)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

fn generate(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    req: &Request,
) -> anyhow::Result<String> {
    // Qwen2.5 ChatML template. Prefer the full multi-turn message list (few-shot
    // demonstrations drive the newline/list/self-correction behavior); fall back
    // to the legacy single system+user pair.
    let mut prompt = String::new();
    if req.messages.is_empty() {
        prompt.push_str(&format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
            req.system, req.user
        ));
    } else {
        for m in &req.messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
        }
    }
    prompt.push_str("<|im_start|>assistant\n");

    // Independent utterance: drop the previous request's tokens so this prompt
    // starts from an empty cache in the reused context.
    ctx.clear_kv_cache();

    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let n_prompt = tokens.len() as i32;

    let mut batch = LlamaBatch::new(4096, 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::greedy();
    let mut n_cur = batch.n_tokens();
    // Accumulate raw BYTES, not per-token Strings. A multi-byte character the
    // vocab has no single token for is emitted as byte-fallback tokens, so its
    // UTF-8 sequence is split across several tokens. `token_to_str` builds a
    // fresh UTF-8 decoder per call and discards it, so each fragment is an
    // incomplete or invalid sequence and the character is dropped SILENTLY —
    // no error, no replacement char, just a hole in the output. Measured
    // 2026-08-17: an echo of "∮ ∑ ∫ ∞ θ ζ π γ ε δ ₀ ² − √ Γ ∂" came back as
    // "∮      π γ ε δ ₀ ² −  Γ" on BOTH the 1.5b and the 4b — identical losses
    // from two different models, which is what proves it is this decode path
    // and not model capability. ∑ ∫ ∞ θ ζ √ ∂ ñ are the casualties; every
    // single-token glyph (∮ π γ ε δ ₀ ² − Γ) survived. Ordinary English is
    // unaffected — em dash, ellipsis, curly quotes, é, °, ½, → all round-trip.
    let mut out_bytes: Vec<u8> = Vec::new();
    let limit = n_prompt + req.max_tokens;

    while n_cur <= limit {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        // The 8-byte guess is what the crate's own helper uses, and it is too
        // small for some pieces — a bare call returns InsufficientBufferSpace(-n)
        // and, with `?`, kills the whole request. Retry at the size it asks for.
        let piece = match model.token_to_piece_bytes(token, 8, true, None) {
            Err(TokenToStringError::InsufficientBufferSpace(n)) => model.token_to_piece_bytes(
                token,
                usize::try_from(-n).expect("buffer size is positive"),
                true,
                None,
            )?,
            other => other?,
        };
        out_bytes.extend_from_slice(&piece);
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }
    // Lossy on purpose: a genuinely malformed byte should cost one replacement
    // char, never the whole dictation. Decoding once at the end is what lets a
    // split sequence reassemble.
    Ok(String::from_utf8_lossy(&out_bytes).trim().to_string())
}
