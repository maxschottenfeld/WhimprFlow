//! Local speech-to-text via whisper.cpp (whisper-rs), implementing
//! [`whimpr_core::AsrEngine`]. Expects 16 kHz mono f32 samples.

use std::path::Path;

use whimpr_core::asr::{AsrCaps, AsrEngine, AsrEngineId, Transcript};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper's ceiling for the decoder's text context is `n_text_ctx`, and it will
/// only ever take half of that as prompt (`n_text_ctx / 2`). For every model in the
/// family `n_text_ctx` is 448, so the usable prompt budget is 224 tokens — the
/// number OpenAI's own prompting guide quotes. Anything past that is silently
/// dropped by whisper.cpp, which is precisely the kind of quiet truncation this
/// project keeps getting bitten by, so we truncate deliberately and say so.
pub const MAX_PROMPT_TOKENS: usize = 224;

/// Decoding options for a single transcription.
///
/// `Default` reproduces exactly what the app shipped before this struct existed, so
/// `transcribe()` is unchanged behaviour.
#[derive(Debug, Clone)]
pub struct RunOpts {
    /// Whisper `initial_prompt` — biases decoding toward particular spellings.
    /// Sanitized and trimmed to [`MAX_PROMPT_TOKENS`] before use.
    pub prompt: Option<String>,
    /// Force whisper to emit a single segment.
    ///
    /// Upstream added this to stop *short* clips being transcribed twice, and it
    /// works for that. The open question was whether it also truncates clips longer
    /// than whisper's 30-second window, since `single_segment` makes the decoder
    /// stop after one window's worth of output. Exposed here so the harness can
    /// answer that against real audio instead of by reading whisper.cpp.
    pub single_segment: bool,
}

impl Default for RunOpts {
    fn default() -> Self {
        Self {
            prompt: None,
            single_segment: true,
        }
    }
}

/// A loaded whisper model ready to transcribe utterances.
pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    /// Load a GGML/GGUF whisper model from `model_path`.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))?;
        Ok(Self { ctx })
    }

    /// Count how many whisper tokens `text` occupies, for prompt-budget decisions.
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        // `tokenize` needs an upper bound; MAX_PROMPT_TOKENS * 4 is far more than any
        // prompt we would consider sending and keeps the failure mode "returns Err"
        // rather than "silently truncates".
        self.ctx
            .tokenize(text, MAX_PROMPT_TOKENS * 4)
            .map(|t| t.len())
            .map_err(|e| anyhow::anyhow!("whisper tokenize: {e}"))
    }

    /// Trim `text` from the FRONT until it fits inside [`MAX_PROMPT_TOKENS`].
    ///
    /// Front, not back, because whisper weights later prompt tokens more heavily —
    /// so the tail is the part worth keeping. Callers should therefore order an
    /// initial prompt **least-important first**; whatever survives truncation is
    /// also what the decoder leans on hardest.
    ///
    /// Returns `None` when even a single word will not fit, which should not happen
    /// for real vocabulary but is not worth panicking over.
    pub fn fit_prompt(&self, text: &str) -> Option<String> {
        let text = sanitize_prompt(text);
        if text.is_empty() {
            return None;
        }
        if self.count_tokens(&text).ok()? <= MAX_PROMPT_TOKENS {
            return Some(text);
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        // Drop leading words until it fits. Prompts are short enough (a few hundred
        // words at most) that walking them is cheaper than being clever.
        for start in 1..words.len() {
            let candidate = words[start..].join(" ");
            if self.count_tokens(&candidate).ok()? <= MAX_PROMPT_TOKENS {
                return Some(candidate);
            }
        }
        None
    }

    /// Transcribe with an optional whisper `initial_prompt`.
    ///
    /// The prompt biases decoding toward particular spellings — it is the supported
    /// way to make whisper get names and jargon right at the acoustic level, rather
    /// than repairing them afterwards. It is a *bias*, not a constraint: whisper is
    /// free to ignore it, and it never guarantees a spelling.
    ///
    /// The prompt is sanitized and budget-trimmed here rather than at the call site,
    /// so no caller can accidentally overflow the 224-token window or hand
    /// `set_initial_prompt` a null byte (which panics).
    pub fn transcribe_with_prompt(
        &self,
        pcm16k: &[f32],
        prompt: Option<&str>,
    ) -> anyhow::Result<Transcript> {
        self.transcribe_with_opts(
            pcm16k,
            &RunOpts {
                prompt: prompt.map(str::to_string),
                ..Default::default()
            },
        )
    }

    /// Full-control entry point. The app uses [`Self::transcribe_with_prompt`]; this
    /// exists so the offline harness can A/B decoding options (notably
    /// `single_segment`) against real audio without a second code path.
    pub fn transcribe_with_opts(
        &self,
        pcm16k: &[f32],
        opts: &RunOpts,
    ) -> anyhow::Result<Transcript> {
        let fitted = opts.prompt.as_deref().and_then(|p| self.fit_prompt(p));
        self.run(pcm16k, fitted.as_deref(), opts.single_segment)
    }

    fn run(
        &self,
        pcm16k: &[f32],
        prompt: Option<&str>,
        single_segment: bool,
    ) -> anyhow::Result<Transcript> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("whisper create_state: {e}"))?;

        // Beam search rather than greedy. whisper.cpp defaults to greedy (best_of 1),
        // but OpenAI's reference implementation uses beam search — it evaluates several
        // candidate token paths and picks the best overall sequence instead of
        // committing to the highest-probability token at each step. That is exactly the
        // class of error that produces plausible-but-wrong words. beam_size 5 matches
        // the reference default; the extra cost is negligible on push-to-talk clips.
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 0.0,
        });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        // Do not carry decoder context between calls. Each dictation is independent,
        // and carrying context lets one bad transcription poison the next. This does
        // NOT disable the initial prompt below: whisper.cpp clears the carried-over
        // context first and prepends the initial prompt afterwards, so the two are
        // independent. (Verified empirically, not just read — see OVERNIGHT-LOG.)
        params.set_no_context(true);
        // Push-to-talk utterances are normally one short clip, not long-form audio.
        // Without this, whisper.cpp can split a short clip into multiple internal
        // segments that repeat the same words — which then get concatenated below,
        // producing the sentence twice. See RunOpts::single_segment for the caveat
        // that matters on clips longer than whisper's 30-second window.
        params.set_single_segment(single_segment);
        if let Some(p) = prompt {
            params.set_initial_prompt(p);
        }

        state
            .full(params, pcm16k)
            .map_err(|e| anyhow::anyhow!("whisper full: {e}"))?;

        let n = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("whisper n_segments: {e}"))?;
        let mut text = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                text.push_str(&seg);
            }
        }

        Ok(Transcript {
            text: text.trim().to_string(),
            confidence: None,
        })
    }
}

/// Strip characters that would break `set_initial_prompt` or waste prompt budget.
///
/// Null bytes make `set_initial_prompt` panic (it builds a `CString`), so they are
/// removed rather than trusted. Newlines and runs of whitespace are collapsed
/// because they cost tokens and buy nothing.
fn sanitize_prompt(text: &str) -> String {
    text.chars()
        .filter(|c| *c != '\0')
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl AsrEngine for WhisperEngine {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::WhisperCpp
    }

    fn caps(&self) -> AsrCaps {
        AsrCaps {
            supports_streaming: false,
        }
    }

    fn transcribe(&self, pcm16k: &[f32]) -> anyhow::Result<Transcript> {
        self.transcribe_with_opts(pcm16k, &RunOpts::default())
    }
}
