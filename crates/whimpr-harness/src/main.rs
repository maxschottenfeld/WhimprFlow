//! Offline dictation-pipeline harness.
//!
//! Runs one or more `.wav` fixtures through the same stages a live dictation goes
//! through, minus the microphone and the paste:
//!
//! ```text
//!   wav on disk
//!     -> decode to mono f32           (hound; mirrors whimpr-audio's downmix)
//!     -> whimpr_audio::resample_to_16k
//!     -> whimpr_asr::WhisperEngine::transcribe[_with_prompt]
//!     -> whimpr_core::cleanup::pre_normalize_layout / needs_cleanup / post_process
//! ```
//!
//! The LLM cleanup call itself is deliberately not made here: it needs the
//! `whimpr-llm-worker` sidecar and ~2s per call, and the questions this harness
//! exists to answer are all upstream of it. What *is* reported is whether the
//! `needs_cleanup()` gate would have fired, which is the part that decides whether
//! the LLM runs at all.
//!
//! Usage:
//! ```text
//!   whimpr-harness <file.wav> [more.wav ...]
//!       [--model <path>]      # default: same preference order as the dev app
//!       [--prompt <text>]     # whisper initial_prompt (decoding bias)
//!       [--prompt-file <path>]
//!       [--repeat <n>]        # run each fixture n times (timing stability)
//!       [--quiet]             # transcript + timing only, no banner
//!       [--no-trim]           # skip the leading-silence trim (pre-fix behaviour)
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

fn support_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join("Library/Application Support/WhimprFlow Dev")
}

/// Mirrors `hotkey.rs::model_path()`. Kept in sync by hand; if the app's list
/// changes and this one doesn't, the harness measures a different model than the
/// app runs, which would make every number here a lie. The banner prints the
/// resolved path for exactly that reason.
fn default_model_path() -> PathBuf {
    let dir = support_dir().join("models");
    for name in [
        "ggml-small.en-dev.bin",
        "ggml-large-v3-turbo.bin",
        "ggml-large-v3-turbo-q5_0.bin",
        "ggml-medium.en.bin",
        "ggml-small.en.bin",
        "ggml-base.en.bin",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return p;
        }
    }
    dir.join("ggml-base.en.bin")
}

/// Decode a wav to mono f32 in [-1, 1], returning (samples, sample_rate).
///
/// Downmixes multi-channel input by averaging, which is what `whimpr-audio`'s
/// capture callback does with the live stream.
fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("read float samples: {e}"))?,
        hound::SampleFormat::Int => {
            // Normalize by the full-scale value for the declared bit depth.
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("read int samples: {e}"))?
                .into_iter()
                .map(|s| s as f32 / scale)
                .collect()
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Peak and RMS of a buffer, as a cheap sanity check that a fixture actually
/// contains what it is supposed to (a "silence" fixture that is not silent, or a
/// speech fixture that decoded to zeros, would otherwise look like an ASR result).
fn levels(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    (peak, rms)
}

struct Args {
    wavs: Vec<PathBuf>,
    model: Option<PathBuf>,
    prompt: Option<String>,
    repeat: usize,
    quiet: bool,
    /// `--no-single-segment` turns off whisper's `single_segment` flag, so a >30s
    /// clip can emit more than one segment. This is the B2 truncation experiment.
    single_segment: bool,
    audio_ctx: Option<i32>,
    /// `--no-trim` skips the leading-silence trim, reproducing the pre-fix
    /// behaviour. Keep this: it is how the truncation bug stays demonstrable.
    trim: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut wavs = Vec::new();
    let mut model = None;
    let mut prompt = None;
    let mut repeat = 1usize;
    let mut quiet = false;
    let mut single_segment = true;
    let mut audio_ctx: Option<i32> = None;
    let mut trim = true;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                model = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow::anyhow!("--model needs a path"))?,
                ))
            }
            "--prompt" => {
                prompt = Some(it.next().ok_or_else(|| anyhow::anyhow!("--prompt needs text"))?)
            }
            "--prompt-file" => {
                let p = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--prompt-file needs a path"))?;
                prompt = Some(std::fs::read_to_string(&p)?.trim().to_string());
            }
            "--repeat" => {
                repeat = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--repeat needs a number"))?
                    .parse()?
            }
            "--quiet" => quiet = true,
            "--no-trim" => trim = false,
            "--no-single-segment" => single_segment = false,
            "--audio-ctx" => {
                audio_ctx = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--audio-ctx needs a number"))?
                        .parse()?,
                )
            }
            other if other.starts_with("--") => {
                anyhow::bail!("unknown flag {other}")
            }
            other => wavs.push(PathBuf::from(other)),
        }
    }
    if wavs.is_empty() {
        anyhow::bail!("usage: whimpr-harness <file.wav> [...] [--model P] [--prompt T] [--repeat N]");
    }
    Ok(Args { wavs, model, prompt, repeat, quiet, single_segment, audio_ctx, trim })
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let model = args.model.clone().unwrap_or_else(default_model_path);

    if !args.quiet {
        println!("model:  {}", model.display());
        match &args.prompt {
            Some(p) => println!("prompt: {:?} ({} chars)", p, p.len()),
            None => println!("prompt: <none>"),
        }
        println!("single_segment: {}", args.single_segment);
        println!();
    }

    let t = Instant::now();
    let engine = whimpr_asr::WhisperEngine::load(&model)?;
    let load_ms = t.elapsed().as_millis();
    if !args.quiet {
        // Model load is a once-per-app-launch cost, not a per-dictation one. Printed
        // separately so it is never mistaken for part of the dictation budget.
        println!("model load: {load_ms} ms (once per launch, not per dictation)\n");
    }

    let mut failures = 0usize;
    for wav in &args.wavs {
        for run in 0..args.repeat {
            match run_one(&engine, wav, &args, run) {
                Ok(()) => {}
                Err(e) => {
                    failures += 1;
                    println!("── {} ── ERROR: {e}", wav.display());
                }
            }
        }
    }
    if failures > 0 {
        anyhow::bail!("{failures} fixture(s) failed");
    }
    Ok(())
}

fn run_one(
    engine: &whimpr_asr::WhisperEngine,
    wav: &Path,
    args: &Args,
    run: usize,
) -> anyhow::Result<()> {
    let repeat = args.repeat;
    let label = if repeat > 1 {
        format!("{} (run {}/{})", wav.display(), run + 1, repeat)
    } else {
        wav.display().to_string()
    };
    println!("── {label}");

    let (samples, rate) = load_wav(wav)?;
    let secs = samples.len() as f32 / rate as f32;
    let (peak, rms) = levels(&samples);
    println!("   input:    {secs:.2}s @ {rate} Hz, {} samples, peak {peak:.3} rms {rms:.4}", samples.len());

    let t = Instant::now();
    // Mirrors the app's order: trim the leading silence at the source rate, then
    // resample. `--no-trim` restores the old behaviour, which is what makes the
    // leading-silence bug demonstrable rather than merely fixed.
    let trimmed: &[f32] = if args.trim {
        whimpr_audio::trim_leading_silence(&samples, rate)
    } else {
        &samples
    };
    let cut_s = (samples.len() - trimmed.len()) as f32 / rate.max(1) as f32;
    let pcm = whimpr_audio::resample_to_16k(trimmed, rate);
    let resample_ms = t.elapsed().as_millis();
    if cut_s > 0.0 {
        println!("   trim:     {cut_s:.2}s of leading silence removed");
    }

    let opts = whimpr_asr::RunOpts {
        prompt: args.prompt.clone(),
        single_segment: args.single_segment,
        audio_ctx: args.audio_ctx,
    };
    let t = Instant::now();
    let transcript = engine.transcribe_with_opts(&pcm, &opts)?;
    let asr_ms = t.elapsed().as_millis();
    let raw = transcript.text;

    // Same deterministic path clean_transcript() takes before it decides whether to
    // call the model at all.
    let raw_norm = whimpr_core::cleanup::pre_normalize_layout(&raw);
    let gated = whimpr_core::cleanup::needs_cleanup(&raw_norm);
    let out = whimpr_core::cleanup::post_process(&raw_norm);

    let words = raw.split_whitespace().count();
    println!("   resample: {resample_ms} ms   asr: {asr_ms} ms   ({:.1}x realtime)", secs * 1000.0 / asr_ms.max(1) as f32);
    println!("   gate:     needs_cleanup = {gated}{}", if gated { "  (LLM would run, ~2s)" } else { "  (LLM skipped)" });
    println!("   words:    {words}");
    println!("   text:     {out:?}");
    println!();
    Ok(())
}
