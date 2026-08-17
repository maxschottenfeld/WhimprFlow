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
//! By default the LLM call is not made: it needs the `whimpr-llm-worker` sidecar
//! and seconds per call, and most questions this harness answers are upstream of
//! it. What is always reported is whether the `needs_cleanup()` gate would have
//! fired, which is the part that decides whether the LLM runs at all.
//!
//! `--math` opts *in* to a real model call and runs the spoken-math stage
//! (`whimpr_core::mathfmt`, G2). It exists because that stage cannot be judged
//! any other way: its output is notation, and whether notation is *correct* is
//! not something a gate or a word count can decide — 🔴 a correct dense
//! conversion is far SHORTER than its input, so every length-based score both
//! rejects good output and passes bad. Someone has to read the text. This prints
//! it, next to the transcript it came from, so that reading takes seconds.
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
//!       [--math]              # run the math stage (spawns the LLM worker)
//!       [--format unicode|latex|both]   # math notation; default unicode
//!       [--llm-model <path>]  # GGUF for the worker; default = the app's choice
//!       [--text "..."]        # skip audio entirely, feed this transcript in
//! ```
//!
//! `--text` takes the microphone *and* whisper out of the loop, which matters
//! when the question is about the math stage rather than about recognition:
//! `say` is not Max, and a synthetic-voice mis-recognition upstream would
//! otherwise be scored as a conversion failure.

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

/// Which notation the math stage should be exercised in. `Both` runs each input
/// twice so the two can be compared on the same transcript — the only fair way
/// to compare them, since ASR variation would otherwise be scored as a format
/// difference.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FormatArg {
    Unicode,
    Latex,
    Both,
}

impl FormatArg {
    fn formats(self) -> &'static [whimpr_core::mathfmt::MathFormat] {
        use whimpr_core::mathfmt::MathFormat::*;
        match self {
            FormatArg::Unicode => &[Unicode],
            FormatArg::Latex => &[Latex],
            FormatArg::Both => &[Unicode, Latex],
        }
    }
}

struct Args {
    wavs: Vec<PathBuf>,
    /// Literal transcripts supplied with `--text`, bypassing audio and ASR.
    texts: Vec<String>,
    model: Option<PathBuf>,
    prompt: Option<String>,
    repeat: usize,
    quiet: bool,
    /// Run the spoken-math stage. Opt-in because it spawns the worker process and
    /// costs seconds per input.
    math: bool,
    format: FormatArg,
    llm_model: Option<PathBuf>,
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
    let mut texts = Vec::new();
    let mut model = None;
    let mut prompt = None;
    let mut repeat = 1usize;
    let mut quiet = false;
    let mut single_segment = true;
    let mut audio_ctx: Option<i32> = None;
    let mut trim = true;
    let mut math = false;
    let mut format = FormatArg::Unicode;
    let mut llm_model = None;

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
            "--text" => texts.push(
                it.next().ok_or_else(|| anyhow::anyhow!("--text needs a transcript"))?,
            ),
            "--math" => math = true,
            "--format" => {
                let v = it.next().ok_or_else(|| anyhow::anyhow!("--format needs a value"))?;
                format = match v.as_str() {
                    "unicode" => FormatArg::Unicode,
                    "latex" => FormatArg::Latex,
                    "both" => FormatArg::Both,
                    other => anyhow::bail!("--format must be unicode, latex or both (got {other})"),
                };
            }
            "--llm-model" => {
                llm_model = Some(PathBuf::from(
                    it.next().ok_or_else(|| anyhow::anyhow!("--llm-model needs a path"))?,
                ))
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
    if wavs.is_empty() && texts.is_empty() {
        anyhow::bail!(
            "usage: whimpr-harness <file.wav> [...] [--text \"...\"] [--model P] [--prompt T] \
             [--repeat N] [--math [--format unicode|latex|both] [--llm-model P]]"
        );
    }
    if !texts.is_empty() && !math {
        anyhow::bail!("--text only has something to run without --math if you also want ASR; pass --math");
    }
    Ok(Args {
        wavs,
        texts,
        model,
        prompt,
        repeat,
        quiet,
        single_segment,
        audio_ctx,
        trim,
        math,
        format,
        llm_model,
    })
}

/// Resolve the GGUF for the worker, mirroring `src-tauri/src/local_llm.rs`'s
/// `model_path()`. Hand-synced, like `default_model_path` above and for the same
/// reason: if the app's preference list changes and this one does not, the
/// harness measures a different model than the app runs and every number it
/// prints is a lie. The banner prints what actually resolved.
fn default_llm_model_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = PathBuf::from(home).join("Library/Application Support/WhimprFlow Dev/models");
    for name in ["qwen3-4b-instruct-2507-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"] {
        let p = dir.join(name);
        if p.exists() {
            return p;
        }
    }
    dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf")
}

/// Locate the built worker binary. Release first (that is what the app bundles
/// and what the latency numbers must come from); debug only as a fallback, and
/// loudly, because a debug llama.cpp is slow enough to make timings meaningless.
fn worker_bin_path() -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let root = PathBuf::from(home).join("WhimprFlow");
    let release = root.join("target/release/whimpr-llm-worker");
    if release.exists() {
        return Ok(release);
    }
    let debug = root.join("target/debug/whimpr-llm-worker");
    if debug.exists() {
        eprintln!(
            "⚠ using the DEBUG worker at {} — timings from it are meaningless. \
             Build with `cargo build --release -p whimpr-llm-worker`.",
            debug.display()
        );
        return Ok(debug);
    }
    anyhow::bail!(
        "no whimpr-llm-worker binary found (looked in target/release and target/debug). \
         Run: cargo build --release -p whimpr-llm-worker"
    )
}

/// Run one transcript through the math stage and print the result for a human to
/// read. Returns the elapsed milliseconds per format, for the timing summary.
///
/// Deliberately prints the transcript above the output every time, even when it
/// is unchanged from the line printed a moment earlier: the only question this
/// answers is "did the conversion of THIS sentence come out right", and that is
/// unanswerable without both halves side by side.
fn run_math(
    worker: &mut whimpr_core::worker::LocalWorker,
    transcript: &str,
    args: &Args,
) -> Vec<(whimpr_core::mathfmt::MathFormat, u128)> {
    let mut timings = Vec::new();
    for &fmt in args.format.formats() {
        let msgs = whimpr_core::mathfmt::build_messages(transcript, fmt);
        let t = Instant::now();
        // `request` turns the worker's `error` field into a real Err. Reading
        // only `text` would render a hard failure as an empty string, which looks
        // exactly like a bad model — that mistake nearly produced a false quality
        // verdict on 2026-08-17.
        let res = worker.request(&msgs, 400);
        let ms = t.elapsed().as_millis();
        let name = match fmt {
            whimpr_core::mathfmt::MathFormat::Unicode => "unicode",
            whimpr_core::mathfmt::MathFormat::Latex => "latex",
        };
        match res {
            Ok(raw) => {
                let out = whimpr_core::mathfmt::finalize(&raw);
                if out.is_empty() {
                    println!("   math[{name}]: {ms} ms  ⚠ EMPTY OUTPUT (call succeeded — this is the model, not an error)");
                } else {
                    println!("   math[{name}]: {ms} ms");
                    println!("     {out}");
                }
            }
            Err(e) => println!("   math[{name}]: {ms} ms  🔴 CALL FAILED: {e}"),
        }
        timings.push((fmt, ms));
    }
    timings
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let model = args.model.clone().unwrap_or_else(default_model_path);

    // Spawn the LLM worker first when it is wanted, so a missing binary or model
    // fails immediately instead of after a whisper load and a run of fixtures.
    let mut worker = if args.math {
        let bin = worker_bin_path()?;
        let gguf = args.llm_model.clone().unwrap_or_else(default_llm_model_path);
        if !gguf.exists() {
            anyhow::bail!("llm model not found at {}", gguf.display());
        }
        if !args.quiet {
            println!("llm:    {}", gguf.display());
            println!("worker: {}", bin.display());
        }
        // Deliberately NOT timed. `spawn` returns as soon as the child process
        // exists; the several seconds of GGUF load happen inside it, and the
        // first request is what waits for them. A "load: 0 ms" line here would be
        // measuring the fork, not the model — an instrument that reports a number
        // for something it did not observe is worse than one that stays quiet.
        // The first per-input timing below absorbs the load; treat it as a warmup.
        Some(whimpr_core::worker::LocalWorker::spawn(&bin, &gguf)?)
    } else {
        None
    };

    if !args.quiet {
        println!("model:  {}", model.display());
        match &args.prompt {
            Some(p) => println!("prompt: {:?} ({} chars)", p, p.len()),
            None => println!("prompt: <none>"),
        }
        println!("single_segment: {}", args.single_segment);
        println!();
    }

    let mut math_ms: Vec<(whimpr_core::mathfmt::MathFormat, u128)> = Vec::new();

    // `--text` inputs skip audio and ASR entirely, so they need no whisper model
    // and are run before it is loaded.
    for text in &args.texts {
        println!("── --text");
        println!("   text:     {text:?}");
        if let Some(w) = worker.as_mut() {
            math_ms.extend(run_math(w, text, &args));
        }
        println!();
    }

    let mut failures = 0usize;
    if !args.wavs.is_empty() {
        let t = Instant::now();
        let engine = whimpr_asr::WhisperEngine::load(&model)?;
        let load_ms = t.elapsed().as_millis();
        if !args.quiet {
            // Model load is a once-per-app-launch cost, not a per-dictation one. Printed
            // separately so it is never mistaken for part of the dictation budget.
            println!("model load: {load_ms} ms (once per launch, not per dictation)\n");
        }

        for wav in &args.wavs {
            for run in 0..args.repeat {
                match run_one(&engine, wav, &args, run, worker.as_mut(), &mut math_ms) {
                    Ok(()) => {}
                    Err(e) => {
                        failures += 1;
                        println!("── {} ── ERROR: {e}", wav.display());
                    }
                }
            }
        }
    }

    if !math_ms.is_empty() {
        print_math_summary(&math_ms);
    }
    if failures > 0 {
        anyhow::bail!("{failures} fixture(s) failed");
    }
    Ok(())
}

/// Median latency per format. Median rather than mean because the distribution is
/// long-tailed and one slow first call would otherwise dominate; this is the same
/// statistic the 2026-08-17 sweep reported, so the numbers stay comparable.
fn print_math_summary(samples: &[(whimpr_core::mathfmt::MathFormat, u128)]) {
    use whimpr_core::mathfmt::MathFormat::*;
    println!("── math latency");
    for (fmt, name) in [(Unicode, "unicode"), (Latex, "latex")] {
        let mut v: Vec<u128> = samples.iter().filter(|(f, _)| *f == fmt).map(|(_, ms)| *ms).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        let median = v[v.len() / 2];
        println!("   {name:<8} n={:<3} median {median} ms   min {} max {}", v.len(), v[0], v[v.len() - 1]);
    }
}

fn run_one(
    engine: &whimpr_asr::WhisperEngine,
    wav: &Path,
    args: &Args,
    run: usize,
    worker: Option<&mut whimpr_core::worker::LocalWorker>,
    math_ms: &mut Vec<(whimpr_core::mathfmt::MathFormat, u128)>,
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
    // The math stage runs on the same text the app would hand it: post-processed
    // whisper output, before any cleanup LLM. Cleanup is not in this path because
    // `needs_cleanup()` is false on all ten math fixtures anyway (measured, not
    // read), so on math dictation the two stages never actually compete.
    if let Some(w) = worker {
        math_ms.extend(run_math(w, &out, args));
    }
    println!();
    Ok(())
}
