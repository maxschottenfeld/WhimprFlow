//! Finds and spawns the local-LLM worker (a separate process, so llama.cpp and
//! whisper.cpp never link into the same binary).
//!
//! The **protocol** — one JSON request per line over stdio — lives in
//! `whimpr_core::worker` so the app and the offline harness share a single
//! implementation and cannot drift apart. What stays here is the part that is
//! genuinely app-specific: where the binary and the model live inside a bundle.

use std::path::{Path, PathBuf};

/// Re-exported so existing call sites (`local_llm::LocalWorker`) keep reading the
/// same. The type itself is `whimpr_core::worker::LocalWorker`.
pub use whimpr_core::worker::LocalWorker;

/// Platform application-support dir: `~/Library/Application Support/WhimprFlow`
/// on macOS, `%APPDATA%\WhimprFlow Dev` on Windows. Deliberately separate from the
/// stable app's dir — see `hotkey.rs::support_dir` and project.md, Phase 0.
fn app_support_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("APPDATA").unwrap_or_default();
        PathBuf::from(base).join("WhimprFlow")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join("Library/Application Support/WhimprFlow")
    }
}

/// Find the worker binary: next to the app executable (bundled), else the dev build dir.
///
/// The bundled copy is put there by Tauri's `externalBin` (see tauri.conf.json and
/// scripts/build-worker.sh), which strips the target-triple suffix when it copies
/// the binary into `Contents/MacOS/`. That is the path that should hit in a real
/// install.
///
/// The dev fallback below is a *checkout-relative guess* and used to be the only
/// thing that made cleanup work at all. It happens to be correct only because the
/// repo lives at ~/WhimprFlow: move the .app anywhere else and cleanup would
/// silently revert to pasting raw transcripts, with no error and no log line. It is
/// kept for `cargo run` during development, but it now says so loudly, because
/// "quietly worse output forever" is the worst way for this to fail.
pub fn worker_bin_path() -> Option<PathBuf> {
    let exe_name = if cfg!(target_os = "windows") {
        "whimpr-llm-worker.exe"
    } else {
        "whimpr-llm-worker"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(exe_name);
            if cand.exists() {
                return Some(cand);
            }
            eprintln!(
                "[whimpr] no bundled cleanup worker at {} — falling back to the dev build dir",
                cand.display()
            );
        }
    }
    // Dev fallback.
    #[cfg(target_os = "windows")]
    let dev = std::env::current_dir()
        .unwrap_or_default()
        .join("target/release")
        .join(exe_name);
    #[cfg(not(target_os = "windows"))]
    let dev = {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join("WhimprFlow/target/release/whimpr-llm-worker")
    };

    if dev.exists() {
        eprintln!(
            "[whimpr] ⚠ using DEV worker path {} — this app bundle is not self-contained; \
             cleanup will silently stop working if it is moved. Rebuild with \
             `./ui/node_modules/.bin/tauri build` to bundle the worker.",
            dev.display()
        );
        Some(dev)
    } else {
        eprintln!(
            "[whimpr] ⚠ cleanup worker not found (looked next to the executable and at {}). \
             Local cleanup is DISABLED — transcripts will be pasted raw.",
            dev.display()
        );
        None
    }
}

/// The local model path (same models dir as whisper/ASR). Prefer the larger,
/// much more capable Qwen3-4B if present (far better at self-corrections and
/// structure than the 1.5B); fall back to the 1.5B otherwise.
///
/// 🔴 **Pinned to the 1.5B on purpose — this is not an oversight, and the 4B must
/// not be added back to this list.** Max's decision, 2026-08-17: the two stages
/// get different models, *"the 4B for math, the 1.5 to the cleanup."* Cleanup runs
/// on the hot path where a human is waiting on ordinary dictation, and the 4B
/// costs roughly three times as long; the math stage runs behind a hotkey pressed
/// deliberately for accuracy, where that wait is the point.
///
/// Pinning also **defuses a footgun**. Until this commit the 4B was preferred
/// here and merely happened to be absent, parked under a `.candidate` suffix — so
/// dropping that suffix would silently have re-modelled the daily driver's
/// cleanup. A rename in the models directory is now safe: it changes the math
/// stage (see [`math_model_path`]) and nothing else.
pub fn model_path() -> PathBuf {
    app_support_dir()
        .join("models")
        .join("qwen2.5-1.5b-instruct-q4_k_m.gguf")
}

/// The model for the spoken-math stage: the 4B, which is markedly better at this
/// specific job than the 1.5B and is allowed to be slower for it.
///
/// Measured 2026-08-17 over the eleven-input evaluation set, and again live on
/// Max's own voice: on *"the contour integral around gamma of f of z over z minus
/// z naught dz"* the 1.5B returned `The contour integral around γ of f(z)/(z − z₀)
/// dz` — leaving the operator as English words — while the 4B returned
/// `∮_γ f(z)/(z − z₀) dz`. Across the set the 1.5B also **silently dropped terms**
/// (`π^S` for "pi to the S minus 1", losing the −1), which is the worst failure
/// available here because the result still looks plausible.
///
/// **Accepts the `.candidate` suffix**, so the 4B works whether or not it has been
/// renamed — the file is currently parked that way and renaming it is a decision
/// nothing here should force. Falls back to the 1.5B if no 4B is present at all,
/// so math mode degrades to worse notation rather than to no feature.
pub fn math_model_path() -> PathBuf {
    let dir = app_support_dir().join("models");
    for name in [
        "qwen3-4b-instruct-2507-q4_k_m.gguf",
        "qwen3-4b-instruct-2507-q4_k_m.gguf.candidate",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return p;
        }
    }
    model_path()
}

/// Spawn the cleanup worker if both the binary and the model are present.
pub fn spawn_default() -> Option<LocalWorker> {
    spawn_with(&model_path(), "cleanup")
}

/// Spawn the math worker. Separate process from cleanup, holding a separate model
/// — that is the whole point, and it costs a second resident model (~2.4 GB) while
/// it is alive. Spawned lazily on first use, never at startup, so a user who never
/// presses the math gesture never pays for it.
pub fn spawn_math() -> Option<LocalWorker> {
    spawn_with(&math_model_path(), "math")
}

fn spawn_with(model: &Path, label: &str) -> Option<LocalWorker> {
    let bin = worker_bin_path()?;
    if !model.exists() {
        eprintln!("[whimpr] {label} model not found at {}", model.display());
        return None;
    }
    match LocalWorker::spawn(&bin, model) {
        Ok(w) => {
            eprintln!(
                "[whimpr] {label} LLM worker started ({} · {})",
                bin.display(),
                model.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()
            );
            Some(w)
        }
        Err(e) => {
            eprintln!("[whimpr] {label} LLM worker failed to start: {e}");
            None
        }
    }
}
