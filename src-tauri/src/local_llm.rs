//! Finds and spawns the local-LLM worker (a separate process, so llama.cpp and
//! whisper.cpp never link into the same binary).
//!
//! The **protocol** — one JSON request per line over stdio — lives in
//! `whimpr_core::worker` so the app and the offline harness share a single
//! implementation and cannot drift apart. What stays here is the part that is
//! genuinely app-specific: where the binary and the model live inside a bundle.

use std::path::PathBuf;

/// Re-exported so existing call sites (`local_llm::LocalWorker`) keep reading the
/// same. The type itself is `whimpr_core::worker::LocalWorker`.
pub use whimpr_core::worker::LocalWorker;

/// Platform application-support dir: `~/Library/Application Support/WhimprFlow Dev`
/// on macOS, `%APPDATA%\WhimprFlow Dev` on Windows. Deliberately separate from the
/// stable app's dir — see `hotkey.rs::support_dir` and project.md, Phase 0.
fn app_support_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var("APPDATA").unwrap_or_default();
        PathBuf::from(base).join("WhimprFlow Dev")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join("Library/Application Support/WhimprFlow Dev")
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
/// ⚠️ **One worker serves both cleanup and the math stage**, so this function
/// decides the model for both — there is no separate math model. The 4B is
/// currently parked in the models dir as `…q4_k_m.gguf.candidate`, which does not
/// match the name below, so this resolves to the **1.5B** — which is what the
/// math stage was chosen on (2026-08-17: 1.5B-Unicode at ~1.4 s median versus
/// ~3.7 s for the 4B). Dropping the `.candidate` suffix therefore changes the
/// math stage's model and latency as well as cleanup's. The math stage logs the
/// resolved model filename on every run so a rename never changes behaviour
/// silently.
pub fn model_path() -> PathBuf {
    let dir = app_support_dir().join("models");
    for name in [
        "qwen3-4b-instruct-2507-q4_k_m.gguf",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ] {
        let p = dir.join(name);
        if p.exists() {
            return p;
        }
    }
    dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf")
}

/// Spawn the worker if both the binary and the model are present.
pub fn spawn_default() -> Option<LocalWorker> {
    let bin = worker_bin_path()?;
    let model = model_path();
    if !model.exists() {
        eprintln!("[whimpr] local model not found at {}", model.display());
        return None;
    }
    match LocalWorker::spawn(&bin, &model) {
        Ok(w) => {
            eprintln!("[whimpr] local LLM worker started ({})", bin.display());
            Some(w)
        }
        Err(e) => {
            eprintln!("[whimpr] local LLM worker failed to start: {e}");
            None
        }
    }
}
