//! Spawns and talks to the local-LLM cleanup worker (a separate process, so
//! llama.cpp and whisper.cpp never link into the same binary). One JSON request
//! per line over stdio: `{system,user}` -> `{text}`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct LocalWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl LocalWorker {
    pub fn spawn(worker_bin: &Path, model: &Path) -> anyhow::Result<Self> {
        let mut child = Command::new(worker_bin)
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?);
        Ok(Self { child, stdin, stdout })
    }

    /// Send one cleanup request (system prompt + few-shot turns + transcript) and
    /// read the response (blocks until the line comes).
    pub fn cleanup(
        &mut self,
        messages: &[whimpr_core::cleanup::CleanupMsg],
    ) -> anyhow::Result<String> {
        let req = serde_json::json!({ "messages": messages, "max_tokens": 400 });
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;

        let mut resp = String::new();
        if self.stdout.read_line(&mut resp)? == 0 {
            anyhow::bail!("local worker closed");
        }
        let v: serde_json::Value = serde_json::from_str(&resp)?;
        if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
            anyhow::bail!("local llm: {err}");
        }
        Ok(v.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string())
    }
}

impl Drop for LocalWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// Platform application-support dir: `~/Library/Application Support/WhimprFlow`
/// on macOS, `%APPDATA%\WhimprFlow` on Windows.
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

/// The local cleanup model path (same models dir as whisper/ASR). Prefer the
/// larger, much more capable Qwen3-4B if present (far better at
/// self-corrections and structure than the 1.5B); fall back to the 1.5B otherwise.
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
