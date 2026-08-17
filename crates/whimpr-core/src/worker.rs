//! Client for the local-LLM sidecar (`whimpr-llm-worker`): spawn it once, then
//! send one JSON request per line over stdio and read one response per line.
//!
//! This lives in `whimpr-core` rather than in the Tauri app so that the app and
//! the offline harness talk to the worker through **one** implementation. The
//! harness exists to answer questions about what the app will do; a second copy
//! of the wire format is a copy that can drift, and a harness that measures a
//! different request shape than the app sends produces numbers that are lies.
//! (`whimpr-harness`'s own `default_model_path` carries the same warning about
//! hand-syncing.)
//!
//! Path resolution (which binary, which model) deliberately stays in the app —
//! it is bundle-layout-specific and has nothing to do with the protocol.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::cleanup::CleanupMsg;

/// A spawned worker process, kept warm across requests (model load is a
/// once-per-launch cost of several seconds; paying it per request was the
/// dominant source of perceived latency before the context was made reusable).
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
        let stdout =
            BufReader::new(child.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?);
        Ok(Self { child, stdin, stdout })
    }

    /// Send one request (system prompt + few-shot turns + transcript) and block
    /// until the response line arrives.
    ///
    /// 🔴 The worker reports failure in an `error` field while still returning a
    /// `text` field, so a caller that reads only `text` renders a hard failure as
    /// `""` — which looks exactly like a bad model rather than a broken call.
    /// That nearly produced a false quality verdict on 2026-08-17 (eight of
    /// fifteen sweep cases were empty for a buffer-size bug, not a model one), so
    /// `error` is checked FIRST here and turned into a real `Err`. Every caller
    /// gets that guarantee by construction; none of them has to remember.
    pub fn request(&mut self, messages: &[CleanupMsg], max_tokens: u32) -> anyhow::Result<String> {
        let req = serde_json::json!({ "messages": messages, "max_tokens": max_tokens });
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

    /// Cleanup's budget. Kept as a named method so the app's call site reads the
    /// same as it always did.
    pub fn cleanup(&mut self, messages: &[CleanupMsg]) -> anyhow::Result<String> {
        self.request(messages, 400)
    }
}

impl Drop for LocalWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
