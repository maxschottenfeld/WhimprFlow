//! Replay `strip_pause_punctuation` over real logged dictations and print every
//! line it would change, so each one can be eyeballed instead of trusted.
//!
//! This is the pre-ship check for the pause strip, and the sibling of
//! `whimpr-core`'s `dict-dryrun`. The rule is cheap to reason about and easy to
//! get subtly wrong — an early draft turned the real dictation
//! `"editing CLAUDE.md and .md files"` into `"and.md files"` — so the standard
//! here is the same one the rest of this project holds: measure over Max's actual
//! text before believing anything.
//!
//! ```text
//! cargo run -p whimpr-asr --example pause-dryrun
//! cargo run -p whimpr-asr --example pause-dryrun -- /path/to/logs
//! ```
//!
//! Logs are `~/Library/Application Support/WhimprFlow[ Dev]/logs/*.log`, and a
//! snapshot of them lives in the vault at
//! `memory/projects/WhimprFlow/log-snapshot-2026-08-16/` — retention is 14 days,
//! so the snapshot outlives the originals.

use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_DIRS: &[&str] = &[
    "log-snapshot/live",
    "log-snapshot/tmp-0815",
];

const TRANSCRIPT: &str = "[whimpr] TRANSCRIPT: \"";
const CLEANED: &str = "[whimpr] CLEANED:   \"";

struct Rec {
    file: String,
    line: usize,
    transcript: String,
    cleaned: Option<String>,
}

impl Rec {
    /// What actually reached the paste path.
    fn final_text(&self) -> &str {
        self.cleaned.as_deref().unwrap_or(&self.transcript)
    }
}

/// Read a possibly multi-line quoted payload starting at `lines[i]`.
/// Returns the payload and the index of the last line consumed.
fn read_quoted(lines: &[&str], i: usize, prefix: &str) -> (String, usize) {
    let first = &lines[i][prefix.len()..];
    if first.ends_with('"') {
        return (first[..first.len() - 1].to_string(), i);
    }
    let mut buf = first.to_string();
    let mut j = i + 1;
    while j < lines.len() {
        buf.push('\n');
        buf.push_str(lines[j]);
        if lines[j].ends_with('"') {
            buf.truncate(buf.len() - 1);
            return (buf, j);
        }
        if lines[j].starts_with("[whimpr") {
            return (buf, j);
        }
        j += 1;
    }
    (buf, lines.len() - 1)
}

fn parse(path: &Path) -> Vec<Rec> {
    let Ok(body) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lines: Vec<&str> = body.lines().collect();
    let mut recs: Vec<Rec> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with(TRANSCRIPT) {
            let (text, end) = read_quoted(&lines, i, TRANSCRIPT);
            recs.push(Rec {
                file: name.clone(),
                line: end + 1,
                transcript: text,
                cleaned: None,
            });
            i = end + 1;
        } else if lines[i].starts_with(CLEANED) {
            let (text, end) = read_quoted(&lines, i, CLEANED);
            if let Some(last) = recs.last_mut() {
                last.cleaned = Some(text);
            }
            i = end + 1;
        } else {
            i += 1;
        }
    }
    recs
}

fn show(s: &str) -> String {
    s.replace('\n', "\\n")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dirs: Vec<PathBuf> = if args.is_empty() {
        DEFAULT_DIRS.iter().map(PathBuf::from).collect()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let mut recs = Vec::new();
    // The vault snapshot holds two trees and the second re-copies most of the
    // first, so the same dictation would otherwise be counted twice. First
    // directory listed wins.
    let mut seen: Vec<String> = Vec::new();
    for dir in &dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            eprintln!("skipping unreadable dir: {}", dir.display());
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "log"))
            .collect();
        paths.sort();
        for p in paths {
            let base = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            if seen.contains(&base) {
                continue;
            }
            seen.push(base);
            recs.extend(parse(&p));
        }
    }

    let total = recs.len();
    let nonempty: Vec<&Rec> = recs.iter().filter(|r| !r.transcript.trim().is_empty()).collect();

    println!("pause-dryrun — replaying strip_pause_punctuation over real dictations");
    println!("  dictations parsed : {total}");
    println!("  non-empty         : {}\n", nonempty.len());

    let mut changed = 0usize;
    let mut emptied = 0usize;
    println!("{}", "=".repeat(78));
    println!("EVERY LINE THE RULE WOULD CHANGE (on the text that reached the paste path)");
    println!("{}", "=".repeat(78));
    for r in &nonempty {
        let before = r.final_text();
        let after = whimpr_asr::strip_pause_punctuation(before);
        if after == before {
            continue;
        }
        changed += 1;
        if after.is_empty() {
            emptied += 1;
        }
        let via = if r.cleaned.is_some() { "CLEANED" } else { "TRANSCRIPT" };
        println!("\n[{}:{}] via {via}", r.file, r.line);
        println!("  BEFORE: {}", show(before));
        println!("  AFTER : {}", show(&after));
    }

    // The safety number: how many dictations does this rule leave alone?
    let untouched = nonempty.len() - changed;
    println!("\n{}", "=".repeat(78));
    println!("SUMMARY");
    println!("{}", "=".repeat(78));
    println!("  changed        : {changed} / {}", nonempty.len());
    println!("  unchanged      : {untouched} / {}", nonempty.len());
    println!("  became empty   : {emptied}  (nothing is pasted for these)");
    println!(
        "\n  A change here is only correct if every BEFORE above contains punctuation\n  \
         Max did not say. Read them; do not trust the count."
    );
}
