//! Dry-run the deterministic dictionary stage over real transcripts.
//!
//! `DictionaryStore::apply` runs unconditionally on the hot lane, so a bad entry is
//! not a missed correction — it is a wrong word pasted into every future dictation
//! that contains the trigger. This tool answers "what would my dictionary actually do
//! to my own dictation?" *before* that ships, using the real transcripts already
//! sitting in the logs.
//!
//! It exists because the first run of it found exactly that failure: an auto-learned
//! entry mapping "Cloud" → "clad" would have rewritten Max's "Claude Code" to
//! "Clad Code" twice in one week's dictation.
//!
//! ```text
//! grep -h '\[whimpr\] TRANSCRIPT: ' ~/Library/Application\ Support/WhimprFlow/logs/*.log \
//!   | sed 's/.*TRANSCRIPT: "//; s/"$//' \
//!   | cargo run --release -p whimpr-core --example dict-dryrun -- \
//!       ~/Library/Application\ Support/WhimprFlow/dictionary.json
//! ```
//!
//! Reads transcripts on stdin, one per line. Prints every change it would make, so
//! each one can be eyeballed as right or wrong.

use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use whimpr_core::DictionaryStore;

fn main() {
    let Some(dict_path) = std::env::args().nth(1) else {
        eprintln!("usage: dict-dryrun <path/to/dictionary.json>   (transcripts on stdin)");
        std::process::exit(2);
    };

    let store = DictionaryStore::load(Path::new(&dict_path));
    if store.entries.is_empty() {
        eprintln!("warning: no entries loaded from {dict_path} — nothing to apply");
    }

    let transcripts: Vec<String> = std::io::stdin()
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    let mut changes = Vec::new();
    let mut total_hits = 0usize;

    let started = Instant::now();
    for t in &transcripts {
        let (out, hits) = store.apply(t);
        if hits > 0 {
            total_hits += hits;
            changes.push((t.clone(), out));
        }
    }
    let elapsed = started.elapsed();

    println!("dictionary entries   : {}", store.entries.len());
    println!("transcripts replayed : {}", transcripts.len());
    println!("transcripts changed  : {}", changes.len());
    println!("total replacements   : {total_hits}");
    println!(
        "cost                 : {:?} for all {}, {:.1} µs per dictation",
        elapsed,
        transcripts.len(),
        elapsed.as_secs_f64() * 1e6 / transcripts.len().max(1) as f64
    );

    if changes.is_empty() {
        println!("\nno changes — every transcript would pass through untouched.");
        return;
    }

    println!("\n--- every change this stage would make (check each one) ---");
    for (before, after) in &changes {
        println!("  BEFORE: {before}");
        println!("  AFTER : {after}\n");
    }
}
