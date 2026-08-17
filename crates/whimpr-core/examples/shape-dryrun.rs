//! Replay real `pasted -> edited` pairs through the edit-shape gate and report what it
//! would learn.
//!
//! Reads TSV on **stdin**: `pasted<TAB>edited`, one pair per line, with literal `\n`
//! and `\t` escaped inside the fields. Prints per-pair verdicts and a summary.
//!
//! ```text
//! python3 scripts/export-wispr-pairs.py \
//!   | cargo run --release -q -p whimpr-core --example shape-dryrun
//! ```
//!
//! ⚠️ Like `dict-dryrun`, this prints a vacuous pass if you forget the pipe. **Check
//! `pairs replayed` is non-zero before believing anything below it.**
//!
//! ⚠️ This is a corpus of real edit SHAPES, not labelled ground truth for correctness.
//! Wispr's own auto-learn kept 6 entries from this data and 2 of them are typos. A pair
//! being accepted here does not make it a good dictionary entry — the vocabulary-quality
//! gates in `autolearn::detect_correction` are a separate stage.

use std::collections::HashSet;

use whimpr_core::editshape::{
    analyse_in_region, is_learnable_pair, word_tokens, PriorContext,
};

/// The OLD detector's shape rule, restated for before/after comparison.
///
/// A faithful restatement of `autolearn::word_diff` + `detect_correction`'s shape half as
/// it stood at commit `80bf363`: a **set difference over the whole field**, accepted only
/// when exactly one distinct word was removed and exactly one added. It is reproduced here
/// rather than called because it is being deleted; if it ever disagrees with the real thing
/// the real thing is right.
fn old_rule_shape(pasted: &str, field: &str) -> Option<(String, String)> {
    let ins = word_tokens(pasted);
    let aft = word_tokens(field);
    if ins.is_empty() || aft.is_empty() {
        return None;
    }
    let ins_lc: HashSet<String> = ins.iter().map(|w| w.to_lowercase()).collect();
    let aft_lc: HashSet<String> = aft.iter().map(|w| w.to_lowercase()).collect();
    let diff = |from: &[String], other: &HashSet<String>| -> Vec<String> {
        let mut seen = HashSet::new();
        from.iter()
            .filter(|w| !other.contains(&w.to_lowercase()))
            .filter(|w| seen.insert(w.to_lowercase()))
            .cloned()
            .collect()
    };
    let (removed, added) = (diff(&ins, &aft_lc), diff(&aft, &ins_lc));
    (removed.len() == 1 && added.len() == 1).then(|| (removed[0].clone(), added[0].clone()))
}

fn unescape(s: &str) -> String {
    s.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")
}

/// Real prior text, taken verbatim from the Notes note Max was dictating into on
/// 2026-08-17 when the whole-field set difference rejected his correction.
const PRIOR_BEFORE: &str = "TODO groceries milk eggs return library books Friday check tire pressure";
const PRIOR_AFTER: &str = "and then some older notes below";

fn main() {
    // `--embed` wraps every pair in the surrounding text a real document has.
    //
    // This is the measurement that matters, and the plain mode CANNOT substitute for it:
    // Wispr's `pastedText`/`editedText` are already region-scoped, so the corpus contains
    // no surrounding-document text at all and the whole-field defect is invisible in it.
    // Embedding the same real edits in real prior text is what exercises the fix.
    let embed = std::env::args().any(|a| a == "--embed");
    let mut pairs = 0usize;
    let mut accepted = 0usize;
    let mut old_shape_ok = 0usize;
    let mut old_full_ok = 0usize;
    let mut new_shape_ok = 0usize;
    let mut merges = 0usize;
    let mut shapes: std::collections::BTreeMap<String, usize> = Default::default();
    let mut examples: Vec<(String, String, String)> = Vec::new();

    for line in std::io::stdin().lines().map_while(Result::ok) {
        let Some((a, b)) = line.split_once('\t') else { continue };
        let (pasted, edited) = (unescape(a), unescape(b));
        if pasted.trim().is_empty() || edited.trim().is_empty() {
            continue;
        }
        pairs += 1;

        // The corpus gives no first-read baseline, so treat the pasted text as the whole
        // field. That is the CONSERVATIVE choice: with no prior context declared, any
        // surrounding text shows up as insertions and can only cause rejection, never a
        // false accept. Real dictations do have a baseline and will do better than this.
        // In embed mode the FIELD carries prior text on both sides; the baseline is that
        // same prior text around the untouched paste, which is exactly what poll 1 sees.
        let (baseline, field) = if embed {
            (
                format!("{PRIOR_BEFORE} {pasted} {PRIOR_AFTER}"),
                format!("{PRIOR_BEFORE} {edited} {PRIOR_AFTER}"),
            )
        } else {
            (pasted.clone(), edited.clone())
        };

        let prior = PriorContext::from_baseline(&pasted, &baseline);
        let v = analyse_in_region(&pasted, &field, &prior);

        if let Some((m, c)) = old_rule_shape(&pasted, &field) {
            old_shape_ok += 1;
            if is_learnable_pair(&m, &c).is_none() {
                old_full_ok += 1;
            }
        }

        let key: String = v.labels.chars().collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect();
        *shapes.entry(if key.is_empty() { "(none)".into() } else { key }).or_default() += 1;

        if let Some((mishear, correct)) = v.correction {
            new_shape_ok += 1;
            if is_learnable_pair(&mishear, &correct).is_none() {
                accepted += 1;
                if mishear.contains(' ') {
                    merges += 1;
                }
                if examples.len() < 25 {
                    examples.push((mishear, correct, v.labels));
                }
            }
        }
    }

    let pct = |n: usize| if pairs == 0 { 0.0 } else { 100.0 * n as f64 / pairs as f64 };
    println!("mode                       : {}", if embed {
        "EMBEDDED in real prior text (the realistic case)"
    } else {
        "bare pairs (region-scoped already — cannot exercise the whole-field fix)"
    });
    println!("pairs replayed             : {pairs}");
    println!();
    println!("OLD  whole-field set diff  : {old_shape_ok} pass shape  ({:.1}%)", pct(old_shape_ok));
    println!("OLD  + vocabulary gates    : {old_full_ok} learned     ({:.1}%)", pct(old_full_ok));
    println!("NEW  aligned shape gate    : {new_shape_ok} pass shape  ({:.1}%)", pct(new_shape_ok));
    println!("NEW  + vocabulary gates    : {accepted} learned     ({:.1}%)", pct(accepted));
    println!();
    println!("of the new ones, word MERGES (impossible under the old rule): {merges}");
    println!("\nlabel-character mix (which operations appear, per pair):");
    let mut mix: Vec<_> = shapes.into_iter().collect();
    mix.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in mix.iter().take(12) {
        println!("  {k:<10} {n}");
    }
    if !examples.is_empty() {
        println!("\nwhat it would learn (check every one — this is not ground truth):");
        for (m, c, l) in &examples {
            println!("  {l:<14} {m:?} -> {c:?}");
        }
    }
}
