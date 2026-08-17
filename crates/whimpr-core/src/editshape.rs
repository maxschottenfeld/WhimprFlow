//! Edit **shape** analysis: align what we pasted against what the field now holds,
//! and decide whether the difference looks like a *correction* rather than continued
//! writing.
//!
//! ## Why this exists — read before changing it
//!
//! Auto-learn's original detector was a **set difference over the whole field**
//! ([`crate::dictionary`]'s sibling, `autolearn::word_diff`). Two consequences, both
//! measured live on 2026-08-17 and both fatal:
//!
//! 1. **Any pre-existing text in the target field made a correction undetectable.**
//!    Every word already there counted as "added", so a clean one-word fix could
//!    never be a 1-for-1 swap. Max corrected `derelict → Dirichlet` in a Notes note
//!    holding 11 unrelated words; the correction was *seen and held across seven
//!    consecutive polls* and was rejected as `-1 +12`. In practice auto-learn only
//!    worked in an **empty** field — not a note with content, not a chat with
//!    history, not a document, not an email with a signature.
//!
//! 2. **Proper nouns mishear into *two* words, not one**, and a set difference with a
//!    1-for-1 requirement rejects all of them. Observed the same evening:
//!    `whisper flow` ← WhimprFlow, `They're elect` ← Dirichlet, `silo row` ← Silero.
//!    Three of four dictations.
//!
//! ## The approach, taken from the reference implementation
//!
//! Wispr Flow does not compare sets and does not gate on edit *distance*. It aligns
//! the two token sequences, collapses each step of the alignment to a one-character
//! label, and matches the resulting string against two regexes. Its alphabet:
//!
//! | `M` | `Z` | `D` | `I` | `S` | `C` | `E` |
//! |---|---|---|---|---|---|---|
//! | Match | None | Delete | Insert | Substitution | Casing | EditCaptureError |
//!
//! **A substitution is learnable only when flanked by untouched tokens.** Adjacent to
//! another substitution, or to an insertion, it is continued writing rather than a
//! correction. That single rule is what makes surrounding text irrelevant, which is
//! the property the set difference lacked.
//!
//! Two shapes are accepted (regexes lifted verbatim from the shipped bundle and
//! verified by execution against WhimprFlow's own observed cases):
//!
//! ```text
//! ISOLATED_SINGLE_SUBSTITUTION  [CMZ]S[CMZ] | ^S[CMZ] | [CMZ]S$
//! DELETION_SUBSTITUTION         [CMZ](DS|SD)[CMZ] | ^(DS|SD)[CMZ] | [CMZ](DS|SD)$
//! ```
//!
//! The second is what admits a word **merge** — two spoken-wrong tokens replaced by
//! one right one — which is the dominant real shape for the proper nouns a dictionary
//! exists to fix.
//!
//! ## The region has to be MEASURED, not inferred
//!
//! Aligning against the whole field is still wrong on its own. Prior text before the
//! insertion point appears as leading `I`s, which is harmless — `MMMS` still ends in
//! `[CMZ]S$`. But prior text **after** the insertion point appears as trailing `I`s, and
//! `MMMSIIII` matches neither regex: the substitution is no longer flanked. So the
//! surrounding text must be removed before the label string is built. Wispr does the
//! equivalent by tracking `originalOffset`/`originalTotal` over an observed region.
//!
//! 🔴 **The obvious shortcut does not work, and it was tried first.** Dropping leading
//! and trailing insertion steps from the *current* read looks equivalent and is not:
//! **prior text and text the user has just typed are both trailing insertions**, and a
//! single snapshot cannot distinguish them. That version accepted
//! `"ping the server foo"` → `"ping the server bar quickly"` as an isolated
//! substitution, when the trailing word means the user is still writing — precisely the
//! case the flanking rule exists to reject.
//!
//! So the region comes from [`PriorContext::from_baseline`], built from the **first**
//! read of the field (~1.2 s after the paste, before the user has edited anything).
//! That read shows what was already there. Everything after it is the user's doing.
//! See `prior_text_and_newly_typed_text_are_told_apart` for the two cases side by side.

/// One step of the alignment between the pasted text and the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    /// Tokens identical.
    Match,
    /// Tokens differ by case only.
    Casing,
    /// One token replaced by one different token.
    Substitution,
    /// A token we pasted is gone.
    Delete,
    /// A token appeared that we did not paste.
    Insert,
    /// A substitution at the edge of the region whose replacement is a prefix or
    /// suffix of what it replaced — almost certainly an artifact of *when we looked*
    /// rather than a correction the user finished making.
    ///
    /// This is the structural kill for the `Hypothesis → Hypothe` class: a read taken
    /// mid-keystroke. It can never satisfy either accept pattern, by construction.
    CaptureError,
}

impl Label {
    fn ch(self) -> char {
        match self {
            Self::Match => 'M',
            Self::Casing => 'C',
            Self::Substitution => 'S',
            Self::Delete => 'D',
            Self::Insert => 'I',
            Self::CaptureError => 'E',
        }
    }

    /// Is this an "untouched" label, i.e. one that may flank a substitution?
    /// Mirrors the `[CMZ]` character class in both regexes.
    fn is_unchanged(self) -> bool {
        matches!(self, Self::Match | Self::Casing)
    }
}

/// One aligned step: its label plus the tokens on each side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub label: Label,
    /// The token from the pasted text, if this step consumed one.
    pub from: Option<String>,
    /// The token from the field, if this step consumed one.
    pub to: Option<String>,
}

/// Split into alphanumeric word tokens, punctuation trimmed from the ends only,
/// original case kept.
///
/// Deliberately identical in behaviour to `autolearn::word_tokens` so the two views of
/// a dictation cannot disagree about what a word is. Note the consequence, which is
/// load-bearing: an **interior** hyphen or apostrophe survives, so `Re-Place` and
/// `They're` are single tokens.
pub fn word_tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Align two token sequences by longest common subsequence, then classify each step.
///
/// The LCS is computed over case-insensitive equality so that a casing-only change
/// aligns as a pair (labelled [`Label::Casing`]) instead of showing up as an unrelated
/// delete-plus-insert. That matters: `DictionaryStore::apply` treats case-only entries
/// as first-class — `Katex → KaTeX` is its motivating example — and the set-difference
/// detector could never produce one, because it lowercased before comparing and the
/// difference vanished.
pub fn align(from: &[String], to: &[String]) -> Vec<Step> {
    let (n, m) = (from.len(), to.len());
    // lcs[i][j] = length of the LCS of from[i..] and to[j..]
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if from[i].eq_ignore_ascii_case(&to[j]) {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    let mut steps = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if from[i].eq_ignore_ascii_case(&to[j]) {
            let label = if from[i] == to[j] {
                Label::Match
            } else {
                Label::Casing
            };
            steps.push(Step {
                label,
                from: Some(from[i].clone()),
                to: Some(to[j].clone()),
            });
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            // Dropping from[i] keeps at least as much in common → it was deleted.
            steps.push(Step {
                label: Label::Delete,
                from: Some(from[i].clone()),
                to: None,
            });
            i += 1;
        } else {
            steps.push(Step {
                label: Label::Insert,
                from: None,
                to: Some(to[j].clone()),
            });
            j += 1;
        }
    }
    while i < n {
        steps.push(Step {
            label: Label::Delete,
            from: Some(from[i].clone()),
            to: None,
        });
        i += 1;
    }
    while j < m {
        steps.push(Step {
            label: Label::Insert,
            from: None,
            to: Some(to[j].clone()),
        });
        j += 1;
    }

    coalesce_substitutions(steps)
}

/// Turn a **run** of deletes followed by a **run** of inserts into substitutions,
/// pairing them positionally.
///
/// An LCS walk has no notion of "replaced" — it only deletes and inserts, and it emits
/// them in runs rather than interleaved. Two adjacent changed words arrive as
/// `D D I I`, not `D I D I`.
///
/// 🔴 **An earlier version of this function only coalesced an immediately adjacent
/// `D`,`I` pair, and it silently mispaired the tokens.** On
/// `"meet at noon monvi"` → `"meet at dawn Manvi"` the walk produces `M M D D I I`;
/// pairwise scanning left the first `D` alone and then fused the *second* delete with
/// the *first* insert, yielding `M M D S` with the nonsense correction
/// `"noon monvi" → "dawn"` — and `M M D S` matches the word-merge accept pattern, so a
/// two-word rewrite would have been learned as a correction. Caught by
/// `two_adjacent_substitutions_are_rewriting_not_correcting` failing on first run.
///
/// Pairing whole runs gives `M M S S`, which is correctly rejected as rewriting.
fn coalesce_substitutions(steps: Vec<Step>) -> Vec<Step> {
    let mut out: Vec<Step> = Vec::with_capacity(steps.len());
    let mut k = 0;
    while k < steps.len() {
        if steps[k].label != Label::Delete {
            out.push(steps[k].clone());
            k += 1;
            continue;
        }
        // Measure the delete run, then any insert run immediately after it.
        let d_start = k;
        while k < steps.len() && steps[k].label == Label::Delete {
            k += 1;
        }
        let d_end = k;
        let i_start = k;
        while k < steps.len() && steps[k].label == Label::Insert {
            k += 1;
        }
        let i_end = k;

        let dels = d_end - d_start;
        let ins = i_end - i_start;
        let pairs = dels.min(ins);

        for n in 0..pairs {
            out.push(Step {
                label: Label::Substitution,
                from: steps[d_start + n].from.clone(),
                to: steps[i_start + n].to.clone(),
            });
        }
        // Whichever run was longer leaves a tail of plain deletes or inserts.
        for n in pairs..dels {
            out.push(steps[d_start + n].clone());
        }
        for n in pairs..ins {
            out.push(steps[i_start + n].clone());
        }
    }
    out
}

/// The tokens that were already in the field, before and after the pasted text.
///
/// Derived from the **first read** of the field, which happens ~1.2 s after the paste
/// and therefore shows the paste sitting in whatever was already there, before the user
/// has edited anything.
///
/// 🔴 **This must be measured, not inferred, and that is the whole point.** An earlier
/// version of this module inferred the region by dropping leading and trailing
/// insertion steps from the current read. That cannot work, because *prior text* and
/// *text the user has just typed* are both trailing insertions and are indistinguishable
/// from a single snapshot. It made a continued-typing edit
/// (`"ping the server foo"` → `"ping the server bar quickly"`) look like an isolated
/// substitution and accepted it. Caught by
/// `a_substitution_beside_an_insertion_is_rejected` failing on first run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PriorContext {
    pub before: Vec<String>,
    pub after: Vec<String>,
}

impl PriorContext {
    /// Work out what was already in the field by aligning the pasted text against the
    /// first read of it.
    pub fn from_baseline(pasted: &str, baseline: &str) -> Self {
        let from = word_tokens(pasted);
        let to = word_tokens(baseline);
        if from.is_empty() || to.is_empty() {
            return Self::default();
        }
        let steps = align(&from, &to);
        let before = steps
            .iter()
            .take_while(|s| s.label == Label::Insert)
            .filter_map(|s| s.to.clone())
            .collect();
        let after: Vec<String> = steps
            .iter()
            .rev()
            .take_while(|s| s.label == Label::Insert)
            .filter_map(|s| s.to.clone())
            .collect();
        Self {
            before,
            after: after.into_iter().rev().collect(),
        }
    }

    /// Remove the known prior tokens from a later read, leaving the region the paste
    /// landed in.
    ///
    /// Matching, not counting: if the user edited the surrounding text too, a token
    /// that no longer matches is left in place rather than silently consumed. The
    /// count is only an upper bound.
    fn strip<'a>(&self, mut field: &'a [String]) -> &'a [String] {
        for want in &self.before {
            match field.first() {
                Some(got) if got.eq_ignore_ascii_case(want) => field = &field[1..],
                _ => break,
            }
        }
        for want in self.after.iter().rev() {
            match field.last() {
                Some(got) if got.eq_ignore_ascii_case(want) => {
                    field = &field[..field.len() - 1]
                }
                _ => break,
            }
        }
        field
    }
}

/// Re-label edge substitutions that look like a mid-keystroke read.
///
/// Applied *after* region trimming, because "edge" means the edge of the pasted
/// region, not of the whole field.
fn mark_capture_errors(steps: &mut [Step]) {
    let n = steps.len();
    if n == 0 {
        return;
    }
    for idx in [0, n - 1] {
        let s = &mut steps[idx];
        if s.label != Label::Substitution {
            continue;
        }
        if let (Some(f), Some(t)) = (s.from.as_ref(), s.to.as_ref()) {
            let (fl, tl) = (f.to_lowercase(), t.to_lowercase());
            if fl != tl && (fl.starts_with(&tl) || fl.ends_with(&tl)) {
                s.label = Label::CaptureError;
            }
        }
    }
}

/// The label string for a trimmed, capture-error-marked alignment.
pub fn label_string(steps: &[Step]) -> String {
    steps.iter().map(|s| s.label.ch()).collect()
}

/// What the shape analysis concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeVerdict {
    /// The label string, for logs and tests.
    pub labels: String,
    /// `Some((mishear, correct))` when the shape is a learnable correction.
    ///
    /// `mishear` may be **two words joined by a space** — that is the word-merge case
    /// (`whisper flow → WhimprFlow`), and admitting it is a large part of why this
    /// module exists.
    pub correction: Option<(String, String)>,
}

/// Analyse the difference between pasted text and current field content.
///
/// This decides *shape* only. Vocabulary-quality judgements — common-word stoplist,
/// length floors, alphabetic-only — stay in `autolearn::detect_correction`, which
/// applies them to the pair this returns. The two concerns are orthogonal and keeping
/// them apart is deliberate: shape says "this is a correction", quality says "this is
/// a correction worth remembering".
pub fn analyse(pasted: &str, field: &str) -> ShapeVerdict {
    analyse_in_region(pasted, field, &PriorContext::default())
}

/// As [`analyse`], but with the surrounding text that was already in the field removed
/// first — so the shape describes the pasted region rather than the whole document.
///
/// Build the `prior` with [`PriorContext::from_baseline`] from the **first** read of the
/// field. Passing [`PriorContext::default`] means "the field held nothing but the paste",
/// which is what [`analyse`] assumes.
///
/// ⚠️ **No insertion is trimmed beyond what `prior` accounts for.** A trailing insertion
/// that is *not* prior text is the user continuing to type, and it must be allowed to
/// reject the substitution — that is the difference between a correction and a rewrite.
pub fn analyse_in_region(pasted: &str, field: &str, prior: &PriorContext) -> ShapeVerdict {
    let from = word_tokens(pasted);
    let field_toks = word_tokens(field);
    let to = prior.strip(&field_toks);
    if from.is_empty() || to.is_empty() {
        return ShapeVerdict {
            labels: String::new(),
            correction: None,
        };
    }
    let mut steps = align(&from, to);
    mark_capture_errors(&mut steps);
    let labels = label_string(&steps);
    let correction = extract_correction(&steps);
    ShapeVerdict { labels, correction }
}

/// Find the single learnable substitution, if the shape permits one.
///
/// Implements the two accept patterns directly rather than with a regex engine — the
/// patterns are short, and expressing them as index arithmetic keeps the "flanked by
/// unchanged tokens" requirement visible instead of encoded in a string.
fn extract_correction(steps: &[Step]) -> Option<(String, String)> {
    let n = steps.len();
    let unchanged_or_edge = |idx: isize| -> bool {
        if idx < 0 || idx as usize >= n {
            return true; // start/end of region satisfies ^ and $
        }
        steps[idx as usize].label.is_unchanged()
    };

    // Only ONE DISTINCT correction may exist in the region. Two *different* isolated
    // fixes in one dictation would each match locally, but picking one of them is
    // guessing, so the whole observation is discarded.
    //
    // ⚠️ Repetitions of the SAME correction are not ambiguous — they are the same fix
    // applied everywhere the word occurred, which is a stronger signal than one
    // occurrence, not a weaker one. Dictating "monvi and monvi again" and correcting
    // both is one decision by the user.
    //
    // 🔴 An earlier version of this function required literally one substitution and
    // regressed exactly that case. The old set-difference detector had handled it
    // (de-duplicating by lowercased word) and there is a test pinning it, from a real
    // 2026-08-06 fix — `learns_when_the_same_mishear_appears_twice` failed on first run
    // after the rewrite. Caught by the suite, not by review.
    let subs: Vec<usize> = (0..n)
        .filter(|&k| steps[k].label == Label::Substitution)
        .collect();
    if subs.is_empty() {
        return None;
    }
    let same_pair = |a: usize, b: usize| {
        let (fa, ta) = (&steps[a].from, &steps[a].to);
        let (fb, tb) = (&steps[b].from, &steps[b].to);
        match ((fa, ta), (fb, tb)) {
            ((Some(fa), Some(ta)), (Some(fb), Some(tb))) => {
                fa.eq_ignore_ascii_case(fb) && ta.eq_ignore_ascii_case(tb)
            }
            _ => false,
        }
    };
    if !subs.iter().all(|&k| same_pair(k, subs[0])) {
        return None;
    }
    // Every occurrence must independently be a properly flanked correction; one that
    // sits next to an insertion is still continued writing.
    if subs.len() > 1 {
        for &k in &subs {
            let ki = k as isize;
            if !(unchanged_or_edge(ki - 1) && unchanged_or_edge(ki + 1)) {
                return None;
            }
        }
    }
    let s = subs[0];
    let si = s as isize;

    // Shape 1 — isolated substitution: [CMZ]S[CMZ] | ^S[CMZ] | [CMZ]S$
    if unchanged_or_edge(si - 1) && unchanged_or_edge(si + 1) {
        let from = steps[s].from.clone()?;
        let to = steps[s].to.clone()?;
        return Some((from, to));
    }

    // Shape 2 — deletion beside the substitution: a word merge.
    // [CMZ](DS|SD)[CMZ] | ^(DS|SD)[CMZ] | [CMZ](DS|SD)$
    //
    // The mishear is both original tokens joined in their original order; the
    // correction is the single replacement token.
    for (d, outer_lo, outer_hi) in [
        (si - 1, si - 2, si + 1), // D S
        (si + 1, si - 1, si + 2), // S D
    ] {
        if d < 0 || d as usize >= n {
            continue;
        }
        if steps[d as usize].label != Label::Delete {
            continue;
        }
        if !(unchanged_or_edge(outer_lo) && unchanged_or_edge(outer_hi)) {
            continue;
        }
        let deleted = steps[d as usize].from.clone()?;
        let sub_from = steps[s].from.clone()?;
        let to = steps[s].to.clone()?;
        let mishear = if d < si {
            format!("{deleted} {sub_from}")
        } else {
            format!("{sub_from} {deleted}")
        };
        return Some((mishear, to));
    }

    None
}

/// Very common words never learned as a "correction" — avoids poisoning the dictionary
/// from ordinary edits (their/there, then/than, sentence rewording).
const COMMON: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "your", "youre", "with", "this", "that",
    "have", "from", "they", "theyre", "their", "there", "would", "could", "should", "about",
    "then", "than", "them", "these", "those", "here", "were", "well", "will", "what", "when",
    "where", "which", "while", "into", "just", "like", "make", "made", "want", "some", "time",
    "know", "take", "come", "back", "good", "much", "also", "been", "over", "only", "more",
    "most", "very", "even", "such", "many", "does", "done", "same", "sure", "okay", "yeah",
    "hey", "hello", "please", "thanks", "thank", "message", "email", "text", "call", "end",
];

fn is_common(w: &str) -> bool {
    COMMON.contains(&w.to_lowercase().as_str())
}

/// Levenshtein distance normalized by the longer length (0 identical, 1 unrelated).
fn norm_levenshtein(a: &str, b: &str) -> f32 {
    let (a, b) = (a.to_lowercase(), b.to_lowercase());
    let m = a.chars().count().max(b.chars().count());
    if m == 0 {
        return 1.0;
    }
    strsim::levenshtein(&a, &b) as f32 / m as f32
}

/// Is `candidate` a prefix of `original`, short by 2 or more characters?
fn is_truncation_of(candidate: &str, original: &str) -> bool {
    let (c, o) = (candidate.to_lowercase(), original.to_lowercase());
    o.len().saturating_sub(c.len()) >= 2 && o.starts_with(&c)
}

/// Is this pair worth putting in a dictionary? Returns `None` if acceptable, or
/// `Some(reason)` naming the gate that rejected it.
///
/// **Vocabulary quality, deliberately separate from shape.** [`analyse`] answers "did the
/// user correct something"; this answers "is that correction worth remembering". Keeping
/// them apart is what lets the shape rules stay faithful to the reference implementation
/// while these stay tuned to WhimprFlow's own measured junk.
///
/// Every gate here was earned by a junk entry that reached Max's dictionary — `nurmsl`
/// 2026-08-06, `clad` 08-15, `Hypothesis → Hypothe` 08-16 — or by measurement over the
/// real corpus. **The distance gate in particular is not redundant with the shape gate:**
/// replaying 46 real edits on 2026-08-17, shape alone accepted `"Bet" → "commit"` and
/// `"Reimann" → "Domain"`, which are shape-perfect isolated substitutions and complete
/// nonsense as vocabulary. Wispr gets away without a distance gate because it runs a
/// model over the candidate afterwards; WhimprFlow has no such stage.
///
/// `mishear` may contain a single interior space — that is the word-merge case
/// (`"In bed" → "Embed"`), and rejecting it here would undo the main gain of the shape
/// rewrite.
pub fn is_learnable_pair(mishear: &str, correct: &str) -> Option<&'static str> {
    let word_ok = |w: &str| {
        w.chars()
            .all(|c| c.is_alphabetic() || c == ' ')
            && w.split(' ').filter(|p| !p.is_empty()).count() <= 2
    };
    if mishear.chars().count() < 3 || correct.chars().count() < 3 {
        return Some("a word is under 3 chars");
    }
    if !word_ok(mishear) || !word_ok(correct) || correct.contains(' ') {
        return Some("non-alphabetic, or not a single word on the correct side");
    }
    if correct.eq_ignore_ascii_case(mishear) {
        return Some("differs by case only");
    }
    if mishear.split(' ').any(is_common) || is_common(correct) {
        return Some("an ordinary English word");
    }
    let d = norm_levenshtein(mishear, correct);
    if d <= 0.0 || d > 0.6 {
        return Some("not phonetically close enough to look like a mishear");
    }
    if is_truncation_of(correct, mishear) {
        return Some("the correction is a truncation of what it replaced");
    }
    let titled = correct.chars().next().is_some_and(|c| c.is_uppercase());
    if !titled && (mishear.chars().count() < 4 || correct.chars().count() < 4) {
        return Some("lowercase and under 4 chars — not distinctive enough");
    }
    None
}

/// The whole decision: shape, then vocabulary quality.
pub fn learn_from(pasted: &str, field: &str, prior: &PriorContext) -> Option<(String, String)> {
    let (mishear, correct) = analyse_in_region(pasted, field, prior).correction?;
    is_learnable_pair(&mishear, &correct).is_none().then_some((mishear, correct))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(a: &str, b: &str) -> String {
        analyse(a, b).labels
    }
    fn corr(a: &str, b: &str) -> Option<(String, String)> {
        analyse(a, b).correction
    }
    /// The realistic path: `baseline` is the FIRST read of the field, so the prior
    /// context is measured rather than guessed.
    fn corr_in(pasted: &str, baseline: &str, field: &str) -> Option<(String, String)> {
        let prior = PriorContext::from_baseline(pasted, baseline);
        analyse_in_region(pasted, field, &prior).correction
    }
    fn labels_in(pasted: &str, baseline: &str, field: &str) -> String {
        let prior = PriorContext::from_baseline(pasted, baseline);
        analyse_in_region(pasted, field, &prior).labels
    }

    // ── The cases that failed live on 2026-08-17 ──────────────────────────────
    //
    // Every test in this block is a real observation from Max's own dictation, not a
    // constructed example. Together they are the reason this module replaced the set
    // difference.

    /// 🔴 THE headline case. Max corrected `derelict → Dirichlet` inside a Notes note
    /// holding 11 words of unrelated text. The correction was seen and held across
    /// seven consecutive polls, and the set difference rejected it as `-1 +12`.
    #[test]
    fn a_correction_inside_a_field_full_of_prior_text_is_now_found() {
        let pasted = "I'm testing this with derelict";
        let prior = "TODO groceries milk eggs return library books Friday check tire pressure";
        let baseline = format!("{prior} {pasted}");
        let field = format!("{prior} I'm testing this with Dirichlet");
        assert_eq!(
            corr_in(pasted, &baseline, &field),
            Some(("derelict".into(), "Dirichlet".into())),
            "labels were {:?}",
            labels_in(pasted, &baseline, &field)
        );
    }

    /// The same fix, but with the unrelated text *after* the insertion point. This is
    /// the case region-trimming exists for — as trailing `I`s it would read `MMMMSIII`
    /// and match no accept pattern, so auto-learn would work at the end of a document
    /// and silently fail in the middle of one.
    #[test]
    fn prior_text_after_the_insertion_point_is_also_ignored() {
        let pasted = "I'm testing this with derelict";
        let tail = "and then some older notes here";
        let baseline = format!("{pasted} {tail}");
        let field = format!("I'm testing this with Dirichlet {tail}");
        assert_eq!(
            corr_in(pasted, &baseline, &field),
            Some(("derelict".into(), "Dirichlet".into())),
            "labels were {:?}",
            labels_in(pasted, &baseline, &field)
        );
    }

    /// Prior text on BOTH sides — the realistic case for editing mid-document.
    #[test]
    fn prior_text_on_both_sides_is_ignored() {
        let pasted = "testing with derelict";
        let baseline = format!("older stuff above {pasted} and older stuff below");
        let field = "older stuff above testing with Dirichlet and older stuff below";
        assert_eq!(
            corr_in(pasted, &baseline, field),
            Some(("derelict".into(), "Dirichlet".into()))
        );
    }

    /// 🔴 The other dominant real shape: a proper noun misheard as TWO words and fixed
    /// to one. Three of four dictations on 2026-08-17 had this shape and the 1-for-1
    /// requirement rejected every one.
    #[test]
    fn a_two_word_mishear_corrected_to_one_word_is_learnable() {
        assert_eq!(
            corr(
                "Okay I'm dictating this with whisper flow today",
                "Okay I'm dictating this with WhimprFlow today"
            ),
            Some(("whisper flow".into(), "WhimprFlow".into()))
        );
    }

    /// Same shape, observed the same evening: Dirichlet → "They're elect".
    #[test]
    fn the_theyre_elect_case() {
        assert_eq!(
            corr("a note about They're elect today", "a note about Dirichlet today"),
            Some(("They're elect".into(), "Dirichlet".into()))
        );
    }

    /// And Silero → "silo row".
    #[test]
    fn the_silo_row_case() {
        assert_eq!(
            corr("testing with silo row here", "testing with Silero here"),
            Some(("silo row".into(), "Silero".into()))
        );
    }

    // ── The guards that must survive ──────────────────────────────────────────

    /// 🔴 `Hypothesis → Hypothe` — a read taken mid-keystroke. Labelled a capture
    /// error because the replacement is a prefix of the original *at the region
    /// edge*, and `E` satisfies neither accept pattern by construction.
    #[test]
    fn a_mid_keystroke_truncation_is_a_capture_error() {
        let v = analyse("look at the Riemann Hypothesis", "look at the Riemann Hypothe");
        assert_eq!(v.labels, "MMMME", "{v:?}");
        assert_eq!(v.correction, None);
    }

    /// The suffix half of the same rule.
    #[test]
    fn an_edge_suffix_substitution_is_also_a_capture_error() {
        let v = analyse("the value is Whimprflow", "the value is flow");
        assert_eq!(v.labels, "MMME");
        assert_eq!(v.correction, None);
    }

    /// …but a truncation in the *middle* of the region is a real edit, not an
    /// artifact of when we looked, so it is not reclassified.
    #[test]
    fn an_interior_truncation_is_not_a_capture_error() {
        let v = analyse("i use Whimprflows daily", "i use Whimprflow daily");
        assert_eq!(v.labels, "MMSM");
        assert_eq!(
            v.correction,
            Some(("Whimprflows".into(), "Whimprflow".into()))
        );
    }

    #[test]
    fn two_adjacent_substitutions_are_rewriting_not_correcting() {
        let v = analyse("meet at noon monvi", "meet at dawn Manvi");
        assert_eq!(v.correction, None, "{v:?}");
    }

    #[test]
    fn a_substitution_beside_an_insertion_is_rejected() {
        let v = analyse("ping the server foo", "ping the server bar quickly");
        assert_eq!(v.correction, None, "{v:?}");
    }

    #[test]
    fn continued_typing_is_not_a_correction() {
        let v = analyse("here is the start", "here is the start and then a lot more text");
        assert_eq!(v.correction, None, "{v:?}");
        assert!(!v.labels.contains('S'), "{v:?}");
    }

    #[test]
    fn two_separate_fixes_are_too_ambiguous_to_learn_from() {
        // Each would match locally; learning from an ambiguous pair is how junk gets in.
        let v = analyse("monvi went to derelict today", "Manvi went to Dirichlet today");
        assert_eq!(v.correction, None, "{v:?}");
    }

    #[test]
    fn no_change_produces_no_correction() {
        let v = analyse("hello there world", "hello there world");
        assert_eq!(v.labels, "MMM");
        assert_eq!(v.correction, None);
    }

    #[test]
    fn an_emptied_field_produces_nothing() {
        assert_eq!(corr("some dictated words", ""), None);
        assert_eq!(corr("", "some text"), None);
    }

    // ── Casing ────────────────────────────────────────────────────────────────

    /// The set difference lowercased before comparing, so a case-only fix produced an
    /// EMPTY diff and could never be learned — even though `DictionaryStore::apply`
    /// treats case-only entries as first-class and names `Katex → KaTeX` as its
    /// motivating example. The alignment sees it.
    #[test]
    fn a_case_only_correction_is_visible_at_last() {
        let v = analyse("rendered with Katex today", "rendered with KaTeX today");
        assert_eq!(v.labels, "MMCM", "{v:?}");
    }

    /// A casing change may *flank* a substitution — it is in the `[CMZ]` class.
    #[test]
    fn a_casing_change_can_flank_a_substitution() {
        let v = analyse("katex and monvi here", "KaTeX and Manvi here");
        assert_eq!(v.labels, "CMSM", "{v:?}");
        assert_eq!(v.correction, Some(("monvi".into(), "Manvi".into())));
    }

    // ── Label alphabet sanity ─────────────────────────────────────────────────

    #[test]
    fn labels_match_the_reference_alphabet() {
        assert_eq!(Label::Match.ch(), 'M');
        assert_eq!(Label::Casing.ch(), 'C');
        assert_eq!(Label::Substitution.ch(), 'S');
        assert_eq!(Label::Delete.ch(), 'D');
        assert_eq!(Label::Insert.ch(), 'I');
        assert_eq!(Label::CaptureError.ch(), 'E');
        // Only M and C may flank a substitution (`[CMZ]`; WhimprFlow has no Z case).
        assert!(Label::Match.is_unchanged() && Label::Casing.is_unchanged());
        for l in [
            Label::Substitution,
            Label::Delete,
            Label::Insert,
            Label::CaptureError,
        ] {
            assert!(!l.is_unchanged(), "{l:?} must not flank a substitution");
        }
    }

    #[test]
    fn prior_context_is_measured_from_the_first_read() {
        let p = PriorContext::from_baseline("b c d", "prior stuff b c d trailing");
        assert_eq!(p.before, vec!["prior".to_string(), "stuff".to_string()]);
        assert_eq!(p.after, vec!["trailing".to_string()]);
    }

    /// 🔴 The distinction the inferred version could not make. Same pasted text, same
    /// trailing extra word — but in one case it was already there (prior context, so
    /// ignore it) and in the other the user typed it (continued writing, so the
    /// substitution is not isolated and must be rejected).
    #[test]
    fn prior_text_and_newly_typed_text_are_told_apart() {
        let pasted = "ping the server foo";

        // Already there at the first read -> prior context -> the fix is isolated.
        let already = PriorContext::from_baseline(pasted, "ping the server foo quickly");
        assert_eq!(
            analyse_in_region(pasted, "ping the server bar quickly", &already).correction,
            Some(("foo".into(), "bar".into()))
        );

        // NOT there at the first read -> the user is still writing -> reject.
        let fresh = PriorContext::from_baseline(pasted, pasted);
        assert_eq!(
            analyse_in_region(pasted, "ping the server bar quickly", &fresh).correction,
            None
        );
    }

    #[test]
    fn an_interior_insertion_still_blocks_a_substitution() {
        // Prior context only ever strips the OUTSIDE. An insertion inside the region
        // is real evidence of writing and must survive.
        let v = analyse("a b c d", "a b extra X d");
        assert!(v.correction.is_none(), "{v:?}");
    }
}
