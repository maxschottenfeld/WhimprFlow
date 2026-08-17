//! Custom dictionary: user vocabulary plus a pre-filter that injects only the
//! entries relevant to a given utterance into the cleanup prompt (fewer distractors
//! → higher LLM precision). Manual entries and auto-learned (✨) entries share the
//! same store; the auto-learn diff engine (needs accessibility reads) layers on top.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cleanup::VocabEntry;

/// How a dictionary entry was created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DictSource {
    Manual,
    Auto,
}

fn default_source() -> DictSource {
    DictSource::Manual
}

/// One vocabulary entry: the authoritative spelling and known mishears.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub correct: String,
    #[serde(default)]
    pub mishears: Vec<String>,
    #[serde(default = "default_source")]
    pub source: DictSource,
}

/// The user's dictionary, persisted as JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DictionaryStore {
    pub entries: Vec<DictionaryEntry>,
}

impl DictionaryStore {
    /// Load from `path`, returning an empty store if missing or unreadable.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist to `path` (creating parent dirs).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(path, json)
    }

    /// Add or merge an entry, de-duplicating by spelling (case-insensitive).
    pub fn add(&mut self, correct: impl Into<String>, mishears: Vec<String>, source: DictSource) {
        let correct = correct.into();
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| e.correct.eq_ignore_ascii_case(&correct))
        {
            for m in mishears {
                if !existing.mishears.iter().any(|x| x.eq_ignore_ascii_case(&m)) {
                    existing.mishears.push(m);
                }
            }
        } else {
            self.entries.push(DictionaryEntry {
                correct,
                mishears,
                source,
            });
        }
    }

    /// Promote an auto-learned entry to an approved one, so it may rewrite text.
    /// Returns true if an entry was found and changed.
    ///
    /// Approval is deliberately modelled as `source: Auto -> Manual` rather than a
    /// separate `approved` flag: an approved suggestion is indistinguishable from a
    /// word the user typed in by hand, both in what it is allowed to do and in how
    /// the Hub should present it, and a second boolean would let the two disagree.
    /// It also means the on-disk format does not change and no migration is needed.
    ///
    /// Idempotent — approving an already-manual entry reports false (nothing to do)
    /// rather than being an error.
    pub fn approve(&mut self, correct: &str) -> bool {
        match self
            .entries
            .iter_mut()
            .find(|e| e.correct.eq_ignore_ascii_case(correct))
        {
            Some(e) if e.source == DictSource::Auto => {
                e.source = DictSource::Manual;
                true
            }
            _ => false,
        }
    }

    /// Remove an entry by its spelling (case-insensitive). Returns true if removed.
    pub fn remove(&mut self, correct: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| !e.correct.eq_ignore_ascii_case(correct));
        self.entries.len() != before
    }

    /// Build a whisper `initial_prompt` from the dictionary, or `None` when there is
    /// nothing to bias toward.
    ///
    /// This is the dictionary's *second* consumer, and the important one. The first,
    /// [`Self::prefilter`], feeds the cleanup LLM — but that only fires when
    /// `needs_cleanup()` says the transcript looks messy, which is a minority of
    /// dictations, so most of the time the dictionary did nothing at all. An
    /// `initial_prompt` biases whisper's decoding directly and applies to *every*
    /// dictation, fixing names at the acoustic level instead of repairing them after
    /// the fact.
    ///
    /// Two deliberate differences from `prefilter`:
    ///
    /// - **It cannot filter by utterance.** The prompt has to be chosen before there
    ///   is a transcript to compare against, so the whole dictionary goes in. That is
    ///   fine: the budget is 224 tokens, which is a lot of proper nouns, and
    ///   `WhisperEngine::fit_prompt` trims from the front if it overflows.
    /// - **Only correct spellings are included, never mishears.** `prefilter` sends
    ///   mishears to the cleanup LLM so it can recognise what to replace. Putting a
    ///   mishear in an ASR prompt would bias decoding *toward* the wrong spelling —
    ///   the exact opposite of the point.
    ///
    /// Entries stay in insertion order, so the newest (most likely to be currently
    /// relevant) sit at the end — which is both what whisper weights most heavily and
    /// what survives front-truncation.
    pub fn asr_prompt(&self) -> Option<String> {
        let terms: Vec<&str> = self
            .entries
            .iter()
            .map(|e| e.correct.trim())
            .filter(|t| !t.is_empty())
            .collect();
        if terms.is_empty() {
            return None;
        }
        // "Glossary: a, b, c." — a plain term list, which measured better than the
        // bare words alone on the proper-noun fixture (3 of 4 mis-spellings fixed).
        Some(format!("Glossary: {}.", terms.join(", ")))
    }

    /// Apply the dictionary's known mishears to `text` deterministically, returning
    /// the corrected text and how many replacements were made.
    ///
    /// This is the dictionary's **third** consumer and the only one that runs on
    /// every dictation with a guaranteed result. [`Self::prefilter`] only reaches the
    /// cleanup LLM, which `needs_cleanup()` skips on the large majority of
    /// dictations; [`Self::asr_prompt`] only *biases* whisper's decoding and can be
    /// overridden by the acoustics. Neither is a promise. This is: if a known mishear
    /// is in the text, it leaves as the correct spelling, with no model involved.
    ///
    /// Rules, all deliberate:
    ///
    /// - **Word boundaries only.** A mishear is matched only when neither side is
    ///   alphanumeric, so an entry for "clad" never rewrites "cladding".
    /// - **Longest pattern first.** Multi-word mishears ("charge bee") win over any
    ///   shorter pattern that overlaps them, so the more specific rule always applies.
    /// - **One pass, left to right, no re-scanning.** Replaced text is never re-examined,
    ///   so A→B→C chains and A→B→A loops are impossible by construction. Two entries
    ///   that disagree produce a stable result rather than an order-dependent one.
    /// - **The dictionary's spelling is authoritative**, used verbatim — that is the
    ///   whole point of an entry that says "ChargeBee", "CUDA" or "KaTeX". The single
    ///   exception is sentence-initial capitalization: a match whose first letter was
    ///   uppercase keeps an uppercase first letter, so "wisper" opening a sentence
    ///   becomes "Whisper" and not "whisper".
    /// - **A mishear identical to its own correct spelling is skipped**, so an entry
    ///   can never be a no-op that still counts as a replacement.
    ///
    /// Cost is O(text × rules) over a handful of entries and a sentence or two of
    /// text — microseconds, with no allocation per candidate position. It belongs on
    /// the hot lane; see the call site in `hotkey.rs`.
    pub fn apply(&self, text: &str) -> (String, usize) {
        let rules = self.replacement_rules();
        if rules.is_empty() || text.is_empty() {
            return (text.to_string(), 0);
        }

        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        let mut hits = 0;

        let is_word = |c: char| c.is_alphanumeric();

        while i < chars.len() {
            // A match may only start on a word boundary.
            let boundary_left = i == 0 || !is_word(chars[i - 1]);
            let mut matched = None;
            if boundary_left {
                for (pattern, replacement) in &rules {
                    let end = i + pattern.len();
                    if end > chars.len() {
                        continue;
                    }
                    let same = chars[i..end]
                        .iter()
                        .zip(pattern.iter())
                        .all(|(a, b)| a.to_lowercase().eq(b.to_lowercase()));
                    if !same {
                        continue;
                    }
                    // …and end on one.
                    if end < chars.len() && is_word(chars[end]) {
                        continue;
                    }
                    matched = Some((end, replacement));
                    break;
                }
            }

            match matched {
                Some((end, replacement)) => {
                    let was_capitalized = chars[i..end]
                        .iter()
                        .find(|c| c.is_alphabetic())
                        .is_some_and(|c| c.is_uppercase());
                    push_replacement(&mut out, replacement, was_capitalized);
                    hits += 1;
                    i = end;
                }
                None => {
                    out.push(chars[i]);
                    i += 1;
                }
            }
        }

        (out, hits)
    }

    /// Flatten the store into `(mishear, correct)` character patterns, longest first.
    ///
    /// Sorting by length is what makes overlapping entries deterministic: with rules
    /// for both "charge bee" and "bee", the longer one is always tried first, so the
    /// result does not depend on the order entries happen to sit in the JSON file.
    fn replacement_rules(&self) -> Vec<(Vec<char>, &str)> {
        let mut rules: Vec<(Vec<char>, &str)> = Vec::new();
        for e in &self.entries {
            // 🔴 Only APPROVED entries may rewrite text. An auto-learned entry is a
            // *suggestion*: it still reaches `asr_prompt()` and biases whisper's
            // decoding, but it does not get to edit what lands in Max's document
            // until he approves it (which flips `source` to `Manual`).
            //
            // This is the split Wispr Flow ships — its auto-learned entries carry
            // `replacement = NULL` and are vocabulary bias only, never a
            // find-and-replace rule (verified against Max's own `flow.sqlite`:
            // all 6 `user_edits` rows, `replacement` NULL). The reason matters,
            // because auto-learn infers from a single observation and has produced
            // junk three times — `nurmsl` 08-06, `clad` 08-15, `Hypothesis ->
            // Hypothe` 08-16 — each caught by Max or by luck, never by the system.
            // Wispr's own auto-learn is no better (2 of its 6 entries are typos);
            // it is harmless there *only* because those entries cannot rewrite.
            //
            // ⚠️ Bias alone is NOT a substitute for this stage, which is why the
            // approve step exists rather than dropping rewriting altogether.
            // Measured on the harness 2026-08-17 over `fixtures/propernouns.wav`:
            // the glossary fixed `WimpherFlow -> Whimprflow`, a surname it had been
            // mishearing, and `Celero -> Silero`, but left `Tauri` as "entry",
            // and it *introduced* `Whisper -> WISPR`, a word that was correct
            // without the prompt and is not in the glossary at all. Bias is
            // probabilistic and has collateral effects on neighbours; only this
            // function is a guarantee.
            if e.source != DictSource::Manual {
                continue;
            }
            let correct = e.correct.trim();
            if correct.is_empty() {
                continue;
            }
            for m in &e.mishears {
                let m = m.trim();
                // Only an *exact* match is a no-op worth skipping. A mishear that
                // differs from the correct spelling by case alone is a real
                // correction — "Katex" → "KaTeX" is precisely the kind of entry a
                // user adds by hand — so it must not be filtered out here.
                if m.is_empty() || m == correct {
                    continue;
                }
                rules.push((m.chars().collect(), correct));
            }
        }
        rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        rules
    }
}

/// Append `replacement`, upper-casing its first character when the text it replaced
/// started with a capital. Everything after the first character is left exactly as the
/// dictionary spells it, so internal capitals ("ChargeBee", "KaTeX") survive.
fn push_replacement(out: &mut String, replacement: &str, capitalize: bool) {
    let mut cs = replacement.chars();
    match cs.next() {
        Some(first) if capitalize && first.is_lowercase() => {
            out.extend(first.to_uppercase());
            out.push_str(cs.as_str());
        }
        Some(first) => {
            out.push(first);
            out.push_str(cs.as_str());
        }
        None => {}
    }
}

impl DictionaryStore {
    /// Select the entries relevant to `utterance` — those whose spelling or a known
    /// mishear is edit-close to a spoken token (or adjacent token pair, to catch
    /// split words like "charge bee" → "ChargeBee") — capped to `max`.
    pub fn prefilter(&self, utterance: &str, max: usize) -> Vec<VocabEntry> {
        let toks: Vec<String> = utterance
            .split_whitespace()
            .map(|t| {
                t.trim_matches(|c: char| c.is_ascii_punctuation())
                    .to_lowercase()
            })
            .filter(|t| !t.is_empty())
            .collect();

        let mut grams: Vec<String> = toks.clone();
        for w in toks.windows(2) {
            grams.push(format!("{}{}", w[0], w[1]));
        }

        let mut out = Vec::new();
        for e in &self.entries {
            let targets: Vec<String> = std::iter::once(e.correct.to_lowercase())
                .chain(e.mishears.iter().map(|m| m.to_lowercase()))
                .collect();
            if grams.iter().any(|g| targets.iter().any(|t| close(g, t))) {
                out.push(VocabEntry {
                    correct: e.correct.clone(),
                    mishears: e.mishears.clone(),
                });
                if out.len() >= max {
                    break;
                }
            }
        }
        out
    }
}

/// Two tokens are "close" if identical or within a normalized edit distance of 0.34.
fn close(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let maxlen = a.chars().count().max(b.chars().count());
    if maxlen == 0 {
        return false;
    }
    (strsim::levenshtein(a, b) as f32 / maxlen as f32) <= 0.34
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> DictionaryStore {
        let mut s = DictionaryStore::default();
        s.add("Manvi", vec!["Monvi".into(), "Manvee".into()], DictSource::Manual);
        s.add("ChargeBee", vec!["charge bee".into()], DictSource::Manual);
        s
    }

    #[test]
    fn prefilter_selects_close_mishear() {
        // "monvi" is an exact mishear of Manvi.
        let v = store().prefilter("send the deck to monvi please", 15);
        assert!(v.iter().any(|e| e.correct == "Manvi"));
        assert!(!v.iter().any(|e| e.correct == "ChargeBee"));
    }

    #[test]
    fn prefilter_catches_split_word_via_bigram() {
        // "charge bee" spoken as two words → bigram "chargebee" matches.
        let v = store().prefilter("we should renew charge bee this month", 15);
        assert!(v.iter().any(|e| e.correct == "ChargeBee"));
    }

    #[test]
    fn prefilter_ignores_unrelated_utterance() {
        let v = store().prefilter("the weather is nice today", 15);
        assert!(v.is_empty());
    }

    #[test]
    fn asr_prompt_lists_correct_spellings_only() {
        let p = store().asr_prompt().unwrap();
        assert!(p.contains("Manvi"), "{p}");
        assert!(p.contains("ChargeBee"), "{p}");
        // Mishears must never appear: they would bias decoding toward the error.
        assert!(!p.contains("Monvi"), "{p}");
        assert!(!p.contains("charge bee"), "{p}");
    }

    #[test]
    fn asr_prompt_is_none_when_empty() {
        assert!(DictionaryStore::default().asr_prompt().is_none());
    }

    #[test]
    fn asr_prompt_preserves_insertion_order() {
        // Newest last, so it survives front-truncation and gets whisper's heavier
        // late-token weighting.
        let mut s = store();
        s.add("Zylophone", vec![], DictSource::Auto);
        let p = s.asr_prompt().unwrap();
        assert!(p.find("Manvi").unwrap() < p.find("Zylophone").unwrap(), "{p}");
    }

    #[test]
    fn apply_replaces_a_known_mishear() {
        let (out, n) = store().apply("send the deck to Monvi please");
        assert_eq!(out, "send the deck to Manvi please");
        assert_eq!(n, 1);
    }

    #[test]
    fn apply_is_case_insensitive_when_matching() {
        // Whisper's casing of a mishear is not something we can rely on.
        let (out, _) = store().apply("send it to MONVI and monvi");
        assert_eq!(out, "send it to Manvi and Manvi");
    }

    #[test]
    fn apply_respects_word_boundaries() {
        // "monvi" inside a longer word is not the word we learned.
        let (out, n) = store().apply("the monvirus spread and premonvi too");
        assert_eq!(out, "the monvirus spread and premonvi too");
        assert_eq!(n, 0);
    }

    #[test]
    fn apply_handles_multi_word_mishears() {
        let (out, n) = store().apply("we should renew charge bee this month");
        assert_eq!(out, "we should renew ChargeBee this month");
        assert_eq!(n, 1);
    }

    #[test]
    fn apply_prefers_the_longest_pattern() {
        // With rules for both "charge bee" and "bee", the longer one must win
        // regardless of the order the entries sit in the store.
        let mut s = DictionaryStore::default();
        s.add("Bee", vec!["bee".into()], DictSource::Manual);
        s.add("ChargeBee", vec!["charge bee".into()], DictSource::Manual);
        let (out, n) = s.apply("renew charge bee now");
        assert_eq!(out, "renew ChargeBee now");
        assert_eq!(n, 1);
    }

    #[test]
    fn apply_uses_the_dictionary_spelling_verbatim() {
        // Internal capitals are the reason the entry exists — never normalize them.
        let mut s = DictionaryStore::default();
        s.add("KaTeX", vec!["Katex".into()], DictSource::Manual);
        let (out, _) = s.apply("rendered with Katex today");
        assert_eq!(out, "rendered with KaTeX today");
    }

    #[test]
    fn apply_capitalizes_at_the_start_of_a_sentence() {
        // The one deliberate deviation from "verbatim": a capitalized match keeps a
        // capital, so a lowercase entry does not lowercase the start of a sentence.
        let mut s = DictionaryStore::default();
        s.add("whisper", vec!["wisper".into()], DictSource::Manual);
        let (out, _) = s.apply("Wisper is the model. I use wisper daily.");
        assert_eq!(out, "Whisper is the model. I use whisper daily.");
    }

    #[test]
    fn apply_never_chains_replacements() {
        // Replaced text is not re-scanned, so A→B and B→C cannot compose into A→C,
        // and A→B / B→A cannot loop. This is the property that makes the stage safe
        // to run unconditionally on the hot lane.
        let mut s = DictionaryStore::default();
        s.add("bravo", vec!["alpha".into()], DictSource::Manual);
        s.add("charlie", vec!["bravo".into()], DictSource::Manual);
        let (out, n) = s.apply("alpha");
        assert_eq!(out, "bravo");
        assert_eq!(n, 1);
    }

    #[test]
    fn apply_fixes_a_case_only_mishear() {
        // Case alone is a real correction, not a no-op — fixing "Katex" to "KaTeX" is
        // exactly the kind of entry a user adds by hand.
        let mut s = DictionaryStore::default();
        s.add("Manvi", vec!["manvi".into()], DictSource::Manual);
        let (out, n) = s.apply("hello manvi");
        assert_eq!(out, "hello Manvi");
        assert_eq!(n, 1);
    }

    #[test]
    fn apply_skips_a_mishear_identical_to_its_correct_spelling() {
        // An exactly-equal pair is the only true no-op, and it must not inflate the
        // replacement count.
        let mut s = DictionaryStore::default();
        s.add("Manvi", vec!["Manvi".into()], DictSource::Manual);
        let (out, n) = s.apply("hello Manvi");
        assert_eq!(out, "hello Manvi");
        assert_eq!(n, 0);
    }

    #[test]
    fn apply_is_a_no_op_on_an_empty_dictionary() {
        let (out, n) = DictionaryStore::default().apply("nothing to do here");
        assert_eq!(out, "nothing to do here");
        assert_eq!(n, 0);
    }

    #[test]
    fn apply_leaves_unrelated_text_byte_identical() {
        let text = "The weather is nice today — 100% nicer, in fact.";
        let (out, n) = store().apply(text);
        assert_eq!(out, text);
        assert_eq!(n, 0);
    }

    #[test]
    fn apply_matches_against_punctuation_boundaries() {
        let (out, n) = store().apply("ask Monvi, then tell (monvi) again.");
        assert_eq!(out, "ask Manvi, then tell (Manvi) again.");
        assert_eq!(n, 2);
    }

    #[test]
    fn add_merges_mishears_case_insensitively() {
        let mut s = store();
        s.add("manvi", vec!["Manvie".into()], DictSource::Auto);
        let e = s.entries.iter().find(|e| e.correct == "Manvi").unwrap();
        assert!(e.mishears.iter().any(|m| m == "Manvie"));
        assert_eq!(s.entries.iter().filter(|e| e.correct.eq_ignore_ascii_case("manvi")).count(), 1);
    }

    /// 🔴 The load-bearing half of the split: an unreviewed auto-learned entry must
    /// never edit the user's text. This is the guard against §8.13 — auto-learn
    /// infers from a single observation and has produced junk three times, and
    /// `apply()` runs on every dictation.
    #[test]
    fn an_auto_entry_never_rewrites_text() {
        let mut s = DictionaryStore::default();
        s.add("Whimprflow", vec!["Wimperslow".into()], DictSource::Auto);
        let (out, n) = s.apply("i use Wimperslow daily");
        assert_eq!(out, "i use Wimperslow daily", "an unapproved entry rewrote text");
        assert_eq!(n, 0);
    }

    /// …and the other half: it is a *suggestion*, not a no-op. It still reaches
    /// whisper as decoding bias, which is exactly Wispr Flow's `replacement = NULL`
    /// design. Losing this would make auto-learn worthless rather than cautious.
    #[test]
    fn an_auto_entry_still_biases_recognition() {
        let mut s = DictionaryStore::default();
        s.add("Whimprflow", vec!["Wimperslow".into()], DictSource::Auto);
        let p = s.asr_prompt().expect("an auto entry must still reach the ASR prompt");
        assert!(p.contains("Whimprflow"), "{p}");
    }

    #[test]
    fn approving_an_auto_entry_grants_it_rewrite_authority() {
        let mut s = DictionaryStore::default();
        s.add("Whimprflow", vec!["Wimperslow".into()], DictSource::Auto);
        assert!(s.approve("Whimprflow"));
        let (out, n) = s.apply("i use Wimperslow daily");
        assert_eq!(out, "i use Whimprflow daily");
        assert_eq!(n, 1);
    }

    #[test]
    fn approve_matches_case_insensitively_and_is_idempotent() {
        let mut s = DictionaryStore::default();
        s.add("Whimprflow", vec!["Wimperslow".into()], DictSource::Auto);
        // Case-insensitive, like `add` and `remove`.
        assert!(s.approve("whimprflow"));
        // Already manual — nothing left to do, and not an error.
        assert!(!s.approve("Whimprflow"));
        assert!(!s.approve("NeverHeardOfIt"));
    }

    /// Approval must not disturb the entry's data — only its authority. A rename or
    /// a dropped mishear here would silently change what `apply()` matches.
    #[test]
    fn approve_changes_only_the_source() {
        let mut s = DictionaryStore::default();
        s.add("Whimprflow", vec!["Wimperslow".into(), "Wimpher Flow".into()], DictSource::Auto);
        let before = s.entries[0].clone();
        assert!(s.approve("Whimprflow"));
        let after = &s.entries[0];
        assert_eq!(after.correct, before.correct);
        assert_eq!(after.mishears, before.mishears);
        assert_eq!(after.source, DictSource::Manual);
    }
}
