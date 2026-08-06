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
    fn add_merges_mishears_case_insensitively() {
        let mut s = store();
        s.add("manvi", vec!["Manvie".into()], DictSource::Auto);
        let e = s.entries.iter().find(|e| e.correct == "Manvi").unwrap();
        assert!(e.mishears.iter().any(|m| m == "Manvie"));
        assert_eq!(s.entries.iter().filter(|e| e.correct.eq_ignore_ascii_case("manvi")).count(), 1);
    }
}
