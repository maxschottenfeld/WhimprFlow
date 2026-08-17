//! Auto-learn: after WhimprFlow pastes dictated text, watch the focused text field
//! for a few seconds. If the user corrects a single distinctive word (typically a
//! mis-heard name), diff it out and add it to the dictionary — so next time ASR/
//! cleanup spell it right. This is the signal source Wispr's ✨ sparkle needs.
//!
//! It is deliberately conservative: it only learns on a clean one-word substitution
//! into an otherwise-unchanged field, where the new word looks like a proper noun
//! and is phonetically close to the word it replaced. That avoids poisoning the
//! dictionary with common-word edits. Reads use the Accessibility API and only run
//! when Accessibility is granted.

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::{c_char, c_void};
    use std::ptr;
    use std::time::Duration;

    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type AXUIElementRef = *const c_void;

    const KCF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    /// Gaps between successive reads of the focused field, in milliseconds.
    ///
    /// This used to be a single 7-second sleep followed by one read, despite the
    /// module docs claiming it "watches" the field. That one-shot window failed in
    /// both directions: correct a word at 8 seconds and it was never seen at all;
    /// correct it at 3 seconds and keep typing, and the 7-second snapshot contained
    /// the later edits too, which tripped the one-word-in/one-word-out filter.
    ///
    /// The schedule is dense early and sparse later — an early read is more likely
    /// to catch the correction *before* further edits pile up on top of it, and the
    /// captured `AXUIElementRef` is also less likely to have gone stale. The first
    /// clean one-for-one swap wins and polling stops.
    ///
    /// Cumulative: 1.2, 2.4, 4, 6, 8, 11, 14, 17, 20s.
    const POLL_GAPS_MS: &[u64] = &[1200, 1200, 1600, 2000, 2000, 3000, 3000, 3000, 3000];

    /// Give up if the field cannot be read this many times in a row. A captured
    /// element goes stale when the app re-renders or focus moves; there is no point
    /// polling a dead reference for the full 20 seconds.
    const MAX_CONSECUTIVE_READ_FAILURES: usize = 4;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFStringGetLength(s: CFStringRef) -> isize;
        fn CFStringGetCString(s: CFStringRef, buf: *mut c_char, size: isize, encoding: u32) -> bool;
        fn CFStringGetMaximumSizeForEncoding(len: isize, encoding: u32) -> isize;
        fn CFGetTypeID(cf: CFTypeRef) -> usize;
        fn CFStringGetTypeID() -> usize;
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut CFTypeRef,
        ) -> i32;
    }

    fn make_cfstring(s: &str) -> CFStringRef {
        let Ok(c) = std::ffi::CString::new(s) else {
            return ptr::null();
        };
        unsafe { CFStringCreateWithCString(ptr::null(), c.as_ptr(), KCF_STRING_ENCODING_UTF8) }
    }

    /// Convert a CFStringRef to a Rust String (None if it isn't actually a string).
    unsafe fn cfstring_to_string(s: CFStringRef) -> Option<String> {
        if s.is_null() || CFGetTypeID(s) != CFStringGetTypeID() {
            return None;
        }
        let len = CFStringGetLength(s);
        let max = CFStringGetMaximumSizeForEncoding(len, KCF_STRING_ENCODING_UTF8) + 1;
        if max <= 0 {
            return Some(String::new());
        }
        let mut buf = vec![0i8; max as usize];
        if CFStringGetCString(s, buf.as_mut_ptr(), max, KCF_STRING_ENCODING_UTF8) {
            std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_str()
                .ok()
                .map(|x| x.to_string())
        } else {
            None
        }
    }

    /// Copy the system-wide focused UI element (retained — caller CFReleases it).
    unsafe fn copy_focused_element() -> AXUIElementRef {
        let system = AXUIElementCreateSystemWide();
        if system.is_null() {
            return ptr::null();
        }
        let attr = make_cfstring("AXFocusedUIElement");
        let mut focused: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(system, attr, &mut focused);
        if !attr.is_null() {
            CFRelease(attr);
        }
        CFRelease(system);
        if err != 0 {
            return ptr::null();
        }
        focused as AXUIElementRef
    }

    /// Read a text element's AXValue as a string.
    unsafe fn element_value(element: AXUIElementRef) -> Option<String> {
        if element.is_null() {
            return None;
        }
        let attr = make_cfstring("AXValue");
        let mut value: CFTypeRef = ptr::null();
        let err = AXUIElementCopyAttributeValue(element, attr, &mut value);
        if !attr.is_null() {
            CFRelease(attr);
        }
        if err != 0 || value.is_null() {
            return None;
        }
        let s = cfstring_to_string(value);
        CFRelease(value);
        s
    }

    /// A raw AX pointer we deliberately move to the observer thread. Safe because
    /// CF/AX types are internally thread-safe and we retain it before sending.
    struct SendPtr(AXUIElementRef);
    unsafe impl Send for SendPtr {}

    /// Right after paste, snapshot the focused field, then check it once after a
    /// short delay for a one-word correction to learn.
    pub fn watch_correction(inserted: &str) {
        // Reads require Accessibility; also skip trivial dictations.
        if !crate::paste::is_trusted() || crate::autolearn::word_tokens(inserted).len() < 2 {
            return;
        }
        let inserted = inserted.to_string();
        let focused = unsafe { copy_focused_element() };
        if focused.is_null() {
            // Observability only. This path used to return in total silence, and that
            // silence is what made "highlight-to-add hasn't been working" undiagnosable
            // from the logs: on 2026-08-15 all 57 dictations produced no auto-learn
            // output whatsoever, and this was the only branch that could account for it
            // (paste never failed, so Accessibility was granted, and the dictations were
            // far longer than the two-token floor). Which apps fail to expose a focused
            // element is still unmeasured — this line is what will measure it.
            eprintln!(
                "[whimpr] auto-learn: no focused UI element from the frontmost app \
                 — not watching this dictation"
            );
            return;
        }
        let holder = SendPtr(focused);
        std::thread::spawn(move || {
            // Force whole-struct capture (2021 disjoint captures would otherwise grab
            // the raw pointer field and lose the `Send` impl on `SendPtr`).
            let holder = holder;

            let mut reads_ok = 0usize;
            let mut consecutive_failures = 0usize;
            let mut learned = false;

            // Diagnostic state only — never consulted by the decision path below.
            // "N read(s), no clean one-word correction found" was logged 93 times out
            // of 95 sessions between 08-09 and 08-16 and never once said which gate
            // did the rejecting, so every "auto-learn doesn't work" report had to be
            // answered by guessing. These three track the shape of what was seen.
            let mut last_text: Option<String> = None;
            let mut empty_reads = 0usize;
            // The read that came CLOSEST to being a correction, i.e. the smallest
            // non-zero `removed + added`. Explaining the *last* read instead was
            // actively misleading on the first run: the field's final state was a
            // placeholder, so the summary blamed a 12-word diff while poll 3 had seen
            // a clean 1-for-1 swap and a gate had thrown it away. The near-miss is the
            // interesting read; the final one only says whether the text was sent.
            let mut best: Option<(usize, String)> = None;

            for (i, gap) in POLL_GAPS_MS.iter().enumerate() {
                std::thread::sleep(Duration::from_millis(*gap));
                match unsafe { element_value(holder.0) } {
                    Some(after) => {
                        reads_ok += 1;
                        consecutive_failures = 0;
                        // Per-poll trace. The end-of-session summary reports only the
                        // FINAL read, and on the first real run (2026-08-16, Max's
                        // "option one" test) that hid the one thing worth knowing:
                        // whether the user's correction was ever visible at all. The
                        // final read was Claude Code's *placeholder* text, so the
                        // whole edit window was invisible between two log lines.
                        let (nrem, nadd) = super::diff_shape(&inserted, &after);
                        eprintln!(
                            "[whimpr] auto-learn: poll {}/{} -{nrem} +{nadd} field={:?}",
                            i + 1,
                            POLL_GAPS_MS.len(),
                            super::preview_text(&after),
                        );
                        if after.trim().is_empty() {
                            // The field emptied under us — the overwhelmingly likely
                            // cause is the user sending the message. Worth counting
                            // separately: a correction made before the send is
                            // invisible by the time we look.
                            empty_reads += 1;
                        } else {
                            last_text = Some(after.clone());
                            let score = nrem + nadd;
                            if score > 0 && best.as_ref().is_none_or(|(b, _)| score < *b) {
                                best = Some((score, after.clone()));
                            }
                        }
                        if let Some((mishear, correct)) =
                            super::detect_correction(&inserted, &after)
                        {
                            eprintln!(
                                "[whimpr] ✨ auto-learned: \"{mishear}\" -> \"{correct}\" \
                                 (poll {}/{})",
                                i + 1,
                                POLL_GAPS_MS.len()
                            );
                            crate::hotkey::dictionary_learn(correct, vec![mishear]);
                            learned = true;
                            break;
                        }
                    }
                    None => {
                        consecutive_failures += 1;
                        if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES {
                            break;
                        }
                    }
                }
            }
            unsafe { CFRelease(holder.0) };

            // Say which of the two failure modes happened. Previously they were
            // indistinguishable, which made every "auto-learn doesn't work" report
            // impossible to act on: a field that was never readable and a field with
            // no correction in it both produced total silence.
            if !learned {
                if reads_ok == 0 {
                    eprintln!(
                        "[whimpr] auto-learn: never read the focused field \
                         (element went stale, or focus moved) — nothing to learn from"
                    );
                } else {
                    eprintln!(
                        "[whimpr] auto-learn: {reads_ok} read(s), no clean one-word \
                         correction found"
                    );
                    // ...and *why*. Judged on the last non-empty read, because that
                    // is the state closest to what the user actually left behind.
                    match best.as_ref().map(|(_, t)| t).or(last_text.as_ref()) {
                        Some(after) => eprintln!(
                            "[whimpr] auto-learn: WHY: {} (closest read; empty reads: \
                             {empty_reads}/{reads_ok})",
                            super::rejection_reason(&inserted, after)
                        ),
                        None => eprintln!(
                            "[whimpr] auto-learn: WHY: every read was an empty field \
                             ({empty_reads}/{reads_ok}) — text was almost certainly sent \
                             or cleared before we looked"
                        ),
                    }
                }
            }
        });
    }
}

#[cfg(target_os = "macos")]
pub use imp::watch_correction;

#[cfg(not(target_os = "macos"))]
pub fn watch_correction(_inserted: &str) {}

/// Split into alphanumeric word tokens (punctuation stripped), original case kept.
pub fn word_tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Very common words we never learn as a "correction" — avoids dictionary poisoning
/// from ordinary edits (their/there, your/you're, then/than, sentence rewording…).
const COMMON: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "your", "youre", "with", "this", "that",
    "have", "from", "they", "theyre", "their", "there", "would", "could", "should", "about",
    "then", "than", "them", "these", "those", "here", "were", "well", "will", "what", "when",
    "where", "which", "while", "your", "into", "just", "like", "make", "made", "want", "some",
    "time", "know", "take", "come", "back", "good", "much", "also", "been", "over", "only",
    "more", "most", "very", "even", "such", "many", "does", "done", "same", "sure", "okay",
    "yeah", "hey", "hello", "please", "thanks", "thank", "message", "email", "text", "call",
];

/// The distinct words removed from `inserted` and added in `after`, as a set
/// difference. `None` when either side has no words at all.
///
/// De-duplicated: a word appearing twice is still one distinct change — the version
/// before 2026-08-06 filtered the token *Vec* while testing set membership, so
/// dictating "monvi and monvi" and fixing both produced `removed.len() == 2` and was
/// silently rejected as "too ambiguous".
///
/// ⚠️ **This is a difference over the WHOLE field**, which is the single biggest
/// reason auto-learn rejects real corrections: every word already in the target field
/// that was not in the dictation counts as "added". A one-word fix inside a field that
/// held any prior text is therefore never a clean 1-for-1 swap.
fn word_diff(inserted: &str, after: &str) -> Option<(Vec<String>, Vec<String>)> {
    use std::collections::HashSet;
    let ins = word_tokens(inserted);
    let aft = word_tokens(after);
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
    Some((diff(&ins, &aft_lc), diff(&aft, &ins_lc)))
}

/// Explain, **for the log only**, why [`detect_correction`] rejected a field read.
///
/// This mirrors that function's gate order rather than being wired into it, so the
/// decision path stays byte-for-byte unchanged and switching the diagnostic on cannot
/// alter what gets learned. It exists because `"no clean one-word correction found"`
/// was logged on 93 of 95 watch sessions (2026-08-09 → 08-16) without ever saying
/// which gate fired — leaving every "auto-learn doesn't work" report unanswerable.
///
/// ⚠️ Keep the gate order in sync with [`detect_correction`]; the shared front half is
/// [`word_diff`], but the checks below are deliberately duplicated so that a change
/// here can never change behaviour there.
pub fn rejection_reason(inserted: &str, after: &str) -> String {
    let Some((removed, added)) = word_diff(inserted, after) else {
        return "one side had no words at all (field empty, or nothing pasted)".into();
    };

    if removed.is_empty() && added.is_empty() {
        // No visible change at all. Worth its own message because one important case
        // lands here rather than at any gate: `word_diff` compares lowercased words,
        // so a CASE-ONLY correction cancels out and is invisible before any gate runs.
        // `DictionaryStore::apply` explicitly supports case-only entries and names
        // "Katex" -> "KaTeX" as its motivating example — but auto-learn can never
        // produce one. (This is also why the `eq_ignore_ascii_case` check inside
        // `detect_correction` is unreachable in practice: a pair that differs by case
        // alone never survives the diff to reach it.)
        return "no word-level difference between the pasted text and the field \
                — note the diff is CASE-INSENSITIVE, so a case-only fix such as \
                \"Katex\" -> \"KaTeX\" is invisible to auto-learn by construction"
            .into();
    }

    if removed.len() != 1 || added.len() != 1 {
        return format!(
            "not a 1-for-1 swap — {} word(s) removed {:?}, {} word(s) added {:?} \
             (whole-field diff: text already in the field counts as added)",
            removed.len(),
            preview(&removed),
            added.len(),
            preview(&added),
        );
    }

    let mishear = &removed[0];
    let correct = &added[0];
    let alpha = |w: &str| w.chars().all(|c| c.is_alphabetic());

    if mishear.chars().count() < 3 || correct.chars().count() < 3 {
        return format!("\"{mishear}\" -> \"{correct}\": rejected, a word is under 3 chars");
    }
    if !alpha(mishear) || !alpha(correct) {
        return format!("\"{mishear}\" -> \"{correct}\": rejected, non-alphabetic");
    }
    if correct.eq_ignore_ascii_case(mishear) {
        // Unreachable in practice — see the empty-diff branch above. Kept so the two
        // gate sequences stay structurally aligned.
        return format!("\"{mishear}\" -> \"{correct}\": rejected, differs by case only");
    }
    if is_common(correct) || is_common(mishear) {
        return format!("\"{mishear}\" -> \"{correct}\": rejected, an ordinary English word");
    }
    let d = norm_levenshtein(mishear, correct);
    if d <= 0.0 || d > 0.6 {
        return format!(
            "\"{mishear}\" -> \"{correct}\": rejected, normalized edit distance {d:.2} \
             outside (0.0, 0.6] — not phonetically close enough to look like a mishear"
        );
    }
    let titled = correct.chars().next().is_some_and(|c| c.is_uppercase());
    if !titled && (mishear.chars().count() < 4 || correct.chars().count() < 4) {
        return format!(
            "\"{mishear}\" -> \"{correct}\": rejected, lowercase and under 4 chars \
             (not distinctive enough)"
        );
    }

    format!("\"{mishear}\" -> \"{correct}\": all gates passed (unexpected — should have been learned)")
}

/// First few words of a diff list, so a 200-word field does not fill the log.
fn preview(words: &[String]) -> Vec<&str> {
    words.iter().take(4).map(|w| w.as_str()).collect()
}

/// How many distinct words differ each way, for the per-poll trace. `(0, 0)` when
/// either side is wordless.
pub fn diff_shape(inserted: &str, after: &str) -> (usize, usize) {
    word_diff(inserted, after).map_or((0, 0), |(rem, add)| (rem.len(), add.len()))
}

/// A short, single-line window onto the field, so the trace shows *what* was read
/// without dumping a whole document into the log on every poll.
///
/// This exists because the field turned out to contain things no one predicted —
/// Claude Code reports its empty input box as the placeholder `"Type / for commands"`,
/// which is why the emptied-field check never fired on the first real run.
pub fn preview_text(s: &str) -> String {
    const MAX: usize = 48;
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX {
        return flat;
    }
    let head: String = flat.chars().take(MAX).collect();
    format!("{head}…")
}

/// Detect a single clean one-word correction: exactly one word removed from the
/// inserted text and one word added in the field, both distinctive and phonetically
/// close, with the new word looking like a proper noun. Returns `(mishear, correct)`.
pub fn detect_correction(inserted: &str, after: &str) -> Option<(String, String)> {
    let (removed, added) = word_diff(inserted, after)?;
    if removed.len() != 1 || added.len() != 1 {
        return None; // only learn on a clean 1-for-1 swap
    }
    let mishear = removed[0].clone();
    let correct = added[0].clone();

    let alpha = |w: &str| w.chars().all(|c| c.is_alphabetic());
    if mishear.chars().count() < 3 || correct.chars().count() < 3 {
        return None;
    }
    if !alpha(&mishear) || !alpha(&correct) {
        return None;
    }
    if correct.eq_ignore_ascii_case(&mishear) {
        return None;
    }
    if is_common(&correct) || is_common(&mishear) {
        return None;
    }
    // The correction must be phonetically close to what it replaced — a real
    // mishear, not an unrelated rewrite.
    let d = norm_levenshtein(&mishear, &correct);
    if d <= 0.0 || d > 0.6 {
        return None;
    }

    // Two accept paths.
    //
    // Titlecase was previously the *only* one, on the theory that auto-learn exists
    // for proper nouns. But it silently threw away every lowercase correction —
    // "wisper" -> "whisper", "tarry" -> "tauri", "cuda" -> "CUDA" — which is a large
    // share of what actually gets mis-heard when you dictate technical text, and it
    // does not match how the feature was described ("if I highlight a word, delete
    // it, and replace it with another one, it'll save it").
    //
    // So: Titlecase still qualifies on its own, and a lowercase pair qualifies when
    // both words are long enough to be distinctive and neither is an ordinary
    // English word. That second path is looser, which is the point, but it is still
    // fenced by everything above: exactly one word swapped, both >= 3 chars, both
    // alphabetic, neither in COMMON, and phonetically close. The realistic failure
    // is one junk dictionary entry, which is visible in the Hub and removable in a
    // click — a much smaller cost than silently learning nothing.
    let titled = correct.chars().next().is_some_and(|c| c.is_uppercase());
    const MIN_LOWERCASE_LEN: usize = 4;
    let distinctive_lowercase = mishear.chars().count() >= MIN_LOWERCASE_LEN
        && correct.chars().count() >= MIN_LOWERCASE_LEN;

    if titled || distinctive_lowercase {
        Some((mishear, correct))
    } else {
        None
    }
}

fn is_common(w: &str) -> bool {
    let lc = w.to_lowercase();
    COMMON.contains(&lc.as_str())
}

/// Levenshtein distance normalized by the longer length (0 = identical, 1 = totally
/// different).
fn norm_levenshtein(a: &str, b: &str) -> f32 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    let m = a.chars().count().max(b.chars().count());
    if m == 0 {
        return 1.0;
    }
    strsim::levenshtein(&a, &b) as f32 / m as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_a_name_correction() {
        // We inserted "monvi"; the user fixed it to "Manvi".
        let got = detect_correction("send the deck to monvi please", "send the deck to Manvi please");
        assert_eq!(got, Some(("monvi".to_string(), "Manvi".to_string())));
    }

    #[test]
    fn ignores_common_word_edits() {
        // "there" -> "their" is a common-word edit, never learned.
        assert_eq!(detect_correction("i left there bag", "i left their bag"), None);
    }

    #[test]
    fn ignores_multi_word_changes() {
        // More than one word changed → too ambiguous, skip.
        assert_eq!(detect_correction("meet at noon monvi", "see you later Manvi"), None);
    }

    #[test]
    fn ignores_unrelated_replacement() {
        // Not phonetically close → not a mishear.
        assert_eq!(detect_correction("ping the server foo", "ping the server Xylophone"), None);
    }

    #[test]
    fn no_change_learns_nothing() {
        assert_eq!(detect_correction("hello there world", "hello there world"), None);
    }

    #[test]
    fn learns_lowercase_technical_correction() {
        // Previously discarded: the accept path required a Titlecase replacement, so
        // every lowercase technical fix was silently thrown away.
        assert_eq!(
            detect_correction("i use wisper for this", "i use whisper for this"),
            Some(("wisper".to_string(), "whisper".to_string()))
        );
    }

    #[test]
    fn learns_when_the_same_mishear_appears_twice() {
        // Set difference, not a filtered Vec: two occurrences of one wrong word are
        // still one distinct correction. This used to be rejected as ambiguous.
        assert_eq!(
            detect_correction("monvi and monvi again", "Manvi and Manvi again"),
            Some(("monvi".to_string(), "Manvi".to_string()))
        );
    }

    #[test]
    fn still_ignores_short_lowercase_edits() {
        // "cat" -> "bat" is close and lowercase, but too short to be distinctive.
        assert_eq!(detect_correction("the cat sat down", "the bat sat down"), None);
    }

    #[test]
    fn still_ignores_unrelated_lowercase_replacement() {
        // Phonetic-closeness gate still applies on the lowercase path.
        assert_eq!(
            detect_correction("ping the server quickly", "ping the server xylophone"),
            None
        );
    }

    /// The diagnostic must never disagree with the decision it explains. This is the
    /// guard against the two gate sequences drifting apart: `rejection_reason` says
    /// "all gates passed" if and only if `detect_correction` actually returns a pair.
    #[test]
    fn rejection_reason_never_contradicts_detect_correction() {
        let cases = [
            ("send the deck to monvi please", "send the deck to Manvi please"),
            ("i left there bag", "i left their bag"),
            ("meet at noon monvi", "see you later Manvi"),
            ("ping the server foo", "ping the server Xylophone"),
            ("hello there world", "hello there world"),
            ("i use wisper for this", "i use whisper for this"),
            ("monvi and monvi again", "Manvi and Manvi again"),
            ("the cat sat down", "the bat sat down"),
            ("rendered with Katex", "rendered with KaTeX"),
            ("a dictation", ""),
            ("", "some text"),
            ("fix this word", "i had already typed a lot here fix this term"),
        ];
        for (ins, aft) in cases {
            let passed = rejection_reason(ins, aft).contains("all gates passed");
            assert_eq!(
                passed,
                detect_correction(ins, aft).is_some(),
                "disagreement on ({ins:?}, {aft:?}): reason said {:?}",
                rejection_reason(ins, aft)
            );
        }
    }

    #[test]
    fn a_case_only_correction_is_invisible_before_any_gate() {
        // Found by this test failing on its first run, against my own wrong belief
        // that the `eq_ignore_ascii_case` gate was what rejected this pair.
        //
        // `word_diff` compares lowercased words, so "Katex" -> "KaTeX" produces an
        // EMPTY diff — zero removed, zero added — and never reaches a gate at all.
        // The consequence is the one worth recording: `DictionaryStore::apply` treats
        // case-only entries as first-class and names this exact pair as its
        // motivating example, yet auto-learn cannot ever produce one.
        assert_eq!(
            word_diff("rendered with Katex", "rendered with KaTeX"),
            Some((vec![], vec![]))
        );
        let r = rejection_reason("rendered with Katex", "rendered with KaTeX");
        assert!(r.contains("CASE-INSENSITIVE"), "{r}");
        assert_eq!(detect_correction("rendered with Katex", "rendered with KaTeX"), None);
    }

    #[test]
    fn rejection_reason_names_the_whole_field_diff() {
        // Pre-existing text in the target field is the dominant suspected cause.
        let r = rejection_reason("fix this word", "i had already typed a lot here fix this term");
        assert!(r.contains("not a 1-for-1 swap"), "{r}");
        assert!(r.contains("counts as added"), "{r}");
    }

    /// Observed in the wild, 2026-08-16: Max dictated "…the Find and Replace system",
    /// corrected "Replace" to "Re-Place", and poll 3 saw a clean `-1 +1`. It was still
    /// not learned — `word_tokens` only trims non-alphanumerics from the *ends* of a
    /// token, so the interior hyphen survives and the all-alphabetic gate rejects it.
    ///
    /// Hyphenated and possessive corrections are ordinary English, so this quietly
    /// discards a whole class of real fixes.
    #[test]
    fn an_interior_hyphen_rejects_an_otherwise_perfect_correction() {
        let ins = "a test for the Find and Replace system";
        let aft = "a test for the Find and Re-Place system";
        assert_eq!(word_diff(ins, aft), Some((vec!["Replace".into()], vec!["Re-Place".into()])));
        assert_eq!(detect_correction(ins, aft), None);
        assert!(rejection_reason(ins, aft).contains("non-alphabetic"));
    }

    /// Observed in the wild, 2026-08-16, and it settles project.md §8 item 3: the
    /// watcher reads the field *mid-edit*. Max was still typing when poll 5 fired and
    /// it learned `"Hypothesis" -> "Hypothe"` — a truncation, written to the dictionary
    /// as authoritative.
    ///
    /// The gate that should have caught this is the one that lets it through:
    /// a truncation is a *prefix*, so its edit distance is small by construction. The
    /// phonetic-closeness test does not merely fail to reject mid-edit captures, it
    /// actively selects for them.
    #[test]
    fn a_mid_edit_truncation_passes_every_gate() {
        assert_eq!(
            detect_correction("look at the Riemann Hypothesis", "look at the Riemann Hypothe"),
            Some(("Hypothesis".to_string(), "Hypothe".to_string()))
        );
        // …and it is "close" precisely because it is a prefix of the real word.
        assert!(norm_levenshtein("Hypothesis", "Hypothe") < 0.6);
    }

    #[test]
    fn rejection_reason_handles_an_emptied_field() {
        // The "user hit send" shape.
        let r = rejection_reason("some dictated words", "");
        assert!(r.contains("no words at all"), "{r}");
    }

    #[test]
    fn still_ignores_common_word_edits_lowercase() {
        // The COMMON stoplist is what keeps ordinary typo fixes out.
        assert_eq!(detect_correction("i left there bag", "i left their bag"), None);
        assert_eq!(detect_correction("go form here now", "go from here now"), None);
    }
}
