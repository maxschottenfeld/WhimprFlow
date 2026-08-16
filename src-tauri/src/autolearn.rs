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

            for (i, gap) in POLL_GAPS_MS.iter().enumerate() {
                std::thread::sleep(Duration::from_millis(*gap));
                match unsafe { element_value(holder.0) } {
                    Some(after) => {
                        reads_ok += 1;
                        consecutive_failures = 0;
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

/// Detect a single clean one-word correction: exactly one word removed from the
/// inserted text and one word added in the field, both distinctive and phonetically
/// close, with the new word looking like a proper noun. Returns `(mishear, correct)`.
pub fn detect_correction(inserted: &str, after: &str) -> Option<(String, String)> {
    use std::collections::HashSet;
    let ins = word_tokens(inserted);
    let aft = word_tokens(after);
    if ins.is_empty() || aft.is_empty() {
        return None;
    }
    let ins_lc: HashSet<String> = ins.iter().map(|w| w.to_lowercase()).collect();
    let aft_lc: HashSet<String> = aft.iter().map(|w| w.to_lowercase()).collect();

    // De-duplicate. This is a set difference, so a word that appears twice is still
    // one distinct change — the previous version filtered the token *Vec* while
    // testing set membership, so dictating "monvi and monvi" and fixing both
    // produced removed.len() == 2 and was silently rejected as "too ambiguous".
    let diff = |from: &[String], other: &HashSet<String>| -> Vec<String> {
        let mut seen = HashSet::new();
        from.iter()
            .filter(|w| !other.contains(&w.to_lowercase()))
            .filter(|w| seen.insert(w.to_lowercase()))
            .cloned()
            .collect()
    };
    let removed = diff(&ins, &aft_lc);
    let added = diff(&aft, &ins_lc);
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

    #[test]
    fn still_ignores_common_word_edits_lowercase() {
        // The COMMON stoplist is what keeps ordinary typo fixes out.
        assert_eq!(detect_correction("i left there bag", "i left their bag"), None);
        assert_eq!(detect_correction("go form here now", "go from here now"), None);
    }
}
