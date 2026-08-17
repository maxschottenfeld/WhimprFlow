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
    use std::sync::{Arc, Condvar, Mutex, OnceLock};
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

    /// Why an observation window closed. Mirrors Wispr Flow's
    /// `contentObservationEndReason`, and the distinction is the whole point of this
    /// module's 2026-08-17 rewrite.
    ///
    /// Measured over Max's own 1,175 Wispr dictations, restricted to rows carrying a
    /// real edit (n ≈ 692):
    ///
    /// | terminator | rows |
    /// |---|---|
    /// | user pressed Return | 258 |
    /// | next dictation started | 147 |
    /// | textbox emptied | 22 |
    /// | **observation window elapsed (timeout)** | **18** |
    ///
    /// WhimprFlow used to implement *only* the last row — a fixed 20 s ladder with
    /// no early exit — i.e. the ~2.6% path. Both of Max's own successful Wispr
    /// captures came through `next_dictation_started`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum EndReason {
        /// The user started dictating again. They are done with the old field.
        NextDictationStarted,
        /// The field went empty under us — overwhelmingly "the user hit send".
        TextboxEmptied,
        /// The full poll ladder ran out.
        WindowElapsed,
        /// The element stopped answering (focus moved, or the app re-rendered).
        ElementWentStale,
    }

    impl EndReason {
        /// Does this terminator prove the user *finished editing*?
        ///
        /// This is load-bearing, because it decides how much evidence is needed
        /// before writing to the dictionary. The in-window path requires a candidate
        /// to survive two consecutive polls — that rule exists because a read taken
        /// mid-keystroke looks exactly like a finished correction, and on 2026-08-16
        /// one wrote `"Hypothesis" -> "Hypothe"` from a single observation.
        ///
        /// A semantic terminator supplies that same assurance by a different route:
        /// if the user has moved on to another dictation, or sent the message, the
        /// text they left behind **is** the finished state. Waiting for it to "hold"
        /// is meaningless once nothing can change it again. A timeout proves nothing
        /// of the kind — the user may be mid-word — so it keeps the stricter rule.
        fn is_settled(self) -> bool {
            matches!(self, Self::NextDictationStarted | Self::TextboxEmptied)
        }

        fn label(self) -> &'static str {
            match self {
                Self::NextDictationStarted => "next_dictation_started",
                Self::TextboxEmptied => "textbox_emptied",
                Self::WindowElapsed => "observation_window_elapsed",
                Self::ElementWentStale => "element_went_stale",
            }
        }
    }

    /// Handle to a running observation, so a later dictation can close it.
    ///
    /// The poll loop sleeps on the condvar rather than on `thread::sleep`, so ending
    /// a watch wakes it **immediately** instead of up to 3 s later (the widest gap in
    /// the ladder). That matters: the whole value of the `next_dictation_started`
    /// terminator is evaluating the field state *promptly* once it is final.
    pub(super) struct WatchHandle {
        ended: Mutex<Option<EndReason>>,
        wake: Condvar,
    }

    impl WatchHandle {
        fn new() -> Self {
            Self {
                ended: Mutex::new(None),
                wake: Condvar::new(),
            }
        }

        /// Record the terminator and wake the poll loop. First writer wins — a
        /// window that already closed for its own reason keeps that reason.
        fn end(&self, reason: EndReason) {
            let mut g = self.ended.lock().unwrap();
            if g.is_none() {
                *g = Some(reason);
            }
            self.wake.notify_all();
        }

        fn reason(&self) -> Option<EndReason> {
            *self.ended.lock().unwrap()
        }

        /// Sleep for `gap`, waking early if the watch is ended. Returns the
        /// terminator if one fired.
        fn sleep_or_end(&self, gap: Duration) -> Option<EndReason> {
            let g = self.ended.lock().unwrap();
            if let Some(r) = *g {
                return Some(r);
            }
            let (g, _timeout) = self.wake.wait_timeout(g, gap).unwrap();
            *g
        }
    }

    /// The one in-flight observation, if any.
    ///
    /// Deliberately at most one. Before this existed every dictation spawned a
    /// detached thread that ran the full 20 s regardless, so a burst of dictations
    /// left several watchers polling stale handles at once, each able to write to
    /// the dictionary. Now a new dictation closes the old window first, which both
    /// bounds the thread count at one and supplies the highest-value terminator.
    static ACTIVE: OnceLock<Mutex<Option<Arc<WatchHandle>>>> = OnceLock::new();

    fn active() -> &'static Mutex<Option<Arc<WatchHandle>>> {
        ACTIVE.get_or_init(|| Mutex::new(None))
    }

    /// Close the outstanding observation, if there is one, and let its thread do the
    /// final evaluation. Called at the top of every dictation.
    fn end_active_watch(reason: EndReason) {
        // Release the registry lock *before* touching the handle's own lock. The
        // nesting would be harmless today (the poll thread never grabs the registry
        // while holding `ended`), but the two locks are reached from three call
        // sites and an ordering rule nobody can see is a deadlock waiting for a
        // fourth.
        let h = active().lock().unwrap().take();
        if let Some(h) = h {
            h.end(reason);
        }
    }

    /// Drop this watch from the registry once its thread is finished, unless a newer
    /// dictation already replaced it. Without the identity check, a watch that ended
    /// on the timeout would clear whichever watch happened to be running by then.
    fn clear_active_if(handle: &Arc<WatchHandle>) {
        let mut slot = active().lock().unwrap();
        if slot.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, handle)) {
            *slot = None;
        }
    }

    /// Right after paste, snapshot the focused field, then check it once after a
    /// short delay for a one-word correction to learn.
    pub fn watch_correction(inserted: &str) {
        // 🔴 FIRST, before any early return: a new dictation has started, so the
        // previous observation window is over no matter what happens next. This is
        // the highest-value terminator in the measured distribution (147 of ~692
        // real-edit captures in Wispr's own telemetry, and both of Max's successful
        // captures), and it must fire even when *this* dictation is not watchable —
        // an untrusted process or a two-word utterance still means the user moved on.
        end_active_watch(EndReason::NextDictationStarted);

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
        let handle = Arc::new(WatchHandle::new());
        *active().lock().unwrap() = Some(Arc::clone(&handle));
        std::thread::spawn(move || {
            // Force whole-struct capture (2021 disjoint captures would otherwise grab
            // the raw pointer field and lose the `Send` impl on `SendPtr`).
            let holder = holder;

            let mut reads_ok = 0usize;
            let mut consecutive_failures = 0usize;
            let mut learned = false;

            // What the window saw. Added as diagnostics — "N read(s), no clean
            // one-word correction found" was logged 93 times out of 95 sessions
            // between 08-09 and 08-16 without ever saying which gate did the
            // rejecting, so every "auto-learn doesn't work" report had to be answered
            // by guessing.
            //
            // ⚠️ These are NO LONGER diagnostic-only (changed 2026-08-17). `best` and
            // `last_text` are now the inputs to `finalize_observation` when the window
            // closes on a semantic terminator. That is the entire point of the change:
            // the reads were always being taken, and were being thrown away.
            let mut last_text: Option<String> = None;
            let mut empty_reads = 0usize;
            // The read that came CLOSEST to being a correction, i.e. the smallest
            // non-zero `removed + added`. Explaining the *last* read instead was
            // actively misleading on the first run: the field's final state was a
            // placeholder, so the summary blamed a 12-word diff while poll 3 had seen
            // a clean 1-for-1 swap and a gate had thrown it away. The near-miss is the
            // interesting read; the final one only says whether the text was sent.
            let mut best: Option<(usize, String)> = None;
            // The correction seen on the previous poll, awaiting confirmation.
            let mut pending: Option<(String, String)> = None;

            // How the window closed. Defaults to the timeout, and is overwritten the
            // moment anything better happens.
            let mut end_reason = EndReason::WindowElapsed;

            for (i, gap) in POLL_GAPS_MS.iter().enumerate() {
                // Interruptible sleep: a new dictation wakes this immediately rather
                // than leaving the field unevaluated for up to 3 s.
                if let Some(r) = handle.sleep_or_end(Duration::from_millis(*gap)) {
                    end_reason = r;
                    break;
                }
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
                            //
                            // This now TERMINATES the window instead of polling on.
                            // Nothing further can be learned from a field the user
                            // has already sent, and the state we want — what they
                            // left just before sending — is already in `best` /
                            // `last_text`. Continuing to poll only risked the handle
                            // going stale before anything used those reads.
                            empty_reads += 1;
                            end_reason = EndReason::TextboxEmptied;
                            break;
                        } else {
                            last_text = Some(after.clone());
                            let score = nrem + nadd;
                            if score > 0 && best.as_ref().is_none_or(|(b, _)| score < *b) {
                                best = Some((score, after.clone()));
                            }
                        }
                        // A correction must survive TWO CONSECUTIVE polls before it is
                        // written. A real correction is something the user typed and
                        // then left alone; a mid-edit read is gone by the next poll,
                        // because they are still typing. On 2026-08-16 this exact loop
                        // caught Max part-way through a word and wrote
                        // "Hypothesis" -> "Hypothe" to the dictionary from a single
                        // observation — with `apply()` now live on the hot lane, that
                        // would have started silently rewriting the word in every
                        // dictation. One sample is not evidence that an edit is done.
                        match super::detect_correction(&inserted, &after) {
                            Some(pair) if pending.as_ref() == Some(&pair) => {
                                let (mishear, correct) = pair;
                                eprintln!(
                                    "[whimpr] ✨ auto-learned: \"{mishear}\" -> \"{correct}\" \
                                     (held across 2 polls, confirmed at {}/{})",
                                    i + 1,
                                    POLL_GAPS_MS.len()
                                );
                                crate::hotkey::dictionary_learn(correct, vec![mishear]);
                                learned = true;
                                break;
                            }
                            Some(pair) => {
                                eprintln!(
                                    "[whimpr] auto-learn: candidate \"{}\" -> \"{}\" at poll \
                                     {}/{} — waiting for it to hold",
                                    pair.0,
                                    pair.1,
                                    i + 1,
                                    POLL_GAPS_MS.len()
                                );
                                pending = Some(pair);
                            }
                            None => pending = None,
                        }
                    }
                    None => {
                        consecutive_failures += 1;
                        if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES {
                            end_reason = EndReason::ElementWentStale;
                            break;
                        }
                    }
                }
            }

            // A new dictation may have ended this watch while we were mid-poll rather
            // than asleep, in which case the loop ran to the end of the ladder and
            // still thinks it timed out. Pick that up — but only when nothing more
            // specific already happened.
            //
            // The guard matters: without it, a window that ended because the element
            // went stale (NOT a settled terminator) would be retroactively upgraded to
            // `NextDictationStarted` (settled) by the next dictation, and the reads it
            // is holding would be judged from a single observation. Whichever
            // terminator actually fired first is the true one, and the local reasons
            // are all set at the moment they occur.
            if end_reason == EndReason::WindowElapsed {
                if let Some(r) = handle.reason() {
                    end_reason = r;
                }
            }

            // ── Evaluate on termination ──────────────────────────────────────────
            //
            // Until 2026-08-17 `best` fed exactly one thing: the `WHY:` diagnostic
            // printer. A correction that was seen, logged, and then wiped by the user
            // pressing Return was thrown away — and "correct the word, then send" is
            // the single most common real-world shape (258 of ~692 in Wispr's
            // telemetry ended on a trailing newline, another 22 on the textbox
            // emptying). The detector never got to look at it.
            //
            // ⚠️ Two properties of this block are deliberate and worth keeping:
            //
            //  1. It evaluates reads **already taken**; it never issues a fresh AX
            //     read at termination. On `NextDictationStarted` the user has moved
            //     on and the captured element may now resolve to a *different* field
            //     — a stale `AXUIElement` does not error, it returns plausible
            //     garbage. Judging only what we saw while the window was genuinely
            //     open sidesteps that entirely.
            //  2. It runs **only** for a settled terminator. On a timeout the user
            //     may be mid-word, so the two-consecutive-poll rule still governs;
            //     relaxing that is how `"Hypothesis" -> "Hypothe"` got written.
            if !learned {
                if let Some((mishear, correct)) = super::finalize_observation(
                    &inserted,
                    best.as_ref().map(|(_, t)| t.as_str()),
                    last_text.as_deref(),
                    end_reason.is_settled(),
                ) {
                    eprintln!(
                        "[whimpr] ✨ auto-learned: \"{mishear}\" -> \"{correct}\" \
                         (settled on {})",
                        end_reason.label()
                    );
                    crate::hotkey::dictionary_learn(correct, vec![mishear]);
                    learned = true;
                }
            }

            eprintln!(
                "[whimpr] auto-learn: observation ended — {} ({reads_ok} read(s), \
                 learned={learned})",
                end_reason.label()
            );
            clear_active_if(&handle);
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
    if is_truncation_of(correct, mishear) {
        return format!(
            "\"{mishear}\" -> \"{correct}\": rejected, \"{correct}\" is a TRUNCATION of \
             \"{mishear}\" — almost certainly read while still being typed"
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
    if is_truncation_of(&correct, &mishear) {
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

/// Decide what, if anything, to learn when an observation window closes.
///
/// Split out of the poll loop so the *policy* can be tested without an
/// Accessibility handle, a focused text field, or a 20-second wait. The loop it
/// came from is untestable by construction, which is how it went ten days without
/// anyone noticing it was throwing away the reads it had already taken.
///
/// - `best` is the closest read seen (smallest non-zero word diff) — where a clean
///   1-for-1 swap lands.
/// - `last_text` is the last non-empty read — the state the user actually left.
/// - `settled` says whether the terminator proves the user finished editing.
///
/// Both candidates are tried, in that order, because they disagree in a real case:
/// a stray extra word elsewhere in the field can make some other read "closer" while
/// the actual correction sits in the final one.
///
/// 🔴 **`settled` is not decoration.** A single observation is exactly what wrote
/// `"Hypothesis" -> "Hypothe"` to the dictionary on 2026-08-16, caught mid-keystroke.
/// The in-window path defends against that by requiring a candidate to survive two
/// consecutive polls. This path may skip that rule *only* because a semantic
/// terminator supplies the same assurance another way — once the user has moved to
/// the next dictation or sent the message, the text cannot change again, so waiting
/// for it to "hold" would prove nothing that has not already been proven. On a
/// timeout none of that holds, and this returns `None`.
pub fn finalize_observation(
    inserted: &str,
    best: Option<&str>,
    last_text: Option<&str>,
    settled: bool,
) -> Option<(String, String)> {
    if !settled {
        return None;
    }
    [best, last_text]
        .into_iter()
        .flatten()
        .find_map(|candidate| detect_correction(inserted, candidate))
}

/// Is `candidate` a strict prefix of `original`, short by 2 or more characters?
///
/// This is the mid-edit guard, and it exists because the phonetic gate cannot be it.
/// Measured over every entry auto-learn has ever produced, the junk and the real
/// corrections **overlap**: `Hypothesis -> Hypothe` (a truncation caught while Max was
/// still typing) and `Wimperslow -> Whimprflow` (the one entry this feature ever got
/// right) both score exactly 0.30, so no threshold on edit distance can separate them.
/// A truncation is a *prefix*, and prefixes are among the closest strings to a word by
/// construction — so distance is least informative exactly where it needs to be sharpest.
///
/// The 2-character floor keeps genuine one-character fixes alive: dropping a stray
/// plural ("Whimprflows" -> "Whimprflow") is a real correction, while dropping four
/// characters mid-word is someone still typing.
///
/// ⚠️ This is a heuristic that shrinks the error, not a fix that removes it. Only an
/// explicit "add this" gesture removes the guessing — see §0 G3.
fn is_truncation_of(candidate: &str, original: &str) -> bool {
    let (c, o) = (candidate.to_lowercase(), original.to_lowercase());
    o.len().saturating_sub(c.len()) >= 2 && o.starts_with(&c)
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
    fn a_mid_edit_truncation_is_now_rejected() {
        // It is "close" precisely because it is a prefix of the real word — which is
        // why the distance gate admitted it, and why the fix had to be a different
        // test rather than a different threshold.
        assert!(norm_levenshtein("Hypothesis", "Hypothe") < 0.6);
        assert_eq!(
            detect_correction("look at the Riemann Hypothesis", "look at the Riemann Hypothe"),
            None
        );
        assert!(rejection_reason(
            "look at the Riemann Hypothesis",
            "look at the Riemann Hypothe"
        )
        .contains("TRUNCATION"));
    }

    #[test]
    fn the_truncation_gate_spares_a_one_character_fix() {
        // Dropping a stray plural is a real correction, not someone mid-word. This is
        // the case the 2-character floor exists to protect.
        assert_eq!(
            detect_correction("i use Whimprflows daily", "i use Whimprflow daily"),
            Some(("Whimprflows".to_string(), "Whimprflow".to_string()))
        );
    }

    #[test]
    fn the_truncation_gate_leaves_real_corrections_alone() {
        // Every non-truncation case from the measured corpus must still pass.
        assert!(detect_correction("i use Wimperslow", "i use Whimprflow").is_some());
        assert!(detect_correction("send it to monvi", "send it to Manvi").is_some());
        assert!(detect_correction("i use wisper here", "i use whisper here").is_some());
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

    // ── Observation lifetime (2026-08-17) ────────────────────────────────────
    //
    // These cover the change from "poll for a fixed 20 seconds and judge only what
    // is visible at the moment of judging" to "stop when something semantic happens
    // and judge what was actually seen".

    /// The shape the old loop threw away, and the dominant real-world one: the user
    /// corrects the word, then presses Return. The corrected text was read, scored,
    /// logged — and then only ever handed to the `WHY:` printer.
    #[test]
    fn a_correction_then_send_is_now_learned() {
        let inserted = "i use Wimperslow every day";
        let corrected = "i use Whimprflow every day";
        assert_eq!(
            finalize_observation(inserted, Some(corrected), Some(corrected), true),
            Some(("Wimperslow".to_string(), "Whimprflow".to_string()))
        );
    }

    /// 🔴 The guard that keeps this from re-opening the `Hypothesis -> Hypothe` hole.
    /// On a timeout we have no evidence the user stopped typing, so a single
    /// observation is not enough no matter how clean it looks.
    #[test]
    fn an_unsettled_window_learns_nothing_even_from_a_perfect_read() {
        let inserted = "i use Wimperslow every day";
        let corrected = "i use Whimprflow every day";
        // Identical inputs to the test above; only the terminator differs.
        assert_eq!(
            finalize_observation(inserted, Some(corrected), Some(corrected), false),
            None
        );
    }

    /// The two candidates genuinely disagree, which is why both are tried. Here an
    /// unrelated word typed elsewhere makes the *closest* read useless while the
    /// final read holds the real correction.
    #[test]
    fn it_falls_back_to_the_last_read_when_the_closest_one_is_not_a_swap() {
        let inserted = "ping the server monvi";
        // Closest read: one word added, nothing removed — not a 1-for-1 swap.
        let best = "ping the server monvi now";
        let last = "ping the server Manvi";
        assert_eq!(detect_correction(inserted, best), None);
        assert_eq!(
            finalize_observation(inserted, Some(best), Some(last), true),
            Some(("monvi".to_string(), "Manvi".to_string()))
        );
    }

    /// A window that saw nothing usable stays silent rather than inventing a pair.
    #[test]
    fn nothing_observed_learns_nothing() {
        assert_eq!(finalize_observation("some dictated words", None, None, true), None);
    }

    /// Termination must not weaken any accept gate — it changes *when* the detector
    /// is consulted, never *what* it accepts. A common-word edit is still refused on
    /// the settled path.
    #[test]
    fn a_settled_terminator_does_not_relax_the_gates() {
        assert_eq!(
            finalize_observation("i left there bag", Some("i left their bag"), None, true),
            None
        );
        assert_eq!(
            finalize_observation(
                "look at the Riemann Hypothesis",
                Some("look at the Riemann Hypothe"),
                None,
                true
            ),
            None,
            "a truncation must stay rejected even when the window closed cleanly"
        );
    }
}
