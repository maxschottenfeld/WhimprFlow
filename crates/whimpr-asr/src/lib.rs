//! Local speech-to-text via whisper.cpp (whisper-rs), implementing
//! [`whimpr_core::AsrEngine`]. Expects 16 kHz mono f32 samples.

use std::path::Path;

use whimpr_core::asr::{AsrCaps, AsrEngine, AsrEngineId, Transcript};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Whisper's ceiling for the decoder's text context is `n_text_ctx`, and it will
/// only ever take half of that as prompt (`n_text_ctx / 2`). For every model in the
/// family `n_text_ctx` is 448, so the usable prompt budget is 224 tokens — the
/// number OpenAI's own prompting guide quotes. Anything past that is silently
/// dropped by whisper.cpp, which is precisely the kind of quiet truncation this
/// project keeps getting bitten by, so we truncate deliberately and say so.
pub const MAX_PROMPT_TOKENS: usize = 224;

/// Decoding options for a single transcription.
///
/// `Default` reproduces exactly what the app shipped before this struct existed, so
/// `transcribe()` is unchanged behaviour.
#[derive(Debug, Clone)]
pub struct RunOpts {
    /// Whisper `initial_prompt` — biases decoding toward particular spellings.
    /// Sanitized and trimmed to [`MAX_PROMPT_TOKENS`] before use.
    pub prompt: Option<String>,
    /// Force whisper to emit a single segment.
    ///
    /// Upstream added this to stop *short* clips being transcribed twice, and it
    /// works for that. The open question was whether it also truncates clips longer
    /// than whisper's 30-second window, since `single_segment` makes the decoder
    /// stop after one window's worth of output. Exposed here so the harness can
    /// answer that against real audio instead of by reading whisper.cpp.
    pub single_segment: bool,
    /// Override whisper's encoder context length (`n_audio_ctx`, default 1500 =
    /// 30 seconds at 50 frames/sec).
    ///
    /// Whisper pads every clip to a full 30-second window and runs the encoder over
    /// all of it, which is why ASR time tracks model size rather than clip length.
    /// Shrinking this makes the encoder skip the padding — the single largest
    /// latency lever available that is not a model change. Accuracy degrades if it
    /// is set below what the real audio needs, so it must be measured, not assumed.
    pub audio_ctx: Option<i32>,
}

impl Default for RunOpts {
    fn default() -> Self {
        Self {
            prompt: None,
            single_segment: true,
            audio_ctx: None,
        }
    }
}

/// A loaded whisper model ready to transcribe utterances.
pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    /// Load a GGML/GGUF whisper model from `model_path`.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))?;
        Ok(Self { ctx })
    }

    /// Count how many whisper tokens `text` occupies, for prompt-budget decisions.
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        // `tokenize` needs an upper bound; MAX_PROMPT_TOKENS * 4 is far more than any
        // prompt we would consider sending and keeps the failure mode "returns Err"
        // rather than "silently truncates".
        self.ctx
            .tokenize(text, MAX_PROMPT_TOKENS * 4)
            .map(|t| t.len())
            .map_err(|e| anyhow::anyhow!("whisper tokenize: {e}"))
    }

    /// Trim `text` from the FRONT until it fits inside [`MAX_PROMPT_TOKENS`].
    ///
    /// Front, not back, because whisper weights later prompt tokens more heavily —
    /// so the tail is the part worth keeping. Callers should therefore order an
    /// initial prompt **least-important first**; whatever survives truncation is
    /// also what the decoder leans on hardest.
    ///
    /// Returns `None` when even a single word will not fit, which should not happen
    /// for real vocabulary but is not worth panicking over.
    pub fn fit_prompt(&self, text: &str) -> Option<String> {
        let text = sanitize_prompt(text);
        if text.is_empty() {
            return None;
        }
        if self.count_tokens(&text).ok()? <= MAX_PROMPT_TOKENS {
            return Some(text);
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        // Drop leading words until it fits. Prompts are short enough (a few hundred
        // words at most) that walking them is cheaper than being clever.
        for start in 1..words.len() {
            let candidate = words[start..].join(" ");
            if self.count_tokens(&candidate).ok()? <= MAX_PROMPT_TOKENS {
                return Some(candidate);
            }
        }
        None
    }

    /// Transcribe with an optional whisper `initial_prompt`.
    ///
    /// The prompt biases decoding toward particular spellings — it is the supported
    /// way to make whisper get names and jargon right at the acoustic level, rather
    /// than repairing them afterwards. It is a *bias*, not a constraint: whisper is
    /// free to ignore it, and it never guarantees a spelling.
    ///
    /// The prompt is sanitized and budget-trimmed here rather than at the call site,
    /// so no caller can accidentally overflow the 224-token window or hand
    /// `set_initial_prompt` a null byte (which panics).
    pub fn transcribe_with_prompt(
        &self,
        pcm16k: &[f32],
        prompt: Option<&str>,
    ) -> anyhow::Result<Transcript> {
        self.transcribe_with_opts(
            pcm16k,
            &RunOpts {
                prompt: prompt.map(str::to_string),
                ..Default::default()
            },
        )
    }

    /// Full-control entry point. The app uses [`Self::transcribe_with_prompt`]; this
    /// exists so the offline harness can A/B decoding options (notably
    /// `single_segment`) against real audio without a second code path.
    pub fn transcribe_with_opts(
        &self,
        pcm16k: &[f32],
        opts: &RunOpts,
    ) -> anyhow::Result<Transcript> {
        let fitted = opts.prompt.as_deref().and_then(|p| self.fit_prompt(p));
        self.run(pcm16k, fitted.as_deref(), opts.single_segment, opts.audio_ctx)
    }

    fn run(
        &self,
        pcm16k: &[f32],
        prompt: Option<&str>,
        single_segment: bool,
        audio_ctx: Option<i32>,
    ) -> anyhow::Result<Transcript> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("whisper create_state: {e}"))?;

        // Beam search rather than greedy. whisper.cpp defaults to greedy (best_of 1),
        // but OpenAI's reference implementation uses beam search — it evaluates several
        // candidate token paths and picks the best overall sequence instead of
        // committing to the highest-probability token at each step. That is exactly the
        // class of error that produces plausible-but-wrong words. beam_size 5 matches
        // the reference default; the extra cost is negligible on push-to-talk clips.
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 0.0,
        });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        // Do not carry decoder context between calls. Each dictation is independent,
        // and carrying context lets one bad transcription poison the next. This does
        // NOT disable the initial prompt below: whisper.cpp clears the carried-over
        // context first and prepends the initial prompt afterwards, so the two are
        // independent. (Verified empirically, not just read — see OVERNIGHT-LOG.)
        params.set_no_context(true);
        // Push-to-talk utterances are normally one short clip, not long-form audio.
        // Without this, whisper.cpp can split a short clip into multiple internal
        // segments that repeat the same words — which then get concatenated below,
        // producing the sentence twice. See RunOpts::single_segment for the caveat
        // that matters on clips longer than whisper's 30-second window.
        params.set_single_segment(single_segment);
        if let Some(ctx) = audio_ctx {
            params.set_audio_ctx(ctx);
        }
        if let Some(p) = prompt {
            params.set_initial_prompt(p);
        }

        let padded = pad_tail(pcm16k);

        state
            .full(params, &padded)
            .map_err(|e| anyhow::anyhow!("whisper full: {e}"))?;

        let n = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("whisper n_segments: {e}"))?;
        let mut text = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                text.push_str(&seg);
            }
        }

        // Two independent cleanups of whisper's own output, in order: its
        // bracketed non-speech annotations, then the punctuation it emits when it
        // is decoding a thinking-pause rather than speech. Both are text-domain
        // and both can return empty, which the paste path already treats as
        // "nothing to paste".
        Ok(Transcript {
            text: strip_pause_punctuation(&strip_nonspeech_markers(&text)),
            confidence: None,
        })
    }
}

/// How much silence to append before handing audio to whisper.
///
/// Whisper drops the final word of a clip whose speech runs right up to the last
/// sample. Push-to-talk hits that case on *every* dictation by construction — the
/// key comes up right after the last word, so there is never a natural trailing
/// pause. Measured on an 11.9 s fixture ending in "...for on-device transcription":
///
/// | trailing pad | final word |
/// |---|---|
/// | 0 ms   | lost |
/// | 250 ms | lost |
/// | 500 ms | present |
/// | 1000 ms| present |
///
/// 500 ms is the first value that works; the constant sits at 750 ms for margin
/// because the padding is *free*. Whisper pads every clip out to a 30-second
/// window and runs the encoder over all of it, so ASR time is set by model size,
/// not clip length — measured 1317/1227/1317/1261 ms across those four runs, i.e.
/// noise. See project.md's latency section for the underlying measurement.
///
/// One edge case, accepted: a clip landing within 750 ms *below* a 30-second
/// multiple gets pushed into an extra window, which costs one more encoder pass.
/// That band is narrow and those dictations are already the slow ones.
const TAIL_PAD_MS: usize = 750;
const SAMPLE_RATE: usize = 16_000;

fn pad_tail(pcm16k: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pcm16k.len() + TAIL_PAD_MS * SAMPLE_RATE / 1000);
    out.extend_from_slice(pcm16k);
    out.resize(out.len() + TAIL_PAD_MS * SAMPLE_RATE / 1000, 0.0);
    out
}

/// Substrings that mark a bracketed span as one of whisper's non-speech
/// annotations rather than something the user actually said.
const NONSPEECH_HINTS: &[&str] = &[
    "blank_audio", "blankaudio", "blank audio", "silence", "no speech", "nospeech",
    "inaudible", "music", "applause", "laughter", "background noise", "pause",
];

/// Remove whisper's non-speech annotations from a transcript.
///
/// On silence whisper does not stay quiet — it emits a literal `[BLANK_AUDIO]`
/// token as ordinary segment text, and the app pastes it. That is almost certainly
/// the "blank audio gets inserted when I don't say anything" complaint this project
/// was scoped around: not a hallucinated sentence, just this marker.
///
/// Deliberately narrow. Only `[...]` / `(...)` spans whose contents match a known
/// non-speech hint are removed, so a genuinely dictated parenthetical survives. A
/// clip that was *only* a marker returns an empty string, and an empty transcript
/// is already handled upstream as "nothing to paste".
fn strip_nonspeech_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let close = match chars[i] {
            '[' => Some(']'),
            '(' => Some(')'),
            _ => None,
        };
        match close.and_then(|c| chars[i..].iter().position(|&x| x == c).map(|p| i + p)) {
            Some(end) => {
                let inner: String = chars[i + 1..end].iter().collect::<String>().to_lowercase();
                let squashed: String =
                    inner.chars().filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '_').collect();
                if NONSPEECH_HINTS.iter().any(|h| squashed.contains(h)) {
                    i = end + 1;
                    continue;
                }
                out.extend(&chars[i..=end]);
                i = end + 1;
            }
            None => {
                out.push(chars[i]);
                i += 1;
            }
        }
    }
    // Collapse whitespace left behind by a removed marker.
    let out = out.split_whitespace().collect::<Vec<_>>().join(" ");

    // Second rule, and the more robust of the two: if what survives is *entirely*
    // one bracketed span, it is not speech. Whisper describes non-speech audio
    // rather than staying quiet, and the vocabulary is open-ended — loud room tone
    // produced "(water running)" here, and "(wind blowing)", "(typing)",
    // "(door closes)" are all in the same family. Enumerating them is a losing
    // game; noticing that the whole utterance is parenthetical is not.
    //
    // Nobody dictates a message consisting solely of a parenthetical, so the false
    // positive is close to hypothetical, and pasting "(water running)" into
    // whatever Max is writing is a much worse outcome than dropping it.
    if is_wholly_bracketed(&out) {
        return String::new();
    }
    out
}

/// Whether `s` is exactly one `[...]` or `(...)` span and nothing else.
fn is_wholly_bracketed(s: &str) -> bool {
    let s = s.trim();
    let mut chars = s.chars();
    let close = match chars.next() {
        Some('[') => ']',
        Some('(') => ')',
        _ => return false,
    };
    // The closing bracket must be the final character, and must not appear earlier
    // (otherwise "(a) and (b)" would qualify).
    match s.char_indices().skip(1).find(|(_, c)| *c == close) {
        Some((i, c)) => i + c.len_utf8() == s.len(),
        None => false,
    }
}

/// Sentence punctuation that whisper emits *instead of* words when it is
/// decoding silence. Deliberately not "all punctuation" — a line of `---` or a
/// stray `}` from dictated code is content, and must survive.
const PAUSE_PUNCT: &[char] = &['.', ',', '!', '?', ';', ':', '…'];

fn is_pause_punct(c: char) -> bool {
    PAUSE_PUNCT.contains(&c)
}

/// Closing delimiters, which never want a space in front of them either. Whisper
/// puts a trail-off inside the quote — `I could be done..."` — and the
/// replacement space would otherwise strand the quote: `done " That`.
const CLOSING: &[char] = &['"', '\'', ')', ']', '}', '”', '’'];

fn is_closing(c: char) -> bool {
    CLOSING.contains(&c)
}

/// Whether `s` carries nothing but sentence punctuation and whitespace.
fn is_wholly_pause_punct(s: &str) -> bool {
    !s.trim().is_empty() && s.chars().all(|c| c.is_whitespace() || is_pause_punct(c))
}

/// Drop punctuation sitting in front of the first word of a line.
///
/// Nobody dictates a line that opens `". Alpha checkpoint"`. Whisper produces
/// exactly that when a clip opens on silence — reproducible today with
/// `cargo run --release -p whimpr-harness -- --no-trim fixtures/lead-5s.wav`,
/// which returns `". Alpha checkpoint 1, looking at the zeta function…"`. This is
/// the literal shape of G1: a period added because of a pause.
///
/// In production `trim_leading_silence` (§6b) usually removes the lead before
/// whisper sees it — which is why the 206-dictation corpus contains no leading
/// period at all — but that trim deliberately ignores any lead under one second,
/// so the case is defended rather than assumed away.
///
/// **The trailing space is the whole guard.** `". Alpha"` opens with a period
/// that is followed by a space and is stripped; `".md files are fine"` opens with
/// a period that is *not*, and is left exactly alone.
fn strip_leading_orphan(line: &str) -> String {
    let mut s = line;
    loop {
        let mut it = s.chars();
        match (it.next(), it.next()) {
            (Some(c), Some(next)) if is_pause_punct(c) && next.is_whitespace() => {
                s = s[c.len_utf8()..].trim_start();
            }
            _ => return s.to_string(),
        }
    }
}

/// Remove punctuation that whisper inserted because Max *paused*, not because he
/// said anything.
///
/// Whisper renders a thinking-pause as an ellipsis, because that is what an
/// ellipsis means in the text it was trained on. Measured over 206 real logged
/// dictations (2026-08-09 → 08-15, `Reports/2026-08-16-overnight-summary.md`):
/// `...` appears in **15 of 206 (7.3%)**, 18 occurrences, and it is always in
/// whisper's raw output — the cleanup LLM added one **zero** times. Those
/// dictations are the sparse ones: 0.55 s/word median against 0.38 for the rest,
/// worst case `"Plus... work."` at 7.4 seconds of audio for two words.
///
/// This is a **text-domain** strip and cannot cause the audio-domain failures
/// this project has hit before (§6b's leading-silence trim). It removes three
/// dots after transcription; it never touches the samples.
///
/// The rule, in order:
///
/// 1. Every ellipsis form → a single space. Replacing rather than deleting is
///    load-bearing: `"That is...entertainment"` must not become
///    `"That isentertainment"`. The space is skipped when the next character is
///    punctuation the speaker dictated, or a closing delimiter, so
///    `"Plus..., work"` lands on `"Plus, work"` rather than `"Plus , work"`, and
///    `I could be done..."` keeps its quote attached.
/// 2. Collapse runs of spaces, per line. Newlines survive, because this also
///    runs after cleanup, which is what turns a dictated "new line" into one.
/// 3. Drop any line that is nothing but sentence punctuation. A *blank* line is
///    kept — a dictated paragraph break is content. Then drop punctuation
///    stranded in front of a line's first word — see [`strip_leading_orphan`].
/// 4. If nothing but sentence punctuation survives, return empty. Same reasoning
///    as [`is_wholly_bracketed`] — a 23.5-second clip in the corpus transcribed
///    to the single character `","` and pasted it. Nobody dictates a message
///    that is one comma.
/// 5. Trim the ends. `"How..."` becomes `"How"`, **adding nothing** — inventing a
///    period where a trail-off was is inventing content Max did not say. That is
///    his explicit call, 2026-08-16.
///
/// Not stripped, on purpose: a lone period that ends a real sentence. `"Katie."`,
/// `"Run."` and `"Thank you."` are dictations in the corpus and their periods are
/// correct. Only punctuation with *no words attached to it* is removable.
pub fn strip_pause_punctuation(text: &str) -> String {
    // 1. Every ellipsis form → one space. `…` is U+2026; the ASCII run is 3-or-more
    //    so a 4-dot `....` cannot slip past (it has never appeared in the corpus,
    //    but the cost of covering it is one character of pattern). The spaced
    //    `. . .` form is handled by scanning across whitespace between dots.
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '…' {
            out.push(' ');
            i += 1;
            continue;
        }
        if chars[i] == '.' {
            // Count dots, allowing spaces/tabs (not newlines) between them, so
            // both `...` and `. . .` are recognised as one token.
            let mut j = i;
            let mut dots = 0;
            let mut last_dot = i;
            while j < chars.len() {
                if chars[j] == '.' {
                    dots += 1;
                    last_dot = j;
                    j += 1;
                } else if chars[j] == ' ' || chars[j] == '\t' {
                    j += 1;
                } else {
                    break;
                }
            }
            if dots >= 3 {
                // The replacement is a space *unless* the next thing along is
                // punctuation the speaker did dictate — `"Plus..., work"` has to
                // land on `"Plus, work"`, not `"Plus , work"`.
                //
                // Deciding this here, rather than closing up spaces in a later
                // pass, is what keeps the rule off text it did not create: a
                // global "no space before a dot" rule turns the real dictation
                // `"editing CLAUDE.md and .md files"` into `"and.md files"`.
                let next = chars[last_dot + 1..]
                    .iter()
                    .find(|c| !c.is_whitespace())
                    .copied();
                if !next.is_some_and(|c| is_pause_punct(c) || is_closing(c)) {
                    out.push(' ');
                }
                i = last_dot + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }

    // 2 + 3. Per line: collapse runs of spaces, then drop the line entirely if no
    //    words are left on it. A *blank* line is kept — dictated paragraph breaks
    //    are content, and only a line carrying punctuation and nothing else is
    //    evidence of a pause.
    let mut lines: Vec<String> = Vec::new();
    for line in out.split('\n') {
        let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if is_wholly_pause_punct(&collapsed) {
            continue;
        }
        lines.push(strip_leading_orphan(&collapsed));
    }
    let joined = lines.join("\n");

    // 5. Nothing but punctuation survived — there was no speech here.
    if is_wholly_pause_punct(&joined) {
        return String::new();
    }
    // 6. Trim. Whitespace-only can only reach here from an all-whitespace input,
    //    and the paste path treats empty as "nothing to paste".
    joined.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_blank_audio_marker() {
        assert_eq!(strip_nonspeech_markers("[BLANK_AUDIO]"), "");
        assert_eq!(strip_nonspeech_markers("  [BLANK_AUDIO]  "), "");
    }

    #[test]
    fn strips_marker_but_keeps_speech() {
        assert_eq!(
            strip_nonspeech_markers("Hello there. [BLANK_AUDIO] How are you?"),
            "Hello there. How are you?"
        );
    }

    #[test]
    fn strips_common_nonspeech_annotations() {
        for m in ["[ Silence ]", "(upbeat music)", "[INAUDIBLE]", "[Applause]", "[ pause ]"] {
            assert_eq!(strip_nonspeech_markers(m), "", "failed to strip {m}");
        }
    }

    #[test]
    fn keeps_genuine_parentheticals() {
        let s = "Send it to Max (the one from lacrosse) before noon.";
        assert_eq!(strip_nonspeech_markers(s), s);
        let s2 = "Use the array [0] index.";
        assert_eq!(strip_nonspeech_markers(s2), s2);
    }

    #[test]
    fn drops_a_wholly_bracketed_transcript() {
        // Whisper describes non-speech audio instead of staying quiet, with an
        // open-ended vocabulary. Loud room tone produced exactly this one.
        assert_eq!(strip_nonspeech_markers("(water running)"), "");
        assert_eq!(strip_nonspeech_markers("  (wind blowing)  "), "");
        assert_eq!(strip_nonspeech_markers("[door closes]"), "");
    }

    #[test]
    fn wholly_bracketed_rule_does_not_eat_real_sentences() {
        // A parenthetical that is only *part* of the utterance must survive, and so
        // must a sentence containing more than one bracketed span.
        let s = "(the one from lacrosse) will be there";
        assert_eq!(strip_nonspeech_markers(s), s);
        let s2 = "(a) and (b)";
        assert_eq!(strip_nonspeech_markers(s2), s2);
    }

    #[test]
    fn is_wholly_bracketed_edges() {
        assert!(is_wholly_bracketed("(x)"));
        assert!(!is_wholly_bracketed(""));
        assert!(!is_wholly_bracketed("("));
        assert!(!is_wholly_bracketed("(x) y"));
        assert!(!is_wholly_bracketed("y (x)"));
    }

    #[test]
    fn unmatched_bracket_is_left_alone() {
        assert_eq!(strip_nonspeech_markers("a [ b c"), "a [ b c");
    }

    #[test]
    fn pad_tail_appends_silence_and_preserves_input() {
        let input = vec![0.5f32; 1600];
        let out = pad_tail(&input);
        assert_eq!(out.len(), 1600 + TAIL_PAD_MS * SAMPLE_RATE / 1000);
        assert_eq!(&out[..1600], &input[..]);
        assert!(out[1600..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn sanitize_prompt_removes_nulls_and_collapses_space() {
        assert_eq!(sanitize_prompt("a\0b"), "ab");
        assert_eq!(sanitize_prompt("  a \n\t b  "), "a b");
    }

    // ---------------------------------------------------------------- G1: the
    // pause-punctuation strip. Every row of the table Max signed off on
    // 2026-08-16, then the traps, then the shapes that must NOT change.

    /// The five real shapes, verbatim from the agreed rule table.
    #[test]
    fn pause_strip_agreed_rule_table() {
        assert_eq!(
            strip_pause_punctuation("That is...entertainment"),
            "That is entertainment"
        );
        assert_eq!(strip_pause_punctuation("Plus... work."), "Plus work.");
        assert_eq!(strip_pause_punctuation("How..."), "How");
        assert_eq!(
            strip_pause_punctuation("...where did you put"),
            "where did you put"
        );
        assert_eq!(
            strip_pause_punctuation("put...where did you put"),
            "put where did you put"
        );
    }

    /// Trailing adds nothing. `"How..."` is `"How"`, never `"How."` — inventing a
    /// period is inventing content. Max's explicit call, 2026-08-16.
    #[test]
    fn pause_strip_trailing_adds_no_period() {
        assert_eq!(strip_pause_punctuation("How..."), "How");
        assert_eq!(strip_pause_punctuation("Can you..."), "Can you");
        assert_eq!(strip_pause_punctuation("Onward earbuds is..."), "Onward earbuds is");
        assert!(!strip_pause_punctuation("How...").ends_with('.'));
    }

    /// Trap 1: naive deletion joins words. This is the one that silently corrupts.
    #[test]
    fn pause_strip_never_joins_words() {
        for s in [
            "That is...entertainment",
            "put...where did you put",
            "Does indexing...can you define the series",
            "Yeah, where can you put...where did you put that statement?",
        ] {
            assert!(
                !strip_pause_punctuation(s).contains("isentertainment"),
                "joined words in {s:?}"
            );
            assert!(
                !strip_pause_punctuation(s).contains("putwhere"),
                "joined words in {s:?}"
            );
            assert!(
                !strip_pause_punctuation(s).contains("indexingcan"),
                "joined words in {s:?}"
            );
        }
    }

    /// Trap 3: the inserted space must close up against following punctuation.
    #[test]
    fn pause_strip_closes_space_before_punctuation() {
        assert_eq!(strip_pause_punctuation("Plus..., work"), "Plus, work");
        assert_eq!(strip_pause_punctuation("so..., then"), "so, then");
        assert_eq!(strip_pause_punctuation("really...? yes"), "really? yes");
        assert_eq!(strip_pause_punctuation("so...; next"), "so; next");
        // A trail-off inside a quotation — real, whimpr-2026-08-11.log:694.
        assert_eq!(
            strip_pause_punctuation("like, \"I could be done...\" That lesson"),
            "like, \"I could be done\" That lesson"
        );
        assert_eq!(strip_pause_punctuation("(so on...) next"), "(so on) next");
        // Four dots spread across spaces is one long trail-off, not an ellipsis
        // plus a sentence period — there is no way to tell them apart and no
        // instance of either in the corpus.
        assert_eq!(strip_pause_punctuation("wait... . done"), "wait done");
    }

    /// The rule must never close up a space it did not create. This exact string
    /// is a real dictation (whimpr-2026-08-12.log:481) and an early draft turned
    /// it into `"and.md files"`.
    #[test]
    fn pause_strip_leaves_untouched_spaces_alone() {
        let s = "Go ahead and action all of those just for editing CLAUDE.md and .md files.";
        assert_eq!(strip_pause_punctuation(s), s);
        assert_eq!(strip_pause_punctuation("the . thing"), "the . thing");
    }

    /// Trap 2: an ellipsis-only clip strips to empty, not to a stray space.
    #[test]
    fn pause_strip_punctuation_only_becomes_empty() {
        assert_eq!(strip_pause_punctuation("..."), "");
        assert_eq!(strip_pause_punctuation("  ...  "), "");
        assert_eq!(strip_pause_punctuation("…"), "");
        // The real one: a 23.5-second clip in the corpus transcribed to `","`.
        assert_eq!(strip_pause_punctuation(","), "");
        assert_eq!(strip_pause_punctuation("."), "");
        assert_eq!(strip_pause_punctuation(" . , "), "");
        assert_eq!(strip_pause_punctuation(""), "");
    }

    /// All three written forms of the token are the same token.
    #[test]
    fn pause_strip_covers_every_ellipsis_form() {
        assert_eq!(strip_pause_punctuation("a...b"), "a b");
        assert_eq!(strip_pause_punctuation("a…b"), "a b");
        assert_eq!(strip_pause_punctuation("a. . .b"), "a b");
        assert_eq!(strip_pause_punctuation("a....b"), "a b");
        assert_eq!(strip_pause_punctuation("a.....b"), "a b");
    }

    /// The cleanup-created artifact: a dictated "new line" leaves a period alone
    /// on its own line. Whisper said `"Wimperslow, Aeropod, Bug. New line."`
    /// (whimpr-2026-08-12.log:567) and cleanup pasted `"…Bug.\n."`.
    #[test]
    fn pause_strip_drops_punctuation_only_lines() {
        assert_eq!(
            strip_pause_punctuation("Wimprflow, Airpod, Bug.\n."),
            "Wimprflow, Airpod, Bug."
        );
        assert_eq!(strip_pause_punctuation("one\n.\ntwo"), "one\ntwo");
        // A blank line is a dictated paragraph break, not a pause. It stays.
        assert_eq!(strip_pause_punctuation("one\n\ntwo"), "one\n\ntwo");
    }

    /// A lone period that ends a real sentence is CORRECT and must survive. These
    /// are all real dictations from the corpus. Over-stripping here would be a
    /// worse bug than the one being fixed.
    #[test]
    fn pause_strip_keeps_real_sentence_periods() {
        for s in [
            "Katie.",
            "Run.",
            "Thank you.",
            "That makes sense.",
            "To clarify.",
            "S equals 1/2.",
            "Obsidian looks good.",
            "Oh, it's the Fibonacci.",
            "Hello, hello, hello.",
            "Run my morning brief.",
            "Well, my daily briefing and then let's run review/plan.",
        ] {
            assert_eq!(strip_pause_punctuation(s), s, "changed a clean dictation");
        }
    }

    /// Decimals, abbreviations, file extensions and ratios keep their dots.
    #[test]
    fn pause_strip_keeps_ordinary_dots() {
        for s in [
            "It costs 3.50 today.",
            "Edit the .md files, i.e. the notes.",
            "Version 1.2.3 shipped.",
            "See U.S. policy.",
            "Go ahead and action all of those just for editing CLAUDE.md and .md files.",
        ] {
            assert_eq!(strip_pause_punctuation(s), s, "mangled ordinary dots");
        }
    }

    /// Content that merely *looks* like punctuation must survive: a markdown rule
    /// and a dictated closing brace are not pauses.
    #[test]
    fn pause_strip_keeps_non_sentence_punctuation_lines() {
        assert_eq!(strip_pause_punctuation("one\n---\ntwo"), "one\n---\ntwo");
        assert_eq!(strip_pause_punctuation("fn main() {\n}\n"), "fn main() {\n}");
    }

    /// A period stranded in front of the first word is G1's literal shape.
    /// The BEFORE string is real harness output:
    /// `whimpr-harness -- --no-trim fixtures/lead-5s.wav`.
    #[test]
    fn pause_strip_drops_leading_orphan_punctuation() {
        assert_eq!(
            strip_pause_punctuation(
                ". Alpha checkpoint 1, looking at the zeta function, its magnitude \
                 only depends on the real part of S."
            ),
            "Alpha checkpoint 1, looking at the zeta function, its magnitude only \
             depends on the real part of S."
        );
        assert_eq!(strip_pause_punctuation(", hello there"), "hello there");
        assert_eq!(strip_pause_punctuation(". . hello"), "hello");
        assert_eq!(strip_pause_punctuation("one\n. two"), "one\ntwo");
    }

    /// ...but the space after it is the guard, and a file extension has none.
    #[test]
    fn pause_strip_keeps_leading_dot_that_starts_a_word() {
        assert_eq!(strip_pause_punctuation(".md files are fine"), ".md files are fine");
        assert_eq!(strip_pause_punctuation(".gitignore needs a line"), ".gitignore needs a line");
        assert_eq!(strip_pause_punctuation("...md files"), "md files");
    }

    /// Running it twice changes nothing — it also runs after cleanup.
    #[test]
    fn pause_strip_is_idempotent() {
        for s in [
            "That is...entertainment",
            "Plus... work.",
            "How...",
            "Wimprflow, Airpod, Bug.\n.",
            "Katie.",
            ",",
        ] {
            let once = strip_pause_punctuation(s);
            assert_eq!(strip_pause_punctuation(&once), once, "not idempotent on {s:?}");
        }
    }

    /// The whole ASR exit path: markers first, then pause punctuation.
    #[test]
    fn asr_exit_strips_markers_then_pause_punctuation() {
        let run = |s: &str| strip_pause_punctuation(&strip_nonspeech_markers(s));
        assert_eq!(run("[BLANK_AUDIO]"), "");
        assert_eq!(run("Plus... work."), "Plus work.");
        assert_eq!(run("Hello there. [BLANK_AUDIO] How..."), "Hello there. How");
        assert_eq!(run("(water running)"), "");
    }
}

/// Strip characters that would break `set_initial_prompt` or waste prompt budget.
///
/// Null bytes make `set_initial_prompt` panic (it builds a `CString`), so they are
/// removed rather than trusted. Newlines and runs of whitespace are collapsed
/// because they cost tokens and buy nothing.
fn sanitize_prompt(text: &str) -> String {
    text.chars()
        .filter(|c| *c != '\0')
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

impl AsrEngine for WhisperEngine {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::WhisperCpp
    }

    fn caps(&self) -> AsrCaps {
        AsrCaps {
            supports_streaming: false,
        }
    }

    fn transcribe(&self, pcm16k: &[f32]) -> anyhow::Result<Transcript> {
        self.transcribe_with_opts(pcm16k, &RunOpts::default())
    }
}
