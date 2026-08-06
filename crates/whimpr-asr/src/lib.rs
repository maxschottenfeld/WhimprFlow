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
}

impl Default for RunOpts {
    fn default() -> Self {
        Self {
            prompt: None,
            single_segment: true,
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
        self.run(pcm16k, fitted.as_deref(), opts.single_segment)
    }

    fn run(
        &self,
        pcm16k: &[f32],
        prompt: Option<&str>,
        single_segment: bool,
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

        Ok(Transcript {
            text: strip_nonspeech_markers(&text),
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
