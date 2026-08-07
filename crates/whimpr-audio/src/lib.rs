//! Microphone capture for WhimprFlow.
//!
//! [`start`] opens the default input device and streams audio. While it runs it
//! downmixes to mono, accumulates the whole utterance, and invokes a throttled
//! callback with a small rolling window of RMS levels (0..1) for the pill's live
//! waveform. [`CaptureHandle::stop`] returns the accumulated mono samples plus the
//! device sample rate, ready for resampling to 16 kHz and handing to ASR.
//!
//! cpal's macOS `Stream` is not `Send`, so the stream is created and owned on a
//! dedicated thread; control flows over channels.

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Number of bars in the rolling waveform window (matches the pill's bar count).
const WAVE_BARS: usize = 6;
/// Emit the waveform at ~30 fps.
const EMIT_INTERVAL: Duration = Duration::from_millis(33);
/// Perceptual gain applied to raw RMS so speech fills the meter without clipping.
const LEVEL_GAIN: f32 = 14.0;

/// The captured audio for one utterance.
pub struct CaptureResult {
    /// Mono samples at `sample_rate` (device-native; resample before ASR).
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl CaptureResult {
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }
}

/// A running capture. Drop or [`stop`](Self::stop) to end it.
pub struct CaptureHandle {
    stop_tx: Sender<()>,
    join: Option<JoinHandle<Option<CaptureResult>>>,
}

impl CaptureHandle {
    /// Stop capture and return the accumulated audio (None if the device failed).
    pub fn stop(mut self) -> Option<CaptureResult> {
        let _ = self.stop_tx.send(());
        self.join.take().and_then(|h| h.join().ok().flatten())
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        // If dropped without an explicit stop, still end the capture thread.
        let _ = self.stop_tx.send(());
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

/// Start capturing from the default input device.
///
/// `on_bars` is called ~30x/second with `WAVE_BARS` RMS levels in `[0, 1]`
/// (oldest→newest), from the audio thread. Returns once the stream is playing (so
/// a microphone-permission failure surfaces here, not silently).
pub fn start<F>(on_bars: F) -> anyhow::Result<CaptureHandle>
where
    F: Fn(&[f32]) + Send + 'static,
{
    let (stop_tx, stop_rx) = channel::<()>();
    let (ready_tx, ready_rx) = channel::<anyhow::Result<()>>();

    let join = std::thread::spawn(move || -> Option<CaptureResult> {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("no default input device")));
                return None;
            }
        };
        let supported = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("no default input config: {e}")));
                return None;
            }
        };

        let sample_format = supported.sample_format();
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels().max(1) as usize;
        let config = supported.config();

        // Log which device we actually opened, at capture start.
        //
        // Without this the "hotkeys die after sleep/wake" reports could only ever be
        // inferred from the sample rate: the built-in mic runs 48 kHz, an AirPods
        // HFP link runs 24 kHz, and a capture that lands on a Bluetooth mic
        // immediately after it connects returns digital silence. That inference was
        // right but unprovable. The name makes it a fact, and it costs one line per
        // dictation.
        eprintln!(
            "[whimpr-audio] input device: {:?} @ {} Hz, {} ch, {:?}",
            device.name().unwrap_or_else(|_| "<unknown>".to_string()),
            sample_rate,
            channels,
            sample_format
        );

        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
        let buf_cb = buffer.clone();

        let mut ring: VecDeque<f32> = VecDeque::from(vec![0.0f32; WAVE_BARS]);
        let mut last_emit = Instant::now();
        let err_fn = |e| eprintln!("[whimpr-audio] stream error: {e}");

        // Only the common f32 input format is handled for now; other formats error
        // out clearly rather than capturing silence.
        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let frames = data.len() / channels;
                    let mut sumsq = 0.0f32;
                    {
                        let mut buf = buf_cb.lock().unwrap();
                        buf.reserve(frames);
                        for f in 0..frames {
                            let mut acc = 0.0f32;
                            for c in 0..channels {
                                acc += data[f * channels + c];
                            }
                            let mono = acc / channels as f32;
                            buf.push(mono);
                            sumsq += mono * mono;
                        }
                    }
                    if last_emit.elapsed() >= EMIT_INTERVAL {
                        last_emit = Instant::now();
                        let rms = if frames > 0 {
                            (sumsq / frames as f32).sqrt()
                        } else {
                            0.0
                        };
                        let level = (rms * LEVEL_GAIN).clamp(0.0, 1.0);
                        ring.pop_front();
                        ring.push_back(level);
                        let bars: Vec<f32> = ring.iter().copied().collect();
                        on_bars(&bars);
                    }
                },
                err_fn,
                None,
            ),
            other => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("unsupported sample format {other:?}")));
                return None;
            }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow::anyhow!("failed to build input stream: {e}")));
                return None;
            }
        };
        if let Err(e) = stream.play() {
            let _ = ready_tx.send(Err(anyhow::anyhow!("failed to start stream: {e}")));
            return None;
        }
        let _ = ready_tx.send(Ok(()));

        // Keep the stream alive on this thread until asked to stop.
        let _ = stop_rx.recv();
        drop(stream);

        let samples = std::mem::take(&mut *buffer.lock().unwrap());
        Some(CaptureResult {
            samples,
            sample_rate,
        })
    });

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(CaptureHandle {
            stop_tx,
            join: Some(join),
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(anyhow::anyhow!("capture thread exited before starting")),
    }
}

/// Frame size for the leading-silence scan. 20 ms is short enough to place a
/// speech onset precisely and long enough for an RMS to mean something.
const TRIM_FRAME_MS: usize = 20;

/// How much lead-in to keep in front of the detected onset.
///
/// Not zero, deliberately. Whisper drops the *final* word of a clip that ends on
/// a hard speech edge (see `whimpr-asr`'s `TAIL_PAD_MS`), and a hard edge at the
/// *front* is the same class of hazard. 250 ms is comfortably inside the safe
/// band measured below — 1 s of lead transcribes perfectly — while being far
/// short of the 2 s where damage starts.
const TRIM_KEEP_MS: usize = 250;

/// Don't touch a clip whose leading silence is under this.
///
/// Combined with [`TRIM_KEEP_MS`] the trim engages once the speech onset is at
/// least `TRIM_MIN_MS + TRIM_KEEP_MS` = **1.25 s** in.
///
/// Measured, on a 6 s sentence behind a room-tone lead: 0.4 s and 1 s transcribe
/// perfectly; **1.25 s** is the first damage ("Alpha checkpoint" degrading to
/// "A check point", no words lost); **1.5 s** is the first real loss. So the engage
/// point lands exactly on the first observable damage and 250 ms below the first
/// actual word loss.
///
/// **Do not lower this to buy more margin — it was tried and it backfired.**
/// Dropping it to 700 ms (engage at ~1 s) made `speech-gap.wav` — 1 s of lead, a
/// sentence, a 6 s interior gap, another sentence — lose its entire second
/// sentence, 18 words down to 9, deterministically across three runs. Untrimmed it
/// is correct. Whisper's behaviour around a clip that opens on a near-immediate
/// speech onset *and* contains a long interior gap is genuinely erratic, and a
/// shorter remaining lead walks into it.
///
/// That is the same non-monotonicity the sweep table shows, seen from the other
/// side, and it is the real argument for this constant being conservative: leaving
/// genuinely short leads bit-identical is not just about avoiding needless work,
/// it is about not perturbing clips that already transcribe correctly.
const TRIM_MIN_MS: usize = 1000;

/// Strip a long run of silence from the FRONT of a capture.
///
/// **This fixes a real, reproduced correctness bug.** Whisper does not merely
/// ignore leading silence — past about 2 seconds of it, it starts losing or
/// inventing the speech that follows. Measured on `small.en` with synthetic
/// fixtures (`scripts/make-pause-fixtures.sh`), one 6 s sentence behind a room-tone
/// lead, transcript vs. the 18 words actually spoken:
///
/// | lead | result |
/// |---|---|
/// | 0.4 s  | 18 words, correct |
/// | 1 s    | 18 words, correct |
/// | 1.25 s | "Alpha checkpoint" degraded to "A check point" — damage starts here |
/// | 1.5 s  | **15 words** — "Alpha checkpoint one" lost, "looking" → "Look" |
/// | 2 s    | **15 words** — same loss |
/// | 3 s    | **15 words** — same loss |
/// | 4 s    | 18 words, but "Alpha" degraded to "A" |
/// | 5 s    | 19 words — a bare `"."` prepended, no words lost |
/// | 6 s    | 19 words — a bare `"."` prepended, no words lost |
/// | 8 s    | **10 words** — half the sentence lost, ASR 679 ms vs ~900 ms |
/// | 10 s   | **hallucinated** "just like this." prepended |
///
/// Note the curve is **not monotonic** — 4 s recovers to 18 words, 8 s collapses to
/// 10, 10 s invents text. The trigger is established; what the decoder is doing
/// internally is not, and this code does not claim to know. It removes the trigger
/// rather than the mechanism, which is why it does not need to know.
///
/// Push-to-talk hits this constantly by construction: the key goes down, the user
/// thinks about what to say, *then* speaks. It is the mechanism behind the
/// 2026-08-07 front-truncation reports and behind the spurious leading periods
/// reported independently — the same artefact seen from two directions. Trimming
/// the lead restored all of the above to a full, correct 18 words.
///
/// Deliberately not a VAD. It finds one boundary at the front, never splits on
/// interior silence, and never drops audio after the onset — a mid-utterance
/// thinking-pause is measured-good (5 s mid-clip transcribes perfectly) and must
/// survive untouched. A clip that is silent throughout is returned unchanged, so
/// the existing empty-transcript path still handles it.
pub fn trim_leading_silence(input: &[f32], sample_rate: u32) -> &[f32] {
    if sample_rate == 0 || input.is_empty() {
        return input;
    }
    let frame = sample_rate as usize * TRIM_FRAME_MS / 1000;
    if frame == 0 || input.len() < frame * 2 {
        return input;
    }

    let rms: Vec<f32> = input
        .chunks_exact(frame)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt())
        .collect();
    if rms.is_empty() {
        return input;
    }

    // Noise floor from the 10th percentile rather than the minimum: a single
    // freakishly quiet frame inside real room tone must not drag the floor down
    // and make every later frame look like speech.
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[sorted.len() / 10];
    let peak = sorted[sorted.len() - 1];

    // Three-way gate. The relative terms adapt to a noisy room and to a quiet
    // talker respectively; the absolute term (~ -58 dBFS) keeps digital-zero
    // silence, where both relative terms collapse to zero, from matching the
    // very first frame.
    let threshold = (floor * 4.0).max(peak * 0.08).max(40.0 / 32_768.0);

    let Some(first) = rms.iter().position(|&v| v > threshold) else {
        // Nothing above the floor anywhere: the whole capture is silence. Leave it
        // alone and let the empty-transcript path deal with it.
        return input;
    };

    let keep = sample_rate as usize * TRIM_KEEP_MS / 1000;
    let cut = (first * frame).saturating_sub(keep);
    if cut < sample_rate as usize * TRIM_MIN_MS / 1000 {
        return input;
    }
    &input[cut..]
}

/// Cutoff for the anti-alias filter, comfortably under the 8 kHz Nyquist of the
/// 16 kHz target. Speech carries very little above this, and backing off from 8 kHz
/// buys a usable transition band without needing a huge number of taps.
const ANTIALIAS_CUTOFF_HZ: f32 = 7_000.0;
/// Odd so the kernel is symmetric about an integer center (linear phase — no group
/// delay to compensate for).
const ANTIALIAS_TAPS: usize = 81;

/// Windowed-sinc low-pass FIR (Hamming window), normalized to unity DC gain.
fn lowpass_kernel(cutoff_hz: f32, sample_rate: u32, taps: usize) -> Vec<f32> {
    let fc = (cutoff_hz / sample_rate as f32).min(0.5); // cycles/sample
    let center = (taps / 2) as f32;
    let mut k: Vec<f32> = (0..taps)
        .map(|i| {
            let n = i as f32 - center;
            // sinc(2*fc*n), with the removable singularity at n == 0 handled.
            let sinc = if n == 0.0 {
                2.0 * fc
            } else {
                (2.0 * std::f32::consts::PI * fc * n).sin() / (std::f32::consts::PI * n)
            };
            let w = 0.54
                - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos();
            sinc * w
        })
        .collect();
    let sum: f32 = k.iter().sum();
    if sum != 0.0 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

/// Zero-padded "same"-length convolution, centered so the output stays time-aligned.
fn convolve_same(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let half = kernel.len() / 2;
    (0..input.len())
        .map(|i| {
            let mut acc = 0.0f32;
            for (j, &k) in kernel.iter().enumerate() {
                let idx = i as isize + j as isize - half as isize;
                if idx >= 0 && (idx as usize) < input.len() {
                    acc += input[idx as usize] * k;
                }
            }
            acc
        })
        .collect()
}

/// Resample mono `input` from `src_rate` to 16 kHz (what ASR models expect).
///
/// Downsampling low-passes first. Dropping 48 kHz straight to 16 kHz by
/// interpolation alone lets everything above 8 kHz fold back down into the speech
/// band as aliasing distortion — sibilants and fricatives (s, sh, f, th) carry
/// much of their energy up there, so the audible result is smeared consonants and
/// mis-heard words. Band-limiting first removes the content that would fold;
/// linear interpolation is then adequate, because the signal it interpolates no
/// longer contains the high frequencies that made it inaccurate.
///
/// Upsampling needs no filter (no content above the source Nyquist to fold), and a
/// rate that already matches is returned unchanged.
pub fn resample_to_16k(input: &[f32], src_rate: u32) -> Vec<f32> {
    const DST: u32 = 16_000;
    if src_rate == DST || src_rate == 0 || input.is_empty() {
        return input.to_vec();
    }

    let filtered;
    let src: &[f32] = if src_rate > DST {
        let kernel = lowpass_kernel(ANTIALIAS_CUTOFF_HZ, src_rate, ANTIALIAS_TAPS);
        filtered = convolve_same(input, &kernel);
        &filtered
    } else {
        input
    };

    let ratio = DST as f64 / src_rate as f64;
    let out_len = ((src.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = src.get(idx).copied().unwrap_or(0.0);
        let b = src.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-room-tone. `rand` is not a dependency of this crate and
    /// is not worth adding for a test fixture.
    fn noise(rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8_388_608.0 - 1.0) * amp
            })
            .collect()
    }

    /// A stand-in for speech: loud enough to clear the gate, unlike room tone.
    fn loud(rate: u32, secs: f32) -> Vec<f32> {
        tone(440.0, rate, secs).iter().map(|s| s * 0.5).collect()
    }

    #[test]
    fn trims_a_long_room_tone_lead() {
        let rate = 48_000;
        let mut clip = noise(rate, 5.0, 0.005);
        clip.extend(loud(rate, 3.0));
        let out = trim_leading_silence(&clip, rate);
        // 5 s of lead, 250 ms kept => ~4.75 s removed.
        let removed = (clip.len() - out.len()) as f32 / rate as f32;
        assert!(
            (4.6..4.9).contains(&removed),
            "expected ~4.75s removed, got {removed}"
        );
    }

    #[test]
    fn trims_digital_zero_lead_too() {
        // Both relative terms of the gate collapse to zero here; the absolute
        // floor is what has to carry it.
        let rate = 48_000;
        let mut clip = vec![0.0f32; rate as usize * 5];
        clip.extend(loud(rate, 3.0));
        let out = trim_leading_silence(&clip, rate);
        assert!(out.len() < clip.len(), "digital-zero lead was not trimmed");
    }

    #[test]
    fn leaves_a_short_lead_completely_alone() {
        // The measured-safe band. The common case must be bit-identical, so that
        // this fix cannot regress a normal dictation.
        let rate = 48_000;
        for lead in [0.0f32, 0.25, 0.4, 0.6] {
            let mut clip = noise(rate, lead, 0.005);
            clip.extend(loud(rate, 3.0));
            let out = trim_leading_silence(&clip, rate);
            assert_eq!(out.len(), clip.len(), "lead of {lead}s should not be trimmed");
        }
    }

    #[test]
    fn engages_by_the_shortest_harmful_lead() {
        // 1.25s is where damage was first measured and 1.5s is where words start
        // being lost outright, so the trim has to be engaging through that band.
        //
        // Starts at 1.3 rather than 1.25 on purpose: the engage point IS a 1.25s
        // onset, so an onset of exactly 1.25s falls inside the 20ms frame
        // quantization and can land either side. The real lead-1_25s fixture does
        // trim (its `say` onset sits a few ms later); a synthetic onset at exactly
        // 1.25s does not. Asserting 1.25 here would be asserting a coin-flip.
        let rate = 48_000;
        for lead in [1.3f32, 1.5, 2.0] {
            let mut clip = noise(rate, lead, 0.005);
            clip.extend(loud(rate, 3.0));
            let out = trim_leading_silence(&clip, rate);
            assert!(
                out.len() < clip.len(),
                "lead of {lead}s must be trimmed -- it is at or above where whisper breaks"
            );
        }
    }

    #[test]
    fn does_not_engage_on_a_one_second_lead() {
        // Pinned deliberately, against a regression that was actually introduced and
        // caught: lowering TRIM_MIN_MS so this case *did* trim cost speech-gap.wav
        // its whole second sentence. A 1s lead transcribes correctly untouched, so
        // touching it is pure downside.
        let rate = 48_000;
        let mut clip = noise(rate, 1.0, 0.005);
        clip.extend(loud(rate, 3.0));
        assert_eq!(trim_leading_silence(&clip, rate).len(), clip.len());
    }

    #[test]
    fn never_touches_a_mid_utterance_pause() {
        // The thinking-pause case, which measured fine and must stay untouched --
        // this is the difference between a leading trim and a VAD.
        let rate = 48_000;
        let mut clip = loud(rate, 2.0);
        clip.extend(noise(rate, 6.0, 0.005));
        clip.extend(loud(rate, 2.0));
        let out = trim_leading_silence(&clip, rate);
        assert_eq!(out.len(), clip.len(), "interior silence must survive");
    }

    #[test]
    fn returns_an_all_silent_clip_unchanged() {
        let rate = 48_000;
        let clip = noise(rate, 6.0, 0.005);
        assert_eq!(trim_leading_silence(&clip, rate).len(), clip.len());
        let zeros = vec![0.0f32; rate as usize * 6];
        assert_eq!(trim_leading_silence(&zeros, rate).len(), zeros.len());
    }

    #[test]
    fn trim_edge_cases_do_not_panic() {
        assert!(trim_leading_silence(&[], 48_000).is_empty());
        assert_eq!(trim_leading_silence(&[0.1, 0.2], 48_000).len(), 2);
        // A zero sample rate would make the frame size zero and divide by it.
        assert_eq!(trim_leading_silence(&[0.1, 0.2], 0).len(), 2);
    }

    #[test]
    fn resample_48k_to_16k_thirds_the_length() {
        let input = vec![0.0f32; 48_000];
        let out = resample_to_16k(&input, 48_000);
        assert!((out.len() as i64 - 16_000).abs() <= 1);
    }

    #[test]
    fn resample_noop_at_16k() {
        let input = vec![0.1f32, 0.2, 0.3];
        assert_eq!(resample_to_16k(&input, 16_000), input);
    }

    fn tone(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// The actual anti-aliasing guarantee. A 12 kHz tone sampled at 48 kHz is above
    /// the 8 kHz Nyquist of the 16 kHz target, so naive decimation folds it down to
    /// |12000 - 16000| = 4 kHz — a loud phantom tone sitting in the middle of the
    /// speech band. With the low-pass in place it should be attenuated to near
    /// silence instead. Trimmed at both ends to ignore filter edge transients.
    #[test]
    fn rejects_above_nyquist_instead_of_aliasing_it() {
        let out = resample_to_16k(&tone(12_000.0, 48_000, 0.25), 48_000);
        let body = &out[out.len() / 8..out.len() * 7 / 8];
        assert!(
            rms(body) < 0.05,
            "12 kHz tone should be filtered out, not aliased into the speech band \
             (got RMS {:.4}; unfiltered decimation yields ~0.7)",
            rms(body)
        );
    }

    /// The filter must not eat ordinary speech-band content while doing that.
    #[test]
    fn preserves_speech_band_content() {
        let out = resample_to_16k(&tone(1_000.0, 48_000, 0.25), 48_000);
        let body = &out[out.len() / 8..out.len() * 7 / 8];
        assert!(
            rms(body) > 0.6,
            "1 kHz tone should pass essentially untouched (got RMS {:.4}, expected ~0.707)",
            rms(body)
        );
    }
}
