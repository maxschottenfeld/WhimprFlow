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
