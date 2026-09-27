//! Measure how long an input device takes to deliver audio after it is opened.
//!
//! Instrument for the AirPods bug (project notes §6). For each trial it opens a
//! device with the same cpal calls `whimpr_audio::start` uses and prints:
//!
//! - `open`: time from lookup to `stream.play()` returning (what `start()` blocks on)
//! - `first`: time from lookup to the first non-zero sample
//!
//! Three targets, alternated so none always runs warm:
//!
//! - `default`: `host.default_input_device()`, what the app does today
//! - `builtin`: the input device whose CoreAudio transport type is built-in,
//!   found by transport type rather than by name, then opened through cpal by name
//! - `start()`: `whimpr_audio::start` itself, so the device the app would choose
//!   (its choice is logged to stderr)
//!
//! Run with AirPods connected and again with them disconnected. Launch from a
//! terminal that has Microphone access, or every trial reads as silence.
//!
//!     cargo run --release -p whimpr-audio --example mic-open-latency -- [trials] [gap_ms]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use whimpr_audio::coreaudio;

fn transport_str(t: u32) -> String {
    let b = t.to_be_bytes();
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        String::from_utf8_lossy(&b).into_owned()
    } else {
        format!("0x{t:08x}")
    }
}

// ---- one trial ---------------------------------------------------------------

struct Trial {
    name: String,
    rate: u32,
    open_ms: f64,
    first_ms: Option<f64>,
}

fn trial(builtin: Option<&str>, listen: Duration) -> anyhow::Result<Trial> {
    let t0 = Instant::now();
    let ms = move |t: Instant| t.duration_since(t0).as_secs_f64() * 1000.0;
    let host = cpal::default_host();
    let device = match builtin {
        None => host.default_input_device().ok_or_else(|| anyhow::anyhow!("no default input"))?,
        Some(want) => host
            .input_devices()?
            .find(|d| d.name().map(|n| n == want).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("cpal has no input named {want:?}"))?,
    };
    let supported = device.default_input_config()?;
    let rate = supported.sample_rate().0;
    let name = device.name().unwrap_or_default();

    let seen = Arc::new(AtomicBool::new(false));
    let first = Arc::new(Mutex::new(None::<f64>));
    let (seen_cb, first_cb) = (seen.clone(), first.clone());
    let stream = device.build_input_stream(
        &supported.config(),
        move |data: &[f32], _| {
            if !seen_cb.load(Ordering::Relaxed) && data.iter().any(|&s| s != 0.0) {
                seen_cb.store(true, Ordering::Relaxed);
                *first_cb.lock().unwrap() = Some(ms(Instant::now()));
            }
        },
        |e| eprintln!("stream error: {e}"),
        None,
    )?;
    stream.play()?;
    let open_ms = ms(Instant::now());

    let deadline = Instant::now() + listen;
    while !seen.load(Ordering::Relaxed) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(stream);
    let first_ms = *first.lock().unwrap();
    Ok(Trial { name, rate, open_ms, first_ms })
}

/// The production path: `whimpr_audio::start`, whatever device it chooses.
fn trial_start(listen: Duration) -> anyhow::Result<Trial> {
    let t0 = Instant::now();
    let handle = whimpr_audio::start(t0, |_: &[f32]| {})?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    std::thread::sleep(listen);
    let res = handle.stop().ok_or_else(|| anyhow::anyhow!("capture returned nothing"))?;
    Ok(Trial {
        name: "(see stderr)".into(),
        rate: res.sample_rate,
        open_ms,
        first_ms: res.start_timing.first_sample_ms,
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let gap = Duration::from_millis(args.next().and_then(|s| s.parse().ok()).unwrap_or(2000));
    let listen = Duration::from_secs(6);

    println!("input devices (CoreAudio):");
    let default_id = coreaudio::default_input();
    for id in coreaudio::devices().into_iter().filter(|&id| coreaudio::has_input(id)) {
        let t = coreaudio::transport_type(id).map(transport_str).unwrap_or("?".into());
        let mark = if Some(id) == default_id { "  <- default" } else { "" };
        println!("  [{id}] {:?} transport={t}{mark}", coreaudio::name(id).unwrap_or_default());
    }
    let builtin = coreaudio::builtin_input_name();
    println!("built-in by transport type: {builtin:?}");
    println!("{trials} trials per target, {} ms apart, listening up to {} s\n", gap.as_millis(), listen.as_secs());

    let mut rows: Vec<(&str, Trial)> = Vec::new();
    for i in 0..trials {
        for label in ["default", "builtin", "start()"] {
            if label == "builtin" && builtin.is_none() {
                continue;
            }
            std::thread::sleep(gap);
            let r = match label {
                "default" => trial(None, listen),
                "builtin" => trial(builtin.as_deref(), listen),
                _ => trial_start(Duration::from_millis(1500)),
            };
            match r {
                Ok(t) => {
                    let first = t.first_ms.map(|v| format!("{v:8.1} ms")).unwrap_or("  NEVER    ".into());
                    println!(
                        "#{i} {label:<7} {:<28} {:>5} Hz  open {:7.1} ms  first {first}",
                        format!("{:?}", t.name), t.rate, t.open_ms
                    );
                    rows.push((label, t));
                }
                Err(e) => println!("#{i} {label:<7} ERROR {e}"),
            }
        }
    }

    println!("\nmedians:");
    for label in ["default", "builtin", "start()"] {
        let mut open: Vec<f64> = rows.iter().filter(|r| r.0 == label).map(|r| r.1.open_ms).collect();
        let mut first: Vec<f64> = rows.iter().filter(|r| r.0 == label).filter_map(|r| r.1.first_ms).collect();
        let never = rows.iter().filter(|r| r.0 == label && r.1.first_ms.is_none()).count();
        if open.is_empty() {
            continue;
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        println!(
            "  {label:<7} open {:7.1} ms  first {:7.1} ms  never-delivered {never}/{}",
            med(&mut open),
            med(&mut first),
            open.len()
        );
    }
}
