//! Does dropping a cpal input stream actually stop it?
//!
//! cpal 0.15.3 on macOS registers a disconnect listener for any device that was
//! not obtained as the default. The listener's closure owns a clone of the
//! stream, and the stream owns the listener, so the stream can never be freed:
//! its audio unit keeps running and the data callback keeps firing after
//! `drop()`. `choose_input_device()` takes exactly that path whenever the
//! default input is Bluetooth (it picks the built-in mic out of
//! `input_devices()`), and each dictation's callback then appends samples to a
//! buffer nobody reads, forever.
//!
//! This opens the default input both ways, drops each stream after one second,
//! and counts callbacks during the two seconds after the drop. A stream that
//! stopped reports 0. `--pause` calls `whimpr_audio::stop_stream` (pause, then
//! drop) instead of a plain drop, which is the fix.
//!
//! ```text
//! cargo run -p whimpr-audio --example stream-leak
//! cargo run -p whimpr-audio --example stream-leak -- --pause
//! ```
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn run(label: &str, device: cpal::Device, pause: bool) -> anyhow::Result<()> {
    let config = device.default_input_config()?.config();
    let calls = Arc::new(AtomicU64::new(0));
    let calls_cb = calls.clone();
    let stream = device.build_input_stream(
        &config,
        move |_: &[f32], _| {
            calls_cb.fetch_add(1, Ordering::Relaxed);
        },
        |e| eprintln!("stream error: {e}"),
        None,
    )?;
    stream.play()?;
    std::thread::sleep(Duration::from_secs(1));
    let before = calls.load(Ordering::Relaxed);
    if pause {
        whimpr_audio::stop_stream(stream);
    } else {
        drop(stream);
    }
    let at_drop = calls.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(2));
    let after = calls.load(Ordering::Relaxed) - at_drop;
    println!(
        "{label:<40} callbacks while open: {before:>4}   after drop (2 s): {after:>4}   {}",
        if after == 0 { "stopped" } else { "STILL RUNNING" }
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let pause = std::env::args().any(|a| a == "--pause");
    let host = cpal::default_host();
    let default = host.default_input_device().ok_or_else(|| anyhow::anyhow!("no default input"))?;
    let name = default.name()?;
    println!("default input: {name:?}   mode: {}", if pause { "pause + drop" } else { "drop" });

    run("default_input_device()", default, pause)?;

    let enumerated = host
        .input_devices()?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("default input not found in input_devices()"))?;
    run("same device via input_devices()", enumerated, pause)?;
    Ok(())
}
