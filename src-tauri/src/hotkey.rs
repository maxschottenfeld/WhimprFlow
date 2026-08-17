//! Hold-Fn → pill wiring for the demo shell.
//!
//! This installs an in-process CoreGraphics event tap that feeds Fn key-down /
//! key-up into the real [`whimpr_core`] dictation state machine, and turns the
//! machine's actions into `whimpr://flowbar/state` events the overlay pill
//! renders. There is no audio or ASR yet, so a finalized session is simulated as
//! completing shortly after key release — enough to see the full
//! recording → transcribing → done → idle loop driven by the actual state machine.
//!
//! In the shipping product this hook lives in a separate sidecar process (so heavy
//! inference can't stall it); running it in-process is an acceptable macOS-only
//! path for this demo and the early milestones.

/// Dictionary entry shape sent to the Hub UI (auto-learned entries flagged).
#[derive(Clone, serde::Serialize)]
pub struct DictEntryDto {
    pub correct: String,
    pub mishears: Vec<String>,
    pub auto: bool,
}

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::c_void;
    use std::path::PathBuf;
    use super::DictEntryDto;
    use std::ptr::{null, null_mut};
    use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use tauri::{AppHandle, Emitter};
    use whimpr_core::state::{Action, BarState};
    use whimpr_core::{
        CleanupContext, CleanupMode, CleanupProvider, Input, PipelineEvent, StateMachine,
        TriggerToken,
    };
    use whimpr_ipc::BindingId;

    const OVERLAY_LABEL: &str = "whimpr_bar";

    // --- CoreGraphics / CoreFoundation FFI (listen-only Fn tap) -----------
    type CFMachPortRef = *mut c_void;
    type CFRunLoopSourceRef = *mut c_void;
    type CFRunLoopRef = *mut c_void;
    type CFStringRef = *const c_void;
    type CFAllocatorRef = *const c_void;
    type CGEventRef = *mut c_void;
    type CGEventTapProxy = *mut c_void;
    type CGEventTapCallBack =
        extern "C" fn(CGEventTapProxy, u32, CGEventRef, *mut c_void) -> CGEventRef;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventTapCreate(
            tap: u32,
            place: u32,
            options: u32,
            events_of_interest: u64,
            callback: CGEventTapCallBack,
            user_info: *mut c_void,
        ) -> CFMachPortRef;
        fn CGEventTapEnable(tap: CFMachPortRef, enable: bool);
        fn CGEventTapIsEnabled(tap: CFMachPortRef) -> bool;
        fn CGEventGetFlags(event: CGEventRef) -> u64;
        fn CGEventGetIntegerValueField(event: CGEventRef, field: u32) -> i64;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFMachPortCreateRunLoopSource(
            allocator: CFAllocatorRef,
            port: CFMachPortRef,
            order: isize,
        ) -> CFRunLoopSourceRef;
        fn CFRunLoopGetCurrent() -> CFRunLoopRef;
        fn CFRunLoopAddSource(rl: CFRunLoopRef, source: CFRunLoopSourceRef, mode: CFStringRef);
        fn CFRunLoopRun();
        static kCFRunLoopDefaultMode: CFStringRef;
    }

    const K_CG_SESSION_EVENT_TAP: u32 = 1;
    const K_CG_HEAD_INSERT: u32 = 0;
    const K_CG_TAP_OPTION_LISTEN_ONLY: u32 = 1;
    const K_CG_EVENT_FLAGS_CHANGED: u32 = 12;
    const EVENTS_OF_INTEREST: u64 = 1 << K_CG_EVENT_FLAGS_CHANGED;
    // WhimprFlow Dev binds Right Option instead of Fn, so it can run side-by-side
    // with the stable /Applications/WhimprFlow.app without fighting over the same
    // physical key. See memory/projects/WhimprFlow/project.md, Phase 0.
    const FLAG_HOTKEY_MODIFIER: u64 = 0x0080_0000; // kCGEventFlagMaskSecondaryFn
    const K_CG_KEYBOARD_EVENT_KEYCODE: u32 = 9;
    const KEYCODE_HOTKEY: i64 = 63; // kVK_Function
    /// Hold Shift while pressing the dictation hotkey to dictate MATHEMATICS —
    /// the transcript is converted to notation instead of being cleaned up (G2).
    ///
    /// A modifier on the existing key rather than a second global hotkey, for
    /// three reasons. It needs no new keycode, so it cannot collide with a key
    /// the user has bound elsewhere. It needs no change to
    /// `promote-to-stable.sh`'s de-brand — that script rewrites `KEYCODE_HOTKEY`
    /// from Right Option to Fn, and a Shift modifier rides along with whichever
    /// key that is, so stable and dev keep their separate bindings for free
    /// (a shared second keycode would have had BOTH apps firing at once, since
    /// they are designed to run side by side). And the gesture composes: the
    /// modifier is read at key-DOWN from the event's own flags, which already
    /// carry the full modifier state, so Shift simply has to be held first.
    const FLAG_MATH_MODIFIER: u64 = 0x0002_0000; // kCGEventFlagMaskShift
    const K_CG_TAP_DISABLED_BY_TIMEOUT: u32 = 0xFFFF_FFFE;
    const K_CG_TAP_DISABLED_BY_USER_INPUT: u32 = 0xFFFF_FFFF;

    static APP: OnceLock<AppHandle> = OnceLock::new();
    static MACHINE: OnceLock<Mutex<StateMachine>> = OnceLock::new();
    static CLOCK: OnceLock<Instant> = OnceLock::new();
    static FN_IS_DOWN: AtomicBool = AtomicBool::new(false);
    static TAP_PORT: AtomicPtr<c_void> = AtomicPtr::new(null_mut());
    /// Bundle id of the app that was frontmost at record-start = the paste target.
    /// Cleanup uses it to format for the medium (email vs. text vs. chat); the
    /// math stage uses it to pick Unicode or LaTeX.
    static TARGET_APP: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    /// Whether the dictation now being recorded was started with Shift held, i.e.
    /// is a MATH dictation. Latched at key-DOWN and read at finalize, because the
    /// user has long since let go of Shift by the time the transcript exists.
    static MATH_MODE: AtomicBool = AtomicBool::new(false);
    static CAPTURE: OnceLock<Mutex<Option<whimpr_audio::CaptureHandle>>> = OnceLock::new();
    static ASR: OnceLock<Arc<whimpr_asr::WhisperEngine>> = OnceLock::new();
    static OPENAI: OnceLock<Mutex<Option<whimpr_cleanup::OpenAiProvider>>> = OnceLock::new();
    static ANTHROPIC: OnceLock<Mutex<Option<whimpr_cleanup::AnthropicProvider>>> = OnceLock::new();
    static LOCAL: OnceLock<Mutex<Option<crate::local_llm::LocalWorker>>> = OnceLock::new();
    /// The math stage's own worker, holding its own (larger) model. Separate from
    /// `LOCAL` because the two stages deliberately run different models — the 4B
    /// is much better at notation and much slower, which is the right trade behind
    /// a deliberate hotkey and the wrong one on ordinary dictation.
    ///
    /// Spawned **lazily**, on the first math key-down, so the ~2.4 GB model is
    /// never resident for someone who does not use the gesture. `MATH_SPAWNING`
    /// guards against a second key-press starting a second process while the first
    /// is still loading.
    static MATH_LOCAL: OnceLock<Mutex<Option<crate::local_llm::LocalWorker>>> = OnceLock::new();
    static MATH_SPAWNING: AtomicBool = AtomicBool::new(false);
    static SETTINGS: OnceLock<Mutex<whimpr_core::Settings>> = OnceLock::new();
    static DICTIONARY: OnceLock<Mutex<whimpr_core::DictionaryStore>> = OnceLock::new();
    static STATS: OnceLock<Mutex<whimpr_core::StatsStore>> = OnceLock::new();

    #[derive(Clone, Serialize)]
    struct BarPayload {
        state: &'static str,
    }

    #[derive(Clone, Serialize)]
    struct WavePayload {
        bars: Vec<f32>,
    }

    #[derive(Clone, Serialize)]
    struct TranscriptPayload {
        text: String,
    }

    /// One machine-parseable line per completed dictation (3b). Emitted via
    /// `eprintln!` like everything else, prefixed `[whimpr-metrics]` so
    /// `scripts/read-logs.py` can pull just these lines out of a day's log and
    /// ignore the human-readable prose around them.
    ///
    /// `capture_start_ms` is `None` when the capture never delivered a sample
    /// (see `whimpr_audio::CaptureStartTiming`) -- distinct from 0, which would
    /// falsely imply an instant-open.
    #[derive(Serialize)]
    struct DictationMetrics {
        ts: u64,
        words: u32,
        clip_pretrim_s: f32,
        clip_posttrim_s: f32,
        trim_engaged: bool,
        trim_cut_s: f32,
        capture_start_ms: Option<f64>,
        resample_ms: u64,
        asr_ms: u64,
        cleanup_fired: bool,
        cleanup_ms: u64,
        /// True when this dictation was started with Shift held, so `cleanup_ms`
        /// is the MATH stage's time rather than cleanup's. Without this flag the
        /// two are indistinguishable in the metrics, and they have very different
        /// latency profiles — which would quietly corrupt any latency baseline
        /// computed from a day's log.
        math_mode: bool,
        /// Deterministic dictionary replacements applied on the hot lane.
        dict_hits: usize,
        /// Microseconds, not milliseconds — the stage is expected to round to 0ms and
        /// a millisecond field would only ever record that it was free, which is not
        /// evidence. This is the number that has to stay small.
        dict_us: u64,
        paste_ms: u64,
        total_ms: u64,
    }

    /// The whisper ASR model to load: prefer the most accurate one present, in
    /// descending quality order, falling back to the small base model. Bigger
    /// English models mis-hear names/technical terms far less (and better ASR means
    /// less for cleanup and the dictionary to fix downstream).
    fn model_path() -> PathBuf {
        let dir = support_dir().join("models");
        // Ordered for LATENCY, not raw accuracy — measured 2026-08-05, see
        // orientation.md. Whisper pads every clip to a 30-second window and runs the
        // encoder over all of it, so ASR time is set by model size and is essentially
        // independent of how briefly you actually spoke: large-v3-turbo costs ~2.7s
        // per dictation on an M2 Air whether the clip is 2s or 9s. small.en's encoder
        // is roughly 7x smaller, which is what makes sub-second dictation possible.
        //
        // The "-dev" suffixes are deliberate. Both apps share this models directory by
        // symlink, and the stable app's compiled list ranks a plain "ggml-small.en.bin"
        // ABOVE base.en — so dropping one in under its canonical name would silently
        // change which model the stable app loads. Dev-only filenames keep that from
        // happening. Rename a file to its canonical name only if you also want the
        // stable app to pick it up.
        for name in [
            // Speed-first default.
            "ggml-small.en-dev.bin",
            // Accuracy-first fallbacks, kept for easy A/B by renaming on disk.
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-medium.en.bin",
            "ggml-small.en.bin",
            "ggml-base.en.bin",
        ] {
            let p = dir.join(name);
            if p.exists() {
                return p;
            }
        }
        dir.join("ggml-base.en.bin")
    }

    // WhimprFlow Dev uses its own app-support dir so it never reads/writes the
    // stable app's dictionary, stats, or settings. Models are symlinked in
    // (Phase 0 step 4) rather than duplicated. See project.md, Phase 0.
    fn support_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join("Library/Application Support/WhimprFlow")
    }
    fn settings_path() -> PathBuf {
        support_dir().join("settings.json")
    }
    fn dict_path() -> PathBuf {
        support_dir().join("dictionary.json")
    }
    fn stats_path() -> PathBuf {
        support_dir().join("stats.json")
    }
    fn logs_dir() -> PathBuf {
        support_dir().join("logs")
    }

    // ── Always-on file logging (3a) ─────────────────────────────────────────
    //
    // Finder/Dock/launchd launch WhimprFlow with fd 0/1/2 all on `/dev/null` --
    // verified 2026-08-07 -- so all 64 `eprintln!` calls in this codebase vanish
    // on every normal launch, including the one that would have settled the §6
    // AirPods bug on 2026-08-06. This section `dup2`s a dated log file onto fd 2
    // at startup so every existing `eprintln!` lands in a readable file, with
    // ZERO call-site changes anywhere else. See project.md §6.
    //
    // Deliberately does nothing when stderr is NOT `/dev/null` (a terminal, an
    // explicit `open --stderr` redirect) -- a dev session run by hand keeps
    // printing to the terminal exactly as before.

    /// True if `fd` and `/dev/null` are the same underlying file (device + inode).
    /// Comparing identity rather than assuming fd 2 -> /dev/null lets this also
    /// correctly no-op under an explicit `open --stderr /tmp/x.log` redirect,
    /// which points fd 2 at a real file, not /dev/null.
    fn fd_is_dev_null(fd: std::os::unix::io::RawFd) -> bool {
        use std::os::unix::io::AsRawFd;
        unsafe {
            let mut target: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut target) != 0 {
                return false;
            }
            let Ok(null_file) = std::fs::File::open("/dev/null") else {
                return false;
            };
            let mut null_stat: libc::stat = std::mem::zeroed();
            if libc::fstat(null_file.as_raw_fd(), &mut null_stat) != 0 {
                return false;
            }
            target.st_dev == null_stat.st_dev && target.st_ino == null_stat.st_ino
        }
    }

    /// Today's date, in the *local* timezone (log files are named for the day
    /// Max experiences, not UTC). `libc::localtime_r` rather than pulling in a
    /// date/time crate -- this codebase stays dependency-light on purpose (see
    /// stats.rs, settings.rs), and one POSIX call is all "what's today's local
    /// date" needs.
    fn local_date() -> (i32, u32, u32) {
        unsafe {
            let now = libc::time(null_mut());
            let mut tm: libc::tm = std::mem::zeroed();
            libc::localtime_r(&now, &mut tm);
            (tm.tm_year + 1900, (tm.tm_mon + 1) as u32, tm.tm_mday as u32)
        }
    }

    fn local_date_string() -> String {
        let (y, m, d) = local_date();
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// Days since the civil epoch (0000-03-01), Howard Hinnant's
    /// `days_from_civil`. Used only to diff two calendar dates for the retention
    /// prune below -- correct for the whole proleptic Gregorian calendar,
    /// including leap years, with no date library.
    fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400; // [0, 399]
        let mp = (m as i64 + 9) % 12; // [0, 11], Mar=0 .. Feb=11
        let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
        era * 146_097 + doe - 719_468
    }

    /// Delete log files whose embedded date is more than `keep_days` before
    /// today. ~42 KB/day at ~50 dictations (6 lines each) means this is about not
    /// leaving an unbounded file on a machine with no maintainer for nine months,
    /// not about disk pressure -- council finding, 2026-08-08.
    fn prune_old_logs(dir: &std::path::Path, keep_days: i64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let (ty, tm, td) = local_date();
        let today = days_from_civil(ty as i64, tm, td);
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(date_part) = name
                .strip_prefix("whimpr-")
                .and_then(|s| s.strip_suffix(".log"))
            else {
                continue;
            };
            let parts: Vec<&str> = date_part.split('-').collect();
            let (Some(y), Some(m), Some(d)) = (
                parts.first().and_then(|s| s.parse::<i64>().ok()),
                parts.get(1).and_then(|s| s.parse::<u32>().ok()),
                parts.get(2).and_then(|s| s.parse::<u32>().ok()),
            ) else {
                continue;
            };
            if today - days_from_civil(y, m, d) > keep_days {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    /// Open today's log file and `dup2` it onto fd 2. Returns the date string the
    /// file was opened for, so the rotation watchdog knows when it's stale.
    fn open_and_redirect(dir: &std::path::Path) -> std::io::Result<String> {
        use std::os::unix::io::AsRawFd;
        std::fs::create_dir_all(dir)?;
        let date = local_date_string();
        let path = dir.join(format!("whimpr-{date}.log"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let rc = unsafe { libc::dup2(file.as_raw_fd(), 2) };
        if rc < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // fd 2 now points at the same open file description as `file`'s fd.
        // Leak `file` itself so its fd isn't closed out from under fd 2 when this
        // function returns -- fd 2 stays valid for the process lifetime either
        // way (dup2 gives it its own reference), but there's no reason to close
        // and immediately lose the handle.
        std::mem::forget(file);
        Ok(date)
    }

    /// Poll for the local date changing and re-point fd 2 at the new day's file
    /// when it does, pruning old files on every rotation. Polls rather than
    /// sleeping-until-midnight to avoid timezone/offset arithmetic entirely --
    /// log rotation has no reason to be precise to the second.
    fn spawn_rotation_watchdog(dir: PathBuf, mut current: String) {
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(300));
            let today = local_date_string();
            if today != current {
                match open_and_redirect(&dir) {
                    Ok(d) => {
                        current = d;
                        prune_old_logs(&dir, 14);
                        eprintln!("[whimpr] log rotated for {current}");
                    }
                    Err(e) => eprintln!("[whimpr] log rotation failed: {e}"),
                }
            }
        });
    }

    /// Entry point: redirect fd 2 to a dated log file, if it isn't already
    /// pointed somewhere real. Called once, as the very first thing `run()`
    /// does, so every subsequent `eprintln!` -- including the ones from
    /// `build_overlay` before `install()` even runs -- is captured.
    ///
    /// Never panics. A logging failure must not take down a dictation tool: on
    /// any error this silently leaves fd 2 exactly as it was.
    pub fn init_logging() {
        if !fd_is_dev_null(2) {
            return;
        }
        let dir = logs_dir();
        prune_old_logs(&dir, 14);
        match open_and_redirect(&dir) {
            Ok(date) => {
                eprintln!("[whimpr] logging to {}", dir.join(format!("whimpr-{date}.log")).display());
                spawn_rotation_watchdog(dir, date);
            }
            Err(_) => {
                // Nowhere to report this -- stderr is still /dev/null, and the
                // log file we'd log the failure to is the thing that failed to
                // open. Carry on silently, per the "never panic" rule.
            }
        }
    }

    /// Seconds since the Unix epoch (UTC), or 0 if the clock is before the epoch.
    fn unix_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Log one completed dictation to the stats store (words, speaking time, text,
    /// target app) and persist it. Powers both the Hub stats and the history list.
    pub fn record_dictation(text: &str, duration_secs: f32) {
        let words = whimpr_core::stats::count_words(text);
        if words == 0 {
            return;
        }
        let app = TARGET_APP.get().and_then(|m| m.lock().unwrap().clone());
        if let Some(m) = STATS.get() {
            let mut store = m.lock().unwrap();
            let duration_ms = (duration_secs.max(0.0) * 1000.0) as u32;
            let chars = text.chars().count() as u32;
            store.record(words, duration_ms, chars, unix_now(), text.to_string(), app);
            let _ = store.save(&stats_path());
        }
    }

    /// The most recent dictations for the Hub Home history list.
    pub fn history(limit: usize) -> Vec<whimpr_core::HistoryItem> {
        STATS
            .get()
            .map(|m| m.lock().unwrap().history(limit))
            .unwrap_or_default()
    }

    /// The dictionary entries for the Hub Dictionary screen (auto-learned flagged).
    pub fn dictionary_entries() -> Vec<DictEntryDto> {
        DICTIONARY
            .get()
            .map(|m| {
                m.lock()
                    .unwrap()
                    .entries
                    .iter()
                    .map(|e| DictEntryDto {
                        correct: e.correct.clone(),
                        mishears: e.mishears.clone(),
                        auto: matches!(e.source, whimpr_core::DictSource::Auto),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add a manual dictionary entry and persist.
    pub fn dictionary_add(correct: String, mishears: Vec<String>) {
        if let Some(m) = DICTIONARY.get() {
            let mut store = m.lock().unwrap();
            store.add(correct, mishears, whimpr_core::DictSource::Manual);
            let _ = store.save(&dict_path());
        }
    }

    /// Approve an auto-learned entry, granting it authority to rewrite text, and
    /// persist. Until this is called the entry only biases whisper's decoding —
    /// see `DictionaryStore::replacement_rules`.
    pub fn dictionary_approve(correct: &str) {
        if let Some(m) = DICTIONARY.get() {
            let mut store = m.lock().unwrap();
            if store.approve(correct) {
                let _ = store.save(&dict_path());
            }
        }
    }

    /// Remove a dictionary entry by spelling and persist.
    pub fn dictionary_remove(correct: &str) {
        if let Some(m) = DICTIONARY.get() {
            let mut store = m.lock().unwrap();
            if store.remove(correct) {
                let _ = store.save(&dict_path());
            }
        }
    }

    /// Apply the dictionary's known mishears to `text`, returning the corrected text
    /// and the number of replacements.
    ///
    /// This is the hot lane's deterministic stage: no model, no network, no
    /// conditional. It runs on **every** dictation, which is the whole point — the
    /// dictionary's other two consumers are both best-effort. `prefilter()` only
    /// reaches the cleanup LLM, and `needs_cleanup()` skipped it on 186 of the 201
    /// dictations logged between 2026-08-08 and 08-15; `asr_prompt()` only biases
    /// whisper and loses to the acoustics often enough that "Whimprflow" was still
    /// being transcribed "Wimperslow" with the entry already in the dictionary.
    ///
    /// Returns the input unchanged when the store is missing or the lock is poisoned:
    /// a dictation that pastes uncorrected text is a far better failure than one that
    /// pastes nothing.
    fn dictionary_apply(text: &str) -> (String, usize) {
        match DICTIONARY.get().and_then(|m| m.lock().ok()) {
            Some(store) => store.apply(text),
            None => (text.to_string(), 0),
        }
    }

    /// Add an AUTO-learned entry (from the post-paste correction observer) and persist.
    /// Marked ✨ auto in the UI. No-op if it would duplicate an existing entry's data.
    pub fn dictionary_learn(correct: String, mishears: Vec<String>) {
        if let Some(m) = DICTIONARY.get() {
            let mut store = m.lock().unwrap();
            store.add(correct, mishears, whimpr_core::DictSource::Auto);
            let _ = store.save(&dict_path());
        }
    }

    /// Aggregated stats for the Hub. `tz_offset_minutes` is the UI's
    /// `Date.getTimezoneOffset()` so day math matches the user's local clock.
    pub fn stats_summary(tz_offset_minutes: i32) -> whimpr_core::StatsSummary {
        STATS
            .get()
            .map(|m| m.lock().unwrap().summary(tz_offset_minutes, unix_now()))
            .unwrap_or_else(|| {
                whimpr_core::StatsStore::default().summary(tz_offset_minutes, unix_now())
            })
    }

    /// Read an API key from an env var or the OS keychain (never a plaintext file).
    fn read_key(account: &str, env_var: &str) -> Option<String> {
        if let Ok(k) = std::env::var(env_var) {
            let k = k.trim().to_string();
            if !k.is_empty() {
                return Some(k);
            }
        }
        keyring::Entry::new("com.whimpr.whimprflow", account)
            .ok()
            .and_then(|e| e.get_password().ok())
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
    }
    fn read_openai_key() -> Option<String> {
        read_key("openai_api_key", "OPENAI_API_KEY")
    }
    fn read_anthropic_key() -> Option<String> {
        read_key("anthropic_api_key", "ANTHROPIC_API_KEY")
    }

    /// A snapshot of the current settings.
    pub fn current_settings() -> whimpr_core::Settings {
        SETTINGS
            .get()
            .map(|m| m.lock().unwrap().clone())
            .unwrap_or_default()
    }
    /// Apply new settings and rebuild the cloud providers (picks up model changes).
    pub fn update_settings(new: whimpr_core::Settings) {
        if let Some(m) = SETTINGS.get() {
            *m.lock().unwrap() = new.clone();
        }
        let _ = new.save(&settings_path());
        rebuild_providers();
    }

    /// (Re)build the cloud cleanup providers from the current keys + settings. Called
    /// at startup and whenever a key or model changes, so edits take effect live.
    pub fn rebuild_providers() {
        let settings = current_settings();
        let openai = read_openai_key().map(|k| {
            whimpr_cleanup::OpenAiProvider::with_base_url(
                k,
                settings.openai_model.clone(),
                Some(settings.openai_base_url.clone()),
            )
        });
        let anthropic = read_anthropic_key()
            .map(|k| whimpr_cleanup::AnthropicProvider::new(k, settings.anthropic_model.clone()));
        eprintln!(
            "[whimpr] cleanup providers: openai={}, anthropic={}",
            openai.is_some(),
            anthropic.is_some()
        );
        match OPENAI.get() {
            Some(m) => *m.lock().unwrap() = openai,
            None => {
                let _ = OPENAI.set(Mutex::new(openai));
            }
        }
        match ANTHROPIC.get() {
            Some(m) => *m.lock().unwrap() = anthropic,
            None => {
                let _ = ANTHROPIC.set(Mutex::new(anthropic));
            }
        }
    }

    /// Clean a raw transcript per the current settings (mode + level), feeding in the
    /// dictionary vocabulary relevant to this utterance. Falls back to raw whenever
    /// cleanup is off, the provider is unavailable, it errors, or the gates reject it.
    ///
    /// Returns `(text, fired)` where `fired` is whether an LLM call was actually
    /// made -- distinct from "the gate wanted cleanup," which can be true while
    /// `fired` is false (e.g. no provider configured). `fired` is the field the
    /// per-dictation metrics line (3b) reports, because it's the one that costs
    /// wall-clock time: the 2026-08-08 council found the cleanup LLM is the
    /// dominant contributor to the p90 tail (~3.1s median when it runs).
    fn clean_transcript(raw: &str) -> (String, bool) {
        let settings = current_settings();
        let level = settings.cleanup_level;
        if matches!(settings.cleanup_mode, CleanupMode::Raw) || level.bypasses_llm() {
            return (raw.to_string(), false);
        }
        // Turn explicit spoken layout cues ("new line", "new paragraph") into break
        // markers up front — the model passes an opaque marker through reliably but
        // mangles the literal cue words. The model sees `raw` (with markers); the gate
        // and any raw fallback use `raw_out` (markers restored to real breaks) so we
        // never paste a "[[NL]]" token or lose an explicit break.
        let raw_norm = whimpr_core::cleanup::pre_normalize_layout(raw);
        let raw = raw_norm.as_str();
        let raw_out = whimpr_core::cleanup::post_process(&raw_norm);
        // Skip the model entirely when the transcript shows no evidence of the mess
        // cleanup exists to fix. The LLM call costs ~2s on an M2 Air regardless of how
        // short the input is (it is almost all fixed prompt prefill), and Whisper
        // already emits punctuated, filler-free text for most short dictations — so
        // paying that toll unconditionally was the single largest avoidable component
        // of end-to-end latency. Deterministic post-processing still runs.
        if !whimpr_core::cleanup::needs_cleanup(raw) {
            eprintln!("[whimpr] cleanup skipped — transcript already clean");
            return (raw_out, false);
        }
        let vocab = DICTIONARY
            .get()
            .map(|d| d.lock().unwrap().prefilter(raw, 15))
            .unwrap_or_default();
        let app_bundle_id = TARGET_APP.get().and_then(|m| m.lock().unwrap().clone());
        if let Some(app) = app_bundle_id.as_deref() {
            eprintln!("[whimpr] cleanup target app: {app}");
        }
        let ctx = CleanupContext {
            level,
            vocab,
            app_bundle_id,
            ..Default::default()
        };
        // Run the on-device model with the same prompt + per-app formatting.
        let run_local = || -> Option<anyhow::Result<String>> {
            LOCAL.get().and_then(|m| {
                m.lock().unwrap().as_mut().map(|w| {
                    // System prompt + few-shot demonstration turns + the transcript,
                    // so the on-device model actually produces newlines/lists and
                    // resolves self-corrections instead of just being told to.
                    let messages = whimpr_core::cleanup::build_messages(raw, &ctx);
                    w.cleanup(&messages)
                })
            })
        };
        // Selected provider, falling back to local when a cloud key can't be read
        // (so cleanup still runs) — and Local mode uses the worker directly.
        let result: Option<anyhow::Result<String>> = match settings.cleanup_mode {
            CleanupMode::OpenAi => OPENAI
                .get()
                .and_then(|m| m.lock().unwrap().as_ref().map(|p| p.cleanup(raw, &ctx)))
                .or_else(run_local),
            CleanupMode::Anthropic => ANTHROPIC
                .get()
                .and_then(|m| m.lock().unwrap().as_ref().map(|p| p.cleanup(raw, &ctx)))
                .or_else(run_local),
            CleanupMode::Local => run_local(),
            CleanupMode::Raw => None,
        };
        // `Some(_)` means a provider was actually called (Ok or Err) -- that's the
        // branch that spent wall-clock time. `None` means the gate wanted cleanup
        // but nothing was available to run it, which costs nothing.
        let fired = result.is_some();
        let text = match result {
            Some(Ok(cleaned)) => {
                // Deterministic safety net: convert any leftover spoken layout cue the
                // model missed into real line breaks, strip stray code fences, cap blank
                // lines. Guarantees no "new line"/"new paragraph" word reaches the cursor.
                let cleaned = whimpr_core::cleanup::post_process(&cleaned);
                if whimpr_core::cleanup::evaluate_gates(&raw_out, &cleaned, level).passed() {
                    cleaned
                } else {
                    eprintln!("[whimpr] cleanup gate rejected the edit — pasting raw");
                    raw_out
                }
            }
            Some(Err(e)) => {
                eprintln!("[whimpr] cleanup failed ({e}) — pasting raw");
                raw_out
            }
            None => {
                if matches!(settings.cleanup_mode, CleanupMode::Local) {
                    eprintln!("[whimpr] local cleanup model not wired yet — pasting raw");
                } else {
                    eprintln!("[whimpr] cleanup provider has no API key — pasting raw");
                }
                raw_out
            }
        };
        (text, fired)
    }

    /// Spawn the math worker if it is not already up, on a background thread.
    ///
    /// Called from the key-down path, which runs on the CGEventTap callback — that
    /// thread must return promptly or macOS disables the tap (and the hotkey dies
    /// until relaunch). Loading a 2.4 GB model there would be several seconds of
    /// exactly the stall that guard exists to prevent.
    ///
    /// Idempotent and safe to call on every math key-press: `MATH_SPAWNING` stops
    /// a second press from starting a second process while the first is loading,
    /// and the slot check stops it from replacing a live worker.
    fn ensure_math_worker() {
        let slot = MATH_LOCAL.get_or_init(|| Mutex::new(None));
        if slot.lock().unwrap().is_some() {
            return;
        }
        // `swap` rather than a check-then-set: two key-presses in quick succession
        // would otherwise both see "not spawning" and both spawn.
        if MATH_SPAWNING.swap(true, Ordering::SeqCst) {
            return;
        }
        std::thread::spawn(|| {
            eprintln!("[whimpr] math worker: loading (first use — this takes a few seconds)");
            let t = Instant::now();
            let mut w = crate::local_llm::spawn_math();
            // 🔴 Spawning is NOT loading, and the difference is the whole point of
            // starting early. `spawn` returns as soon as the child PROCESS exists;
            // llama.cpp then loads the GGUF lazily, so without this the load lands
            // inside the user's first real request instead. Measured 2026-08-17:
            // first math dictation 8500 ms versus 5125 ms warm — a 3.4 s penalty
            // that the key-down head start appeared to remove and did not.
            //
            // So force the load here, on this thread, with a throwaway request.
            // Its content does not matter and its answer is discarded; what
            // matters is that it returns only once the model is resident.
            if let Some(worker) = w.as_mut() {
                let warm = [whimpr_core::cleanup::CleanupMsg {
                    role: "user",
                    content: "hi".to_string(),
                }];
                match worker.request(&warm, 1) {
                    Ok(_) => eprintln!("[whimpr] math worker: ready ({} ms)", t.elapsed().as_millis()),
                    // A failed warmup is worth saying out loud but not worth
                    // discarding the worker over — the real request may still
                    // work, and dropping it here would turn a warning into a
                    // silently dead feature.
                    Err(e) => eprintln!("[whimpr] ⚠ math worker warmup failed ({e}) — keeping it anyway"),
                }
            }
            let slot = MATH_LOCAL.get_or_init(|| Mutex::new(None));
            *slot.lock().unwrap() = w;
            MATH_SPAWNING.store(false, Ordering::SeqCst);
        });
    }

    /// Convert a spoken-mathematics transcript into notation (G2: "'f of g' turns
    /// into `f(g)`"). Returns `(text, fired)`; on any failure the raw transcript
    /// comes back unchanged, because a dictation that is merely un-notated is
    /// still the user's words, and one that is empty is not.
    ///
    /// Runs INSTEAD of cleanup, not after it. Two LLM calls on one dictation
    /// would double the wait for no benefit: `needs_cleanup()` is false on all
    /// ten math fixtures (measured by running it, not by reading it), so on real
    /// math dictation cleanup almost never fires anyway — and where it did, the
    /// two prompts would be arguing over the same text.
    ///
    /// Notation is chosen from the paste target: Unicode by default, LaTeX where
    /// it is actually typeset. See `whimpr_core::mathfmt::format_for_app`.
    fn format_math(raw: &str) -> (String, bool) {
        let app_bundle_id = TARGET_APP.get().and_then(|m| m.lock().unwrap().clone());
        let format = whimpr_core::mathfmt::format_for_app(app_bundle_id.as_deref());
        // Log the model as well as the format. The 4B and the 1.5B differ enough
        // in both accuracy and latency that "which one ran" is the first question
        // to ask of a bad result — and the 4B arrives by a file RENAME in the
        // models dir, which is otherwise an invisible change to this behaviour.
        eprintln!(
            "[whimpr] MATH MODE: format={format:?} app={} model={}",
            app_bundle_id.as_deref().unwrap_or("<unknown>"),
            crate::local_llm::math_model_path()
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default()
        );
        let raw_out = whimpr_core::cleanup::post_process(raw);
        let messages = whimpr_core::mathfmt::build_messages(raw, format);
        // On the very first math dictation the worker is still loading — it was
        // started at key-down, which usually covers it, but a two-second utterance
        // can finish first. Wait rather than fall back to raw: the user asked for
        // notation and silently not converting is the one outcome that looks like
        // the feature is broken. Bounded so a worker that will never come up
        // cannot hang the paste forever.
        if MATH_SPAWNING.load(Ordering::SeqCst) {
            eprintln!("[whimpr] math worker still loading — waiting");
            let t = Instant::now();
            while MATH_SPAWNING.load(Ordering::SeqCst) && t.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(50));
            }
            eprintln!("[whimpr] math worker wait: {} ms", t.elapsed().as_millis());
        }
        let result = MATH_LOCAL
            .get()
            .and_then(|m| m.lock().unwrap().as_mut().map(|w| w.request(&messages, 400)));
        match result {
            Some(Ok(out)) => {
                let out = whimpr_core::mathfmt::finalize(&out);
                // The ONLY rejection here is empty output. Deliberately no
                // length, retention, or similarity gate: a correct dense
                // conversion is far shorter than its input (the Cauchy formula
                // scores 0.26 retention and is perfect), so any such gate rejects
                // good output and passes bad. That was built, measured and killed
                // on 2026-08-17 — see whimpr_core::mathfmt's module header.
                if out.is_empty() {
                    eprintln!("[whimpr] math stage returned nothing — pasting raw");
                    (raw_out, true)
                } else {
                    (out, true)
                }
            }
            Some(Err(e)) => {
                eprintln!("[whimpr] math stage failed ({e}) — pasting raw");
                (raw_out, true)
            }
            None => {
                eprintln!(
                    "[whimpr] ⚠ MATH MODE requested but the local LLM worker is not running — \
                     pasting the raw transcript. Nothing was converted."
                );
                (raw_out, false)
            }
        }
    }

    fn now_ms() -> u64 {
        CLOCK.get().map(|c| c.elapsed().as_millis() as u64).unwrap_or(0)
    }

    fn bar_name(b: BarState) -> &'static str {
        match b {
            BarState::Idle => "idle",
            BarState::Recording => "recording",
            BarState::Locked => "locked",
            BarState::Transcribing => "transcribing",
            BarState::Done => "done",
            BarState::Cancelled => "cancelled",
            BarState::Error => "error",
        }
    }

    fn emit_bar(app: &AppHandle, state: &'static str) {
        eprintln!("[whimpr] pill -> {state}");
        let _ = app.emit_to(OVERLAY_LABEL, "whimpr://flowbar/state", BarPayload { state });
    }

    /// Feed one input into the shared state machine and enact its actions.
    fn handle_input(input: Input) {
        let (Some(app), Some(machine)) = (APP.get(), MACHINE.get()) else {
            return;
        };
        let actions = {
            let mut m = machine.lock().unwrap();
            m.step(input)
        };
        for action in actions {
            apply_action(app, action);
        }
    }

    fn apply_action(app: &AppHandle, action: Action) {
        match action {
            Action::ShowBar(bar) => {
                emit_bar(app, bar_name(bar));
                // Let the "done" tick linger briefly before returning to idle.
                if bar == BarState::Done {
                    let app2 = app.clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(500));
                        emit_bar(&app2, "idle");
                    });
                }
            }
            // Start the microphone; stream real RMS bars to the pill waveform.
            // Runs off the tap thread so the mic-permission prompt can't stall keys.
            Action::StartCapture { .. } => {
                // Everything from here to whimpr_audio::start()'s device-open call
                // is synchronous dispatch (tap callback -> state machine -> this
                // arm), so this is effectively the key-down instant -- see
                // whimpr_audio::CaptureStartTiming.
                let key_down_at = Instant::now();
                let app_thread = app.clone();
                std::thread::spawn(move || {
                    let app_cb = app_thread.clone();
                    match whimpr_audio::start(key_down_at, move |bars| {
                        let _ = app_cb.emit_to(
                            OVERLAY_LABEL,
                            "whimpr://audio/waveform",
                            WavePayload { bars: bars.to_vec() },
                        );
                    }) {
                        Ok(handle) => {
                            *CAPTURE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(handle);
                        }
                        Err(e) => eprintln!("[whimpr] mic capture failed to start: {e}"),
                    }
                });
            }
            // Stop the mic, transcribe the buffered audio, and advance the machine.
            Action::StopCaptureAndFinalize { session } => {
                let app2 = app.clone();
                let handle = CAPTURE.get().and_then(|slot| slot.lock().unwrap().take());
                std::thread::spawn(move || {
                    // Whatever happens, return the pill to idle (done -> idle).
                    let finish =
                        || handle_input(Input::Pipeline(PipelineEvent::Committed { session }));
                    let Some(res) = handle.and_then(|h| h.stop()) else {
                        eprintln!("[whimpr] no audio captured");
                        finish();
                        return;
                    };
                    let peak = res.samples.iter().fold(0f32, |m, &s| m.max(s.abs()));
                    eprintln!(
                        "[whimpr] captured {} samples @ {} Hz (~{:.2}s), peak {:.4}",
                        res.samples.len(),
                        res.sample_rate,
                        res.duration_secs(),
                        peak
                    );
                    if peak < 0.005 {
                        // Do NOT say "grant access to your terminal" here. This app is
                        // normally launched from Finder/Dock/launchd, where the terminal
                        // is not in the picture at all, and that wording has already sent
                        // Max toggling permissions for a problem that was never about
                        // permissions (the real cause that time was the input device
                        // flipping to a Bluetooth mic that delivers silence).
                        eprintln!(
                            "[whimpr] ⚠ audio is silent — nothing was captured. Either the \
                             selected input device isn't delivering audio (check the device \
                             name logged at capture start — a Bluetooth mic that just \
                             connected will return silence for a few seconds), or WhimprFlow \
                             itself lacks Microphone access (System Settings → Privacy & \
                             Security → Microphone). Grant it to WhimprFlow, not to a terminal."
                        );
                    }
                    let Some(asr) = ASR.get().cloned() else {
                        eprintln!("[whimpr] ASR not ready (model still loading or missing)");
                        finish();
                        return;
                    };
                    // Per-stage timing. Everything here is serial after key release, so
                    // this is exactly the latency the user waits through before text
                    // appears — the number to optimize against.
                    let t_start = Instant::now();
                    // Strip a long thinking-pause from the front before whisper sees
                    // it. Past ~2s of leading silence whisper starts losing the speech
                    // that follows outright -- see whimpr_audio::trim_leading_silence
                    // for the measured table. Trimming here rather than after the
                    // resample also means the anti-alias filter runs over less audio.
                    let trimmed = whimpr_audio::trim_leading_silence(&res.samples, res.sample_rate);
                    let cut = res.samples.len() - trimmed.len();
                    let trim_cut_s = cut as f32 / res.sample_rate.max(1) as f32;
                    if cut > 0 {
                        eprintln!("[whimpr] trimmed {trim_cut_s:.2}s of leading silence");
                    }
                    let clip_posttrim_s = trimmed.len() as f32 / res.sample_rate.max(1) as f32;
                    let pcm = whimpr_audio::resample_to_16k(trimmed, res.sample_rate);
                    let ms_resample = t_start.elapsed().as_millis();
                    // Bias decoding toward the user's own vocabulary. Unlike the
                    // cleanup prompt this cannot be filtered to the utterance — there
                    // is no transcript yet — so the whole dictionary goes in, trimmed
                    // to whisper's 224-token budget inside the ASR crate. Costs no
                    // measurable time and applies to every dictation, including the
                    // majority that the cleanup gate skips.
                    let asr_prompt = DICTIONARY
                        .get()
                        .and_then(|d| d.lock().unwrap().asr_prompt());
                    if let Some(p) = asr_prompt.as_deref() {
                        eprintln!("[whimpr] asr prompt: \"{p}\"");
                    }
                    let t_asr = Instant::now();
                    match asr.transcribe_with_prompt(&pcm, asr_prompt.as_deref()) {
                        Ok(t) => {
                            let ms_asr = t_asr.elapsed().as_millis();
                            let raw = t.text;
                            eprintln!("[whimpr] TRANSCRIPT: \"{}\"", raw);
                            // Either convert spoken mathematics to notation (the
                            // Shift+hotkey gesture) or clean the transcript — not
                            // both. See format_math() for why they are exclusive.
                            let math_mode = MATH_MODE.load(Ordering::SeqCst);
                            let t_clean = Instant::now();
                            let (text, cleanup_fired) = if math_mode {
                                format_math(&raw)
                            } else {
                                clean_transcript(&raw)
                            };
                            let ms_clean = t_clean.elapsed().as_millis();
                            if text != raw {
                                eprintln!(
                                    "[whimpr] {}: \"{}\"",
                                    if math_mode { "MATH     " } else { "CLEANED  " },
                                    text
                                );
                            }
                            // Hot lane, deterministic stage. Placed *after* cleanup on
                            // purpose: cleanup is an LLM and can rewrite a word we just
                            // fixed, so the dictionary has to get the last word for the
                            // correction to actually be a guarantee. Cleanup is not left
                            // blind by this — `prefilter()` already puts the relevant
                            // entries in its prompt.
                            let t_dict = Instant::now();
                            let (text, dict_hits) = dictionary_apply(&text);
                            let us_dict = t_dict.elapsed().as_micros();
                            if dict_hits > 0 {
                                eprintln!(
                                    "[whimpr] DICTIONARY: \"{}\" ({} replacement(s), {}µs)",
                                    text, dict_hits, us_dict
                                );
                            }

                            // Second, idempotent application of the pause strip.
                            // The ASR crate already ran it on whisper's own output,
                            // which is where 15 of the 16 measured occurrences live.
                            // The sixteenth is *created here*: whisper heard
                            // "Wimperslow, Aeropod, Bug. New line.", and cleanup's
                            // layout-cue rewrite turned "New line." into a newline
                            // and left the period behind on a line of its own
                            // (whimpr-2026-08-12.log:567). Cleanup has never added an
                            // ellipsis in 206 dictations, so this costs nothing on the
                            // normal path — but it is the only thing standing between
                            // that stray period and Max's document.
                            let pre_strip = text;
                            let text = whimpr_asr::strip_pause_punctuation(&pre_strip);
                            if text != pre_strip {
                                eprintln!("[whimpr] PAUSE-STRIP: \"{}\"", text);
                            }
                            if !text.is_empty() {
                                let t_paste = Instant::now();
                                if let Err(e) = crate::paste::paste_text(&text) {
                                    eprintln!("[whimpr] paste failed: {e}");
                                }
                                let ms_paste = t_paste.elapsed().as_millis();
                                let ms_total = t_start.elapsed().as_millis();
                                eprintln!(
                                    "[whimpr] ⏱ audio {:.1}s | resample {}ms | asr {}ms | \
                                     cleanup {}ms | paste {}ms | TOTAL {}ms",
                                    res.duration_secs(),
                                    ms_resample,
                                    ms_asr,
                                    ms_clean,
                                    ms_paste,
                                    ms_total,
                                );
                                // One structured line per dictation (3b) -- everything
                                // above in one greppable, machine-parseable record. See
                                // scripts/read-logs.py.
                                let metrics = DictationMetrics {
                                    ts: unix_now(),
                                    words: whimpr_core::stats::count_words(&text),
                                    clip_pretrim_s: res.duration_secs(),
                                    clip_posttrim_s,
                                    trim_engaged: cut > 0,
                                    trim_cut_s,
                                    capture_start_ms: res.start_timing.first_sample_ms,
                                    resample_ms: ms_resample as u64,
                                    asr_ms: ms_asr as u64,
                                    cleanup_fired,
                                    cleanup_ms: ms_clean as u64,
                                    math_mode,
                                    dict_hits,
                                    dict_us: us_dict as u64,
                                    paste_ms: ms_paste as u64,
                                    total_ms: ms_total as u64,
                                };
                                match serde_json::to_string(&metrics) {
                                    Ok(json) => eprintln!("[whimpr-metrics] {json}"),
                                    Err(e) => eprintln!("[whimpr] metrics serialize failed: {e}"),
                                }
                                // Log words + speaking time for the Hub stats (WPM, streak…).
                                record_dictation(&text, res.duration_secs());
                                // Watch the field for a post-paste correction to learn (✨).
                                crate::autolearn::watch_correction(&text);
                            }
                            let _ = app2.emit_to(
                                OVERLAY_LABEL,
                                "whimpr://transcript",
                                TranscriptPayload { text },
                            );
                        }
                        Err(e) => eprintln!("[whimpr] ASR error: {e}"),
                    }
                    finish();
                });
            }
            Action::DiscardCapture { .. } => {
                if let Some(slot) = CAPTURE.get() {
                    if let Some(handle) = slot.lock().unwrap().take() {
                        let _ = handle.stop();
                    }
                }
            }
            // The ASR path (StopCaptureAndFinalize) now drives pipeline completion.
            Action::RunPipeline { .. } => {}
            // PlayPing / WarnSessionCap: no-ops for now.
            _ => {}
        }
    }

    extern "C" fn tap_callback(
        _proxy: CGEventTapProxy,
        etype: u32,
        event: CGEventRef,
        _info: *mut c_void,
    ) -> CGEventRef {
        if etype == K_CG_TAP_DISABLED_BY_TIMEOUT || etype == K_CG_TAP_DISABLED_BY_USER_INPUT {
            // macOS disables a tap silently under load or around sleep/wake, and it
            // stays dead until something re-enables it. Re-enabling here is not new
            // — it dates to the original PoC — but it used to do so without a word,
            // which is why nobody could tell from a log whether this path had ever
            // fired. The "hotkey died after I closed the lid" reports have never been
            // reproduced; if they recur, this line is the evidence to look for.
            let why = if etype == K_CG_TAP_DISABLED_BY_TIMEOUT {
                "timeout"
            } else {
                "user input"
            };
            let port = TAP_PORT.load(Ordering::SeqCst);
            if !port.is_null() {
                unsafe { CGEventTapEnable(port, true) };
                eprintln!("[whimpr] event tap was disabled by macOS ({why}) — re-enabled");
            } else {
                eprintln!(
                    "[whimpr] ⚠ event tap disabled by macOS ({why}) but the port is null — \
                     the hotkey is now dead until relaunch"
                );
            }
            return event;
        }
        if etype == K_CG_EVENT_FLAGS_CHANGED {
            let keycode =
                unsafe { CGEventGetIntegerValueField(event, K_CG_KEYBOARD_EVENT_KEYCODE) };
            // ✅ ANSWERED 2026-08-17: **Fn's flagsChanged event does carry the
            // Shift bit**, so the math gesture works on stable exactly as it does
            // on dev. Measured with a temporary probe on this line, pressing
            // Shift+Fn: `flags=0x00820102 shift=true fn=true`. This was worth
            // checking rather than assuming — Fn is not an ordinary modifier, and
            // had it not carried Shift, the feature would have installed over the
            // daily driver and silently done nothing, looking like a bad build.
            // The probe is deleted; the answer is the reason it does not need to
            // exist.
            if keycode == KEYCODE_HOTKEY {
                let flags = unsafe { CGEventGetFlags(event) };
                let down = (flags & FLAG_HOTKEY_MODIFIER) != 0;
                let was_down = FN_IS_DOWN.swap(down, Ordering::SeqCst);
                let at_ms = now_ms();
                if down && !was_down {
                    // Latch math mode from the modifier state carried on THIS
                    // event. It has to be read here and stored: by the time the
                    // transcript exists the key is long released, and re-reading
                    // the keyboard then would report whatever the user happens to
                    // be holding seconds later.
                    let math = (flags & FLAG_MATH_MODIFIER) != 0;
                    MATH_MODE.store(math, Ordering::SeqCst);
                    if math {
                        // Start loading the math model NOW rather than at
                        // finalize. The user is about to speak for several
                        // seconds, and the load overlaps with that instead of
                        // being added to the wait after they stop. Off the tap
                        // thread: the keyboard callback must never block, or
                        // macOS disables the tap out from under us.
                        ensure_math_worker();
                    }
                    eprintln!(
                        "[whimpr] Fn DOWN{}",
                        if math { " + Shift — MATH MODE" } else { "" }
                    );
                    // Snapshot the paste target now, while the user's app is focused.
                    let target = crate::appctx::frontmost_bundle_id();
                    *TARGET_APP.get_or_init(|| Mutex::new(None)).lock().unwrap() = target;
                    handle_input(Input::Trigger(TriggerToken::Down {
                        binding: BindingId::PushToTalk,
                        at_ms,
                    }));
                } else if !down && was_down {
                    eprintln!("[whimpr] Fn UP");
                    handle_input(Input::Trigger(TriggerToken::Up {
                        binding: BindingId::PushToTalk,
                        at_ms,
                    }));
                }
            }
        }
        event
    }

    pub fn install(app: AppHandle) {
        let _ = APP.set(app);
        let _ = MACHINE.set(Mutex::new(StateMachine::new()));
        let _ = CLOCK.set(Instant::now());

        // Load the speech-to-text model off the main thread (it takes ~1s).
        std::thread::spawn(|| {
            let path = model_path();
            if !path.exists() {
                eprintln!("[whimpr] ASR model not found at {}", path.display());
                return;
            }
            match whimpr_asr::WhisperEngine::load(&path) {
                Ok(engine) => {
                    let _ = ASR.set(Arc::new(engine));
                    eprintln!("[whimpr] ASR model loaded — ready to transcribe");
                }
                Err(e) => eprintln!("[whimpr] ASR model load failed: {e}"),
            }
        });

        // Load settings + dictionary, and build cloud providers from stored keys.
        let settings = whimpr_core::Settings::load(&settings_path());
        let dict = whimpr_core::DictionaryStore::load(&dict_path());
        eprintln!(
            "[whimpr] cleanup mode: {:?}, level: {:?}",
            settings.cleanup_mode, settings.cleanup_level
        );
        let _ = SETTINGS.set(Mutex::new(settings));
        let _ = DICTIONARY.set(Mutex::new(dict));
        let _ = STATS.set(Mutex::new(whimpr_core::StatsStore::load(&stats_path())));
        rebuild_providers();

        // Start the local cleanup worker in the background (model load takes a few
        // seconds; the first local cleanup waits for it, subsequent ones are fast).
        std::thread::spawn(|| {
            let worker = crate::local_llm::spawn_default();
            let _ = LOCAL.set(Mutex::new(worker));
        });

        // Accessibility is the ONE permission that makes the Fn CGEventTap global AND
        // lets us post the Cmd+V paste into other apps. Without it, a keyboard tap is
        // silently limited to frontmost-only — the exact bug. Prompt for it up front.
        if crate::paste::is_trusted() {
            eprintln!(
                "[whimpr] Accessibility granted — Fn works in every app, paste enabled"
            );
        } else {
            eprintln!(
                "[whimpr] ⚠ Accessibility NOT granted — Fn only works while WhimprFlow \
                 is frontmost and paste is disabled. Prompting; grant WhimprFlow under System \
                 Settings → Privacy & Security → Accessibility (no relaunch needed)."
            );
            crate::paste::prompt_accessibility();
        }
        // Input Monitoring is NOT the gate for a CGEventTap — kept only as diagnostics.
        eprintln!(
            "[whimpr] (info) Input Monitoring: {}",
            crate::paste::input_monitoring_granted()
        );

        // Periodic tick drives the double-tap timeout / session cap.
        std::thread::spawn(|| loop {
            std::thread::sleep(Duration::from_millis(100));
            handle_input(Input::Tick { now_ms: now_ms() });
        });

        // The event tap runs on a thread with its own CFRunLoop. CRITICAL: create it
        // ONLY after the process is trusted for Accessibility. macOS fixes a keyboard
        // tap's privilege at CGEventTapCreate time — a tap born untrusted is
        // permanently frontmost-only and is NOT upgraded when the grant later arrives.
        // Polling here also means the hotkey starts working the moment the user grants
        // Accessibility, without a relaunch.
        std::thread::spawn(|| {
            while !crate::paste::is_trusted() {
                std::thread::sleep(Duration::from_millis(500));
            }
            eprintln!("[whimpr] Accessibility present — creating global Fn tap");

            // Retry rather than give up. AXIsProcessTrusted() flipping true and the
            // process actually being allowed to create a keyboard tap are not the
            // same instant, so the first attempt after a fresh grant can return null
            // on a race. This used to `return`, killing the hotkey until the app was
            // relaunched — which matches the recurring "Fn opens the
            // overlay but nothing completes, and a clean relaunch fixes it" reports
            // better than anything else in this file.
            //
            // Backoff to 2s and keep trying: a permanently-failing tap costs one log
            // line every two seconds, and a hotkey that fixes itself thirty seconds
            // later is far better than one that is dead until relaunch.
            let mut port: CFMachPortRef = null_mut();
            let mut attempt = 0u32;
            while port.is_null() {
                attempt += 1;
                port = unsafe {
                    CGEventTapCreate(
                        K_CG_SESSION_EVENT_TAP,
                        K_CG_HEAD_INSERT,
                        K_CG_TAP_OPTION_LISTEN_ONLY,
                        EVENTS_OF_INTEREST,
                        tap_callback,
                        null_mut(),
                    )
                };
                if port.is_null() {
                    if attempt == 1 || attempt % 15 == 0 {
                        eprintln!(
                            "[whimpr] ⚠ hotkey tap null despite Accessibility (attempt {attempt}) \
                             — retrying. If this persists, a stale TCC entry from an earlier \
                             build is the usual cause: run `tccutil reset Accessibility \
                             com.whimpr.whimprflow`, re-grant, and relaunch."
                        );
                    }
                    std::thread::sleep(Duration::from_millis(
                        std::cmp::min(200 * attempt as u64, 2000),
                    ));
                }
            }
            if attempt > 1 {
                eprintln!("[whimpr] hotkey tap created on attempt {attempt}");
            }
            TAP_PORT.store(port, Ordering::SeqCst);

            // Watchdog. The callback above re-enables a tap that macOS disabled, but
            // that only works if the disable is delivered as an event. This checks
            // the tap's own state directly, so a tap that went dead without a
            // callback — the shape the unreproduced sleep/wake reports would take —
            // recovers within a few seconds instead of lasting until relaunch.
            std::thread::spawn(|| loop {
                std::thread::sleep(Duration::from_secs(3));
                let port = TAP_PORT.load(Ordering::SeqCst);
                if !port.is_null() && !unsafe { CGEventTapIsEnabled(port) } {
                    eprintln!("[whimpr] ⚠ watchdog: event tap found disabled — re-enabling");
                    unsafe { CGEventTapEnable(port, true) };
                }
            });

            unsafe {
                let source = CFMachPortCreateRunLoopSource(null(), port, 0);
                CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopDefaultMode);
                CGEventTapEnable(port, true);
                CFRunLoopRun();
            }
        });
    }
}

#[cfg(target_os = "macos")]
pub use imp::{
    current_settings, dictionary_add, dictionary_approve, dictionary_entries, dictionary_learn,
    dictionary_remove,
    history, init_logging, install, rebuild_providers, stats_summary, update_settings,
};

// Windows uses the real (but unverified) platform layer in `crate::win`.
#[cfg(target_os = "windows")]
pub use crate::win::{
    current_settings, dictionary_add, dictionary_approve, dictionary_entries, dictionary_learn,
    dictionary_remove,
    history, init_logging, install, rebuild_providers, stats_summary, update_settings,
};

// Other platforms (Linux, etc.): inert stubs so the crate still builds.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod other {
    pub fn install(_app: tauri::AppHandle) {}
    pub fn current_settings() -> whimpr_core::Settings {
        whimpr_core::Settings::default()
    }
    pub fn update_settings(_new: whimpr_core::Settings) {}
    pub fn rebuild_providers() {}
    pub fn stats_summary(tz_offset_minutes: i32) -> whimpr_core::StatsSummary {
        whimpr_core::StatsStore::default().summary(tz_offset_minutes, 0)
    }
    pub fn history(_limit: usize) -> Vec<whimpr_core::HistoryItem> {
        Vec::new()
    }
    pub fn dictionary_entries() -> Vec<super::DictEntryDto> {
        Vec::new()
    }
    pub fn dictionary_add(_correct: String, _mishears: Vec<String>) {}
    pub fn dictionary_remove(_correct: &str) {}
    pub fn dictionary_approve(_correct: &str) {}
    pub fn dictionary_learn(_correct: String, _mishears: Vec<String>) {}
    pub fn init_logging() {}
}
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub use other::{
    current_settings, dictionary_add, dictionary_approve, dictionary_entries, dictionary_learn,
    dictionary_remove,
    history, init_logging, install, rebuild_providers, stats_summary, update_settings,
};
