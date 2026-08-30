//! User settings, persisted as JSON. Drives the cleanup engine (which provider,
//! how aggressive) and other behavior. Kept dependency-light so it lives in core.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cleanup::CleanupLevel;

/// Which cleanup engine processes transcripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CleanupMode {
    /// Paste the raw transcript (no cleanup).
    Raw,
    /// Local on-device model (default — works offline, no API key).
    #[default]
    Local,
    /// OpenAI cloud.
    OpenAi,
    /// Anthropic cloud.
    Anthropic,
}

/// Persisted user configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub cleanup_mode: CleanupMode,
    pub cleanup_level: CleanupLevel,
    pub openai_model: String,
    /// API root for the "OpenAI" cleanup mode, e.g. `https://openrouter.ai/api/v1`
    /// to route through OpenRouter instead of OpenAI directly (same wire format).
    /// Empty string (the default) means OpenAI's own endpoint.
    #[serde(default)]
    pub openai_base_url: String,
    pub anthropic_model: String,
    /// Play the record-start ping.
    pub sound_on_start: bool,
    /// Show the Flow Bar pill while idle (nothing being dictated). Off means the
    /// pill only appears once the hotkey is held, and disappears again after.
    ///
    /// `serde(default)` is load-bearing: `Settings::load` falls back to
    /// `Settings::default()` on ANY deserialize error, so a settings.json written
    /// before this field existed would otherwise reset every other setting.
    #[serde(default = "default_true")]
    pub show_idle_pill: bool,
}

fn default_true() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            cleanup_mode: CleanupMode::default(),
            cleanup_level: CleanupLevel::Light,
            openai_model: "gpt-4o-mini".to_string(),
            openai_base_url: String::new(),
            anthropic_model: "claude-haiku-4-5".to_string(),
            sound_on_start: true,
            show_idle_pill: true,
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let s = Settings::default();
        assert_eq!(s.cleanup_mode, CleanupMode::Local);
        assert_eq!(s.cleanup_level, CleanupLevel::Light);
    }

    /// A settings.json written before `show_idle_pill` existed must keep every
    /// other value. Without the field's `serde(default)`, `load` would fail to
    /// deserialize and silently hand back a wholly default Settings.
    #[test]
    fn older_settings_json_keeps_its_other_values() {
        let legacy = r#"{
            "cleanup_mode": "raw",
            "cleanup_level": "high",
            "openai_model": "gpt-4o-mini",
            "openai_base_url": "",
            "anthropic_model": "claude-haiku-4-5",
            "sound_on_start": false
        }"#;
        let s: Settings = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.cleanup_mode, CleanupMode::Raw);
        assert_eq!(s.cleanup_level, CleanupLevel::High);
        assert!(!s.sound_on_start);
        assert!(s.show_idle_pill, "missing field defaults to visible");
    }

    #[test]
    fn round_trips_json() {
        let s = Settings {
            cleanup_mode: CleanupMode::Local,
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cleanup_mode, CleanupMode::Local);
    }
}
