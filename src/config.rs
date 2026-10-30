use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera_id: Option<String>,
    pub scan_seconds: f32,
    pub muted: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_id: None,
            scan_seconds: 3.0,
            muted: false,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        Self::load_from(&Self::path())
    }

    fn load_from(path: &Path) -> Self {
        let mut config = fs::read(path)
            .ok()
            .and_then(|data| serde_json::from_slice::<Self>(&data).ok())
            .unwrap_or_default();
        config.scan_seconds = config.scan_seconds.clamp(2.0, 6.0);
        config
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::path())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("写入配置失败：{}", temporary.display()))?;
        atomic_replace(&temporary, path)?;
        Ok(())
    }

    fn path() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("NBPick")
            .join("config.json")
    }
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::Storage::FileSystem::{
            MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
        },
        core::PCWSTR,
    };
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    unsafe {
        MoveFileExW(
            PCWSTR(source.as_ptr()),
            PCWSTR(destination.as_ptr()),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    }
    .map_err(|error| std::io::Error::other(error.to_string()))
}

#[cfg(not(windows))]
fn atomic_replace(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

pub fn choose_camera<T: Clone>(
    devices: &[T],
    remembered: &str,
    id: impl Fn(&T) -> &str,
    name: impl Fn(&T) -> &str,
) -> Option<usize> {
    if devices.is_empty() {
        return None;
    }
    devices
        .iter()
        .position(|d| id(d) == remembered)
        .or_else(|| {
            devices
                .iter()
                .position(|d| id(d).contains("USB") || name(d).to_ascii_lowercase().contains("usb"))
        })
        .or(Some(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_config_and_out_of_range_duration_are_safe() {
        let mut c: Config = serde_json::from_slice(b"not json").unwrap_or_default();
        c.scan_seconds = 99.0;
        c.scan_seconds = c.scan_seconds.clamp(2.0, 6.0);
        assert_eq!(c.scan_seconds, 6.0);
    }
    #[test]
    fn device_fallback_prefers_remembered_then_usb() {
        let d = [("builtin", "Integrated"), ("usb", "Class USB Camera")];
        assert_eq!(choose_camera(&d, "usb", |x| x.0, |x| x.1), Some(1));
        assert_eq!(choose_camera(&d, "missing", |x| x.0, |x| x.1), Some(1));
        assert_eq!(choose_camera(&d[..1], "missing", |x| x.0, |x| x.1), Some(0));
    }
    #[test]
    fn settings_round_trip_to_disk() {
        let path = std::env::temp_dir().join(format!(
            "nb-pick-config-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let expected = Config {
            camera_id: Some("camera-usb-1".into()),
            scan_seconds: 5.0,
            muted: true,
        };
        expected.save_to(&path).unwrap();
        assert_eq!(Config::load_from(&path), expected);
        let _ = fs::remove_file(path);
    }
}
