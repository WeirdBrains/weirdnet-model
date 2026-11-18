//! Where model files live on disk, and which Hub they come from.

use std::path::{Path, PathBuf};

use crate::catalog::ModelSpec;

/// The public Hugging Face Hub.
pub const HUB: &str = "https://huggingface.co";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStore {
    dir: PathBuf,
    hub: String,
}

impl ModelStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), hub: HUB.into() }
    }

    /// `$WEIRDNET_MODELS_DIR`, else `<user data dir>/weirdnet/models`; the Hub from
    /// `$WEIRDNET_HF_URL` (a mirror) when set.
    pub fn from_env() -> Self {
        let dir = std::env::var_os("WEIRDNET_MODELS_DIR")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("weirdnet").join("models"));
        let mut store = Self::new(dir);
        if let Some(hub) = std::env::var("WEIRDNET_HF_URL").ok().filter(|u| !u.trim().is_empty()) {
            store.hub = hub.trim_end_matches('/').to_string();
        }
        store
    }

    /// Use another Hub (a mirror, or a local server in tests).
    pub fn with_hub(mut self, hub: impl Into<String>) -> Self {
        self.hub = hub.into().trim_end_matches('/').to_string();
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn hub(&self) -> &str {
        &self.hub
    }

    /// Where `spec`'s file goes.
    pub fn path(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(&spec.file)
    }

    /// Downloaded and complete: the size matches (the checksum was checked when it finished).
    pub fn installed(&self, spec: &ModelSpec) -> bool {
        std::fs::metadata(self.path(spec)).is_ok_and(|m| m.len() == spec.size)
    }

    /// Bytes of a partial download, if any.
    pub fn partial_bytes(&self, spec: &ModelSpec) -> u64 {
        std::fs::metadata(self.part_path(spec)).map(|m| m.len()).unwrap_or(0)
    }

    pub(crate) fn part_path(&self, spec: &ModelSpec) -> PathBuf {
        self.dir.join(format!("{}.part", spec.file))
    }

    /// Deletes the file and any partial download. Returns whether a file was removed.
    pub fn remove(&self, spec: &ModelSpec) -> bool {
        let _ = std::fs::remove_file(self.part_path(spec));
        std::fs::remove_file(self.path(spec)).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;

    #[test]
    fn paths_follow_the_file_name_and_remove_clears_parts() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let spec = catalog::default_for(catalog::Device::Phone);
        assert_eq!(store.path(&spec), tmp.path().join("gemma-4-E2B-it-Q4_0.gguf"));
        assert!(!store.installed(&spec));
        std::fs::write(store.part_path(&spec), b"abc").unwrap();
        assert_eq!(store.partial_bytes(&spec), 3);
        assert!(!store.remove(&spec));
        assert_eq!(store.partial_bytes(&spec), 0);
    }

    #[test]
    fn a_mirror_loses_its_trailing_slash() {
        assert_eq!(ModelStore::new("/x").with_hub("http://127.0.0.1:1/").hub(), "http://127.0.0.1:1");
    }
}
