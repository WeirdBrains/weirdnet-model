//! Where model files live on disk, and which Hub they come from.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::ModelSpec;
use crate::hf::{Hub, split_part};
use crate::token::HfToken;

/// The public Hugging Face Hub.
pub const HUB: &str = "https://huggingface.co";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStore {
    dir: PathBuf,
    hub: String,
    token: Option<HfToken>,
    /// The token goes to any hub, not only the public one (set by [`ModelStore::with_token`]).
    token_any_hub: bool,
}

/// A model file (or split model) in the store, from [`ModelStore::list`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredModel {
    /// The file stem without the split suffix, as in [`RepoModel::id`](crate::hf::RepoModel::id).
    pub id: String,
    /// Paths relative to the store directory, first part first.
    pub files: Vec<String>,
    pub size: u64,
    /// Every part of a split model is there.
    pub complete: bool,
}

impl ModelStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), hub: HUB.into(), token: None, token_any_hub: false }
    }

    /// `$WEIRDNET_MODELS_DIR`, else `<user data dir>/weirdnet/models`; the Hub from
    /// `$WEIRDNET_HF_URL` (a mirror) when set; this node's Hugging Face token
    /// ([`HfToken::from_env`]), which is sent only to the public Hub, never to a mirror.
    pub fn from_env() -> Self {
        let dir = std::env::var_os("WEIRDNET_MODELS_DIR")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("weirdnet").join("models"));
        let mut store = Self::new(dir);
        if let Some(hub) = std::env::var("WEIRDNET_HF_URL").ok().filter(|u| !u.trim().is_empty()) {
            store.hub = hub.trim_end_matches('/').to_string();
        }
        store.token = HfToken::from_env();
        store
    }

    /// Sends `token` to this store's Hub, whichever it is (a mirror too).
    pub fn with_token(mut self, token: Option<HfToken>) -> Self {
        self.token = token;
        self.token_any_hub = true;
        self
    }

    /// The token requests to this store's Hub carry, if any.
    pub(crate) fn token(&self) -> Option<&HfToken> {
        self.token.as_ref().filter(|_| self.token_any_hub || self.hub == HUB)
    }

    /// Whether requests to the Hub carry a token.
    pub fn has_token(&self) -> bool {
        self.token().is_some()
    }

    /// A [`Hub`] client for this store's Hub, with the token when it may go there.
    pub fn hub_client(&self) -> Hub {
        Hub::new(&self.hub).with_token(self.token().cloned())
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

    /// The GGUF models in the store (split parts grouped; projector, draft and `.part` files
    /// left out), sorted by id.
    pub fn list(&self) -> Vec<StoredModel> {
        let mut files = Vec::new();
        walk(&self.dir, &self.dir, 0, &mut files);
        let mut out: std::collections::BTreeMap<String, (Vec<Part>, u32)> = Default::default();
        for (rel, size) in files {
            let (key, part, of) = match split_part(&rel) {
                Some((base, part, of)) => {
                    let dir = rel.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
                    (format!("{dir}{base}"), part, of)
                }
                None => (rel.strip_suffix(".gguf").or_else(|| rel.strip_suffix(".GGUF")).unwrap_or(&rel).to_string(), 1, 1),
            };
            let e = out.entry(key).or_default();
            e.0.push((part, rel, size));
            e.1 = of;
        }
        let mut models: Vec<StoredModel> = out
            .into_iter()
            .map(|(key, (mut parts, of))| {
                parts.sort();
                let complete = parts.len() as u32 == of && parts[0].0 == 1;
                StoredModel {
                    id: key.rsplit('/').next().unwrap_or(&key).to_string(),
                    size: parts.iter().map(|p| p.2).sum(),
                    files: parts.into_iter().map(|p| p.1).collect(),
                    complete,
                }
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.files.cmp(&b.files)));
        models
    }

    /// Deletes a listed model's files (and their `.part` files) and the directories that leaves
    /// empty. Returns the bytes freed. Paths outside the store are refused.
    pub fn remove_local(&self, model: &StoredModel) -> crate::Result<u64> {
        let mut freed = 0;
        for rel in &model.files {
            crate::hf::check_file(rel)?;
            let path = self.dir.join(rel);
            let _ = std::fs::remove_file(self.dir.join(format!("{rel}.part")));
            if let Ok(m) = std::fs::metadata(&path) {
                std::fs::remove_file(&path)?;
                freed += m.len();
            }
            let mut dir = path.parent().map(Path::to_path_buf);
            while let Some(d) = dir.filter(|d| d.starts_with(&self.dir) && *d != self.dir) {
                if std::fs::remove_dir(&d).is_err() {
                    break;
                }
                dir = d.parent().map(Path::to_path_buf);
            }
        }
        Ok(freed)
    }
}

/// A file of a stored model: part number, path relative to the store, size.
type Part = (u32, String, u64);

/// Model GGUFs under `dir`, as `/`-separated paths relative to `root` (four levels deep at most).
fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let Ok(ft) = e.file_type() else { continue };
        let path = e.path();
        if ft.is_dir() && depth < 4 {
            walk(root, &path, depth + 1, out);
        } else if ft.is_file() {
            let name = e.file_name().to_string_lossy().to_ascii_lowercase();
            if !name.ends_with(".gguf") || name.starts_with("mmproj") || name.starts_with("mtp-") {
                continue;
            }
            let Ok(rel) = path.strip_prefix(root) else { continue };
            let rel: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            out.push((rel.join("/"), e.metadata().map(|m| m.len()).unwrap_or(0)));
        }
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
    fn lists_and_removes_models_with_their_parts() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let w = |rel: &str, n: usize| {
            let p = tmp.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        w("a-Q4_0.gguf", 3);
        w("mmproj-a.gguf", 1);
        w("b.gguf.part", 1);
        w("Q8_0/b-Q8_0-00001-of-00002.gguf", 5);
        w("Q8_0/b-Q8_0-00002-of-00002.gguf", 6);
        w("c-00002-of-00002.gguf", 2);
        let list = store.list();
        assert_eq!(
            list.iter().map(|m| (m.id.as_str(), m.size, m.complete)).collect::<Vec<_>>(),
            [("a-Q4_0", 3, true), ("b-Q8_0", 11, true), ("c", 2, false)]
        );
        assert_eq!(list[1].files, ["Q8_0/b-Q8_0-00001-of-00002.gguf", "Q8_0/b-Q8_0-00002-of-00002.gguf"]);
        w("Q8_0/b-Q8_0-00002-of-00002.gguf.part", 1);
        assert_eq!(store.remove_local(&list[1]).unwrap(), 11);
        assert!(!tmp.path().join("Q8_0").exists(), "the emptied directory goes too");
        assert_eq!(store.list().len(), 2);
        let mut evil = list[0].clone();
        evil.files = vec!["../x.gguf".into()];
        assert!(store.remove_local(&evil).is_err());
    }

    #[test]
    fn tokens_go_to_the_public_hub_or_where_they_were_set() {
        let t = HfToken::new("hf_x");
        let mut store = ModelStore::new("/x");
        store.token = t.clone();
        assert!(store.has_token() && store.hub_client().has_token());
        let mirror = store.clone().with_hub("http://mirror");
        assert!(!mirror.has_token(), "an environment token never goes to a mirror");
        assert!(mirror.with_token(t).has_token());
        assert!(format!("{store:?}").contains("<redacted>"));
    }

    #[test]
    fn a_mirror_loses_its_trailing_slash() {
        assert_eq!(ModelStore::new("/x").with_hub("http://127.0.0.1:1/").hub(), "http://127.0.0.1:1");
    }
}
