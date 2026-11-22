//! Where model files live on disk, and which Hub they come from. LoRA adapters live apart, in
//! `adapters/<id>/` ([`ModelStore::adapters`]), and are never listed as models.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::ModelSpec;
use crate::hf::{AdapterFormat, Hub, adapter_hint, split_part};
use crate::token::HfToken;

/// The store's subdirectory for adapters: one directory per adapter id.
pub const ADAPTERS_DIR: &str = "adapters";
/// What [`download_adapter`](crate::download::download_adapter) records next to an adapter's files.
pub const ADAPTER_INFO: &str = "adapter.json";

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

/// A LoRA adapter in the store, from [`ModelStore::adapters`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredAdapter {
    /// Its directory's name under `adapters/`.
    pub id: String,
    pub format: AdapterFormat,
    /// The GGUF file (llama.cpp `--lora`), or the PEFT directory (vLLM `--lora-modules`).
    pub path: PathBuf,
    /// All its files together.
    pub size: u64,
    /// The base model as a Hub id, when known.
    pub base_model: Option<String>,
    /// `general.architecture` of a GGUF adapter.
    pub architecture: Option<String>,
    /// The Hub repo it was downloaded from, when known.
    pub repo: Option<String>,
    pub rank: Option<u32>,
    pub alpha: Option<f32>,
}

/// `adapter.json`: a [`StoredAdapter`] without machine paths (the file name only).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdapterInfo {
    id: String,
    format: AdapterFormat,
    /// The GGUF's file name; `None` for PEFT (the directory).
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    base_model: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    rank: Option<u32>,
    #[serde(default)]
    alpha: Option<f32>,
}

pub(crate) fn write_adapter_info(dir: &Path, a: &StoredAdapter) -> crate::Result<()> {
    let file = match a.format {
        AdapterFormat::GgufLora => a.path.file_name().map(|f| f.to_string_lossy().into_owned()),
        _ => None,
    };
    let info = AdapterInfo {
        id: a.id.clone(),
        format: a.format,
        file,
        base_model: a.base_model.clone(),
        architecture: a.architecture.clone(),
        repo: a.repo.clone(),
        rank: a.rank,
        alpha: a.alpha,
    };
    let json = serde_json::to_vec_pretty(&info).map_err(|e| crate::Error::Model(e.to_string()))?;
    std::fs::write(dir.join(ADAPTER_INFO), json)?;
    Ok(())
}

/// A LoRA GGUF by its header.
fn is_adapter_gguf(path: &Path) -> bool {
    std::fs::File::open(path).ok().and_then(|f| crate::gguf::read_header(std::io::BufReader::new(f)).ok()).is_some_and(|h| h.kind.is_adapter())
}

/// Bytes of an adapter's files in `dir` (not `.part` files nor `adapter.json`).
fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()) && !e.file_name().to_string_lossy().ends_with(".part") && e.file_name() != ADAPTER_INFO)
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
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
    /// left out), sorted by id. Adapters are not models: `adapters/` is skipped, and so is a
    /// GGUF whose name hints at an adapter ([`adapter_hint`]) when its header says it is one.
    pub fn list(&self) -> Vec<StoredModel> {
        let mut files = Vec::new();
        walk(&self.dir, &self.dir, 0, &mut files);
        files.retain(|(rel, _)| !(adapter_hint(rel) && is_adapter_gguf(&self.dir.join(rel))));
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

impl ModelStore {
    /// Where adapter `id` goes: `<store>/adapters/<id>`. The id must be one plain path part.
    pub fn adapter_dir(&self, id: &str) -> crate::Result<PathBuf> {
        if id.contains('/') || crate::hf::check_file(id).is_err() {
            return Err(crate::Error::Model(format!("{id:?} is not an adapter id")));
        }
        Ok(self.dir.join(ADAPTERS_DIR).join(id))
    }

    /// The complete LoRA adapters under `adapters/`, sorted by id: from `adapter.json` when
    /// [`download_adapter`](crate::download::download_adapter) wrote one, else a LoRA GGUF (by
    /// its header) or a PEFT directory (`adapter_config.json` and its weights) found there.
    pub fn adapters(&self) -> Vec<StoredAdapter> {
        let mut out: Vec<StoredAdapter> = std::fs::read_dir(self.dir.join(ADAPTERS_DIR))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| Self::adapter_in(&e.path(), &e.file_name().to_string_lossy()))
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    fn adapter_in(dir: &Path, id: &str) -> Option<StoredAdapter> {
        let info: Option<AdapterInfo> = std::fs::read(dir.join(ADAPTER_INFO)).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let peft_weights = ["adapter_model.safetensors", "adapter_model.bin"].iter().any(|w| dir.join(w).is_file());
        let peft_config = || -> Option<serde_json::Value> { serde_json::from_slice(&std::fs::read(dir.join("adapter_config.json")).ok()?).ok() };
        let gguf = || {
            std::fs::read_dir(dir)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && p.is_file() && is_adapter_gguf(p))
        };
        let mut a = StoredAdapter {
            id: id.to_string(),
            format: AdapterFormat::Peft,
            path: dir.to_path_buf(),
            size: dir_size(dir),
            base_model: None,
            architecture: None,
            repo: None,
            rank: None,
            alpha: None,
        };
        match info {
            Some(i) => {
                a.format = i.format;
                (a.base_model, a.architecture, a.repo, a.rank, a.alpha) = (i.base_model, i.architecture, i.repo, i.rank, i.alpha);
                match i.format {
                    AdapterFormat::GgufLora => {
                        let file = dir.join(i.file.filter(|f| !f.contains('/') && !f.contains('\\'))?);
                        if !file.is_file() {
                            return None;
                        }
                        a.path = file;
                    }
                    _ if !(peft_weights && dir.join("adapter_config.json").is_file()) => return None,
                    _ => {}
                }
            }
            None if peft_weights => {
                let (rank, alpha, base) = crate::hf::peft_lora(&peft_config()?)?;
                (a.rank, a.alpha, a.base_model) = (rank, alpha, base);
            }
            None => {
                let file = gguf()?;
                let h = crate::gguf::read_header(std::io::BufReader::new(std::fs::File::open(&file).ok()?)).ok()?;
                a.format = AdapterFormat::GgufLora;
                a.alpha = match h.kind {
                    crate::gguf::GgufKind::Adapter { alpha, .. } => alpha,
                    _ => None,
                };
                (a.architecture, a.base_model, a.path) = (Some(h.architecture), h.base_model, file);
            }
        }
        Some(a)
    }

    /// Deletes a stored adapter's directory. Returns the bytes freed. Ids that are not one plain
    /// path part are refused.
    pub fn remove_adapter(&self, adapter: &StoredAdapter) -> crate::Result<u64> {
        let dir = self.adapter_dir(&adapter.id)?;
        if !dir.exists() {
            return Ok(0);
        }
        let freed = dir_size(&dir);
        std::fs::remove_dir_all(&dir)?;
        let _ = std::fs::remove_dir(self.dir.join(ADAPTERS_DIR));
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
        if ft.is_dir() && depth == 0 && e.file_name() == ADAPTERS_DIR {
            continue;
        } else if ft.is_dir() && depth < 4 {
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
    fn adapters_are_found_without_adapter_json_and_never_listed_as_models() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let w = |rel: &str, b: &[u8]| {
            let p = tmp.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b).unwrap();
        };
        w("m-Q4_0.gguf", b"GGUF....");
        w("stray-LoRA-F16.gguf", &crate::gguf::lora_header("qwen3"));
        w("lora-merged-Q4_0.gguf", b"not an adapter header");
        w("adapters/hand/hand.gguf", &crate::gguf::lora_header("llama"));
        let config = br#"{"peft_type":"LORA","r":4,"lora_alpha":8,"base_model_name_or_path":"Org/B"}"#;
        w("adapters/peft/adapter_config.json", config);
        w("adapters/peft/adapter_model.safetensors", &[1; 10]);
        w("adapters/half/adapter_config.json", br#"{"peft_type":"LORA"}"#);
        w("adapters/ia3/adapter_config.json", br#"{"peft_type":"IA3"}"#);
        w("adapters/ia3/adapter_model.safetensors", &[1; 10]);
        assert_eq!(store.list().iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["lora-merged-Q4_0", "m-Q4_0"], "the header decides");
        let a = store.adapters();
        assert_eq!(
            a.iter().map(|a| (a.id.as_str(), a.format)).collect::<Vec<_>>(),
            [("hand", AdapterFormat::GgufLora), ("peft", AdapterFormat::Peft)]
        );
        assert_eq!((a[0].architecture.as_deref(), a[0].path.clone()), (Some("llama"), tmp.path().join("adapters/hand/hand.gguf")));
        assert_eq!((a[1].rank, a[1].alpha, a[1].base_model.as_deref(), a[1].size), (Some(4), Some(8.0), Some("Org/B"), 10 + config.len() as u64));
        assert!(store.adapter_dir("a/b").is_err() && store.adapter_dir("..").is_err());
        let mut evil = a[0].clone();
        evil.id = "../m-Q4_0.gguf".into();
        assert!(store.remove_adapter(&evil).is_err());
        assert!(tmp.path().join("m-Q4_0.gguf").exists());
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
