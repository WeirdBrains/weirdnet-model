//! Where model files live on disk, and which Hub they come from. LoRA adapters live apart, in
//! `adapters/<id>/` ([`ModelStore::adapters`]), and are never listed as models.
//!
//! Files a person brings ([`ModelStore::import`]) land in their own directory with a
//! [`Provenance`] record next to them: a GGUF model in `<id>/`, an adapter in `adapters/<id>/`,
//! a projector in `projectors/<id>/`, a safetensors model in `safetensors/<id>/` and an MLX
//! model in `mlx/<id>/` ([`ModelStore::dir_models`]).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::artifact::{Artifact, ArtifactFormat, ArtifactKind, PROVENANCE_FILE, Provenance, ProvenanceKind, inspect_staged};
use crate::catalog::ModelSpec;
use crate::download::Progress;
use crate::hf::{AdapterFormat, Hub, adapter_hint, split_part};
use crate::limits::{LIMITS_FILE, LIMITS_SUFFIX, LimitsRecord};
use crate::token::HfToken;

/// The store's subdirectory for adapters: one directory per adapter id.
pub const ADAPTERS_DIR: &str = "adapters";
/// What [`download_adapter`](crate::download::download_adapter) records next to an adapter's files.
pub const ADAPTER_INFO: &str = "adapter.json";
/// The store's subdirectory for multimodal projectors: one directory per projector id.
pub const PROJECTORS_DIR: &str = "projectors";
/// The store's subdirectory for safetensors model directories (vLLM), one per model id.
pub const SAFETENSORS_DIR: &str = "safetensors";
/// The store's subdirectory for MLX model directories (MLX-LM), one per model id.
pub const MLX_DIR: &str = "mlx";
/// Directories [`ModelStore::import`] works in while it places files; never listed.
const IMPORTING: &str = ".import-";
const REPLACED: &str = ".replaced-";

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
    /// How its files are laid out: GGUF files (from [`ModelStore::list`]), or a safetensors or
    /// MLX directory (from [`ModelStore::dir_models`]).
    #[serde(default)]
    pub layout: ModelLayout,
    /// Where it came from, for models placed by [`ModelStore::import`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// The limits a node recorded when it loaded the model ([`ModelStore::record_limits`]), when
    /// there is a current record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<LimitsRecord>,
}

/// [`StoredModel::layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelLayout {
    /// GGUF files, for llama.cpp.
    #[default]
    Gguf,
    /// A safetensors directory (`safetensors/<id>/`), for vLLM.
    Safetensors,
    /// An MLX directory (`mlx/<id>/`), for MLX-LM.
    Mlx,
}

/// A multimodal projector in the store, from [`ModelStore::projectors`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredProjector {
    /// Its directory's name under `projectors/`.
    pub id: String,
    /// The GGUF (llama.cpp `--mmproj`).
    pub path: PathBuf,
    pub size: u64,
    /// The model it is for (its provenance's base), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub for_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// What [`ModelStore::import`] placed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Imported {
    Model(StoredModel),
    Adapter(StoredAdapter),
    Projector(StoredProjector),
}

impl Imported {
    pub fn id(&self) -> &str {
        match self {
            Imported::Model(m) => &m.id,
            Imported::Adapter(a) => &a.id,
            Imported::Projector(p) => &p.id,
        }
    }

    /// Bytes of its files.
    pub fn size(&self) -> u64 {
        match self {
            Imported::Model(m) => m.size,
            Imported::Adapter(a) => a.size,
            Imported::Projector(p) => p.size,
        }
    }

    pub fn provenance(&self) -> Option<&Provenance> {
        match self {
            Imported::Model(m) => m.provenance.as_ref(),
            Imported::Adapter(a) => a.provenance.as_ref(),
            Imported::Projector(p) => p.provenance.as_ref(),
        }
    }
}

/// Free and total bytes on the store's filesystem ([`ModelStore::disk_space`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiskSpace {
    /// Bytes this process may still write.
    pub free: u64,
    pub total: u64,
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
    /// Where it came from, for adapters placed by [`ModelStore::import`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
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

/// Bytes of an entry's files in `dir`, subdirectories too (not `.part` or hidden files, nor
/// `adapter.json` or `provenance.json`).
fn dir_size(dir: &Path) -> u64 {
    dir_files(dir, dir).iter().map(|(_, size)| size).sum()
}

/// An entry's files under `dir` (as in [`dir_size`]), as `/`-separated paths relative to `root`,
/// sorted.
fn dir_files(root: &Path, dir: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    collect_files(root, dir, 0, &mut out);
    out.sort();
    out
}

fn collect_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, u64)>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Ok(ft) = e.file_type() else { continue };
        if name.starts_with('.') || name.ends_with(".part") || name == ADAPTER_INFO || name == PROVENANCE_FILE {
            continue;
        }
        if ft.is_dir() && depth < 3 {
            collect_files(root, &e.path(), depth + 1, out);
        } else if ft.is_file() {
            let path = e.path();
            let Ok(rel) = path.strip_prefix(root) else { continue };
            let rel: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            out.push((rel.join("/"), e.metadata().map(|m| m.len()).unwrap_or(0)));
        }
    }
}

/// A directory [`ModelStore::import`] is working in.
fn is_work_dir(name: &str) -> bool {
    name.starts_with(IMPORTING) || name.starts_with(REPLACED)
}

/// Moves `from` to `to`: a rename, or a copy then delete across filesystems.
fn move_file(from: &Path, to: &Path) -> std::io::Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to)?;
    std::fs::File::open(to)?.sync_all()?;
    std::fs::remove_file(from)
}

/// A name for a work directory next to the entry: unique within this process and time.
fn work_name(prefix: &str, id: &str) -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}{id}-{}-{nanos}", std::process::id())
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
                let mut id = key.rsplit('/').next().unwrap_or(&key).to_string();
                let provenance = self.provenance_of(&parts[0].1);
                if let Some(p) = &provenance {
                    id = p.id.clone();
                }
                let mut m = StoredModel {
                    id,
                    size: parts.iter().map(|p| p.2).sum(),
                    files: parts.into_iter().map(|p| p.1).collect(),
                    complete,
                    layout: ModelLayout::Gguf,
                    provenance,
                    limits: None,
                };
                m.limits = self.limits(&m);
                m
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.files.cmp(&b.files)));
        models
    }

    /// The provenance record next to stored GGUF file `rel` (in its own directory, not the store
    /// root) when the record lists that file.
    fn provenance_of(&self, rel: &str) -> Option<Provenance> {
        let (dir, file) = rel.rsplit_once('/')?;
        let p = Provenance::read(&self.dir.join(dir))?;
        (p.kind == ProvenanceKind::Model && !p.id.is_empty() && p.files.iter().any(|f| f.path == file)).then_some(p)
    }

    /// Deletes a listed model's files (and their `.part` files) and the directories that leaves
    /// empty (with a provenance record left alone in one). Returns the bytes freed. Paths
    /// outside the store are refused.
    pub fn remove_local(&self, model: &StoredModel) -> crate::Result<u64> {
        let mut freed = 0;
        for rel in &model.files {
            crate::hf::check_file(rel)?;
        }
        self.clear_limits(model);
        for rel in &model.files {
            let path = self.dir.join(rel);
            let _ = std::fs::remove_file(self.dir.join(format!("{rel}.part")));
            if let Ok(m) = std::fs::metadata(&path) {
                std::fs::remove_file(&path)?;
                freed += m.len();
            }
        }
        for rel in &model.files {
            let path = self.dir.join(rel);
            if let Some(d) = path.parent().filter(|d| *d != self.dir)
                && std::fs::read_dir(d).is_ok_and(|mut e| e.all(|e| e.is_ok_and(|e| e.file_name() == PROVENANCE_FILE)))
            {
                let _ = std::fs::remove_file(d.join(PROVENANCE_FILE));
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
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()) && !e.file_name().to_string_lossy().starts_with('.'))
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
            provenance: Provenance::read(dir).filter(|p| p.kind == ProvenanceKind::Adapter),
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

impl ModelStore {
    /// Where `model`'s limits record goes: `<first file>.limits.json` next to a GGUF model's
    /// first file, `.limits.json` inside a safetensors or MLX directory.
    pub fn limits_path(&self, model: &StoredModel) -> crate::Result<PathBuf> {
        match model.layout {
            ModelLayout::Gguf => {
                let first = model.files.first().ok_or_else(|| crate::Error::Model(format!("{} has no files", model.id)))?;
                crate::hf::check_file(first)?;
                Ok(self.dir.join(format!("{first}{LIMITS_SUFFIX}")))
            }
            layout => Ok(self.model_dir(layout, &model.id)?.join(LIMITS_FILE)),
        }
    }

    /// Keeps `record` next to `model`'s files, replacing any earlier one atomically, with the
    /// model's current size so a later change to its files makes the record stale. A node calls
    /// this when it loads the model (see [`limits::measure`](crate::limits::measure)).
    pub fn record_limits(&self, model: &StoredModel, record: &LimitsRecord) -> crate::Result<()> {
        let path = self.limits_path(model)?;
        let dir = path.parent().filter(|d| d.is_dir()).ok_or_else(|| crate::Error::Model(format!("{} is not in the store", model.id)))?;
        let mut record = record.clone();
        record.model_bytes = Some(model.size);
        let json = serde_json::to_vec_pretty(&record).map_err(|e| crate::Error::Model(e.to_string()))?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let tmp = dir.join(format!(".{}.tmp", name.trim_start_matches('.')));
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// `model`'s limits record, if there is a current one: `None` when there is none, it is not
    /// a regular file or doesn't parse, its format is another, or the model's files changed size
    /// since it was recorded.
    pub fn limits(&self, model: &StoredModel) -> Option<LimitsRecord> {
        let path = self.limits_path(model).ok()?;
        if !std::fs::symlink_metadata(&path).ok()?.is_file() {
            return None;
        }
        let r: LimitsRecord = serde_json::from_slice(&std::fs::read(&path).ok()?).ok()?;
        (r.format == crate::limits::LIMITS_FORMAT && r.model_bytes.is_none_or(|b| b == model.size)).then_some(r)
    }

    /// Deletes `model`'s limits record. Returns whether there was one.
    pub fn clear_limits(&self, model: &StoredModel) -> bool {
        self.limits_path(model).is_ok_and(|p| std::fs::remove_file(p).is_ok())
    }
}

impl ModelStore {
    /// The safetensors (`safetensors/<id>/`) and MLX (`mlx/<id>/`) model directories, sorted by
    /// id. [`ModelStore::list`] has the GGUF models. A directory is complete when it has its
    /// `config.json`.
    pub fn dir_models(&self) -> Vec<StoredModel> {
        let mut out = Vec::new();
        for (sub, layout) in [(SAFETENSORS_DIR, ModelLayout::Safetensors), (MLX_DIR, ModelLayout::Mlx)] {
            for e in std::fs::read_dir(self.dir.join(sub)).into_iter().flatten().flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with('.') || !e.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                let files = dir_files(&self.dir, &e.path());
                if files.is_empty() {
                    continue;
                }
                let mut m = StoredModel {
                    id: name,
                    size: files.iter().map(|f| f.1).sum(),
                    complete: e.path().join("config.json").is_file(),
                    files: files.into_iter().map(|f| f.0).collect(),
                    layout,
                    provenance: Provenance::read(&e.path()).filter(|p| p.kind == ProvenanceKind::Model),
                    limits: None,
                };
                m.limits = self.limits(&m);
                out.push(m);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| a.files.cmp(&b.files)));
        out
    }

    /// Where an imported model of `layout` with id `id` goes: `<id>/` for GGUF,
    /// `safetensors/<id>/` or `mlx/<id>/`. The id must be one plain path part.
    pub fn model_dir(&self, layout: ModelLayout, id: &str) -> crate::Result<PathBuf> {
        check_id(id)?;
        Ok(match layout {
            ModelLayout::Gguf => self.dir.join(id),
            ModelLayout::Safetensors => self.dir.join(SAFETENSORS_DIR).join(id),
            ModelLayout::Mlx => self.dir.join(MLX_DIR).join(id),
        })
    }

    /// Where projector `id` goes: `<store>/projectors/<id>`.
    pub fn projector_dir(&self, id: &str) -> crate::Result<PathBuf> {
        check_id(id)?;
        Ok(self.dir.join(PROJECTORS_DIR).join(id))
    }

    /// The projectors under `projectors/`, sorted by id: each directory's GGUF.
    pub fn projectors(&self) -> Vec<StoredProjector> {
        let mut out: Vec<StoredProjector> = std::fs::read_dir(self.dir.join(PROJECTORS_DIR))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()) && !e.file_name().to_string_lossy().starts_with('.'))
            .filter_map(|e| {
                let dir = e.path();
                let path = std::fs::read_dir(&dir)
                    .ok()?
                    .flatten()
                    .map(|f| f.path())
                    .find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && p.is_file())?;
                let provenance = Provenance::read(&dir).filter(|p| p.kind == ProvenanceKind::Projector);
                Some(StoredProjector {
                    id: e.file_name().to_string_lossy().into_owned(),
                    size: dir_size(&dir),
                    for_model: provenance.as_ref().and_then(|p| p.base.as_ref()).and_then(|b| b.id.clone()),
                    path,
                    provenance,
                })
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Deletes a stored projector's directory. Returns the bytes freed.
    pub fn remove_projector(&self, projector: &StoredProjector) -> crate::Result<u64> {
        let dir = self.projector_dir(&projector.id)?;
        if !dir.exists() {
            return Ok(0);
        }
        let freed = dir_size(&dir);
        std::fs::remove_dir_all(&dir)?;
        let _ = std::fs::remove_dir(self.dir.join(PROJECTORS_DIR));
        Ok(freed)
    }

    /// The store's directory, or its nearest existing parent.
    fn existing_dir(&self) -> PathBuf {
        let mut d = self.dir.clone();
        while !d.exists() {
            match d.parent() {
                Some(p) => d = p.to_path_buf(),
                None => break,
            }
        }
        d
    }

    /// Free and total bytes on the filesystem the store is on (or would be on, before it
    /// exists).
    pub fn disk_space(&self) -> crate::Result<DiskSpace> {
        let d = self.existing_dir();
        Ok(DiskSpace { free: fs4::available_space(&d)?, total: fs4::total_space(&d)? })
    }

    /// Bytes this process may still write on the store's filesystem (0 when that can't be read,
    /// so callers checking for room refuse rather than overfill).
    pub fn free_bytes(&self) -> u64 {
        self.disk_space().map(|d| d.free).unwrap_or(0)
    }

    /// Bytes of everything stored (models, model directories, adapters and projectors) whose
    /// provenance has this origin ([`ORIGIN_UPLOAD`](crate::artifact::ORIGIN_UPLOAD) and so on).
    pub fn bytes_by_origin(&self, origin: &str) -> u64 {
        let of = |p: Option<&Provenance>| p.is_some_and(|p| p.origin == origin);
        let models: u64 = self.list().iter().chain(self.dir_models().iter()).filter(|m| of(m.provenance.as_ref())).map(|m| m.size).sum();
        let adapters: u64 = self.adapters().iter().filter(|a| of(a.provenance.as_ref())).map(|a| a.size).sum();
        let projectors: u64 = self.projectors().iter().filter(|p| of(p.provenance.as_ref())).map(|p| p.size).sum();
        models + adapters + projectors
    }

    /// Places files received into `staged` (under their artifact paths, as an upload stages them;
    /// keep `staged` outside the store) into the store as `id`, with `provenance` next to them.
    ///
    /// The staged files are checked first: each must be in `artifact.files` with the same size
    /// and SHA-256, and their headers must say what `artifact` says. The record's id, kind, files
    /// and revision are then set from the bytes. Files are moved (copied across filesystems)
    /// into a work directory next to the target, which is renamed into place last, so the entry
    /// appears whole or not at all. An adapter also gets its `adapter.json`.
    ///
    /// Where it goes: a GGUF model in `<id>/`, an adapter in `adapters/<id>/`, a projector in
    /// `projectors/<id>/`, a safetensors model in `safetensors/<id>/`, an MLX model in
    /// `mlx/<id>/`. An id already in use is refused; see [`ModelStore::import_replacing`].
    pub fn import(&self, staged: &Path, artifact: &Artifact, provenance: &Provenance, id: &str) -> crate::Result<Imported> {
        self.import_with(staged, artifact, provenance, id, false, &mut |_| true)
    }

    /// [`ModelStore::import`], replacing an entry this store imported before under the same id
    /// (models in the store's own GGUF files are never replaced).
    pub fn import_replacing(&self, staged: &Path, artifact: &Artifact, provenance: &Provenance, id: &str) -> crate::Result<Imported> {
        self.import_with(staged, artifact, provenance, id, true, &mut |_| true)
    }

    /// [`ModelStore::import`] with `replace` and a `progress` callback for the hash check
    /// (`false` cancels).
    pub fn import_with(
        &self,
        staged: &Path,
        artifact: &Artifact,
        provenance: &Provenance,
        id: &str,
        replace: bool,
        progress: &mut dyn FnMut(Progress) -> bool,
    ) -> crate::Result<Imported> {
        check_id(id)?;
        let dest = self.dest(artifact, id)?;
        self.check_free(artifact, id, &dest, replace)?;
        let found = inspect_staged(staged, &artifact.files, progress)?;
        if (found.kind, found.format, &found.revision) != (artifact.kind, artifact.format, &artifact.revision) {
            return Err(crate::Error::Model(format!("{id}: the staged files are not what was described")));
        }
        self.place(staged, &found, provenance, id, replace)
    }

    /// [`ModelStore::import_with`] for files known only by their list (an upload's offer): what
    /// they are comes from their headers, read once with the hash check. The entry's kind and
    /// format are in the returned [`Imported`].
    pub fn import_files(
        &self,
        staged: &Path,
        files: &[crate::artifact::ArtifactFile],
        provenance: &Provenance,
        id: &str,
        replace: bool,
        progress: &mut dyn FnMut(Progress) -> bool,
    ) -> crate::Result<Imported> {
        check_id(id)?;
        let found = inspect_staged(staged, files, progress)?;
        self.place(staged, &found, provenance, id, replace)
    }

    /// Moves checked staged files into place (see [`ModelStore::import`]).
    fn place(&self, staged: &Path, found: &Artifact, provenance: &Provenance, id: &str, replace: bool) -> crate::Result<Imported> {
        let dest = self.dest(found, id)?;
        self.check_free(found, id, &dest, replace)?;
        let mut record = provenance.clone();
        record.id = id.to_string();
        record.kind = found.kind.provenance_kind();
        record.files = found.files.clone();
        record.revision = found.revision.clone();

        let parent = dest.parent().ok_or_else(|| crate::Error::Model(format!("{id}: no parent directory")))?;
        std::fs::create_dir_all(parent)?;
        let work = parent.join(work_name(IMPORTING, id));
        let placed = (|| -> crate::Result<()> {
            std::fs::create_dir(&work)?;
            for f in &found.files {
                let to = work.join(&f.path);
                if let Some(d) = to.parent() {
                    std::fs::create_dir_all(d)?;
                }
                move_file(&staged.join(&f.path), &to)?;
            }
            record.write(&work)?;
            if let ArtifactKind::Adapter { format } = found.kind {
                let mut a = self.adapter_record(found, format, id, &work);
                a.path = match format {
                    AdapterFormat::GgufLora => dest.join(&found.files[0].path),
                    _ => dest.clone(),
                };
                write_adapter_info(&work, &a)?;
            }
            Ok(())
        })();
        if let Err(e) = placed {
            // Give the staged files back so the upload can be retried without sending them again.
            for f in &found.files {
                let at = work.join(&f.path);
                if at.exists() {
                    let _ = move_file(&at, &staged.join(&f.path));
                }
            }
            let _ = std::fs::remove_dir_all(&work);
            return Err(e);
        }
        let old = parent.join(work_name(REPLACED, id));
        let had_old = replace && dest.exists();
        if had_old {
            std::fs::rename(&dest, &old)?;
        }
        if let Err(e) = std::fs::rename(&work, &dest) {
            if had_old {
                let _ = std::fs::rename(&old, &dest);
            }
            let _ = std::fs::remove_dir_all(&work);
            return Err(e.into());
        }
        if had_old {
            let _ = std::fs::remove_dir_all(&old);
        }
        Ok(self.imported(found, id, &dest, record))
    }

    /// Where an artifact stored as `id` goes.
    fn dest(&self, artifact: &Artifact, id: &str) -> crate::Result<PathBuf> {
        match (artifact.kind, artifact.format) {
            (ArtifactKind::Adapter { .. }, _) => self.adapter_dir(id),
            (ArtifactKind::Projector, _) => self.projector_dir(id),
            (_, ArtifactFormat::Safetensors) => self.model_dir(ModelLayout::Safetensors, id),
            (_, ArtifactFormat::Mlx) => self.model_dir(ModelLayout::Mlx, id),
            _ => self.model_dir(ModelLayout::Gguf, id),
        }
    }

    /// Refuses an id in use (unless `replace` and the entry at `dest` was imported) and a
    /// target that a model's own GGUF files are in.
    fn check_free(&self, artifact: &Artifact, id: &str, dest: &Path, replace: bool) -> crate::Result<()> {
        let taken = |what: &str| crate::Error::Model(format!("{id} exists ({what}); remove it first or replace it"));
        let imported_here = Provenance::read(dest).is_some();
        if dest.exists() && !(replace && imported_here) {
            return Err(taken("its directory is in use"));
        }
        let model = artifact.kind == ArtifactKind::Model;
        if model
            && self.list().iter().chain(self.dir_models().iter()).any(|m| m.id == id && !m.files.iter().all(|f| self.dir.join(f).starts_with(dest)))
        {
            return Err(taken("a stored model has that id"));
        }
        Ok(())
    }

    fn adapter_record(&self, found: &Artifact, format: AdapterFormat, id: &str, dir: &Path) -> StoredAdapter {
        StoredAdapter {
            id: id.to_string(),
            format,
            path: dir.to_path_buf(),
            size: found.size(),
            base_model: found.base_model.clone(),
            architecture: found.architecture.clone(),
            repo: None,
            rank: found.rank,
            alpha: found.alpha,
            provenance: None,
        }
    }

    fn imported(&self, found: &Artifact, id: &str, dest: &Path, record: Provenance) -> Imported {
        let rel = |f: &str| {
            let full = dest.join(f);
            let rel: Vec<String> =
                full.strip_prefix(&self.dir).unwrap_or(&full).components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            rel.join("/")
        };
        match found.kind {
            ArtifactKind::Adapter { format } => {
                let mut a = self.adapter_record(found, format, id, dest);
                if format == AdapterFormat::GgufLora {
                    a.path = dest.join(&found.files[0].path);
                }
                a.provenance = Some(record);
                Imported::Adapter(a)
            }
            ArtifactKind::Projector => Imported::Projector(StoredProjector {
                id: id.to_string(),
                path: dest.join(&found.files[0].path),
                size: found.size(),
                for_model: record.base.as_ref().and_then(|b| b.id.clone()),
                provenance: Some(record),
            }),
            _ => Imported::Model(StoredModel {
                id: id.to_string(),
                files: found.files.iter().map(|f| rel(&f.path)).collect(),
                size: found.size(),
                complete: true,
                layout: match found.format {
                    ArtifactFormat::Safetensors => ModelLayout::Safetensors,
                    ArtifactFormat::Mlx => ModelLayout::Mlx,
                    _ => ModelLayout::Gguf,
                },
                provenance: Some(record),
                limits: None,
            }),
        }
    }
}

/// An entry id: one plain path part, not hidden.
fn check_id(id: &str) -> crate::Result<()> {
    if id.contains('/') || id.starts_with('.') || crate::hf::check_file(id).is_err() {
        return Err(crate::Error::Model(format!("{id:?} is not a valid id (one plain name, no '/', not hidden)")));
    }
    Ok(())
}

/// A file of a stored model: part number, path relative to the store, size.
type Part = (u32, String, u64);

/// Model GGUFs under `dir`, as `/`-separated paths relative to `root` (four levels deep at most).
fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let Ok(ft) = e.file_type() else { continue };
        let path = e.path();
        let skip = [ADAPTERS_DIR, PROJECTORS_DIR, SAFETENSORS_DIR, MLX_DIR];
        if ft.is_dir() && ((depth == 0 && skip.iter().any(|d| e.file_name() == *d)) || is_work_dir(&e.file_name().to_string_lossy())) {
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
    fn limits_records_sit_next_to_gguf_models_and_go_with_them() {
        use crate::limits::{LimitSource, ModelLimits};
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let w = |rel: &str, n: usize| {
            let p = tmp.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        w("a-Q4_0.gguf", 3);
        w("Q8_0/b-Q8_0-00001-of-00002.gguf", 5);
        w("Q8_0/b-Q8_0-00002-of-00002.gguf", 6);
        let list = store.list();
        assert!(list.iter().all(|m| m.limits.is_none()), "no record yet");
        let record = LimitsRecord::new(ModelLimits::new().with_context_tokens(8192).with_audio_chunks(30, 187), LimitSource::Measured)
            .with_engine("llama.cpp");
        store.record_limits(&list[0], &record).unwrap();
        store.record_limits(&list[1], &record.clone()).unwrap();
        assert!(tmp.path().join("a-Q4_0.gguf.limits.json").is_file());
        assert!(tmp.path().join("Q8_0/b-Q8_0-00001-of-00002.gguf.limits.json").is_file());
        let again = store.list();
        assert_eq!(again.len(), 2, "records are not models");
        let a = again[0].limits.clone().unwrap();
        assert_eq!((a.limits, a.model_bytes, a.engine.as_deref()), (record.limits, Some(3), Some("llama.cpp")));
        assert_eq!(store.limits(&again[1]).unwrap().model_bytes, Some(11));
        // A record whose model changed size is stale.
        w("a-Q4_0.gguf", 4);
        assert!(store.list()[0].limits.is_none());
        // Another format, junk or a directory in its place reads as none.
        std::fs::write(tmp.path().join("a-Q4_0.gguf.limits.json"), br#"{"format":"other/1"}"#).unwrap();
        assert!(store.limits(&store.list()[0]).is_none());
        std::fs::write(tmp.path().join("a-Q4_0.gguf.limits.json"), b"{").unwrap();
        assert!(store.limits(&store.list()[0]).is_none());
        // Removing a model removes its record, and the emptied directory still goes.
        assert_eq!(store.remove_local(&again[1]).unwrap(), 11);
        assert!(!tmp.path().join("Q8_0").exists());
        assert!(store.clear_limits(&store.list()[0]));
        assert!(!store.clear_limits(&store.list()[0]));
        // Paths that leave the store are refused.
        let mut evil = again[0].clone();
        evil.files = vec!["../x.gguf".into()];
        assert!(store.limits_path(&evil).is_err() && store.record_limits(&evil, &record).is_err());
        let mut none = again[0].clone();
        none.files.clear();
        assert!(store.limits_path(&none).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_limits_record_is_not_read() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        std::fs::write(tmp.path().join("a.gguf"), b"GGUF").unwrap();
        let elsewhere = tmp.path().join("elsewhere.json");
        std::fs::write(&elsewhere, serde_json::to_vec(&LimitsRecord::default()).unwrap()).unwrap();
        std::os::unix::fs::symlink(&elsewhere, tmp.path().join("a.gguf.limits.json")).unwrap();
        assert!(store.list()[0].limits.is_none());
    }

    #[test]
    fn stored_models_from_before_limits_still_read() {
        let old = serde_json::json!({"id": "a", "files": ["a.gguf"], "size": 3, "complete": true});
        let m: StoredModel = serde_json::from_value(old).unwrap();
        assert!(m.limits.is_none() && m.provenance.is_none());
        assert_eq!(m.layout, ModelLayout::Gguf);
        let j = serde_json::to_value(&m).unwrap();
        assert!(j.get("limits").is_none(), "absent limits stay off the wire");
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

    mod import {
        use super::*;
        use crate::artifact::tests::{model_gguf, safetensors};
        use crate::artifact::{ORIGIN_TRAIN, ORIGIN_UPLOAD, ProvenanceBase, inspect};

        fn stage(dir: &Path, files: &[(&str, Vec<u8>)]) -> Artifact {
            for (name, body) in files {
                let p = dir.join(name);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(p, body).unwrap();
            }
            let paths: Vec<PathBuf> =
                if files.len() == 1 && !files[0].0.contains('/') { vec![dir.join(files[0].0)] } else { vec![dir.to_path_buf()] };
            inspect(&paths).unwrap()
        }

        #[test]
        fn a_gguf_model_lands_in_its_own_directory_with_provenance() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            let staged = tmp.path().join("up");
            let a = stage(&staged, &[("tune-Q4_0.gguf", model_gguf(&[]))]);
            let p = Provenance::new(ORIGIN_UPLOAD).with_license("apache-2.0");
            let got = store.import(&staged, &a, &p, "support-r3").unwrap();
            let Imported::Model(m) = &got else { panic!("{got:?}") };
            assert_eq!(
                (m.id.as_str(), m.files.as_slice(), m.layout),
                ("support-r3", &["support-r3/tune-Q4_0.gguf".to_string()][..], ModelLayout::Gguf)
            );
            assert!(!staged.join("tune-Q4_0.gguf").exists(), "moved, not copied");
            let listed = store.list();
            assert_eq!(listed.len(), 1);
            assert_eq!(&listed[0], m, "list reads the same entry back");
            let rec = listed[0].provenance.as_ref().unwrap();
            assert_eq!((rec.revision.as_str(), rec.license.as_deref(), rec.kind), (a.revision.as_str(), Some("apache-2.0"), ProvenanceKind::Model));
            assert_eq!(store.bytes_by_origin(ORIGIN_UPLOAD), a.size());
            assert_eq!(store.bytes_by_origin(ORIGIN_TRAIN), 0);

            let again = tmp.path().join("again");
            let b = stage(&again, &[("tune-Q4_0.gguf", model_gguf(&[("general.name", serde_json::json!("v2"))]))]);
            assert!(store.import(&again, &b, &p, "support-r3").unwrap_err().to_string().contains("exists"));
            assert!(again.join("tune-Q4_0.gguf").exists(), "a refusal keeps the staged files");
            let replaced = store.import_replacing(&again, &b, &p, "support-r3").unwrap();
            assert_eq!(replaced.provenance().unwrap().revision, b.revision);
            assert_eq!(store.list().len(), 1);

            assert_eq!(store.remove_local(&store.list()[0]).unwrap(), b.size());
            assert!(!tmp.path().join("store/support-r3").exists(), "the provenance record goes with the files");
        }

        #[test]
        fn an_id_used_by_a_stored_model_or_bad_ids_are_refused() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            std::fs::create_dir_all(store.dir()).unwrap();
            std::fs::write(store.dir().join("m-Q4_0.gguf"), model_gguf(&[])).unwrap();
            let staged = tmp.path().join("up");
            let a = stage(&staged, &[("x.gguf", model_gguf(&[]))]);
            let p = Provenance::new(ORIGIN_UPLOAD);
            assert!(store.import_replacing(&staged, &a, &p, "m-Q4_0").unwrap_err().to_string().contains("exists"));
            for bad in ["a/b", "..", ".hidden", ""] {
                assert!(store.import(&staged, &a, &p, bad).is_err(), "{bad}");
            }
        }

        #[test]
        fn files_known_by_their_list_are_classified_from_their_headers() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            let staged = tmp.path().join("up");
            let a = stage(&staged, &[("tune-LoRA.gguf", crate::gguf::lora_header("llama"))]);
            let got = store.import_files(&staged, &a.files, &Provenance::new(ORIGIN_UPLOAD), "tune", false, &mut |_| true).unwrap();
            assert!(matches!(&got, Imported::Adapter(ad) if ad.format == AdapterFormat::GgufLora));
            assert_eq!(got.provenance().unwrap().kind, ProvenanceKind::Adapter);
        }

        #[test]
        fn tampered_staged_files_are_refused() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            let staged = tmp.path().join("up");
            let a = stage(&staged, &[("x.gguf", model_gguf(&[]))]);
            let mut body = std::fs::read(staged.join("x.gguf")).unwrap();
            *body.last_mut().unwrap() ^= 1;
            std::fs::write(staged.join("x.gguf"), body).unwrap();
            assert!(matches!(store.import(&staged, &a, &Provenance::new(ORIGIN_UPLOAD), "x"), Err(crate::Error::Checksum { .. })));
            assert!(store.list().is_empty());
        }

        #[test]
        fn adapters_get_adapter_json_and_their_size_leaves_provenance_out() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            let staged = tmp.path().join("up");
            let a = stage(&staged, &[("tune-LoRA.gguf", crate::gguf::lora_header("llama"))]);
            let mut p = Provenance::new(ORIGIN_TRAIN);
            p.base = Some(ProvenanceBase { id: Some("qwen3-8b".into()), ..Default::default() });
            let got = store.import(&staged, &a, &p, "tune").unwrap();
            let Imported::Adapter(ad) = got else { panic!() };
            assert_eq!((ad.format, ad.architecture.as_deref(), ad.alpha), (AdapterFormat::GgufLora, Some("llama"), Some(32.5)));
            let listed = store.adapters();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].path, store.dir().join("adapters/tune/tune-LoRA.gguf"));
            assert_eq!(listed[0].size, a.size(), "adapter.json and provenance.json are not counted");
            assert_eq!(listed[0].provenance.as_ref().unwrap().base.as_ref().unwrap().id.as_deref(), Some("qwen3-8b"));
            assert!(store.dir().join("adapters/tune/adapter.json").is_file());
            assert!(store.list().is_empty(), "adapters are not models");

            let peft = tmp.path().join("peft");
            let b = stage(
                &peft,
                &[
                    ("adapter_config.json", br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"base_model_name_or_path":"Org/B"}"#.to_vec()),
                    ("adapter_model.safetensors", safetensors(3, None)),
                ],
            );
            store.import(&peft, &b, &Provenance::new(ORIGIN_UPLOAD), "peft").unwrap();
            let peft = store.adapters().into_iter().find(|a| a.id == "peft").unwrap();
            assert_eq!((peft.format, peft.rank, peft.path.clone()), (AdapterFormat::Peft, Some(8), store.dir().join("adapters/peft")));
        }

        #[test]
        fn safetensors_and_mlx_directories_and_projectors_have_their_own_layouts() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("store"));
            let hf = tmp.path().join("hf");
            let a = stage(&hf, &[("config.json", br#"{"architectures":["X"]}"#.to_vec()), ("model.safetensors", safetensors(2, None))]);
            store.import(&hf, &a, &Provenance::new(ORIGIN_UPLOAD), "x-st").unwrap();
            let mlx = tmp.path().join("mlx");
            let b = stage(
                &mlx,
                &[
                    ("config.json", br#"{"architectures":["X"]}"#.to_vec()),
                    ("model.safetensors", safetensors(2, Some(serde_json::json!({"format":"mlx"})))),
                ],
            );
            store.import(&mlx, &b, &Provenance::new(ORIGIN_UPLOAD), "x-mlx").unwrap();
            let dirs = store.dir_models();
            assert_eq!(
                dirs.iter().map(|m| (m.id.as_str(), m.layout, m.complete)).collect::<Vec<_>>(),
                [("x-mlx", ModelLayout::Mlx, true), ("x-st", ModelLayout::Safetensors, true)]
            );
            assert_eq!(dirs[1].files, ["safetensors/x-st/config.json", "safetensors/x-st/model.safetensors"]);
            assert!(store.list().is_empty(), "GGUF list only");

            let mm = tmp.path().join("mm");
            let c = stage(&mm, &[("mmproj-F16.gguf", crate::gguf::header(&[("general.architecture", serde_json::json!("clip"))]))]);
            let mut p = Provenance::new(ORIGIN_UPLOAD);
            p.base = Some(ProvenanceBase { id: Some("x-st".into()), ..Default::default() });
            store.import(&mm, &c, &p, "vision").unwrap();
            let proj = store.projectors();
            assert_eq!((proj[0].id.as_str(), proj[0].for_model.as_deref()), ("vision", Some("x-st")));
            assert_eq!(store.remove_projector(&proj[0]).unwrap(), c.size());
            assert_eq!(store.remove_local(&dirs[0]).unwrap(), b.size());
            assert!(!store.dir().join("mlx/x-mlx").exists());
            assert_eq!(store.bytes_by_origin(ORIGIN_UPLOAD), a.size());
        }

        #[test]
        fn directory_models_keep_their_limits_record_inside_and_hidden() {
            use crate::limits::{LimitSource, ModelLimits};
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path());
            let dir = tmp.path().join("safetensors/x-st");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), b"{}").unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; 8]).unwrap();
            let m = store.dir_models().remove(0);
            let r = LimitsRecord::new(ModelLimits::new().with_image_tokens(256), LimitSource::Operator);
            store.record_limits(&m, &r).unwrap();
            assert!(dir.join(".limits.json").is_file());
            let again = store.dir_models().remove(0);
            assert_eq!(again.files, m.files, "the record is not one of the model's files");
            assert_eq!(again.size, m.size);
            assert_eq!(again.limits.as_ref().map(|l| (l.limits, l.source)), Some((r.limits, LimitSource::Operator)));
            store.remove_local(&again).unwrap();
            assert!(!dir.exists(), "the record doesn't keep the directory alive");
        }

        #[test]
        fn disk_space_is_read_before_the_store_exists() {
            let tmp = tempfile::tempdir().unwrap();
            let store = ModelStore::new(tmp.path().join("not/yet"));
            let d = store.disk_space().unwrap();
            assert!(d.total >= d.free && d.total > 0);
            assert!(store.free_bytes() > 0);
        }
    }
}
