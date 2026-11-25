//! Model files a person brings themselves (a fine-tune made elsewhere, a trainer's output):
//! [`inspect`] reads and hashes them, [`Provenance`] records where they came from, and
//! [`ModelStore::import`](crate::ModelStore::import) places them in the store.
//!
//! Weights are data, never code: pickled weights (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`),
//! Python files and other code are refused, and so are symlinks and anything that isn't a
//! regular file. GGUF and safetensors files are read by their headers only, with size caps.
//!
//! What can be inspected:
//!
//! - GGUF files: one model, a split model (every `-0000k-of-0000n` part), a LoRA adapter or a
//!   multimodal projector, told apart by the header ([`gguf::read_header`]).
//! - A safetensors model directory (`config.json` with `architectures`, `*.safetensors`, an
//!   index covering every shard when there are several, tokenizer files), for vLLM.
//! - An MLX model directory (the same, saved by MLX-LM), for MLX-LM.
//! - A PEFT LoRA adapter directory (`adapter_config.json`, `adapter_model.safetensors`), for vLLM.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::download::Progress;
use crate::gguf::{self, GgufHeader, GgufInfo, GgufKind};
use crate::hf::{AdapterFormat, check_file, check_repo, peft_lora, split_part};
use crate::{Error, Result};

/// Where [`ModelStore::import`](crate::ModelStore::import) writes an entry's [`Provenance`].
pub const PROVENANCE_FILE: &str = "provenance.json";
/// [`Provenance::format`] for records written by this crate.
pub const PROVENANCE_FORMAT: &str = "weirdnet-provenance/1";
/// [`Provenance::origin`]: sent from another machine.
pub const ORIGIN_UPLOAD: &str = "upload";
/// [`Provenance::origin`]: made by a training run.
pub const ORIGIN_TRAIN: &str = "train";
/// [`Provenance::origin`]: downloaded from a Hub.
pub const ORIGIN_HUB: &str = "hub";
/// The largest JSON file (`config.json`, `adapter_config.json`, an index, a provenance record)
/// that is parsed.
pub const JSON_LIMIT: u64 = 16 << 20;
/// The largest safetensors header that is parsed (the format's own limit).
pub const SAFETENSORS_HEADER_LIMIT: u64 = 100 << 20;
/// File extensions that may be inspected (lower case). Anything else is refused.
pub const ALLOWED_EXTENSIONS: &[&str] = &["gguf", "safetensors", "json", "txt", "model", "jinja", "tiktoken", "md"];
/// File names without an allowed extension that may come along (license and readme files).
pub const ALLOWED_NAMES: &[&str] = &["LICENSE", "NOTICE", "README", "USE_POLICY"];
/// Pickled weights: they can run code when loaded, so they are never accepted.
const PICKLES: &[&str] = &["bin", "pt", "pth", "ckpt", "pkl", "pickle", "joblib", "npy", "npz"];
/// Code.
const CODE: &[&str] = &["py", "pyc", "pyo", "pyx", "so", "dll", "dylib", "exe", "sh", "bat", "cmd", "ps1", "js"];
/// Directories are read this many levels deep at most.
const MAX_DEPTH: usize = 3;

/// What an artifact is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactKind {
    /// A full model.
    Model,
    /// A LoRA adapter for a base model.
    Adapter { format: AdapterFormat },
    /// A multimodal projector (llama.cpp `--mmproj`) for a model.
    Projector,
}

impl ArtifactKind {
    pub fn is_adapter(&self) -> bool {
        matches!(self, ArtifactKind::Adapter { .. })
    }

    /// The [`ProvenanceKind`] for records about it.
    pub fn provenance_kind(&self) -> ProvenanceKind {
        match self {
            ArtifactKind::Model => ProvenanceKind::Model,
            ArtifactKind::Adapter { .. } => ProvenanceKind::Adapter,
            ArtifactKind::Projector => ProvenanceKind::Projector,
        }
    }
}

/// How an artifact's weights are stored. A PEFT adapter is [`ArtifactFormat::Safetensors`]
/// with [`ArtifactKind::Adapter`] of [`AdapterFormat::Peft`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactFormat {
    /// GGUF, in `parts` files (1 unless split).
    Gguf { parts: u32 },
    /// A safetensors directory (Hugging Face Transformers layout), for vLLM.
    Safetensors,
    /// An MLX directory (MLX-LM's layout), for MLX-LM.
    Mlx,
}

/// One file of an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactFile {
    /// `/`-separated path relative to the artifact (a file name for single files).
    pub path: String,
    pub size: u64,
    /// Lower-case hex SHA-256.
    pub sha256: String,
}

/// What [`inspect`] found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub format: ArtifactFormat,
    /// GGUF `general.architecture`, or the first of `config.json`'s `architectures`. `None` for
    /// PEFT adapters (their config doesn't say).
    pub architecture: Option<String>,
    /// The base model as a Hub id: the GGUF header's `general.base_model`, PEFT's
    /// `base_model_name_or_path`, or `config.json`'s `_name_or_path`, when they name one.
    pub base_model: Option<String>,
    /// A GGUF model's KV shape, when its header has it.
    pub info: Option<GgufInfo>,
    /// PEFT `r`.
    pub rank: Option<u32>,
    /// GGUF `adapter.lora.alpha` or PEFT `lora_alpha`.
    pub alpha: Option<f32>,
    /// Split GGUF parts in order; otherwise sorted by path.
    pub files: Vec<ArtifactFile>,
    /// See [`revision`].
    pub revision: String,
}

impl Artifact {
    /// All files together.
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// A content revision for `files`: `sha256:` and the first 16 hex digits of the SHA-256 of the
/// `path:sha256` lines, sorted by path. The same files give the same revision wherever they are.
pub fn revision(files: &[ArtifactFile]) -> String {
    let mut lines: Vec<String> = files.iter().map(|f| format!("{}:{}\n", f.path, f.sha256)).collect();
    lines.sort();
    let digest = Sha256::digest(lines.concat().as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{}", &hex[..16])
}

/// Whether `path` may be part of an artifact: a relative, `/`-separated path of plain parts
/// (no `..`, no hidden parts, at most three levels), with an allowed extension. Pickles, Python
/// and other code are refused with the reason.
pub fn check_name(path: &str) -> Result<()> {
    check_file(path).map_err(|_| Error::Model(format!("{path:?} is not a plain relative file name")))?;
    if path.split('/').any(|p| p.starts_with('.')) {
        return Err(Error::Model(format!("{path}: hidden files are not accepted")));
    }
    if path.split('/').count() > MAX_DEPTH {
        return Err(Error::Model(format!("{path}: nested more than {MAX_DEPTH} levels")));
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some(e) if PICKLES.contains(&e) => {
            Err(Error::Model(format!("{path}: pickled weights can run code and are never accepted; convert them to safetensors or GGUF")))
        }
        Some(e) if CODE.contains(&e) => Err(Error::Model(format!("{path}: code is never accepted with model files"))),
        Some(e) if ALLOWED_EXTENSIONS.contains(&e) => Ok(()),
        _ if ALLOWED_NAMES.iter().any(|n| name.eq_ignore_ascii_case(n) || name.to_ascii_uppercase().starts_with(&format!("{n}."))) => Ok(()),
        _ => Err(Error::Model(format!("{path}: not a model file type ({})", ALLOWED_EXTENSIONS.join(", ")))),
    }
}

/// A file to inspect: its artifact path and where it is.
#[derive(Debug, Clone)]
struct Found {
    rel: String,
    abs: PathBuf,
    size: u64,
}

/// A regular file, never a symlink or a device.
fn regular(path: &Path) -> Result<u64> {
    let m = std::fs::symlink_metadata(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    if m.file_type().is_symlink() {
        return Err(Error::Model(format!("{}: symlinks are not accepted", path.display())));
    }
    if !m.is_file() {
        return Err(Error::Model(format!("{}: not a regular file", path.display())));
    }
    Ok(m.len())
}

fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<Found>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = e.path();
        let ft = e.file_type()?;
        if ft.is_symlink() {
            return Err(Error::Model(format!("{}: symlinks are not accepted", path.display())));
        } else if ft.is_dir() {
            if depth + 1 >= MAX_DEPTH {
                return Err(Error::Model(format!("{}: nested more than {MAX_DEPTH} levels", path.display())));
            }
            walk(root, &path, depth + 1, out)?;
        } else if ft.is_file() {
            let rel: Vec<String> =
                path.strip_prefix(root).unwrap_or(&path).components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            let rel = rel.join("/");
            check_name(&rel)?;
            out.push(Found { rel, size: regular(&path)?, abs: path });
        } else {
            return Err(Error::Model(format!("{}: not a regular file", path.display())));
        }
    }
    Ok(())
}

/// One directory (its files, recursively), or files given one by one (by file name).
fn gather(paths: &[PathBuf]) -> Result<Vec<Found>> {
    let mut out = Vec::new();
    match paths {
        [] => return Err(Error::Model("nothing to inspect".into())),
        [one] if std::fs::symlink_metadata(one).is_ok_and(|m| m.is_dir()) => walk(one, one, 0, &mut out)?,
        _ => {
            for p in paths {
                let size = regular(p)?;
                let rel = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                check_name(&rel)?;
                if out.iter().any(|f: &Found| f.rel == rel) {
                    return Err(Error::Model(format!("{rel} is given twice")));
                }
                out.push(Found { rel, abs: p.clone(), size });
            }
        }
    }
    if out.is_empty() {
        return Err(Error::Model("no files to inspect".into()));
    }
    Ok(out)
}

/// A JSON file, with [`JSON_LIMIT`].
fn read_json(path: &Path) -> Result<Value> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.take(JSON_LIMIT + 1).read_to_end(&mut buf)?;
    if buf.len() as u64 > JSON_LIMIT {
        return Err(Error::Model(format!("{}: larger than {JSON_LIMIT} bytes", path.display())));
    }
    serde_json::from_slice(&buf).map_err(|e| Error::Model(format!("{}: not valid JSON: {e}", path.display())))
}

/// What a safetensors header says ([`read_safetensors`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SafetensorsHeader {
    /// The header's length in bytes (after the 8-byte length).
    pub header_len: u64,
    pub tensors: usize,
    /// Bytes of tensor data after the header.
    pub data_bytes: u64,
    /// `__metadata__` (MLX-LM writes `format` = `mlx`).
    pub metadata: BTreeMap<String, String>,
}

/// Bytes per element of a safetensors dtype, when it is a whole number of bytes.
fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" | "C64" => 8,
        _ => return None,
    })
}

/// Reads and checks a safetensors header from the start of a file of `file_size` bytes: the
/// 8-byte length, the JSON header (at most [`SAFETENSORS_HEADER_LIMIT`]), and the tensors'
/// offsets, which must cover the data after the header exactly, with no gaps or overlaps, each
/// as long as its dtype and shape say.
pub fn read_safetensors(mut r: impl Read, file_size: u64) -> Result<SafetensorsHeader> {
    let bad = |m: String| Error::Model(format!("safetensors: {m}"));
    let mut len = [0u8; 8];
    r.read_exact(&mut len).map_err(|e| bad(format!("header cut short: {e}")))?;
    let header_len = u64::from_le_bytes(len);
    if header_len > SAFETENSORS_HEADER_LIMIT {
        return Err(bad(format!("header is larger than {SAFETENSORS_HEADER_LIMIT} bytes")));
    }
    if header_len.saturating_add(8) > file_size {
        return Err(bad("header is longer than the file".into()));
    }
    let mut buf = vec![0; header_len as usize];
    r.read_exact(&mut buf).map_err(|e| bad(format!("header cut short: {e}")))?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&buf).map_err(|e| bad(format!("header is not a JSON object: {e}")))?;
    let data_bytes = file_size - 8 - header_len;
    let mut metadata = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, t) in &header {
        if name == "__metadata__" {
            for (k, v) in t.as_object().into_iter().flatten() {
                metadata.insert(k.clone(), v.as_str().ok_or_else(|| bad(format!("metadata {k} is not a string")))?.to_string());
            }
            continue;
        }
        let dtype = t["dtype"].as_str().ok_or_else(|| bad(format!("{name} has no dtype")))?;
        let shape: Vec<u64> = t["shape"]
            .as_array()
            .ok_or_else(|| bad(format!("{name} has no shape")))?
            .iter()
            .map(Value::as_u64)
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("{name}: bad shape")))?;
        let offsets: Vec<u64> = t["data_offsets"]
            .as_array()
            .filter(|a| a.len() == 2)
            .and_then(|a| a.iter().map(Value::as_u64).collect::<Option<Vec<_>>>())
            .ok_or_else(|| bad(format!("{name} has no data_offsets")))?;
        let (start, end) = (offsets[0], offsets[1]);
        if end < start {
            return Err(bad(format!("{name}: offsets run backwards")));
        }
        if let Some(size) = dtype_size(dtype) {
            let want = shape.iter().try_fold(size, |acc, d| acc.checked_mul(*d)).ok_or_else(|| bad(format!("{name}: shape overflows")))?;
            if end - start != want {
                return Err(bad(format!("{name}: {} bytes for {dtype} {shape:?}, which needs {want}", end - start)));
            }
        }
        spans.push((start, end, name.clone()));
    }
    spans.sort();
    let mut at = 0;
    for (start, end, name) in &spans {
        if *start != at {
            return Err(bad(format!("{name} starts at {start}, not {at}: gaps and overlaps are not allowed")));
        }
        at = *end;
    }
    if at != data_bytes {
        return Err(bad(format!("tensors cover {at} bytes; the file has {data_bytes} after the header")));
    }
    Ok(SafetensorsHeader { header_len, tensors: spans.len(), data_bytes, metadata })
}

fn safetensors_of(f: &Found) -> Result<SafetensorsHeader> {
    let file = std::fs::File::open(&f.abs)?;
    read_safetensors(std::io::BufReader::new(file), f.size).map_err(|e| Error::Model(format!("{}: {e}", f.rel)))
}

/// The header read so far from the start of a file being received ([`early_header`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EarlyHeader {
    Gguf(GgufHeader),
    Safetensors(SafetensorsHeader),
}

/// A reader over a prefix that notes whether a read reached its end.
struct Prefix<'a> {
    bytes: &'a [u8],
    hit_end: bool,
}

impl Read for Prefix<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.bytes.read(buf)?;
        if n < buf.len() {
            self.hit_end = true;
        }
        Ok(n)
    }
}

/// Checks a file's header from the first bytes received (`prefix`) of a file of `file_size`
/// bytes named `path`, so a bad file is refused before the rest is sent. `Ok(None)` when more
/// bytes are needed (or the file type has no header); an error when the header is bad.
pub fn early_header(path: &str, prefix: &[u8], file_size: u64) -> Result<Option<EarlyHeader>> {
    let lower = path.to_ascii_lowercase();
    let complete = prefix.len() as u64 >= file_size;
    if lower.ends_with(".gguf") {
        let mut r = Prefix { bytes: prefix, hit_end: false };
        return match gguf::read_header(&mut r) {
            Ok(h) => Ok(Some(EarlyHeader::Gguf(h))),
            Err(_) if r.hit_end && !complete => Ok(None),
            Err(e) => Err(Error::Model(format!("{path}: {e}"))),
        };
    }
    if lower.ends_with(".safetensors") {
        let need = prefix.get(..8).map(|l| u64::from_le_bytes(l.try_into().expect("8 bytes")).saturating_add(8));
        return match need {
            Some(n) if n <= SAFETENSORS_HEADER_LIMIT + 8 && (prefix.len() as u64) < n && !complete => Ok(None),
            None if !complete => Ok(None),
            _ => read_safetensors(prefix, file_size).map(|h| Some(EarlyHeader::Safetensors(h))).map_err(|e| Error::Model(format!("{path}: {e}"))),
        };
    }
    Ok(None)
}

/// Kind, format and what the headers say, before hashing.
struct Classified {
    kind: ArtifactKind,
    format: ArtifactFormat,
    architecture: Option<String>,
    base_model: Option<String>,
    info: Option<GgufInfo>,
    rank: Option<u32>,
    alpha: Option<f32>,
    /// `files` reordered: split parts in order, else by path.
    files: Vec<Found>,
}

fn classify(mut files: Vec<Found>) -> Result<Classified> {
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    let is = |f: &Found, ext: &str| f.rel.to_ascii_lowercase().ends_with(ext);
    let ggufs = files.iter().filter(|f| is(f, ".gguf")).count();
    if ggufs > 0 {
        if ggufs != files.len() {
            return Err(Error::Model("GGUF files come alone: give only the .gguf file or its split parts".into()));
        }
        return classify_gguf(files);
    }
    let root = |name: &str| files.iter().find(|f| f.rel == name).cloned();
    if let Some(config) = root("adapter_config.json") {
        let cfg = read_json(&config.abs)?;
        let (rank, alpha, base) = peft_lora(&cfg).ok_or_else(|| Error::Model("adapter_config.json: only LoRA adapters are accepted".into()))?;
        root("adapter_model.safetensors").ok_or_else(|| Error::Model("a PEFT adapter needs adapter_model.safetensors".into()))?;
        for f in files.iter().filter(|f| is(f, ".safetensors")) {
            safetensors_of(f)?;
        }
        return Ok(Classified {
            kind: ArtifactKind::Adapter { format: AdapterFormat::Peft },
            format: ArtifactFormat::Safetensors,
            architecture: None,
            base_model: base,
            info: None,
            rank,
            alpha,
            files,
        });
    }
    if let Some(config) = root("config.json") {
        let cfg = read_json(&config.abs)?;
        let architecture = cfg["architectures"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| Error::Model("config.json has no architectures".into()))?
            .to_string();
        let shards: Vec<&Found> = files.iter().filter(|f| is(f, ".safetensors")).collect();
        if shards.is_empty() {
            return Err(Error::Model("a model directory needs its *.safetensors weights".into()));
        }
        let mut mlx = cfg.get("quantization").is_some_and(Value::is_object);
        for s in &shards {
            let h = safetensors_of(s)?;
            mlx |= h.metadata.get("format").is_some_and(|f| f == "mlx");
        }
        let indexes: Vec<&Found> = files.iter().filter(|f| is(f, ".safetensors.index.json")).collect();
        match indexes.as_slice() {
            [] if shards.len() > 1 => return Err(Error::Model("several *.safetensors files need a *.safetensors.index.json".into())),
            [] => {}
            [index] => {
                let map = read_json(&index.abs)?;
                let named: BTreeSet<&str> = map["weight_map"]
                    .as_object()
                    .ok_or_else(|| Error::Model(format!("{}: no weight_map", index.rel)))?
                    .values()
                    .map(|v| v.as_str().ok_or_else(|| Error::Model(format!("{}: bad weight_map", index.rel))))
                    .collect::<Result<_>>()?;
                let dir = index.rel.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
                let have: BTreeSet<String> = shards.iter().map(|s| s.rel.clone()).collect();
                let want: BTreeSet<String> = named.iter().map(|n| format!("{dir}{n}")).collect();
                if let Some(missing) = want.difference(&have).next() {
                    return Err(Error::Model(format!("{}: names {missing}, which is missing", index.rel)));
                }
                if let Some(extra) = have.difference(&want).next() {
                    return Err(Error::Model(format!("{extra} is not in {}", index.rel)));
                }
            }
            _ => return Err(Error::Model("more than one *.safetensors.index.json".into())),
        }
        let base_model = cfg["_name_or_path"].as_str().filter(|b| check_repo(b).is_ok()).map(str::to_string);
        return Ok(Classified {
            kind: ArtifactKind::Model,
            format: if mlx { ArtifactFormat::Mlx } else { ArtifactFormat::Safetensors },
            architecture: Some(architecture),
            base_model,
            info: None,
            rank: None,
            alpha: None,
            files,
        });
    }
    Err(Error::Model(
        "not a model: give GGUF files, a safetensors or MLX model directory (config.json and *.safetensors), or a PEFT adapter directory (adapter_config.json)".into(),
    ))
}

fn classify_gguf(mut files: Vec<Found>) -> Result<Classified> {
    let parts = if let [one] = files.as_slice() {
        if let Some((_, part, of)) = split_part(&one.rel)
            && (part, of) != (1, 1)
        {
            return Err(Error::Model(format!("{} is part {part} of {of}: give every part", one.rel)));
        }
        1
    } else {
        let mut seen: Vec<(u32, Found)> = Vec::new();
        let mut set: Option<(String, u32)> = None;
        for f in &files {
            let (base, part, of) = split_part(&f.rel)
                .ok_or_else(|| Error::Model("one model per upload: several GGUF files must be the parts of one split model".into()))?;
            let dir = f.rel.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
            let key = (format!("{dir}{base}"), of);
            if set.get_or_insert_with(|| key.clone()) != &key {
                return Err(Error::Model("one model per upload: the GGUF parts belong to different models".into()));
            }
            seen.push((part, f.clone()));
        }
        let of = set.map(|s| s.1).unwrap_or(0);
        seen.sort_by_key(|(p, _)| *p);
        if seen.len() as u32 != of || seen.iter().enumerate().any(|(i, (p, _))| *p != i as u32 + 1) {
            return Err(Error::Model(format!("a split model needs all {of} parts, once each")));
        }
        files = seen.into_iter().map(|(_, f)| f).collect();
        of
    };
    let first = &files[0];
    let header =
        gguf::read_header(std::io::BufReader::new(std::fs::File::open(&first.abs)?)).map_err(|e| Error::Model(format!("{}: {e}", first.rel)))?;
    if let Some(info) = &header.info
        && parts > 1
        && info.split_count != parts as u64
    {
        return Err(Error::Model(format!("{}: the header says {} parts, not {parts}", first.rel, info.split_count)));
    }
    let (kind, alpha) = match &header.kind {
        GgufKind::Model => (ArtifactKind::Model, None),
        GgufKind::Adapter { alpha, .. } if header.kind.is_lora() => (ArtifactKind::Adapter { format: AdapterFormat::GgufLora }, *alpha),
        GgufKind::Adapter { adapter_type, .. } => {
            return Err(Error::Model(format!("{}: only LoRA adapters are accepted, not {adapter_type:?}", first.rel)));
        }
        GgufKind::Projector => (ArtifactKind::Projector, None),
    };
    if parts > 1 && kind != ArtifactKind::Model {
        return Err(Error::Model(format!("{}: only models come in split parts", first.rel)));
    }
    Ok(Classified {
        kind,
        format: ArtifactFormat::Gguf { parts },
        architecture: Some(header.architecture),
        base_model: header.base_model,
        info: header.info,
        rank: None,
        alpha,
        files,
    })
}

/// SHA-256 of `path`, read in chunks, adding to `done` and calling `progress` as it goes.
fn hash_file(path: &Path, done: &mut u64, total: u64, progress: &mut dyn FnMut(Progress) -> bool) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        *done += n as u64;
        if !progress(Progress { done: *done, total }) {
            return Err(Error::Cancelled { done: crate::human(*done), total: crate::human(total) });
        }
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn finish(c: Classified, progress: &mut dyn FnMut(Progress) -> bool) -> Result<Artifact> {
    let total = c.files.iter().map(|f| f.size).sum();
    let mut done = 0;
    let mut files = Vec::with_capacity(c.files.len());
    for f in &c.files {
        let sha256 = hash_file(&f.abs, &mut done, total, progress)?;
        files.push(ArtifactFile { path: f.rel.clone(), size: f.size, sha256 });
    }
    let revision = revision(&files);
    Ok(Artifact {
        kind: c.kind,
        format: c.format,
        architecture: c.architecture,
        base_model: c.base_model,
        info: c.info,
        rank: c.rank,
        alpha: c.alpha,
        files,
        revision,
    })
}

/// Reads and hashes local model files: one directory, or files given one by one (a GGUF, every
/// part of a split GGUF). Headers are checked before anything is hashed. See the module docs
/// for what is accepted and refused.
pub fn inspect(paths: &[PathBuf]) -> Result<Artifact> {
    inspect_with(paths, &mut |_| true)
}

/// [`inspect`], calling `progress` while hashing; it returns `false` to cancel.
pub fn inspect_with(paths: &[PathBuf], progress: &mut dyn FnMut(Progress) -> bool) -> Result<Artifact> {
    finish(classify(gather(paths)?)?, progress)
}

/// Inspects files that were received into `dir` under their artifact paths (as an upload
/// stages them) and checks each against `files`: present, a regular file, the same size and
/// SHA-256. The headers are checked as [`inspect`] does.
pub fn inspect_staged(dir: &Path, files: &[ArtifactFile], progress: &mut dyn FnMut(Progress) -> bool) -> Result<Artifact> {
    if files.is_empty() {
        return Err(Error::Model("no files".into()));
    }
    let mut found = Vec::with_capacity(files.len());
    let mut names = BTreeSet::new();
    for f in files {
        check_name(&f.path)?;
        if !names.insert(f.path.as_str()) {
            return Err(Error::Model(format!("{} is listed twice", f.path)));
        }
        let abs = dir.join(&f.path);
        let size = regular(&abs)?;
        if size != f.size {
            return Err(Error::Model(format!("{}: {size} bytes, expected {}", f.path, f.size)));
        }
        found.push(Found { rel: f.path.clone(), abs, size });
    }
    let artifact = finish(classify(found)?, progress)?;
    for want in files {
        let got = artifact.files.iter().find(|g| g.path == want.path).expect("every file is inspected");
        if !got.sha256.eq_ignore_ascii_case(&want.sha256) {
            return Err(Error::Checksum { name: want.path.clone(), got: got.sha256.clone() });
        }
    }
    Ok(artifact)
}

/// [`Provenance::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProvenanceKind {
    #[default]
    Model,
    Adapter,
    Projector,
}

/// [`Provenance::base`]: the model an artifact was made from or runs on.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProvenanceBase {
    /// The base's id where it runs (a served or stored model id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Its Hub id (`owner/name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// [`Provenance::dataset`]: what it was trained on (hashes, counts and licenses; never rows).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProvenanceDataset {
    /// The dataset manifest's format, such as an app's `name/1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// SHA-256 of the manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    /// Other counts by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counts: BTreeMap<String, u64>,
    /// Licenses of the data (and of a teacher model's output, when one made it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub licenses: Vec<String>,
}

/// [`Provenance::trainer`]: what made it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProvenanceTrainer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
}

fn provenance_format() -> String {
    PROVENANCE_FORMAT.into()
}

/// Where a model, adapter or projector came from, kept next to its files
/// ([`PROVENANCE_FILE`]). It is what the uploader said, checked against the bytes where it
/// can be (files, hashes and revision are set from them on import); it is signed by nobody and
/// is not a trust anchor. Unknown fields (an app's own) are kept in [`Provenance::extra`] and
/// written back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Provenance {
    /// [`PROVENANCE_FORMAT`], or an app's own name.
    #[serde(default = "provenance_format")]
    pub format: String,
    #[serde(default)]
    pub kind: ProvenanceKind,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub files: Vec<ArtifactFile>,
    /// See [`revision`].
    #[serde(default)]
    pub revision: String,
    /// [`ORIGIN_UPLOAD`], [`ORIGIN_TRAIN`], [`ORIGIN_HUB`] or an app's own.
    #[serde(default)]
    pub origin: String,
    /// An SPDX id, or `unknown` when the uploader said so explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<ProvenanceBase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<ProvenanceDataset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainer: Option<ProvenanceTrainer>,
    /// The uploading device's label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uploaded_by: Option<String>,
    /// Unix time in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uploaded_ms: Option<u64>,
    /// Any other fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            format: provenance_format(),
            kind: ProvenanceKind::Model,
            id: String::new(),
            files: Vec::new(),
            revision: String::new(),
            origin: String::new(),
            license: None,
            base: None,
            dataset: None,
            trainer: None,
            uploaded_by: None,
            uploaded_ms: None,
            extra: BTreeMap::new(),
        }
    }
}

impl Provenance {
    /// A record with this origin ([`ORIGIN_UPLOAD`] and so on).
    pub fn new(origin: impl Into<String>) -> Self {
        Self { origin: origin.into(), ..Default::default() }
    }

    /// A record for `artifact` stored as `id`: its kind, files and revision, and its base
    /// model's Hub id when the files name one.
    pub fn for_artifact(artifact: &Artifact, id: impl Into<String>, origin: impl Into<String>) -> Self {
        let mut p = Self::new(origin);
        p.id = id.into();
        p.kind = artifact.kind.provenance_kind();
        p.files = artifact.files.clone();
        p.revision = artifact.revision.clone();
        if let Some(hub) = &artifact.base_model {
            p.base = Some(ProvenanceBase { hub: Some(hub.clone()), ..Default::default() });
        }
        p
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Bytes of the files it lists.
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// The record in `dir` ([`PROVENANCE_FILE`]), if there is a readable one.
    pub fn read(dir: &Path) -> Option<Provenance> {
        let path = dir.join(PROVENANCE_FILE);
        regular(&path).ok()?;
        serde_json::from_value(read_json(&path).ok()?).ok()
    }

    /// Writes the record to `dir` ([`PROVENANCE_FILE`]), replacing it atomically.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self).map_err(|e| Error::Model(e.to_string()))?;
        let tmp = dir.join(format!(".{PROVENANCE_FILE}.tmp"));
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, dir.join(PROVENANCE_FILE))?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A safetensors file with one F32 tensor of `n` elements and optional metadata.
    pub(crate) fn safetensors(n: u64, metadata: Option<Value>) -> Vec<u8> {
        let mut header = json!({"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, n * 4]}});
        if let Some(m) = metadata {
            header["__metadata__"] = m;
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(vec![7u8; (n * 4) as usize]);
        out
    }

    /// A small llama GGUF model header, padded.
    pub(crate) fn model_gguf(extra: &[(&str, Value)]) -> Vec<u8> {
        let mut kv = vec![
            ("general.architecture", json!("llama")),
            ("llama.block_count", json!(2)),
            ("llama.attention.head_count", json!(4)),
            ("llama.attention.head_count_kv", json!(2)),
            ("llama.attention.key_length", json!(8)),
        ];
        kv.extend(extra.iter().cloned());
        let mut out = gguf::header(&kv);
        out.extend([1u8; 100]);
        out
    }

    fn write(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn names_refuse_pickles_code_and_hidden_files() {
        for ok in ["m.gguf", "config.json", "tokenizer.model", "d/model-00001-of-00002.safetensors", "LICENSE", "LICENSE.txt", "chat_template.jinja"]
        {
            check_name(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        for (bad, why) in [
            ("pytorch_model.bin", "pickled"),
            ("w.pt", "pickled"),
            ("w.ckpt", "pickled"),
            ("modeling.py", "code"),
            (".hidden.json", "hidden"),
            ("../x.gguf", "plain"),
            ("/abs.gguf", "plain"),
            ("a/b/c/d.json", "nested"),
            ("x.exe", "code"),
            ("x.zip", "not a model file"),
        ] {
            let e = check_name(bad).unwrap_err().to_string();
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn a_gguf_model_is_read_hashed_and_given_a_revision() {
        let tmp = tempfile::tempdir().unwrap();
        let body = model_gguf(&[("general.base_model.0.repo_url", json!("https://huggingface.co/Org/Base"))]);
        let p = write(tmp.path(), "tune-Q4_0.gguf", &body);
        let mut calls = 0;
        let a = inspect_with(std::slice::from_ref(&p), &mut |pr| {
            calls += 1;
            pr.total == body.len() as u64
        })
        .unwrap();
        assert!(calls > 0);
        assert_eq!((a.kind, a.format), (ArtifactKind::Model, ArtifactFormat::Gguf { parts: 1 }));
        assert_eq!((a.architecture.as_deref(), a.base_model.as_deref()), (Some("llama"), Some("Org/Base")));
        assert_eq!(a.info.as_ref().map(|i| i.layers), Some(2));
        assert_eq!(a.files[0].sha256, crate::download::sha256_of(&p).unwrap());
        assert_eq!(a.size(), body.len() as u64);
        assert!(a.revision.starts_with("sha256:") && a.revision.len() == 23);
        assert_eq!(a.revision, revision(&a.files), "stable");
        assert!(matches!(inspect_with(&[p], &mut |_| false), Err(Error::Cancelled { .. })));
    }

    #[test]
    fn revisions_ignore_order_but_not_paths() {
        let f = |p: &str, h: &str| ArtifactFile { path: p.into(), size: 1, sha256: h.into() };
        assert_eq!(revision(&[f("a", "1"), f("b", "2")]), revision(&[f("b", "2"), f("a", "1")]));
        assert_ne!(revision(&[f("a", "1")]), revision(&[f("c", "1")]));
    }

    #[test]
    fn split_ggufs_need_every_part() {
        let tmp = tempfile::tempdir().unwrap();
        let p1 = write(tmp.path(), "m-00001-of-00002.gguf", &model_gguf(&[("split.count", json!(2))]));
        let p2 = write(tmp.path(), "m-00002-of-00002.gguf", &[0; 50]);
        let a = inspect(&[p2.clone(), p1.clone()]).unwrap();
        assert_eq!(a.format, ArtifactFormat::Gguf { parts: 2 });
        assert_eq!(a.files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]);
        assert!(inspect(std::slice::from_ref(&p1)).unwrap_err().to_string().contains("give every part"));
        let other = write(tmp.path(), "x.gguf", &model_gguf(&[]));
        assert!(inspect(&[p1, other]).unwrap_err().to_string().contains("one model per upload"));
    }

    #[test]
    fn gguf_adapters_and_projectors_are_told_apart() {
        let tmp = tempfile::tempdir().unwrap();
        let lora = write(tmp.path(), "tune-LoRA.gguf", &gguf::lora_header("llama"));
        let a = inspect(&[lora]).unwrap();
        assert_eq!(a.kind, ArtifactKind::Adapter { format: AdapterFormat::GgufLora });
        assert_eq!((a.alpha, a.base_model.as_deref()), (Some(32.5), Some("Org/Base-8B")));
        let mm = write(tmp.path(), "mmproj-F16.gguf", &gguf::header(&[("general.architecture", json!("clip")), ("general.type", json!("mmproj"))]));
        assert_eq!(inspect(&[mm]).unwrap().kind, ArtifactKind::Projector);
        let other = write(
            tmp.path(),
            "cv.gguf",
            &gguf::header(&[("general.architecture", json!("llama")), ("general.type", json!("adapter")), ("adapter.type", json!("control_vector"))]),
        );
        assert!(inspect(&[other]).unwrap_err().to_string().contains("only LoRA"));
        let junk = write(tmp.path(), "junk.gguf", b"not a gguf");
        assert!(inspect(&[junk]).is_err());
    }

    #[test]
    fn a_safetensors_directory_is_a_model_and_mlx_is_recognised() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("hf");
        write(&d, "config.json", br#"{"architectures":["Qwen3ForCausalLM"],"_name_or_path":"Qwen/Qwen3-8B"}"#);
        write(&d, "tokenizer.json", b"{}");
        write(&d, "model.safetensors", &safetensors(4, None));
        write(&d, ".gitattributes", b"skipped");
        let a = inspect(std::slice::from_ref(&d)).unwrap();
        assert_eq!((a.kind, a.format), (ArtifactKind::Model, ArtifactFormat::Safetensors));
        assert_eq!((a.architecture.as_deref(), a.base_model.as_deref()), (Some("Qwen3ForCausalLM"), Some("Qwen/Qwen3-8B")));
        assert_eq!(a.files.len(), 3, "hidden files are left out");

        let m = tmp.path().join("mlx");
        write(&m, "config.json", br#"{"architectures":["LlamaForCausalLM"]}"#);
        write(&m, "model.safetensors", &safetensors(2, Some(json!({"format": "mlx"}))));
        assert_eq!(inspect(&[m]).unwrap().format, ArtifactFormat::Mlx);
        let q = tmp.path().join("mlxq");
        write(&q, "config.json", br#"{"architectures":["LlamaForCausalLM"],"quantization":{"bits":4,"group_size":64}}"#);
        write(&q, "model.safetensors", &safetensors(2, None));
        assert_eq!(inspect(&[q]).unwrap().format, ArtifactFormat::Mlx);
    }

    #[test]
    fn shards_must_match_their_index() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("m");
        write(&d, "config.json", br#"{"architectures":["X"]}"#);
        write(&d, "model-00001-of-00002.safetensors", &safetensors(2, None));
        write(&d, "model-00002-of-00002.safetensors", &safetensors(2, None));
        assert!(inspect(std::slice::from_ref(&d)).unwrap_err().to_string().contains("index"));
        write(&d, "model.safetensors.index.json", br#"{"weight_map":{"a":"model-00001-of-00002.safetensors"}}"#);
        assert!(inspect(std::slice::from_ref(&d)).unwrap_err().to_string().contains("not in"));
        write(
            &d,
            "model.safetensors.index.json",
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        );
        assert_eq!(inspect(&[d]).unwrap().files.len(), 4);
    }

    #[test]
    fn peft_adapters_are_lora_only() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().join("peft");
        write(&d, "adapter_config.json", br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"base_model_name_or_path":"Org/B"}"#);
        write(&d, "adapter_model.safetensors", &safetensors(3, None));
        let a = inspect(std::slice::from_ref(&d)).unwrap();
        assert_eq!((a.kind, a.format), (ArtifactKind::Adapter { format: AdapterFormat::Peft }, ArtifactFormat::Safetensors));
        assert_eq!((a.rank, a.alpha, a.base_model.as_deref()), (Some(8), Some(16.0), Some("Org/B")));
        write(&d, "adapter_config.json", br#"{"peft_type":"IA3"}"#);
        assert!(inspect(std::slice::from_ref(&d)).unwrap_err().to_string().contains("only LoRA"));
        write(&d, "adapter_config.json", br#"{"peft_type":"LORA"}"#);
        write(&d, "adapter_model.bin", b"pickle");
        assert!(inspect(&[d]).unwrap_err().to_string().contains("pickled"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let real = write(tmp.path(), "real.gguf", &model_gguf(&[]));
        let link = tmp.path().join("link.gguf");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(inspect(&[link]).unwrap_err().to_string().contains("symlink"));
        let d = tmp.path().join("d");
        std::fs::create_dir(&d).unwrap();
        std::os::unix::fs::symlink(&real, d.join("m.gguf")).unwrap();
        assert!(inspect(&[d]).unwrap_err().to_string().contains("symlink"));
    }

    #[test]
    fn bad_safetensors_headers_are_refused() {
        let good = safetensors(4, None);
        read_safetensors(&good[..], good.len() as u64).unwrap();
        assert!(read_safetensors(&good[..], good.len() as u64 + 1).unwrap_err().to_string().contains("cover"));
        let h = br#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let mut short = (h.len() as u64).to_le_bytes().to_vec();
        short.extend(h);
        short.extend([0u8; 8]);
        assert!(read_safetensors(&short[..], short.len() as u64).unwrap_err().to_string().contains("needs 16"));
        let huge = (SAFETENSORS_HEADER_LIMIT + 1).to_le_bytes();
        assert!(read_safetensors(&huge[..], u64::MAX).unwrap_err().to_string().contains("larger"));
    }

    #[test]
    fn early_headers_wait_for_enough_bytes_and_refuse_bad_ones() {
        let body = model_gguf(&[]);
        assert_eq!(early_header("m.gguf", &body[..10], 10_000).unwrap(), None);
        assert!(matches!(early_header("m.gguf", &body, 10_000).unwrap(), Some(EarlyHeader::Gguf(h)) if h.architecture == "llama"));
        assert!(early_header("m.gguf", b"NOPE and more bytes", 10_000).is_err());
        let st = safetensors(4, None);
        assert_eq!(early_header("model.safetensors", &st[..4], st.len() as u64).unwrap(), None);
        assert_eq!(early_header("model.safetensors", &st[..12], st.len() as u64).unwrap(), None);
        assert!(matches!(early_header("model.safetensors", &st, st.len() as u64).unwrap(), Some(EarlyHeader::Safetensors(h)) if h.tensors == 1));
        assert!(early_header("model.safetensors", &st, st.len() as u64 + 5).is_err());
        assert_eq!(early_header("config.json", b"{", 100).unwrap(), None);
    }

    #[test]
    fn staged_files_must_match_the_offer() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "src/m.gguf", &model_gguf(&[]));
        let a = inspect(std::slice::from_ref(&src)).unwrap();
        let stage = tmp.path().join("stage");
        write(&stage, "m.gguf", &std::fs::read(&src).unwrap());
        assert_eq!(inspect_staged(&stage, &a.files, &mut |_| true).unwrap(), a);
        let mut wrong = a.files.clone();
        wrong[0].sha256 = "00".repeat(32);
        assert!(matches!(inspect_staged(&stage, &wrong, &mut |_| true), Err(Error::Checksum { .. })));
        wrong[0].size += 1;
        assert!(inspect_staged(&stage, &wrong, &mut |_| true).unwrap_err().to_string().contains("expected"));
    }

    #[test]
    fn provenance_round_trips_with_app_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = Provenance::new(ORIGIN_UPLOAD).with_license("apache-2.0");
        p.id = "tune".into();
        p.extra.insert("app_field".into(), json!({"x": 1}));
        p.write(tmp.path()).unwrap();
        let back = Provenance::read(tmp.path()).unwrap();
        assert_eq!(back, p);
        assert_eq!(back.format, PROVENANCE_FORMAT);
        let raw: Value = serde_json::from_slice(&std::fs::read(tmp.path().join(PROVENANCE_FILE)).unwrap()).unwrap();
        assert_eq!(raw["app_field"]["x"], 1, "app fields sit at the top level");
        assert!(raw.get("base").is_none(), "unset fields are left out");
        let app: Provenance = serde_json::from_value(json!({"format": "app-provenance/1", "kind": "adapter", "origin": "train"})).unwrap();
        assert_eq!((app.kind, app.origin.as_str()), (ProvenanceKind::Adapter, ORIGIN_TRAIN));
    }
}
