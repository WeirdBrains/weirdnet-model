//! The parts of a GGUF header that size a model's KV cache, read from the start of the file
//! (over HTTP with a `Range` request, see [`Hub::gguf_info`](crate::hf::Hub::gguf_info)), and
//! whether a file is a model, a LoRA adapter or a multimodal projector ([`read_header`]), with
//! what a projector's header says about the images and audio it takes ([`ProjectorInfo`]).

use std::collections::HashMap;
use std::io::Read;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// How many header bytes [`read_info`] reads at most (tokenizer vocabularies are the bulk).
pub const HEADER_LIMIT: u64 = 64 << 20;

/// What a GGUF header says about the model's shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GgufInfo {
    /// `general.architecture`, such as `llama` or `qwen3`.
    pub architecture: String,
    /// Transformer blocks.
    pub layers: u64,
    /// KV heads summed over the blocks (per-block counts differ on some models).
    pub kv_heads_total: u64,
    pub key_length: u64,
    pub value_length: u64,
    /// The context the model was trained for.
    pub context_length: Option<u64>,
    /// `split.count` on split files (1 otherwise).
    pub split_count: u64,
}

impl GgufInfo {
    /// KV cache bytes per token at f16: K and V for every KV head of every block.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_heads_total * (self.key_length + self.value_length) * 2
    }
}

/// What kind of GGUF file this is (`general.type`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GgufKind {
    /// A model (`general.type` is `model` or missing).
    Model,
    /// An adapter (`general.type` is `adapter`), applied on top of a base model of the same
    /// architecture.
    Adapter {
        /// `adapter.type`, such as `lora`.
        adapter_type: String,
        /// `adapter.lora.alpha`, when set.
        alpha: Option<f32>,
    },
    /// A multimodal projector (`general.type` is `mmproj`, or the `clip` architecture), the
    /// file llama.cpp's `--mmproj` takes next to a model.
    Projector,
}

impl GgufKind {
    /// A LoRA adapter (`adapter.type` is `lora`), the kind llama.cpp's `--lora` takes.
    pub fn is_lora(&self) -> bool {
        matches!(self, GgufKind::Adapter { adapter_type, .. } if adapter_type == "lora")
    }

    pub fn is_adapter(&self) -> bool {
        matches!(self, GgufKind::Adapter { .. })
    }

    pub fn is_projector(&self) -> bool {
        matches!(self, GgufKind::Projector)
    }
}

/// What a GGUF header says about a file of any kind (see [`read_header`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GgufHeader {
    pub kind: GgufKind,
    /// `general.architecture`, such as `llama` or `qwen3`. An adapter needs a base model with
    /// the same one.
    pub architecture: String,
    /// The model this one was made from, as a Hub id (`owner/name`) when the header has a
    /// Hugging Face `general.base_model.0.repo_url`. For an adapter this is its base model.
    pub base_model: Option<String>,
    /// The repo the weights came from, as a Hub id when the header has a Hugging Face
    /// `general.source.repo_url` or `general.repo_url`.
    pub repo: Option<String>,
    /// The KV shape of a model; `None` for adapters (they have no blocks of their own) and
    /// headers that lack the figures.
    pub info: Option<GgufInfo>,
    /// What a projector takes (`clip.*` keys); `None` for models and adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<ProjectorInfo>,
}

/// What a multimodal projector's header (llama.cpp `mtmd`, the `clip.*` keys) says about the
/// images and audio it takes. Every figure is as written in the header, `None` when the key is
/// missing; [`limits::from_projector`](crate::limits::from_projector) turns them into token costs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProjectorInfo {
    /// `clip.has_vision_encoder`: the model reads images.
    pub vision: bool,
    /// `clip.has_audio_encoder`: the model reads audio.
    pub audio: bool,
    /// The image projector type (`clip.vision.projector_type`, else `clip.projector_type`), such
    /// as `gemma3` or `qwen2.5vl_merger`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_projector: Option<String>,
    /// The audio projector type (`clip.audio.projector_type`, else `clip.projector_type`), such
    /// as `voxtral` or `ultravox`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_projector: Option<String>,
    /// `clip.vision.image_size`: the edge in pixels the encoder works at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<u32>,
    /// `clip.vision.patch_size`: the edge in pixels of one encoder patch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_size: Option<u32>,
    /// `clip.vision.spatial_merge_size`: patches merged per side (Qwen-VL and others).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_merge: Option<u32>,
    /// `clip.vision.projector.scale_factor`: patches pooled per side (Gemma 3, Idefics 3 and
    /// others).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_factor: Option<u32>,
    /// `clip.vision.image_max_pixels`: the most pixels one image is resized to (dynamic-size
    /// encoders).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_max_pixels: Option<u64>,
    /// `clip.audio.projector.stack_factor`: audio frames stacked into one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_stack_factor: Option<u32>,
    /// `clip.audio.num_mel_bins`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_mel_bins: Option<u32>,
}

impl ProjectorInfo {
    /// The projector type that applies to images: the vision type, else the shared one when the
    /// projector has a vision encoder.
    pub fn image_type(&self) -> Option<&str> {
        self.vision.then_some(self.vision_projector.as_deref()).flatten()
    }

    /// The projector type that applies to audio, when the projector has an audio encoder.
    pub fn audio_type(&self) -> Option<&str> {
        self.audio.then_some(self.audio_projector.as_deref()).flatten()
    }
}

#[derive(Debug, Clone)]
enum Val {
    Int(u64),
    Float(f64),
    Str(String),
    Ints(Vec<u64>),
    Other,
}

struct Reader<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: u64) -> Result<Vec<u8>> {
        if self.read + n > HEADER_LIMIT {
            return Err(Error::Model(format!("GGUF header is larger than {HEADER_LIMIT} bytes")));
        }
        let mut buf = vec![0; n as usize];
        self.inner.read_exact(&mut buf).map_err(|e| Error::Model(format!("GGUF header cut short: {e}")))?;
        self.read += n;
        Ok(buf)
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        if self.read + n > HEADER_LIMIT {
            return Err(Error::Model(format!("GGUF header is larger than {HEADER_LIMIT} bytes")));
        }
        let got = std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())?;
        if got != n {
            return Err(Error::Model("GGUF header cut short".into()));
        }
        self.read += n;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().expect("4 bytes")))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().expect("8 bytes")))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        String::from_utf8(self.bytes(n)?).map_err(|_| Error::Model("GGUF string is not UTF-8".into()))
    }

    /// Scalar of type `t`: integers as `u64` (negative ones as `None`), others skipped.
    fn scalar(&mut self, t: u32) -> Result<Option<u64>> {
        Ok(match t {
            0 | 7 => Some(self.bytes(1)?[0] as u64),
            1 => Some(self.bytes(1)?[0] as i8 as i64).and_then(|v| u64::try_from(v).ok()),
            2 => Some(u16::from_le_bytes(self.bytes(2)?.try_into().expect("2 bytes")) as u64),
            3 => u64::try_from(i16::from_le_bytes(self.bytes(2)?.try_into().expect("2 bytes"))).ok(),
            4 => Some(self.u32()? as u64),
            5 => u64::try_from(self.u32()? as i32).ok(),
            10 => Some(self.u64()?),
            11 => u64::try_from(self.u64()? as i64).ok(),
            6 => {
                self.skip(4)?;
                None
            }
            12 => {
                self.skip(8)?;
                None
            }
            t => return Err(Error::Model(format!("GGUF value type {t} is unknown"))),
        })
    }

    fn value(&mut self, t: u32) -> Result<Val> {
        match t {
            8 => {
                let n = self.u64()?;
                if n > 4096 {
                    self.skip(n)?;
                    return Ok(Val::Other);
                }
                String::from_utf8(self.bytes(n)?).map(Val::Str).map_err(|_| Error::Model("GGUF string is not UTF-8".into()))
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if et == 8 {
                    for _ in 0..n {
                        let len = self.u64()?;
                        self.skip(len)?;
                    }
                    return Ok(Val::Other);
                }
                let mut ints = Vec::new();
                for _ in 0..n {
                    match self.scalar(et)? {
                        Some(v) if n <= 4096 => ints.push(v),
                        _ => {}
                    }
                }
                Ok(if n <= 4096 && ints.len() as u64 == n { Val::Ints(ints) } else { Val::Other })
            }
            6 => Ok(Val::Float(f32::from_le_bytes(self.bytes(4)?.try_into().expect("4 bytes")) as f64)),
            12 => Ok(Val::Float(f64::from_le_bytes(self.bytes(8)?.try_into().expect("8 bytes")))),
            t => Ok(self.scalar(t)?.map(Val::Int).unwrap_or(Val::Other)),
        }
    }
}

type Kv = HashMap<String, Val>;

/// The key-value metadata at the start of a GGUF file (versions 2 and 3), stopping before the
/// tensor table.
fn read_kv(r: impl Read) -> Result<Kv> {
    let mut r = Reader { inner: r, read: 0 };
    if r.bytes(4)? != b"GGUF" {
        return Err(Error::Model("not a GGUF file".into()));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(Error::Model(format!("GGUF version {version} is not supported")));
    }
    let _tensors = r.u64()?;
    let count = r.u64()?;
    let mut kv: Kv = HashMap::new();
    for _ in 0..count {
        let key = r.string()?;
        let t = r.u32()?;
        let v = r.value(t)?;
        kv.insert(key, v);
    }
    Ok(kv)
}

fn architecture(kv: &Kv) -> Result<String> {
    match kv.get("general.architecture") {
        Some(Val::Str(a)) => Ok(a.clone()),
        _ => Err(Error::Model("GGUF header has no general.architecture".into())),
    }
}

fn info_from(kv: &Kv) -> Result<GgufInfo> {
    let arch = architecture(kv)?;
    let int = |k: &str| match kv.get(&format!("{arch}.{k}")) {
        Some(Val::Int(v)) => Some(*v),
        _ => None,
    };
    let layers = int("block_count").ok_or_else(|| Error::Model(format!("GGUF header has no {arch}.block_count")))?;
    let heads = match kv.get(&format!("{arch}.attention.head_count")) {
        Some(Val::Int(v)) => Some(*v),
        Some(Val::Ints(v)) => v.iter().copied().max(),
        _ => None,
    };
    let kv_heads_total = match kv.get(&format!("{arch}.attention.head_count_kv")) {
        Some(Val::Int(v)) => v * layers,
        Some(Val::Ints(v)) => v.iter().sum(),
        _ => heads.unwrap_or(0) * layers,
    };
    let head_dim = int("embedding_length").zip(heads.filter(|h| *h > 0)).map(|(e, h)| e / h);
    let key_length = int("attention.key_length").or(head_dim).ok_or_else(|| Error::Model("GGUF header has no head size".into()))?;
    let value_length = int("attention.value_length").unwrap_or(key_length);
    let split_count = match kv.get("split.count") {
        Some(Val::Int(v)) => *v,
        _ => 1,
    };
    let context_length = int("context_length");
    Ok(GgufInfo { architecture: arch, layers, kv_heads_total, key_length, value_length, context_length, split_count })
}

/// Reads the metadata at the start of a GGUF file (versions 2 and 3) and stops before the
/// tensor table. Fails on adapters, which have no blocks; see [`read_header`].
pub fn read_info(r: impl Read) -> Result<GgufInfo> {
    info_from(&read_kv(r)?)
}

/// `owner/name` from a Hugging Face repo URL (`https://huggingface.co/owner/name`).
pub fn hub_id(url: &str) -> Option<String> {
    let rest = url.strip_prefix("https://huggingface.co/").or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let id = rest.trim_end_matches('/');
    crate::hf::check_repo(id).ok().map(|_| id.to_string())
}

/// Reads a GGUF header of any kind: a model (with its KV shape when the header has it), an
/// adapter (`general.type` = `adapter`, with `adapter.type` and `adapter.lora.alpha`) or a
/// multimodal projector (`general.type` = `mmproj`, or architecture `clip`). Unlike
/// [`read_info`] it doesn't fail on adapters.
pub fn read_header(r: impl Read) -> Result<GgufHeader> {
    let kv = read_kv(r)?;
    let architecture = architecture(&kv)?;
    let text = |k: &str| match kv.get(k) {
        Some(Val::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let kind = match text("general.type").as_deref() {
        Some("adapter") => GgufKind::Adapter {
            adapter_type: text("adapter.type").unwrap_or_default(),
            alpha: match kv.get("adapter.lora.alpha") {
                Some(Val::Float(a)) => Some(*a as f32),
                Some(Val::Int(a)) => Some(*a as f32),
                _ => None,
            },
        },
        Some("mmproj") => GgufKind::Projector,
        _ if architecture == "clip" => GgufKind::Projector,
        _ => GgufKind::Model,
    };
    let info = if matches!(kind, GgufKind::Model) { info_from(&kv).ok() } else { None };
    let base_model = text("general.base_model.0.repo_url").and_then(|u| hub_id(&u));
    let repo = text("general.source.repo_url").or_else(|| text("general.repo_url")).and_then(|u| hub_id(&u));
    let projector = kind.is_projector().then(|| projector_from(&kv));
    Ok(GgufHeader { kind, architecture, base_model, repo, info, projector })
}

/// The `clip.*` keys of a projector header.
fn projector_from(kv: &Kv) -> ProjectorInfo {
    let text = |k: &str| match kv.get(k) {
        Some(Val::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let int = |k: &str| match kv.get(k) {
        Some(Val::Int(v)) => Some(*v),
        _ => None,
    };
    let small = |k: &str| int(k).and_then(|v| u32::try_from(v).ok());
    let shared = text("clip.projector_type");
    let vision = int("clip.has_vision_encoder").is_some_and(|v| v != 0);
    let audio = int("clip.has_audio_encoder").is_some_and(|v| v != 0);
    ProjectorInfo {
        vision,
        audio,
        vision_projector: text("clip.vision.projector_type").or_else(|| shared.clone().filter(|_| vision)),
        audio_projector: text("clip.audio.projector_type").or_else(|| shared.filter(|_| audio)),
        image_size: small("clip.vision.image_size"),
        patch_size: small("clip.vision.patch_size"),
        spatial_merge: small("clip.vision.spatial_merge_size"),
        scale_factor: small("clip.vision.projector.scale_factor"),
        image_max_pixels: int("clip.vision.image_max_pixels"),
        audio_stack_factor: small("clip.audio.projector.stack_factor"),
        audio_mel_bins: small("clip.audio.num_mel_bins"),
    }
}

/// GGUF header bytes for tests.
#[cfg(test)]
pub(crate) fn header(kv: &[(&str, serde_json::Value)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    let s = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (k, v) in kv {
        s(&mut out, k);
        match v {
            serde_json::Value::String(text) => {
                out.extend(8u32.to_le_bytes());
                s(&mut out, text);
            }
            serde_json::Value::Number(n) if n.is_u64() => {
                out.extend(4u32.to_le_bytes());
                out.extend((n.as_u64().unwrap() as u32).to_le_bytes());
            }
            serde_json::Value::Number(n) => {
                out.extend(6u32.to_le_bytes());
                out.extend((n.as_f64().unwrap() as f32).to_le_bytes());
            }
            serde_json::Value::Array(items) if items.first().is_some_and(|i| i.is_string()) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for i in items {
                    s(&mut out, i.as_str().unwrap());
                }
            }
            serde_json::Value::Array(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(5u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for i in items {
                    out.extend((i.as_u64().unwrap() as i32).to_le_bytes());
                }
            }
            serde_json::Value::Bool(b) => {
                out.extend(7u32.to_le_bytes());
                out.push(*b as u8);
            }
            _ => unreachable!("test headers use strings, numbers, arrays and bools"),
        }
    }
    out
}

/// A LoRA adapter header with the keys `convert_lora_to_gguf.py` writes (llama.cpp b12590), for
/// tests.
#[cfg(test)]
pub(crate) fn lora_header(arch: &str) -> Vec<u8> {
    use serde_json::json;
    header(&[
        ("general.architecture", json!(arch)),
        ("general.type", json!("adapter")),
        ("general.name", json!("Tune")),
        ("general.base_model.count", json!(1)),
        ("general.base_model.0.repo_url", json!("https://huggingface.co/Org/Base-8B")),
        ("adapter.type", json!("lora")),
        ("adapter.lora.alpha", json!(32.5)),
    ])
}

/// Projector headers with the `clip.*` keys of real llama.cpp `mmproj` files, for tests:
/// `voxtral` (Voxtral Mini 3B), `ultravox`, `gemma3` (Gemma 3 4B), `qwen2.5vl` (Qwen2.5-VL 3B),
/// `qwen2.5o` (Qwen2.5-Omni 3B) and `qwen3vl` (Qwen3-VL 2B).
#[cfg(test)]
pub(crate) fn projector_header(which: &str) -> Vec<u8> {
    use serde_json::json;
    let mut kv = vec![("general.architecture", json!("clip")), ("general.type", json!("mmproj"))];
    match which {
        "voxtral" => kv.extend([
            ("clip.has_audio_encoder", json!(true)),
            ("clip.projector_type", json!("voxtral")),
            ("clip.audio.num_mel_bins", json!(128)),
            ("clip.audio.projector.stack_factor", json!(4)),
        ]),
        "ultravox" => kv.extend([
            ("clip.has_audio_encoder", json!(true)),
            ("clip.projector_type", json!("ultravox")),
            ("clip.audio.num_mel_bins", json!(128)),
            ("clip.audio.projector.stack_factor", json!(8)),
        ]),
        "gemma3" => kv.extend([
            ("clip.projector_type", json!("gemma3")),
            ("clip.has_text_encoder", json!(false)),
            ("clip.has_vision_encoder", json!(true)),
            ("clip.vision.image_size", json!(896)),
            ("clip.vision.patch_size", json!(14)),
        ]),
        "qwen2.5vl" => kv.extend([
            ("clip.has_vision_encoder", json!(true)),
            ("clip.vision.image_size", json!(560)),
            ("clip.vision.patch_size", json!(14)),
            ("clip.projector_type", json!("qwen2.5vl_merger")),
        ]),
        "qwen2.5o" => kv.extend([
            ("clip.has_vision_encoder", json!(true)),
            ("clip.vision.image_size", json!(560)),
            ("clip.vision.patch_size", json!(14)),
            ("clip.has_audio_encoder", json!(true)),
            ("clip.projector_type", json!("qwen2.5o")),
            ("clip.audio.num_mel_bins", json!(128)),
        ]),
        "qwen3vl" => kv.extend([
            ("clip.has_vision_encoder", json!(true)),
            ("clip.vision.image_size", json!(768)),
            ("clip.vision.patch_size", json!(16)),
            ("clip.projector_type", json!("qwen3vl_merger")),
            ("clip.vision.spatial_merge_size", json!(2)),
        ]),
        other => panic!("no test projector {other}"),
    }
    header(&kv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_the_kv_shape() {
        let h = header(&[
            ("general.architecture", json!("qwen3")),
            ("general.name", json!("Qwen3 8B")),
            ("tokenizer.ggml.tokens", json!(["a", "b", "c"])),
            ("qwen3.block_count", json!(36)),
            ("qwen3.context_length", json!(40960)),
            ("qwen3.embedding_length", json!(4096)),
            ("qwen3.attention.head_count", json!(32)),
            ("qwen3.attention.head_count_kv", json!(8)),
            ("qwen3.attention.key_length", json!(128)),
            ("qwen3.attention.value_length", json!(128)),
            ("general.flag", json!(true)),
        ]);
        let info = read_info(&h[..]).unwrap();
        assert_eq!((info.architecture.as_str(), info.layers, info.kv_heads_total, info.context_length), ("qwen3", 36, 288, Some(40960)));
        // The pinned catalog's figure for Qwen3 8B.
        assert_eq!(info.kv_bytes_per_token(), 147_456);
        assert_eq!(info.split_count, 1);
    }

    #[test]
    fn per_block_heads_and_derived_head_size() {
        let h = header(&[
            ("general.architecture", json!("x")),
            ("x.block_count", json!(3)),
            ("x.embedding_length", json!(512)),
            ("x.attention.head_count", json!(8)),
            ("x.attention.head_count_kv", json!([2, 0, 2])),
            ("split.count", json!(3)),
        ]);
        let info = read_info(&h[..]).unwrap();
        assert_eq!((info.kv_heads_total, info.key_length, info.value_length, info.split_count), (4, 64, 64, 3));
        assert_eq!(info.kv_bytes_per_token(), 4 * 128 * 2);
    }

    #[test]
    fn adapters_read_as_headers_but_not_as_info() {
        let h = lora_header("qwen3");
        assert!(read_info(&h[..]).is_err(), "an adapter has no block_count");
        let head = read_header(&h[..]).unwrap();
        assert_eq!(head.kind, GgufKind::Adapter { adapter_type: "lora".into(), alpha: Some(32.5) });
        assert!(head.kind.is_lora() && head.kind.is_adapter());
        assert_eq!((head.architecture.as_str(), head.base_model.as_deref(), head.info), ("qwen3", Some("Org/Base-8B"), None));
    }

    #[test]
    fn models_read_as_headers_with_their_shape() {
        let h = header(&[
            ("general.architecture", json!("llama")),
            ("general.type", json!("model")),
            ("general.source.repo_url", json!("https://huggingface.co/Org/Base-8B")),
            ("llama.block_count", json!(2)),
            ("llama.attention.head_count", json!(4)),
            ("llama.attention.key_length", json!(16)),
        ]);
        let head = read_header(&h[..]).unwrap();
        assert_eq!(head.kind, GgufKind::Model);
        assert!(!head.kind.is_lora());
        assert_eq!(head.repo.as_deref(), Some("Org/Base-8B"));
        assert_eq!(head.info.map(|i| i.layers), Some(2));
        let bare = header(&[("general.architecture", json!("x"))]);
        let head = read_header(&bare[..]).unwrap();
        assert_eq!((head.kind, head.info), (GgufKind::Model, None), "a model header without the shape still reads");
        assert!(read_header(&header(&[("general.type", json!("adapter"))])[..]).is_err(), "no architecture");
        assert_eq!(hub_id("https://huggingface.co/a/b/"), Some("a/b".into()));
        assert_eq!(hub_id("https://example.com/a/b"), None);
        assert_eq!(hub_id("https://huggingface.co/a/b/tree/main"), None);
    }

    #[test]
    fn junk_and_short_headers_fail() {
        assert!(read_info(&b"nope"[..]).is_err());
        let h = header(&[("general.architecture", json!("x")), ("x.block_count", json!(3))]);
        assert!(read_info(&h[..h.len() - 2]).is_err());
        assert!(read_info(&h[..]).is_err(), "no head size");
    }

    #[test]
    fn projectors_are_their_own_kind() {
        let typed = header(&[("general.architecture", json!("clip")), ("general.type", json!("mmproj"))]);
        let h = read_header(&typed[..]).unwrap();
        assert!(h.kind.is_projector() && !h.kind.is_adapter() && h.info.is_none());
        let untyped = header(&[("general.architecture", json!("clip"))]);
        assert_eq!(read_header(&untyped[..]).unwrap().kind, GgufKind::Projector);
    }

    #[test]
    fn projector_headers_say_what_they_take() {
        let v = read_header(&projector_header("voxtral")[..]).unwrap().projector.unwrap();
        assert!(v.audio && !v.vision);
        assert_eq!((v.audio_type(), v.image_type(), v.audio_stack_factor, v.audio_mel_bins), (Some("voxtral"), None, Some(4), Some(128)));
        let g = read_header(&projector_header("gemma3")[..]).unwrap().projector.unwrap();
        assert!(g.vision && !g.audio);
        assert_eq!((g.image_type(), g.audio_type(), g.image_size, g.patch_size, g.scale_factor), (Some("gemma3"), None, Some(896), Some(14), None));
        let o = read_header(&projector_header("qwen2.5o")[..]).unwrap().projector.unwrap();
        assert_eq!((o.image_type(), o.audio_type()), (Some("qwen2.5o"), Some("qwen2.5o")), "one shared type for both encoders");
        let q = read_header(&projector_header("qwen3vl")[..]).unwrap().projector.unwrap();
        assert_eq!(q.spatial_merge, Some(2));
        // Split types, as newer converters write them (Gemma 4).
        let split = header(&[
            ("general.architecture", json!("clip")),
            ("clip.has_vision_encoder", json!(true)),
            ("clip.has_audio_encoder", json!(true)),
            ("clip.vision.projector_type", json!("gemma4v")),
            ("clip.audio.projector_type", json!("gemma4a")),
        ]);
        let s = read_header(&split[..]).unwrap().projector.unwrap();
        assert_eq!((s.image_type(), s.audio_type()), (Some("gemma4v"), Some("gemma4a")));
        // Models and adapters have none; a header from before the field still deserializes.
        assert!(read_header(&lora_header("llama")[..]).unwrap().projector.is_none());
        let old: GgufHeader =
            serde_json::from_value(json!({"kind": "projector", "architecture": "clip", "base_model": null, "repo": null, "info": null})).unwrap();
        assert!(old.projector.is_none());
        let back: GgufHeader =
            serde_json::from_str(&serde_json::to_string(&read_header(&projector_header("ultravox")[..]).unwrap()).unwrap()).unwrap();
        assert_eq!(back.projector.unwrap().audio_stack_factor, Some(8));
    }
}
