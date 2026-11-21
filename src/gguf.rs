//! The parts of a GGUF header that size a model's KV cache, read from the start of the file
//! (over HTTP with a `Range` request, see [`Hub::gguf_info`](crate::hf::Hub::gguf_info)).

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

#[derive(Debug, Clone)]
enum Val {
    Int(u64),
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
            t => Ok(self.scalar(t)?.map(Val::Int).unwrap_or(Val::Other)),
        }
    }
}

/// Reads the metadata at the start of a GGUF file (versions 2 and 3) and stops before the
/// tensor table.
pub fn read_info(r: impl Read) -> Result<GgufInfo> {
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
    let mut kv: HashMap<String, Val> = HashMap::new();
    for _ in 0..count {
        let key = r.string()?;
        let t = r.u32()?;
        let v = r.value(t)?;
        kv.insert(key, v);
    }
    let arch = match kv.get("general.architecture") {
        Some(Val::Str(a)) => a.clone(),
        _ => return Err(Error::Model("GGUF header has no general.architecture".into())),
    };
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
            serde_json::Value::Number(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend((n.as_u64().unwrap() as u32).to_le_bytes());
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
    fn junk_and_short_headers_fail() {
        assert!(read_info(&b"nope"[..]).is_err());
        let h = header(&[("general.architecture", json!("x")), ("x.block_count", json!(3))]);
        assert!(read_info(&h[..h.len() - 2]).is_err());
        assert!(read_info(&h[..]).is_err(), "no head size");
    }
}
