//! Which model files Weirdnet knows how to fetch.

use serde::{Deserialize, Serialize};

/// A downloadable GGUF.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Short id, such as `gemma-4-e4b`.
    pub id: String,
    pub label: String,
    /// Hugging Face repo (`owner/name`).
    pub repo: String,
    pub file: String,
    pub size: u64,
    /// Lower-case hex SHA-256 of the file, when known.
    pub sha256: Option<String>,
}

/// Device class, which picks the default model size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Desktop,
    Phone,
}

pub const GEMMA_4_E2B: &str = "gemma-4-e2b";
pub const GEMMA_4_E4B: &str = "gemma-4-e4b";

/// Pinned files: ggml-org's Q4_0 conversions, sizes and SHA-256 from the Hugging Face API (2026-11-18).
pub fn catalog() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: GEMMA_4_E2B.into(),
            label: "Gemma 4 E2B instruct (Q4_0)".into(),
            repo: "ggml-org/gemma-4-E2B-it-GGUF".into(),
            file: "gemma-4-E2B-it-Q4_0.gguf".into(),
            size: 2_841_481_184,
            sha256: Some("8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52".into()),
        },
        ModelSpec {
            id: GEMMA_4_E4B.into(),
            label: "Gemma 4 E4B instruct (Q4_0)".into(),
            repo: "ggml-org/gemma-4-E4B-it-GGUF".into(),
            file: "gemma-4-E4B-it-Q4_0.gguf".into(),
            size: 4_590_807_392,
            sha256: Some("a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2".into()),
        },
    ]
}

/// The catalog entry with this id.
pub fn find(id: &str) -> Option<ModelSpec> {
    catalog().into_iter().find(|m| m.id == id)
}

/// E4B on desktop, E2B on phones.
pub fn default_for(device: Device) -> ModelSpec {
    find(match device {
        Device::Desktop => GEMMA_4_E4B,
        Device::Phone => GEMMA_4_E2B,
    })
    .expect("the default models are in the catalog")
}

impl ModelSpec {
    /// Where the file is downloaded from, on the Hub at `hub`.
    pub fn url(&self, hub: &str) -> String {
        format!("{}/{}/resolve/main/{}", hub.trim_end_matches('/'), self.repo, self.file)
    }

    /// What a consent prompt shows.
    pub fn consent_text(&self, hub: &str) -> Vec<String> {
        vec![
            format!("Download {} ({}, {} bytes)", self.label, human(self.size), self.size),
            format!("from {}/{}", hub.trim_end_matches('/'), self.repo),
            "Nothing is downloaded without your OK; the file is checked (SHA-256) when done.".into(),
        ]
    }
}

/// `2.84 GB` (decimal, like the download pages).
pub fn human(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.2} GB", b as f64 / 1e9)
    } else if b >= 1_000_000 {
        format!("{:.1} MB", b as f64 / 1e6)
    } else {
        format!("{b} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_device() {
        assert_eq!(default_for(Device::Desktop).id, "gemma-4-e4b");
        assert_eq!(default_for(Device::Phone).id, "gemma-4-e2b");
    }

    #[test]
    fn entries_are_pinned_and_urls_point_at_the_hub() {
        for m in catalog() {
            assert_eq!(m.sha256.as_deref().map(str::len), Some(64), "{}", m.id);
            assert!(m.file.ends_with(".gguf"));
        }
        let e2b = find("gemma-4-e2b").unwrap();
        assert_eq!(e2b.url("https://huggingface.co/"), "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf");
        assert!(find("nope").is_none());
    }

    #[test]
    fn sizes_read_like_download_pages() {
        assert_eq!(human(2_841_481_184), "2.84 GB");
        assert_eq!(human(557_368_096), "557.4 MB");
        assert_eq!(human(12), "12 bytes");
        assert!(default_for(Device::Desktop).consent_text("https://h").iter().any(|l| l.contains("4.59 GB")));
    }
}
