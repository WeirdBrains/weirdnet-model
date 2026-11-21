//! Which file of a repo to download: the largest quantization whose weights plus KV cache fit
//! the node's memory [`Budget`].
//!
//! The KV cache comes from the model's GGUF header ([`GgufInfo::kv_bytes_per_token`], read
//! with [`Hub::gguf_info`](crate::hf::Hub::gguf_info)) when known, else a conservative guess
//! from the file size ([`estimate_kv_bytes_per_token`]). The footprint is the same as the
//! tiered catalog's ([`CatalogEntry::footprint`](crate::hw::CatalogEntry::footprint)):
//! weights plus KV for every slot, plus 10% for compute buffers.

use serde::{Deserialize, Serialize};

use crate::gguf::GgufInfo;
use crate::hf::RepoModel;
use crate::hw::Budget;

/// What the model must hold besides its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FitOptions {
    /// Context per slot, in tokens (default 8192, the catalog's).
    pub context: u32,
    /// Concurrent requests (default 1).
    pub slots: u32,
    /// KV bytes per token, when known (default: guessed per file).
    pub kv_bytes_per_token: Option<u64>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self { context: 8192, slots: 1, kv_bytes_per_token: None }
    }
}

impl FitOptions {
    pub fn with_context(mut self, context: u32) -> Self {
        self.context = context;
        self
    }

    pub fn with_slots(mut self, slots: u32) -> Self {
        self.slots = slots.max(1);
        self
    }

    /// The model's KV shape from its header. A trained context shorter than `context` caps it.
    pub fn with_info(mut self, info: &GgufInfo) -> Self {
        self.kv_bytes_per_token = Some(info.kv_bytes_per_token());
        if let Some(max) = info.context_length.and_then(|c| u32::try_from(c).ok()).filter(|c| *c > 0) {
            self.context = self.context.min(max);
        }
        self
    }

    pub fn with_kv_bytes_per_token(mut self, bytes: u64) -> Self {
        self.kv_bytes_per_token = Some(bytes);
        self
    }
}

/// Where a [`Fit`]'s KV figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KvSource {
    /// The model's header.
    Header,
    /// [`estimate_kv_bytes_per_token`].
    Estimate,
}

/// One model measured against the budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Fit {
    pub model: RepoModel,
    pub kv_bytes: u64,
    pub kv_source: KvSource,
    /// Weights plus KV, plus 10%.
    pub footprint: u64,
    pub fits: bool,
}

/// A guess when the header is unknown: one byte of KV per token for every 32 KiB of weights, and
/// at least 128 KiB per token (1 GiB at 8192 tokens). It errs high: the catalog's Qwen3 8B to
/// 32B (Q4_K_M) need one byte per 34,000 to 75,000 bytes of weights, and its small models
/// (SmolLM2 135M, Qwen3 Embedding 0.6B) at most 112 KiB per token. Read the header when you can.
pub fn estimate_kv_bytes_per_token(weights: u64) -> u64 {
    (weights / 32_768).max(128 << 10)
}

/// The footprint of `model` under `opts`.
pub fn measure(model: &RepoModel, budget: &Budget, opts: &FitOptions) -> Fit {
    let (per_token, kv_source) = match opts.kv_bytes_per_token {
        Some(b) => (b, KvSource::Header),
        None => (estimate_kv_bytes_per_token(model.size), KvSource::Estimate),
    };
    let kv_bytes = per_token.saturating_mul(opts.context as u64).saturating_mul(opts.slots.max(1) as u64);
    let total = model.size.saturating_add(kv_bytes);
    let footprint = total.saturating_add(total / 10);
    Fit { model: model.clone(), kv_bytes, kv_source, footprint, fits: model.complete && footprint <= budget.usable_bytes }
}

/// Every model measured, largest first.
pub fn rank(models: &[RepoModel], budget: &Budget, opts: &FitOptions) -> Vec<Fit> {
    let mut out: Vec<Fit> = models.iter().map(|m| measure(m, budget, opts)).collect();
    out.sort_by(|a, b| b.model.size.cmp(&a.model.size).then_with(|| a.model.id.cmp(&b.model.id)));
    out
}

/// The largest complete model that fits (the best quality the node can hold), if any.
pub fn choose(models: &[RepoModel], budget: &Budget, opts: &FitOptions) -> Option<Fit> {
    rank(models, budget, opts).into_iter().find(|f| f.fits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::{RepoFile, group_files};
    use crate::hw::GIB;

    fn models() -> Vec<RepoModel> {
        let f = |name: &str, size: u64| RepoFile { name: name.into(), size, sha256: None };
        group_files(&[
            f("m-Q4_K_M.gguf", 5 * GIB),
            f("m-Q8_0.gguf", 9 * GIB),
            f("BF16/m-BF16-00001-of-00002.gguf", 8 * GIB),
            f("BF16/m-BF16-00002-of-00002.gguf", 8 * GIB),
            f("F32/m-F32-00001-of-00003.gguf", GIB),
        ])
    }

    fn budget(usable: u64) -> Budget {
        Budget { pool_bytes: usable, reserve_bytes: 0, usable_bytes: usable }
    }

    #[test]
    fn picks_the_largest_that_fits() {
        // Qwen3 8B's KV: 147456 bytes per token; 8192 tokens is 1.125 GiB.
        let opts = FitOptions::default().with_kv_bytes_per_token(147_456);
        assert_eq!(choose(&models(), &budget(12 * GIB), &opts).unwrap().model.id, "m-Q8_0");
        assert_eq!(choose(&models(), &budget(20 * GIB), &opts).unwrap().model.id, "m-BF16");
        assert_eq!(choose(&models(), &budget(7 * GIB), &opts).unwrap().model.id, "m-Q4_K_M");
        assert!(choose(&models(), &budget(4 * GIB), &opts).is_none());
        // More slots need more KV.
        assert_eq!(choose(&models(), &budget(12 * GIB), &opts.with_slots(4)).unwrap().model.id, "m-Q4_K_M");
    }

    #[test]
    fn incomplete_splits_never_fit_and_ranks_are_largest_first() {
        let ranked = rank(&models(), &budget(1000 * GIB), &FitOptions::default());
        assert_eq!(
            ranked.iter().map(|f| (f.model.id.as_str(), f.fits)).collect::<Vec<_>>(),
            [("m-BF16", true), ("m-Q8_0", true), ("m-Q4_K_M", true), ("m-F32", false)]
        );
        assert!(ranked.iter().all(|f| f.kv_source == KvSource::Estimate));
    }

    #[test]
    fn the_header_caps_the_context_and_footprints_match_the_catalog() {
        let info = GgufInfo { kv_heads_total: 8, key_length: 64, value_length: 64, context_length: Some(4096), ..Default::default() };
        let opts = FitOptions::default().with_info(&info);
        assert_eq!((opts.context, opts.kv_bytes_per_token), (4096, Some(2048)));
        let e = crate::hw::find("qwen3-8b").unwrap();
        let m = RepoModel { id: "q".into(), quant: None, parts: vec![], size: e.spec.size, complete: true };
        let fit = measure(&m, &budget(u64::MAX), &FitOptions::default().with_kv_bytes_per_token(e.kv_bytes_per_token).with_slots(2));
        assert_eq!(fit.footprint, e.footprint(2));
        for e in crate::hw::catalog() {
            assert!(estimate_kv_bytes_per_token(e.spec.size) >= e.kv_bytes_per_token, "{}", e.id());
        }
    }
}
