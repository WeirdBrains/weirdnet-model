//! Hardware detection and model sizing.
//!
//! - [`detect()`]: GPUs (NVIDIA through `nvidia-smi`, AMD through sysfs, Apple silicon), CPU
//!   threads and system memory. Never fails; missing tools mean fewer devices.
//! - [`budget()`]: the memory models may use (85% of VRAM, or of unified memory after an OS
//!   reserve), with overrides.
//! - [`catalog()`] and [`find()`]: a small tiered catalog of chat, embedding and speech-to-text
//!   ([`Purpose::Transcribe`]) models with their memory footprints; entries are
//!   [`ModelSpec`](crate::ModelSpec)s where this crate pins the file (and the projector a model
//!   needs, [`CatalogEntry::projector`]), so they download through the consent gate.
//! - [`plan()`]: the main chat model, a stand-in and an embedding model that fit the budget;
//!   [`transcriber()`]: the speech-to-text model that fits, for apps that want one.

mod detect;
mod sizing;

pub use detect::{Accel, Gpu, Hardware, detect, parse_meminfo_total, parse_nvidia_smi};
pub use sizing::{Budget, CatalogEntry, Plan, Purpose, VOXTRAL_MINI_3B, budget, by_purpose, catalog, find, plan, transcriber};

/// 1 GiB.
pub const GIB: u64 = 1 << 30;

/// Parses sizes like `12GiB`, `512MiB`, `100GB`, `8G` or a plain byte count.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let num: f64 = num.parse().ok()?;
    let mult: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kib" => 1024.0,
        "kb" => 1e3,
        "m" | "mib" => 1024.0 * 1024.0,
        "mb" => 1e6,
        "g" | "gib" => GIB as f64,
        "gb" => 1e9,
        "t" | "tib" => GIB as f64 * 1024.0,
        "tb" => 1e12,
        _ => return None,
    };
    Some((num * mult) as u64)
}

/// `11.8 GiB` (binary, like memory budgets).
pub fn human_gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("12GiB"), Some(12 * GIB));
        assert_eq!(parse_size("100GB"), Some(100_000_000_000));
        assert_eq!(parse_size("512 MiB"), Some(512 << 20));
        assert_eq!(parse_size("1.5G"), Some(GIB + GIB / 2));
        assert_eq!(parse_size("42"), Some(42));
        assert_eq!(parse_size("lots"), None);
    }
}
