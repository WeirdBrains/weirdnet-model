//! Can this CPU run the llama.cpp compiled into this build?
//!
//! llama.cpp's build turns on AVX, AVX2, BMI2, FMA and F16C by default. A static build made that
//! way dies with an illegal instruction on x86 CPUs without them (most made before 2013) the first
//! time ggml runs. [`check`] compares what the build contains (worked out by `build.rs`) with what
//! this CPU has, and [`LlamaEngine::load`](crate::llama::LlamaEngine) calls it before any llama.cpp
//! code runs, so such a CPU gets [`Error::UnsupportedCpu`] instead of a crash.
//!
//! With the `cpu-dispatch` feature ggml is built for every CPU level and picks the best one at run
//! time, so nothing is required and [`check`] always passes. ARM builds use the target's baseline
//! (armv8-a on Linux, the Apple M1 level on macOS), which every CPU of that target has.

use crate::{Error, Result};

/// x86 instructions this build's llama.cpp contains (`is_x86_feature_detected!` names). Empty
/// without the `llama` feature, with `cpu-dispatch`, and on non-x86 targets.
pub fn required() -> Vec<&'static str> {
    env!("WEIRDNET_MODEL_GGML_X86").split(',').filter(|s| !s.is_empty()).collect()
}

/// The [`required`] instructions this CPU lacks (empty when local inference can run).
pub fn missing() -> Vec<&'static str> {
    missing_from(&required(), has)
}

/// `Ok` when this CPU can run the compiled-in llama.cpp, else [`Error::UnsupportedCpu`]. Cheap
/// (the detection is cached), so apps can call it to decide whether to offer local models.
pub fn check() -> Result<()> {
    let missing = missing();
    if missing.is_empty() { Ok(()) } else { Err(Error::UnsupportedCpu { missing: missing.into_iter().map(String::from).collect() }) }
}

fn missing_from<'a>(required: &[&'a str], has: impl Fn(&str) -> bool) -> Vec<&'a str> {
    required.iter().copied().filter(|f| !has(f)).collect()
}

/// Whether this CPU (and OS, for the AVX register state) supports `feature`. Unknown names count
/// as missing.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn has(feature: &str) -> bool {
    match feature {
        "sse4.2" => std::is_x86_feature_detected!("sse4.2"),
        "avx" => std::is_x86_feature_detected!("avx"),
        "avx2" => std::is_x86_feature_detected!("avx2"),
        "bmi2" => std::is_x86_feature_detected!("bmi2"),
        "fma" => std::is_x86_feature_detected!("fma"),
        "f16c" => std::is_x86_feature_detected!("f16c"),
        "avxvnni" => std::is_x86_feature_detected!("avxvnni"),
        "avx512f" => std::is_x86_feature_detected!("avx512f"),
        "avx512cd" => std::is_x86_feature_detected!("avx512cd"),
        "avx512vl" => std::is_x86_feature_detected!("avx512vl"),
        "avx512dq" => std::is_x86_feature_detected!("avx512dq"),
        "avx512bw" => std::is_x86_feature_detected!("avx512bw"),
        "avx512vbmi" => std::is_x86_feature_detected!("avx512vbmi"),
        "avx512vnni" => std::is_x86_feature_detected!("avx512vnni"),
        "avx512bf16" => std::is_x86_feature_detected!("avx512bf16"),
        _ => false,
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn has(_feature: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What llama.cpp's default x86 build needs.
    const DEFAULT_BUILD: &[&str] = &["avx", "avx2", "bmi2", "f16c", "fma", "sse4.2"];

    #[test]
    fn a_nehalem_lacks_what_the_default_build_needs() {
        let nehalem = |f: &str| f == "sse4.2";
        assert_eq!(missing_from(DEFAULT_BUILD, nehalem), ["avx", "avx2", "bmi2", "f16c", "fma"]);
        let sandy_bridge = |f: &str| matches!(f, "sse4.2" | "avx");
        assert_eq!(missing_from(DEFAULT_BUILD, sandy_bridge), ["avx2", "bmi2", "f16c", "fma"]);
    }

    #[test]
    fn a_haswell_has_everything_the_default_build_needs() {
        let haswell = |f: &str| matches!(f, "sse4.2" | "avx" | "avx2" | "bmi2" | "fma" | "f16c");
        assert!(missing_from(DEFAULT_BUILD, haswell).is_empty());
    }

    #[test]
    fn unsupported_cpu_names_the_missing_instructions() {
        let e = Error::UnsupportedCpu { missing: vec!["avx2".into(), "fma".into()] };
        let s = e.to_string();
        assert!(s.contains("avx2, fma"), "{s}");
    }

    #[test]
    fn the_build_machine_can_run_its_own_build() {
        // Tests run where they were built, so a mismatch here is a build.rs bug.
        assert_eq!(missing(), Vec::<&str>::new(), "required {:?}", required());
        check().unwrap();
    }

    #[test]
    fn the_requirement_matches_the_features() {
        let r = required();
        if !cfg!(feature = "llama") || !cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
            assert!(r.is_empty(), "{r:?}");
        }
        for f in &r {
            assert!(!f.is_empty() && !f.contains(','), "{r:?}");
        }
    }
}
