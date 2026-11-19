//! Works out which x86 instructions the statically linked llama.cpp (ggml) will contain, so the
//! crate can refuse local inference on a CPU without them instead of dying on an illegal
//! instruction. The result goes to `src/cpu.rs` as `WEIRDNET_MODEL_GGML_X86` (comma-separated
//! `is_x86_feature_detected!` names).
//!
//! This mirrors how `llama-cpp-sys-2` (0.1.161) configures ggml's CMake build:
//! - `target-cpu=native` in RUSTFLAGS turns on `GGML_NATIVE` (`-march=native`): the build
//!   machine's own instructions.
//! - Otherwise `GGML_NATIVE` is off and each Rust target feature (`-C target-feature`,
//!   `target-cpu=<name>`) switches the matching `GGML_<ISA>` on. Every option left unset gets
//!   ggml's default, which is on for SSE4.2, AVX, AVX2, BMI2, FMA and F16C unless
//!   `SOURCE_DATE_EPOCH` is set (or CMake is cross-compiling), and `GGML_<ISA>` environment
//!   variables override those defaults.
//! - With `cpu-dispatch` ggml is built for every CPU level and picks one at run time, so
//!   nothing beyond the target's baseline is required.
//!
//! When in doubt this requires more, never less: a wrong guess refuses a CPU that would have
//! worked instead of crashing on one that would not.

use std::env;

/// ggml CMake option, the `is_x86_feature_detected!` names it compiles in, and whether ggml turns
/// it on by default (when `INS_ENB` is on).
const OPTIONS: &[(&str, &[&str], bool)] = &[
    ("GGML_SSE42", &["sse4.2"], true),
    ("GGML_AVX", &["avx"], true),
    ("GGML_AVX2", &["avx2"], true),
    ("GGML_BMI2", &["bmi2"], true),
    ("GGML_FMA", &["fma"], true),
    ("GGML_F16C", &["f16c"], true),
    ("GGML_AVX_VNNI", &["avxvnni"], false),
    ("GGML_AVX512", &["avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"], false),
    ("GGML_AVX512_VBMI", &["avx512vbmi"], false),
    ("GGML_AVX512_VNNI", &["avx512vnni"], false),
    ("GGML_AVX512_BF16", &["avx512bf16"], false),
];

/// Every name `src/cpu.rs` can check at run time.
const KNOWN: &[&str] = &[
    "sse4.2",
    "avx",
    "avx2",
    "bmi2",
    "fma",
    "f16c",
    "avxvnni",
    "avx512f",
    "avx512cd",
    "avx512vl",
    "avx512dq",
    "avx512bw",
    "avx512vbmi",
    "avx512vnni",
    "avx512bf16",
];

fn cmake_bool(v: &str) -> Option<bool> {
    match v.trim().to_ascii_uppercase().as_str() {
        "ON" | "1" | "TRUE" | "YES" | "Y" => Some(true),
        "OFF" | "0" | "FALSE" | "NO" | "N" | "IGNORE" | "NOTFOUND" | "" => Some(false),
        s if s.ends_with("-NOTFOUND") => Some(false),
        _ => None,
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    for (opt, _, _) in OPTIONS {
        println!("cargo:rerun-if-env-changed={opt}");
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mut need: Vec<&str> = Vec::new();
    if (arch == "x86_64" || arch == "x86") && env::var_os("CARGO_FEATURE_LLAMA").is_some() && env::var_os("CARGO_FEATURE_CPU_DISPATCH").is_none() {
        let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
        let native = rustflags.split('\x1f').any(|f| f.contains("target-cpu=native"));
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        // Instructions Rust itself was told it may use are compiled into ggml too.
        for f in target_features.split(',') {
            if let Some(k) = KNOWN.iter().find(|k| **k == f) {
                need.push(k);
            }
        }
        if native {
            // `-march=native`: whatever the build machine has (Rust is built for it too).
            need.extend(build_machine_features());
        } else {
            let ins_enb = env::var_os("SOURCE_DATE_EPOCH").is_none();
            let msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
            for (opt, names, default) in OPTIONS {
                let in_target = names.iter().any(|n| target_features.split(',').any(|f| f == *n));
                // An unparseable value is treated as on.
                let on = in_target || env::var(opt).ok().map(|v| cmake_bool(&v).unwrap_or(true)).unwrap_or(*default && ins_enb);
                if on {
                    need.extend_from_slice(names);
                    // MSVC's /arch:AVX2 and /arch:AVX512 imply FMA and F16C.
                    if msvc && (*opt == "GGML_AVX2" || *opt == "GGML_AVX512") {
                        need.extend_from_slice(&["fma", "f16c"]);
                    }
                }
            }
        }
    }
    need.sort_unstable();
    need.dedup();
    println!("cargo:rustc-env=WEIRDNET_MODEL_GGML_X86={}", need.join(","));
}

/// What `-march=native` would pick up here. Build scripts run on the build machine.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn build_machine_features() -> Vec<&'static str> {
    let mut v = Vec::new();
    macro_rules! detect {
        ($($n:tt),*) => {$( if std::is_x86_feature_detected!($n) { v.push($n); } )*};
    }
    detect!(
        "sse4.2",
        "avx",
        "avx2",
        "bmi2",
        "fma",
        "f16c",
        "avxvnni",
        "avx512f",
        "avx512cd",
        "avx512vl",
        "avx512dq",
        "avx512bw",
        "avx512vbmi",
        "avx512vnni",
        "avx512bf16"
    );
    v
}

/// Building on a non-x86 machine with `target-cpu=native` for x86 can't be read: require all.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn build_machine_features() -> Vec<&'static str> {
    KNOWN.to_vec()
}
