//! The llama.cpp [`Engine`]: loads a GGUF, renders the model's own chat template, and streams
//! sampled text. Without a usable GPU backend it runs on the CPU. Adapted from Mouser's
//! `mouser-llm` (copied, BSD-3-Clause here).

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;

use crate::prompt;
use crate::worker::{Engine, GenerateRequest, Stats, StopCause};
use crate::{Error, Result};

pub const DEFAULT_CONTEXT: u32 = 8192;
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
const BATCH: usize = 512;

/// The process-wide llama.cpp backend. Checks the CPU first ([`crate::cpu::check`]): on a CPU
/// without the instructions this build's ggml was compiled with, llama.cpp must never run.
fn backend() -> Result<&'static LlamaBackend> {
    crate::cpu::check()?;
    static B: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    #[cfg(feature = "cpu-dispatch")]
    static NO_CPU: OnceLock<std::path::PathBuf> = OnceLock::new();
    B.get_or_init(|| {
        #[cfg(feature = "cpu-dispatch")]
        if let Err(dir) = dispatch::load() {
            NO_CPU.set(dir).ok();
            return Err(String::new());
        }
        LlamaBackend::init()
            .map(|mut b| {
                b.void_logs();
                b
            })
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| {
        #[cfg(feature = "cpu-dispatch")]
        if let Some(dir) = NO_CPU.get() {
            return Error::NoCpuBackend { dir: dir.clone() };
        }
        Error::Model(format!("llama.cpp: {e}"))
    })
}

#[cfg(feature = "cpu-dispatch")]
pub use dispatch::{backends_dir, set_backends_dir};

/// `cpu-dispatch`: ggml's backends (one CPU library per CPU level, plus GPU ones) are shared
/// libraries loaded at run time; ggml asks each CPU library whether this CPU can run it and keeps
/// the best. Only the chosen directory is searched, never the working directory.
#[cfg(feature = "cpu-dispatch")]
mod dispatch {
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    use llama_cpp_2::{LlamaBackendDeviceType, list_llama_ggml_backend_devices, llama_backend};

    static DIR: OnceLock<PathBuf> = OnceLock::new();

    /// Load the ggml backend libraries from `dir` instead of the default (see [`backends_dir`]).
    /// Call it before the first model loads (the backends load once per process); returns `false`
    /// when a directory was already set.
    pub fn set_backends_dir(dir: PathBuf) -> bool {
        DIR.set(dir).is_ok()
    }

    /// Where the ggml backend libraries (`libggml-cpu-*.so`, `ggml-cpu-*.dll`, …) are loaded from:
    /// the directory given to [`set_backends_dir`], else `WEIRDNET_GGML_BACKENDS_DIR`, else the
    /// executable's directory. A development build whose executable's directory has none falls
    /// back to the directory llama-cpp-sys-2 built them in, so `cargo test` and `cargo run` work.
    pub fn backends_dir() -> PathBuf {
        if let Some(d) = DIR.get() {
            return d.clone();
        }
        if let Some(d) = std::env::var_os("WEIRDNET_GGML_BACKENDS_DIR").filter(|d| !d.is_empty()) {
            return d.into();
        }
        std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)).unwrap_or_default()
    }

    fn has_cpu_library(dir: &Path) -> bool {
        std::fs::read_dir(dir).map(|it| it.flatten().any(|e| is_cpu_library(&e.file_name().to_string_lossy()))).unwrap_or(false)
    }

    fn is_cpu_library(name: &str) -> bool {
        let name = name.strip_prefix("lib").unwrap_or(name);
        name.starts_with("ggml-cpu") && (name.ends_with(".so") || name.ends_with(".dll"))
    }

    fn has_cpu_device() -> bool {
        list_llama_ggml_backend_devices().iter().any(|d| d.device_type == LlamaBackendDeviceType::Cpu)
    }

    /// Loads the backends once; `Err(dir)` when no CPU backend could be loaded from `dir`.
    pub(super) fn load() -> Result<(), PathBuf> {
        let dir = backends_dir();
        if has_cpu_library(&dir) {
            llama_backend::load_backends_from_path(&dir);
        } else if DIR.get().is_none() && std::env::var_os("WEIRDNET_GGML_BACKENDS_DIR").is_none() {
            // Not shipped next to the executable: a development build run from the target dir.
            llama_backend::load_backends();
        }
        if has_cpu_device() { Ok(()) } else { Err(dir) }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn cpu_libraries_are_recognised_on_every_platform() {
            for n in ["libggml-cpu-haswell.so", "libggml-cpu-x64.so", "ggml-cpu-sse42.dll", "libggml-cpu-apple_m1.so", "libggml-cpu.so"] {
                assert!(is_cpu_library(n), "{n}");
            }
            for n in ["libggml-base.so", "libggml-vulkan.so", "libllama.so", "libggml-cpu-x64.so.txt", "ggml.dll"] {
                assert!(!is_cpu_library(n), "{n}");
            }
        }

        #[test]
        fn a_cpu_backend_loads_in_a_development_build() {
            super::super::backend().unwrap();
            assert!(has_cpu_device());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaOptions {
    /// Layers to offload to the GPU (all of them by default; ignored without a GPU backend).
    pub gpu_layers: u32,
    /// Context length in tokens.
    pub context: u32,
}

impl Default for LlamaOptions {
    fn default() -> Self {
        Self { gpu_layers: 999, context: DEFAULT_CONTEXT }
    }
}

pub struct LlamaEngine {
    model: LlamaModel,
    opts: LlamaOptions,
}

/// `llama.cpp (Vulkan)`, `(Metal)` or `(CPU)`, for a settings line.
pub fn describe() -> String {
    if crate::cpu::check().is_err() {
        return "llama.cpp (unsupported CPU)".into();
    }
    let gpu = backend().map(|b| b.supports_gpu_offload()).unwrap_or(false);
    format!("llama.cpp ({})", if gpu { if cfg!(target_os = "macos") { "Metal" } else { "GPU" } } else { "CPU" })
}

fn model_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!("{what}: {e}"))
}

impl LlamaEngine {
    /// Loads `path`. Fails with [`Error::UnsupportedCpu`] before touching llama.cpp when this CPU
    /// can't run it.
    pub fn load(path: &Path, opts: LlamaOptions) -> Result<Self> {
        let b = backend()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(opts.gpu_layers);
        let model = LlamaModel::load_from_file(b, path, &params).map_err(|e| model_err(&format!("loading {}", path.display()), e))?;
        Ok(Self { model, opts })
    }

    /// What this engine runs the model with, as loaded: the context in tokens
    /// ([`LlamaOptions::context`]). It reads text only, so no image or audio figures.
    pub fn limits(&self) -> crate::limits::ModelLimits {
        crate::limits::ModelLimits::new().with_context_tokens(self.opts.context)
    }

    /// A [`Worker`](crate::Worker) loader for `path`, for `Worker::spawn`.
    pub fn loader(path: std::path::PathBuf, opts: LlamaOptions) -> impl FnMut() -> Result<Box<dyn Engine>> + Send + 'static {
        move || Ok(Box::new(LlamaEngine::load(&path, opts)?) as Box<dyn Engine>)
    }
}

impl Engine for LlamaEngine {
    fn generate(&mut self, req: &GenerateRequest, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Stats> {
        let b = backend()?;
        let model = &self.model;
        let prompt = match model.meta_val_str("tokenizer.chat_template") {
            Ok(t) => prompt::render(&t, &req.messages, "").unwrap_or_else(|_| prompt::render_fallback(&req.messages)),
            Err(_) => prompt::render_fallback(&req.messages),
        };
        let vocab = model.vocab();
        let mut tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.is_empty() {
            return Err(Error::Model("empty prompt".into()));
        }
        let n_ctx = self.opts.context;
        let max_new = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as usize;
        if tokens.len() + 8 >= n_ctx as usize {
            // Keep the end of an over-long prompt.
            tokens.drain(..tokens.len() + 8 + max_new.min(n_ctx as usize / 4) - n_ctx as usize);
        }
        let cparams = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_batch(BATCH as u32);
        let mut ctx = model.new_context(b, cparams).map_err(|e| model_err("llama.cpp context", e))?;

        let mut batch = LlamaBatch::new(BATCH, 1);
        let mut pos = 0i32;
        for chunk in tokens.chunks(BATCH) {
            batch.clear();
            for (i, t) in chunk.iter().enumerate() {
                let last = pos as usize + i + 1 == tokens.len();
                batch.add(*t, pos + i as i32, &[0], last).map_err(|e| model_err("llama.cpp batch", e))?;
            }
            ctx.decode(&mut batch).map_err(|e| model_err("llama.cpp decode", e))?;
            pos += chunk.len() as i32;
        }

        let temp = req.temperature.unwrap_or(1.0);
        let mut sampler = if temp <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(64),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::temp(temp),
                LlamaSampler::dist(0x7765_6972),
            ])
        };
        let mut pending: Vec<u8> = Vec::new();
        let mut produced = 0u64;
        let mut stop = StopCause::End;
        let mut idx = batch.n_tokens() - 1;
        loop {
            if produced as usize >= max_new {
                stop = StopCause::MaxTokens;
                break;
            }
            let tok = sampler.sample(&ctx, idx);
            sampler.accept(tok);
            if vocab.is_eog(tok) {
                break;
            }
            pending.extend(vocab.token_to_piece(tok, false, None));
            // Emit only complete UTF-8.
            let valid = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let s = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                if !on_text(&s) {
                    stop = StopCause::Cancelled;
                    break;
                }
            }
            produced += 1;
            batch.clear();
            batch.add(tok, pos, &[0], true).map_err(|e| model_err("llama.cpp batch", e))?;
            pos += 1;
            idx = 0;
            if pos as u32 >= n_ctx {
                stop = StopCause::MaxTokens;
                break;
            }
            ctx.decode(&mut batch).map_err(|e| model_err("llama.cpp decode", e))?;
        }
        if stop != StopCause::Cancelled && !pending.is_empty() {
            on_text(&String::from_utf8_lossy(&pending));
        }
        Ok(Stats { prompt_tokens: tokens.len() as u64, generated_tokens: produced, stop })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a real GGUF: `WEIRDNET_TEST_GGUF=/path/model.gguf cargo test --features llama -- --ignored`.
    #[test]
    #[ignore = "needs a real model file"]
    fn a_real_model_answers_and_cancels() {
        let path = std::env::var("WEIRDNET_TEST_GGUF").expect("set WEIRDNET_TEST_GGUF");
        let worker = crate::Worker::spawn(LlamaEngine::loader(path.into(), LlamaOptions { context: 2048, ..Default::default() }));
        let mut req = GenerateRequest::new(vec![crate::ChatMessage::new(crate::ChatRole::User, "Reply with one short sentence about the sea.")]);
        req.max_tokens = Some(48);
        req.temperature = Some(0.0);
        let (text, stats) = worker.generate(req.clone()).collect_blocking().unwrap();
        assert!(!text.trim().is_empty());
        assert!(stats.generated_tokens > 0);
        req.max_tokens = Some(512);
        let mut g = worker.generate(req);
        assert!(matches!(g.blocking_recv(), Some(crate::Chunk::Text(_))));
        g.cancel();
        let mut last = None;
        while let Some(c) = g.blocking_recv() {
            last = Some(c);
        }
        assert!(matches!(last, Some(crate::Chunk::Done(Stats { stop: StopCause::Cancelled, .. }))));
    }

    /// Needs a real GGUF, as above.
    #[test]
    #[ignore = "needs a real model file"]
    fn a_real_model_reports_the_context_it_runs_with() {
        let path = std::path::PathBuf::from(std::env::var("WEIRDNET_TEST_GGUF").expect("set WEIRDNET_TEST_GGUF"));
        let engine = LlamaEngine::load(&path, LlamaOptions { context: 1024, ..Default::default() }).unwrap();
        assert_eq!(engine.limits(), crate::limits::ModelLimits::new().with_context_tokens(1024));
        let record = crate::limits::measure(&path, None, engine.limits().context_tokens).unwrap();
        assert_eq!(record.limits.context_tokens, Some(1024));
        assert!(record.trained_context.is_some_and(|t| t > 0), "the header's trained context is kept for reference");
    }

    #[test]
    fn the_backend_describes_itself() {
        assert!(describe().starts_with("llama.cpp ("));
    }
}
