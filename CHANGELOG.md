# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-18

### Added

- `examples/model.rs`: `fetch`, `chat` (with timings) and `cpu` commands.
- `cpu` module: `cpu::check()`, `cpu::required()` and `cpu::missing()` say whether this CPU can run
  the compiled-in llama.cpp. `build.rs` works out which x86 instructions the build contains.
- `cpu-dispatch` feature: llama.cpp built for every CPU level (`GGML_BACKEND_DL` and
  `GGML_CPU_ALL_VARIANTS` through llama-cpp-2's `dynamic-backends`) and chosen at run time, with
  `llama::backends_dir()` and `llama::set_backends_dir()`. Apps ship the llama.cpp shared libraries;
  the README lists them.
- `Error::UnsupportedCpu` and `Error::NoCpuBackend`. Adding variants to `Error` is a breaking
  change.

### Changed

- `LlamaEngine::load` checks the CPU before any llama.cpp code runs. On a CPU without the
  instructions the build uses (AVX, AVX2, BMI2, FMA and F16C by default, so most x86 CPUs from
  before 2013) it returns `Error::UnsupportedCpu` instead of crashing with an illegal instruction.
  `llama::describe()` says `llama.cpp (unsupported CPU)` there.

### Fixed

- README named the cache override `WEIRDNET_MODEL_DIR`; the variable is `WEIRDNET_MODELS_DIR`.

## [0.1.0] - 2026-11-18

### Added

- Model catalog with Gemma 4 E2B and E4B GGUF defaults, a model store, and consent-gated Hugging Face downloads with resume, progress and SHA-256 verification.
- Hugging Face repo listing for GGUF files.
- Chat prompt rendering through the model's chat template, and a streaming tool-call parser.
- Worker thread with streaming generation and cancel, and an `Engine` trait with a mock for tests.
- `llama` feature (llama.cpp), `vulkan` and `metal` GPU features, and `agent` feature implementing `weirdnet_agent::ModelBackend`.
