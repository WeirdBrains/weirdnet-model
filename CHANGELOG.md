# Changelog

## [Unreleased]

### Changed

- Metadata and docs scrub; no API changes.

## [0.4.2] - 2026-11-19

### Changed

- The `agent` feature depends on `weirdnet-agent` `"0.4"` (any 0.4.x, 0.4.1 and later included) instead of exactly `=0.4.0`, so agent patch releases no longer need a weirdnet-model release.

## [0.4.1] - 2026-11-19

### Fixed

- `hw::plan()`: the note when no catalog chat model fits no longer names an application command (`ymirctl`); it says to add a smaller GGUF model.

## [0.4.0] - 2026-11-19

### Added

- `hw` module (from Ymir's `ymir-hw`): `detect()` finds NVIDIA (`nvidia-smi`), AMD (sysfs) and Apple silicon GPUs, CPU threads and system memory (Linux, Android, macOS, BSD, Windows); `budget()` gives the memory models may use; `catalog()`/`find()` list tiered chat and embedding models with footprints; `plan()` picks a main model, a stand-in and an embedding model that fit.

### Changed

- **Breaking for `agent` users:** the `agent` feature now depends on `weirdnet-agent` `=0.4.0` (was `=0.3.0`). `ModelBackend` is unchanged; roles added later to the now non-exhaustive `Role` are not sent to the model.

## [0.3.0] - 2026-11-19

### Changed

- **Breaking for `agent` users:** the `agent` feature now depends on `weirdnet-agent` `=0.3.0` (was `=0.2.0`), so `LocalModel` implements the 0.3.0 `ModelBackend`. `ModelBackend` itself is unchanged and no source changes were needed.

## [0.2.1] - 2026-11-19

### Changed

- The `agent` feature now depends on `weirdnet-agent` `=0.2.0` (was `^0.1`). `ModelBackend` is unchanged, so apps on agent 0.2 can use `LocalModel`.

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
