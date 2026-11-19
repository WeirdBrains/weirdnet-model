# weirdnet-model

Local models for WeirdBrains apps: download a GGUF from Hugging Face (only after consent, with
resume and a SHA-256 check), load it with llama.cpp, and stream tokens from a worker thread you can
cancel. App-agnostic; the model is optional at build time.

- `catalog`: Gemma 4 E2B (phones, small laptops) and E4B (desktop) GGUF defaults, with repo, file,
  size and checksum. `catalog::default_for(Device)` picks one.
- `ModelStore`, `download`: a cache directory (`WEIRDNET_MODELS_DIR` overrides it), a `Consent` the
  user must give for exactly that model, `Progress` callbacks, `.part` resume and verification.
- `hf`: list a repo's GGUF files and sizes for a model picker.
- `prompt`: chat turns rendered through the model's own chat template, and a streaming parser that
  turns `<tool_call>` tags into tool calls.
- `Worker`: one thread owns the engine; `Worker::generate` returns a `Generation` that streams
  `Chunk`s and can be cancelled from any thread. The `Engine` trait lets tests use a mock.
- `llama` feature: `llama::LlamaEngine` (llama.cpp, CPU). `vulkan` and `metal` add GPU offload.
  Building needs cmake and a C++ compiler.
- `cpu-dispatch` feature: llama.cpp built for every CPU level and picked at run time (see below).
- `cpu`: whether this CPU can run the compiled-in llama.cpp. Loading a model checks it first.
- `agent` feature: `agent::LocalModel` implements `weirdnet_agent::ModelBackend`, so the agent
  loop can run on a local model with streamed text and parsed tool calls.

```toml
weirdnet-model = { version = "0.2", features = ["llama", "agent"] }
```

Tests never download a model. `WEIRDNET_TEST_GGUF=/path/model.gguf cargo test --features llama -- --ignored`
runs the real-model check.

## CPU support

llama.cpp's build turns on AVX, AVX2, BMI2, FMA and F16C by default, so a plain `llama` build runs
only on x86 CPUs that have them (Intel Haswell and AMD Excavator, 2013 to 2015, and later). Two
things keep older CPUs from crashing with an illegal instruction:

- **The guard, always on.** `build.rs` works out which instructions the linked llama.cpp contains
  (from the target features, `target-cpu=native`, `GGML_*` environment variables and
  `SOURCE_DATE_EPOCH`, the way `llama-cpp-sys-2` configures it). `LlamaEngine::load` calls
  `cpu::check()` before any llama.cpp code runs and returns `Error::UnsupportedCpu { missing }` on a
  CPU without them. Apps can call `cpu::check()` themselves to hide local models, or to offer
  another machine instead. `examples/model.rs cpu` prints the result.
- **`cpu-dispatch`, for release binaries that should run local models on any CPU.** It enables
  llama-cpp-2's `dynamic-backends` (`GGML_BACKEND_DL` plus `GGML_CPU_ALL_VARIANTS`): ggml is built
  once per CPU level (x86: baseline x86-64, SSE4.2, Sandy Bridge, Haswell, AVX-512 levels up to
  Sapphire Rapids; ARM: armv8.0 to armv9.2, Apple M1 to M4) and asks each library at load time
  whether this CPU can run it, keeping the best. Speed on a modern CPU is the same as the static
  build, because the same AVX2 code runs.

`cpu-dispatch` makes llama.cpp shared libraries that the app has to ship:

- `libllama`, `libllama-common`, `libggml`, `libggml-base` (`.so`, `.dylib` or `.dll`), which the
  executable links against. On Linux and macOS, point the executable's rpath at them (for example
  `-C link-arg=-Wl,-rpath,$ORIGIN` on Linux); on Windows put the DLLs next to the `.exe`. Without
  them the app does not start at all, not just local models.
- one `libggml-cpu-<level>` library per CPU level (`.so` on Linux and macOS, `.dll` on Windows),
  plus `libggml-vulkan` or `libggml-metal` with those features. They load from
  `llama::backends_dir()`: the directory given to `llama::set_backends_dir`, else
  `WEIRDNET_GGML_BACKENDS_DIR`, else the executable's directory (a development build falls back to
  where `llama-cpp-sys-2` built them). The working directory is never searched. If none loads,
  loading a model fails with `Error::NoCpuBackend`.

On x86-64 Linux (llama-cpp-sys-2 0.1.161, release, stripped) that is a 4.3 MB executable plus four
shared libraries (11 MB) and 14 CPU libraries (20 MB), 35 MB in all, against 9.4 MB for the static
example. Ship the versioned names the executable asks for (`libllama.so.0` and so on).

`llama-cpp-sys-2` 0.1.161's shared-library build does not work with Rust 1.100's build directory
layout (its build script fails with "No such file or directory"), so build `cpu-dispatch` with Rust
1.99 for now. A second build in the same target directory can also fail with "File exists" on the
library symlinks it left there; deleting `target/<profile>/{,deps/,examples/}lib*.so*` symlinks
between builds avoids it. The static `llama` build has neither problem.

Without `cpu-dispatch`, a static build can still run everywhere at a speed cost by building ggml
for an older level, for example `GGML_AVX=OFF GGML_AVX2=OFF GGML_BMI2=OFF GGML_FMA=OFF
GGML_F16C=OFF` in the build environment (SSE4.2, Nehalem and later). The guard follows those
variables. ARM builds need nothing: `llama-cpp-sys-2` builds for armv8-a on Linux, and macOS uses
the compiler's Apple M1 baseline. Do not use `target-cpu=native` for anything you distribute.

## License

BSD-3-Clause.
