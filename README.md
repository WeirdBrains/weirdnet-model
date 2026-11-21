# weirdnet-model

Local models for [WeirdBrains](https://weirdbrains.com) apps: download a GGUF from Hugging Face (only after consent, with
resume and a SHA-256 check), load it with llama.cpp, and stream tokens from a worker thread you can
cancel. App-agnostic; the model is optional at build time.

- `catalog`: Gemma 4 E2B (phones, small laptops) and E4B (desktop) GGUF defaults, with repo, file,
  size and checksum. `catalog::default_for(Device)` picks one.
- `ModelStore`, `download`, `download_parts`: a cache directory (`WEIRDNET_MODELS_DIR` overrides
  it), a `Consent` the user must give for exactly that model, `Progress` callbacks, `.part` resume
  and verification, split models part by part; `ModelStore::list` and `remove_local` for what is
  on disk.
- `hf`: search the Hub for GGUF repos, list a repo's models (split GGUFs grouped into one model,
  quantization read from the file name) and read a file's header for KV sizing. `HfToken` for
  gated and private repos, read on the downloading node only.
- `fit`: pick the largest file whose weights plus KV cache fit the node's memory budget.
- `prompt`: chat turns rendered through the model's own chat template, and a streaming parser that
  turns `<tool_call>` tags into tool calls.
- `Worker`: one thread owns the engine; `Worker::generate` returns a `Generation` that streams
  `Chunk`s and can be cancelled from any thread. The `Engine` trait lets tests use a mock.
- `llama` feature: `llama::LlamaEngine` (llama.cpp, CPU). `vulkan` and `metal` add GPU offload.
  Building needs cmake and a C++ compiler.
- `cpu-dispatch` feature: llama.cpp built for every CPU level and picked at run time (see below).
- `cpu`: whether this CPU can run the compiled-in llama.cpp. Loading a model checks it first.
- `agent` feature: `agent::LocalModel` implements `weirdnet_agent::ModelBackend`, so the agent
  loop can run on a local model with streamed text and parsed tool calls (`weirdnet-agent` 0.5;
  text only: messages with image or audio parts are refused).
- `hw`: hardware detection and model sizing (see below).

```toml
weirdnet-model = { version = "0.5", features = ["llama", "agent"] }
```

Tests never download a model. `WEIRDNET_TEST_GGUF=/path/model.gguf cargo test --features llama -- --ignored`
runs the real-model check.

## Hardware and sizing

`hw::detect()` reports the OS, CPU threads, system memory and GPUs: NVIDIA through `nvidia-smi`,
AMD through sysfs, and Apple silicon. Devices that share system memory (Apple, GB10, AMD APUs) are
marked unified. It never fails; a missing tool means fewer devices.

`hw::budget(&hw, memory_budget, reserve)` is the memory models may use: 85% of VRAM, or of
unified memory after an OS reserve; both can be overridden. `hw::plan(&hw, budget)` picks a main
chat model, a smaller stand-in and an embedding model from `hw::catalog()`, a tiered catalog with
each model's footprint; the Gemma entries are this crate's pinned `ModelSpec`s, so they download
through the consent gate.

```rust
let hw = weirdnet_model::hw::detect();
let plan = weirdnet_model::hw::plan(&hw, weirdnet_model::hw::budget(&hw, None, None));
```

## Hub search, fit and downloads

```rust
use weirdnet_model::{ModelStore, Consent, download_parts, fit, hf::Search, hw};

let store = ModelStore::from_env();
let hub = store.hub_client();
let hits = hub.search(&Search::new("qwen3 8b").with_limit(10))?;
let repo = &hits[0].id;
let models = hub.models(repo)?; // split GGUFs come back as one model with all its parts
let info = hub.gguf_info(repo, &models[0].first().name)?; // header only, through a Range request
let hw = hw::detect();
let opts = fit::FitOptions::default().with_slots(2).with_info(&info);
let best = fit::choose(&models, &hw::budget(&hw, None, None), &opts).expect("something fits");
let parts = best.model.specs(repo)?;
// Show the user what will be fetched, then:
let path = download_parts(&store, &parts, &Consent::given(&parts[0]), &mut |_| true)?;
```

Search and files come from the [Hugging Face Hub](https://huggingface.co). The footprint is the weights plus the KV cache for every slot at the chosen context, plus 10%
for compute buffers, the same as the tiered catalog's. Without a header the KV cache is guessed
high from the file size.

The Hugging Face token comes from this node only: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the
token file at `HF_TOKEN_PATH`, `HF_HOME/token` or `~/.cache/huggingface/token`. `ModelStore::from_env`
sends it only to the public Hub, never to a `WEIRDNET_HF_URL` mirror (`with_token` sends it to
any Hub you pick). Redirects to other hosts don't carry it, its `Debug` output is redacted, and it
has no serde support, so it can't travel to another node in a message.

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
