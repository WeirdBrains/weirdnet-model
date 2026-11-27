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
- `hw`: hardware detection and model sizing, with a speech-to-text model (see below).
- `limits`: per-model media and request limits, read when a model loads and kept next to its
  files (see below).

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

## Speech to text

Each catalog entry has a `Purpose`, its role on a node: `chat`, `embedding`, `transcribe` or `speak`
(`Purpose::as_str` and `Purpose::parse` give the wire names). The `transcribe` entry is
[Voxtral Mini 3B](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF)
(`hw::VOXTRAL_MINI_3B`, Q4_K_M, Apache-2.0), which llama.cpp runs with its audio encoder:
`CatalogEntry::projector()` is that pinned `mmproj` file, downloaded through the same consent
gate and counted in `footprint`. Audio goes in as an `input_audio` part and the transcript comes
back as text, so a soundtrack can become text for models without audio.

`hw::plan` never proposes it, since transcription is optional; `hw::transcriber(&budget)` returns
it when it fits, and `hw::by_purpose(Purpose::Transcribe)` lists the entries.

## Text to speech

The `speak` role (`Purpose::Speak`) is text in, audio out. Each entry has a `SpeechProfile`
(`CatalogEntry::speech()`): its preset voices, whether it designs a voice from a description,
takes style instructions, clones a voice from a reference clip (and whether the clip's
transcript is used or required), speaks multi-speaker scripts, takes raw engine input or a
speed, streams; its languages, output formats and sample rate; and its limits. Every field is
cited to the model's card and its engine's documentation, or absent: nothing is defaulted, and a
model is never said to do what it doesn't.

| Id | Model | Engine | Does |
|---|---|---|---|
| `qwen3-tts-0.6b-customvoice` | [Qwen3-TTS 0.6B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | vLLM-Omni | nine preset voices |
| `qwen3-tts-0.6b-base` | [Qwen3-TTS 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | vLLM-Omni | voice cloning |
| `qwen3-tts-1.7b-customvoice` | [Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | vLLM-Omni | nine preset voices, style instructions |
| `qwen3-tts-1.7b-voicedesign` | [Qwen3-TTS 1.7B VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | vLLM-Omni | a voice from a description |
| `qwen3-tts-1.7b-base` | [Qwen3-TTS 1.7B Base GGUF](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF) | llama.cpp | voice cloning |
| `kokoro-82m` | [Kokoro 82M GGUF](https://huggingface.co/ggml-org/Kokoro-82M-GGUF) | llama.cpp | 54 preset voices, phonemes, speed |
| `voxcpm2` | [VoxCPM2 GGUF](https://huggingface.co/ggml-org/VoxCPM2-GGUF) | llama.cpp | voice design and cloning, 30 languages |
| `fish-audio-drama-3` | [Fish Audio Drama 3 GGUF](https://huggingface.co/ggml-org/drama-3-GGUF) | llama.cpp | scripts of up to 16 speakers |

GGUF entries (`CatalogEntry::is_gguf`) pin the model and its projector (`CatalogEntry::projector`,
the codec and voices), both counted in `footprint`. The others name the Hugging Face repo
vLLM-Omni serves and pin its main weights for sizing.

Fish Audio Drama 3 is under the [Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE)
(`speech::FISH_AUDIO_COMMUNITY`): `CatalogEntry::restricted()` is true, `hw::speaker` never
picks it, and `download` refuses its files (by id, or from its repos under any id) unless the
consent also accepts the license:

```rust
use weirdnet_model::{Consent, download, hw, speech::FISH_AUDIO_COMMUNITY};

let drama = hw::find(hw::FISH_AUDIO_DRAMA_3).unwrap();
let consent = Consent::given(&drama.spec).accepting_license(FISH_AUDIO_COMMUNITY);
```

`hw::plan` never proposes a speaker; `hw::speaker(&budget)` returns the largest unrestricted GGUF
one that fits, and `hw::by_purpose(Purpose::Speak)` lists them. A hand-placed GGUF speech model
gets its profile from its headers: `speech::from_gguf(model, projector)` (Qwen3-TTS, Kokoro with
the voices stored in its projector, VoxCPM2, Drama 3, as llama.cpp documents them), which
`limits::measure` puts in `LimitsRecord::speech` so it is kept with the model's limits. With the
`agent` feature `SpeechProfile` converts to and from `weirdnet_agent::speech::SpeechProfile`, with
the same form on the wire.

## Model limits

A client fitting images, audio or a video's frames to a model needs that model's limits, and
there are no defaults: what isn't known stays absent. `ModelLimits` holds the context the model
runs with on the node, images per request, what an image costs (a fixed number, or one token per
`image_patch` pixels square up to `image_tokens`), the longest image edge, audio length and cost
(per minute, and per piece for encoders that work on 30-second pieces), video length, bytes and
frames per minute,
and the largest request. `or` puts an operator's settings over measured ones, `tightest` merges
copies, and `image_cost` and `audio_cost` price one input. With the `agent` feature it converts
to and from `weirdnet_agent::MediaLimits` for catalog entries.

When a node loads a model, `limits::measure(model, projector, context)` reads the GGUF and its
projector: what the projector's header says images and audio cost (for Gemma 3, Qwen-VL,
Voxtral, Ultravox and Qwen2 audio projectors, as llama.cpp computes it) and the context passed in,
never the trained context. `ModelStore::record_limits` keeps the result next to the model's
files and `StoredModel::limits` reads it back, until the model's files change.

```rust
use weirdnet_model::{ModelStore, limits};

let store = ModelStore::from_env();
let model = &store.list()[0];
let first = store.dir().join(&model.files[0]);
let record = limits::measure(&first, Some(&store.dir().join("mmproj-model.gguf")), Some(8192))?;
store.record_limits(model, &record.with_engine("llama.cpp"))?;
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

## LoRA adapters

```rust
use weirdnet_model::{ModelStore, Consent, download_adapter};

let store = ModelStore::from_env();
let hub = store.hub_client();
let adapters = hub.adapters("owner/my-tune")?; // GGUF LoRA files and PEFT directories
let tune = &adapters[0]; // id, format, files, size, architecture, base_model, rank, alpha
// Show the user what will be fetched, then:
let path = download_adapter(&store, "owner/my-tune", tune, &Consent::given_adapter(tune), &mut |_| true)?;
for a in store.adapters() {
    println!("{} ({:?}) for {:?}", a.id, a.format, a.base_model);
}
```

A GGUF file is an adapter when its header says `general.type` = `adapter`; LoRA adapters also
have `adapter.type` = `lora` and `adapter.lora.alpha`, the keys
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `convert_lora_to_gguf.py` writes.
`gguf::read_header` reads any GGUF header (models and adapters; `read_info` still fails on
adapters, which have no blocks). A PEFT adapter is an `adapter_config.json` with `peft_type`
LORA next to `adapter_model.safetensors` (or `.bin`), the layout vLLM serves.

Adapters are never models: `Hub::models` reads the header of every file whose name hints at an
adapter (a `lora` or `adapter` word) and leaves real adapters out, so fits never rank one, and
`ModelStore::list` skips `adapters/` and hinted GGUFs whose header says adapter. Hub searches
leave adapter repos out unless `Search::with_adapters(true)`. Adapters download into
`adapters/<id>/` under the store with the same consent, resume and SHA-256 checks as models, plus
an `adapter.json` with the repo, format and base model (no paths).

## Your own model files

For weights a person brings themselves (a fine-tune made on another machine, a trainer's
output), `artifact::inspect` reads and hashes the files and `ModelStore::import` places them in
the store with a provenance record next to them.

```rust
use std::path::PathBuf;
use weirdnet_model::artifact::{self, ORIGIN_UPLOAD, Provenance};
use weirdnet_model::{Imported, ModelStore};

let store = ModelStore::from_env();
// One directory, or files one by one (a GGUF, every part of a split GGUF).
let found = artifact::inspect(&[PathBuf::from("out/qwen3-8b-support-r3.Q4_K_M.gguf")])?;
println!("{:?} {:?} {:?} {}", found.kind, found.format, found.architecture, found.revision);
let record = Provenance::for_artifact(&found, "support-r3", ORIGIN_UPLOAD).with_license("apache-2.0");
// `staged` holds the files under their artifact paths, outside the store (an upload's parts).
let staged = PathBuf::from("/tmp/upload-staging");
match store.import(&staged, &found, &record, "support-r3")? {
    Imported::Model(m) => println!("{} in {:?}", m.id, m.files),
    other => println!("{other:?}"),
}
println!("{} bytes free, {} bytes uploaded", store.free_bytes(), store.bytes_by_origin(ORIGIN_UPLOAD));
```

What `inspect` accepts: GGUF models (split ones with every part), GGUF LoRA adapters and
multimodal projectors (`GgufKind::Projector`: `general.type` = `mmproj` or the `clip`
architecture), safetensors model directories (`config.json` with `architectures`, an index that
covers every shard), MLX directories (MLX-LM's `format` = `mlx` metadata or a `quantization`
block in `config.json`) and PEFT LoRA directories. Weights are data, never code: pickled weights
(`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`), Python files and other code are refused, and so are
symlinks, hidden files given by name and anything that isn't a regular file. Headers are read
with size caps before anything is hashed; `artifact::early_header` checks a header from the
first bytes of a file still being received. The revision is `sha256:` and the first 16 hex
digits of the SHA-256 of the sorted `path:sha256` lines.

`import` checks the staged files against the artifact (size, SHA-256 and headers), sets the
record's id, kind, files and revision from the bytes, moves the files into a work directory
next to the target and renames it into place last. Layout: a GGUF model in `<id>/`, an adapter in
`adapters/<id>/` (with its `adapter.json`), a projector in `projectors/<id>/`, a safetensors
model in `safetensors/<id>/` and an MLX model in `mlx/<id>/`. `list` keeps listing GGUF models
(an imported one under its id, with `provenance`); `dir_models` lists the safetensors and MLX
directories, `projectors` the projectors. An adapter's `size` leaves `adapter.json` and
`provenance.json` out. `free_bytes` and `disk_space` read the store's filesystem.

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
