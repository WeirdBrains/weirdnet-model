# Handoff

Running log for [weirdnet-model](https://github.com/WeirdBrains/weirdnet-model), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

## 2026-11-27: text to speech: Purpose::Speak, speech catalog and profiles

What changed (on main, for release 0.5.5; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`), the `weirdnet-model` part of Ymir's audio design (`docs/design/audio.md`, "Weirdnet crate spec", on Ymir's `audio-tts-design` branch), requested for Michael:

- `hw::Purpose::Speak` (`speak` on the wire, `Purpose::ALL`). Eight catalog entries with memory footprints: `qwen3-tts-0.6b-customvoice`, `qwen3-tts-0.6b-base`, `qwen3-tts-1.7b-customvoice` and `qwen3-tts-1.7b-voicedesign` (the Qwen repos vLLM-Omni serves; the shared speech tokenizer counts in `footprint`), and `qwen3-tts-1.7b-base`, `kokoro-82m`, `voxcpm2` and `fish-audio-drama-3` (ggml-org GGUFs with their pinned projectors). Sizes and SHA-256 from the Hugging Face API (2026-11-27). `hw::speaker` (largest unrestricted GGUF speaker that fits; `hw::plan` never proposes one), `CatalogEntry::speaks`, `speech`, `license`, `restricted`, `is_gguf`.
- `speech`: `SpeechProfile`, `AudioFormat`, `CloneMode` with the same fields and wire form as `weirdnet_agent::speech` (0.5.11), `From` both ways under `agent`, `SpeechProfile::or` (engine report over the catalog). Each catalog profile is cited to the model card and to llama.cpp's `docs/multimodal/tts.md` and llama-server README (GGUF) or vLLM-Omni's speech API (Qwen repos). Notable: ggml-org's Drama 3 projector has no codec encoder, so that entry does not clone ("Voice cloning is not supported yet", its card); ggml-org's Qwen3-TTS 1.7B Base projector has the speaker encoder only, so llama.cpp clones from the clip alone (`CloneMode::Audio`).
- Drama 3 is under the Fish Audio Community License (`speech::FISH_AUDIO_COMMUNITY`): `restricted`, never picked by `speaker`, and `download` refuses its files (by id, or from `ggml-org/drama-3-GGUF` or `fishaudio/drama-3` under any id) unless the consent carries `Consent::accepting_license("fish-audio-community")`; `hw::restricted_license`.
- Hand-placed GGUF speech models: `speech::from_gguf(model, projector)` from `clip.has_gen_audio_encoder`, `clip.gen.audio.projector_type` (`qwen3tts_gen`, `kokoro`, `voxcpm2_gen`, `drama3_gen`), the reference encoder, `general.languages`, the license and the projector's `voice.*` tensors. `limits::measure` fills the new `LimitsRecord::speech` so `weirdnet-backends` keeps it with `set_limits_store`. `gguf`: `GgufHeader::languages`, `license`; `ProjectorInfo::speech`, `speech_projector`, `default_voice`, `voices`, `speech_type`, `reads_audio`.
- `ModelLimits::video_frames_per_minute`; with `agent`, `to_media`/`from_media` now carry `video_max_seconds`, `video_max_bytes` and `video_frames_per_minute` (`weirdnet-agent` 0.5.10). The `agent` feature needs `weirdnet-agent` 0.5.11.

Checks run: `cargo fmt --check`; clippy `-D warnings` with default features, `agent` and `llama,agent`; tests with default features, `--no-default-features`, `agent` and `llama,agent`; `cargo doc -D warnings` with `llama,agent`; `cargo package --locked`; `cargo semver-checks` against 0.5.4 with `agent` and with `llama,agent` (no semver update required). `from_gguf` was run on the published headers of the four ggml-org speech GGUFs and their projectors and matches each catalog profile exactly.

Not tested: no speech model was downloaded or run; the profiles of the vLLM-Omni entries were not checked against a running vLLM-Omni; a Drama 3 projector with the codec encoder (its type name is assumed only through `clip.has_audio_encoder`).

What's left and next: release 0.5.5. Then `weirdnet-backends` 0.4.10 can merge `CatalogEntry::speech` and `LimitsRecord::speech` with what the engine reports at load (`SpeechProfile::or`, then `to_agent()`), and Ymir's download path passes `--accept-license` through `Consent::accepting_license`.

## 2026-11-26: release 0.5.4

What changed: release `v0.5.4` (tag [v0.5.4](https://github.com/WeirdBrains/weirdnet-model/releases/tag/v0.5.4), the `release: v0.5.4` commit right after this entry). It ships speech to text and measured model limits (`86980ef`, entry below) and CI on macOS and Windows (`5a70d34`) to [crates.io](https://crates.io/crates/weirdnet-model/0.5.4); docs at [docs.rs](https://docs.rs/weirdnet-model/0.5.4).

Release approval: m3talsmith approved patch releases without asking ("Don't ask about publishing").

Checks run: CI green on `86980ef` (Linux check and cpu-dispatch, macOS, Windows); `cargo semver-checks` against 0.5.3 with `agent` and with `llama,agent` (no semver update required); the release script's checks from [release.toml](release.toml) (fmt, clippy `-D warnings` and tests with `llama,agent`, `cargo doc -D warnings`), then `cargo publish`.

Not tested: as in the entry below.

What's left and next: as in the entry below (weirdnet-agent `Transcribe` purpose and route; weirdnet-backends measuring limits at load and calling the transcriber).

## 2026-11-26: speech to text and measured model limits

What changed (on main, for release 0.5.4; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`), the `weirdnet-model` part of the video design ([Ymir](https://github.com/WeirdBrains/ymir/blob/feature/video-upload/docs/design/video.md), [Bragi](https://github.com/WeirdBrains/bragi/blob/feature/video-upload/docs/design/video.md)):

- `hw::Purpose::Transcribe` (`transcribe` on the wire), with `Purpose::as_str`, `parse` and `Display`. The catalog gains Voxtral Mini 3B (`hw::VOXTRAL_MINI_3B`, `ggml-org/Voxtral-Mini-3B-2507-GGUF`, Q4_K_M) with its pinned audio projector (`CatalogEntry::projector`), which `footprint` now counts; `hw::transcriber`, `hw::by_purpose`, `CatalogEntry::transcribes`. `hw::plan` never picks it.
- `limits`: `ModelLimits` (context in use, images, audio per minute and per 30-second piece, video, request bytes; every field absent when unknown), `or`, `tightest`, `image_cost`, `audio_cost`; `from_projector` (Gemma 3, Qwen-VL, Voxtral, Ultravox and Qwen2 audio projectors, as llama.cpp's `mtmd` computes them); `measure` (model and projector headers at load, with the context the node runs at, never the trained one); `LimitsRecord` (`weirdnet-limits/1`, `LimitSource`, unknown fields kept). With `agent`: `to_media`, `from_media` and `From` both ways with `weirdnet_agent::MediaLimits` (the agent requirement is now 0.5.7).
- `ModelStore::record_limits`, `limits`, `clear_limits`, `limits_path`: the record sits next to a GGUF's first file (`<file>.limits.json`) or inside a directory model (`.limits.json`), is dropped when the model's size changes, and goes with `remove_local`. `StoredModel::limits` is filled by `list` and `dir_models`.
- `gguf::ProjectorInfo` on `GgufHeader::projector`; `LlamaEngine::limits`.

Checks run: `cargo fmt --check`; clippy `-D warnings` with default features, `agent`, `--no-default-features` and `llama,agent`; tests with default features and `--no-default-features` (99 passed), `agent` (106) and `llama,agent` (107 passed, 2 ignored); the 2 ignored real-model tests with a local Qwen2.5 0.5B GGUF; `cargo doc -D warnings` with `llama,agent`; `cargo semver-checks` against 0.5.3 with `agent` and with `llama,agent` (no semver update required). `from_projector` and `measure` were also run against the headers of the published Voxtral, Gemma 3, Gemma 4, Qwen2.5-VL, Qwen2.5-Omni and Ultravox projector files.

Not tested: transcribing real audio with Voxtral (no model was downloaded or run); token counts against a running llama-server; Gemma 4's projectors (`gemma4v`, `gemma4a`) and other projector types are left absent rather than computed.

What's left and next: release 0.5.4. Then [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends) can call `limits::measure` when it loads a model (with its `--ctx-size` per slot and any `--image-max-tokens`) and record it; [weirdnet-agent](https://github.com/WeirdBrains/weirdnet-agent)'s catalog `Purpose` needs a `transcribe` variant for the transcription route.

## 2026-11-26: CI on macOS and Windows

What changed: [CI](.github/workflows/ci.yml) gets a `platforms` job on `macos-latest` and `windows-latest`: clippy `-D warnings` and tests with default features, tests with `agent` only, and clippy and tests with `llama,agent` (llama.cpp built from source with the runner's compiler). Linux keeps the full `check` and `cpu-dispatch` jobs.

Checks run: none locally for this commit (workflow only); the first run on main is the check.

Not tested: `cpu-dispatch`, Vulkan and Metal on macOS and Windows.

What's left and next: the speech-to-text catalog entry and measured model limits for the [video design](https://github.com/WeirdBrains/ymir/blob/feature/video-upload/docs/design/video.md), then release 0.5.4.

## 2026-11-24: release 0.5.3

What changed: release `v0.5.3` (tag [v0.5.3](https://github.com/WeirdBrains/weirdnet-model/releases/tag/v0.5.3), the `release: v0.5.3` commit right after this entry). It ships the model upload work below (`a3c3ad1`) to [crates.io](https://crates.io/crates/weirdnet-model/0.5.3); docs at [docs.rs](https://docs.rs/weirdnet-model/0.5.3).

Release approval: m3talsmith approved publishing 0.5.3 to crates.io once main is green (2026-11-24, 9:21 PM ET).

Checks run: CI green on `a3c3ad1`; the release script's checks from [release.toml](release.toml) (fmt, clippy `-D warnings` and tests with `llama,agent`, `cargo doc -D warnings`), then `cargo publish`.

Not tested: as in the entry below.

What's left and next: [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends) 0.4.4 moves to `weirdnet-model = "0.5.3"` and adds uploads to `models/1`.

## 2026-11-24: model uploads (artifact inspection, import, provenance)

What changed (on main, for release 0.5.3; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`), the model side of the [model upload plan](https://github.com/WeirdBrains/ymir/blob/feature/model-upload-plan/docs/design/model-upload.md), for uploads in [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends):

- `artifact`: `inspect` / `inspect_with` / `inspect_staged` read model files and folders made elsewhere into an `Artifact` (`ArtifactKind`: model, adapter, projector; `ArtifactFormat`: GGUF with its split parts, safetensors, MLX; `ArtifactFile` with SHA-256 hashed while streaming; a content `revision`). GGUF, PEFT (`adapter_config.json` + `adapter_model.safetensors`), Hugging Face safetensors (`config.json`, sharded with an index that must cover every shard) and MLX folders are accepted; pickles, `.py` files, symlinks, other extensions and paths that leave the folder are refused with `Error::Model`. `read_safetensors` and `early_header` check headers from the first bytes. There is no PEFT format variant: adapters reuse `hf::AdapterFormat`.
- `Provenance` (`provenance.json`, format `weirdnet-provenance/1`): origin, license, base, dataset, trainer, uploader and time, unknown fields kept. `StoredModel` and `StoredAdapter` carry it; `StoredModel::layout` (`ModelLayout`). An adapter's size leaves `provenance.json` out.
- `gguf`: `GgufKind::Projector` for vision projector files (`general.type` = `mmproj` or the `clip` architecture).
- `ModelStore`: `import`, `import_replacing`, `import_with` and `import_files` (inspects once) move staged files into place (GGUF models in `<id>/` under their own names, `adapters/`, `projectors/`, `safetensors/`, `mlx/`), write `provenance.json` (and `adapter.json` for adapters), roll back on failure and swap atomically on replace. `dir_models`, `model_dir`, `projectors`, `projector_dir`, `remove_projector`, `disk_space`, `free_bytes`, `bytes_by_origin`. Staging for partial uploads lives outside the store (the caller's choice).

Release approval: m3talsmith approved publishing 0.5.3 to crates.io once main is green (2026-11-24, 9:21 PM ET), together with weirdnet-backends 0.4.4.

Checks run: the `release.toml` checks (fmt; clippy `-D warnings` with `llama,agent`; tests with `llama,agent`, 91 passed, 2 ignored; `cargo doc -D warnings`); `cargo semver-checks` against 0.5.2 with `llama,agent` (no semver update required). Tests use temporary folders with generated GGUF, safetensors, PEFT and MLX files.

Not tested: real Hugging Face, MLX or PEFT checkpoints from training tools; very large files (multi-GB) and imports across filesystems (the copy fallback runs only when rename fails); Windows and macOS.

What's left and next: release 0.5.3, then weirdnet-backends 0.4.4 (upload, fetch and peer pull in `models/1`) moves to `weirdnet-model = "0.5.3"`. Device roles in weirdnet-sync grants are already free-form app strings, so no sync or agent change is needed; m3talsmith chose that any linked device may upload, so no any-of-roles helper was added to `RosterAuth`.

## 2026-11-22: LoRA adapters merged to main

What changed: fast-forwards main to [lora-adapters](https://github.com/WeirdBrains/weirdnet-model/tree/lora-adapters) (`78711f0`); main is ready for 0.5.2, not released. No code changes in the merge.

Checks run: the `release.toml` checks on this checkout after the fast-forward (fmt, clippy `-D warnings` with `llama,agent`, tests with `llama,agent` — 70 passed, 2 ignored; `cargo doc -D warnings`); `cargo semver-checks` against 0.5.1 earlier today (no semver update required beyond a patch).

Not tested: as in the entry below.

What's left and next: release 0.5.2 (approved), then weirdnet-backends 0.4.0 (its lora-adapters branch) moves to `weirdnet-model = "0.5.2"`.


## 2026-11-22: LoRA adapters (branch `lora-adapters`)

What changed (branch [lora-adapters](https://github.com/WeirdBrains/weirdnet-model/tree/lora-adapters), unreleased; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`), for LoRA serving in [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends):

- `gguf::read_header` with `GgufHeader` and `GgufKind`: reads adapter GGUFs (`general.type` = `adapter`, `adapter.type`, `adapter.lora.alpha`, base model from `general.base_model.0.repo_url`); `read_info` is unchanged.
- `hf`: `Hub::adapters` (`RepoAdapter`, `AdapterFormat`: GGUF LoRA and PEFT), `Hub::gguf_header`, `adapter_hint`, `Search::with_adapters` and `SearchHit::adapter`. `Hub::models` drops hinted files whose header says adapter.
- `download_adapter` with `Consent::given_adapter` into `adapters/<id>/` (same resume and SHA-256 checks, plus a path-free `adapter.json`); `ModelStore::adapters`, `remove_adapter`, `adapter_dir`; `list` skips adapters.

Checks run: `cargo fmt --check`; clippy `-D warnings` default, `agent` and `llama,agent`; tests with `llama,agent` (70 passed, MockHub with a LoRA GGUF header and PEFT layouts); `cargo doc -D warnings`.

Not tested: a real Hub adapter repo; `cargo semver-checks` against 0.5.1 had not finished at this push.

What's left and next: semver check; merge to main; proposed release 0.5.2 (additive) once m3talsmith approves, before weirdnet-backends 0.4.0.

## 2026-11-21: release 0.5.1

What changed: release `v0.5.1` (tag [v0.5.1](https://github.com/WeirdBrains/weirdnet-model/releases/tag/v0.5.1), the `release: v0.5.1` commit right after this entry), approved by m3talsmith. It ships the Hub search, fit chooser, token, split-download, list and remove work below (5b0c043) to [crates.io](https://crates.io/crates/weirdnet-model/0.5.1); docs at [docs.rs](https://docs.rs/weirdnet-model/0.5.1).

Checks run: the release script's checks from [release.toml](release.toml) (fmt, clippy `-D warnings` and tests with `llama,agent`, `cargo doc -D warnings`), then `cargo publish`.

Not tested: as in the entry below.

What's left and next: [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends) 0.3.0 moves to `weirdnet-model = "0.5.1"`.

## 2026-11-21: Hub search, fit chooser, token, split downloads, list and remove

What changed (commit 5b0c043, on main, unreleased; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `hf::Hub`: `search` (GGUF repos, most downloaded first), `models` (a repo's GGUFs with split files grouped into one `RepoModel`, quantization from the file name), `gguf_info` (the header through a `Range` request, for KV sizing). `check_file` keeps repo file names inside the store.
- `HfToken`: read on the downloading node from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or the Hugging Face token files. `ModelStore::from_env` sends it to the public Hub only; `with_token` to any Hub. Redacted `Debug`, no serde, and redirects to other hosts drop it.
- `gguf::read_info`, `fit::{choose, rank, measure, FitOptions}`: the largest complete model whose weights plus KV cache (plus 10%) fit a `hw::Budget`.
- `download_parts` for split models under one consent; `download` creates subdirectories. `ModelStore::list` and `remove_local`.

Checks run: `cargo fmt --check`; clippy `-D warnings` default, `--features agent` and `--features llama,agent`; tests default, `--no-default-features --features agent` and `--features llama,agent` (62 passed); `cargo doc` with `-D warnings`; `cargo package`. All tests use a local mock Hub. `live_hub` (ignored) was run once against the real Hub: search, a repo's models, a Gemma 4 header and a fit all worked.

Not tested: a real gated repo with a token; a real split-model download (only the mock); Windows; the `cpu-dispatch` build.

What's left and next: a patch release (proposed 0.5.1, additive) once m3talsmith approves it; then `weirdnet-backends` moves its `models/1` branch onto it.
