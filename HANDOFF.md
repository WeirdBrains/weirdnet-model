# Handoff

Running log for [weirdnet-model](https://github.com/WeirdBrains/weirdnet-model), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

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
