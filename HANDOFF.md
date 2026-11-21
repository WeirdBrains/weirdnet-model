# Handoff

Running log for [weirdnet-model](https://github.com/WeirdBrains/weirdnet-model), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

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
