# Changelog

## [Unreleased]

## [0.1.0] - 2026-11-18

### Added

- Model catalog with Gemma 4 E2B and E4B GGUF defaults, a model store, and consent-gated Hugging Face downloads with resume, progress and SHA-256 verification.
- Hugging Face repo listing for GGUF files.
- Chat prompt rendering through the model's chat template, and a streaming tool-call parser.
- Worker thread with streaming generation and cancel, and an `Engine` trait with a mock for tests.
- `llama` feature (llama.cpp), `vulkan` and `metal` GPU features, and `agent` feature implementing `weirdnet_agent::ModelBackend`.
