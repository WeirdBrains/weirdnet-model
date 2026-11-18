# weirdnet-model

Local models for WeirdBrains apps: download a GGUF from Hugging Face (only after consent, with
resume and a SHA-256 check), load it with llama.cpp, and stream tokens from a worker thread you can
cancel. App-agnostic; the model is optional at build time.

- `catalog`: Gemma 4 E2B (phones, small laptops) and E4B (desktop) GGUF defaults, with repo, file,
  size and checksum. `catalog::default_for(Device)` picks one.
- `ModelStore`, `download`: a cache directory (`WEIRDNET_MODEL_DIR` overrides it), a `Consent` the
  user must give for exactly that model, `Progress` callbacks, `.part` resume and verification.
- `hf`: list a repo's GGUF files and sizes for a model picker.
- `prompt`: chat turns rendered through the model's own chat template, and a streaming parser that
  turns `<tool_call>` tags into tool calls.
- `Worker`: one thread owns the engine; `Worker::generate` returns a `Generation` that streams
  `Chunk`s and can be cancelled from any thread. The `Engine` trait lets tests use a mock.
- `llama` feature: `llama::LlamaEngine` (llama.cpp, CPU). `vulkan` and `metal` add GPU offload.
  Building needs cmake and a C++ compiler.
- `agent` feature: `agent::LocalModel` implements `weirdnet_agent::ModelBackend`, so the agent
  loop can run on a local model with streamed text and parsed tool calls.

```toml
weirdnet-model = { version = "0.1", features = ["llama", "agent"] }
```

Tests never download a model. `WEIRDNET_TEST_GGUF=/path/model.gguf cargo test --features llama -- --ignored`
runs the real-model check.

## License

BSD-3-Clause.
