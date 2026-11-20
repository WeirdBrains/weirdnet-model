# weirdnet-model agents

The collective `AGENTS.md` in `WeirdBrains/weirdnet-agents` is mandatory and wins on conflict.

- No app-specific types (Mouser chat, Morph panels, app settings). Apps pass paths, consent and callbacks.
- Nothing downloads without a `Consent` the app creates after showing the user what will be fetched.
- Tests never download a model or touch the real network: use the mock hub in `tests/`. A real-model test is `#[ignore]` and reads `WEIRDNET_TEST_GGUF`.
- Extracted from `mouser-llm` and `mouser-core::models` by copying; Mouser is not changed here. Switching Mouser over is Mouser's owner's follow-up.
- On a shared dev box, build outside the checkout with `CARGO_TARGET_DIR` on a scratch disk; clean targets afterwards. GUI work only inside `xvfb-run`.
