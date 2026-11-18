//! Weirdnet local models.
//!
//! - [`catalog`]: pinned Gemma 4 E2B and E4B GGUF files (Hugging Face repo, file, size, SHA-256).
//! - [`store`] and [`mod@download`]: where model files live, and a consent-gated download that
//!   resumes a `.part` file and checks size and SHA-256 before the file appears.
//! - [`hf`]: read a Hugging Face repo's GGUF files, so any repo can become a [`ModelSpec`].
//! - [`prompt`]: render chat turns with the model's own Jinja template, and the `<tool_call>`
//!   tag format for tools (prompt text and a streaming parser).
//! - [`worker`]: an [`Engine`] on its own thread, streaming text with cancel.
//! - `llama` (feature): the llama.cpp [`Engine`].
//! - `agent` (feature): `weirdnet_agent::ModelBackend` over the worker.
//!
//! Nothing here knows an app: callers pass paths, consent and callbacks.
#![forbid(unsafe_code)]

pub mod catalog;
pub mod download;
pub mod hf;
pub mod prompt;
pub mod store;
#[cfg(test)]
mod testing;
pub mod worker;

#[cfg(feature = "agent")]
pub mod agent;
#[cfg(feature = "llama")]
pub mod llama;

pub use catalog::{Device, ModelSpec, human};
pub use download::{Consent, Progress, download};
pub use prompt::{ChatMessage, ChatRole};
pub use store::ModelStore;
pub use worker::{Chunk, Engine, GenerateRequest, Generation, Stats, Worker};

/// Everything that can go wrong here.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("network: {0}")]
    Http(String),
    #[error("{name}: checksum mismatch (got {got}); the file was deleted")]
    Checksum { name: String, got: String },
    #[error("download stopped at {done} of {total} (run it again to resume)")]
    Incomplete { done: String, total: String },
    #[error("download cancelled at {done} of {total} (run it again to resume)")]
    Cancelled { done: String, total: String },
    #[error("consent was given for another model")]
    Consent,
    #[error("{0:?} is not a Hugging Face model id (owner/name)")]
    BadRepo(String),
    #[error("{0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, Error>;
