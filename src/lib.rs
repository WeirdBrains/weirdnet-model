//! Weirdnet local models.
//!
//! - [`artifact`]: inspect and hash model files a person brings (GGUF models, split models,
//!   LoRA adapters and projectors; safetensors and MLX model directories; PEFT adapters),
//!   refusing pickles, code and symlinks, and the [`Provenance`] record kept next to them;
//!   [`ModelStore::import`] places them in the store.
//! - [`catalog`]: pinned Gemma 4 E2B and E4B GGUF files (Hugging Face repo, file, size, SHA-256).
//! - [`limits`]: per-model media and request limits ([`ModelLimits`]: context in use, images per
//!   request and their token cost, audio length and cost, video, request bytes), read from a
//!   model and its projector when it loads ([`limits::measure`]) and kept next to its files
//!   ([`ModelStore::record_limits`]). Unknown figures stay absent, never defaulted.
//! - [`store`] and [`mod@download`]: where model files and LoRA adapters live (list and remove
//!   them too), and a consent-gated download that resumes a `.part` file and checks size and
//!   SHA-256 before the file appears, for single and split models and for adapters.
//! - [`hf`]: search the Hugging Face Hub, read a repo's GGUF files (split files grouped) and a
//!   file's header, so any repo can become a [`ModelSpec`]; find a repo's LoRA adapters (GGUF
//!   and PEFT), which are never listed as models; [`HfToken`] for gated repos, read on the
//!   downloading node only.
//! - [`gguf`]: the GGUF header fields that size the KV cache, and whether a file is a model, a
//!   LoRA adapter or a projector (with what a projector takes, [`gguf::ProjectorInfo`]).
//! - [`fit`]: pick the largest file whose weights and KV cache fit the memory budget.
//! - [`prompt`]: render chat turns with the model's own Jinja template, and the `<tool_call>`
//!   tag format for tools (prompt text and a streaming parser).
//! - [`worker`]: an [`Engine`] on its own thread, streaming text with cancel.
//! - [`cpu`]: whether this CPU can run the compiled-in llama.cpp (checked before it loads).
//! - [`hw`]: what accelerators and memory this machine has, the memory budget for models, a
//!   tiered catalog with footprints (chat, embedding and a speech-to-text model with the
//!   `transcribe` role, [`hw::Purpose::Transcribe`]), [`hw::plan`] to pick models that fit and
//!   [`hw::transcriber`] for the speech-to-text one.
//! - `llama` (feature): the llama.cpp [`Engine`].
//! - `agent` (feature): `weirdnet_agent::ModelBackend` over the worker.
//!
//! Nothing here knows an app: callers pass paths, consent and callbacks.
#![forbid(unsafe_code)]

pub mod artifact;
pub mod catalog;
pub mod cpu;
pub mod download;
pub mod fit;
pub mod gguf;
pub mod hf;
pub mod hw;
pub mod limits;
pub mod prompt;
pub mod store;
#[cfg(test)]
mod testing;
pub mod token;
pub mod worker;

#[cfg(feature = "agent")]
pub mod agent;
#[cfg(feature = "llama")]
pub mod llama;

pub use artifact::{Artifact, ArtifactFile, ArtifactFormat, ArtifactKind, Provenance, inspect};
pub use catalog::{Device, ModelSpec, human};
pub use download::{Consent, Progress, download, download_adapter, download_parts};
pub use hf::{AdapterFormat, Hub, RepoAdapter};
pub use limits::{LimitSource, LimitsRecord, ModelLimits};
pub use prompt::{ChatMessage, ChatRole};
pub use store::{DiskSpace, Imported, ModelLayout, ModelStore, StoredAdapter, StoredModel, StoredProjector};
pub use token::HfToken;
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
    /// This CPU lacks instructions the compiled-in llama.cpp uses; see [`cpu`].
    #[error("local models need CPU instructions this one lacks: {}", missing.join(", "))]
    UnsupportedCpu { missing: Vec<String> },
    /// `cpu-dispatch` builds: no llama.cpp CPU backend library could be loaded.
    #[error("no llama.cpp CPU backend found in {} (ship the ggml-cpu libraries there)", dir.display())]
    NoCpuBackend { dir: std::path::PathBuf },
}

pub type Result<T> = std::result::Result<T, Error>;
