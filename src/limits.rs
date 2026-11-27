//! Per-model limits for media and requests, and the record a node keeps of them.
//!
//! What a client needs to fit images, audio (or a video's frames and soundtrack) to one model:
//! the context the model runs with, images per request and what one costs, audio length and
//! what it costs, and the largest request. Every field is absent when it isn't known, never
//! defaulted, so a client can say what is missing instead of guessing.
//!
//! - [`ModelLimits`]: the figures, with [`or`](ModelLimits::or) (operator settings over measured
//!   ones), [`tightest`](ModelLimits::tightest) and cost helpers. With the `agent` feature they
//!   convert to and from `weirdnet_agent::MediaLimits`, which catalog entries carry.
//! - [`from_projector`]: what a llama.cpp projector's header (`mmproj`) says images and audio
//!   cost, for the projector types whose costs follow from the header alone.
//! - [`measure`]: read a model's GGUF (and its projector) when it loads, with the context the
//!   node runs it at, into a [`LimitsRecord`]; [`ModelStore::record_limits`](crate::ModelStore::record_limits)
//!   keeps it next to the model's files and [`StoredModel::limits`](crate::StoredModel::limits)
//!   reads it back.
//!
//! Costs from headers follow llama.cpp's `mtmd` (2026-11): a flag such as `--image-max-tokens`
//! changes them, so a node that passes one sets the result over the header's with
//! [`ModelLimits::or`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::gguf::{GgufKind, ProjectorInfo};
use crate::speech::SpeechProfile;
use crate::{Error, Result};

/// [`LimitsRecord::format`].
pub const LIMITS_FORMAT: &str = "weirdnet-limits/1";
/// The name a GGUF model's limits record gets next to its first file: `<file>.limits.json`.
pub const LIMITS_SUFFIX: &str = ".limits.json";
/// The limits record's name inside a safetensors or MLX model directory.
pub const LIMITS_FILE: &str = ".limits.json";

/// Seconds of audio llama.cpp's Whisper-style encoders take at a time (3000 mel frames at
/// 16 kHz with a 160-sample hop); longer audio is cut into pieces this long, the last one
/// padded.
pub const WHISPER_CHUNK_SECONDS: u32 = 30;
const WHISPER_CHUNK_FRAMES: u32 = 3000;
/// llama.cpp's own cap on one image for Qwen-VL projectors, when the header has no
/// `clip.vision.image_max_pixels` and the node passes no `--image-max-tokens`.
const QWEN_VL_MAX_IMAGE_TOKENS: u32 = 4096;

/// Media and request limits for one model. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelLimits {
    /// The context in tokens the model runs with on this node (per slot: llama.cpp's
    /// `--ctx-size` divided by `--parallel`), not the context it was trained for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u32>,
    /// Most images one request may carry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_images: Option<u32>,
    /// Tokens one image costs. With [`image_patch`](Self::image_patch) set, the most one image
    /// costs; otherwise every image costs this much.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u32>,
    /// For models whose image cost grows with size: the edge in pixels of the square one image
    /// token covers. Absent: a fixed cost per image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_patch: Option<u32>,
    /// The longest image edge in pixels the model uses; larger images are scaled down to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_edge: Option<u32>,
    /// Longest audio input in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_max_seconds: Option<u32>,
    /// Tokens one minute of audio costs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_tokens_per_minute: Option<u32>,
    /// Audio is encoded in pieces this many seconds long, the last one padded, so a short clip
    /// costs a whole piece ([`audio_chunk_tokens`](Self::audio_chunk_tokens)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_chunk_seconds: Option<u32>,
    /// Tokens one piece of [`audio_chunk_seconds`](Self::audio_chunk_seconds) costs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_chunk_tokens: Option<u32>,
    /// Longest video the model takes whole, in seconds (models that read video natively).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_max_seconds: Option<u32>,
    /// Largest video the model takes whole, in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_max_bytes: Option<u64>,
    /// Most frames the model reads from one minute of video sent whole (120 for 2 a second),
    /// kept per minute so the type stays `Eq`. Since 0.5.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_frames_per_minute: Option<u32>,
    /// Largest request body in bytes the serving side accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
}

impl ModelLimits {
    /// No limits known.
    pub const UNKNOWN: Self = Self {
        context_tokens: None,
        max_images: None,
        image_tokens: None,
        image_patch: None,
        image_edge: None,
        audio_max_seconds: None,
        audio_tokens_per_minute: None,
        audio_chunk_seconds: None,
        audio_chunk_tokens: None,
        video_max_seconds: None,
        video_max_bytes: None,
        video_frames_per_minute: None,
        max_request_bytes: None,
    };

    pub fn new() -> Self {
        Self::UNKNOWN
    }

    /// Whether no limit is known.
    pub fn is_unknown(&self) -> bool {
        *self == Self::UNKNOWN
    }

    pub fn with_context_tokens(mut self, tokens: u32) -> Self {
        self.context_tokens = Some(tokens);
        self
    }

    pub fn with_max_images(mut self, n: u32) -> Self {
        self.max_images = Some(n);
        self
    }

    /// Every image costs `tokens` (or at most that, with [`with_image_patch`](Self::with_image_patch)).
    pub fn with_image_tokens(mut self, tokens: u32) -> Self {
        self.image_tokens = Some(tokens);
        self
    }

    /// The image cost grows with size: one token per `patch` x `patch` pixels.
    pub fn with_image_patch(mut self, patch: u32) -> Self {
        self.image_patch = Some(patch);
        self
    }

    pub fn with_image_edge(mut self, pixels: u32) -> Self {
        self.image_edge = Some(pixels);
        self
    }

    pub fn with_audio_max_seconds(mut self, seconds: u32) -> Self {
        self.audio_max_seconds = Some(seconds);
        self
    }

    pub fn with_audio_tokens_per_minute(mut self, tokens: u32) -> Self {
        self.audio_tokens_per_minute = Some(tokens);
        self
    }

    /// Audio is encoded in `seconds`-long pieces of `tokens` each; sets
    /// [`audio_tokens_per_minute`](Self::audio_tokens_per_minute) to match.
    pub fn with_audio_chunks(mut self, seconds: u32, tokens: u32) -> Self {
        self.audio_chunk_seconds = Some(seconds);
        self.audio_chunk_tokens = Some(tokens);
        if seconds > 0 {
            self.audio_tokens_per_minute = u32::try_from(u64::from(tokens) * 60 / u64::from(seconds)).ok();
        }
        self
    }

    pub fn with_video_max_seconds(mut self, seconds: u32) -> Self {
        self.video_max_seconds = Some(seconds);
        self
    }

    pub fn with_video_max_bytes(mut self, bytes: u64) -> Self {
        self.video_max_bytes = Some(bytes);
        self
    }

    /// The model reads at most `frames` frames per minute of video (120 for 2 a second).
    pub fn with_video_frames_per_minute(mut self, frames: u32) -> Self {
        self.video_frames_per_minute = Some(frames);
        self
    }

    /// [`video_frames_per_minute`](Self::video_frames_per_minute) per second.
    pub fn video_frame_rate(&self) -> Option<f64> {
        self.video_frames_per_minute.map(|f| f64::from(f) / 60.0)
    }

    pub fn with_max_request_bytes(mut self, bytes: u64) -> Self {
        self.max_request_bytes = Some(bytes);
        self
    }

    /// [`audio_tokens_per_minute`](Self::audio_tokens_per_minute) per second.
    pub fn audio_tokens_per_second(&self) -> Option<f64> {
        self.audio_tokens_per_minute.map(|t| f64::from(t) / 60.0)
    }

    /// What one `width` x `height` image costs in tokens, if known: the fixed cost, or with
    /// [`image_patch`](Self::image_patch) the patches after scaling down to
    /// [`image_edge`](Self::image_edge), capped at [`image_tokens`](Self::image_tokens).
    pub fn image_cost(&self, width: u32, height: u32) -> Option<u32> {
        let Some(patch) = self.image_patch.filter(|p| *p > 0) else {
            return self.image_tokens;
        };
        let (w, h) = self.scaled(width, height);
        let tokens = u64::from(w.div_ceil(patch)) * u64::from(h.div_ceil(patch));
        let tokens = u32::try_from(tokens).unwrap_or(u32::MAX);
        Some(self.image_tokens.map_or(tokens, |cap| tokens.min(cap)))
    }

    /// `width` x `height` scaled down (aspect kept) so the longer edge fits
    /// [`image_edge`](Self::image_edge); unchanged when it fits or no edge is known.
    pub fn scaled(&self, width: u32, height: u32) -> (u32, u32) {
        let long = width.max(height);
        match self.image_edge {
            Some(edge) if edge > 0 && long > edge => {
                let s = |v: u32| ((u64::from(v) * u64::from(edge)).div_ceil(u64::from(long)) as u32).max(1);
                (s(width), s(height))
            }
            _ => (width, height),
        }
    }

    /// What `millis` of audio costs in tokens, if known: whole pieces when the model encodes
    /// audio in pieces ([`audio_chunk_seconds`](Self::audio_chunk_seconds)), else the per-minute
    /// rate rounded up.
    pub fn audio_cost(&self, millis: u64) -> Option<u64> {
        if let (Some(secs), Some(tokens)) = (self.audio_chunk_seconds.filter(|s| *s > 0), self.audio_chunk_tokens) {
            return Some(millis.div_ceil(u64::from(secs) * 1000).max(1) * u64::from(tokens));
        }
        self.audio_tokens_per_minute.map(|t| (u64::from(t) * millis).div_ceil(60_000))
    }

    /// Each field from `self` when known, else from `fallback`: operator settings over measured
    /// ones (`operator.or(measured)`).
    pub fn or(self, fallback: Self) -> Self {
        Self {
            context_tokens: self.context_tokens.or(fallback.context_tokens),
            max_images: self.max_images.or(fallback.max_images),
            image_tokens: self.image_tokens.or(fallback.image_tokens),
            image_patch: self.image_patch.or(fallback.image_patch),
            image_edge: self.image_edge.or(fallback.image_edge),
            audio_max_seconds: self.audio_max_seconds.or(fallback.audio_max_seconds),
            audio_tokens_per_minute: self.audio_tokens_per_minute.or(fallback.audio_tokens_per_minute),
            audio_chunk_seconds: self.audio_chunk_seconds.or(fallback.audio_chunk_seconds),
            audio_chunk_tokens: self.audio_chunk_tokens.or(fallback.audio_chunk_tokens),
            video_max_seconds: self.video_max_seconds.or(fallback.video_max_seconds),
            video_max_bytes: self.video_max_bytes.or(fallback.video_max_bytes),
            video_frames_per_minute: self.video_frames_per_minute.or(fallback.video_frames_per_minute),
            max_request_bytes: self.max_request_bytes.or(fallback.max_request_bytes),
        }
    }

    /// The stricter of two copies, field by field: less context, fewer images, a smaller edge,
    /// shorter audio and video, fewer video frames per minute, fewer request bytes, a smaller patch and higher token costs. A
    /// field known on one side only keeps that side's value; a field neither side knows stays
    /// absent.
    pub fn tightest(self, other: Self) -> Self {
        fn pick<T: Ord>(a: Option<T>, b: Option<T>, f: fn(T, T) -> T) -> Option<T> {
            match (a, b) {
                (Some(a), Some(b)) => Some(f(a, b)),
                (a, b) => a.or(b),
            }
        }
        Self {
            context_tokens: pick(self.context_tokens, other.context_tokens, Ord::min),
            max_images: pick(self.max_images, other.max_images, Ord::min),
            image_tokens: pick(self.image_tokens, other.image_tokens, Ord::max),
            image_patch: pick(self.image_patch, other.image_patch, Ord::min),
            image_edge: pick(self.image_edge, other.image_edge, Ord::min),
            audio_max_seconds: pick(self.audio_max_seconds, other.audio_max_seconds, Ord::min),
            audio_tokens_per_minute: pick(self.audio_tokens_per_minute, other.audio_tokens_per_minute, Ord::max),
            audio_chunk_seconds: pick(self.audio_chunk_seconds, other.audio_chunk_seconds, Ord::max),
            audio_chunk_tokens: pick(self.audio_chunk_tokens, other.audio_chunk_tokens, Ord::max),
            video_max_seconds: pick(self.video_max_seconds, other.video_max_seconds, Ord::min),
            video_max_bytes: pick(self.video_max_bytes, other.video_max_bytes, Ord::min),
            video_frames_per_minute: pick(self.video_frames_per_minute, other.video_frames_per_minute, Ord::min),
            max_request_bytes: pick(self.max_request_bytes, other.max_request_bytes, Ord::min),
        }
    }
}

#[cfg(feature = "agent")]
impl ModelLimits {
    /// The fields `weirdnet_agent::MediaLimits` carries (catalog entries and capability
    /// records), video included (since 0.5.5, `weirdnet-agent` 0.5.10); the context goes in the
    /// catalog entry's own `context`, and the audio piece size is left out (the agent's type
    /// carries the per-minute rate).
    pub fn to_media(&self) -> weirdnet_agent::MediaLimits {
        let mut m = weirdnet_agent::MediaLimits::new();
        m.max_images = self.max_images;
        m.image_tokens = self.image_tokens;
        m.image_patch = self.image_patch;
        m.image_edge = self.image_edge;
        m.audio_max_seconds = self.audio_max_seconds;
        m.audio_tokens_per_minute = self.audio_tokens_per_minute;
        m.video_max_seconds = self.video_max_seconds;
        m.video_max_bytes = self.video_max_bytes;
        m.video_frames_per_minute = self.video_frames_per_minute;
        m.max_request_bytes = self.max_request_bytes;
        m
    }

    /// The fields of `weirdnet_agent::MediaLimits` (an operator's settings, say); the rest absent.
    pub fn from_media(m: &weirdnet_agent::MediaLimits) -> Self {
        Self {
            max_images: m.max_images,
            image_tokens: m.image_tokens,
            image_patch: m.image_patch,
            image_edge: m.image_edge,
            audio_max_seconds: m.audio_max_seconds,
            audio_tokens_per_minute: m.audio_tokens_per_minute,
            video_max_seconds: m.video_max_seconds,
            video_max_bytes: m.video_max_bytes,
            video_frames_per_minute: m.video_frames_per_minute,
            max_request_bytes: m.max_request_bytes,
            ..Self::UNKNOWN
        }
    }
}

#[cfg(feature = "agent")]
impl From<ModelLimits> for weirdnet_agent::MediaLimits {
    fn from(l: ModelLimits) -> Self {
        l.to_media()
    }
}

#[cfg(feature = "agent")]
impl From<weirdnet_agent::MediaLimits> for ModelLimits {
    fn from(m: weirdnet_agent::MediaLimits) -> Self {
        Self::from_media(&m)
    }
}

/// What image and audio inputs cost, from a projector's header, for the projector types whose
/// costs follow from the header alone (as llama.cpp's `mtmd` computes them, 2026-11):
///
/// - `gemma3`: every image is resized to `image_size` square and costs
///   `(image_size / patch_size / scale)²` tokens (scale 4 unless the header sets one): 256 for
///   Gemma 3.
/// - `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger` and the image side of `qwen2.5o`:
///   one token per `2 * patch_size` pixels square, at most `image_max_pixels` (or llama.cpp's
///   4096 tokens) per image.
/// - `voxtral`, `ultravox`, `qwen2a` and the audio side of `qwen2.5o`: 30-second pieces of
///   3000 mel frames, stacked (`stack_factor`), halved by the encoder and halved again by
///   pooling (`voxtral`, `qwen2a`): 187 tokens a piece for Voxtral and Ultravox, 750 for Qwen2
///   audio.
///
/// Other types, and figures a header lacks, stay absent. Nothing here says how many images a
/// request may carry, how long audio may be or what context the model runs at.
pub fn from_projector(p: &ProjectorInfo) -> ModelLimits {
    let mut l = ModelLimits::new();
    match p.image_type() {
        Some("gemma3") => {
            let merge = p.scale_factor.unwrap_or(4);
            if let (Some(size), Some(patch)) = (p.image_size, p.patch_size)
                && patch > 0
                && merge > 0
                && size % patch == 0
                && (size / patch) % merge == 0
            {
                let side = size / patch / merge;
                l = l.with_image_tokens(side * side).with_image_edge(size);
            }
        }
        Some("qwen2vl_merger" | "qwen2.5vl_merger" | "qwen3vl_merger" | "qwen2.5o") => {
            if let Some(patch) = p.patch_size.filter(|p| *p > 0)
                && p.spatial_merge.unwrap_or(2) == 2
            {
                let unit = patch * 2;
                let cap = match p.image_max_pixels {
                    Some(px) => u32::try_from(px / (u64::from(unit) * u64::from(unit))).unwrap_or(u32::MAX),
                    None => QWEN_VL_MAX_IMAGE_TOKENS,
                };
                l = l.with_image_patch(unit).with_image_tokens(cap);
            }
        }
        _ => {}
    }
    let audio = match p.audio_type() {
        Some("voxtral") => p.audio_stack_factor.map(|s| (s, true)),
        Some("ultravox") => p.audio_stack_factor.map(|s| (s, false)),
        Some("qwen2a" | "qwen2.5o") => Some((1, true)),
        _ => None,
    };
    if let Some((stack, pool)) = audio.filter(|(s, _)| *s > 0) {
        let mut n = WHISPER_CHUNK_FRAMES.div_ceil(stack) / 2;
        if pool {
            n /= 2;
        }
        l = l.with_audio_chunks(WHISPER_CHUNK_SECONDS, n);
    }
    l
}

/// How a [`LimitsRecord`]'s figures were found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum LimitSource {
    /// Read from the model's files and the settings it was loaded with ([`measure`]).
    #[default]
    Measured,
    /// Set by the node's operator for this hardware.
    Operator,
    /// Something this version doesn't know.
    #[serde(other)]
    Other,
}

fn limits_format() -> String {
    LIMITS_FORMAT.into()
}

/// A model's limits as a node found them when it loaded the model, kept next to its files
/// ([`ModelStore::record_limits`](crate::ModelStore::record_limits)). Unknown fields are kept in
/// [`extra`](Self::extra) and written back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LimitsRecord {
    /// [`LIMITS_FORMAT`].
    #[serde(default = "limits_format")]
    pub format: String,
    #[serde(default)]
    pub limits: ModelLimits,
    #[serde(default)]
    pub source: LimitSource,
    /// What ran the model, such as `llama.cpp`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// The projector file's name, when one was loaded with the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<String>,
    /// The context the model was trained for (`<arch>.context_length`), for reference: the one
    /// it runs at is [`ModelLimits::context_tokens`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_context: Option<u64>,
    /// Bytes of the model's files when this was recorded; a record whose model has since changed
    /// size is stale and not read back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_bytes: Option<u64>,
    /// When it was recorded, in seconds since the Unix epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<u64>,
    /// What a speech model does ([`Purpose::Speak`](crate::hw::Purpose::Speak)): from the
    /// headers at load ([`measure`], [`speech::from_gguf`](crate::speech::from_gguf)) or what
    /// the engine reported, kept with the limits. `None` for models that don't speak, and in
    /// records from before 0.5.5. Since 0.5.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech: Option<SpeechProfile>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for LimitsRecord {
    fn default() -> Self {
        Self::new(ModelLimits::UNKNOWN, LimitSource::Measured)
    }
}

impl LimitsRecord {
    pub fn new(limits: ModelLimits, source: LimitSource) -> Self {
        Self {
            format: LIMITS_FORMAT.into(),
            limits,
            source,
            engine: None,
            projector: None,
            trained_context: None,
            model_bytes: None,
            recorded_at: None,
            speech: None,
            extra: Default::default(),
        }
    }

    pub fn with_engine(mut self, engine: impl Into<String>) -> Self {
        self.engine = Some(engine.into());
        self
    }

    /// Sets the speech profile ([`speech`](Self::speech)), such as what the engine reported
    /// over what [`measure`] read.
    pub fn with_speech(mut self, speech: SpeechProfile) -> Self {
        self.speech = Some(speech);
        self
    }

    /// Stamps the record with the current time.
    pub fn stamped(mut self) -> Self {
        self.recorded_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs());
        self
    }
}

fn header_of(path: &Path) -> Result<crate::gguf::GgufHeader> {
    let f = std::fs::File::open(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    crate::gguf::read_header(std::io::BufReader::new(f)).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

/// Reads a GGUF model (the first file of a split model) and its projector, if any, as a node
/// loads them, into a record: what the projector's header says images and audio cost
/// ([`from_projector`]), the context the node runs the model at (`context_tokens`, per slot;
/// `None` when the caller doesn't know it, never the trained context) and the trained context
/// for reference. A speech model's projector also gives its speech profile
/// ([`LimitsRecord::speech`], see [`speech::from_gguf`](crate::speech::from_gguf)). Fails when
/// `model` is not a GGUF model or `projector` not a projector.
pub fn measure(model: &Path, projector: Option<&Path>, context_tokens: Option<u32>) -> Result<LimitsRecord> {
    let m = header_of(model)?;
    if m.kind != GgufKind::Model {
        return Err(Error::Model(format!("{} is not a GGUF model", model.display())));
    }
    let mut limits = ModelLimits::new();
    let mut projector_name = None;
    let mut speech = None;
    if let Some(path) = projector {
        let p = header_of(path)?;
        let Some(info) = p.projector.as_ref().filter(|_| p.kind.is_projector()) else {
            return Err(Error::Model(format!("{} is not a projector", path.display())));
        };
        limits = from_projector(info);
        speech = crate::speech::from_gguf(&m, &p);
        projector_name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    }
    limits.context_tokens = context_tokens;
    let mut record = LimitsRecord::new(limits, LimitSource::Measured).stamped();
    record.projector = projector_name;
    record.speech = speech;
    record.trained_context = m.info.and_then(|i| i.context_length);
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{header, projector_header, read_header};
    use serde_json::json;

    fn info(which: &str) -> ProjectorInfo {
        read_header(&projector_header(which)[..]).unwrap().projector.unwrap()
    }

    #[test]
    fn unknown_is_empty_on_the_wire() {
        assert!(ModelLimits::default().is_unknown());
        assert_eq!(serde_json::to_string(&ModelLimits::UNKNOWN).unwrap(), "{}");
        let back: ModelLimits = serde_json::from_str("{}").unwrap();
        assert!(back.is_unknown());
        let l = ModelLimits::new().with_context_tokens(8192).with_max_images(4).with_audio_chunks(30, 187).with_video_max_bytes(1 << 20);
        let j = serde_json::to_value(l).unwrap();
        assert_eq!(
            j,
            json!({"context_tokens": 8192, "max_images": 4, "audio_tokens_per_minute": 374, "audio_chunk_seconds": 30, "audio_chunk_tokens": 187, "video_max_bytes": 1048576})
        );
        assert_eq!(serde_json::from_value::<ModelLimits>(j).unwrap(), l);
        // Fields from a newer version are ignored, not an error.
        let newer: ModelLimits = serde_json::from_value(json!({"max_images": 2, "video_frame_rate": 2})).unwrap();
        assert_eq!(newer, ModelLimits::new().with_max_images(2));
    }

    #[test]
    fn costs() {
        let fixed = ModelLimits::new().with_image_tokens(256);
        assert_eq!(fixed.image_cost(4000, 3000), Some(256));
        assert_eq!(ModelLimits::UNKNOWN.image_cost(10, 10), None);
        let patched = ModelLimits::new().with_image_patch(28).with_image_edge(896);
        assert_eq!(patched.scaled(1792, 896), (896, 448));
        assert_eq!(patched.image_cost(1792, 896), Some(32 * 16));
        assert_eq!(patched.with_image_tokens(100).image_cost(1792, 896), Some(100), "capped");
        let chunks = ModelLimits::new().with_audio_chunks(30, 187);
        assert_eq!(chunks.audio_tokens_per_minute, Some(374));
        assert_eq!(chunks.audio_cost(5_000), Some(187), "a short clip costs a whole piece");
        assert_eq!(chunks.audio_cost(30_000), Some(187));
        assert_eq!(chunks.audio_cost(30_001), Some(374));
        assert_eq!(chunks.audio_cost(0), Some(187));
        let rate = ModelLimits::new().with_audio_tokens_per_minute(375);
        assert_eq!(rate.audio_tokens_per_second(), Some(6.25));
        assert_eq!(rate.audio_cost(1_000), Some(7));
        assert_eq!(ModelLimits::UNKNOWN.audio_cost(1_000), None);
        assert_eq!(ModelLimits::new().with_audio_chunks(0, 10).audio_tokens_per_minute, None, "no rate from a zero-length piece");
    }

    #[test]
    fn operator_over_measured_and_tightest() {
        let measured = ModelLimits::new().with_context_tokens(8192).with_image_tokens(256).with_image_edge(896);
        let operator = ModelLimits::new().with_max_images(2).with_image_tokens(300);
        let both = operator.or(measured);
        assert_eq!((both.max_images, both.image_tokens, both.image_edge, both.context_tokens), (Some(2), Some(300), Some(896), Some(8192)));
        let a = ModelLimits::new().with_context_tokens(4096).with_max_images(8).with_audio_chunks(30, 187).with_video_max_seconds(60);
        let b = ModelLimits::new().with_context_tokens(8192).with_max_images(2).with_audio_chunks(30, 750).with_max_request_bytes(1 << 20);
        let t = a.tightest(b);
        assert_eq!((t.context_tokens, t.max_images, t.audio_chunk_tokens, t.audio_tokens_per_minute), (Some(4096), Some(2), Some(750), Some(1500)));
        assert_eq!((t.video_max_seconds, t.max_request_bytes), (Some(60), Some(1 << 20)), "one side only keeps its value");
        assert_eq!(a.tightest(ModelLimits::UNKNOWN), a);
    }

    #[test]
    fn projector_costs_follow_llama_cpp() {
        // Gemma 3: 896 / 14 = 64 patches a side, pooled by 4: 256 tokens.
        let g = from_projector(&info("gemma3"));
        assert_eq!(g, ModelLimits::new().with_image_tokens(256).with_image_edge(896));
        // Qwen2.5-VL: one token per 28 px square, up to llama.cpp's 4096.
        let q = from_projector(&info("qwen2.5vl"));
        assert_eq!(q, ModelLimits::new().with_image_patch(28).with_image_tokens(4096));
        assert_eq!(q.image_cost(560, 280), Some(20 * 10));
        assert_eq!(from_projector(&info("qwen3vl")).image_patch, Some(32));
        // Voxtral: 3000 frames / 4 stacked / 2 (conv) / 2 (pool) = 187 per 30 s.
        let v = from_projector(&info("voxtral"));
        assert_eq!(v, ModelLimits::new().with_audio_chunks(30, 187));
        assert_eq!(v.audio_tokens_per_minute, Some(374));
        // Ultravox: 3000 / 8 / 2 = 187, no pooling.
        assert_eq!(from_projector(&info("ultravox")).audio_chunk_tokens, Some(187));
        // Qwen2.5-Omni: both sides; Qwen2 audio is 3000 / 2 / 2 = 750 per piece.
        let o = from_projector(&info("qwen2.5o"));
        assert_eq!((o.image_patch, o.audio_chunk_tokens, o.audio_tokens_per_minute), (Some(28), Some(750), Some(1500)));
        // Nothing about images per request, audio length or context comes from a header.
        for l in [g, q, v, o] {
            assert_eq!((l.max_images, l.audio_max_seconds, l.context_tokens, l.max_request_bytes), (None, None, None, None));
        }
    }

    #[test]
    fn unknown_projectors_and_missing_figures_stay_absent() {
        let mut p = ProjectorInfo {
            vision: true,
            audio: true,
            vision_projector: Some("gemma4v".into()),
            audio_projector: Some("gemma4a".into()),
            ..Default::default()
        };
        p.image_size = Some(224);
        p.patch_size = Some(16);
        assert!(from_projector(&p).is_unknown(), "types this version doesn't know are not guessed");
        let mut g = info("gemma3");
        g.patch_size = None;
        assert!(from_projector(&g).is_unknown());
        let mut g = info("gemma3");
        g.scale_factor = Some(3);
        assert!(from_projector(&g).is_unknown(), "64 patches don't pool evenly by 3");
        let mut v = info("voxtral");
        v.audio_stack_factor = None;
        assert!(from_projector(&v).is_unknown(), "Voxtral needs its stack factor");
        let mut q = info("qwen2.5vl");
        q.spatial_merge = Some(4);
        assert!(from_projector(&q).is_unknown(), "llama.cpp refuses a merge other than 2");
        q.spatial_merge = None;
        q.image_max_pixels = Some(28 * 28 * 1280);
        assert_eq!(from_projector(&q).image_tokens, Some(1280), "the header's max pixels set the cap");
        let mut no_encoder = info("gemma3");
        no_encoder.vision = false;
        assert!(from_projector(&no_encoder).is_unknown(), "a type without its encoder flag says nothing");
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn voxtral_model() -> Vec<u8> {
        header(&[
            ("general.architecture", json!("llama")),
            ("general.type", json!("model")),
            ("llama.block_count", json!(30)),
            ("llama.context_length", json!(131072)),
            ("llama.embedding_length", json!(3072)),
            ("llama.attention.head_count", json!(32)),
            ("llama.attention.head_count_kv", json!(8)),
            ("llama.attention.key_length", json!(128)),
            ("llama.attention.value_length", json!(128)),
        ])
    }

    #[test]
    fn measures_a_model_and_projector_at_load() {
        let tmp = tempfile::tempdir().unwrap();
        let model = write(tmp.path(), "Voxtral-Mini-3B-2507-Q4_K_M.gguf", &voxtral_model());
        let mmproj = write(tmp.path(), "mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf", &projector_header("voxtral"));
        let r = measure(&model, Some(&mmproj), Some(16384)).unwrap();
        assert_eq!(r.format, LIMITS_FORMAT);
        assert_eq!(r.source, LimitSource::Measured);
        assert_eq!(r.limits, ModelLimits::new().with_audio_chunks(30, 187).with_context_tokens(16384));
        assert_eq!((r.projector.as_deref(), r.trained_context), (Some("mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf"), Some(131072)));
        assert!(r.recorded_at.is_some());
        // Without the context or a projector, those stay absent; the trained context is not used.
        let bare = measure(&model, None, None).unwrap();
        assert!(bare.limits.is_unknown());
        assert_eq!(bare.trained_context, Some(131072));
        // Wrong kinds and missing files fail.
        assert!(measure(&mmproj, None, None).is_err(), "a projector is not a model");
        assert!(measure(&model, Some(&model), None).is_err(), "a model is not a projector");
        let lora = write(tmp.path(), "lora.gguf", &crate::gguf::lora_header("llama"));
        assert!(measure(&lora, None, None).is_err());
        assert!(measure(&tmp.path().join("gone.gguf"), None, None).is_err());
    }

    #[test]
    fn records_round_trip_and_keep_unknown_fields() {
        let mut r = LimitsRecord::new(ModelLimits::new().with_context_tokens(4096), LimitSource::Operator).with_engine("llama.cpp");
        r.model_bytes = Some(10);
        let back: LimitsRecord = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        let newer: LimitsRecord =
            serde_json::from_value(json!({"format": LIMITS_FORMAT, "limits": {"max_images": 1}, "source": "probe", "app": {"x": 1}})).unwrap();
        assert_eq!((newer.source, newer.limits.max_images), (LimitSource::Other, Some(1)));
        assert_eq!(serde_json::to_value(&newer).unwrap()["app"], json!({"x": 1}));
        let empty: LimitsRecord = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, LimitsRecord::default());
        // A speech profile is kept, and absent from records without one.
        assert!(serde_json::to_value(&r).unwrap().get("speech").is_none());
        let spoken = r.clone().with_speech(SpeechProfile::new().with_presets(["af_heart"]).with_sample_rate(24_000));
        let j = serde_json::to_value(&spoken).unwrap();
        assert_eq!(j["speech"], json!({"presets": ["af_heart"], "sample_rate": 24000}));
        assert_eq!(serde_json::from_value::<LimitsRecord>(j).unwrap(), spoken);
        assert!(empty.speech.is_none());
    }

    #[test]
    fn measures_a_speech_model() {
        let tmp = tempfile::tempdir().unwrap();
        let model =
            write(tmp.path(), "Kokoro-82M-F32.gguf", &header(&[("general.architecture", json!("kokoro")), ("kokoro.context_length", json!(512))]));
        let mmproj = write(tmp.path(), "mmproj-Kokoro-82M-F16.gguf", &crate::gguf::speech_projector_header("kokoro"));
        let r = measure(&model, Some(&mmproj), None).unwrap();
        let s = r.speech.clone().expect("a speech projector gives a profile");
        assert_eq!(s.presets.as_deref().and_then(|p| p.first()).map(String::as_str), Some("af_heart"));
        assert!(r.limits.is_unknown(), "a speech projector's encoder prices no audio input");
        let back: LimitsRecord = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.speech.as_ref(), Some(&s));
        // Models that don't speak have none.
        let vox = write(tmp.path(), "v.gguf", &voxtral_model());
        let vp = write(tmp.path(), "mmproj-v.gguf", &projector_header("voxtral"));
        assert!(measure(&vox, Some(&vp), None).unwrap().speech.is_none());
        assert!(measure(&model, None, None).unwrap().speech.is_none());
    }

    #[cfg(feature = "agent")]
    #[test]
    fn converts_to_and_from_the_agent_limits() {
        let l = ModelLimits::new()
            .with_context_tokens(8192)
            .with_max_images(3)
            .with_image_tokens(256)
            .with_image_edge(896)
            .with_audio_chunks(30, 187)
            .with_max_request_bytes(1 << 20)
            .with_video_max_seconds(10)
            .with_video_max_bytes(50 << 20)
            .with_video_frames_per_minute(120);
        let m: weirdnet_agent::MediaLimits = l.into();
        assert_eq!(
            m,
            weirdnet_agent::MediaLimits::new()
                .with_max_images(3)
                .with_image_tokens(256)
                .with_image_edge(896)
                .with_audio_tokens_per_minute(374)
                .with_max_request_bytes(1 << 20)
                .with_video_max_seconds(10)
                .with_video_max_bytes(50 << 20)
                .with_video_frames_per_minute(120)
        );
        assert_eq!(m.video_frame_rate(), l.video_frame_rate());
        let back = ModelLimits::from(m);
        assert_eq!(back, ModelLimits { context_tokens: None, audio_chunk_seconds: None, audio_chunk_tokens: None, ..l }, "video round-trips");
        assert_eq!((back.video_max_seconds, back.video_max_bytes, back.video_frames_per_minute), (Some(10), Some(50 << 20), Some(120)));
        assert!(ModelLimits::from(weirdnet_agent::MediaLimits::UNKNOWN).is_unknown());
    }
}
