//! Speech out (text to speech): what a speaking model ([`Purpose::Speak`](crate::hw::Purpose::Speak))
//! does, takes and allows, as a [`SpeechProfile`].
//!
//! The profile has the same fields and the same wire form as `weirdnet_agent::speech::SpeechProfile`
//! (since `weirdnet-agent` 0.5.11), so a record written here reads there; with the `agent` feature
//! the two convert with `From` both ways. Without it this crate stays free of the agent.
//!
//! **Profiles are cited or absent.** A bool is true only when the model is known to do it; a list
//! or limit is `None` (or, for [`formats`](SpeechProfile::formats), empty) when nobody reported
//! it, never defaulted. Profiles come from:
//!
//! - the catalog ([`CatalogEntry::speech`](crate::hw::CatalogEntry::speech)), each cited to the
//!   model's card and its engine's documentation;
//! - a hand-placed GGUF's header ([`from_gguf`]), for the speech models llama.cpp documents;
//! - the engine at load (`weirdnet-backends`), kept with the measured limits
//!   ([`LimitsRecord::speech`](crate::LimitsRecord::speech)).

use serde::{Deserialize, Serialize};

use crate::gguf::GgufHeader;

/// The license name of Fish Audio Drama 3 ([`FISH_AUDIO_DRAMA_3`](crate::hw::FISH_AUDIO_DRAMA_3)),
/// the Fish Audio Community License: free for research and non-commercial use, commercial use
/// limited (an annual revenue cap, and no hosted text-to-speech service). Downloads need the
/// operator's explicit acceptance ([`Consent::accepting_license`](crate::Consent::accepting_license)).
pub const FISH_AUDIO_COMMUNITY: &str = "fish-audio-community";

/// Output sample rates llama.cpp documents for its speech models (`docs/multimodal/tts.md`).
const QWEN3_TTS_RATE: u32 = 24_000;
const KOKORO_RATE: u32 = 24_000;
const VOXCPM2_RATE: u32 = 48_000;
const DRAMA3_RATE: u32 = 44_100;
/// llama.cpp keeps the first 30 seconds of a reference clip (VoxCPM2, Drama 3).
const LLAMA_REFERENCE_SECONDS: u32 = 30;
/// Drama 3's speaker tags, `<|speaker:0|>` to `<|speaker:15|>`.
const DRAMA3_SPEAKERS: u8 = 16;

/// An audio encoding for speech out: `wav`, `pcm16` (raw little-endian 16-bit samples), `mp3`,
/// `opus` or `flac` on the wire, as `weirdnet_agent::speech::AudioFormat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioFormat {
    #[default]
    Wav,
    Pcm16,
    Mp3,
    Opus,
    Flac,
}

impl AudioFormat {
    /// Every format, in wire order.
    pub const ALL: [AudioFormat; 5] = [AudioFormat::Wav, AudioFormat::Pcm16, AudioFormat::Mp3, AudioFormat::Opus, AudioFormat::Flac];

    /// The name on the wire: `wav`, `pcm16`, `mp3`, `opus`, `flac`.
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Pcm16 => "pcm16",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Opus => "opus",
            AudioFormat::Flac => "flac",
        }
    }

    /// Parses [`as_str`](Self::as_str)'s names, and `pcm` (OpenAI's name for 16-bit PCM) as
    /// [`Pcm16`](Self::Pcm16); `None` for anything else.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "wav" => Some(AudioFormat::Wav),
            "pcm16" | "pcm" => Some(AudioFormat::Pcm16),
            "mp3" => Some(AudioFormat::Mp3),
            "opus" => Some(AudioFormat::Opus),
            "flac" => Some(AudioFormat::Flac),
            _ => None,
        }
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a voice-cloning model needs with the reference clip: `audio` (the clip; a transcript is
/// not used), `audio_and_text` (the clip; its transcript helps but may be left out) or
/// `audio_text_required` on the wire, as `weirdnet_agent::speech::CloneMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CloneMode {
    /// The reference clip only; a transcript sent with it is not used.
    Audio,
    /// The clip, with an optional transcript the model uses when given.
    AudioAndText,
    /// The clip and its transcript, both required.
    AudioTextRequired,
}

/// What one speaking model does, takes and allows. See the [module docs](self); the fields are
/// those of `weirdnet_agent::speech::SpeechProfile`, with the same names on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpeechProfile {
    /// The named preset voices, the model's default first when known. `None`: the model takes no
    /// preset by name (or none was reported).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presets: Option<Vec<String>>,
    /// Designs a voice from a description.
    #[serde(default, skip_serializing_if = "is_false")]
    pub design: bool,
    /// Takes style or emotion instructions for the chosen voice.
    #[serde(default, skip_serializing_if = "is_false")]
    pub instructions: bool,
    /// Clones a voice from a reference clip, and what the clip needs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clone: Option<CloneMode>,
    /// Speaks multi-speaker scripts with up to this many speakers (speaker numbers below it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_speakers: Option<u8>,
    /// Most voices one request may set (the voice plus per-speaker voices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_voices: Option<u8>,
    /// Takes raw engine input with no template (such as Kokoro phonemes).
    #[serde(default, skip_serializing_if = "is_false")]
    pub raw: bool,
    /// Takes a speaking speed.
    #[serde(default, skip_serializing_if = "is_false")]
    pub speed: bool,
    /// Streams audio as it is made.
    #[serde(default, skip_serializing_if = "is_false")]
    pub streaming: bool,
    /// The languages it speaks from text (ISO 639-1 or the engine's codes). `None`: not reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub languages: Option<Vec<String>>,
    /// The output formats it produces, the default first. Empty: not reported.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub formats: Vec<AudioFormat>,
    /// Its output sample rate in Hz.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Longest input in characters one request may carry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_chars: Option<u32>,
    /// Longest reference clip in seconds (what the engine uses of a longer one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_reference_seconds: Option<u32>,
    /// Largest reference clip in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_reference_bytes: Option<u64>,
    /// Longest audio one request produces, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_seconds: Option<u32>,
    /// The model's license when it restricts how speech may be served, such as
    /// [`FISH_AUDIO_COMMUNITY`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl SpeechProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_presets<S: Into<String>>(mut self, presets: impl IntoIterator<Item = S>) -> Self {
        self.presets = Some(presets.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_design(mut self, design: bool) -> Self {
        self.design = design;
        self
    }

    pub fn with_instructions(mut self, instructions: bool) -> Self {
        self.instructions = instructions;
        self
    }

    pub fn with_clone(mut self, clone: CloneMode) -> Self {
        self.clone = Some(clone);
        self
    }

    pub fn with_script_speakers(mut self, speakers: u8) -> Self {
        self.script_speakers = Some(speakers);
        self
    }

    pub fn with_max_voices(mut self, voices: u8) -> Self {
        self.max_voices = Some(voices);
        self
    }

    pub fn with_raw(mut self, raw: bool) -> Self {
        self.raw = raw;
        self
    }

    pub fn with_speed(mut self, speed: bool) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    pub fn with_languages<S: Into<String>>(mut self, languages: impl IntoIterator<Item = S>) -> Self {
        self.languages = Some(languages.into_iter().map(Into::into).collect());
        self
    }

    /// The output formats, the default first.
    pub fn with_formats(mut self, formats: impl IntoIterator<Item = AudioFormat>) -> Self {
        self.formats = formats.into_iter().collect();
        self
    }

    pub fn with_sample_rate(mut self, hz: u32) -> Self {
        self.sample_rate = Some(hz);
        self
    }

    pub fn with_max_input_chars(mut self, chars: u32) -> Self {
        self.max_input_chars = Some(chars);
        self
    }

    pub fn with_max_reference_seconds(mut self, seconds: u32) -> Self {
        self.max_reference_seconds = Some(seconds);
        self
    }

    pub fn with_max_reference_bytes(mut self, bytes: u64) -> Self {
        self.max_reference_bytes = Some(bytes);
        self
    }

    pub fn with_max_output_seconds(mut self, seconds: u32) -> Self {
        self.max_output_seconds = Some(seconds);
        self
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Nothing is reported (every field unset).
    pub fn is_unknown(&self) -> bool {
        *self == Self::default()
    }

    /// The format a request without one gets: the first reported.
    pub fn default_format(&self) -> Option<AudioFormat> {
        self.formats.first().copied()
    }

    /// Each field from `self` when it is set, else from `fallback`: what an engine reports at
    /// load over a catalog entry or a header (`reported.or(&cataloged)`). A bool is set when it is
    /// true; empty formats count as unset. Nothing here narrows: operator overrides go through
    /// `weirdnet_agent::speech::SpeechProfile::narrowed_by`.
    pub fn or(&self, fallback: &Self) -> Self {
        Self {
            presets: self.presets.clone().or_else(|| fallback.presets.clone()),
            design: self.design || fallback.design,
            instructions: self.instructions || fallback.instructions,
            clone: self.clone.or(fallback.clone),
            script_speakers: self.script_speakers.or(fallback.script_speakers),
            max_voices: self.max_voices.or(fallback.max_voices),
            raw: self.raw || fallback.raw,
            speed: self.speed || fallback.speed,
            streaming: self.streaming || fallback.streaming,
            languages: self.languages.clone().or_else(|| fallback.languages.clone()),
            formats: if self.formats.is_empty() { fallback.formats.clone() } else { self.formats.clone() },
            sample_rate: self.sample_rate.or(fallback.sample_rate),
            max_input_chars: self.max_input_chars.or(fallback.max_input_chars),
            max_reference_seconds: self.max_reference_seconds.or(fallback.max_reference_seconds),
            max_reference_bytes: self.max_reference_bytes.or(fallback.max_reference_bytes),
            max_output_seconds: self.max_output_seconds.or(fallback.max_output_seconds),
            license: self.license.clone().or_else(|| fallback.license.clone()),
        }
    }
}

/// What llama-server's `/v1/audio/speech` produces for every speech model: `wav` (the default)
/// and `pcm` (16-bit), streamed with `stream_format: "audio"`.
pub(crate) fn llama_server() -> SpeechProfile {
    SpeechProfile::new().with_formats([AudioFormat::Wav, AudioFormat::Pcm16]).with_streaming(true)
}

/// What vLLM-Omni's `/v1/audio/speech` produces: `wav`, `mp3`, `flac`, `pcm` and `opus`,
/// streamed with `stream: true`, with a `speed`.
pub(crate) fn vllm_omni() -> SpeechProfile {
    SpeechProfile::new()
        .with_formats([AudioFormat::Wav, AudioFormat::Mp3, AudioFormat::Flac, AudioFormat::Pcm16, AudioFormat::Opus])
        .with_streaming(true)
        .with_speed(true)
}

/// Qwen3-TTS's ten languages (its model card; llama.cpp's `--tts-lang` codes).
pub(crate) const QWEN3_TTS_LANGUAGES: [&str; 10] = ["zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"];
/// Kokoro speaks English from text (`en-us`, `en-gb`); other voices need phonemes (`raw`).
pub(crate) const KOKORO_LANGUAGES: [&str; 2] = ["en-us", "en-gb"];

/// The speech profile a hand-placed GGUF speech model's headers give, for llama-server (the
/// engine that runs GGUF speech models): `None` unless the projector generates audio
/// (`clip.has_gen_audio_encoder`), which is how llama.cpp marks a speech model's `mmproj`.
///
/// Per projector type (`clip.gen.audio.projector_type`), following llama.cpp's
/// `docs/multimodal/tts.md` (2026-11):
///
/// - `qwen3tts_gen` (Qwen3-TTS): ten languages, 24 kHz; cloning from the clip when the projector
///   has the speaker encoder (`clip.has_audio_encoder`). A transcript only helps with a speech
///   tokenizer encoder the header doesn't name, so the clone mode is `audio`.
/// - `kokoro` (Kokoro): the preset voices stored in the projector (`voice.*` tensors, the
///   default `clip.gen.audio.default_voice` first), English from text, raw phonemes, speed,
///   24 kHz.
/// - `voxcpm2_gen` (VoxCPM2): voice design (a description in parentheses), cloning from the clip
///   (its first 30 s; no transcript) with the AudioVAE encoder, 48 kHz.
/// - `drama3_gen` (Fish Audio Drama 3): scripts of 16 speakers; cloning one voice from a clip
///   and its transcript only with a projector that has the codec encoder
///   (`clip.has_audio_encoder`), 44.1 kHz.
/// - Any other type: what llama-server does for every speech model.
///
/// Every profile has llama-server's formats (`wav`, `pcm16`) and streaming, the languages in the
/// model's `general.languages` when the list above has none, the presets stored in the
/// projector, and the license when it restricts use (not a permissive SPDX id; Fish Audio's
/// licenses as [`FISH_AUDIO_COMMUNITY`]).
pub fn from_gguf(model: &GgufHeader, projector: &GgufHeader) -> Option<SpeechProfile> {
    let p = projector.projector.as_ref().filter(|p| p.speech)?;
    let mut s = llama_server();
    let voices = || {
        let mut v = p.voices.clone();
        if let Some(d) = &p.default_voice
            && let Some(i) = v.iter().position(|x| x == d)
        {
            let d = v.remove(i);
            v.insert(0, d);
        }
        v
    };
    match p.speech_type() {
        Some("qwen3tts_gen") => {
            s = s.with_languages(QWEN3_TTS_LANGUAGES).with_sample_rate(QWEN3_TTS_RATE);
            if p.audio {
                s = s.with_clone(CloneMode::Audio);
            }
        }
        Some("kokoro") => {
            s = s.with_languages(KOKORO_LANGUAGES).with_raw(true).with_speed(true).with_sample_rate(KOKORO_RATE);
        }
        Some("voxcpm2_gen") => {
            s = s.with_design(true).with_sample_rate(VOXCPM2_RATE);
            if p.audio {
                s = s.with_clone(CloneMode::Audio).with_max_reference_seconds(LLAMA_REFERENCE_SECONDS);
            }
        }
        Some("drama3_gen") => {
            s = s.with_script_speakers(DRAMA3_SPEAKERS).with_sample_rate(DRAMA3_RATE);
            if p.audio {
                s = s.with_clone(CloneMode::AudioTextRequired).with_max_voices(1).with_max_reference_seconds(LLAMA_REFERENCE_SECONDS);
            }
        }
        _ => {}
    }
    let v = voices();
    if !v.is_empty() {
        s.presets = Some(v);
    }
    if s.languages.is_none() {
        let langs = if model.languages.is_empty() { &projector.languages } else { &model.languages };
        if !langs.is_empty() {
            s.languages = Some(langs.clone());
        }
    }
    s.license = restricting_license(model.license.as_deref()).or_else(|| restricting_license(projector.license.as_deref()));
    Some(s)
}

/// Licenses that don't restrict how speech is served (lower-case SPDX ids).
const PERMISSIVE: [&str; 9] = ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense", "mpl-2.0"];

/// `license` (a header's or a card's) when it restricts use: anything but a permissive SPDX id,
/// with Fish Audio's license names as [`FISH_AUDIO_COMMUNITY`]. `None` when absent or permissive.
pub fn restricting_license(license: Option<&str>) -> Option<String> {
    let l = license?.trim().to_ascii_lowercase();
    if l.is_empty() || PERMISSIVE.contains(&l.as_str()) {
        return None;
    }
    if l.starts_with("fish-audio") {
        return Some(FISH_AUDIO_COMMUNITY.into());
    }
    Some(l)
}

#[cfg(feature = "agent")]
mod agent_conv {
    use super::*;
    use weirdnet_agent::speech as a;

    impl From<AudioFormat> for a::AudioFormat {
        fn from(f: AudioFormat) -> Self {
            match f {
                AudioFormat::Wav => a::AudioFormat::Wav,
                AudioFormat::Pcm16 => a::AudioFormat::Pcm16,
                AudioFormat::Mp3 => a::AudioFormat::Mp3,
                AudioFormat::Opus => a::AudioFormat::Opus,
                AudioFormat::Flac => a::AudioFormat::Flac,
            }
        }
    }

    impl AudioFormat {
        /// The agent's format, if this version knows it.
        pub fn from_agent(f: a::AudioFormat) -> Option<Self> {
            AudioFormat::parse(f.as_str())
        }
    }

    impl From<CloneMode> for a::CloneMode {
        fn from(c: CloneMode) -> Self {
            match c {
                CloneMode::Audio => a::CloneMode::Audio,
                CloneMode::AudioAndText => a::CloneMode::AudioAndText,
                CloneMode::AudioTextRequired => a::CloneMode::AudioTextRequired,
            }
        }
    }

    impl CloneMode {
        /// The agent's clone mode, if this version knows it.
        pub fn from_agent(c: a::CloneMode) -> Option<Self> {
            match c {
                a::CloneMode::Audio => Some(CloneMode::Audio),
                a::CloneMode::AudioAndText => Some(CloneMode::AudioAndText),
                a::CloneMode::AudioTextRequired => Some(CloneMode::AudioTextRequired),
                _ => None,
            }
        }
    }

    impl SpeechProfile {
        /// The same profile as `weirdnet_agent::speech::SpeechProfile`, which catalog entries and
        /// `ModelStatus` carry.
        pub fn to_agent(&self) -> a::SpeechProfile {
            let mut p = a::SpeechProfile::new();
            p.presets = self.presets.clone();
            p.design = self.design;
            p.instructions = self.instructions;
            p.clone = self.clone.map(Into::into);
            p.script_speakers = self.script_speakers;
            p.max_voices = self.max_voices;
            p.raw = self.raw;
            p.speed = self.speed;
            p.streaming = self.streaming;
            p.languages = self.languages.clone();
            p.formats = self.formats.iter().copied().map(Into::into).collect();
            p.sample_rate = self.sample_rate;
            p.max_input_chars = self.max_input_chars;
            p.max_reference_seconds = self.max_reference_seconds;
            p.max_reference_bytes = self.max_reference_bytes;
            p.max_output_seconds = self.max_output_seconds;
            p.license = self.license.clone();
            p
        }

        /// An agent profile as this crate's. A clone mode or format this version doesn't know is
        /// left out (the clone mode as no cloning), never guessed.
        pub fn from_agent(p: &a::SpeechProfile) -> Self {
            Self {
                presets: p.presets.clone(),
                design: p.design,
                instructions: p.instructions,
                clone: p.clone.and_then(CloneMode::from_agent),
                script_speakers: p.script_speakers,
                max_voices: p.max_voices,
                raw: p.raw,
                speed: p.speed,
                streaming: p.streaming,
                languages: p.languages.clone(),
                formats: p.formats.iter().copied().filter_map(AudioFormat::from_agent).collect(),
                sample_rate: p.sample_rate,
                max_input_chars: p.max_input_chars,
                max_reference_seconds: p.max_reference_seconds,
                max_reference_bytes: p.max_reference_bytes,
                max_output_seconds: p.max_output_seconds,
                license: p.license.clone(),
            }
        }
    }

    impl From<SpeechProfile> for a::SpeechProfile {
        fn from(p: SpeechProfile) -> Self {
            p.to_agent()
        }
    }

    impl From<a::SpeechProfile> for SpeechProfile {
        fn from(p: a::SpeechProfile) -> Self {
            Self::from_agent(&p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{header, read_header, speech_projector_header};
    use serde_json::json;

    fn model(arch: &str, extra: &[(&str, serde_json::Value)]) -> GgufHeader {
        let mut kv = vec![("general.architecture", json!(arch))];
        kv.extend_from_slice(extra);
        read_header(&header(&kv)[..]).unwrap()
    }

    fn proj(which: &str) -> GgufHeader {
        read_header(&speech_projector_header(which)[..]).unwrap()
    }

    #[test]
    fn unknown_is_empty_on_the_wire_and_fields_round_trip() {
        assert!(SpeechProfile::default().is_unknown());
        assert_eq!(serde_json::to_string(&SpeechProfile::new()).unwrap(), "{}");
        let p = SpeechProfile::new()
            .with_presets(["vivian"])
            .with_clone(CloneMode::AudioAndText)
            .with_formats([AudioFormat::Wav, AudioFormat::Pcm16])
            .with_sample_rate(24_000)
            .with_streaming(true)
            .with_license(FISH_AUDIO_COMMUNITY);
        let j = serde_json::to_value(&p).unwrap();
        assert_eq!(
            j,
            json!({"presets": ["vivian"], "clone": "audio_and_text", "streaming": true, "formats": ["wav", "pcm16"], "sample_rate": 24000, "license": "fish-audio-community"})
        );
        assert_eq!(serde_json::from_value::<SpeechProfile>(j).unwrap(), p);
        assert_eq!(p.default_format(), Some(AudioFormat::Wav));
        // Fields from a newer version are ignored.
        let newer: SpeechProfile = serde_json::from_value(json!({"design": true, "emotions": ["happy"]})).unwrap();
        assert_eq!(newer, SpeechProfile::new().with_design(true));
        for f in AudioFormat::ALL {
            assert_eq!(AudioFormat::parse(f.as_str()), Some(f));
            assert_eq!(serde_json::to_value(f).unwrap(), json!(f.as_str()));
        }
        assert_eq!(AudioFormat::parse("PCM"), Some(AudioFormat::Pcm16));
        assert_eq!(AudioFormat::parse("aac"), None);
    }

    #[test]
    fn reported_over_cataloged() {
        let catalog = SpeechProfile::new().with_presets(["a", "b"]).with_languages(["en"]).with_formats([AudioFormat::Wav]).with_speed(true);
        let engine = SpeechProfile::new().with_presets(["a"]).with_sample_rate(24_000);
        let both = engine.or(&catalog);
        assert_eq!(both.presets, Some(vec!["a".to_string()]));
        assert_eq!((both.languages.as_deref(), both.sample_rate, both.speed), (Some(&["en".to_string()][..]), Some(24_000), true));
        assert_eq!(both.formats, [AudioFormat::Wav]);
        assert_eq!(SpeechProfile::new().or(&catalog), catalog);
    }

    #[test]
    fn licenses_that_restrict() {
        assert_eq!(restricting_license(None), None);
        assert_eq!(restricting_license(Some("Apache-2.0")), None);
        assert_eq!(restricting_license(Some("mit")), None);
        assert_eq!(restricting_license(Some("fish-audio-research-license")).as_deref(), Some(FISH_AUDIO_COMMUNITY));
        assert_eq!(restricting_license(Some("fish-audio-community")).as_deref(), Some(FISH_AUDIO_COMMUNITY));
        assert_eq!(restricting_license(Some("cc-by-nc-4.0")).as_deref(), Some("cc-by-nc-4.0"));
    }

    #[test]
    fn hand_placed_gguf_speech_models() {
        // Kokoro: the projector's voices, the default first; English from text, phonemes, speed.
        let k = from_gguf(&model("kokoro", &[("general.license", json!("apache-2.0"))]), &proj("kokoro")).unwrap();
        assert_eq!(k.presets.as_deref(), Some(&["af_heart".to_string(), "af_alloy".into(), "bm_george".into()][..]));
        assert!(k.raw && k.speed && k.streaming && k.clone.is_none() && !k.design);
        assert_eq!(
            (k.languages.unwrap(), k.sample_rate, k.formats),
            (vec!["en-us".to_string(), "en-gb".into()], Some(24_000), vec![AudioFormat::Wav, AudioFormat::Pcm16])
        );
        assert_eq!(k.license, None);
        // Qwen3-TTS Base with the speaker encoder: clones from the clip alone.
        let q = from_gguf(&model("qwen3tts", &[]), &proj("qwen3tts")).unwrap();
        assert_eq!((q.clone, q.sample_rate, q.languages.as_ref().map(Vec::len)), (Some(CloneMode::Audio), Some(24_000), Some(10)));
        assert!(!q.instructions && !q.speed && q.presets.is_none());
        // VoxCPM2: design and cloning; languages from the header.
        let v = from_gguf(&model("minicpm", &[("general.languages", json!(["zh", "en", "fr"]))]), &proj("voxcpm2")).unwrap();
        assert!(v.design);
        assert_eq!((v.clone, v.max_reference_seconds, v.sample_rate), (Some(CloneMode::Audio), Some(30), Some(48_000)));
        assert_eq!(v.languages.unwrap(), ["zh", "en", "fr"]);
        // Drama 3 as ggml-org publishes it: scripts, no cloning (no codec encoder), restricted.
        let d = from_gguf(
            &model("qwen35moe", &[("general.license", json!("other")), ("general.license.name", json!("fish-audio-research-license"))]),
            &proj("drama3"),
        )
        .unwrap();
        assert_eq!((d.script_speakers, d.clone, d.max_voices, d.sample_rate), (Some(16), None, None, Some(44_100)));
        assert_eq!(d.license.as_deref(), Some(FISH_AUDIO_COMMUNITY));
        // With the codec encoder: one cloned voice, its transcript required.
        let dc = from_gguf(&model("qwen35moe", &[]), &proj("drama3-codec")).unwrap();
        assert_eq!((dc.clone, dc.max_voices, dc.max_reference_seconds), (Some(CloneMode::AudioTextRequired), Some(1), Some(30)));
        assert_eq!(dc.license.as_deref(), Some(FISH_AUDIO_COMMUNITY), "the projector's license counts too");
        // Projectors that don't generate audio give no profile.
        let voxtral = read_header(&crate::gguf::projector_header("voxtral")[..]).unwrap();
        assert_eq!(from_gguf(&model("llama", &[]), &voxtral), None);
    }

    #[cfg(feature = "agent")]
    #[test]
    fn same_wire_form_as_the_agent() {
        use weirdnet_agent::speech as a;
        for e in crate::hw::by_purpose(crate::hw::Purpose::Speak) {
            let p = e.speech().unwrap();
            let agent = p.to_agent();
            assert_eq!(serde_json::to_value(&p).unwrap(), serde_json::to_value(&agent).unwrap(), "{}", e.id());
            let back: a::SpeechProfile = serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
            assert_eq!(back, agent);
            assert_eq!(SpeechProfile::from_agent(&agent), p);
            assert_eq!(SpeechProfile::from(a::SpeechProfile::from(p.clone())), p);
        }
        let full = SpeechProfile::new()
            .with_presets(["x"])
            .with_design(true)
            .with_instructions(true)
            .with_clone(CloneMode::AudioTextRequired)
            .with_script_speakers(4)
            .with_max_voices(2)
            .with_raw(true)
            .with_speed(true)
            .with_streaming(true)
            .with_languages(["en"])
            .with_formats(AudioFormat::ALL)
            .with_sample_rate(44_100)
            .with_max_input_chars(2000)
            .with_max_reference_seconds(30)
            .with_max_reference_bytes(1 << 20)
            .with_max_output_seconds(180)
            .with_license("x");
        assert_eq!(serde_json::to_value(&full).unwrap(), serde_json::to_value(full.to_agent()).unwrap());
        assert_eq!(SpeechProfile::from_agent(&full.to_agent()), full);
        assert!(full.to_agent().covers(&a::SpeechNeeds::default()).is_ok());
    }
}
