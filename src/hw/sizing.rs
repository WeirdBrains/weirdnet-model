//! The usable memory budget and which models fit in it.

use crate::ModelSpec;
use crate::speech::{self, CloneMode, SpeechProfile};
use serde::{Deserialize, Serialize};

use super::{Accel, GIB, Hardware};

/// What a model is for on a node (its role): `chat`, `embedding`, `transcribe` or `speak` on the
/// wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Purpose {
    Chat,
    Embedding,
    /// Speech to text: audio in (an `input_audio` part), its transcript out, so a soundtrack can
    /// become text for models that don't take audio. Never picked as a chat model.
    Transcribe,
    /// Text to speech: text (or a multi-speaker script) in, audio out, in the voices its
    /// [`SpeechProfile`] lists ([`CatalogEntry::speech`]). Never picked as a chat model, never
    /// in a [`plan`]. Since 0.5.5.
    Speak,
}

impl Purpose {
    /// Every purpose, in wire order.
    pub const ALL: [Purpose; 4] = [Purpose::Chat, Purpose::Embedding, Purpose::Transcribe, Purpose::Speak];

    /// The name on the wire and in settings: `chat`, `embedding`, `transcribe` or `speak`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Purpose::Chat => "chat",
            Purpose::Embedding => "embedding",
            Purpose::Transcribe => "transcribe",
            Purpose::Speak => "speak",
        }
    }

    /// The purpose named `name` (as [`as_str`](Self::as_str) writes it), if any.
    pub fn parse(name: &str) -> Option<Purpose> {
        Purpose::ALL.into_iter().find(|p| p.as_str() == name)
    }
}

impl std::fmt::Display for Purpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The speech-to-text entry ([`Purpose::Transcribe`]): Voxtral Mini 3B, which llama.cpp runs
/// with its projector ([`CatalogEntry::projector`]).
pub const VOXTRAL_MINI_3B: &str = "voxtral-mini-3b";

/// Text-to-speech entries ([`Purpose::Speak`]), one per checkpoint (each mode of Qwen3-TTS is
/// its own checkpoint, so its own id). Since 0.5.5.
///
/// Qwen3-TTS 0.6B CustomVoice: nine preset voices, for vLLM-Omni.
pub const QWEN3_TTS_0_6B_CUSTOMVOICE: &str = "qwen3-tts-0.6b-customvoice";
/// Qwen3-TTS 0.6B Base: voice cloning, for vLLM-Omni.
pub const QWEN3_TTS_0_6B_BASE: &str = "qwen3-tts-0.6b-base";
/// Qwen3-TTS 1.7B CustomVoice: nine preset voices with style instructions, for vLLM-Omni.
pub const QWEN3_TTS_1_7B_CUSTOMVOICE: &str = "qwen3-tts-1.7b-customvoice";
/// Qwen3-TTS 1.7B VoiceDesign: a voice from a description, for vLLM-Omni.
pub const QWEN3_TTS_1_7B_VOICEDESIGN: &str = "qwen3-tts-1.7b-voicedesign";
/// Qwen3-TTS 1.7B Base: voice cloning, as ggml-org's GGUF with its projector, for llama.cpp.
pub const QWEN3_TTS_1_7B_BASE: &str = "qwen3-tts-1.7b-base";
/// Kokoro 82M: 54 preset voices, as ggml-org's GGUF with its projector, for llama.cpp.
pub const KOKORO_82M: &str = "kokoro-82m";
/// VoxCPM2: voice design and cloning, as ggml-org's GGUF with its projector, for llama.cpp.
pub const VOXCPM2: &str = "voxcpm2";
/// Fish Audio Drama 3: multi-speaker scripts, as ggml-org's GGUF with its projector, for
/// llama.cpp. Under the Fish Audio Community License ([`speech::FISH_AUDIO_COMMUNITY`]):
/// [`CatalogEntry::restricted`], so [`speaker`] never picks it and a download needs the
/// operator's explicit acceptance ([`Consent::accepting_license`](crate::Consent::accepting_license)).
pub const FISH_AUDIO_DRAMA_3: &str = "fish-audio-drama-3";

/// Bytes of Qwen3-TTS's speech tokenizer (`speech_tokenizer/model.safetensors`, the same file in
/// every Qwen3-TTS repo), which vLLM-Omni loads next to the checkpoint.
const QWEN3_TTS_TOKENIZER_BYTES: u64 = 682_293_092;
/// Where Drama 3's weights come from, for the license check on hand-built specs.
const DRAMA_3_UPSTREAM: &str = "fishaudio/drama-3";

/// A catalog model with what it needs in memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub spec: ModelSpec,
    pub purpose: Purpose,
    pub quant: String,
    /// Total and active parameters, in billions (they differ for mixture-of-experts models).
    pub params_b: f32,
    pub active_params_b: f32,
    /// KV cache bytes per token at f16 (layers x KV heads x head dim x 2 x 2).
    pub kv_bytes_per_token: u64,
    /// Context the sizing assumes per slot.
    pub context: u32,
}

impl CatalogEntry {
    pub fn id(&self) -> &str {
        &self.spec.id
    }

    /// Weights (and the projector's, for entries that need one, and Qwen3-TTS's speech tokenizer
    /// for its vLLM-Omni checkpoints) plus KV for `slots` concurrent requests at the entry's
    /// context, plus 10% for compute buffers.
    pub fn footprint(&self, slots: u32) -> u64 {
        let kv = self.kv_bytes_per_token * self.context as u64 * slots as u64;
        let weights = self.spec.size + self.projector().map_or(0, |p| p.size) + self.companion_bytes();
        (weights + kv) + (weights + kv) / 10
    }

    fn companion_bytes(&self) -> u64 {
        match self.spec.id.as_str() {
            QWEN3_TTS_0_6B_CUSTOMVOICE | QWEN3_TTS_0_6B_BASE | QWEN3_TTS_1_7B_CUSTOMVOICE | QWEN3_TTS_1_7B_VOICEDESIGN => QWEN3_TTS_TOKENIZER_BYTES,
            _ => 0,
        }
    }

    /// Whether the entry is a GGUF file llama.cpp runs (with its [`projector`](Self::projector)
    /// when it has one). The others are Hugging Face checkpoints for vLLM-Omni: `spec.repo` is the
    /// repo the engine serves, and `spec.file`, `size` and `sha256` pin its main weights (the
    /// engine fetches the rest of the repo). Since 0.5.5.
    pub fn is_gguf(&self) -> bool {
        self.spec.file.to_ascii_lowercase().ends_with(".gguf")
    }

    /// Whether this is a text-to-speech model ([`Purpose::Speak`]). Since 0.5.5.
    pub fn speaks(&self) -> bool {
        self.purpose == Purpose::Speak
    }

    /// The model's license as its card names it (an SPDX id such as `apache-2.0`, or
    /// [`speech::FISH_AUDIO_COMMUNITY`]), for the entries that record it: the speech-to-text and
    /// text-to-speech models. `None` for the others. Since 0.5.5.
    pub fn license(&self) -> Option<&'static str> {
        match self.spec.id.as_str() {
            FISH_AUDIO_DRAMA_3 => Some(speech::FISH_AUDIO_COMMUNITY),
            VOXTRAL_MINI_3B
            | QWEN3_TTS_0_6B_CUSTOMVOICE
            | QWEN3_TTS_0_6B_BASE
            | QWEN3_TTS_1_7B_CUSTOMVOICE
            | QWEN3_TTS_1_7B_VOICEDESIGN
            | QWEN3_TTS_1_7B_BASE
            | KOKORO_82M
            | VOXCPM2 => Some("apache-2.0"),
            _ => None,
        }
    }

    /// Whether the license restricts use ([`speech::restricting_license`]): such an entry is never
    /// picked by [`speaker`] or a [`plan`], and [`download`](crate::download()) refuses it (and its
    /// projector) without a [`Consent::accepting_license`](crate::Consent::accepting_license) for
    /// that license. Since 0.5.5.
    pub fn restricted(&self) -> bool {
        speech::restricting_license(self.license()).is_some()
    }

    /// What a text-to-speech entry does, cited to the model's card and its engine's documentation
    /// (llama.cpp `docs/multimodal/tts.md` and the llama-server README for GGUF entries; the
    /// vLLM-Omni speech API for the others), read 2026-11-27. Never measured figures: what an
    /// engine reports at load goes over it ([`SpeechProfile::or`]). `None` for entries that
    /// don't speak. Since 0.5.5.
    ///
    /// - Qwen3-TTS (24 kHz, ten languages): CustomVoice has nine preset voices (the 1.7B also
    ///   takes style instructions; the 0.6B doesn't), VoiceDesign designs a voice from a
    ///   description, Base clones a voice from a clip whose transcript is optional on vLLM-Omni;
    ///   ggml-org's 1.7B Base projector has the speaker encoder only, so llama.cpp clones from
    ///   the clip alone.
    /// - Kokoro (24 kHz): the 54 voices in ggml-org's projector, `af_heart` (the default) first;
    ///   English from text (`en-us`, `en-gb`), other languages as raw phonemes; speed.
    /// - VoxCPM2 (48 kHz, 30 languages from the text): voice design, cloning from a clip (its
    ///   first 30 s; no transcript).
    /// - Fish Audio Drama 3 (44.1 kHz, 83 languages from the text): scripts of up to 16
    ///   speakers. ggml-org's projector has no codec encoder, so it does not clone ("Voice
    ///   cloning is not supported yet", its card). License [`speech::FISH_AUDIO_COMMUNITY`].
    pub fn speech(&self) -> Option<SpeechProfile> {
        if !self.speaks() {
            return None;
        }
        let qwen_vllm = || speech::vllm_omni().with_languages(speech::QWEN3_TTS_LANGUAGES).with_sample_rate(24_000);
        Some(match self.spec.id.as_str() {
            QWEN3_TTS_0_6B_CUSTOMVOICE => qwen_vllm().with_presets(QWEN3_TTS_SPEAKERS),
            QWEN3_TTS_1_7B_CUSTOMVOICE => qwen_vllm().with_presets(QWEN3_TTS_SPEAKERS).with_instructions(true),
            QWEN3_TTS_1_7B_VOICEDESIGN => qwen_vllm().with_design(true),
            QWEN3_TTS_0_6B_BASE => qwen_vllm().with_clone(CloneMode::AudioAndText),
            QWEN3_TTS_1_7B_BASE => {
                speech::llama_server().with_languages(speech::QWEN3_TTS_LANGUAGES).with_sample_rate(24_000).with_clone(CloneMode::Audio)
            }
            KOKORO_82M => speech::llama_server()
                .with_presets(KOKORO_VOICES)
                .with_languages(speech::KOKORO_LANGUAGES)
                .with_raw(true)
                .with_speed(true)
                .with_sample_rate(24_000),
            VOXCPM2 => speech::llama_server()
                .with_design(true)
                .with_clone(CloneMode::Audio)
                .with_max_reference_seconds(30)
                .with_languages(VOXCPM2_LANGUAGES)
                .with_sample_rate(48_000),
            FISH_AUDIO_DRAMA_3 => speech::llama_server()
                .with_script_speakers(16)
                .with_languages(DRAMA_3_LANGUAGES)
                .with_sample_rate(44_100)
                .with_license(speech::FISH_AUDIO_COMMUNITY),
            _ => return None,
        })
    }

    /// Repos whose files fall under this entry's license (its own, and where the weights come
    /// from), for [`restricted_license`].
    fn license_repos(&self) -> Vec<&str> {
        let mut repos = vec![self.spec.repo.as_str()];
        if self.spec.id == FISH_AUDIO_DRAMA_3 {
            repos.push(DRAMA_3_UPSTREAM);
        }
        repos
    }

    /// The multimodal projector (llama.cpp `--mmproj`) the entry needs, pinned like the model
    /// file: the audio encoder for [`VOXTRAL_MINI_3B`], and the audio generator (codec, voices,
    /// reference encoder) of each GGUF text-to-speech entry. `None` for text models and the
    /// vLLM-Omni checkpoints.
    /// Download it with the same consent gate as the model.
    pub fn projector(&self) -> Option<ModelSpec> {
        match self.spec.id.as_str() {
            VOXTRAL_MINI_3B => Some(spec(
                "voxtral-mini-3b-mmproj",
                "Voxtral Mini 3B audio encoder (Q8_0)",
                "ggml-org/Voxtral-Mini-3B-2507-GGUF",
                "mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf",
                715_714_080,
                "4f24c4ef3ce929d02ed9d1cfb050ae9a7365f057c0ddec0d489580982ebe0d02",
            )),
            QWEN3_TTS_1_7B_BASE => Some(spec(
                "qwen3-tts-1.7b-base-mmproj",
                "Qwen3-TTS 1.7B Base speaker encoder and codec (Q8_0)",
                "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF",
                "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
                446_422_912,
                "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2",
            )),
            KOKORO_82M => Some(spec(
                "kokoro-82m-mmproj",
                "Kokoro 82M decoder and voices (F16)",
                "ggml-org/Kokoro-82M-GGUF",
                "mmproj-Kokoro-82M-F16.gguf",
                208_456_512,
                "071d20f53b7c5d64b08a24bff5dfb77c2ea0df843a7c6c2356998a735ec6929d",
            )),
            VOXCPM2 => Some(spec(
                "voxcpm2-mmproj",
                "VoxCPM2 LocDiT and AudioVAE (BF16)",
                "ggml-org/VoxCPM2-GGUF",
                "mmproj-VoxCPM2-BF16.gguf",
                2_007_741_504,
                "2d0be367461d11e12a8544487491936a8f1d61d70d5313bb25dd50a4693be2ee",
            )),
            FISH_AUDIO_DRAMA_3 => Some(spec(
                "fish-audio-drama-3-mmproj",
                "Fish Audio Drama 3 codebook tower and codec (BF16)",
                "ggml-org/drama-3-GGUF",
                "mmproj-drama-3-BF16.gguf",
                2_942_644_160,
                "c29b9c90cf180cf81da39f31f023ba33641d54d7898e5e9be032c4d111ecbced",
            )),
            _ => None,
        }
    }

    /// Whether this is a speech-to-text model ([`Purpose::Transcribe`]).
    pub fn transcribes(&self) -> bool {
        self.purpose == Purpose::Transcribe
    }
}

fn spec(id: &str, label: &str, repo: &str, file: &str, size: u64, sha256: &str) -> ModelSpec {
    ModelSpec { id: id.into(), label: label.into(), repo: repo.into(), file: file.into(), size, sha256: Some(sha256.into()) }
}

#[allow(clippy::too_many_arguments)]
fn chat(spec: ModelSpec, quant: &str, params_b: f32, active_params_b: f32, kv_bytes_per_token: u64, context: u32) -> CatalogEntry {
    CatalogEntry { spec, purpose: Purpose::Chat, quant: quant.into(), params_b, active_params_b, kv_bytes_per_token, context }
}

#[allow(clippy::too_many_arguments)]
fn speaker_entry(spec: ModelSpec, quant: &str, params_b: f32, active_params_b: f32, kv_bytes_per_token: u64, context: u32) -> CatalogEntry {
    CatalogEntry { spec, purpose: Purpose::Speak, quant: quant.into(), params_b, active_params_b, kv_bytes_per_token, context }
}

/// Qwen3-TTS CustomVoice's preset speakers (its model card, as `config.json` names them).
const QWEN3_TTS_SPEAKERS: [&str; 9] = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"];

/// Kokoro's voices (`VOICES.md`; all 54 are in ggml-org's projector), the default first. The first
/// letter is the language: American and British English, Spanish, French, Hindi, Italian,
/// Japanese, Brazilian Portuguese, Mandarin.
const KOKORO_VOICES: [&str; 54] = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
];

/// VoxCPM2's 30 languages (its model card's `language` list).
const VOXCPM2_LANGUAGES: [&str; 30] = [
    "zh", "en", "ar", "my", "da", "nl", "fi", "fr", "de", "el", "he", "hi", "id", "it", "ja", "km", "ko", "lo", "ms", "no", "pl", "pt", "ru", "es",
    "sw", "sv", "tl", "th", "tr", "vi",
];

/// Drama 3's 83 languages (its model card's `language` list).
const DRAMA_3_LANGUAGES: [&str; 83] = [
    "zh", "en", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "sv", "it", "tr", "no", "nl", "cy", "eu", "ca", "da", "gl", "ta", "hu", "fi", "pl",
    "et", "hi", "la", "ur", "th", "vi", "jw", "bn", "yo", "sl", "cs", "sw", "nn", "he", "ms", "uk", "id", "kk", "bg", "lv", "my", "tl", "sk", "ne",
    "fa", "af", "el", "bo", "hr", "ro", "sn", "mi", "yi", "am", "be", "km", "is", "az", "sd", "br", "sq", "ps", "mn", "ht", "ml", "sr", "sa", "te",
    "ka", "bs", "pa", "lt", "kn", "si", "hy", "mr", "as", "gu", "fo",
];

/// The tiered catalog: chat models smallest first, then the embedding model, then the
/// speech-to-text model, then the text-to-speech models. Sizes and SHA-256 from the Hugging Face
/// API (2026-11-19; Voxtral 2026-11-26; text to speech 2026-11-27); the Gemma 4 files are
/// `weirdnet-model`'s own pinned entries.
pub fn catalog() -> Vec<CatalogEntry> {
    let gemma = |id: &str| crate::catalog::find(id).expect("weirdnet-model ships Gemma 4");
    vec![
        chat(
            spec(
                "smollm2-135m",
                "SmolLM2 135M instruct (Q8_0)",
                "bartowski/SmolLM2-135M-Instruct-GGUF",
                "SmolLM2-135M-Instruct-Q8_0.gguf",
                144_811_360,
                "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83",
            ),
            "Q8_0",
            0.135,
            0.135,
            23_040,
            8192,
        ),
        chat(gemma(crate::catalog::GEMMA_4_E2B), "Q4_0", 5.1, 2.0, 32_768, 8192),
        chat(gemma(crate::catalog::GEMMA_4_E4B), "Q4_0", 8.0, 4.0, 65_536, 8192),
        chat(
            spec(
                "qwen3-8b",
                "Qwen3 8B (Q4_K_M)",
                "unsloth/Qwen3-8B-GGUF",
                "Qwen3-8B-Q4_K_M.gguf",
                5_027_784_512,
                "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4",
            ),
            "Q4_K_M",
            8.2,
            8.2,
            147_456,
            8192,
        ),
        chat(
            spec(
                "qwen3-14b",
                "Qwen3 14B (Q4_K_M)",
                "unsloth/Qwen3-14B-GGUF",
                "Qwen3-14B-Q4_K_M.gguf",
                9_001_753_984,
                "5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66",
            ),
            "Q4_K_M",
            14.8,
            14.8,
            163_840,
            8192,
        ),
        chat(
            spec(
                "gpt-oss-20b",
                "gpt-oss 20B (MXFP4)",
                "ggml-org/gpt-oss-20b-GGUF",
                "gpt-oss-20b-MXFP4.gguf",
                12_109_566_624,
                "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
            ),
            "MXFP4",
            21.0,
            3.6,
            49_152,
            8192,
        ),
        chat(
            spec(
                "qwen3-30b-a3b",
                "Qwen3 30B-A3B (Q4_K_M)",
                "unsloth/Qwen3-30B-A3B-GGUF",
                "Qwen3-30B-A3B-Q4_K_M.gguf",
                18_556_686_912,
                "9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48",
            ),
            "Q4_K_M",
            30.5,
            3.3,
            98_304,
            8192,
        ),
        chat(
            spec(
                "qwen3-32b",
                "Qwen3 32B (Q4_K_M)",
                "unsloth/Qwen3-32B-GGUF",
                "Qwen3-32B-Q4_K_M.gguf",
                19_762_150_048,
                "8df67573b2c23484e02ec7af295e39bed7ee774f3771d5fda2978265b59370e7",
            ),
            "Q4_K_M",
            32.8,
            32.8,
            262_144,
            8192,
        ),
        CatalogEntry {
            spec: spec(
                "qwen3-embedding-0.6b",
                "Qwen3 Embedding 0.6B (Q8_0)",
                "Qwen/Qwen3-Embedding-0.6B-GGUF",
                "Qwen3-Embedding-0.6B-Q8_0.gguf",
                639_150_592,
                "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
            ),
            purpose: Purpose::Embedding,
            quant: "Q8_0".into(),
            params_b: 0.6,
            active_params_b: 0.6,
            kv_bytes_per_token: 114_688,
            context: 2048,
        },
        // Mistral's Voxtral Mini 3B (Apache-2.0): a Whisper-style audio encoder in front of a 3B
        // decoder, transcription in eight languages; 30 seconds of audio cost 187 tokens in
        // llama.cpp, so 16K of context holds about 20 minutes with its transcript.
        CatalogEntry {
            spec: spec(
                VOXTRAL_MINI_3B,
                "Voxtral Mini 3B speech to text (Q4_K_M)",
                "ggml-org/Voxtral-Mini-3B-2507-GGUF",
                "Voxtral-Mini-3B-2507-Q4_K_M.gguf",
                2_473_001_920,
                "4705be8ec22ca23d12632f4b4a3691faa95917d90a06d3cf3c3ec0e91958f1a8",
            ),
            purpose: Purpose::Transcribe,
            quant: "Q4_K_M".into(),
            params_b: 4.7,
            active_params_b: 4.7,
            kv_bytes_per_token: 122_880,
            context: 16_384,
        },
        // Text to speech (Apache-2.0 unless noted). KV per token from each config or GGUF header
        // (layers x KV heads x (key + value dim) x 2 bytes). Speech runs one utterance at a
        // time, and the context these entries are sized at (4096: a prompt plus minutes of
        // audio frames; Drama 3's documented cap in llama.cpp) is an assumption for the budget,
        // not a limit the model reports.
        //
        // Qwen3-TTS (Qwen, 12.5 Hz codec): 28 layers x 8 KV heads x (128 + 128) for both sizes.
        // ggml-org converted only the 1.7B Base checkpoint; the others are the Qwen repos that
        // vLLM-Omni serves (BF16 safetensors, plus the shared speech tokenizer).
        speaker_entry(
            spec(
                QWEN3_TTS_0_6B_CUSTOMVOICE,
                "Qwen3-TTS 0.6B CustomVoice (BF16, vLLM-Omni)",
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "model.safetensors",
                1_811_626_576,
                "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
            ),
            "BF16",
            0.6,
            0.6,
            114_688,
            4096,
        ),
        speaker_entry(
            spec(
                QWEN3_TTS_0_6B_BASE,
                "Qwen3-TTS 0.6B Base, voice cloning (BF16, vLLM-Omni)",
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "model.safetensors",
                1_829_344_272,
                "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
            ),
            "BF16",
            0.6,
            0.6,
            114_688,
            4096,
        ),
        speaker_entry(
            spec(
                QWEN3_TTS_1_7B_CUSTOMVOICE,
                "Qwen3-TTS 1.7B CustomVoice (BF16, vLLM-Omni)",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "model.safetensors",
                3_833_402_552,
                "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
            ),
            "BF16",
            1.7,
            1.7,
            114_688,
            4096,
        ),
        speaker_entry(
            spec(
                QWEN3_TTS_1_7B_VOICEDESIGN,
                "Qwen3-TTS 1.7B VoiceDesign (BF16, vLLM-Omni)",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "model.safetensors",
                3_833_402_552,
                "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522",
            ),
            "BF16",
            1.7,
            1.7,
            114_688,
            4096,
        ),
        speaker_entry(
            spec(
                QWEN3_TTS_1_7B_BASE,
                "Qwen3-TTS 1.7B Base, voice cloning (Q8_0)",
                "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF",
                "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
                1_847_874_400,
                "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1",
            ),
            "Q8_0",
            1.7,
            1.7,
            114_688,
            4096,
        ),
        // Kokoro 82M (hexgrad): not autoregressive; 12 layers x 12 heads x (64 + 64) over its
        // 512-phoneme context.
        speaker_entry(
            spec(
                KOKORO_82M,
                "Kokoro 82M (F32)",
                "ggml-org/Kokoro-82M-GGUF",
                "Kokoro-82M-F32.gguf",
                61_774_240,
                "be19298184f1a2577eb20e4689f1e2b315cbf474c39e113f192991fef3c85c6e",
            ),
            "F32",
            0.082,
            0.082,
            36_864,
            512,
        ),
        // VoxCPM2 (OpenBMB, 2B): MiniCPM4 backbone, 28 layers x 2 KV heads x (128 + 128), plus
        // the projector's residual LM (12 layers x 2 x (64 + 64)), which keeps one context per
        // utterance.
        speaker_entry(
            spec(
                VOXCPM2,
                "VoxCPM2, voice design and cloning (Q8_0)",
                "ggml-org/VoxCPM2-GGUF",
                "VoxCPM2-Q8_0.gguf",
                1_565_764_064,
                "b09cd3afab454f23922632fd474aa0d5d667e128e8bf36e81e77ec7e71fac64b",
            ),
            "Q8_0",
            2.0,
            2.0,
            28_672 + 6_144,
            4096,
        ),
        // Fish Audio Drama 3 (Fish Audio Community License): Qwen3.5-MoE, 35B with 3B active;
        // 10 of its 40 layers attend (the rest are recurrent), 2 KV heads x (256 + 256), plus
        // the projector's codebook tower (13 layers x 2 x (128 + 128)), which keeps the whole
        // sequence, prompt and frames, in its 4096 positions.
        speaker_entry(
            spec(
                FISH_AUDIO_DRAMA_3,
                "Fish Audio Drama 3, multi-speaker (Q4_K_M)",
                "ggml-org/drama-3-GGUF",
                "drama-3-Q4_K_M.gguf",
                20_556_053_760,
                "5b727d669a7a555713403b7f6d744402dafd6fd37154f0fe834a5310548a8b34",
            ),
            "Q4_K_M",
            35.0,
            3.0,
            20_480 + 13_312,
            4096,
        ),
    ]
}

/// The catalog entries for `purpose`, in catalog order.
pub fn by_purpose(purpose: Purpose) -> Vec<CatalogEntry> {
    catalog().into_iter().filter(|e| e.purpose == purpose).collect()
}

/// The speech-to-text model ([`Purpose::Transcribe`]) that fits `budget` with one request at
/// a time, projector included, if any. [`plan`] never proposes one: transcription is optional,
/// so an app adds it when it wants soundtracks turned into text.
pub fn transcriber(budget: &Budget) -> Option<CatalogEntry> {
    by_purpose(Purpose::Transcribe).into_iter().filter(|e| e.footprint(1) <= budget.usable_bytes).max_by(|a, b| a.params_b.total_cmp(&b.params_b))
}

/// A text-to-speech model ([`Purpose::Speak`]) that fits `budget` with one request at a time,
/// projector included: the largest GGUF entry (llama.cpp runs it, and [`download`](crate::download())
/// fetches it) whose license doesn't restrict use, if any. [`plan`] never proposes one: speech
/// is optional, so an app adds it when it wants text read aloud. Since 0.5.5.
pub fn speaker(budget: &Budget) -> Option<CatalogEntry> {
    by_purpose(Purpose::Speak)
        .into_iter()
        .filter(|e| e.is_gguf() && !e.restricted() && e.footprint(1) <= budget.usable_bytes)
        .max_by(|a, b| a.params_b.total_cmp(&b.params_b))
}

/// The license that restricts downloading `spec`, if any: the license of a restricted catalog
/// entry ([`CatalogEntry::restricted`]) whose model or projector `spec` is (by id), or whose repo
/// (or the repo its weights come from) `spec` downloads from. [`download`](crate::download())
/// refuses such a file without [`Consent::accepting_license`](crate::Consent::accepting_license).
/// Since 0.5.5.
pub fn restricted_license(spec: &ModelSpec) -> Option<&'static str> {
    catalog()
        .into_iter()
        .filter(CatalogEntry::restricted)
        .find(|e| {
            e.spec.id == spec.id
                || e.projector().is_some_and(|p| p.id == spec.id)
                || e.license_repos().iter().any(|r| r.eq_ignore_ascii_case(&spec.repo))
        })
        .and_then(|e| e.license())
}

/// The catalog entry with this id.
pub fn find(id: &str) -> Option<CatalogEntry> {
    catalog().into_iter().find(|e| e.spec.id == id)
}

/// How much memory models may use on this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// VRAM of the first GPU, or system memory on unified and CPU-only machines.
    pub pool_bytes: u64,
    /// Kept for the OS (unified and CPU-only only).
    pub reserve_bytes: u64,
    pub usable_bytes: u64,
}

/// 85% of VRAM, or 85% of unified memory after the OS reserve (default: an eighth of RAM, at least
/// 4 GiB). `memory_budget` overrides the result; `reserve` overrides the reserve.
pub fn budget(hw: &Hardware, memory_budget: Option<u64>, reserve: Option<u64>) -> Budget {
    let unified = hw.unified();
    let pool_bytes = if unified { hw.ram_bytes } else { hw.gpus.first().map(|g| g.memory_bytes).unwrap_or(hw.ram_bytes) };
    let reserve_bytes = if unified { reserve.unwrap_or_else(|| (hw.ram_bytes / 8).max(4 * GIB)) } else { 0 };
    let usable = pool_bytes.saturating_sub(reserve_bytes) / 100 * 85;
    Budget { pool_bytes, reserve_bytes, usable_bytes: memory_budget.unwrap_or(usable) }
}

/// What setup proposes for this node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub budget: Budget,
    pub main: Option<CatalogEntry>,
    pub standin: Option<CatalogEntry>,
    pub embedding: Option<CatalogEntry>,
    /// Concurrent requests each chat model is sized for.
    pub slots: u32,
    pub notes: Vec<String>,
}

/// Picks the largest chat model that fits with its KV budget, a Gemma 4 stand-in and the embedding
/// model. CPU-only machines only get models with at most 4B active parameters, so answers stay
/// usable.
pub fn plan(hw: &Hardware, budget: Budget) -> Plan {
    let slots = 2;
    let cpu_only = hw.accel() == Accel::Cpu;
    let mut notes = Vec::new();
    let all = catalog();
    let fits = |e: &&CatalogEntry| e.footprint(slots) <= budget.usable_bytes;
    let main = all
        .iter()
        .filter(|e| e.purpose == Purpose::Chat && e.spec.id != "smollm2-135m")
        .filter(|e| !cpu_only || e.active_params_b <= 4.0)
        .filter(fits)
        .max_by(|a, b| a.params_b.total_cmp(&b.params_b))
        .cloned();
    if cpu_only {
        notes.push("CPU only: limited to models with at most 4B active parameters".into());
    }
    if main.is_none() {
        notes.push("no catalog chat model fits this budget; add a smaller GGUF model".into());
    }
    let standin_id = if budget.usable_bytes >= 16 * GIB { crate::catalog::GEMMA_4_E4B } else { crate::catalog::GEMMA_4_E2B };
    let standin = all.iter().find(|e| e.spec.id == standin_id).filter(fits).filter(|e| main.as_ref().is_none_or(|m| m.spec.id != e.spec.id)).cloned();
    let embedding = all.iter().find(|e| e.purpose == Purpose::Embedding).filter(|e| e.footprint(1) <= budget.usable_bytes).cloned();
    Plan { budget, main, standin, embedding, slots, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::{Gpu, parse_nvidia_smi};

    fn box_with(gpus: Vec<Gpu>, ram_gib: u64) -> Hardware {
        Hardware { os: "linux".into(), arch: "x86_64".into(), cpu_threads: 16, ram_bytes: ram_gib * GIB, gpus }
    }

    #[test]
    fn rtx_3060_gets_an_8b() {
        let hw = box_with(parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 8.6, 615.78.08", 62 * GIB), 62);
        let b = budget(&hw, None, None);
        assert_eq!(b.pool_bytes, 12 * GIB);
        let p = plan(&hw, b);
        assert_eq!(p.main.unwrap().id(), "qwen3-8b");
        assert_eq!(p.standin.unwrap().id(), "gemma-4-e2b");
        assert!(p.embedding.is_some());
    }

    #[test]
    fn spark_gets_a_big_model() {
        let hw = box_with(parse_nvidia_smi("NVIDIA GB10, [N/A], 12.1, 580.95.05", 128 * GIB), 128);
        let b = budget(&hw, None, None);
        assert_eq!(b.reserve_bytes, 16 * GIB);
        let p = plan(&hw, b);
        assert_eq!(p.main.unwrap().id(), "qwen3-32b");
        assert_eq!(p.standin.unwrap().id(), "gemma-4-e4b");
    }

    #[test]
    fn cpu_box_prefers_few_active_params() {
        let hw = box_with(Vec::new(), 62);
        let p = plan(&hw, budget(&hw, None, None));
        assert_eq!(p.main.unwrap().id(), "qwen3-30b-a3b");
    }

    #[test]
    fn tiny_box_gets_no_main() {
        let hw = box_with(Vec::new(), 4);
        let p = plan(&hw, budget(&hw, Some(GIB), None));
        assert!(p.main.is_none());
        assert!(!p.notes.is_empty());
    }

    #[test]
    fn override_budget() {
        let hw = box_with(Vec::new(), 64);
        assert_eq!(budget(&hw, Some(10 * GIB), None).usable_bytes, 10 * GIB);
        assert_eq!(budget(&hw, None, Some(14 * GIB)).reserve_bytes, 14 * GIB);
    }

    #[test]
    fn transcribe_is_a_purpose_of_its_own() {
        assert_eq!(serde_json::to_string(&Purpose::Transcribe).unwrap(), "\"transcribe\"");
        for p in Purpose::ALL {
            assert_eq!(serde_json::from_str::<Purpose>(&format!("\"{p}\"")).unwrap(), p);
            assert_eq!(Purpose::parse(p.as_str()), Some(p));
        }
        assert_eq!(Purpose::parse("speech"), None);
        // Stored entries from before the variant still read.
        assert_eq!(serde_json::from_str::<Purpose>("\"chat\"").unwrap(), Purpose::Chat);
    }

    #[test]
    fn the_catalog_has_a_pinned_speech_to_text_model() {
        let stt = by_purpose(Purpose::Transcribe);
        assert_eq!(stt.len(), 1);
        let v = &stt[0];
        assert_eq!((v.id(), v.quant.as_str(), v.context), (VOXTRAL_MINI_3B, "Q4_K_M", 16_384));
        assert!(v.transcribes());
        assert_eq!(find(VOXTRAL_MINI_3B).as_ref(), Some(v));
        assert_eq!(v.spec.repo, "ggml-org/Voxtral-Mini-3B-2507-GGUF");
        assert_eq!(v.spec.sha256.as_deref().map(str::len), Some(64));
        let p = v.projector().expect("Voxtral needs its audio encoder");
        assert_eq!((p.repo.as_str(), p.file.as_str(), p.size), (v.spec.repo.as_str(), "mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf", 715_714_080));
        assert!(p.sha256.as_deref().is_some_and(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit())));
        assert_ne!(p.file, v.spec.file);
        // 30 layers x 8 KV heads x (128 + 128) x 2 bytes.
        assert_eq!(v.kv_bytes_per_token, 30 * 8 * 256 * 2);
        // The projector counts in the footprint; text models have none.
        let without = (v.spec.size + v.kv_bytes_per_token * 16_384) * 11 / 10;
        assert!(v.footprint(1) > without);
        assert!(catalog().iter().filter(|e| !e.transcribes() && !e.speaks()).all(|e| e.projector().is_none()));
        // Catalog ids stay unique.
        let mut ids: Vec<_> = catalog().iter().map(|e| e.id().to_string()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), catalog().len());
    }

    #[test]
    fn plans_never_pick_the_transcriber_and_it_fits_on_its_own() {
        for (gpus, ram) in [(parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 8.6, 615.78.08", 62 * GIB), 62), (Vec::new(), 62), (Vec::new(), 8)] {
            let hw = box_with(gpus, ram);
            let p = plan(&hw, budget(&hw, None, None));
            for e in [&p.main, &p.standin, &p.embedding].into_iter().flatten() {
                assert!(!e.transcribes() && !e.speaks(), "{} in a plan", e.id());
            }
        }
        let hw = box_with(Vec::new(), 62);
        assert_eq!(transcriber(&budget(&hw, None, None)).map(|e| e.spec.id), Some(VOXTRAL_MINI_3B.into()));
        assert_eq!(transcriber(&budget(&hw, Some(GIB), None)), None, "too small a budget");
        assert!(by_purpose(Purpose::Embedding).iter().all(|e| e.purpose == Purpose::Embedding));
    }

    #[test]
    fn speak_is_a_purpose_of_its_own() {
        assert_eq!(serde_json::to_string(&Purpose::Speak).unwrap(), "\"speak\"");
        assert_eq!(Purpose::parse("speak"), Some(Purpose::Speak));
        assert_eq!(Purpose::Speak.to_string(), "speak");
        assert_eq!(Purpose::parse("tts"), None);
    }

    #[test]
    fn the_catalog_has_cited_speech_models() {
        let ids: Vec<_> = by_purpose(Purpose::Speak).iter().map(|e| e.id().to_string()).collect();
        assert_eq!(
            ids,
            [
                QWEN3_TTS_0_6B_CUSTOMVOICE,
                QWEN3_TTS_0_6B_BASE,
                QWEN3_TTS_1_7B_CUSTOMVOICE,
                QWEN3_TTS_1_7B_VOICEDESIGN,
                QWEN3_TTS_1_7B_BASE,
                KOKORO_82M,
                VOXCPM2,
                FISH_AUDIO_DRAMA_3
            ]
        );
        let hex = |h: &Option<String>| h.as_deref().is_some_and(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()));
        for e in by_purpose(Purpose::Speak) {
            assert!(e.speaks() && !e.transcribes());
            assert!(hex(&e.spec.sha256), "{} pinned", e.id());
            let s = e.speech().unwrap_or_else(|| panic!("{} has a profile", e.id()));
            assert!(s.sample_rate.is_some() && !s.formats.is_empty() && s.languages.is_some(), "{}", e.id());
            assert!(e.license().is_some());
            if e.is_gguf() {
                let p = e.projector().unwrap_or_else(|| panic!("{} needs its projector", e.id()));
                assert!(hex(&p.sha256) && p.file.starts_with("mmproj-") && p.repo == e.spec.repo);
                assert_eq!(s.formats, [crate::speech::AudioFormat::Wav, crate::speech::AudioFormat::Pcm16], "llama-server's formats");
                assert!(e.footprint(1) > (e.spec.size + p.size) * 11 / 10);
            } else {
                assert!(e.projector().is_none());
                assert!(e.spec.repo.starts_with("Qwen/") && e.spec.file == "model.safetensors");
                assert!(e.footprint(1) > (e.spec.size + QWEN3_TTS_TOKENIZER_BYTES) * 11 / 10, "the speech tokenizer counts");
                assert!(s.formats.len() == 5 && s.speed, "vLLM-Omni's formats and speed");
            }
        }
        // Each Qwen3-TTS checkpoint does one thing.
        let q = |id| find(id).unwrap().speech().unwrap();
        assert_eq!(q(QWEN3_TTS_1_7B_CUSTOMVOICE).presets.as_ref().map(Vec::len), Some(9));
        assert!(q(QWEN3_TTS_1_7B_CUSTOMVOICE).instructions && !q(QWEN3_TTS_0_6B_CUSTOMVOICE).instructions);
        assert!(q(QWEN3_TTS_1_7B_VOICEDESIGN).design && q(QWEN3_TTS_1_7B_VOICEDESIGN).clone.is_none());
        assert_eq!(q(QWEN3_TTS_0_6B_BASE).clone, Some(CloneMode::AudioAndText));
        assert_eq!(q(QWEN3_TTS_1_7B_BASE).clone, Some(CloneMode::Audio), "ggml-org's projector has the speaker encoder only");
        let k = q(KOKORO_82M);
        assert_eq!((k.presets.as_ref().map(Vec::len), k.presets.as_ref().and_then(|p| p.first()).map(String::as_str)), (Some(54), Some("af_heart")));
        assert!(k.raw && k.speed);
        assert!(q(VOXCPM2).design && q(VOXCPM2).clone == Some(CloneMode::Audio));
        assert_eq!(q(VOXCPM2).languages.map(|l| l.len()), Some(30));
        let d = q(FISH_AUDIO_DRAMA_3);
        assert_eq!((d.script_speakers, d.clone, d.sample_rate), (Some(16), None, Some(44_100)));
        assert_eq!(d.languages.map(|l| l.len()), Some(83));
        // Only Drama 3 is restricted, and its profile names the license.
        let drama = find(FISH_AUDIO_DRAMA_3).unwrap();
        assert!(drama.restricted() && drama.license() == Some(crate::speech::FISH_AUDIO_COMMUNITY));
        assert_eq!(d.license.as_deref(), Some(crate::speech::FISH_AUDIO_COMMUNITY));
        assert_eq!(catalog().iter().filter(|e| e.restricted()).count(), 1);
        assert!(catalog().iter().filter(|e| !e.speaks()).all(|e| e.speech().is_none()));
    }

    #[test]
    fn a_speaker_that_fits_and_never_a_restricted_one() {
        let b = |usable_bytes| Budget { pool_bytes: usable_bytes, reserve_bytes: 0, usable_bytes };
        assert_eq!(speaker(&b(128 * GIB)).map(|e| e.spec.id), Some(VOXCPM2.into()), "never Drama 3, however big");
        assert_eq!(speaker(&b(GIB * 7 / 2)).map(|e| e.spec.id), Some(QWEN3_TTS_1_7B_BASE.into()));
        assert_eq!(speaker(&b(GIB)).map(|e| e.spec.id), Some(KOKORO_82M.into()));
        assert_eq!(speaker(&b(100 << 20)), None);
        assert!(by_purpose(Purpose::Speak).iter().all(|e| e.footprint(1) > 0));
    }

    #[test]
    fn downloads_under_a_restricted_license() {
        let drama = find(FISH_AUDIO_DRAMA_3).unwrap();
        let lic = Some(crate::speech::FISH_AUDIO_COMMUNITY);
        assert_eq!(restricted_license(&drama.spec), lic);
        assert_eq!(restricted_license(&drama.projector().unwrap()), lic);
        // A hand-built spec from the same repos, under any id.
        let mut other = drama.spec.clone();
        other.id = "my-drama".into();
        other.file = "drama-3-Q8_0.gguf".into();
        assert_eq!(restricted_license(&other), lic);
        other.repo = "FishAudio/Drama-3".into();
        assert_eq!(restricted_license(&other), lic);
        for e in catalog().iter().filter(|e| e.spec.id != FISH_AUDIO_DRAMA_3) {
            assert_eq!(restricted_license(&e.spec), None, "{}", e.id());
        }
    }
}
