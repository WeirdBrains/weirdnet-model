//! A Hugging Face access token, for gated and private repos.
//!
//! The token is read on the node that downloads, from its own environment or token file, and
//! only goes into `Authorization` headers to the Hub. It has no serde support, so it can't end
//! up in a message to another node, and its `Debug` output is redacted.

use std::path::{Path, PathBuf};

/// A Hugging Face access token. `Debug` prints `HfToken(<redacted>)`.
#[derive(Clone, PartialEq, Eq)]
pub struct HfToken(String);

impl std::fmt::Debug for HfToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HfToken(<redacted>)")
    }
}

impl HfToken {
    /// A token from its text (surrounding whitespace dropped); `None` when empty.
    pub fn new(token: impl AsRef<str>) -> Option<Self> {
        let t = token.as_ref().trim();
        (!t.is_empty()).then(|| Self(t.to_string()))
    }

    /// The token in a file, such as `huggingface-cli login`'s `token` file. A missing file is
    /// `Ok(None)`.
    pub fn from_file(path: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Self::new(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// This node's token, the way the Hugging Face tools find it: `$HF_TOKEN`, then
    /// `$HUGGING_FACE_HUB_TOKEN`, then the file at `$HF_TOKEN_PATH`, `$HF_HOME/token` or
    /// `~/.cache/huggingface/token`. Unreadable files count as no token.
    pub fn from_env() -> Option<Self> {
        Self::from_sources(|k| std::env::var(k).ok(), dirs::home_dir())
    }

    pub(crate) fn from_sources(env: impl Fn(&str) -> Option<String>, home: Option<PathBuf>) -> Option<Self> {
        if let Some(t) = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| env(k).and_then(Self::new)) {
            return Some(t);
        }
        let set = |k: &str| env(k).filter(|v| !v.trim().is_empty()).map(PathBuf::from);
        let file = set("HF_TOKEN_PATH")
            .or_else(|| set("HF_HOME").map(|h| h.join("token")))
            .or_else(|| home.map(|h| h.join(".cache").join("huggingface").join("token")))?;
        Self::from_file(&file).ok().flatten()
    }

    /// The `Authorization` header value.
    pub(crate) fn bearer(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_is_redacted_and_empty_is_none() {
        let t = HfToken::new(" hf_secret \n").unwrap();
        assert_eq!(format!("{t:?}"), "HfToken(<redacted>)");
        assert_eq!(t.bearer(), "Bearer hf_secret");
        assert!(HfToken::new("  ").is_none());
    }

    #[test]
    fn found_like_the_hugging_face_tools() {
        let tmp = tempfile::tempdir().unwrap();
        let none = |_: &str| None::<String>;
        assert_eq!(HfToken::from_sources(none, Some(tmp.path().into())), None);
        let cache = tmp.path().join(".cache/huggingface");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join("token"), "hf_file\n").unwrap();
        assert_eq!(HfToken::from_sources(none, Some(tmp.path().into())), HfToken::new("hf_file"));
        let hf_home = tmp.path().join("hfhome");
        std::fs::create_dir_all(&hf_home).unwrap();
        std::fs::write(hf_home.join("token"), "hf_home").unwrap();
        let home_env = hf_home.display().to_string();
        let env = move |k: &str| (k == "HF_HOME").then(|| home_env.clone());
        assert_eq!(HfToken::from_sources(&env, Some(tmp.path().into())), HfToken::new("hf_home"));
        let env = |k: &str| match k {
            "HF_TOKEN" => Some("".to_string()),
            "HUGGING_FACE_HUB_TOKEN" => Some("hf_env".to_string()),
            _ => None,
        };
        assert_eq!(HfToken::from_sources(env, None), HfToken::new("hf_env"));
    }
}
