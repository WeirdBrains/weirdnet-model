//! Reading a Hugging Face repo's GGUF files, so any repo can become a [`ModelSpec`].

use serde_json::Value;

use crate::catalog::ModelSpec;
use crate::{Error, Result};

/// A repo id: `owner/name`, Hub characters only (never a path trick).
pub fn check_repo(id: &str) -> Result<()> {
    let ok = |p: &str| !p.is_empty() && p != "." && p != ".." && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
    match id.split_once('/') {
        Some((o, n)) if ok(o) && ok(n) => Ok(()),
        _ => Err(Error::BadRepo(id.to_string())),
    }
}

/// One GGUF file in a repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// The GGUF files of `repo` on the Hub at `hub` (names, sizes and hashes from the file metadata;
/// no weights are downloaded). Projector (`mmproj-`) and draft (`mtp-`) files are left out.
pub fn repo_files(hub: &str, repo: &str) -> Result<Vec<RepoFile>> {
    check_repo(repo)?;
    let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
    let mut resp = agent
        .get(format!("{}/api/models/{repo}?blobs=true", hub.trim_end_matches('/')))
        .header("Accept", "application/json")
        .call()
        .map_err(|e| Error::Http(e.to_string()))?;
    if resp.status().as_u16() != 200 {
        return Err(Error::Http(format!("the Hub answered {} for {repo}", resp.status().as_u16())));
    }
    let text = resp.body_mut().read_to_string().map_err(|e| Error::Http(e.to_string()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| Error::Http(format!("the Hub's answer is not JSON: {e}")))?;
    Ok(parse_files(&v))
}

pub fn parse_files(v: &Value) -> Vec<RepoFile> {
    v["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            let name = s["rfilename"].as_str()?;
            let file = name.rsplit('/').next().unwrap_or(name);
            let lower = file.to_ascii_lowercase();
            if !lower.ends_with(".gguf") || lower.starts_with("mmproj") || lower.starts_with("mtp-") {
                return None;
            }
            Some(RepoFile { name: name.to_string(), size: s["size"].as_u64().unwrap_or(0), sha256: s["lfs"]["sha256"].as_str().map(str::to_string) })
        })
        .collect()
}

impl ModelSpec {
    /// A spec for one file of `repo`, as listed by [`repo_files`]. The id is the file stem.
    pub fn from_repo_file(repo: &str, file: &RepoFile) -> Result<Self> {
        check_repo(repo)?;
        let name = file.name.rsplit('/').next().unwrap_or(&file.name);
        let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF")).unwrap_or(name);
        Ok(Self {
            id: stem.to_string(),
            label: stem.to_string(),
            repo: repo.to_string(),
            file: file.name.clone(),
            size: file.size,
            sha256: file.sha256.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockHub;

    const REPO_JSON: &str = r#"{"id":"o/n","siblings":[
        {"rfilename":"README.md","size":10},
        {"rfilename":"m-Q4_0.gguf","size":100,"lfs":{"sha256":"abc"}},
        {"rfilename":"mmproj-m-BF16.gguf","size":5,"lfs":{"sha256":"d"}},
        {"rfilename":"mtp-m-Q4_0.gguf","size":6},
        {"rfilename":"sub/m-Q8_0.GGUF","size":200}]}"#;

    #[test]
    fn repo_ids_must_look_like_owner_slash_name() {
        assert!(check_repo("ggml-org/gemma-4-E2B-it-GGUF").is_ok());
        for bad in ["", "a", "a/b/c", "../x", "a/..", "a b/c", "/x"] {
            assert!(check_repo(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_model_ggufs_are_listed() {
        let files = parse_files(&serde_json::from_str(REPO_JSON).unwrap());
        assert_eq!(files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["m-Q4_0.gguf", "sub/m-Q8_0.GGUF"]);
        assert_eq!(files[0].sha256.as_deref(), Some("abc"));
        assert_eq!(files[1].sha256, None);
        let spec = ModelSpec::from_repo_file("o/n", &files[1]).unwrap();
        assert_eq!((spec.id.as_str(), spec.file.as_str(), spec.size), ("m-Q8_0", "sub/m-Q8_0.GGUF", 200));
    }

    #[test]
    fn reads_a_repo_from_the_hub_api() {
        let hub = MockHub::start_with(vec![("/api/models/o/n".into(), "application/json", REPO_JSON.as_bytes().to_vec())]);
        assert_eq!(repo_files(hub.url(), "o/n").unwrap().len(), 2);
        assert!(matches!(repo_files(hub.url(), "o/missing"), Err(Error::Http(ref m)) if m.contains("404")));
        assert!(matches!(repo_files(hub.url(), "bad"), Err(Error::BadRepo(_))));
    }
}
