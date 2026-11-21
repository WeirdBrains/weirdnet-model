//! Hugging Face Hub: search for GGUF repos, read a repo's GGUF files (split files grouped into
//! one model), read a file's header for KV sizing, so any repo can become a [`ModelSpec`].
//!
//! [`Hub`] sends this node's [`HfToken`] (if any) with every request; redirects to other hosts
//! (the file CDN) don't carry it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::ModelSpec;
use crate::gguf::{self, GgufInfo};
use crate::token::HfToken;
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoFile {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// The GGUF files of `repo` on the Hub at `hub` (names, sizes and hashes from the file metadata;
/// no weights are downloaded). Projector (`mmproj-`) and draft (`mtp-`) files are left out. No
/// token; see [`Hub::repo_files`].
pub fn repo_files(hub: &str, repo: &str) -> Result<Vec<RepoFile>> {
    Hub::new(hub).repo_files(repo)
}

/// The Hub's file list in a repo's model info (`siblings`), GGUF model files only.
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

/// A file name inside a repo: relative, `/`-separated, no `.`/`..` parts (never a path trick).
pub fn check_file(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name.split('/').all(|p| !p.is_empty() && p != "." && p != ".." && !p.contains(':'));
    if ok { Ok(()) } else { Err(Error::Model(format!("{name:?} is not a file name inside a repo"))) }
}

/// A Hub search: repos matching `query`, most downloaded first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Search {
    pub query: String,
    /// At most this many hits (capped at 100).
    pub limit: usize,
    /// Only repos tagged `gguf` (default on).
    pub gguf_only: bool,
}

impl Search {
    pub fn new(query: impl Into<String>) -> Self {
        Self { query: query.into(), limit: 20, gguf_only: true }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_gguf_only(mut self, on: bool) -> Self {
        self.gguf_only = on;
        self
    }
}

/// One repo in [`Hub::search`] results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SearchHit {
    /// `owner/name`.
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    /// The repo needs an accepted licence and a token (`auto` or `manual` approval on the Hub).
    pub gated: bool,
    pub pipeline_tag: Option<String>,
    pub last_modified: Option<String>,
}

/// Parses the Hub's `/api/models` answer; entries without a valid repo id are dropped.
pub fn parse_search(v: &Value) -> Vec<SearchHit> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m["id"].as_str().or_else(|| m["modelId"].as_str())?;
            check_repo(id).ok()?;
            Some(SearchHit {
                id: id.to_string(),
                downloads: m["downloads"].as_u64().unwrap_or(0),
                likes: m["likes"].as_u64().unwrap_or(0),
                gated: match &m["gated"] {
                    Value::Bool(b) => *b,
                    Value::String(s) => !s.is_empty() && s != "false",
                    _ => false,
                },
                pipeline_tag: m["pipeline_tag"].as_str().map(str::to_string),
                last_modified: m["lastModified"].as_str().map(str::to_string),
            })
        })
        .collect()
}

/// One downloadable model in a repo: a single GGUF, or every part of a split GGUF
/// (`name-00001-of-00003.gguf` ...), in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RepoModel {
    /// The file stem without the split suffix, such as `Qwen3-32B-Q4_K_M`; the [`ModelSpec`] id.
    pub id: String,
    /// The quantization read from the file name, such as `Q4_K_M` (see [`quant_of`]).
    pub quant: Option<String>,
    /// The files, first part first. llama.cpp loads a split model from its first part.
    pub parts: Vec<RepoFile>,
    /// All parts together.
    pub size: u64,
    /// Every part of a split model is in the repo.
    pub complete: bool,
}

impl RepoModel {
    /// The first (or only) file.
    pub fn first(&self) -> &RepoFile {
        &self.parts[0]
    }

    pub fn is_split(&self) -> bool {
        self.parts.len() > 1
    }

    /// One [`ModelSpec`] per part, all with this model's id, for
    /// [`download_parts`](crate::download::download_parts).
    pub fn specs(&self, repo: &str) -> Result<Vec<ModelSpec>> {
        check_repo(repo)?;
        self.parts
            .iter()
            .map(|f| {
                check_file(&f.name)?;
                Ok(ModelSpec {
                    id: self.id.clone(),
                    label: self.id.clone(),
                    repo: repo.to_string(),
                    file: f.name.clone(),
                    size: f.size,
                    sha256: f.sha256.clone(),
                })
            })
            .collect()
    }
}

/// `(stem, part, of)` for a split file name such as `dir/m-Q4_K_M-00002-of-00003.gguf`.
pub fn split_part(name: &str) -> Option<(&str, u32, u32)> {
    let file = name.rsplit('/').next().unwrap_or(name);
    let stem = file.strip_suffix(".gguf").or_else(|| file.strip_suffix(".GGUF"))?;
    let (rest, of) = stem.rsplit_once("-of-")?;
    let (base, part) = rest.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.chars().all(|c| c.is_ascii_digit());
    if !digits(part) || !digits(of) || base.is_empty() {
        return None;
    }
    let (part, of): (u32, u32) = (part.parse().ok()?, of.parse().ok()?);
    (part >= 1 && part <= of).then_some((base, part, of))
}

/// GGUF quantization names, longer ones first so `Q4_K_M` wins over `Q4_K`.
const QUANTS: &[&str] = &[
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL", "TQ1_0", "TQ2_0", "Q2_K_XL",
    "Q2_K_L", "Q2_K_S", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S",
    "Q6_K_XL", "Q6_K_L", "Q8_K_XL", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "MXFP4", "BF16", "F16", "F32",
];

/// The quantization in a GGUF file name (`Qwen3-8B-Q4_K_M.gguf` is `Q4_K_M`), if it names one.
pub fn quant_of(name: &str) -> Option<String> {
    let file = name.rsplit('/').next().unwrap_or(name).to_ascii_uppercase();
    let stem = file.strip_suffix(".GGUF").unwrap_or(&file);
    let bounded = |at: usize, len: usize| {
        let before = stem[..at].chars().next_back();
        let after = stem[at + len..].chars().next();
        before.is_none_or(|c| !c.is_ascii_alphanumeric()) && after.is_none_or(|c| !c.is_ascii_alphanumeric())
    };
    QUANTS.iter().find(|q| stem.match_indices(**q).any(|(at, _)| bounded(at, q.len()))).map(|q| q.to_string())
}

/// Groups GGUF files into models: split parts together (ordered), single files alone. Sorted
/// by size, smallest first.
pub fn group_files(files: &[RepoFile]) -> Vec<RepoModel> {
    let mut singles = Vec::new();
    let mut splits: std::collections::BTreeMap<(String, u32), Vec<(u32, RepoFile)>> = Default::default();
    for f in files {
        match split_part(&f.name) {
            Some((base, part, of)) => {
                let dir = f.name.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
                splits.entry((format!("{dir}{base}"), of)).or_default().push((part, f.clone()));
            }
            None => singles.push(f.clone()),
        }
    }
    let mut out: Vec<RepoModel> = singles
        .into_iter()
        .map(|f| {
            let file = f.name.rsplit('/').next().unwrap_or(&f.name);
            let id = file.strip_suffix(".gguf").or_else(|| file.strip_suffix(".GGUF")).unwrap_or(file).to_string();
            RepoModel { id, quant: quant_of(&f.name), size: f.size, parts: vec![f], complete: true }
        })
        .collect();
    for ((path, of), mut parts) in splits {
        parts.sort_by_key(|(n, _)| *n);
        parts.dedup_by_key(|(n, _)| *n);
        let complete = parts.len() as u32 == of && parts[0].0 == 1;
        let id = path.rsplit('/').next().unwrap_or(&path).to_string();
        let parts: Vec<RepoFile> = parts.into_iter().map(|(_, f)| f).collect();
        let size = parts.iter().map(|p| p.size).sum();
        out.push(RepoModel { id, quant: quant_of(&format!("{path}.gguf")), size, parts, complete });
    }
    out.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.id.cmp(&b.id)));
    out
}

/// A Hub client: the base URL and this node's token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hub {
    url: String,
    token: Option<HfToken>,
}

impl Hub {
    /// The Hub at `url` (such as [`HUB`](crate::store::HUB) or a mirror), without a token.
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into().trim_end_matches('/').to_string(), token: None }
    }

    /// Sends `token` with requests (gated and private repos).
    pub fn with_token(mut self, token: Option<HfToken>) -> Self {
        self.token = token;
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    pub(crate) fn agent() -> ureq::Agent {
        ureq::Agent::config_builder().http_status_as_error(false).redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost).build().into()
    }

    pub(crate) fn get(&self, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
        let req = Self::agent().get(url);
        match &self.token {
            Some(t) => req.header("Authorization", t.bearer()),
            None => req,
        }
    }

    fn json(&self, req: ureq::RequestBuilder<ureq::typestate::WithoutBody>, what: &str) -> Result<Value> {
        let mut resp = req.header("Accept", "application/json").call().map_err(|e| Error::Http(e.to_string()))?;
        match resp.status().as_u16() {
            200 => {}
            s @ (401 | 403) => {
                let hint = if self.token.is_some() { "the token has no access" } else { "it may be gated or private; set HF_TOKEN on this node" };
                return Err(Error::Http(format!("the Hub answered {s} for {what} ({hint})")));
            }
            s => return Err(Error::Http(format!("the Hub answered {s} for {what}"))),
        }
        let text = resp.body_mut().read_to_string().map_err(|e| Error::Http(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| Error::Http(format!("the Hub's answer is not JSON: {e}")))
    }

    /// Repos matching the search, most downloaded first.
    pub fn search(&self, search: &Search) -> Result<Vec<SearchHit>> {
        let mut req = self
            .get(&format!("{}/api/models", self.url))
            .query("search", search.query.trim())
            .query("sort", "downloads")
            .query("direction", "-1")
            .query("limit", search.limit.clamp(1, 100).to_string());
        if search.gguf_only {
            req = req.query("filter", "gguf");
        }
        Ok(parse_search(&self.json(req, "the search")?))
    }

    /// The GGUF model files of `repo` (see [`repo_files`]).
    pub fn repo_files(&self, repo: &str) -> Result<Vec<RepoFile>> {
        check_repo(repo)?;
        let v = self.json(self.get(&format!("{}/api/models/{repo}?blobs=true", self.url)), repo)?;
        Ok(parse_files(&v))
    }

    /// The downloadable models of `repo`, split files grouped ([`group_files`]).
    pub fn models(&self, repo: &str) -> Result<Vec<RepoModel>> {
        Ok(group_files(&self.repo_files(repo)?))
    }

    /// The header of `file` in `repo` (its first [`gguf::HEADER_LIMIT`] bytes at most, through a
    /// `Range` request), for KV sizing. For a split model pass its first part.
    pub fn gguf_info(&self, repo: &str, file: &str) -> Result<GgufInfo> {
        check_repo(repo)?;
        check_file(file)?;
        let url = format!("{}/{repo}/resolve/main/{file}", self.url);
        let mut resp =
            self.get(&url).header("Range", format!("bytes=0-{}", gguf::HEADER_LIMIT - 1)).call().map_err(|e| Error::Http(e.to_string()))?;
        match resp.status().as_u16() {
            200 | 206 => {}
            s => return Err(Error::Http(format!("the Hub answered {s} for {file}"))),
        }
        gguf::read_info(resp.body_mut().as_reader())
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
    fn file_names_stay_inside_the_repo() {
        for ok in ["m.gguf", "Q4_K_M/m-00001-of-00002.gguf"] {
            assert!(check_file(ok).is_ok(), "{ok}");
        }
        for bad in ["", "/m.gguf", "../m.gguf", "a/../m.gguf", "a//m.gguf", "a\\m.gguf", "C:m.gguf", "./m.gguf"] {
            assert!(check_file(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn quants_come_from_file_names() {
        assert_eq!(quant_of("Qwen3-8B-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of("gpt-oss-20b-MXFP4.gguf").as_deref(), Some("MXFP4"));
        assert_eq!(quant_of("Q8_0/Big-Q8_0-00001-of-00002.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_of("m-UD-Q4_K_XL.gguf").as_deref(), Some("Q4_K_XL"));
        assert_eq!(quant_of("m-iq4_xs.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_of("m-bf16.gguf").as_deref(), Some("BF16"));
        assert_eq!(quant_of("model.gguf"), None);
        assert_eq!(quant_of("XQ4_0Y.gguf"), None);
    }

    fn rf(name: &str, size: u64) -> RepoFile {
        RepoFile { name: name.into(), size, sha256: None }
    }

    #[test]
    fn split_files_group_into_one_model() {
        assert_eq!(split_part("d/m-Q4_K_M-00002-of-00003.gguf"), Some(("m-Q4_K_M", 2, 3)));
        assert_eq!(split_part("m-00004-of-00003.gguf"), None);
        assert_eq!(split_part("m-1-of-3.gguf"), None);
        let files = vec![
            rf("Q8_0/m-Q8_0-00002-of-00002.gguf", 20),
            rf("m-Q4_K_M.gguf", 10),
            rf("Q8_0/m-Q8_0-00001-of-00002.gguf", 25),
            rf("BF16/m-BF16-00001-of-00003.gguf", 30),
            rf("BF16/m-BF16-00003-of-00003.gguf", 30),
        ];
        let models = group_files(&files);
        assert_eq!(
            models.iter().map(|m| (m.id.as_str(), m.size, m.complete)).collect::<Vec<_>>(),
            [("m-Q4_K_M", 10, true), ("m-Q8_0", 45, true), ("m-BF16", 60, false)]
        );
        let q8 = &models[1];
        assert!(q8.is_split());
        assert_eq!(q8.first().name, "Q8_0/m-Q8_0-00001-of-00002.gguf");
        assert_eq!(q8.quant.as_deref(), Some("Q8_0"));
        let specs = q8.specs("o/n").unwrap();
        assert_eq!(
            specs.iter().map(|s| (s.id.as_str(), s.file.as_str())).collect::<Vec<_>>(),
            [("m-Q8_0", "Q8_0/m-Q8_0-00001-of-00002.gguf"), ("m-Q8_0", "Q8_0/m-Q8_0-00002-of-00002.gguf")]
        );
        let mut bad = q8.clone();
        bad.parts[1].name = "../x.gguf".into();
        assert!(bad.specs("o/n").is_err());
    }

    const SEARCH_JSON: &str = r#"[
        {"id":"o/a-GGUF","downloads":50,"likes":3,"gated":false,"pipeline_tag":"text-generation","lastModified":"2026-11-01T00:00:00.000Z"},
        {"id":"o/b-GGUF","downloads":7,"gated":"manual"},
        {"id":"../bad","downloads":1}]"#;

    #[test]
    fn searches_send_the_query_and_the_token() {
        let hub = MockHub::start_with(vec![("/api/models".into(), "application/json", SEARCH_JSON.as_bytes().to_vec())]);
        let client = Hub::new(hub.url()).with_token(HfToken::new("hf_x"));
        let hits = client.search(&Search::new("qwen 3").with_limit(500)).unwrap();
        assert_eq!(hits.iter().map(|h| (h.id.as_str(), h.downloads, h.gated)).collect::<Vec<_>>(), [("o/a-GGUF", 50, false), ("o/b-GGUF", 7, true)]);
        assert_eq!(hits[0].pipeline_tag.as_deref(), Some("text-generation"));
        let seen = hub.seen();
        let path = &seen[0].path;
        assert!(
            path.contains("search=qwen") && path.contains("filter=gguf") && path.contains("limit=100") && path.contains("sort=downloads"),
            "{path}"
        );
        assert_eq!(seen[0].auth.as_deref(), Some("Bearer hf_x"));
        Hub::new(hub.url()).search(&Search::new("q").with_gguf_only(false)).unwrap();
        let seen = hub.seen();
        assert!(!seen[1].path.contains("filter=gguf"));
        assert_eq!(seen[1].auth, None, "no token, no header");
    }

    #[test]
    fn missing_repos_report_the_status() {
        let hub = MockHub::start_with(vec![]);
        let err = Hub::new(hub.url()).repo_files("o/n").unwrap_err();
        assert!(err.to_string().contains("404"), "{err}");
    }

    #[test]
    fn reads_a_header_with_a_range_request() {
        let mut body = crate::gguf::header(&[
            ("general.architecture", serde_json::json!("llama")),
            ("llama.block_count", serde_json::json!(2)),
            ("llama.attention.head_count", serde_json::json!(4)),
            ("llama.attention.head_count_kv", serde_json::json!(2)),
            ("llama.attention.key_length", serde_json::json!(16)),
        ]);
        body.extend(vec![0u8; 1000]);
        let hub = MockHub::start("/o/n/resolve/main/d/m.gguf", body);
        let info = Hub::new(hub.url()).gguf_info("o/n", "d/m.gguf").unwrap();
        assert_eq!(info.kv_bytes_per_token(), 2 * 2 * 32 * 2);
        assert_eq!(hub.last_range().as_deref(), Some("bytes=0-67108863"));
        assert!(Hub::new(hub.url()).gguf_info("o/n", "../m.gguf").is_err());
    }

    #[test]
    fn reads_a_repo_from_the_hub_api() {
        let hub = MockHub::start_with(vec![("/api/models/o/n".into(), "application/json", REPO_JSON.as_bytes().to_vec())]);
        assert_eq!(repo_files(hub.url(), "o/n").unwrap().len(), 2);
        assert!(matches!(repo_files(hub.url(), "o/missing"), Err(Error::Http(ref m)) if m.contains("404")));
        assert!(matches!(repo_files(hub.url(), "bad"), Err(Error::BadRepo(_))));
    }

    /// The real Hub: `cargo test -- --ignored live_hub` (reads a few MB of one header).
    #[test]
    #[ignore = "uses the network"]
    fn live_hub() {
        let hub = Hub::new(crate::store::HUB).with_token(HfToken::from_env());
        let hits = hub.search(&Search::new("Qwen3-0.6B").with_limit(5)).unwrap();
        assert!(!hits.is_empty());
        let models = hub.models("ggml-org/gemma-4-E2B-it-GGUF").unwrap();
        let q4 = models.iter().find(|m| m.quant.as_deref() == Some("Q4_0")).expect("a Q4_0 file");
        let info = hub.gguf_info("ggml-org/gemma-4-E2B-it-GGUF", &q4.first().name).unwrap();
        assert!(info.layers > 0 && info.kv_bytes_per_token() > 0, "{info:?}");
        let fit = crate::fit::choose(
            &models,
            &crate::hw::Budget { pool_bytes: 0, reserve_bytes: 0, usable_bytes: 8 << 30 },
            &crate::fit::FitOptions::default().with_info(&info),
        );
        assert!(fit.is_some());
        eprintln!("{} hits; {} models; {info:?}; best {:?}", hits.len(), models.len(), fit.map(|f| (f.model.id, f.footprint)));
    }
}
