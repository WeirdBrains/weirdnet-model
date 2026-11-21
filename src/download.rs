//! Consent-gated download of one model file, or every part of a split model: resumes a `.part`,
//! checks the size and the SHA-256, and only then renames it into place. Requests carry the
//! store's Hugging Face token when it may go to the store's Hub (see [`ModelStore::from_env`]).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::catalog::{ModelSpec, human};
use crate::store::ModelStore;
use crate::{Error, Result};

/// The user said yes to downloading this model. [`download`] needs one, so nothing downloads
/// without an explicit OK. Create it only after showing [`ModelSpec::consent_text`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consent {
    id: String,
}

impl Consent {
    pub fn given(spec: &ModelSpec) -> Self {
        Self { id: spec.id.clone() }
    }
}

/// How far a download is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

impl Progress {
    /// 0.0 to 1.0.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { (self.done as f64 / self.total as f64).min(1.0) as f32 }
    }
}

/// Downloads `spec` into `store`. `progress` returns `false` to cancel (the `.part` is kept so a
/// later call resumes). Returns the file's path; an already complete, matching file is kept.
pub fn download(store: &ModelStore, spec: &ModelSpec, consent: &Consent, progress: &mut dyn FnMut(Progress) -> bool) -> Result<PathBuf> {
    if consent.id != spec.id {
        return Err(Error::Consent);
    }
    crate::hf::check_file(&spec.file)?;
    let dest = store.path(spec);
    let part = store.part_path(spec);
    std::fs::create_dir_all(dest.parent().unwrap_or(store.dir()))?;
    let total = spec.size;
    if total > 0 && std::fs::metadata(&dest).is_ok_and(|m| m.len() == total) {
        match &spec.sha256 {
            Some(want) if !sha256_of(&dest)?.eq_ignore_ascii_case(want) => std::fs::remove_file(&dest)?,
            _ => {
                progress(Progress { done: total, total });
                return Ok(dest);
            }
        }
    }
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut done = have;
    if have < total || total == 0 {
        let mut req = store.hub_client().get(&spec.url(store.hub()));
        if have > 0 && have < total {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut resp = req.call().map_err(|e| Error::Http(e.to_string()))?;
        let status = resp.status().as_u16();
        done = match status {
            206 => have,
            200 => 0,
            s @ (401 | 403) if !store.has_token() => {
                return Err(Error::Http(format!("the Hub answered {s}; {} may be gated or private (set HF_TOKEN on this node)", spec.repo)));
            }
            s => return Err(Error::Http(format!("the Hub answered {s}"))),
        };
        let mut file = std::fs::OpenOptions::new().create(true).write(true).append(done > 0).truncate(done == 0).open(&part)?;
        let mut body = resp.body_mut().as_reader();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            if !progress(Progress { done, total }) {
                return Err(Error::Cancelled { done: human(done), total: human(total) });
            }
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            done += n as u64;
            if done > total {
                break;
            }
        }
        file.flush()?;
    }
    if done != total {
        if done > total {
            let _ = std::fs::remove_file(&part);
        }
        return Err(Error::Incomplete { done: human(done), total: human(total) });
    }
    if let Some(want) = &spec.sha256 {
        let got = sha256_of(&part)?;
        if !got.eq_ignore_ascii_case(want) {
            let _ = std::fs::remove_file(&part);
            return Err(Error::Checksum { name: spec.file.clone(), got });
        }
    }
    std::fs::rename(&part, &dest)?;
    progress(Progress { done, total });
    Ok(dest)
}

/// Downloads every part of a split model (or the one file of a single one), in order, with one
/// [`Consent`] for the model: all `parts` must share its id (as
/// [`RepoModel::specs`](crate::hf::RepoModel::specs) makes them). Progress counts all parts
/// together. Returns the first part's path, which llama.cpp loads the whole model from.
pub fn download_parts(store: &ModelStore, parts: &[ModelSpec], consent: &Consent, progress: &mut dyn FnMut(Progress) -> bool) -> Result<PathBuf> {
    let first = parts.first().ok_or_else(|| Error::Model("no files to download".into()))?;
    if parts.iter().any(|p| p.id != consent.id) {
        return Err(Error::Consent);
    }
    let total: u64 = parts.iter().map(|p| p.size).sum();
    let mut before = 0;
    for p in parts {
        download(store, p, consent, &mut |pr| progress(Progress { done: before + pr.done.min(p.size), total }))?;
        before += p.size;
    }
    Ok(store.path(first))
}

/// Lower-case hex SHA-256 of a file.
pub fn sha256_of(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockHub;

    fn body() -> Vec<u8> {
        (0..2_000_000u32).map(|i| (i % 251) as u8).collect()
    }

    fn spec_for(body: &[u8]) -> ModelSpec {
        let sum = Sha256::digest(body).iter().map(|b| format!("{b:02x}")).collect::<String>();
        ModelSpec { id: "t".into(), label: "Test".into(), repo: "o/n".into(), file: "t.gguf".into(), size: body.len() as u64, sha256: Some(sum) }
    }

    #[test]
    fn downloads_checks_and_renames() {
        let b = body();
        let hub = MockHub::start("/o/n/resolve/main/t.gguf", b.clone());
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url());
        let spec = spec_for(&b);
        let mut last = 0.0;
        let path = download(&store, &spec, &Consent::given(&spec), &mut |p| {
            last = p.fraction();
            true
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b);
        assert_eq!(last, 1.0);
        assert!(store.installed(&spec));
        assert_eq!(store.partial_bytes(&spec), 0);
        // A second call keeps the verified file and makes no request.
        let hits = hub.requests();
        download(&store, &spec, &Consent::given(&spec), &mut |_| true).unwrap();
        assert_eq!(hub.requests(), hits);
    }

    #[test]
    fn a_cancelled_download_resumes_with_a_range_request() {
        let b = body();
        let hub = MockHub::start("/o/n/resolve/main/t.gguf", b.clone());
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url());
        let spec = spec_for(&b);
        let mut calls = 0;
        let err = download(&store, &spec, &Consent::given(&spec), &mut |_| {
            calls += 1;
            calls < 2
        })
        .unwrap_err();
        assert!(matches!(err, Error::Cancelled { .. }), "{err}");
        let kept = store.partial_bytes(&spec);
        assert!(kept > 0 && kept < b.len() as u64, "kept {kept}");
        download(&store, &spec, &Consent::given(&spec), &mut |_| true).unwrap();
        assert_eq!(std::fs::read(store.path(&spec)).unwrap(), b);
        assert_eq!(hub.last_range(), Some(format!("bytes={kept}-")));
    }

    #[test]
    fn a_wrong_checksum_deletes_the_part_and_names_the_file() {
        let b = body();
        let hub = MockHub::start("/o/n/resolve/main/t.gguf", b.clone());
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url());
        let mut spec = spec_for(&b);
        spec.sha256 = Some("0".repeat(64));
        let err = download(&store, &spec, &Consent::given(&spec), &mut |_| true).unwrap_err();
        assert!(matches!(err, Error::Checksum { ref name, .. } if name == "t.gguf"), "{err}");
        assert_eq!(store.partial_bytes(&spec), 0);
        assert!(!store.installed(&spec));
    }

    #[test]
    fn consent_for_another_model_is_refused_and_http_errors_are_reported() {
        let b = body();
        let hub = MockHub::start("/o/n/resolve/main/other.gguf", b.clone());
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url());
        let spec = spec_for(&b);
        let mut other = spec.clone();
        other.id = "other".into();
        assert!(matches!(download(&store, &spec, &Consent::given(&other), &mut |_| true), Err(Error::Consent)));
        let err = download(&store, &spec, &Consent::given(&spec), &mut |_| true).unwrap_err();
        assert!(matches!(err, Error::Http(ref m) if m.contains("404")), "{err}");
    }

    #[test]
    fn split_models_download_every_part_into_their_directory() {
        let (a, b) = (body(), body()[..500_000].to_vec());
        let hub = MockHub::start_with(vec![
            ("/o/n/resolve/main/Q8_0/m-00001-of-00002.gguf".into(), "application/octet-stream", a.clone()),
            ("/o/n/resolve/main/Q8_0/m-00002-of-00002.gguf".into(), "application/octet-stream", b.clone()),
        ]);
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url()).with_token(crate::HfToken::new("hf_t"));
        let mut parts = Vec::new();
        for (name, data) in [("Q8_0/m-00001-of-00002.gguf", &a), ("Q8_0/m-00002-of-00002.gguf", &b)] {
            let mut s = spec_for(data);
            (s.id, s.file) = ("m".into(), name.into());
            parts.push(s);
        }
        let mut seen = Vec::new();
        let path = download_parts(&store, &parts, &Consent::given(&parts[0]), &mut |p| {
            seen.push(p);
            true
        })
        .unwrap();
        assert_eq!(path, tmp.path().join("Q8_0/m-00001-of-00002.gguf"));
        assert_eq!(std::fs::read(tmp.path().join("Q8_0/m-00002-of-00002.gguf")).unwrap(), b);
        let total = (a.len() + b.len()) as u64;
        assert!(seen.iter().all(|p| p.total == total));
        assert!(seen.windows(2).all(|w| w[0].done <= w[1].done), "progress only goes up");
        assert_eq!(seen.last().unwrap().done, total);
        assert!(hub.seen().iter().all(|s| s.auth.as_deref() == Some("Bearer hf_t")));
        let mut other = parts[0].clone();
        other.id = "x".into();
        assert!(matches!(download_parts(&store, &parts, &Consent::given(&other), &mut |_| true), Err(Error::Consent)));
        let mut evil = parts[0].clone();
        evil.file = "../evil.gguf".into();
        assert!(download(&store, &evil, &Consent::given(&evil), &mut |_| true).is_err());
    }

    #[test]
    fn a_short_body_is_incomplete_and_kept_for_resume() {
        let b = body();
        let hub = MockHub::start("/o/n/resolve/main/t.gguf", b[..1000].to_vec());
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path()).with_hub(hub.url());
        let spec = spec_for(&b);
        let err = download(&store, &spec, &Consent::given(&spec), &mut |_| true).unwrap_err();
        assert!(matches!(err, Error::Incomplete { .. }), "{err}");
        assert_eq!(store.partial_bytes(&spec), 1000);
    }
}
