//! A tiny HTTP server for tests: fixed routes, optional Range, counts requests and records the
//! `Range` and `Authorization` headers.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// One request: the path with its query, the `Range` header and the `Authorization` header.
#[derive(Debug, Clone)]
pub struct Seen {
    pub path: String,
    pub range: Option<String>,
    pub auth: Option<String>,
}

pub struct MockHub {
    url: String,
    log: Arc<Mutex<Vec<Seen>>>,
}

impl MockHub {
    /// Serves `body` at `path` (200, or 206 for `Range: bytes=N-`); any other path is a 404.
    pub fn start(path: &str, body: Vec<u8>) -> Self {
        Self::start_with(vec![(path.to_string(), "application/octet-stream", body)])
    }

    /// Routes match the path without its query string.
    pub fn start_with(routes: Vec<(String, &'static str, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let log: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let seen = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                if reader.read_line(&mut first).unwrap_or(0) == 0 {
                    continue;
                }
                let path = first.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (mut range, mut auth) = (None, None);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("range:") {
                        range = Some(v.trim().to_string());
                    }
                    if lower.starts_with("authorization:") {
                        auth = Some(line["authorization:".len()..].trim().to_string());
                    }
                }
                seen.lock().unwrap().push(Seen { path: path.clone(), range: range.clone(), auth });
                let Some((_, ctype, body)) = routes.iter().find(|(p, _, _)| *p == path.split('?').next().unwrap_or("")) else {
                    let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    continue;
                };
                let spec = range.as_deref().and_then(|r| r.strip_prefix("bytes="));
                let start = spec.and_then(|r| r.split('-').next()).and_then(|n| n.parse::<usize>().ok());
                let end = spec.and_then(|r| r.split('-').nth(1)).and_then(|n| n.parse::<usize>().ok());
                let (status, slice, extra) = match start {
                    Some(s) if s < body.len() => {
                        let e = end.map(|e| e.min(body.len() - 1)).unwrap_or(body.len() - 1);
                        ("206 Partial Content", &body[s..=e], format!("Content-Range: bytes {s}-{e}/{}\r\n", body.len()))
                    }
                    _ => ("200 OK", &body[..], String::new()),
                };
                let head =
                    format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n", slice.len());
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(slice);
            }
        });
        Self { url, log }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn requests(&self) -> usize {
        self.log.lock().unwrap().len()
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.log.lock().unwrap().clone()
    }

    pub fn last_range(&self) -> Option<String> {
        self.log.lock().unwrap().iter().rev().find_map(|s| s.range.clone())
    }
}

/// A Hub with adapters: repo `o/n` holds a model GGUF (`m-Q4_0.gguf`), a LoRA GGUF
/// (`m-LoRA-F16.gguf`, llama.cpp's keys) and a PEFT adapter in `tune/`; repo `o/peft` is a PEFT
/// adapter at its root, tagged as one. Files carry their SHA-256 like LFS files.
pub fn adapter_hub() -> MockHub {
    use sha2::{Digest, Sha256};
    let sum = |b: &[u8]| Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect::<String>();
    let mut model = crate::gguf::header(&[
        ("general.architecture", serde_json::json!("qwen3")),
        ("qwen3.block_count", serde_json::json!(2)),
        ("qwen3.attention.head_count", serde_json::json!(4)),
        ("qwen3.attention.key_length", serde_json::json!(16)),
    ]);
    model.resize(4000, 0);
    let mut lora = crate::gguf::lora_header("qwen3");
    lora.resize(1500, 1);
    let config = br#"{"peft_type":"LORA","r":16,"lora_alpha":32,"base_model_name_or_path":"Org/Base-8B","target_modules":["q_proj"]}"#.to_vec();
    let root_config = br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"base_model_name_or_path":"/home/someone/base"}"#.to_vec();
    let weights = vec![9u8; 2500];
    let repo = format!(
        r#"{{"id":"o/n","tags":["gguf"],"siblings":[
            {{"rfilename":"m-Q4_0.gguf","size":{},"lfs":{{"sha256":"{}"}}}},
            {{"rfilename":"m-LoRA-F16.gguf","size":{},"lfs":{{"sha256":"{}"}}}},
            {{"rfilename":"tune/adapter_config.json","size":{}}},
            {{"rfilename":"tune/adapter_model.safetensors","size":{},"lfs":{{"sha256":"{}"}}}},
            {{"rfilename":"README.md","size":3}}]}}"#,
        model.len(),
        sum(&model),
        lora.len(),
        sum(&lora),
        config.len(),
        weights.len(),
        sum(&weights)
    );
    let peft = format!(
        r#"{{"id":"o/peft","library_name":"peft","tags":["peft","base_model:adapter:Org/Base-8B"],"siblings":[
            {{"rfilename":"adapter_config.json","size":{}}},
            {{"rfilename":"adapter_model.safetensors","size":{},"lfs":{{"sha256":"{}"}}}}]}}"#,
        root_config.len(),
        weights.len(),
        sum(&weights)
    );
    let bin = "application/octet-stream";
    MockHub::start_with(vec![
        ("/api/models/o/n".into(), "application/json", repo.into_bytes()),
        ("/api/models/o/peft".into(), "application/json", peft.into_bytes()),
        ("/o/n/resolve/main/m-Q4_0.gguf".into(), bin, model),
        ("/o/n/resolve/main/m-LoRA-F16.gguf".into(), bin, lora),
        ("/o/n/resolve/main/tune/adapter_config.json".into(), "application/json", config),
        ("/o/n/resolve/main/tune/adapter_model.safetensors".into(), bin, weights.clone()),
        ("/o/peft/resolve/main/adapter_config.json".into(), "application/json", root_config),
        ("/o/peft/resolve/main/adapter_model.safetensors".into(), bin, weights),
    ])
}
