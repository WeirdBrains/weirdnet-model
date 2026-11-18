//! A tiny HTTP server for tests: one path, optional Range, counts requests.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

type Log = Vec<(String, Option<String>)>;

pub struct MockHub {
    url: String,
    log: Arc<Mutex<Log>>,
}

impl MockHub {
    /// Serves `body` at `path` (200, or 206 for `Range: bytes=N-`); any other path is a 404.
    pub fn start(path: &str, body: Vec<u8>) -> Self {
        Self::start_with(vec![(path.to_string(), "application/octet-stream", body)])
    }

    pub fn start_with(routes: Vec<(String, &'static str, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let log: Arc<Mutex<Log>> = Arc::default();
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
                let mut range = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range:") {
                        range = Some(v.trim().to_string());
                    }
                }
                seen.lock().unwrap().push((path.clone(), range.clone()));
                let Some((_, ctype, body)) = routes.iter().find(|(p, _, _)| *p == path.split('?').next().unwrap_or("")) else {
                    let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    continue;
                };
                let start =
                    range.as_deref().and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.strip_suffix('-')).and_then(|n| n.parse::<usize>().ok());
                let (status, slice, extra) = match start {
                    Some(s) if s < body.len() => {
                        ("206 Partial Content", &body[s..], format!("Content-Range: bytes {s}-{}/{}\r\n", body.len() - 1, body.len()))
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

    pub fn last_range(&self) -> Option<String> {
        self.log.lock().unwrap().iter().rev().find_map(|(_, r)| r.clone())
    }
}
