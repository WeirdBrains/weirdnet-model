//! Inference on a worker thread: callers submit a request and read streamed text, and can cancel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::prompt::ChatMessage;
use crate::{Error, Result};

/// One generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    /// Longest reply in tokens; the engine's default when `None`.
    pub max_tokens: Option<u32>,
    /// 0 or less is greedy; the engine's default when `None`.
    pub temperature: Option<f32>,
}

impl GenerateRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self { messages, max_tokens: None, temperature: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCause {
    End,
    MaxTokens,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub stop: StopCause,
}

/// A loaded model. `on_text` gets each piece of text as it is sampled and returns `false` to stop.
pub trait Engine: Send + 'static {
    fn generate(&mut self, req: &GenerateRequest, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Stats>;
}

/// What a generation streams back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chunk {
    Text(String),
    /// The last chunk of a successful or cancelled generation.
    Done(Stats),
    /// The last chunk of a failed one (the model could not load, or sampling failed).
    Failed(String),
}

/// A running generation.
pub struct Generation {
    rx: UnboundedReceiver<Chunk>,
    cancel: Arc<AtomicBool>,
}

/// Stops a generation from another thread or task.
#[derive(Debug, Clone)]
pub struct Canceller(Arc<AtomicBool>);

impl Canceller {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Generation {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn canceller(&self) -> Canceller {
        Canceller(self.cancel.clone())
    }

    /// The next chunk; `None` once the generation has ended.
    pub async fn recv(&mut self) -> Option<Chunk> {
        self.rx.recv().await
    }

    /// Like [`recv`](Self::recv), blocking the thread (not for async code).
    pub fn blocking_recv(&mut self) -> Option<Chunk> {
        self.rx.blocking_recv()
    }

    /// Waits for the whole reply: its text and how it ended.
    pub fn collect_blocking(mut self) -> Result<(String, Stats)> {
        let mut text = String::new();
        while let Some(chunk) = self.blocking_recv() {
            match chunk {
                Chunk::Text(t) => text.push_str(&t),
                Chunk::Done(stats) => return Ok((text, stats)),
                Chunk::Failed(e) => return Err(Error::Model(e)),
            }
        }
        Err(Error::Model("the worker stopped".into()))
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

struct Job {
    req: GenerateRequest,
    tx: UnboundedSender<Chunk>,
    cancel: Arc<AtomicBool>,
}

/// An engine on its own thread. The engine is loaded on the first request (and again after a
/// failed load), so creating a worker is cheap. Requests run one at a time, in order.
pub struct Worker {
    jobs: Sender<Job>,
}

impl Worker {
    pub fn spawn(mut load: impl FnMut() -> Result<Box<dyn Engine>> + Send + 'static) -> Self {
        let (jobs, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("weirdnet-model".into())
            .spawn(move || {
                let mut engine: Option<Box<dyn Engine>> = None;
                while let Ok(job) = rx.recv() {
                    if job.cancel.load(Ordering::SeqCst) {
                        let stats = Stats { prompt_tokens: 0, generated_tokens: 0, stop: StopCause::Cancelled };
                        let _ = job.tx.send(Chunk::Done(stats));
                        continue;
                    }
                    if engine.is_none() {
                        match load() {
                            Ok(e) => engine = Some(e),
                            Err(e) => {
                                let _ = job.tx.send(Chunk::Failed(e.to_string()));
                                continue;
                            }
                        }
                    }
                    let Some(e) = engine.as_mut() else { continue };
                    let (tx, cancel) = (job.tx.clone(), job.cancel.clone());
                    let result = e.generate(&job.req, &mut |piece| !cancel.load(Ordering::SeqCst) && tx.send(Chunk::Text(piece.to_string())).is_ok());
                    let _ = job.tx.send(match result {
                        Ok(mut stats) => {
                            if job.cancel.load(Ordering::SeqCst) {
                                stats.stop = StopCause::Cancelled;
                            }
                            Chunk::Done(stats)
                        }
                        Err(e) => Chunk::Failed(e.to_string()),
                    });
                }
            })
            .expect("spawn the model worker");
        Self { jobs }
    }

    /// Queues a request and returns its stream.
    pub fn generate(&self, req: GenerateRequest) -> Generation {
        let (tx, rx) = unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if self.jobs.send(Job { req, tx: tx.clone(), cancel: cancel.clone() }).is_err() {
            let _ = tx.send(Chunk::Failed("the worker stopped".into()));
        }
        Generation { rx, cancel }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::prompt::ChatRole;

    /// Says each word of the last user message back, one piece per word.
    pub struct Echo {
        pub delay_ms: u64,
    }

    impl Engine for Echo {
        fn generate(&mut self, req: &GenerateRequest, on_text: &mut dyn FnMut(&str) -> bool) -> Result<Stats> {
            let last = req.messages.iter().rev().find(|m| m.role == ChatRole::User).map(|m| m.content.clone()).unwrap_or_default();
            let mut n = 0;
            for w in last.split_whitespace() {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
                if !on_text(&format!("{w} ")) {
                    return Ok(Stats { prompt_tokens: 1, generated_tokens: n, stop: StopCause::Cancelled });
                }
                n += 1;
            }
            Ok(Stats { prompt_tokens: 1, generated_tokens: n, stop: StopCause::End })
        }
    }

    fn req(text: &str) -> GenerateRequest {
        GenerateRequest::new(vec![ChatMessage::new(ChatRole::User, text)])
    }

    #[test]
    fn text_streams_in_order_and_ends_with_stats() {
        let w = Worker::spawn(|| Ok(Box::new(Echo { delay_ms: 0 })));
        let mut g = w.generate(req("one two three"));
        let mut pieces = Vec::new();
        let stats = loop {
            match g.blocking_recv().unwrap() {
                Chunk::Text(t) => pieces.push(t),
                Chunk::Done(s) => break s,
                Chunk::Failed(e) => panic!("{e}"),
            }
        };
        assert_eq!(pieces, ["one ", "two ", "three "]);
        assert_eq!((stats.generated_tokens, stats.stop), (3, StopCause::End));
    }

    #[test]
    fn cancelling_stops_a_slow_reply() {
        let w = Worker::spawn(|| Ok(Box::new(Echo { delay_ms: 20 })));
        let mut g = w.generate(req(&"w ".repeat(500)));
        assert!(matches!(g.blocking_recv(), Some(Chunk::Text(_))));
        g.canceller().cancel();
        let mut last = None;
        while let Some(c) = g.blocking_recv() {
            last = Some(c);
        }
        assert!(matches!(last, Some(Chunk::Done(Stats { stop: StopCause::Cancelled, generated_tokens, .. })) if generated_tokens < 500));
    }

    #[test]
    fn a_failed_load_is_reported_and_retried_on_the_next_request() {
        let tries = Arc::new(AtomicUsize::new(0));
        let t = tries.clone();
        let w = Worker::spawn(move || {
            if t.fetch_add(1, Ordering::SeqCst) == 0 { Err(Error::Model("no file".into())) } else { Ok(Box::new(Echo { delay_ms: 0 })) }
        });
        let err = w.generate(req("a")).collect_blocking().unwrap_err();
        assert!(err.to_string().contains("no file"), "{err}");
        let (text, _) = w.generate(req("b")).collect_blocking().unwrap();
        assert_eq!(text, "b ");
        assert_eq!(tries.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn requests_run_one_at_a_time_in_order_and_a_dropped_stream_cancels() {
        let w = Worker::spawn(|| Ok(Box::new(Echo { delay_ms: 5 })));
        let first = w.generate(req("a b"));
        let second = w.generate(req("c d"));
        assert_eq!(first.collect_blocking().unwrap().0, "a b ");
        assert_eq!(second.collect_blocking().unwrap().0, "c d ");
        let dropped = w.generate(req(&"x ".repeat(1000)));
        drop(dropped);
        assert_eq!(w.generate(req("after")).collect_blocking().unwrap().0, "after ");
    }

    #[tokio::test]
    async fn async_callers_can_await_chunks() {
        let w = Worker::spawn(|| Ok(Box::new(Echo { delay_ms: 0 })));
        let mut g = w.generate(req("hi there"));
        let mut text = String::new();
        while let Some(c) = g.recv().await {
            if let Chunk::Text(t) = c {
                text.push_str(&t);
            }
        }
        assert_eq!(text, "hi there ");
    }
}
