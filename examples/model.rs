//! A small CLI over the crate: `fetch [id]` downloads (consent is the command itself) with
//! progress; `chat [id] "question"` loads the model and streams the answer with timings.
//!
//!     cargo run --release --example model --features llama -- fetch gemma-4-e4b
//!     cargo run --release --example model --features llama -- chat gemma-4-e4b "Why is the sky blue?"

use std::io::Write;
use std::time::Instant;

use weirdnet_model::llama::{LlamaEngine, LlamaOptions};
use weirdnet_model::{ChatMessage, ChatRole, Chunk, Consent, GenerateRequest, ModelStore, Worker, catalog, download, human};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let id = args.get(1).map(String::as_str).unwrap_or(catalog::GEMMA_4_E4B);
    let spec = catalog::find(id).unwrap_or_else(|| panic!("unknown model {id}"));
    let store = ModelStore::from_env();
    match cmd {
        "fetch" => {
            println!("{}", spec.consent_text(store.hub()).join("\n"));
            println!("into {}", store.dir().display());
            let mut last = 0;
            let path = download(&store, &spec, &Consent::given(&spec), &mut |p| {
                let pct = (p.fraction() * 100.0) as u32;
                if pct != last {
                    last = pct;
                    println!("{pct}% ({} of {})", human(p.done), human(p.total));
                }
                true
            })
            .expect("download");
            println!("done: {} (size and SHA-256 verified)", path.display());
        }
        "chat" => {
            let q = args.get(2).cloned().unwrap_or_else(|| "Say hello in one sentence.".into());
            assert!(store.installed(&spec), "run `fetch` first");
            let t0 = Instant::now();
            let worker = Worker::spawn(LlamaEngine::loader(store.path(&spec), LlamaOptions::default()));
            let mut req = GenerateRequest::new(vec![ChatMessage::new(ChatRole::User, q)]);
            req.max_tokens = Some(128);
            let mut g = worker.generate(req);
            let (mut first, mut n) = (None, 0u64);
            while let Some(c) = g.blocking_recv() {
                match c {
                    Chunk::Text(t) => {
                        first.get_or_insert_with(|| t0.elapsed());
                        n += 1;
                        print!("{t}");
                        std::io::stdout().flush().ok();
                    }
                    Chunk::Done(s) => println!("\n[stats {s:?}]"),
                    Chunk::Failed(e) => panic!("{e}"),
                }
            }
            let total = t0.elapsed();
            println!("[load+first-chunk {:?}, total {:?}, {n} chunks]", first.unwrap_or_default(), total);
        }
        _ => eprintln!("usage: model fetch|chat [id] [question]"),
    }
}
