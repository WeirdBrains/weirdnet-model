//! `weirdnet_agent::ModelBackend` over a [`Worker`]: the agent's messages become chat turns (tools
//! in the `<tool_call>` tag format), text streams to the session, and tool calls come back parsed.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use weirdnet_agent::{CancelToken, DeltaSink, Message, ModelBackend, ModelError, ModelTurn, Role, ToolCall, ToolSpec};

use crate::prompt::{self, ChatMessage, ChatRole, Piece, ToolDesc, ToolStream};
use crate::worker::{Chunk, GenerateRequest, StopCause, Worker};

/// A local model for the agent loop.
pub struct LocalModel {
    name: String,
    worker: Arc<Worker>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl LocalModel {
    /// `name` shows in traces, for example `gemma-4-e4b`.
    pub fn new(name: impl Into<String>, worker: Arc<Worker>) -> Self {
        Self { name: name.into(), worker, max_tokens: None, temperature: None }
    }
}

/// The agent's messages as chat turns: tools described in the system turn, calls and results
/// written in tag form.
pub fn turns(messages: &[Message], tools: &[ToolSpec]) -> Vec<ChatMessage> {
    let descs: Vec<ToolDesc> =
        tools.iter().map(|t| ToolDesc { name: t.name.clone(), description: t.description.clone(), schema: t.input_schema.clone() }).collect();
    let tool_text = prompt::tool_prompt(&descs);
    let mut out: Vec<ChatMessage> = Vec::new();
    let mut system: Vec<String> = messages.iter().filter(|m| m.role == Role::System).map(|m| m.content.clone()).collect();
    if !tool_text.is_empty() {
        system.push(tool_text);
    }
    if !system.is_empty() {
        out.push(ChatMessage::new(ChatRole::System, system.join("\n\n")));
    }
    for m in messages {
        match m.role {
            Role::System => {}
            Role::User => out.push(ChatMessage::new(ChatRole::User, m.content.clone())),
            Role::Assistant => {
                let mut text = m.content.clone();
                for c in &m.tool_calls {
                    text.push_str(&prompt::call_text(&c.name, &c.arguments));
                }
                out.push(ChatMessage::new(ChatRole::Assistant, text));
            }
            Role::Tool => {
                let name = messages
                    .iter()
                    .flat_map(|a| a.tool_calls.iter())
                    .find(|c| Some(&c.id) == m.tool_call_id.as_ref())
                    .map(|c| c.name.as_str())
                    .unwrap_or("tool");
                out.push(ChatMessage::new(ChatRole::User, prompt::result_text(name, &m.content, false)));
            }
            // Roles added to `weirdnet_agent::Role` later are not sent to the model.
            _ => {}
        }
    }
    out
}

#[async_trait]
impl ModelBackend for LocalModel {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn complete(&self, messages: &[Message], tools: &[ToolSpec], deltas: &DeltaSink, cancel: &CancelToken) -> Result<ModelTurn, ModelError> {
        let mut req = GenerateRequest::new(turns(messages, tools));
        req.max_tokens = self.max_tokens;
        req.temperature = self.temperature;
        let mut generation = self.worker.generate(req);
        let canceller = generation.canceller();
        let mut parser = ToolStream::default();
        let mut turn = ModelTurn::default();
        let take = |pieces: Vec<Piece>, turn: &mut ModelTurn| {
            for p in pieces {
                match p {
                    Piece::Text(t) => {
                        deltas.text(t.clone());
                        turn.text.push_str(&t);
                    }
                    Piece::Call { name, arguments } => {
                        let id = format!("call_{}", turn.tool_calls.len());
                        turn.tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments: if arguments.is_null() { Value::Object(Default::default()) } else { arguments },
                        });
                    }
                }
            }
        };
        loop {
            let chunk = tokio::select! {
                c = generation.recv() => c,
                () = cancel.cancelled() => {
                    canceller.cancel();
                    return Err(ModelError::Cancelled);
                }
            };
            match chunk {
                Some(Chunk::Text(t)) => take(parser.feed(&t), &mut turn),
                Some(Chunk::Done(stats)) => {
                    if stats.stop == StopCause::Cancelled {
                        return Err(ModelError::Cancelled);
                    }
                    take(parser.finish(), &mut turn);
                    return Ok(turn);
                }
                Some(Chunk::Failed(e)) => return Err(ModelError::Failed(e)),
                None => return Err(ModelError::Failed("the model worker stopped".into())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::tests::Echo;
    use crate::{Error, Result};

    fn worker(delay_ms: u64) -> Arc<Worker> {
        Arc::new(Worker::spawn(move || -> Result<Box<dyn crate::Engine>> { Ok(Box::new(Echo { delay_ms })) }))
    }

    #[test]
    fn turns_carry_tools_calls_and_results() {
        let spec = ToolSpec::new("open_url", "Open a URL");
        let call = ToolCall { id: "c1".into(), name: "open_url".into(), arguments: serde_json::json!({"url": "https://a.com"}) };
        let msgs = [Message::system("be brief"), Message::user("open it"), Message::assistant("", vec![call]), Message::tool("c1", "opened")];
        let t = turns(&msgs, &[spec]);
        assert_eq!(t[0].role, ChatRole::System);
        assert!(t[0].content.starts_with("be brief\n\nYou can call tools"));
        assert_eq!(t[1].content, "open it");
        assert!(t[2].content.starts_with("<tool_call>") && t[2].role == ChatRole::Assistant);
        assert_eq!(t[3].content, "<tool_result name=\"open_url\">opened</tool_result>");
    }

    #[tokio::test]
    async fn streams_text_to_the_session_and_returns_the_turn() {
        let model = LocalModel::new("echo", worker(0));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = DeltaSink::new(Some(tx));
        let turn = model.complete(&[Message::user("hello world")], &[], &sink, &CancelToken::new()).await.unwrap();
        assert_eq!(turn.text, "hello world ");
        drop(sink);
        let mut deltas = 0;
        while rx.recv().await.is_some() {
            deltas += 1;
        }
        assert_eq!(deltas, 2);
        assert_eq!(model.name(), "echo");
    }

    #[tokio::test]
    async fn a_tool_call_in_the_output_becomes_a_tool_call() {
        let model = LocalModel::new("echo", worker(0));
        let said = "<tool_call>{\"name\":\"open_url\",\"arguments\":{\"url\":\"https://a.com\"}}</tool_call>";
        let turn = model.complete(&[Message::user(said)], &[], &DeltaSink::discard(), &CancelToken::new()).await.unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "open_url");
        assert_eq!(turn.tool_calls[0].arguments["url"], "https://a.com");
    }

    #[tokio::test]
    async fn cancelling_the_token_stops_the_model() {
        let model = LocalModel::new("echo", worker(10));
        let cancel = CancelToken::new();
        let c2 = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            c2.cancel();
        });
        let long = "w ".repeat(1000);
        let err = model.complete(&[Message::user(long)], &[], &DeltaSink::discard(), &cancel).await.unwrap_err();
        assert_eq!(err, ModelError::Cancelled);
    }

    #[tokio::test]
    async fn a_load_failure_is_a_model_failure() {
        let w = Arc::new(Worker::spawn(|| -> Result<Box<dyn crate::Engine>> { Err(Error::Model("missing file".into())) }));
        let err = LocalModel::new("x", w).complete(&[Message::user("hi")], &[], &DeltaSink::discard(), &CancelToken::new()).await.unwrap_err();
        assert!(matches!(err, ModelError::Failed(ref m) if m.contains("missing file")), "{err}");
    }
}
