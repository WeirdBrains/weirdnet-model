//! Chat prompts: the model's own Jinja template, a Gemma-style fallback, and the `<tool_call>`
//! tag format for tools.

use serde_json::Value;

use crate::{Error, Result};

pub const TOOL_OPEN: &str = "<tool_call>";
pub const TOOL_CLOSE: &str = "</tool_call>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// A tool as the prompt describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDesc {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments.
    pub schema: Value,
}

/// System text describing the tools and the call format; empty without tools.
pub fn tool_prompt(tools: &[ToolDesc]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "You can call tools. To call one, reply with only\n<tool_call>{\"name\": \"TOOL\", \"arguments\": {...}}</tool_call>\nand wait for the result. Tools (JSON Schema arguments):\n",
    );
    for t in tools {
        s.push_str(&format!("- {}: {} {}\n", t.name, t.description, t.schema));
    }
    s
}

/// A call written back into a prompt (an earlier assistant turn).
pub fn call_text(name: &str, args: &Value) -> String {
    format!("{TOOL_OPEN}{}{TOOL_CLOSE}", serde_json::json!({"name": name, "arguments": args}))
}

/// A tool result written into a prompt.
pub fn result_text(name: &str, content: &str, is_error: bool) -> String {
    format!("<tool_result name=\"{name}\"{}>{content}</tool_result>", if is_error { " error=\"true\"" } else { "" })
}

/// Renders `turns` with the model's Jinja chat template (GGUF `tokenizer.chat_template`), ending
/// with the generation prompt.
pub fn render(template: &str, turns: &[ChatMessage], bos: &str) -> Result<String> {
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |m: String| -> std::result::Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
    });
    let model = |e: minijinja::Error| Error::Model(format!("chat template: {e}"));
    env.add_template("chat", template).map_err(model)?;
    let msgs: Vec<minijinja::Value> = turns.iter().map(|m| minijinja::context! { role => m.role.as_str(), content => &m.content }).collect();
    env.get_template("chat")
        .map_err(model)?
        .render(minijinja::context! { messages => minijinja::Value::from(msgs), add_generation_prompt => true, bos_token => bos, eos_token => "" })
        .map_err(model)
}

/// A plain Gemma-style prompt for a model with no usable template.
pub fn render_fallback(turns: &[ChatMessage]) -> String {
    let mut s = String::new();
    let mut pending_sys = String::new();
    for m in turns {
        match m.role {
            ChatRole::System => pending_sys = format!("{}\n\n", m.content),
            ChatRole::User => s.push_str(&format!("<start_of_turn>user\n{}{}<end_of_turn>\n", std::mem::take(&mut pending_sys), m.content)),
            ChatRole::Assistant => s.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", m.content)),
        }
    }
    s.push_str("<start_of_turn>model\n");
    s
}

/// What a [`ToolStream`] found in the text.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Text(String),
    Call { name: String, arguments: Value },
}

/// Splits streamed text into plain text and `<tool_call>…</tool_call>` calls, holding back a
/// possible partial tag at the end of a chunk.
#[derive(Debug, Default)]
pub struct ToolStream {
    buf: String,
    in_call: bool,
    calls: usize,
}

impl ToolStream {
    pub fn calls(&self) -> usize {
        self.calls
    }

    pub fn feed(&mut self, piece: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        self.buf.push_str(piece);
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(TOOL_CLOSE) else { return out };
                let body: String = self.buf[..end].to_string();
                self.buf.drain(..end + TOOL_CLOSE.len());
                self.in_call = false;
                out.push(self.call(&body));
            } else if let Some(i) = self.buf.find(TOOL_OPEN) {
                if i > 0 {
                    out.push(Piece::Text(self.buf[..i].to_string()));
                }
                self.buf.drain(..i + TOOL_OPEN.len());
                self.in_call = true;
            } else {
                let keep = (1..TOOL_OPEN.len())
                    .rev()
                    .find(|&k| {
                        self.buf.len() >= k && self.buf.is_char_boundary(self.buf.len() - k) && TOOL_OPEN.starts_with(&self.buf[self.buf.len() - k..])
                    })
                    .unwrap_or(0);
                let cut = self.buf.len() - keep;
                if cut > 0 {
                    out.push(Piece::Text(self.buf[..cut].to_string()));
                    self.buf.drain(..cut);
                }
                return out;
            }
        }
    }

    fn call(&mut self, body: &str) -> Piece {
        let v: Value = serde_json::from_str(body.trim()).unwrap_or(Value::Null);
        self.calls += 1;
        let arguments = if v["arguments"].is_null() { Value::Object(Default::default()) } else { v["arguments"].clone() };
        Piece::Call { name: v["name"].as_str().unwrap_or("").to_string(), arguments }
    }

    /// End of stream: flushes held text; an unterminated call is parsed as is.
    pub fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        if self.in_call {
            self.in_call = false;
            vec![self.call(&rest)]
        } else if rest.is_empty() {
            Vec::new()
        } else {
            vec![Piece::Text(rest)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(p: &[Piece]) -> String {
        p.iter().filter_map(|p| if let Piece::Text(t) = p { Some(t.as_str()) } else { None }).collect()
    }

    #[test]
    fn a_template_renders_roles_and_the_generation_prompt() {
        let t = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<model>{% endif %}";
        let turns = [ChatMessage::new(ChatRole::System, "be brief"), ChatMessage::new(ChatRole::User, "hi")];
        assert_eq!(render(t, &turns, "").unwrap(), "<system>be brief\n<user>hi\n<model>");
        assert!(render("{{ raise_exception('no') }}", &turns, "").is_err());
        assert!(render("{% bad", &turns, "").is_err());
    }

    #[test]
    fn the_fallback_folds_the_system_text_into_the_first_user_turn() {
        let turns = [ChatMessage::new(ChatRole::System, "S"), ChatMessage::new(ChatRole::User, "U"), ChatMessage::new(ChatRole::Assistant, "A")];
        assert_eq!(render_fallback(&turns), "<start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\nA<end_of_turn>\n<start_of_turn>model\n");
    }

    #[test]
    fn tool_text_round_trips_through_the_stream_parser() {
        let call = call_text("open_url", &serde_json::json!({"url": "https://a.com"}));
        let mut s = ToolStream::default();
        let mut got = Vec::new();
        for chunk in ["Sure. <tool", "_call>{\"name\":\"open_url\",\"argu", "ments\":{\"url\":\"https://a.com\"}}</tool_", "call> done"] {
            got.extend(s.feed(chunk));
        }
        got.extend(s.finish());
        assert_eq!(text(&got), "Sure.  done");
        assert!(got.contains(&Piece::Call { name: "open_url".into(), arguments: serde_json::json!({"url": "https://a.com"}) }));
        assert_eq!(s.calls(), 1);
        let mut s2 = ToolStream::default();
        assert!(matches!(s2.feed(&call).as_slice(), [Piece::Call { name, .. }] if name == "open_url"));
    }

    #[test]
    fn a_partial_tag_is_held_back_and_plain_text_flows() {
        let mut s = ToolStream::default();
        assert_eq!(s.feed("a <"), vec![Piece::Text("a ".into())]);
        assert_eq!(s.feed("b"), vec![Piece::Text("<b".into())]);
        assert!(s.finish().is_empty());
        let mut s = ToolStream::default();
        assert!(s.feed("<tool_call>{\"name\":\"x\"}").is_empty());
        assert_eq!(s.finish(), vec![Piece::Call { name: "x".into(), arguments: serde_json::json!({}) }]);
    }

    #[test]
    fn tools_are_described_and_results_marked() {
        assert_eq!(tool_prompt(&[]), "");
        let p = tool_prompt(&[ToolDesc { name: "t".into(), description: "does t".into(), schema: serde_json::json!({"type": "object"}) }]);
        assert!(p.contains("- t: does t {\"type\":\"object\"}"));
        assert_eq!(result_text("t", "ok", true), "<tool_result name=\"t\" error=\"true\">ok</tool_result>");
    }
}
