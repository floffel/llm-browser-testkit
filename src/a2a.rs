//! A2A (Agent-to-Agent) protocol client.
//!
//! Implements the A2A JSON-RPC over HTTP protocol for sending tasks to
//! remote agents and receiving responses.

use std::time::Duration;

use anyhow::Context;
use serde_json::Value;

/// A2A client for communicating with a remote agent.
#[derive(Debug)]
pub struct A2aClient {
    /// Agent base URL.
    url: String,
    /// HTTP client for requests.
    client: reqwest::Client,
}

impl A2aClient {
    /// Creates a new A2A client for the given agent URL.
    #[must_use]
    pub fn new(url: &str, timeout: Duration) -> Self {
        Self {
            url: url.trim_end_matches('/').to_owned(),
            client: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .expect("build reqwest client"),
        }
    }

    /// Sends a task to the agent and returns the response text.
    ///
    /// Uses the A2A JSON-RPC protocol:
    /// `{"jsonrpc": "2.0", "method": "tasks/send", "params": {...}, "id": 1}`
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot
    /// be parsed.
    pub async fn send_task(&self, task: &str) -> anyhow::Result<String> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task}]
                }
            },
            "id": 1
        });

        let resp = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("A2A: request failed")?;

        let json: Value = resp.json().await.context("A2A: failed to parse response")?;

        // Try extracting the result from the A2A response.
        // The response structure is:
        // {"jsonrpc": "2.0", "result": {"id": "...", "messages": [{"parts": [{"text": "..."}]}]}, "id": 1}
        if let Some(text) = extract_a2a_text(&json) {
            return Ok(text);
        }

        // Fallback: try error extraction
        if let Some(error) = json["error"]["message"].as_str() {
            anyhow::bail!("A2A error: {error}");
        }

        Ok(serde_json::to_string(&json)?)
    }
}

/// Extracts the text content from an A2A response.
fn extract_a2a_text(value: &Value) -> Option<String> {
    value["result"]["messages"]
        .as_array()
        .and_then(|msgs| msgs.last())
        .and_then(|msg| msg["parts"].as_array())
        .and_then(|parts| parts.iter().find_map(|p| p["text"].as_str()))
        .map(String::from)
}
