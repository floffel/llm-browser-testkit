//! LLM-driven browser test framework.
//!
//! Provides reusable building blocks for browser-based test scenarios:
//! - Browser client via Chrome `DevTools` Protocol (headless)
//! - LLM client for natural language element targeting and assertions
//! - A2A agent integration for agent-to-agent communication
//! - MCP client/server integration for tool-calling
//! - Cost tracking, token counting, and budget enforcement
//! - Declarative TOML scenario runner
//! - `#[browser_test]` macros for `cargo test` integration

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

/// A2A agent protocol client.
pub mod a2a;
/// Budget tracking and enforcement.
pub mod budgets;
/// Cost calculation, usage tracking, and pricing.
pub mod costs;
/// Endpoint registry and routing resolver.
pub mod endpoints;
/// MCP client for connecting to external MCP servers.
pub mod mcp_client;
/// MCP server for exposing the framework as an MCP server.
pub mod mcp_server;
/// Cost and token report printer.
pub mod reporting;
/// Step-by-step scenario executor (navigate, click, type, wait, assert).
pub mod runner;
/// Declarative TOML-based test scenario types.
pub mod scenario;

/// `#[browser_test]` macros for `cargo test` integration.
#[cfg(feature = "macros")]
pub mod macros;

use std::collections::HashMap;
use std::time::Duration;

use serde_json::Value;

pub use costs::LlmResponse;
pub use costs::LlmUsage;

/// Configuration for the LLM client — bundles URL, model, auth, timeouts,
/// and provider-specific options into a single struct passed everywhere.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// OpenAI-compatible API base URL (without trailing `/v1/…`).
    pub url: String,
    /// Model name (e.g. `gpt-4o-mini`, `deepseek`).
    pub model: String,
    /// API key sent as `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Custom headers appended to every LLM request.
    pub headers: HashMap<String, String>,
    /// HTTP timeout.
    pub timeout: Duration,
    /// Sampling temperature (0.0–1.0).
    pub temperature: f64,
    /// Enable extended thinking / reasoning tokens.
    /// `None` = don't send any thinking key (provider default).
    pub thinking: Option<bool>,
    /// Provider-specific parameters merged into the request body
    /// (e.g. `effort = "high"` for Anthropic).
    pub model_params: HashMap<String, Value>,
}

impl LlmConfig {
    /// Build a config from environment defaults, falling back to safe
    /// values when no env vars are set.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            url: llm_base_url(),
            model: llm_model(),
            api_key: std::env::var("HARNESS_LLM_API_KEY").ok(),
            headers: parse_headers_env(),
            timeout: Duration::from_secs(60),
            temperature: 0.0,
            thinking: None,
            model_params: HashMap::new(),
        }
    }
}

/// Parses `HARNESS_LLM_HEADERS` env var (JSON object) into a header map.
///
/// Exposed for use by `endpoints.rs` and tests.
#[must_use]
pub fn parse_headers_env() -> HashMap<String, String> {
    let Ok(raw) = std::env::var("HARNESS_LLM_HEADERS") else {
        return HashMap::new();
    };
    let Ok(json) = serde_json::from_str::<Value>(&raw) else {
        return HashMap::new();
    };
    let Some(obj) = json.as_object() else {
        return HashMap::new();
    };
    obj.iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
        .collect()
}

/// Returns the target base URL from `HARNESS_BROWSER_BASE_URL` env,
/// defaulting to `http://localhost:4200`.
#[must_use]
pub fn base_url() -> String {
    std::env::var("HARNESS_BROWSER_BASE_URL").unwrap_or_else(|_| "http://localhost:4200".to_owned())
}

/// Returns the LLM server base URL from `HARNESS_LLM_TEST_URL` env,
/// defaulting to `http://localhost:8080`.
#[must_use]
pub fn llm_base_url() -> String {
    std::env::var("HARNESS_LLM_TEST_URL")
        .unwrap_or_else(|_| "http://localhost:8080".to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// Returns the LLM model name from `HARNESS_LLM_TEST_MODEL` env,
/// defaulting to `deepseek`.
#[must_use]
pub fn llm_model() -> String {
    std::env::var("HARNESS_LLM_TEST_MODEL").unwrap_or_else(|_| "deepseek".to_owned())
}

/// Returns whether to run the browser in headless mode from
/// `HARNESS_BROWSER_HEADLESS` env, defaulting to `true`.
#[must_use]
pub fn browser_headless() -> bool {
    std::env::var("HARNESS_BROWSER_HEADLESS")
        .map_or(true, |v| v != "0" && v.to_lowercase() != "false")
}

/// Builds a `reqwest::Client` with the given timeout.
#[must_use]
pub fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("build reqwest client")
}

/// Sends a chat completion request to the LLM.
///
/// Returns `Some(content)` on success, `None` on any error.
///
/// Prefer `llm_chat_with_usage` if you need token counting.
#[must_use]
pub async fn llm_chat(llm: &LlmConfig, system: &str, user: &str) -> Option<String> {
    llm_chat_with_usage(llm, system, user)
        .await
        .map(|r| r.content)
}

/// Sends a chat completion request to the LLM and returns both the content
/// and token usage from the API response.
#[must_use]
pub async fn llm_chat_with_usage(llm: &LlmConfig, system: &str, user: &str) -> Option<LlmResponse> {
    let client = http_client(llm.timeout);
    let mut payload = serde_json::json!({
        "model": llm.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 4096,
        "temperature": llm.temperature
    });
    if let Some(think) = llm.thinking {
        if think {
            payload["thinking"] = serde_json::json!({"type": "enabled"});
        } else {
            payload["thinking"] = serde_json::json!({"type": "disabled"});
        }
    }
    // Merge provider-specific parameters into the request body.
    if !llm.model_params.is_empty() {
        if let Value::Object(ref mut map) = payload {
            for (key, val) in &llm.model_params {
                map.insert(key.clone(), val.clone());
            }
        }
    }
    let mut req = client
        .post(format!("{}/v1/chat/completions", llm.url))
        .header("Content-Type", "application/json");

    if let Some(ref key) = llm.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    for (name, value) in &llm.headers {
        req = req.header(name.as_str(), value.as_str());
    }

    let resp = req.json(&payload).send().await.ok()?;
    let json: Value = resp.json().await.ok()?;
    let usage = costs::extract_usage(&json);
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)?;

    Some(LlmResponse { content, usage })
}

/// JavaScript to extract interactive elements from the current page.
/// Returns a JSON array of objects with tag, selector, and label.
pub const DOM_EXTRACT_JS: &str = r#"
(() => {
  const interactive = 'a, button, input, textarea, select, [role="button"], [onclick], [tabindex], [data-testid], [aria-label]';
  const els = document.querySelectorAll(interactive);
  const info = [];
  const seen = new Set();
  els.forEach((el, i) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const tag = el.tagName.toLowerCase();
    let selector = '';
    if (el.id) selector = '#' + CSS.escape(el.id);
    else if (el.getAttribute('data-testid')) selector = '[data-testid="' + el.getAttribute('data-testid') + '"]';
    else if (el.name) selector = '[name="' + CSS.escape(el.name) + '"]';
    else if (el.className && typeof el.className === 'string') {
      const cls = el.className.trim().split(/\\s+/)[0];
      if (cls) selector = tag + '.' + CSS.escape(cls);
    }
    if (!selector) selector = tag;
    if (seen.has(selector)) return;
    seen.add(selector);

    let label = '';
    const aria = el.getAttribute('aria-label');
    if (aria) {
      label = aria;
    } else if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      label = el.placeholder || el.name || el.getAttribute('aria-label') || '';
      if (el.type && !label) label = el.type;
    } else {
      label = (el.textContent || '').trim().substring(0, 80);
    }

    info.push(i + ': ' + selector + ' [' + tag + '] "' + label + '"');
  });
  return JSON.stringify(info);
})()
"#;

/// Truncates a string to the given maximum length, appending a marker
/// if truncation occurred.
#[must_use]
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        format!("{}...<truncated>", &s[..max_len])
    }
}
