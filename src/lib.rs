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
/// Failure diagnostics — page-state capture and artifact writing.
pub mod diagnostics;
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
/// CSS selector sanitization for LLM-generated selectors.
pub mod selectors;
/// Vision support — screenshot capture/downscale/encode for visual asserts.
pub mod vision;

/// A2A agent server for accepting agent tasks.
#[cfg(feature = "a2a-server")]
pub mod a2a_server;

/// `#[browser_test]` macros for `cargo test` integration.
#[cfg(feature = "macros")]
pub mod macros;

use std::collections::HashMap;
use std::time::Duration;

use serde_json::{json, Value};

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
        .ok()
}

/// Sends a chat completion request to the LLM and returns both the content
/// and token usage from the API response.
///
/// Retries transient failures (network errors, HTTP 429/5xx, invalid
/// responses) with a short backoff, and returns the last underlying error
/// instead of collapsing everything into a generic "server down" message.
/// The error text includes the HTTP status and a truncated response-body
/// snippet, so a gateway that answers with an HTML error page is
/// identifiable in CI logs instead of surfacing as a bare JSON decode
/// error.
///
/// # Errors
///
/// Returns the last underlying error as a human-readable string when every
/// attempt fails (transport error, non-success HTTP status, response that is
/// not valid JSON, or a response missing `choices[0].message.content`).
pub async fn llm_chat_with_usage(
    llm: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<LlmResponse, String> {
    chat_with_retry(llm, system, user, None).await
}

/// Sends a vision-enabled chat completion request: the user message carries
/// both the text prompt and a screenshot (JPEG/PNG data URL) as an
/// OpenAI-compatible `image_url` content part.
///
/// Retries and error reporting behave like [`llm_chat_with_usage`].
///
/// # Errors
///
/// Returns the last underlying error as a human-readable string when every
/// attempt fails (transport error, non-success HTTP status, response that is
/// not valid JSON, or a response missing `choices[0].message.content`).
pub async fn llm_chat_vision_with_usage(
    llm: &LlmConfig,
    system: &str,
    user: &str,
    image_data_url: &str,
) -> Result<LlmResponse, String> {
    chat_with_retry(llm, system, user, Some(image_data_url)).await
}

/// Shared retry loop for text-only and vision chat completions.
async fn chat_with_retry(
    llm: &LlmConfig,
    system: &str,
    user: &str,
    image_data_url: Option<&str>,
) -> Result<LlmResponse, String> {
    let client = http_client(llm.timeout);
    let mut last_err = String::from("LLM call failed");
    let mut attempts: u32 = 0;

    while attempts < LLM_CALL_ATTEMPTS {
        attempts += 1;
        match llm_chat_once(&client, llm, system, user, image_data_url).await {
            Ok(resp) => return Ok(resp),
            Err(err) => {
                last_err = err.to_string();
                if attempts >= LLM_CALL_ATTEMPTS || !err.is_retryable() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500 * u64::from(attempts))).await;
            }
        }
    }

    Err(format!(
        "LLM call failed after {attempts} attempt(s) (endpoint {url}): {last_err}",
        url = llm.url
    ))
}

/// Number of attempts for a single chat completion call.
const LLM_CALL_ATTEMPTS: u32 = 3;

/// Builds the chat messages array. Text-only messages keep the plain
/// string `content` shape (maximum provider compatibility); vision calls
/// use the OpenAI-compatible content-part array with a `data:` image URL.
#[must_use]
fn build_messages(system: &str, user: &str, image_data_url: Option<&str>) -> Value {
    let user_content = image_data_url.map_or_else(
        || Value::String(user.to_owned()),
        |url| {
            json!([
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": url}}
            ])
        },
    );
    json!([
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ])
}

/// Internal error type for a single LLM request attempt, distinguishing
/// transient failures (worth retrying) from deterministic configuration
/// errors (fail immediately).
enum LlmCallError {
    /// Transport-level failure (connect, timeout, TLS, …).
    Transport { message: String },
    /// Non-success HTTP status with a body snippet.
    Http { status: u16, body: String },
    /// Success status but the body is not valid JSON.
    InvalidJson {
        status: u16,
        detail: String,
        body: String,
    },
    /// Valid JSON but missing `choices[0].message.content`.
    MissingContent { json: String },
}

impl std::fmt::Display for LlmCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport { message } => write!(f, "LLM HTTP request failed: {message}"),
            Self::Http { status, body } => {
                write!(
                    f,
                    "LLM endpoint returned HTTP {status}: {}",
                    truncate(body, 300)
                )
            }
            Self::InvalidJson {
                status,
                detail,
                body,
            } => write!(
                f,
                "LLM endpoint returned HTTP {status} with non-JSON body ({detail}): {}",
                truncate(body, 300)
            ),
            Self::MissingContent { json } => write!(
                f,
                "LLM response missing choices[0].message.content: {}",
                truncate(json, 300)
            ),
        }
    }
}

impl LlmCallError {
    /// Whether another attempt may succeed. Network errors, rate limits,
    /// server errors, and 200-with-garbage responses can be transient on
    /// flaky gateways; auth/not-found errors are deterministic.
    #[must_use]
    fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::MissingContent { .. } => true,
            Self::Http { status, .. } => {
                *status == 408 || *status == 429 || (500..600).contains(status)
            }
            Self::InvalidJson { status, .. } => {
                *status == 200 || *status == 408 || *status == 429 || (500..600).contains(status)
            }
        }
    }
}

/// Single LLM chat request attempt; returns the underlying error as text.
async fn llm_chat_once(
    client: &reqwest::Client,
    llm: &LlmConfig,
    system: &str,
    user: &str,
    image_data_url: Option<&str>,
) -> Result<LlmResponse, LlmCallError> {
    let mut payload = serde_json::json!({
        "model": llm.model,
        "messages": build_messages(system, user, image_data_url),
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

    let resp = req
        .json(&payload)
        .send()
        .await
        .map_err(|e| LlmCallError::Transport {
            message: e.to_string(),
        })?;
    let status = resp.status();
    let status_u16 = status.as_u16();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(LlmCallError::Http {
            status: status_u16,
            body,
        });
    }
    let json: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return Err(LlmCallError::InvalidJson {
                status: status_u16,
                detail: e.to_string(),
                body,
            });
        }
    };
    let usage = costs::extract_usage(&json);
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| LlmCallError::MissingContent {
            json: json.to_string(),
        })?;

    Ok(LlmResponse { content, usage })
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

#[cfg(test)]
mod tests {
    use crate::costs::extract_usage;
    use crate::truncate;
    use crate::{llm_base_url, llm_model, parse_headers_env, LlmConfig};

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let result = truncate("hello world", 5);
        assert!(result.contains("<truncated>"));
        assert!(result.starts_with("hello"));
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("abcde", 5), "abcde");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 5), "");
    }

    #[test]
    fn test_parse_headers_env_empty() {
        std::env::remove_var("HARNESS_LLM_HEADERS");
        let h = parse_headers_env();
        assert!(h.is_empty());
    }

    #[test]
    fn test_parse_headers_env_valid() {
        std::env::set_var("HARNESS_LLM_HEADERS", r#"{"X-Org":"acme","X-Version":"1"}"#);
        let h = parse_headers_env();
        assert_eq!(h.get("X-Org").map(String::as_str), Some("acme"));
        assert_eq!(h.get("X-Version").map(String::as_str), Some("1"));
        std::env::remove_var("HARNESS_LLM_HEADERS");
    }

    #[test]
    fn test_parse_headers_env_invalid_json() {
        std::env::set_var("HARNESS_LLM_HEADERS", "not-json");
        let h = parse_headers_env();
        assert!(h.is_empty());
        std::env::remove_var("HARNESS_LLM_HEADERS");
    }

    #[test]
    fn test_llm_config_from_env_defaults() {
        #[allow(clippy::float_cmp)]
        {
            let config = LlmConfig::from_env();
            assert_eq!(config.temperature, 0.0);
            assert!(config.thinking.is_none());
            assert!(config.model_params.is_empty());
        }
    }

    #[test]
    fn test_extract_usage_full() {
        let json = serde_json::json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        });
        let usage = extract_usage(&json);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 200);
        assert_eq!(usage.total_tokens, 300);
    }

    #[test]
    fn test_extract_usage_empty() {
        let json = serde_json::json!({});
        let usage = extract_usage(&json);
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_truncate_unicode() {
        // truncate uses byte-level slicing, so max_len refers to byte count
        // 'é' is 2 bytes, so index 3 captures "hé" (h=0, é=bytes 1-2)
        assert_eq!(truncate("héllo", 3), "hé...<truncated>");
        // Length 5 captures full string (5 bytes)
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_parse_headers_env_non_object() {
        std::env::set_var("HARNESS_LLM_HEADERS", "[1, 2, 3]");
        let h = parse_headers_env();
        assert!(h.is_empty());
        std::env::remove_var("HARNESS_LLM_HEADERS");
    }

    #[test]
    fn test_parse_headers_env_nested_values_filtered() {
        std::env::set_var(
            "HARNESS_LLM_HEADERS",
            r#"{"str":"val","num":42,"bool":true}"#,
        );
        let h = parse_headers_env();
        assert_eq!(h.get("str").map(String::as_str), Some("val"));
        assert!(!h.contains_key("num"));
        assert!(!h.contains_key("bool"));
        std::env::remove_var("HARNESS_LLM_HEADERS");
    }

    #[test]
    fn test_llm_config_has_default_model() {
        let config = LlmConfig::from_env();
        assert!(!config.model.is_empty());
    }

    #[test]
    fn test_llm_base_url_default() {
        std::env::remove_var("HARNESS_LLM_TEST_URL");
        let url = llm_base_url();
        assert_eq!(url, "http://localhost:8080");
    }

    #[test]
    fn test_llm_base_url_custom() {
        std::env::set_var("HARNESS_LLM_TEST_URL", "https://custom.api.com/v1");
        let url = llm_base_url();
        assert_eq!(url, "https://custom.api.com/v1");
        std::env::remove_var("HARNESS_LLM_TEST_URL");
    }

    #[test]
    fn test_llm_base_url_trailing_slash() {
        std::env::set_var("HARNESS_LLM_TEST_URL", "https://api.com/");
        let url = llm_base_url();
        assert_eq!(url, "https://api.com");
        std::env::remove_var("HARNESS_LLM_TEST_URL");
    }

    #[test]
    fn test_llm_model_default() {
        std::env::remove_var("HARNESS_LLM_TEST_MODEL");
        assert_eq!(llm_model(), "deepseek");
    }

    #[test]
    fn test_llm_model_custom() {
        std::env::set_var("HARNESS_LLM_TEST_MODEL", "gpt-4o");
        assert_eq!(llm_model(), "gpt-4o");
        std::env::remove_var("HARNESS_LLM_TEST_MODEL");
    }

    #[test]
    fn test_extract_usage_partial() {
        let json = serde_json::json!({
            "usage": {
                "prompt_tokens": 50
            }
        });
        let usage = extract_usage(&json);
        assert_eq!(usage.prompt_tokens, 50);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_browser_headless_default() {
        std::env::remove_var("HARNESS_BROWSER_HEADLESS");
        assert!(crate::browser_headless());
    }
}
