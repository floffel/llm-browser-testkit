//! LLM-driven browser test framework.
//!
//! Provides reusable building blocks for browser-based test scenarios:
//! - Browser client via Chrome `DevTools` Protocol (headless)
//! - LLM client for natural language element targeting and assertions
//! - Declarative TOML scenario runner

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]
//!
//! Also includes a declarative scenario runner — define test steps in TOML
//! and execute them via `ScenarioRunner` or the `llm-browser-testkit` CLI.

/// Step-by-step scenario executor (navigate, click, type, wait, assert).
pub mod runner;
/// Declarative TOML-based test scenario types.
pub mod scenario;

use std::time::Duration;

use serde_json::Value;

/// Returns the target base URL from `HARNESS_BROWSER_BASE_URL` env,
/// defaulting to `http://localhost:4200`.
#[must_use]
pub fn base_url() -> String {
    std::env::var("HARNESS_BROWSER_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:4200".to_owned())
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
pub async fn llm_chat(
    llm_url: &str,
    model: &str,
    timeout: Duration,
    system: &str,
    user: &str,
    temperature: f64,
    thinking: bool,
) -> Option<String> {
    let client = http_client(timeout);
    let mut payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 4096,
        "temperature": temperature
    });
    if thinking {
        payload["thinking"] = serde_json::json!({"type": "enabled"});
    } else {
        payload["thinking"] = serde_json::json!({"type": "disabled"});
    }
    let resp = client
        .post(format!("{llm_url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .ok()?;
    let json: Value = resp.json().await.ok()?;
    json["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)
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