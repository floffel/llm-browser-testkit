//! Scenario types for human-readable browser test case definitions.
//!
//! Scenarios are written in [TOML](https://toml.io) and describe groups of
//! browser interaction tests with reusable assertion definitions and
//! configurable test-level overrides.
//!
//! # Structure
//!
//! ```toml
//! [config]                         # Global defaults
//! start_url = "/dashboard"
//!
//! [[definitions]]                  # Reusable assertion definitions
//! name = "no_errors"
//! preset = "no_error_on_page"
//!
//! [[test]]                         # Test group
//! name = "Dashboard Smoke"
//! start_url = "/dashboard"         # Override global start_url (optional)
//!
//! [[test.steps]]                   # Ordered steps — the `kind` field
//! kind = "navigate"                # determines which other fields apply
//! url = "/dashboard"
//!
//! [[test.steps]]
//! kind = "click"
//! target = "the Login button"      # Natural language — LLM resolves to selector
//!
//! [[test.steps]]
//! kind = "assert"
//! definition = "no_errors"
//! ```
//!
//! ## Step Kinds
//!
//! | `kind`        | Required fields  | Optional fields                          |
//! |---------------|-----------------|------------------------------------------|
//! | `navigate`    | `url`           | `wait_after_ms`                          |
//! | `click`       | `target`        | `selector`, `wait_after_ms`              |
//! | `type`        | `target`, `text`| `selector`, `wait_after_ms`              |
//! | `wait`        | `target`        | `selector`, `timeout_ms`                 |
//! | `assert`      | *one of below*  | —                                        |
//! | `screenshot`  | —               | `path`                                   |
//!
//! Assert steps require one of: `definition` (references a named
//! `[[definitions]]` entry), `preset` (built-in preset name), or `prompt`
//! (custom LLM evaluation prompt).

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

/// Top-level scenario file, deserialized from TOML.
#[derive(Debug, Deserialize)]
pub struct Scenario {
    /// Global configuration (overridable per test).
    #[serde(default)]
    pub config: ScenarioConfig,

    /// Reusable assertion definitions referenced by name in `assert` steps.
    #[serde(default)]
    pub definitions: Vec<AssertDefinition>,

    /// Ordered test groups to execute.
    #[serde(default)]
    pub test: Vec<TestGroup>,
}

/// Global scenario configuration with per-test overridable fields.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct ScenarioConfig {
    /// Base URL for relative navigation.
    pub base_url: Option<String>,
    /// LLM server base URL.
    pub llm_url: Option<String>,
    /// LLM model name.
    pub llm_model: Option<String>,
    /// LLM API key (Bearer token).
    #[serde(default)]
    pub llm_api_key: Option<String>,
    /// Custom HTTP headers as JSON key-value pairs.
    #[serde(default, deserialize_with = "deserialize_headers")]
    pub llm_headers: HashMap<String, String>,
    /// Run browser in headless mode.
    pub browser_headless: Option<bool>,
    /// HTTP / browser action timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Browser viewport width.
    pub viewport_width: Option<u32>,
    /// Browser viewport height.
    pub viewport_height: Option<u32>,
    /// Default URL every test auto-navigates to before running its steps.
    pub start_url: Option<String>,
    /// Whether to auto-navigate to `start_url` before test steps.
    /// Disable when a test starts with click-based navigation.
    #[serde(default = "default_auto_navigate")]
    pub auto_navigate: bool,
    /// LLM temperature (0.0–1.0). Lower = more deterministic.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Enable thinking/reasoning tokens. `None` means the provider default
    /// is used (no `thinking` key is sent). Set to `true`/`false` to
    /// explicitly enable or disable.
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Provider-specific model parameters merged into the chat completion
    /// request body (e.g. `effort = "high"` for Anthropic).
    #[serde(default, deserialize_with = "deserialize_model_params")]
    pub model_params: HashMap<String, Value>,
}

fn deserialize_headers<'de, D>(deserializer: D) -> Result<HashMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    let Some(json) = raw else {
        return Ok(HashMap::new());
    };
    let serde_json::Value::Object(obj) = json else {
        return Ok(HashMap::new());
    };
    Ok(obj
        .into_iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_owned())))
        .collect())
}

const fn default_auto_navigate() -> bool {
    true
}

const fn default_temperature() -> f64 {
    0.0
}

fn deserialize_model_params<'de, D>(deserializer: D) -> Result<HashMap<String, Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Map(HashMap<String, Value>),
        Table(HashMap<String, Value>),
    }
    let raw: Option<Raw> = Option::deserialize(deserializer)?;
    Ok(match raw {
        Some(Raw::Map(m) | Raw::Table(m)) => m,
        None => HashMap::new(),
    })
}

/// Reusable assertion definition referenced by name from `assert` steps.
///
/// Definitions can either reference a built-in preset via `preset`, supply a
/// custom LLM `prompt`, or define a **custom preset** by providing both
/// `system` and `user_template`. Custom presets support the same template
/// variables as built-in presets: `{url}`, `{title}`, `{content}`,
/// `{expected_text}`, `{description}`.
#[derive(Debug, Deserialize, Clone)]
pub struct AssertDefinition {
    /// Unique name used to reference this definition from steps.
    pub name: String,
    /// Predefined assertion preset name (e.g. `no_error_on_page`, `text_visible`).
    #[serde(default)]
    pub preset: Option<String>,
    /// Custom LLM prompt for assertion evaluation.
    #[serde(default)]
    pub prompt: Option<String>,
    /// System prompt for a custom preset.
    #[serde(default)]
    pub system: Option<String>,
    /// User template (with `{placeholders}`) for a custom preset.
    #[serde(default)]
    pub user_template: Option<String>,
    /// Text that the `text_visible` preset checks for, or the
    /// `{expected_text}` placeholder value for custom presets.
    #[serde(default)]
    pub assert_text: Option<String>,
}

/// A group of steps that form a single test scenario.
#[derive(Debug, Deserialize)]
pub struct TestGroup {
    /// Human-readable test name.
    pub name: String,
    /// Override the global `start_url` for this test.
    #[serde(default)]
    pub start_url: Option<String>,
    /// Override the global `auto_navigate` for this test.
    #[serde(default)]
    pub auto_navigate: Option<bool>,
    /// Override the global `base_url` for this test.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Override the global `timeout_secs` for this test.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Override the global `browser_headless` for this test.
    #[serde(default)]
    pub browser_headless: Option<bool>,
    /// Ordered steps to execute.
    #[serde(default)]
    pub steps: Vec<TestStep>,
}

/// A single step in a test. The `kind` field determines which variant is
/// deserialized and which field constraints apply.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum TestStep {
    /// Navigate the browser to a URL.
    #[serde(rename = "navigate")]
    Navigate {
        /// URL to navigate to (absolute, or relative to the test's base URL).
        url: String,
        /// Milliseconds to wait after navigation completes.
        #[serde(default)]
        wait_after_ms: Option<u64>,
    },

    /// Click an element described in natural language.
    #[serde(rename = "click")]
    Click {
        /// Natural language description of the element. The LLM resolves this
        /// to a CSS selector at runtime.
        target: String,
        /// Explicit CSS selector override (bypasses LLM resolution).
        #[serde(default)]
        selector: Option<String>,
        /// Milliseconds to wait after the click.
        #[serde(default)]
        wait_after_ms: Option<u64>,
    },

    /// Type text into an input element.
    #[serde(rename = "type")]
    Type {
        /// Natural language description of the target input element.
        target: String,
        /// Text to type into the element.
        text: String,
        /// Explicit CSS selector override (bypasses LLM resolution).
        #[serde(default)]
        selector: Option<String>,
        /// Milliseconds to wait after typing.
        #[serde(default)]
        wait_after_ms: Option<u64>,
    },

    /// Wait for an element to appear on the page.
    #[serde(rename = "wait")]
    Wait {
        /// Natural language description of the element to wait for.
        target: String,
        /// Explicit CSS selector override (bypasses LLM resolution).
        #[serde(default)]
        selector: Option<String>,
        /// Maximum milliseconds to wait (default: 10000).
        #[serde(default)]
        timeout_ms: Option<u64>,
    },

    /// Evaluate an assertion against the current page content.
    #[serde(rename = "assert")]
    Assert {
        /// Reference to a named `[[definitions]]` entry.
        #[serde(default)]
        definition: Option<String>,
        /// Inline predefined assertion preset (e.g. `no_error_on_page`).
        #[serde(default)]
        preset: Option<String>,
        /// Inline custom LLM prompt for assertion evaluation.
        #[serde(default)]
        prompt: Option<String>,
        /// Text that the `text_visible` preset checks for.
        #[serde(default)]
        assert_text: Option<String>,
    },

    /// Take a screenshot of the current page.
    #[serde(rename = "screenshot")]
    Screenshot {
        /// File path to save the screenshot (default: `screenshot.png`).
        #[serde(default)]
        path: Option<String>,
    },
}
