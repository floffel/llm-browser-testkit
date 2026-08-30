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
//! | `agent`       | `agent`, `task` | —                                        |
//! | `mcp`         | `server`, `tool`| `args`                                   |
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
    #[serde(default)]
    pub base_url: Option<String>,
    /// LLM server base URL (deprecated; prefer `[config.endpoints]`).
    #[serde(default)]
    pub llm_url: Option<String>,
    /// LLM model name (deprecated; prefer `[config.endpoints]`).
    #[serde(default)]
    pub llm_model: Option<String>,
    /// LLM API key (Bearer token).
    #[serde(default)]
    pub llm_api_key: Option<String>,
    /// Custom HTTP headers as JSON key-value pairs.
    #[serde(default, deserialize_with = "deserialize_headers")]
    pub llm_headers: HashMap<String, String>,
    /// Run browser in headless mode.
    #[serde(default)]
    pub browser_headless: Option<bool>,
    /// HTTP / browser action timeout in seconds.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Browser viewport width.
    #[serde(default)]
    pub viewport_width: Option<u32>,
    /// Browser viewport height.
    #[serde(default)]
    pub viewport_height: Option<u32>,
    /// Default URL every test auto-navigates to before running its steps.
    #[serde(default)]
    pub start_url: Option<String>,
    /// Whether to auto-navigate to `start_url` before test steps.
    ///
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
    /// Named endpoints (LLM, MCP, A2A agents) with pricing.
    #[serde(default)]
    pub endpoints: HashMap<String, EndpointConfig>,
    /// Global and per-test budgets for cost/token/call limits.
    #[serde(default)]
    pub budgets: BudgetsConfig,
    /// MCP server exposure configuration.
    #[serde(default)]
    pub mcp_server: Option<McpServerConfig>,
    /// A2A agent server exposure configuration.
    #[serde(default)]
    pub a2a_server: Option<A2aServerConfig>,
    /// Whether to continue running the remaining steps of a test after a
    /// step fails. Default `false` = fail fast: the first failed step ends
    /// the test and the rest are reported as skipped. Set to `true` to run
    /// every step (more diagnostics, more LLM cost on broken apps).
    #[serde(default)]
    pub continue_on_failure: bool,
    /// Longest edge (px) of screenshots attached to `screenshot = true`
    /// assert steps, before they are JPEG-encoded and sent to the vision
    /// endpoint. Downscaling keeps vision token cost/quality sane.
    /// Default: 1400.
    #[serde(default)]
    pub screenshot_max_dimension: Option<u32>,
    /// Directory for failure artifacts (screenshots, page snapshots).
    /// Defaults to `artifacts`.
    #[serde(default)]
    pub artifacts_dir: Option<String>,
    /// Optional viewport matrix: when set, every test in the scenario is
    /// expanded into one variant per viewport (e.g. mobile/tablet/desktop).
    /// Each variant overrides the test's viewport and gets a ` — <name>`
    /// suffix on the test name. Per-test budgets apply per variant.
    #[serde(default)]
    pub viewport_matrix: Option<ViewportMatrix>,
}

/// A list of named viewports a scenario is expanded across.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ViewportMatrix {
    /// The viewport variants (`{name, width, height}`).
    #[serde(default)]
    pub viewports: Vec<ViewportDef>,
}

/// One named viewport size in a matrix.
#[derive(Debug, Deserialize, Clone)]
pub struct ViewportDef {
    /// Human-readable variant name (appended to test names, e.g.
    /// `— mobile`).
    pub name: String,
    /// Browser viewport width in pixels.
    pub width: u32,
    /// Browser viewport height in pixels.
    pub height: u32,
}

/// A named endpoint definition with pricing.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct EndpointConfig {
    /// Endpoint type: `llm`, `mcp`, or `a2a`.
    #[serde(rename = "type")]
    pub endpoint_type: EndpointType,
    /// Base URL for the endpoint.
    #[serde(default)]
    pub url: Option<String>,
    /// Model name (LLM endpoints only).
    #[serde(default)]
    pub model: Option<String>,
    /// API key / bearer token.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Custom HTTP headers as JSON key-value pairs.
    #[serde(default, deserialize_with = "deserialize_headers")]
    pub headers: HashMap<String, String>,
    /// Pricing configuration.
    #[serde(default)]
    pub pricing: Option<PricingConfig>,
    /// Task types this endpoint serves by default
    /// (e.g. `["targeting", "assertion"]`).
    #[serde(default)]
    pub default_for: Vec<String>,
    /// Command to launch an MCP server subprocess (stdio transport).
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments for the MCP server command.
    #[serde(default)]
    pub args: Vec<String>,
    /// Whether this LLM endpoint accepts image parts (vision) in addition
    /// to text. `assert` steps with `screenshot = true` require a vision
    /// endpoint.
    #[serde(default)]
    pub vision: bool,
}

/// Type discriminator for endpoint configuration.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EndpointType {
    /// OpenAI-compatible LLM API.
    #[default]
    Llm,
    /// Model Context Protocol server.
    Mcp,
    /// Agent-to-Agent protocol agent.
    A2a,
}

/// Pricing configuration for an endpoint.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PricingConfig {
    /// Cost per 1M input tokens (USD).
    #[serde(default)]
    pub input_per_1m_tokens: f64,
    /// Cost per 1M output tokens (USD).
    #[serde(default)]
    pub output_per_1m_tokens: f64,
    /// Flat cost per call (USD), used for MCP/agent endpoints.
    #[serde(default)]
    pub per_call: f64,
}

/// Budget limits for test execution.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct BudgetsConfig {
    /// Global budget across all tests in the scenario.
    #[serde(default)]
    pub global: Option<BudgetDef>,
    /// Default per-test budget. Individual tests can override.
    #[serde(default)]
    pub per_test_default: Option<BudgetDef>,
}

/// A budget definition with limits and enforcement mode.
#[derive(Debug, Deserialize, Clone)]
pub struct BudgetDef {
    /// Maximum cost in USD.
    #[serde(default)]
    pub max_cost: Option<f64>,
    /// Maximum total tokens (input + output).
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Maximum number of calls (LLM, MCP, agent combined).
    #[serde(default)]
    pub max_calls: Option<u64>,
    /// Enforcement mode: `hard` (abort) or `soft` (warn and continue).
    #[serde(default)]
    pub enforcement: Option<BudgetEnforcement>,
}

/// Budget enforcement strategy.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BudgetEnforcement {
    /// Abort the test or run when budget is exceeded.
    Hard,
    /// Log a warning but continue execution.
    Soft,
}

/// MCP server exposure configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct McpServerConfig {
    /// Whether to enable the embedded MCP server.
    #[serde(default)]
    pub enabled: bool,
    /// Port to listen on.
    #[serde(default = "default_mcp_port")]
    pub port: u16,
}

const fn default_mcp_port() -> u16 {
    3000
}

/// A2A agent server exposure configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct A2aServerConfig {
    /// Whether to enable the embedded A2A agent server.
    #[serde(default)]
    pub enabled: bool,
    /// Port to listen on.
    #[serde(default = "default_a2a_port")]
    pub port: u16,
}

const fn default_a2a_port() -> u16 {
    3100
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
    /// Predefined assertion preset name
    /// (e.g. `no_error_on_page`, `text_visible`).
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
    /// Agent endpoint to call for this assertion.
    #[serde(default)]
    pub agent: Option<String>,
    /// Agent task template for this assertion.
    #[serde(default)]
    pub task_template: Option<String>,
}

/// A group of steps that form a single test scenario.
#[derive(Debug, Deserialize, Clone)]
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
    /// Override the global viewport width for this test (applied via CDP
    /// `Emulation.setDeviceMetricsOverride` before the test runs).
    #[serde(default)]
    pub viewport_width: Option<u32>,
    /// Override the global viewport height for this test.
    #[serde(default)]
    pub viewport_height: Option<u32>,
    /// Per-test budget override.
    #[serde(default)]
    pub budget: Option<BudgetDef>,
    /// Endpoint to use for all steps in this test (can be overridden
    /// per-step).
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Ordered steps to execute.
    #[serde(default)]
    pub steps: Vec<TestStep>,
}

/// A single step in a test. The `kind` field determines which variant is
/// deserialized and which field constraints apply.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "kind")]
pub enum TestStep {
    /// Navigate the browser to a URL.
    #[serde(rename = "navigate")]
    Navigate {
        /// URL to navigate to (absolute, or relative to the test's
        /// base URL).
        url: String,
        /// Milliseconds to wait after navigation completes.
        #[serde(default)]
        wait_after_ms: Option<u64>,
    },

    /// Click an element described in natural language.
    #[serde(rename = "click")]
    Click {
        /// Natural language description of the element. The LLM resolves
        /// this to a CSS selector at runtime.
        target: String,
        /// Explicit CSS selector override (bypasses LLM resolution).
        #[serde(default)]
        selector: Option<String>,
        /// Milliseconds to wait after the click.
        #[serde(default)]
        wait_after_ms: Option<u64>,
        /// Endpoint to use for LLM element targeting.
        #[serde(default)]
        endpoint: Option<String>,
        /// Idempotent: when the target element is absent the step is
        /// reported skipped instead of failed (the action was already
        /// done / not applicable).
        #[serde(default)]
        idempotent: bool,
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
        /// Endpoint to use for LLM element targeting.
        #[serde(default)]
        endpoint: Option<String>,
        /// Idempotent: when the target element is absent the step is
        /// reported skipped instead of failed (the action was already
        /// done / not applicable).
        #[serde(default)]
        idempotent: bool,
    },

    /// Wait for an element to appear on the page.
    #[serde(rename = "wait")]
    Wait {
        /// Natural language description of the element to wait for.
        target: String,
        /// Explicit CSS selector override (bypasses LLM resolution).
        #[serde(default)]
        selector: Option<String>,
        /// Wait until the page's visible text contains this substring
        /// (alternative to `selector`; either or both may be set — both are
        /// required to hold when both are set).
        #[serde(default)]
        text: Option<String>,
        /// Maximum milliseconds to wait (default: 10000).
        #[serde(default)]
        timeout_ms: Option<u64>,
        /// Endpoint to use for LLM element targeting.
        #[serde(default)]
        endpoint: Option<String>,
        /// Idempotent: when the condition never becomes true within the
        /// timeout the step is reported skipped instead of failed (the
        /// condition was not applicable, e.g. already-authenticated
        /// pages in a viewport matrix).
        #[serde(default)]
        idempotent: bool,
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
        /// Endpoint to use for this assertion's LLM call.
        #[serde(default)]
        endpoint: Option<String>,
        /// Attach a screenshot of the current viewport to the assertion so
        /// the LLM can evaluate visuals (overlaps, clipping, layout).
        /// Requires the resolved endpoint to declare `vision = true`.
        #[serde(default)]
        screenshot: bool,
    },

    /// Take a screenshot of the current page.
    #[serde(rename = "screenshot")]
    Screenshot {
        /// File path to save the screenshot (default: `screenshot.png`).
        #[serde(default)]
        path: Option<String>,
    },

    /// Call an A2A agent with a task.
    #[serde(rename = "agent")]
    Agent {
        /// Name of the agent endpoint to call.
        agent: String,
        /// Task description / prompt for the agent.
        task: String,
        /// Optional definition name with a task template.
        #[serde(default)]
        definition: Option<String>,
    },

    /// Call an MCP server tool.
    #[serde(rename = "mcp")]
    Mcp {
        /// Name of the MCP server endpoint.
        server: String,
        /// Tool name to invoke on the server.
        tool: String,
        /// Tool arguments as JSON.
        #[serde(default)]
        args: Option<serde_json::Value>,
    },
}
