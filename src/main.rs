//! CLI binary for `llm-browser-testkit` — runs TOML browser test scenarios
//! against a real browser with LLM-assisted element targeting and assertions.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "llm-browser-testkit",
    about = "LLM-driven browser test framework"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a TOML test scenario in a real browser.
    Run {
        /// Path to the scenario file.
        scenario: PathBuf,

        /// Base URL for relative navigation (default: `HARNESS_BROWSER_BASE_URL` or localhost:4200).
        #[arg(long, env = "HARNESS_BROWSER_BASE_URL")]
        base_url: Option<String>,

        /// LLM base URL (default: `HARNESS_LLM_TEST_URL` or localhost:8080).
        #[arg(long, env = "HARNESS_LLM_TEST_URL")]
        llm_url: Option<String>,

        /// LLM model name (default: `HARNESS_LLM_TEST_MODEL` or deepseek).
        #[arg(long, env = "HARNESS_LLM_TEST_MODEL")]
        llm_model: Option<String>,

        /// LLM API key sent as `Authorization: Bearer <key>`.
        #[arg(long, env = "HARNESS_LLM_API_KEY")]
        llm_api_key: Option<String>,

        /// Custom HTTP header `Name:Value` (repeatable, e.g. `--llm-header "X-Org:acme"`).
        #[arg(long = "llm-header", value_parser = parse_header)]
        llm_headers: Vec<(String, String)>,

        /// Model parameter `key=value` merged into the chat completion body
        /// (repeatable, e.g. `--model-param effort=high`).
        #[arg(long = "model-param", value_parser = parse_model_param)]
        model_params: Vec<(String, Value)>,

        /// Run browser in headless mode (default: true).
        #[arg(long, default_value = "true")]
        headless: bool,

        /// HTTP / browser action timeout in seconds (default: 60).
        #[arg(long, default_value = "60")]
        timeout: u64,

        /// Browser viewport width (default: 1280).
        #[arg(long, default_value = "1280")]
        viewport_width: u32,

        /// Browser viewport height (default: 720).
        #[arg(long, default_value = "720")]
        viewport_height: u32,

        /// Default start URL for test auto-navigation (default: /dashboard).
        #[arg(long, default_value = "/dashboard")]
        start_url: String,
    },
}

/// Parses "Name:Value" strings from `--llm-header`.
fn parse_header(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once(':')
        .ok_or_else(|| format!("header must be 'Name:Value', got '{s}'"))?;
    Ok((k.trim().to_owned(), v.trim().to_owned()))
}

/// Parses "key=value" strings from `--model-param`.
/// JSON values (quoted strings, numbers, booleans) are parsed as-is;
/// bare words become JSON strings.
fn parse_model_param(s: &str) -> Result<(String, Value), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("model param must be 'key=value', got '{s}'"))?;
    let key = k.trim().to_owned();
    let val_str = v.trim();
    let val = serde_json::from_str::<Value>(val_str)
        .unwrap_or_else(|_| Value::String(val_str.to_owned()));
    Ok((key, val))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            scenario,
            base_url,
            llm_url,
            llm_model,
            llm_api_key,
            llm_headers,
            model_params,
            headless,
            timeout,
            viewport_width,
            viewport_height,
            start_url,
        } => {
            let toml_content = std::fs::read_to_string(&scenario)
                .with_context(|| format!("reading {}", scenario.display()))?;
            let mut scenario_def: llm_browser_testkit::scenario::Scenario =
                toml::from_str(&toml_content).with_context(|| "parsing scenario TOML")?;

            // Build effective global config: CLI args override scenario [config]
            let mut config = scenario_def.config.clone();

            config.base_url = base_url.or(config.base_url);
            config.llm_url = llm_url.or(config.llm_url);
            config.llm_model = llm_model.or(config.llm_model);
            config.llm_api_key = llm_api_key.or(config.llm_api_key);
            if !llm_headers.is_empty() {
                let mut headers = config.llm_headers;
                for (k, v) in llm_headers {
                    headers.insert(k, v);
                }
                config.llm_headers = headers;
            }
            if !model_params.is_empty() {
                let mut params = config.model_params;
                for (k, v) in model_params {
                    params.insert(k, v);
                }
                config.model_params = params;
            }
            config.browser_headless = Some(headless);
            config.timeout_secs = Some(timeout.max(config.timeout_secs.unwrap_or(60)));
            config.viewport_width = Some(viewport_width.max(config.viewport_width.unwrap_or(1280)));
            config.viewport_height =
                Some(viewport_height.max(config.viewport_height.unwrap_or(720)));
            // Only set start_url from CLI if scenario config didn't set one
            if config.start_url.is_none() {
                config.start_url = Some(start_url);
            }

            eprintln!("Base URL: {}", config.base_url.as_deref().unwrap_or("—"));
            eprintln!(
                "LLM: {} ({})",
                config.llm_url.as_deref().unwrap_or("—"),
                config.llm_model.as_deref().unwrap_or("—"),
            );
            eprintln!(
                "Browser: {} ({}x{})",
                if config.browser_headless.unwrap_or(true) {
                    "headless"
                } else {
                    "visible"
                },
                config.viewport_width.unwrap(),
                config.viewport_height.unwrap(),
            );
            eprintln!(
                "Start URL: {}",
                config.start_url.as_deref().unwrap_or("/dashboard"),
            );
            eprintln!(
                "Tests: {}  Definitions: {}",
                scenario_def.test.len(),
                scenario_def.definitions.len(),
            );

            let definitions = std::mem::take(&mut scenario_def.definitions);
            let runner = llm_browser_testkit::runner::ScenarioRunner::new(config, definitions);

            let report = runner.run(&scenario_def.test)?;

            eprintln!("\n═══════════════════════════════════════");
            eprintln!(
                "  Tests:  ✅ {passed} passed   ❌ {failed} failed",
                passed = report.tests_passed,
                failed = report.tests_failed,
            );
            eprintln!(
                "  Steps:  ✅ {passed} passed   ❌ {failed} failed   ⏭️  {skipped} skipped",
                passed = report.passed,
                failed = report.failed,
                skipped = report.skipped,
            );

            if report.failed > 0 {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
