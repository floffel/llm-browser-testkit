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

        /// Base URL for relative navigation
        /// (default: `HARNESS_BROWSER_BASE_URL` or localhost:4200).
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

        /// Custom HTTP header `Name:Value`
        /// (repeatable, e.g. `--llm-header "X-Org:acme"`).
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

        /// Global max cost in USD across all tests. Exceeding this aborts.
        #[arg(long)]
        max_cost: Option<f64>,

        /// Global max tokens across all tests. Exceeding this aborts.
        #[arg(long)]
        max_tokens: Option<u64>,

        /// Budget enforcement mode: `hard` (abort) or `soft` (warn).
        #[arg(long)]
        budget_enforcement: Option<String>,

        /// Port for the A2A agent server (enables a2a-server mode).
        #[arg(long, env = "A2A_SERVER_PORT")]
        agent_port: Option<u16>,

        /// Directory for failure artifacts (screenshots).
        /// (default: `HARNESS_ARTIFACTS_DIR` or `artifacts`).
        #[arg(long, env = "HARNESS_ARTIFACTS_DIR")]
        artifacts_dir: Option<String>,

        /// Continue running remaining steps after a step failure
        /// (default: fail fast — the first failed step ends the test).
        #[arg(long)]
        continue_on_failure: bool,
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
#[allow(clippy::too_many_lines)]
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
            max_cost,
            max_tokens,
            budget_enforcement,
            agent_port,
            artifacts_dir,
            continue_on_failure,
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
            if config.start_url.is_none() {
                config.start_url = Some(start_url);
            }

            // CLI budget overrides
            let enforce = budget_enforcement
                .as_deref()
                .map(|e| match e.to_lowercase().as_str() {
                    "soft" => llm_browser_testkit::scenario::BudgetEnforcement::Soft,
                    _ => llm_browser_testkit::scenario::BudgetEnforcement::Hard,
                });
            if max_cost.is_some() || max_tokens.is_some() || enforce.is_some() {
                let global =
                    config
                        .budgets
                        .global
                        .get_or_insert(llm_browser_testkit::scenario::BudgetDef {
                            max_cost: None,
                            max_tokens: None,
                            max_calls: None,
                            enforcement: None,
                        });
                if let Some(mc) = max_cost {
                    global.max_cost = Some(mc);
                }
                if let Some(mt) = max_tokens {
                    global.max_tokens = Some(mt);
                }
                if let Some(e) = enforce {
                    global.enforcement = Some(e);
                }
            }

            // CLI A2A server override
            if let Some(port) = agent_port {
                config.a2a_server = Some(llm_browser_testkit::scenario::A2aServerConfig {
                    enabled: true,
                    port,
                });
            }

            // CLI failure-behavior overrides (only when the flag was passed,
            // so per-scenario [config] values keep their precedence).
            if let Some(dir) = artifacts_dir {
                config.artifacts_dir = Some(dir);
            }
            if continue_on_failure {
                config.continue_on_failure = true;
            }
            let artifacts_dir = config
                .artifacts_dir
                .clone()
                .unwrap_or_else(|| "artifacts".to_owned());
            eprintln!(
                "Artifacts: {}  |  Continue on failure: {}",
                artifacts_dir,
                if config.continue_on_failure {
                    "yes"
                } else {
                    "no (fail fast)"
                }
            );

            eprintln!("Base URL: {}", config.base_url.as_deref().unwrap_or("—"));
            eprintln!("Endpoints: {} configured", config.endpoints.len());
            if config.endpoints.is_empty() {
                eprintln!(
                    "  (using default LLM: {} @ {})",
                    config.llm_model.as_deref().unwrap_or("—"),
                    config.llm_url.as_deref().unwrap_or("—"),
                );
            } else {
                for (name, ep) in &config.endpoints {
                    eprintln!(
                        "  {name}: {type:?} @ {url}",
                        type = ep.endpoint_type,
                        url = ep.url.as_deref().unwrap_or("(subprocess)")
                    );
                }
            }
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
            if let Some(ref global_budget) = config.budgets.global {
                if let Some(cost) = global_budget.max_cost {
                    eprintln!("Budget (global): max ${cost:.2}");
                }
                if let Some(tokens) = global_budget.max_tokens {
                    eprintln!("Budget (global): max {tokens} tokens");
                }
            }
            if let Some(ref per_test) = config.budgets.per_test_default {
                if let Some(cost) = per_test.max_cost {
                    eprintln!("Budget (per-test default): max ${cost:.2}");
                }
                if let Some(tokens) = per_test.max_tokens {
                    eprintln!("Budget (per-test default): max {tokens} tokens");
                }
            }

            let definitions = std::mem::take(&mut scenario_def.definitions);

            // Viewport matrix expansion: when [config.viewport_matrix] is
            // set, duplicate every test once per named viewport. Each
            // variant overrides the test's viewport and gets a " — <name>"
            // suffix so the report shows exactly which size ran.
            let mut expanded_tests = std::mem::take(&mut scenario_def.test);
            if let Some(matrix) = &config.viewport_matrix {
                if !matrix.viewports.is_empty() {
                    let mut expanded: Vec<llm_browser_testkit::scenario::TestGroup> = Vec::new();
                    for test in expanded_tests {
                        for vp in &matrix.viewports {
                            let mut variant = test.clone();
                            variant.name = format!("{} — {}", test.name, vp.name);
                            variant.viewport_width = Some(vp.width);
                            variant.viewport_height = Some(vp.height);
                            expanded.push(variant);
                        }
                    }
                    expanded_tests = expanded;
                    eprintln!(
                        "Viewport matrix: {} variants per test ({})",
                        matrix.viewports.len(),
                        matrix
                            .viewports
                            .iter()
                            .map(|v| format!("{}={}x{}", v.name, v.width, v.height))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                }
            }

            let runner = llm_browser_testkit::runner::ScenarioRunner::new(config, definitions);

            let report = runner.run(&expanded_tests)?;

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

            // Print cost report
            llm_browser_testkit::reporting::print_report(
                &runner.usage_tracker().per_test_snapshots(),
                &runner.usage_tracker().global_snapshot(),
            );

            if report.failed > 0 {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn test_cli_run_subcommand_exists() {
        let cmd = super::Cli::command();
        let matches = cmd.try_get_matches_from(["llm-browser-testkit", "run", "scenario.toml"]);
        assert!(matches.is_ok());
    }

    #[test]
    fn test_cli_run_with_all_flags() {
        let cmd = super::Cli::command();
        let matches = cmd.try_get_matches_from([
            "llm-browser-testkit",
            "run",
            "scenario.toml",
            "--llm-url",
            "https://api.example.com",
            "--llm-model",
            "gpt-4o",
            "--llm-api-key",
            "sk-test",
            "--llm-header",
            "X-Org:acme",
            "--model-param",
            "effort=high",
            "--base-url",
            "https://myapp.com",
            "--headless",
            "--timeout",
            "30",
            "--viewport-width",
            "1920",
            "--viewport-height",
            "1080",
            "--start-url",
            "/login",
            "--max-cost",
            "5.0",
            "--max-tokens",
            "500000",
            "--budget-enforcement",
            "soft",
            "--agent-port",
            "3100",
        ]);
        assert!(matches.is_ok());
    }

    #[test]
    fn test_cli_run_minimal() {
        let cmd = super::Cli::command();
        let matches = cmd.try_get_matches_from(["llm-browser-testkit", "run", "test.toml"]);
        assert!(matches.is_ok());
    }
}
