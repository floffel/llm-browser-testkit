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
use clap::{ArgAction, Parser, Subcommand};
use llm_browser_testkit::parallel::{RunOptions, ScenarioFile};
use llm_browser_testkit::reporting::{ColorMode, Level, Reporter};
use llm_browser_testkit::scenario::{
    A2aServerConfig, BudgetDef, BudgetEnforcement, EndpointConfig, EndpointType, Scenario,
    ScenarioConfig, TestGroup,
};
use serde_json::Value;
use std::sync::Arc;

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
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Run a TOML test scenario in a real browser.
    Run {
        /// Path(s) to the scenario file(s). Pass several to run them
        /// concurrently (see `--parallel`); each file runs on its own
        /// isolated browser so cookies and session state never interfere.
        scenario: Vec<PathBuf>,

        /// Exact maximum number of scenario files to run at the same time.
        /// When omitted, concurrency auto-scales to the machine's available
        /// memory (bounded by `--parallel-min` / `--parallel-max`), learning
        /// the real per-browser footprint by trial and error and retrying
        /// out-of-memory launch failures. Each file runs on its own isolated
        /// browser; per-file `[config] concurrency_group` values keep files
        /// that touch the same shared backend state from overlapping.
        #[arg(long)]
        parallel: Option<u32>,

        /// Lower bound for auto-scaling concurrency (used only when
        /// `--parallel` is omitted). Default: 1.
        #[arg(long, default_value = "1")]
        parallel_min: u32,

        /// Upper bound for auto-scaling concurrency (used only when
        /// `--parallel` is omitted). `0` means unlimited. Default: 0.
        #[arg(long, default_value = "0")]
        parallel_max: u32,

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

        /// Fallback LLM base URL (default: `HARNESS_LLM_FALLBACK_URL`).
        /// When set, every LLM call first tries the primary endpoint
        /// (`HARNESS_LLM_TEST_URL`) with its own retry budget, then this
        /// fallback endpoint. Useful for pairing a cheap primary model with
        /// a more powerful/expensive fallback that is only billed when the
        /// primary fails. Only applies when the scenario declares no
        /// `[config.endpoints]` table — with a table, use per-endpoint
        /// `fallbacks = [...]` instead.
        #[arg(long, env = "HARNESS_LLM_FALLBACK_URL")]
        llm_fallback_url: Option<String>,

        /// Fallback LLM model name (default: `HARNESS_LLM_FALLBACK_MODEL`).
        #[arg(long, env = "HARNESS_LLM_FALLBACK_MODEL")]
        llm_fallback_model: Option<String>,

        /// Fallback LLM API key (default: `HARNESS_LLM_FALLBACK_API_KEY`).
        #[arg(long, env = "HARNESS_LLM_FALLBACK_API_KEY")]
        llm_fallback_api_key: Option<String>,

        /// Custom HTTP header `Name:Value`
        /// (repeatable, e.g. `--llm-header "X-Org:acme"`).
        #[arg(long = "llm-header", value_parser = parse_header)]
        llm_headers: Vec<(String, String)>,

        /// Additional literal values redacted from all logs and reports
        /// (repeatable; also `HARNESS_REDACT`, comma-separated).
        #[arg(long, env = "HARNESS_REDACT", value_delimiter = ',')]
        redact: Vec<String>,

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

        /// Quiet output: hide step results (`-q`), then warnings too
        /// (`-qq`). Failures and the run summary always show.
        #[arg(short = 'q', long, action = ArgAction::Count)]
        quiet: u8,

        /// Verbose output: show LLM calls and step starts (`-v`), then
        /// everything (`-vv`).
        #[arg(short = 'v', long, action = ArgAction::Count)]
        verbose: u8,

        /// Write a machine-readable `NDJSON` event log (one JSON object per
        /// test/step/LLM-call event, each with a `type` and `ts` field).
        #[arg(long)]
        log_file: Option<PathBuf>,

        /// Write a `JUnit` XML report for CI systems (Jenkins, GitLab, ...).
        #[arg(long)]
        junit: Option<PathBuf>,

        /// Write a Perfetto-format trace of test/step/LLM spans.
        #[arg(long)]
        trace: Option<PathBuf>,

        /// Colorize console output: `auto` (default, `TTY` + `NO_COLOR` aware),
        /// `always`, or `never`.
        #[arg(long, default_value = "auto")]
        color: String,
    },

    /// Print the version.
    Version,
}

/// Parses "Name:Value" strings from `--llm-header`.
fn parse_header(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once(':')
        .ok_or_else(|| format!("header must be 'Name:Value', got '{s}'"))?;
    Ok((k.trim().to_owned(), v.trim().to_owned()))
}

/// Treats `None`/empty strings (e.g. an interpolated but unset CI var) as
/// `None`, so `${{ vars.X || '' }}` never becomes a phantom endpoint.
fn nonempty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.trim().is_empty())
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

/// The CLI flags that override a scenario's `[config]`, applied to every
/// scenario file before it runs (so a batch shares the same overrides).
struct RunOverrides {
    base_url: Option<String>,
    llm_url: Option<String>,
    llm_model: Option<String>,
    llm_api_key: Option<String>,
    llm_fallback_url: Option<String>,
    llm_fallback_model: Option<String>,
    llm_fallback_api_key: Option<String>,
    llm_headers: Vec<(String, String)>,
    model_params: Vec<(String, Value)>,
    headless: bool,
    timeout: u64,
    viewport_width: u32,
    viewport_height: u32,
    start_url: String,
    max_cost: Option<f64>,
    max_tokens: Option<u64>,
    budget_enforcement: Option<String>,
    agent_port: Option<u16>,
    artifacts_dir: Option<String>,
    continue_on_failure: bool,
}

/// Builds a scenario's effective global config by applying CLI overrides on
/// top of the scenario `[config]`. CLI flags win; per-scenario values keep
/// their precedence for anything the CLI did not set.
fn apply_cli_overrides(config: ScenarioConfig, o: &RunOverrides) -> ScenarioConfig {
    let mut config = config;
    config.base_url = o.base_url.clone().or(config.base_url);
    config.llm_url = o.llm_url.clone().or(config.llm_url);
    config.llm_model = o.llm_model.clone().or(config.llm_model);
    config.llm_api_key = o.llm_api_key.clone().or(config.llm_api_key);
    // Fallback endpoint: only meaningful when the scenario declares
    // no [config.endpoints] table (with a table, per-endpoint
    // `fallbacks = [...]` is the declarative form). Synthesize a
    // two-endpoint table here so CLI/env fallback settings behave
    // exactly like the declarative chain.
    if let Some(fb_url) = nonempty(o.llm_fallback_url.clone()) {
        if config.endpoints.is_empty() {
            let mut endpoints = std::collections::HashMap::new();
            endpoints.insert(
                "default".to_owned(),
                EndpointConfig {
                    endpoint_type: EndpointType::Llm,
                    url: config.llm_url.clone(),
                    model: config.llm_model.clone(),
                    api_key: config.llm_api_key.clone(),
                    headers: config.llm_headers.clone(),
                    default_for: vec!["targeting".to_owned(), "assertion".to_owned()],
                    fallbacks: vec!["fallback".to_owned()],
                    ..Default::default()
                },
            );
            endpoints.insert(
                "fallback".to_owned(),
                EndpointConfig {
                    endpoint_type: EndpointType::Llm,
                    url: Some(fb_url),
                    model: nonempty(o.llm_fallback_model.clone()),
                    api_key: nonempty(o.llm_fallback_api_key.clone()),
                    default_for: Vec::new(),
                    ..Default::default()
                },
            );
            config.endpoints = endpoints;
        }
    }
    if !o.llm_headers.is_empty() {
        let mut headers = config.llm_headers;
        for (k, v) in o.llm_headers.clone() {
            headers.insert(k, v);
        }
        config.llm_headers = headers;
    }
    if !o.model_params.is_empty() {
        let mut params = config.model_params;
        for (k, v) in o.model_params.clone() {
            params.insert(k, v);
        }
        config.model_params = params;
    }
    config.browser_headless = Some(o.headless);
    config.timeout_secs = Some(o.timeout.max(config.timeout_secs.unwrap_or(60)));
    config.viewport_width = Some(o.viewport_width.max(config.viewport_width.unwrap_or(1280)));
    config.viewport_height = Some(o.viewport_height.max(config.viewport_height.unwrap_or(720)));
    if config.start_url.is_none() {
        config.start_url = Some(o.start_url.clone());
    }

    // CLI budget overrides
    let enforce = o
        .budget_enforcement
        .as_deref()
        .map(|e| match e.to_lowercase().as_str() {
            "soft" => BudgetEnforcement::Soft,
            _ => BudgetEnforcement::Hard,
        });
    if o.max_cost.is_some() || o.max_tokens.is_some() || enforce.is_some() {
        let global = config.budgets.global.get_or_insert(BudgetDef {
            max_cost: None,
            max_tokens: None,
            max_calls: None,
            enforcement: None,
        });
        if let Some(mc) = o.max_cost {
            global.max_cost = Some(mc);
        }
        if let Some(mt) = o.max_tokens {
            global.max_tokens = Some(mt);
        }
        if let Some(e) = enforce {
            global.enforcement = Some(e);
        }
    }

    // CLI A2A server override
    if let Some(port) = o.agent_port {
        config.a2a_server = Some(A2aServerConfig {
            enabled: true,
            port,
        });
    }

    // CLI failure-behavior overrides (only when the flag was passed,
    // so per-scenario [config] values keep their precedence).
    if let Some(dir) = o.artifacts_dir.clone() {
        config.artifacts_dir = Some(dir);
    }
    if o.continue_on_failure {
        config.continue_on_failure = true;
    }
    config
}

/// Expands `[config.viewport_matrix]` into one test variant per named
/// viewport (each gets a ` — <name>` suffix). Prints a summary line when the
/// matrix is active. Returns the (possibly expanded) test list.
#[allow(clippy::cast_possible_truncation)]
fn expand_viewport_matrix(
    config: &ScenarioConfig,
    tests: Vec<TestGroup>,
    reporter: &Reporter,
) -> Vec<TestGroup> {
    let mut expanded = tests;
    if let Some(matrix) = &config.viewport_matrix {
        if !matrix.viewports.is_empty() {
            let mut out: Vec<TestGroup> = Vec::new();
            for test in expanded {
                for vp in &matrix.viewports {
                    let mut variant = test.clone();
                    variant.name = format!("{} — {}", test.name, vp.name);
                    variant.viewport_width = Some(vp.width);
                    variant.viewport_height = Some(vp.height);
                    out.push(variant);
                }
            }
            expanded = out;
            reporter.info(format!(
                "Viewport matrix: {} variants per test ({})",
                matrix.viewports.len(),
                matrix
                    .viewports
                    .iter()
                    .map(|v| format!("{}={}x{}", v.name, v.width, v.height))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }
    expanded
}

/// Prints the per-scenario summary header (artifacts, base URL, endpoints,
/// browser, start URL, test/definition counts, budgets).
#[allow(clippy::cast_possible_truncation)]
fn print_scenario_header(
    reporter: &Reporter,
    config: &ScenarioConfig,
    tests_len: usize,
    definitions_len: usize,
) {
    let artifacts_dir = config
        .artifacts_dir
        .clone()
        .unwrap_or_else(|| "artifacts".to_owned());
    reporter.info(format!(
        "Artifacts: {}  |  Continue on failure: {}",
        artifacts_dir,
        if config.continue_on_failure {
            "yes"
        } else {
            "no (fail fast)"
        }
    ));

    reporter.info(format!(
        "Base URL: {}",
        config.base_url.as_deref().unwrap_or("-")
    ));
    reporter.info(format!("Endpoints: {} configured", config.endpoints.len()));
    if config.endpoints.is_empty() {
        reporter.info(format!(
            "  (using default LLM: {} @ {})",
            config.llm_model.as_deref().unwrap_or("-"),
            config.llm_url.as_deref().unwrap_or("-"),
        ));
    } else {
        for (name, ep) in &config.endpoints {
            reporter.info(format!(
                "  {name}: {type:?} @ {url}{fallbacks}",
                type = ep.endpoint_type,
                url = ep.url.as_deref().unwrap_or("(subprocess)"),
                fallbacks = if ep.fallbacks.is_empty() {
                    String::new()
                } else {
                    format!("  ->  fallbacks: {}", ep.fallbacks.join(", "))
                },
            ));
        }
    }
    reporter.info(format!(
        "Browser: {} ({}x{})",
        if config.browser_headless.unwrap_or(true) {
            "headless"
        } else {
            "visible"
        },
        config.viewport_width.unwrap(),
        config.viewport_height.unwrap(),
    ));
    reporter.info(format!(
        "Start URL: {}",
        config.start_url.as_deref().unwrap_or("/dashboard"),
    ));
    reporter.info(format!(
        "Tests: {tests_len}  Definitions: {definitions_len}",
    ));
    if let Some(ref global_budget) = config.budgets.global {
        if let Some(cost) = global_budget.max_cost {
            reporter.info(format!("Budget (global): max ${cost:.2}"));
        }
        if let Some(tokens) = global_budget.max_tokens {
            reporter.info(format!("Budget (global): max {tokens} tokens"));
        }
    }
    if let Some(ref per_test) = config.budgets.per_test_default {
        if let Some(cost) = per_test.max_cost {
            reporter.info(format!("Budget (per-test default): max ${cost:.2}"));
        }
        if let Some(tokens) = per_test.max_tokens {
            reporter.info(format!("Budget (per-test default): max {tokens} tokens"));
        }
    }
}

#[tokio::main(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
        }
        Command::Run {
            scenario,
            parallel,
            parallel_min,
            parallel_max,
            base_url,
            llm_url,
            llm_model,
            llm_api_key,
            llm_fallback_url,
            llm_fallback_model,
            llm_fallback_api_key,
            llm_headers,
            redact,
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
            quiet,
            verbose,
            log_file,
            junit,
            trace,
            color,
        } => {
            let level = Level::from_flags(quiet, verbose);
            let color_mode = match color.to_ascii_lowercase().as_str() {
                "always" | "force" => ColorMode::Always,
                "never" | "none" => ColorMode::Never,
                _ => ColorMode::Auto,
            };
            let github = std::env::var_os("GITHUB_ACTIONS").is_some();
            let reporter = Arc::new(Reporter::new(
                level,
                color_mode,
                log_file.as_deref(),
                junit.as_deref(),
                trace.as_deref(),
                github,
            )?);
            for s in &redact {
                reporter.add_redaction_secret(s);
            }

            let overrides = RunOverrides {
                base_url,
                llm_url,
                llm_model,
                llm_api_key,
                llm_fallback_url,
                llm_fallback_model,
                llm_fallback_api_key,
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
            };

            // Parse every scenario file, apply CLI overrides + the viewport
            // matrix, and build one runnable ScenarioFile per path. Each file
            // keeps its own config/definitions/tests.
            let mut files: Vec<ScenarioFile> = Vec::new();
            for path in scenario {
                let toml_content = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let mut scenario_def: Scenario =
                    toml::from_str(&toml_content).with_context(|| "parsing scenario TOML")?;
                let config = apply_cli_overrides(scenario_def.config.clone(), &overrides);
                let test_count = scenario_def.test.len();
                let definitions = std::mem::take(&mut scenario_def.definitions);
                let tests = expand_viewport_matrix(
                    &config,
                    std::mem::take(&mut scenario_def.test),
                    &reporter,
                );
                print_scenario_header(&reporter, &config, test_count, definitions.len());
                files.push(ScenarioFile {
                    label: path.to_string_lossy().to_string(),
                    config,
                    definitions,
                    tests,
                });
            }

            // Single file: keep the original serial path so existing
            // invocations behave and report exactly as before.
            if files.len() == 1 {
                let file = &files[0];
                let runner = llm_browser_testkit::runner::ScenarioRunner::with_reporter(
                    file.config.clone(),
                    file.definitions.clone(),
                    Arc::clone(&reporter),
                );

                let report = match runner.run(&file.tests) {
                    Ok(report) => report,
                    Err(err) => {
                        // Route through the reporter so the error text is
                        // redacted; anyhow's own `?` rendering would bypass it.
                        reporter.error(format!("{err:#}"));
                        std::process::exit(1);
                    }
                };

                reporter.finish()?;

                // Print cost report
                llm_browser_testkit::reporting::print_report(
                    &runner.usage_tracker().per_test_snapshots(),
                    &runner.usage_tracker().global_snapshot(),
                );

                if report.failed > 0 {
                    std::process::exit(1);
                }
            } else {
                // Multiple files: run the batch concurrently, each file on
                // its own isolated browser. Concurrency auto-scales to the
                // machine's memory (learning the real per-browser footprint)
                // unless `--parallel` pins it; out-of-memory launch failures
                // are guarded and retried.
                let mode = llm_browser_testkit::parallel::mode_from_cli(
                    parallel,
                    parallel_min,
                    parallel_max,
                );
                let memory = llm_browser_testkit::parallel::probe_memory_async().await;
                let run = match llm_browser_testkit::parallel::run_scenarios(
                    files,
                    RunOptions {
                        mode,
                        reporter: Arc::clone(&reporter),
                        memory,
                    },
                ) {
                    Ok(run) => run,
                    Err(err) => {
                        reporter.error(format!("{err:#}"));
                        std::process::exit(1);
                    }
                };

                reporter.finish()?;

                llm_browser_testkit::reporting::print_report(&run.per_test, &run.global);

                if run.report.failed > 0 {
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;
    use std::path::PathBuf;

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
            "--llm-fallback-url",
            "https://fallback.example.com",
            "--llm-fallback-model",
            "gpt-4o",
            "--llm-fallback-api-key",
            "sk-fallback",
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

    #[test]
    fn test_cli_run_reporting_flags() {
        let cmd = super::Cli::command();
        let matches = cmd
            .try_get_matches_from([
                "llm-browser-testkit",
                "run",
                "test.toml",
                "-v",
                "-q",
                "--log-file",
                "run.jsonl",
                "--junit",
                "report.xml",
                "--trace",
                "trace.json",
                "--color",
                "never",
            ])
            .unwrap();
        let sub = matches.subcommand_matches("run").unwrap();
        assert_eq!(sub.get_count("quiet"), 1);
        assert_eq!(sub.get_count("verbose"), 1);
        assert_eq!(
            sub.get_one::<String>("color").map(String::as_str),
            Some("never")
        );
        assert_eq!(
            sub.get_one::<PathBuf>("log_file")
                .map(|p| p.to_string_lossy().into_owned()),
            Some("run.jsonl".to_owned())
        );
        assert_eq!(
            sub.get_one::<PathBuf>("junit")
                .map(|p| p.to_string_lossy().into_owned()),
            Some("report.xml".to_owned())
        );
        assert_eq!(
            sub.get_one::<PathBuf>("trace")
                .map(|p| p.to_string_lossy().into_owned()),
            Some("trace.json".to_owned())
        );
    }

    #[test]
    fn test_cli_run_quiet_counted() {
        let cmd = super::Cli::command();
        let matches = cmd
            .try_get_matches_from(["llm-browser-testkit", "run", "t.toml", "-qq"])
            .unwrap();
        let sub = matches.subcommand_matches("run").unwrap();
        assert_eq!(sub.get_count("quiet"), 2);
    }

    #[test]
    fn test_cli_run_redact_flag_repeatable_and_comma_split() {
        let cmd = super::Cli::command();
        let matches = cmd
            .try_get_matches_from([
                "llm-browser-testkit",
                "run",
                "t.toml",
                "--redact",
                "tok-abc",
                "--redact",
                "one,two,three",
            ])
            .unwrap();
        let sub = matches.subcommand_matches("run").unwrap();
        let vals: Vec<String> = sub
            .get_many::<String>("redact")
            .unwrap()
            .map(Clone::clone)
            .collect();
        assert_eq!(vals, vec!["tok-abc", "one", "two", "three"]);
    }

    #[test]
    fn test_cli_run_accepts_multiple_scenarios_and_parallel() {
        let cmd = super::Cli::command();
        let matches = cmd
            .try_get_matches_from([
                "llm-browser-testkit",
                "run",
                "a.toml",
                "b.toml",
                "c.toml",
                "--parallel",
                "3",
            ])
            .unwrap();
        let sub = matches.subcommand_matches("run").unwrap();
        let scenarios: Vec<String> = sub
            .get_many::<std::path::PathBuf>("scenario")
            .unwrap()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        assert_eq!(scenarios, vec!["a.toml", "b.toml", "c.toml"]);
        assert_eq!(sub.get_one::<u32>("parallel").copied(), Some(3));
    }

    #[test]
    fn test_cli_run_parallel_omitted_means_auto_with_default_bounds() {
        let cmd = super::Cli::command();
        let matches = cmd
            .try_get_matches_from(["llm-browser-testkit", "run", "t.toml"])
            .unwrap();
        let sub = matches.subcommand_matches("run").unwrap();
        assert_eq!(sub.get_one::<u32>("parallel").copied(), None);
        assert_eq!(sub.get_one::<u32>("parallel_min").copied(), Some(1));
        assert_eq!(sub.get_one::<u32>("parallel_max").copied(), Some(0));
    }
}
