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

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            scenario,
            base_url,
            llm_url,
            llm_model,
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

            let report = runner.run(&scenario_def)?;

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
