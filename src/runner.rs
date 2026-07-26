use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use headless_chrome::{Browser, LaunchOptions, Tab};

use crate::a2a::A2aClient;
use crate::budgets::{BudgetStatus, BudgetTracker};
use crate::costs::UsageTracker;
use crate::endpoints::{EndpointRegistry, TaskType};
use crate::llm_chat_with_usage;
use crate::mcp_client::McpClient;
use crate::scenario::{AssertDefinition, ScenarioConfig, TestGroup, TestStep};
use crate::truncate;
use crate::LlmConfig;
use crate::DOM_EXTRACT_JS;

/// Executes a [`Scenario`] against a real browser with optional LLM
/// assistance for element targeting and assertions.
pub struct ScenarioRunner {
    config: ScenarioConfig,
    definitions: HashMap<String, AssertDefinition>,
    llm: LlmConfig,
    timeout: Duration,
    viewport_width: u32,
    viewport_height: u32,
    endpoints: EndpointRegistry,
    usage: Arc<UsageTracker>,
    budgets: BudgetTracker,
}

/// Aggregated results from a scenario run.
#[derive(Debug, Default)]
pub struct RunReport {
    /// Number of tests that passed.
    pub tests_passed: u32,
    /// Number of tests that failed.
    pub tests_failed: u32,
    /// Number of steps that passed.
    pub passed: u32,
    /// Number of steps that failed.
    pub failed: u32,
    /// Number of steps that were skipped.
    pub skipped: u32,
    /// Per-step details.
    pub details: Vec<StepResult>,
}

/// Result of a single step execution.
#[derive(Debug)]
pub struct StepResult {
    /// The step name.
    pub name: String,
    /// Whether the step passed, failed, or was skipped.
    pub status: StepStatus,
    /// Human-readable result message.
    pub message: String,
}

/// Outcome for a single step.
#[derive(Debug, PartialEq, Eq)]
pub enum StepStatus {
    /// Step executed successfully and all assertions passed.
    Passed,
    /// Step execution or assertion failed.
    Failed,
    /// Step was skipped.
    Skipped,
}

/// Predefined assertion preset definition.
struct AssertPreset {
    name: &'static str,
    system: &'static str,
    user_template: &'static str,
}

/// Built-in assertion presets.
#[allow(clippy::literal_string_with_formatting_args)]
const ASSERTION_PRESETS: &[AssertPreset] = &[
    AssertPreset {
        name: "no_error_on_page",
        system: "You are a QA tester. Evaluate if a web page contains error messages, stack traces, exception text, HTTP error codes, 'undefined' errors, or any indication of a malfunction. Be strict — even minor rendering glitches count as errors.",
        user_template: "Check if the following page content contains ANY errors or malfunctions:\n\nURL: {url}\nTitle: {title}\n\nPage Content:\n{content}\n\nRespond with exactly \"PASS\" if there are NO errors, or \"FAIL: <reason>\" if there are errors. Only respond with PASS or FAIL.",
    },
    AssertPreset {
        name: "text_visible",
        system: "You are a QA tester. Your task is to check if specific text is visible in the page content.",
        user_template: "Check if the following text appears in the page content:\n\nTEXT TO FIND: \"{expected_text}\"\n\nURL: {url}\n\nPage Content:\n{content}\n\nRespond with exactly \"PASS\" if the text is present (even partial match is OK), or \"FAIL: text not found\" if it is not.",
    },
    AssertPreset {
        name: "element_exists",
        system: "You are a QA tester. Check if a described UI element exists on a web page.",
        user_template: "Check if the following element exists on the page:\n\nELEMENT: \"{description}\"\n\nURL: {url}\n\nPage Content:\n{content}\n\nRespond with exactly \"PASS\" if the element exists, or \"FAIL: <reason>\" if it does not.",
    },
];

impl ScenarioRunner {
    /// Creates a new runner with the given scenario configuration and
    /// assertion definitions.
    #[must_use]
    pub fn new(scenario_config: ScenarioConfig, definitions: Vec<AssertDefinition>) -> Self {
        let endpoints = EndpointRegistry::from_config(&scenario_config.endpoints);
        let budgets = BudgetTracker::from_config(&scenario_config.budgets);

        let llm = LlmConfig {
            url: scenario_config
                .llm_url
                .clone()
                .unwrap_or_else(crate::llm_base_url),
            model: scenario_config
                .llm_model
                .clone()
                .unwrap_or_else(crate::llm_model),
            api_key: scenario_config
                .llm_api_key
                .clone()
                .or_else(|| std::env::var("HARNESS_LLM_API_KEY").ok()),
            headers: if scenario_config.llm_headers.is_empty() {
                crate::parse_headers_env()
            } else {
                scenario_config.llm_headers.clone()
            },
            timeout: Duration::from_secs(scenario_config.timeout_secs.unwrap_or(60)),
            temperature: scenario_config.temperature,
            thinking: scenario_config.thinking,
            model_params: scenario_config.model_params.clone(),
        };
        let defs_map: HashMap<String, AssertDefinition> = definitions
            .into_iter()
            .map(|d| (d.name.clone(), d))
            .collect();

        Self {
            timeout: Duration::from_secs(scenario_config.timeout_secs.unwrap_or(60)),
            viewport_width: scenario_config.viewport_width.unwrap_or(1280),
            viewport_height: scenario_config.viewport_height.unwrap_or(720),
            config: scenario_config,
            definitions: defs_map,
            llm,
            endpoints,
            usage: Arc::new(UsageTracker::new()),
            budgets,
        }
    }

    /// Returns a clone of the [`UsageTracker`] for reporting.
    #[must_use]
    pub fn usage_tracker(&self) -> Arc<UsageTracker> {
        Arc::clone(&self.usage)
    }

    /// Returns a reference to the [`BudgetTracker`].
    #[must_use]
    pub const fn budget_tracker(&self) -> &BudgetTracker {
        &self.budgets
    }

    /// Executes all test groups in the scenario and returns a report.
    ///
    /// # Errors
    ///
    /// Returns an error if the browser fails to launch.
    #[allow(clippy::too_many_lines)]
    pub fn run(&self, tests: &[TestGroup]) -> anyhow::Result<RunReport> {
        let mut report = RunReport::default();

        if tests.is_empty() {
            eprintln!("No tests defined in scenario.");
            return Ok(report);
        }

        let browser_headless = self.config.browser_headless.unwrap_or(true);

        let launch_opts = LaunchOptions {
            headless: browser_headless,
            window_size: Some((self.viewport_width, self.viewport_height)),
            sandbox: false,
            ..LaunchOptions::default()
        };

        let browser = Browser::new(launch_opts).context("failed to launch browser")?;
        let tab = browser.new_tab().context("failed to open browser tab")?;
        let _ = tab.set_default_timeout(self.timeout);

        // Start MCP server if configured
        #[cfg(feature = "mcp-server")]
        if let Some(ref mcp_cfg) = self.config.mcp_server {
            if mcp_cfg.enabled {
                let port = mcp_cfg.port;
                std::thread::spawn(move || {
                    let _ = crate::mcp_server::start_mcp_server(port);
                });
            }
        }
        #[cfg(not(feature = "mcp-server"))]
        if let Some(mcp_cfg) = &self.config.mcp_server {
            if mcp_cfg.enabled {
                eprintln!("  ⚠️  MCP server configured but 'mcp-server' feature not enabled");
            }
        }

        // Start A2A agent server if configured
        #[cfg(feature = "a2a-server")]
        if let Some(ref a2a_cfg) = self.config.a2a_server {
            if a2a_cfg.enabled {
                let port = a2a_cfg.port;
                tokio::spawn(crate::a2a_server::start_a2a_server(port));
            }
        }
        #[cfg(not(feature = "a2a-server"))]
        if let Some(a2a_cfg) = &self.config.a2a_server {
            if a2a_cfg.enabled {
                eprintln!("  ⚠️  A2A server configured but 'a2a-server' feature not enabled");
            }
        }

        for test in tests {
            eprintln!("\n╔══════════════════════════════");
            eprintln!("║  Test: {}", test.name);
            eprintln!("╚══════════════════════════════");

            self.usage.reset_per_test();

            let test_result = self.run_test(test, &tab);
            self.usage.commit_test(&test.name);

            if test_result.failed == 0 && test_result.total > 0 {
                report.tests_passed += 1;
                eprintln!("  Test ✅ Passed");
            } else if test_result.total > 0 {
                report.tests_failed += 1;
                eprintln!("  Test ❌ Failed");
            }

            report.passed += test_result.passed;
            report.failed += test_result.failed;
            report.skipped += test_result.skipped;
            report.details.extend(test_result.details);
        }

        Ok(report)
    }

    #[allow(clippy::too_many_lines)]
    fn run_test(&self, test: &TestGroup, tab: &Tab) -> TestRunResult {
        let base_url = test
            .base_url
            .clone()
            .or_else(|| self.config.base_url.clone())
            .unwrap_or_else(crate::base_url);

        let auto_navigate = test.auto_navigate.unwrap_or(self.config.auto_navigate);

        let start_url = test
            .start_url
            .clone()
            .or_else(|| self.config.start_url.clone())
            .unwrap_or_else(|| "/dashboard".to_owned());

        if auto_navigate {
            let full_url = resolve_url(&start_url, &base_url);
            eprintln!("  → auto-navigate: {full_url}");
            let _ = tab.navigate_to(&full_url);
            let _ = tab.wait_until_navigated();
            std::thread::sleep(Duration::from_secs(4));
        }

        let mut result = TestRunResult::default();

        for step in &test.steps {
            result.total += 1;

            let wait_ms = match step {
                TestStep::Navigate { wait_after_ms, .. }
                | TestStep::Click { wait_after_ms, .. }
                | TestStep::Type { wait_after_ms, .. } => *wait_after_ms,
                _ => None,
            };

            let step_result = match step {
                TestStep::Navigate { url, .. } => {
                    let full_url = resolve_url(url, &base_url);
                    run_navigate_step(&full_url, tab)
                }
                TestStep::Click {
                    target,
                    selector,
                    endpoint,
                    ..
                } => self.run_click(
                    target,
                    selector.as_deref(),
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    tab,
                ),
                TestStep::Type {
                    target,
                    text,
                    selector,
                    endpoint,
                    ..
                } => self.run_type(
                    target,
                    text,
                    selector.as_deref(),
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    tab,
                ),
                TestStep::Wait {
                    target,
                    selector,
                    timeout_ms,
                    endpoint,
                } => self.run_wait(
                    target,
                    selector.as_deref(),
                    *timeout_ms,
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    tab,
                ),
                TestStep::Assert {
                    definition,
                    preset,
                    prompt,
                    assert_text,
                    endpoint,
                } => self.run_assert(
                    definition.as_deref(),
                    preset.as_deref(),
                    prompt.as_deref(),
                    assert_text.as_deref(),
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    tab,
                ),
                TestStep::Screenshot { path } => Self::run_screenshot(path.as_deref(), tab),
                TestStep::Agent {
                    agent,
                    task,
                    definition,
                } => self.run_agent(agent, task, definition.as_deref(), test.endpoint.as_deref()),
                TestStep::Mcp { server, tool, args } => self.run_mcp(server, tool, args.as_ref()),
            };

            eprintln!(
                "    {} {} — {}",
                if step_result.status == StepStatus::Passed {
                    "✅"
                } else if step_result.status == StepStatus::Failed {
                    "❌"
                } else {
                    "⏭️"
                },
                step_result.name,
                step_result.message,
            );

            match step_result.status {
                StepStatus::Passed => result.passed += 1,
                StepStatus::Failed => result.failed += 1,
                StepStatus::Skipped => result.skipped += 1,
            }

            // Check per-test budget after each step
            let test_usage = self.usage.current_test_snapshot();
            let global_usage = self.usage.global_snapshot();
            let budget_status = self.budgets.check_all(
                &test.name,
                &test_usage,
                &global_usage,
                test.budget.as_ref(),
            );
            match budget_status {
                BudgetStatus::HardExceeded { message, .. } => {
                    crate::reporting::print_budget_error(&message);
                    result.details.push(StepResult {
                        name: "[budget]".into(),
                        status: StepStatus::Failed,
                        message,
                    });
                    result.failed += 1;
                    return result;
                }
                BudgetStatus::SoftExceeded { message, .. } => {
                    crate::reporting::print_budget_warning(&message);
                }
                BudgetStatus::Ok => {}
            }

            if let Some(ms) = wait_ms {
                std::thread::sleep(Duration::from_millis(ms));
            }

            result.details.push(step_result);
        }

        result
    }

    // ── step handlers ───────────────────────────────────────────────────

    fn run_click(
        &self,
        target: &str,
        selector_override: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        let selector = match self.resolve_selector(
            selector_override,
            target,
            step_endpoint,
            test_endpoint,
            tab,
        ) {
            Ok(s) => s,
            Err(msg) => {
                return StepResult {
                    name: format!("[click] {target}"),
                    status: StepStatus::Failed,
                    message: msg,
                };
            }
        };

        match tab.wait_for_element(&selector) {
            Ok(element) => match element.click() {
                Ok(_) => StepResult {
                    name: format!("[click] {target}"),
                    status: StepStatus::Passed,
                    message: format!("clicked {selector}"),
                },
                Err(e) => StepResult {
                    name: format!("[click] {target}"),
                    status: StepStatus::Failed,
                    message: format!("click failed on {selector}: {e}"),
                },
            },
            Err(e) => StepResult {
                name: format!("[click] {target}"),
                status: StepStatus::Failed,
                message: format!("element {selector} not found: {e}"),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_type(
        &self,
        target: &str,
        text: &str,
        selector_override: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        let selector = match self.resolve_selector(
            selector_override,
            target,
            step_endpoint,
            test_endpoint,
            tab,
        ) {
            Ok(s) => s,
            Err(msg) => {
                return StepResult {
                    name: format!("[type] {target}"),
                    status: StepStatus::Failed,
                    message: msg,
                };
            }
        };

        match tab.wait_for_element(&selector) {
            Ok(element) => {
                if let Err(e) = element.click() {
                    return StepResult {
                        name: format!("[type] {target}"),
                        status: StepStatus::Failed,
                        message: format!("click to focus {selector} failed: {e}"),
                    };
                }

                let js = format!(
                    "document.querySelector('{}').value = '';",
                    selector.replace('\'', "\\'")
                );
                let _ = tab.evaluate(&js, false);

                match element.type_into(text) {
                    Ok(_) => StepResult {
                        name: format!("[type] {target}"),
                        status: StepStatus::Passed,
                        message: format!("typed {text:?} into {selector}"),
                    },
                    Err(e) => StepResult {
                        name: format!("[type] {target}"),
                        status: StepStatus::Failed,
                        message: format!("type into {selector} failed: {e}"),
                    },
                }
            }
            Err(e) => StepResult {
                name: format!("[type] {target}"),
                status: StepStatus::Failed,
                message: format!("element {selector} not found: {e}"),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_wait(
        &self,
        target: &str,
        selector_override: Option<&str>,
        timeout_ms: Option<u64>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        let selector = match self.resolve_selector(
            selector_override,
            target,
            step_endpoint,
            test_endpoint,
            tab,
        ) {
            Ok(s) => s,
            Err(msg) => {
                return StepResult {
                    name: format!("[wait] {target}"),
                    status: StepStatus::Failed,
                    message: msg,
                };
            }
        };

        let timeout = Duration::from_millis(timeout_ms.unwrap_or(10_000));

        match tab.wait_for_element_with_custom_timeout(&selector, timeout) {
            Ok(_) => StepResult {
                name: format!("[wait] {target}"),
                status: StepStatus::Passed,
                message: format!("found {selector}"),
            },
            Err(e) => StepResult {
                name: format!("[wait] {target}"),
                status: StepStatus::Failed,
                message: format!("wait for {selector} timed out: {e}"),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_assert(
        &self,
        definition: Option<&str>,
        preset: Option<&str>,
        prompt: Option<&str>,
        assert_text: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        std::thread::sleep(Duration::from_millis(500));

        let page_content = get_page_text(tab);

        if let Some(def_name) = definition {
            if let Some(def) = self.definitions.get(def_name) {
                return self.run_assert_def(def, &page_content, step_endpoint, test_endpoint);
            }
            return StepResult {
                name: format!("[assert] {def_name}"),
                status: StepStatus::Failed,
                message: format!("definition '{def_name}' not found"),
            };
        }

        if let Some(preset_name) = preset {
            return self.run_preset(
                preset_name,
                assert_text,
                &page_content,
                step_endpoint,
                test_endpoint,
            );
        }

        if let Some(prompt_text) = prompt {
            return self.run_custom(prompt_text, &page_content, step_endpoint, test_endpoint);
        }

        StepResult {
            name: "[assert]".into(),
            status: StepStatus::Skipped,
            message: "no definition, preset, or prompt specified".into(),
        }
    }

    fn run_assert_def(
        &self,
        def: &AssertDefinition,
        page_content: &PageContent,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        // Agent-based definition: delegate to an A2A agent
        if let Some(ref agent) = def.agent {
            let task = def
                .task_template
                .as_deref()
                .unwrap_or("Evaluate the assertion")
                .replace("{url}", &page_content.url)
                .replace("{title}", &page_content.title)
                .replace("{content}", &page_content.body_text)
                .replace("{expected_text}", def.assert_text.as_deref().unwrap_or(""));

            return self.run_agent_step(agent, &task, &def.name);
        }

        // Custom preset: system + user_template provided in the definition
        if let (Some(system), Some(template)) = (&def.system, &def.user_template) {
            return self.run_custom_preset(
                &def.name,
                system,
                template,
                def.assert_text.as_deref(),
                page_content,
                step_endpoint,
                test_endpoint,
            );
        }

        def.preset.as_ref().map_or_else(
            || {
                def.prompt.as_ref().map_or_else(
                    || StepResult {
                        name: format!("[assert] {}", def.name),
                        status: StepStatus::Failed,
                        message: "definition has no preset, prompt, or system+user_template".into(),
                    },
                    |prompt| self.run_custom(prompt, page_content, step_endpoint, test_endpoint),
                )
            },
            |preset_name| {
                self.run_preset(
                    preset_name,
                    def.assert_text.as_deref(),
                    page_content,
                    step_endpoint,
                    test_endpoint,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_custom_preset(
        &self,
        name: &str,
        system: &str,
        template: &str,
        assert_text: Option<&str>,
        page_content: &PageContent,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        let user_prompt = template
            .replace("{url}", &page_content.url)
            .replace("{title}", &page_content.title)
            .replace("{content}", &page_content.body_text)
            .replace("{expected_text}", assert_text.unwrap_or(""))
            .replace("{description}", "");

        eprintln!("      assert: {name} (custom preset)");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = system.to_owned();

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(llm_chat_with_usage(&llm, &sys, &user_prompt))
        })
        .join()
        .unwrap();

        response.map_or_else(
            || StepResult {
                name: format!("[assert] {name}"),
                status: StepStatus::Failed,
                message: "LLM assertion call failed (server down?)".into(),
            },
            |lr| {
                usage.record_llm_call(
                    &endpoint_name,
                    endpoint,
                    lr.usage.prompt_tokens,
                    lr.usage.completion_tokens,
                );
                let content_lower = lr.content.to_lowercase().trim().to_owned();
                if content_lower.starts_with("pass") {
                    StepResult {
                        name: format!("[assert] {name}"),
                        status: StepStatus::Passed,
                        message: "PASS".into(),
                    }
                } else {
                    StepResult {
                        name: format!("[assert] {name}"),
                        status: StepStatus::Failed,
                        message: lr.content,
                    }
                }
            },
        )
    }

    fn run_preset(
        &self,
        preset_name: &str,
        assert_text: Option<&str>,
        page_content: &PageContent,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        let Some(preset) = ASSERTION_PRESETS.iter().find(|p| p.name == preset_name) else {
            return StepResult {
                name: format!("[assert] {preset_name}"),
                status: StepStatus::Failed,
                message: format!("unknown assertion preset: {preset_name}"),
            };
        };

        let user_prompt = preset
            .user_template
            .replace("{url}", &page_content.url)
            .replace("{title}", &page_content.title)
            .replace("{content}", &page_content.body_text)
            .replace("{expected_text}", assert_text.unwrap_or(""))
            .replace("{description}", "");

        eprintln!("      assert: {preset_name}");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = preset.system.to_owned();

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(llm_chat_with_usage(&llm, &sys, &user_prompt))
        })
        .join()
        .unwrap();

        response.map_or_else(
            || StepResult {
                name: format!("[assert] {preset_name}"),
                status: StepStatus::Failed,
                message: "LLM assertion call failed (server down?)".into(),
            },
            |lr| {
                usage.record_llm_call(
                    &endpoint_name,
                    endpoint,
                    lr.usage.prompt_tokens,
                    lr.usage.completion_tokens,
                );
                let content_lower = lr.content.to_lowercase().trim().to_owned();
                if content_lower.starts_with("pass") {
                    StepResult {
                        name: format!("[assert] {preset_name}"),
                        status: StepStatus::Passed,
                        message: "PASS".into(),
                    }
                } else {
                    StepResult {
                        name: format!("[assert] {preset_name}"),
                        status: StepStatus::Failed,
                        message: lr.content,
                    }
                }
            },
        )
    }

    fn run_custom(
        &self,
        prompt: &str,
        page_content: &PageContent,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        let system = "You are a QA tester evaluating a web page. Respond with exactly \"PASS\" if the assertion holds, or \"FAIL: <reason>\" if it does not.";

        let user = format!(
            "Page URL: {url}\nPage Title: {title}\n\nPage Content:\n{content}\n\nAssertion: {prompt}",
            url = page_content.url,
            title = page_content.title,
            content = page_content.body_text,
        );

        eprintln!("      custom assert");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = system.to_owned();

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(llm_chat_with_usage(&llm, &sys, &user))
        })
        .join()
        .unwrap();

        response.map_or_else(
            || StepResult {
                name: "[assert] custom".into(),
                status: StepStatus::Failed,
                message: "LLM assertion call failed (server down?)".into(),
            },
            |lr| {
                usage.record_llm_call(
                    &endpoint_name,
                    endpoint,
                    lr.usage.prompt_tokens,
                    lr.usage.completion_tokens,
                );
                let content_lower = lr.content.to_lowercase().trim().to_owned();
                if content_lower.starts_with("pass") {
                    StepResult {
                        name: "[assert] custom".into(),
                        status: StepStatus::Passed,
                        message: "PASS".into(),
                    }
                } else {
                    StepResult {
                        name: "[assert] custom".into(),
                        status: StepStatus::Failed,
                        message: lr.content,
                    }
                }
            },
        )
    }

    fn run_screenshot(path: Option<&str>, tab: &Tab) -> StepResult {
        let path = path.unwrap_or("screenshot.png");

        match tab.capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            true,
        ) {
            Ok(data) => {
                if let Err(e) = std::fs::write(path, &data) {
                    return StepResult {
                        name: format!("[screenshot] {path}"),
                        status: StepStatus::Failed,
                        message: format!("failed to write screenshot: {e}"),
                    };
                }
                StepResult {
                    name: format!("[screenshot] {path}"),
                    status: StepStatus::Passed,
                    message: format!("saved to {path}"),
                }
            }
            Err(e) => StepResult {
                name: format!("[screenshot] {path}"),
                status: StepStatus::Failed,
                message: format!("screenshot failed: {e}"),
            },
        }
    }

    /// Runs an A2A agent step.
    #[allow(clippy::literal_string_with_formatting_args)]
    fn run_agent(
        &self,
        agent_name: &str,
        task: &str,
        definition: Option<&str>,
        _test_endpoint: Option<&str>,
    ) -> StepResult {
        // If a definition is specified, look up the task template
        let resolved_task = if let Some(def_name) = definition {
            if let Some(def) = self.definitions.get(def_name) {
                let tmpl = def.task_template.as_deref().unwrap_or(task);
                tmpl.replace("{task}", task)
            } else {
                return StepResult {
                    name: format!("[agent] {def_name}"),
                    status: StepStatus::Failed,
                    message: format!("definition '{def_name}' not found"),
                };
            }
        } else {
            task.to_owned()
        };

        self.run_agent_step(agent_name, &resolved_task, &format!("agent:{agent_name}"))
    }

    fn run_agent_step(&self, agent_name: &str, task: &str, display_name: &str) -> StepResult {
        let Some(ep) = self.endpoints.get(agent_name) else {
            return StepResult {
                name: format!("[agent] {display_name}"),
                status: StepStatus::Failed,
                message: format!("agent endpoint '{agent_name}' not found"),
            };
        };

        if ep.url.is_empty() {
            return StepResult {
                name: format!("[agent] {display_name}"),
                status: StepStatus::Failed,
                message: format!("agent endpoint '{agent_name}' has no URL"),
            };
        }

        eprintln!("      → agent {agent_name}: {task}");

        let url = ep.url.clone();
        let client = A2aClient::new(&url, self.timeout);
        let task_clone = task.to_owned();

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(client.send_task(&task_clone))
        })
        .join()
        .unwrap();

        // Record the flat-cost call
        self.usage.record_flat_call(agent_name, ep);

        match response {
            Ok(text) => {
                let clean = text.trim().to_owned();
                let lower = clean.to_lowercase();
                if lower.starts_with("pass") {
                    StepResult {
                        name: format!("[agent] {display_name}"),
                        status: StepStatus::Passed,
                        message: format!("PASS: {clean}"),
                    }
                } else if lower.starts_with("fail") {
                    StepResult {
                        name: format!("[agent] {display_name}"),
                        status: StepStatus::Failed,
                        message: clean,
                    }
                } else {
                    StepResult {
                        name: format!("[agent] {display_name}"),
                        status: StepStatus::Passed,
                        message: format!("response: {clean}"),
                    }
                }
            }
            Err(e) => StepResult {
                name: format!("[agent] {display_name}"),
                status: StepStatus::Failed,
                message: format!("agent call failed: {e}"),
            },
        }
    }

    /// Runs an MCP tool call step.
    fn run_mcp(
        &self,
        server_name: &str,
        tool_name: &str,
        args: Option<&serde_json::Value>,
    ) -> StepResult {
        let Some(ep) = self.endpoints.get(server_name) else {
            return StepResult {
                name: format!("[mcp] {server_name}:{tool_name}"),
                status: StepStatus::Failed,
                message: format!("MCP server endpoint '{server_name}' not found"),
            };
        };

        let cmd = ep.command.as_deref().unwrap_or("");
        if cmd.is_empty() {
            return StepResult {
                name: format!("[mcp] {server_name}:{tool_name}"),
                status: StepStatus::Failed,
                message: format!("MCP server '{server_name}' has no command configured"),
            };
        }

        eprintln!("      → mcp {server_name} {tool_name}");

        let args_val = args.cloned().unwrap_or(serde_json::Value::Null);

        let command = cmd.to_owned();
        let args_vec = ep.args.clone();
        let tool = tool_name.to_owned();

        let response = std::thread::spawn(move || {
            let mut mcp_client =
                McpClient::connect_stdio(&command, &args_vec).map_err(|e| e.to_string())?;
            mcp_client
                .call_tool(&tool, &args_val)
                .map_err(|e| e.to_string())
        })
        .join()
        .unwrap();

        // Record the flat-cost call
        self.usage.record_flat_call(server_name, ep);

        match response {
            Ok(result) => {
                if result.isError {
                    StepResult {
                        name: format!("[mcp] {server_name}:{tool_name}"),
                        status: StepStatus::Failed,
                        message: result.to_string(),
                    }
                } else {
                    StepResult {
                        name: format!("[mcp] {server_name}:{tool_name}"),
                        status: StepStatus::Passed,
                        message: result.to_string(),
                    }
                }
            }
            Err(e) => StepResult {
                name: format!("[mcp] {server_name}:{tool_name}"),
                status: StepStatus::Failed,
                message: format!("MCP call failed: {e}"),
            },
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Builds an `LlmConfig` from a resolved endpoint, falling back to
    /// the runner's default LLM config for any unset fields.
    fn build_llm_for_endpoint(&self, endpoint: &crate::endpoints::ResolvedEndpoint) -> LlmConfig {
        LlmConfig {
            url: if endpoint.url.is_empty() {
                self.llm.url.clone()
            } else {
                endpoint.url.clone()
            },
            model: endpoint
                .model
                .clone()
                .unwrap_or_else(|| self.llm.model.clone()),
            api_key: endpoint
                .api_key
                .clone()
                .or_else(|| self.llm.api_key.clone()),
            headers: if endpoint.headers.is_empty() {
                self.llm.headers.clone()
            } else {
                endpoint.headers.clone()
            },
            timeout: self.llm.timeout,
            temperature: self.llm.temperature,
            thinking: self.llm.thinking,
            model_params: self.llm.model_params.clone(),
        }
    }

    /// Resolves a CSS selector for the target element. Uses the explicit
    /// `selector` if provided, otherwise asks the LLM to find the element
    /// from the natural language `target` description and page DOM.
    fn resolve_selector(
        &self,
        css_override: Option<&str>,
        target: &str,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> Result<String, String> {
        if let Some(explicit) = css_override {
            return Ok(explicit.to_owned());
        }

        let dom_info = extract_dom_info(tab)?;
        let page_content = get_page_text(tab);

        let system = concat!(
            "You are a browser automation selector generator. ",
            "Given a web page's content and interactive elements, ",
            "return ONLY the best CSS selector for the described element. ",
            "Output nothing except the CSS selector. ",
            "Prefer selectors in this order: #id, [data-testid=\"...\"], ",
            "[name=\"...\"], tag.class, tag. ",
            "Never output explanations, markdown, or extra text."
        );

        let user = format!(
            "Page URL: {}\nPage Title: {}\n\nPage body text (first 4000 chars):\n{}\n\nInteractive elements:\n{}\n\nFind the CSS selector for: {}",
            page_content.url,
            page_content.title,
            truncate(&page_content.body_text, 4000),
            dom_info,
            target,
        );

        eprintln!("      LLM targeting: {target}");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Targeting);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let endpoint_clone = endpoint.clone();
        let sys = system.to_owned();

        let selector = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(llm_chat_with_usage(&llm, &sys, &user))
        })
        .join()
        .unwrap();

        match selector {
            Some(lr) => {
                usage.record_llm_call(
                    &endpoint_name,
                    &endpoint_clone,
                    lr.usage.prompt_tokens,
                    lr.usage.completion_tokens,
                );
                let clean = lr
                    .content
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .trim_matches('`')
                    .to_owned();
                eprintln!("      resolved selector: {clean}");
                Ok(clean)
            }
            None => Err("LLM element targeting failed (server down?)".to_owned()),
        }
    }
}

// ── Free helper functions ──────────────────────────────────────────────

fn run_navigate_step(full_url: &str, tab: &Tab) -> StepResult {
    let name = format!("[navigate] {full_url}");
    match tab.navigate_to(full_url) {
        Ok(_) => {
            let _ = tab.wait_until_navigated();
            StepResult {
                name,
                status: StepStatus::Passed,
                message: format!("navigated to {full_url}"),
            }
        }
        Err(e) => StepResult {
            name,
            status: StepStatus::Failed,
            message: format!("navigation failed: {e}"),
        },
    }
}

fn extract_dom_info(tab: &Tab) -> Result<String, String> {
    let result = tab
        .evaluate(DOM_EXTRACT_JS, false)
        .map_err(|e| format!("DOM extraction failed: {e}"))?;

    let json_str = result
        .value
        .as_ref()
        .and_then(|v| v.as_str())
        .unwrap_or("[]");

    let elements: Vec<String> = serde_json::from_str(json_str).unwrap_or_default();

    if elements.is_empty() {
        return Ok("(no interactive elements found)".to_owned());
    }

    Ok(elements.join("\n"))
}

fn get_page_text(tab: &Tab) -> PageContent {
    let url = tab.get_url();

    let title = tab
        .evaluate("document.title", false)
        .ok()
        .and_then(|r| r.value)
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".to_owned());

    let body_text = tab
        .evaluate("document.body ? document.body.innerText : ''", false)
        .ok()
        .and_then(|r| r.value)
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default();

    PageContent {
        url,
        title,
        body_text: truncate(&body_text, 8000),
    }
}

fn resolve_url(url: &str, base_url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        return url.to_owned();
    }
    let base = base_url.trim_end_matches('/');
    if url.starts_with('/') {
        format!("{base}{url}")
    } else {
        format!("{base}/{url}")
    }
}

// ── Support types ──────────────────────────────────────────────────────

#[derive(Default)]
struct TestRunResult {
    passed: u32,
    failed: u32,
    skipped: u32,
    total: u32,
    details: Vec<StepResult>,
}

struct PageContent {
    url: String,
    title: String,
    body_text: String,
}
