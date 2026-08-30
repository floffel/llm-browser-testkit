use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use headless_chrome::{Browser, LaunchOptions, Tab};

use crate::a2a::A2aClient;
use crate::budgets::{BudgetStatus, BudgetTracker};
use crate::costs::UsageTracker;
use crate::diagnostics;
use crate::endpoints::{EndpointRegistry, TaskType};
use crate::llm_chat_vision_with_usage;
use crate::llm_chat_with_usage;
use crate::mcp_client::McpClient;
use crate::scenario::{AssertDefinition, ScenarioConfig, TestGroup, TestStep};
use crate::selectors::{sanitize_selector, selector_is_useless, validate_selector};
use crate::truncate;
use crate::LlmConfig;
use crate::DOM_EXTRACT_JS;

/// One detected layout defect (`layout_no_issues` preset).
#[derive(Debug, serde::Deserialize)]
struct LayoutIssue {
    #[serde(rename = "type")]
    issue_type: String,
    element: String,
    detail: String,
}

/// In-browser DOM layout scan for `layout_no_issues`.
///
/// Geometry-only checks (no LLM, no pixels):
/// 1. page horizontal overflow (`scrollWidth` > viewport width);
/// 2. visible, non-fixed elements that stick out of the viewport
///    (right/bottom edge) while still partially on screen;
/// 3. text clipped by `overflow: hidden` containers whose content
///    is measurably larger than the box;
/// 4. interactive elements (buttons/links/inputs) whose center point
///    is covered by a different element that would intercept the click.
///
/// Intentional stacking (off-canvas drawers, dropdown menus, badges,
/// fixed headers) is excluded by the position/relation filters.
const LAYOUT_SCAN_JS: &str = r#"
(() => {
  const issues = [];
  const push = (type, el, detail) => {
    if (issues.length >= 30) return;
    let element = el.tagName.toLowerCase();
    if (el.id) element += '#' + el.id;
    else if (typeof el.className === 'string' && el.className.trim())
      element += '.' + el.className.trim().split(/\s+/).join('.');
    issues.push({ type, element, detail: String(detail).slice(0, 220) });
  };
  const vw = document.documentElement.clientWidth || window.innerWidth;
  const vh = document.documentElement.clientHeight || window.innerHeight;
  if (!vw || !vh) return JSON.stringify(issues);
  const de = document.documentElement;
  // 1. Page-level horizontal overflow.
  if (de.scrollWidth > vw + 2)
    push('page-overflow-x', de,
      'page scrollWidth ' + de.scrollWidth + ' exceeds viewport width ' + vw);
  const all = Array.prototype.slice.call(document.querySelectorAll('body *'));
  const visible = (cs) => cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity || '1') !== 0;
  const hasContent = (el) =>
    ((el.textContent || '').trim().length > 0) ||
    !!el.querySelector('img,svg,video,canvas,iframe,button,input,textarea,select');
  // 2. Elements sticking out of the viewport (partially visible only).
  for (const el of all) {
    const cs = getComputedStyle(el);
    if (!visible(cs)) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (cs.position === 'fixed' || cs.position === 'sticky') continue;
    if (!hasContent(el) && el.children.length === 0) continue;
    if (r.top >= vh || r.left >= vw) continue; // fully offscreen = normal scroll content
    const overRight = r.right - vw;
    const overBottom = r.bottom - vh;
    if (overRight > 2 || overBottom > 2) {
      let where = '';
      if (overRight > 2 && overBottom > 2) where = 'right+bottom edges';
      else if (overRight > 2) where = 'right edge (' + Math.round(r.right) + ' > ' + vw + ')';
      else where = 'bottom edge (' + Math.round(r.bottom) + ' > ' + vh + ')';
      push('element-out-of-viewport', el, 'extends ' + Math.max(overRight, overBottom).toFixed(0) + 'px past the ' + where);
    }
  }
  // 3. Text clipped by overflow:hidden containers.
  for (const el of all) {
    const cs = getComputedStyle(el);
    if (cs.overflowX !== 'hidden' && cs.overflowY !== 'hidden') continue;
    if (el.scrollWidth <= el.clientWidth + 2 && el.scrollHeight <= el.clientHeight + 2) continue;
    if (!(el.textContent || '').trim()) continue;
    push('text-clipped', el,
      'content ' + el.scrollWidth + 'x' + el.scrollHeight +
      ' clipped to ' + el.clientWidth + 'x' + el.clientHeight);
  }
  // 4. Interactive elements covered by a different element.
  const interactive =
    'button, a[href], input, textarea, select, [role="button"], [role="link"], [role="menuitem"], label, .mdc-button, .mat-mdc-button, .mat-mdc-icon-button, .mdc-fab';
  const targets = document.querySelectorAll(interactive);
  for (const el of targets) {
    const r = el.getBoundingClientRect();
    if (r.width < 6 || r.height < 6) continue;
    const cs = getComputedStyle(el);
    if (!visible(cs)) continue;
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx > vw || cy > vh) continue;
    const top = document.elementFromPoint(cx, cy);
    if (!top || top === el || el.contains(top) || top.contains(el)) continue;
    const tcs = getComputedStyle(top);
    if (!visible(tcs)) continue;
    if (tcs.pointerEvents === 'none') continue;
    const tr = top.getBoundingClientRect();
    if (tr.width * tr.height < r.width * r.height * 0.25) continue;
    const tname = top.tagName.toLowerCase() + (top.id ? '#' + top.id : '') +
      (typeof top.className === 'string' && top.className.trim() ? '.' + top.className.trim().split(/\s+/).join('.') : '');
    push('element-overlap', el,
      'center point at ' + Math.round(cx) + ',' + Math.round(cy) +
      ' is covered by <' + tname + '>');
  }
  return JSON.stringify(issues);
})()
"#;

/// How long the CDP connection stays open after the browser goes quiet.
///
/// `headless_chrome` ships a 30s default and tears down the entire connection
/// when no traffic arrives for that long; a run must own its connection for
/// its full duration instead.
const BROWSER_IDLE_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

/// Executes a [`Scenario`] against a real browser with optional LLM
/// assistance for element targeting and assertions.
pub struct ScenarioRunner {
    config: ScenarioConfig,
    definitions: HashMap<String, AssertDefinition>,
    llm: LlmConfig,
    timeout: Duration,
    viewport_width: u32,
    viewport_height: u32,
    /// The viewport currently applied in the browser (CDP emulation).
    /// Per-test overrides switch it mid-run; `(0, 0)` = not yet applied.
    applied_viewport: std::cell::Cell<(u32, u32)>,
    endpoints: EndpointRegistry,
    usage: Arc<UsageTracker>,
    budgets: BudgetTracker,
    /// Directory for failure artifacts (screenshots).
    artifacts_dir: PathBuf,
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
    AssertPreset {
        name: "visual_no_issues",
        system: "You are a visual QA engineer inspecting a website screenshot. Detect clearly visible layout and rendering defects: overlapping elements that hide content or controls, clipped or truncated text, content cut off at the viewport edges, misaligned or broken UI, blank/empty panels where content is expected, broken or missing images, duplicated elements, or rendering glitches. Ignore subjective aesthetics, intentional stacking (dropdowns, tooltips, layered design), and content that is simply not loaded (empty states). Only report defects that a user would actually see or be blocked by.",
        user_template: "Inspect the attached screenshot and the page text below.\n\nURL: {url}\nTitle: {title}\n\nPage Content:\n{content}\n\nAre there any clearly visible layout or rendering defects (overlaps, clipping, cut-off content, broken images, blank panels)? Respond with exactly \"PASS\" if the page renders cleanly, or \"FAIL: <describe each defect and where it appears>\" otherwise.",
    },
    AssertPreset {
        name: "visual_no_overlaps",
        system: "You are a visual QA engineer inspecting a website screenshot for OVERLAPPING elements that harm usability: one element covering another element's text, buttons, links, or input fields (cookie banners, modals, popovers, chat widgets, sticky headers, or mispositioned layers that hide content or intercept clicks). Ignore intentional, non-harmful stacking (dropdowns, tooltips, badges over avatars, layered design where nothing is hidden or unclickable). Only fail on overlaps that visibly hide content or would block a click.",
        user_template: "Inspect the attached screenshot and the page text below.\n\nURL: {url}\nTitle: {title}\n\nPage Content:\n{content}\n\nAre any elements overlapping in a way that hides page content, text, or interactive controls, or that would block clicks? Respond with exactly \"PASS\" if there are no such overlaps, or \"FAIL: <describe the overlapping elements and what they hide>\" otherwise.",
    },
    AssertPreset {
        name: "visual_text_visible",
        system: "You are a visual QA engineer inspecting a website screenshot. Determine whether a specific text is FULLY visible and readable: present in the viewport, not clipped, not cut off, not covered by another element, and not obscured by overlays or low contrast stacking. A partial word or a covered text counts as FAIL.",
        user_template: "Inspect the attached screenshot and the page text below.\n\nTEXT TO CHECK: \"{expected_text}\"\n\nURL: {url}\nTitle: {title}\n\nPage Content:\n{content}\n\nIs the text fully visible and readable in the screenshot (not clipped, covered, or hidden)? Respond with exactly \"PASS\" if it is fully visible, or \"FAIL: <explain what hides or clips it>\" otherwise.",
    },
    AssertPreset {
        name: "layout_no_issues",
        system: "DOM layout scanner: reports horizontal page overflow, elements sticking out of the viewport, clipped text in overflow:hidden containers, and interactive elements covered by other elements. Deterministic in-browser checks — no LLM call.",
        user_template: "Runs a deterministic DOM layout scan in the browser (no LLM call). Fails with the list of detected issues.",
    },
];

impl ScenarioRunner {
    /// Creates a new runner with the given scenario configuration and
    /// assertion definitions.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(scenario_config: ScenarioConfig, definitions: Vec<AssertDefinition>) -> Self {
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
        let endpoints = EndpointRegistry::from_config(&scenario_config.endpoints, Some(&llm));
        let budgets = BudgetTracker::from_config(&scenario_config.budgets);
        let defs_map: HashMap<String, AssertDefinition> = definitions
            .into_iter()
            .map(|d| (d.name.clone(), d))
            .collect();

        Self {
            timeout: Duration::from_secs(scenario_config.timeout_secs.unwrap_or(60)),
            viewport_width: scenario_config.viewport_width.unwrap_or(1280),
            viewport_height: scenario_config.viewport_height.unwrap_or(720),
            applied_viewport: std::cell::Cell::new((0, 0)),
            config: scenario_config.clone(),
            definitions: defs_map,
            llm,
            endpoints,
            usage: Arc::new(UsageTracker::new()),
            budgets,
            artifacts_dir: PathBuf::from(
                scenario_config
                    .artifacts_dir
                    .unwrap_or_else(|| "artifacts".to_owned()),
            ),
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
            // headless_chrome defaults this to 30s and shuts down the whole CDP
            // connection when no messages arrive for that long. A scenario can
            // easily exceed 30s of browser silence (slow LLM targeting/assertion
            // calls, page waits, budget checks between steps), after which every
            // remaining step fails with "Unable to make method calls because
            // underlying connection is closed" — one quiet gap kills the run.
            // Open-ended scenarios must own the connection for their full
            // duration, so keep it alive for 6 hours.
            idle_browser_timeout: BROWSER_IDLE_TIMEOUT,
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

        // Per-test viewport override: switch the browser via CDP
        // device-metrics emulation before this test runs.
        let vw = test.viewport_width.unwrap_or(self.viewport_width);
        let vh = test.viewport_height.unwrap_or(self.viewport_height);
        if self.applied_viewport.get() != (vw, vh) {
            self.apply_viewport(tab, vw, vh);
            self.applied_viewport.set((vw, vh));
        }

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

        for (step_index, step) in test.steps.iter().enumerate() {
            result.total += 1;

            let wait_ms = match step {
                TestStep::Navigate { wait_after_ms, .. }
                | TestStep::Click { wait_after_ms, .. }
                | TestStep::Type { wait_after_ms, .. } => *wait_after_ms,
                _ => None,
            };

            let mut step_result = match step {
                TestStep::Navigate { url, .. } => {
                    let full_url = resolve_url(url, &base_url);
                    run_navigate_step(&full_url, tab)
                }
                TestStep::Click {
                    target,
                    selector,
                    endpoint,
                    idempotent,
                    ..
                } => self.run_click(
                    target,
                    selector.as_deref(),
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    *idempotent,
                    tab,
                ),
                TestStep::Type {
                    target,
                    text,
                    selector,
                    endpoint,
                    idempotent,
                    ..
                } => self.run_type(
                    target,
                    text,
                    selector.as_deref(),
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    *idempotent,
                    tab,
                ),
                TestStep::Wait {
                    target,
                    selector,
                    text,
                    timeout_ms,
                    endpoint,
                    idempotent,
                } => self.run_wait(
                    target,
                    selector.as_deref(),
                    text.as_deref(),
                    *timeout_ms,
                    endpoint.as_deref(),
                    test.endpoint.as_deref(),
                    *idempotent,
                    tab,
                ),
                TestStep::Assert {
                    definition,
                    preset,
                    prompt,
                    assert_text,
                    endpoint,
                    screenshot,
                } => self.run_assert(
                    definition.as_deref(),
                    preset.as_deref(),
                    prompt.as_deref(),
                    assert_text.as_deref(),
                    *screenshot,
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

            // Failure diagnostics: capture the page state and a screenshot so
            // CI logs say WHAT the page looked like when the step failed,
            // instead of a bare "timed out: The event waited for never came".
            if step_result.status == StepStatus::Failed {
                let state = diagnostics::capture(tab);
                let screenshot = diagnostics::save_screenshot(
                    tab,
                    &self.artifacts_dir,
                    &test.name,
                    &test.name,
                    step_index,
                    step_kind_label(step),
                );
                step_result.message = format!(
                    "{base} — {excerpt}",
                    base = step_result.message,
                    excerpt = diagnostics::inline_excerpt(&state),
                );
                eprintln!(
                    "{}",
                    diagnostics::full_context(&state, screenshot.as_deref())
                );
            }

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

            // Fail fast: the first failed step ends the test and the
            // remaining steps are reported as skipped (no LLM budget is
            // burned asserting against a page that is already known broken).
            if step_result.status == StepStatus::Failed
                && !self.config.continue_on_failure
                && step_index + 1 < test.steps.len()
            {
                eprintln!(
                    "      ⏭️  failing fast: {} remaining step(s) skipped (set continue_on_failure = true in [config] to disable)",
                    test.steps.len() - step_index - 1
                );
                for skipped in &test.steps[step_index + 1..] {
                    result.total += 1;
                    result.skipped += 1;
                    eprintln!(
                        "    ⏭️  {} — skipped: previous step failed",
                        step_label(skipped)
                    );
                    result.details.push(StepResult {
                        name: step_label(skipped),
                        status: StepStatus::Skipped,
                        message: "skipped: previous step failed".into(),
                    });
                }
                result.details.push(step_result);
                return result;
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

    /// Applies a viewport size to the current tab via CDP
    /// `Emulation.setDeviceMetricsOverride`. Used by per-test viewport
    /// overrides and the viewport matrix. The initial window size set at
    /// browser launch is replaced by emulation; failures are logged but
    /// do not fail the test (a mismatched viewport only weakens coverage).
    fn apply_viewport(&self, tab: &Tab, width: u32, height: u32) {
        use headless_chrome::protocol::cdp::Emulation;
        let _ = self;
        let params = Emulation::SetDeviceMetricsOverride {
            width,
            height,
            device_scale_factor: 1.0,
            mobile: false,
            scale: None,
            screen_width: Some(width),
            screen_height: Some(height),
            position_x: None,
            position_y: None,
            dont_set_visible_size: None,
            screen_orientation: None,
            viewport: None,
            display_feature: None,
            device_posture: None,
        };
        eprintln!("      ↻ viewport: {width}x{height}");
        if let Err(e) = tab.call_method(params) {
            eprintln!("      ⚠️  viewport switch to {width}x{height} failed: {e}");
        }
    }

    // ── step handlers ───────────────────────────────────────────────────

    #[allow(clippy::too_many_lines)]
    fn run_click(
        &self,
        target: &str,
        selector_override: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        idempotent: bool,
        tab: &Tab,
    ) -> StepResult {
        let name = format!("[click] {target}");
        let selector = match self.resolve_selector(
            selector_override,
            target,
            step_endpoint,
            test_endpoint,
            tab,
        ) {
            Ok(s) => s,
            Err(msg) => {
                if idempotent {
                    return StepResult {
                        name,
                        status: StepStatus::Skipped,
                        message: format!("skipped (idempotent): no target found — {msg}"),
                    };
                }
                return StepResult {
                    name,
                    status: StepStatus::Failed,
                    message: msg,
                };
            }
        };

        // Idempotent steps probe briefly: a missing target means the
        // action was already done / not applicable (e.g. an
        // already-authenticated session), and skipping is the success
        // path, not a failure.
        let probe_secs = if idempotent { 5 } else { 10 };
        match tab.wait_for_element_with_custom_timeout(&selector, Duration::from_secs(probe_secs)) {
            Ok(element) => match element.click() {
                Ok(_) => StepResult {
                    name,
                    status: StepStatus::Passed,
                    message: format!("clicked {selector}"),
                },
                Err(e) => StepResult {
                    name,
                    status: StepStatus::Failed,
                    message: format!("click failed on {selector}: {e}"),
                },
            },
            Err(e) if idempotent => StepResult {
                name,
                status: StepStatus::Skipped,
                message: format!("skipped (idempotent): element {selector} not present — {e}"),
            },
            Err(e) => StepResult {
                name,
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
        idempotent: bool,
        tab: &Tab,
    ) -> StepResult {
        let name = format!("[type] {target}");
        let selector = match self.resolve_selector(
            selector_override,
            target,
            step_endpoint,
            test_endpoint,
            tab,
        ) {
            Ok(s) => s,
            Err(msg) => {
                if idempotent {
                    return StepResult {
                        name,
                        status: StepStatus::Skipped,
                        message: format!("skipped (idempotent): no target found — {msg}"),
                    };
                }
                return StepResult {
                    name,
                    status: StepStatus::Failed,
                    message: msg,
                };
            }
        };

        let probe_secs = if idempotent { 5 } else { 10 };
        match tab.wait_for_element_with_custom_timeout(&selector, Duration::from_secs(probe_secs)) {
            Ok(element) => {
                if let Err(e) = element.click() {
                    return StepResult {
                        name,
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
                        name,
                        status: StepStatus::Passed,
                        message: format!("typed {text:?} into {selector}"),
                    },
                    Err(e) => StepResult {
                        name,
                        status: StepStatus::Failed,
                        message: format!("type into {selector} failed: {e}"),
                    },
                }
            }
            Err(e) if idempotent => StepResult {
                name,
                status: StepStatus::Skipped,
                message: format!("skipped (idempotent): element {selector} not present — {e}"),
            },
            Err(e) => StepResult {
                name,
                status: StepStatus::Failed,
                message: format!("element {selector} not found: {e}"),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn run_wait(
        &self,
        target: &str,
        selector_override: Option<&str>,
        text: Option<&str>,
        timeout_ms: Option<u64>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        idempotent: bool,
        tab: &Tab,
    ) -> StepResult {
        let timeout = Duration::from_millis(timeout_ms.unwrap_or(10_000));
        let step_name = format!("[wait] {target}");

        // Resolve an explicit selector only (text-only waits are LLM-free).
        let selector = match selector_override {
            Some(s) => Some(s.to_owned()),
            None if text.is_some() => None,
            None => match self.resolve_selector(None, target, step_endpoint, test_endpoint, tab) {
                Ok(s) => Some(s),
                Err(msg) => {
                    if idempotent {
                        return StepResult {
                            name: step_name,
                            status: StepStatus::Skipped,
                            message: format!("skipped (idempotent): no target found — {msg}"),
                        };
                    }
                    return StepResult {
                        name: step_name,
                        status: StepStatus::Failed,
                        message: msg,
                    };
                }
            },
        };

        if text.is_some() {
            let sel_js = selector
                .as_deref()
                .map(crate::selectors::selector_matches_js);
            let text_js = text.map(|t| {
                let escaped = t.replace('\\', "\\\\").replace('\'', "\\'");
                format!("document.body ? document.body.innerText.includes('{escaped}') : false")
            });

            let deadline = Instant::now() + timeout;
            loop {
                let sel_ok = sel_js
                    .as_ref()
                    .is_none_or(|js| eval_bool(tab, js).unwrap_or(false));
                let text_ok = text_js
                    .as_ref()
                    .is_none_or(|js| eval_bool(tab, js).unwrap_or(false));
                if sel_ok && text_ok {
                    let mut what = Vec::new();
                    if let Some(sel) = &selector {
                        what.push(format!("found {sel}"));
                    }
                    if let Some(t) = text {
                        what.push(format!("text {t:?} visible"));
                    }
                    return StepResult {
                        name: step_name,
                        status: StepStatus::Passed,
                        message: what.join(" and "),
                    };
                }
                if Instant::now() >= deadline {
                    let mut what = Vec::new();
                    if let Some(sel) = &selector {
                        what.push(sel.clone());
                    }
                    if let Some(t) = text {
                        what.push(format!("text {t:?}"));
                    }
                    let message = format!(
                        "wait for {} timed out after {}ms: the event waited for never came",
                        what.join(" / "),
                        timeout.as_millis(),
                    );
                    if idempotent {
                        return StepResult {
                            name: step_name,
                            status: StepStatus::Skipped,
                            message: format!("skipped (idempotent): {message}"),
                        };
                    }
                    return StepResult {
                        name: step_name,
                        status: StepStatus::Failed,
                        message,
                    };
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }

        match selector.as_deref() {
            Some(sel) => match tab.wait_for_element_with_custom_timeout(sel, timeout) {
                Ok(_) => StepResult {
                    name: step_name,
                    status: StepStatus::Passed,
                    message: format!("found {sel}"),
                },
                Err(e) if idempotent => StepResult {
                    name: step_name,
                    status: StepStatus::Skipped,
                    message: format!(
                        "skipped (idempotent): wait for {sel} timed out after {}ms: {e}",
                        timeout.as_millis()
                    ),
                },
                Err(e) => StepResult {
                    name: step_name,
                    status: StepStatus::Failed,
                    message: format!(
                        "wait for {sel} timed out after {}ms: {e}",
                        timeout.as_millis()
                    ),
                },
            },
            None => StepResult {
                name: step_name,
                status: StepStatus::Failed,
                message: "wait step has neither selector nor text".into(),
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
        screenshot: bool,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        std::thread::sleep(Duration::from_millis(500));

        let page_content = get_page_text(tab);

        // Vision attach: capture the viewport once per assert step and hand
        // the JPEG data URL to the preset/prompt evaluation below.
        let image = if screenshot {
            let endpoint = self
                .endpoints
                .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
            if !endpoint.vision {
                return StepResult {
                    name: "[assert]".into(),
                    status: StepStatus::Failed,
                    message: format!(
                        "screenshot requested but endpoint '{name}' does not declare vision = true (add vision = true to its [config.endpoints] entry)",
                        name = endpoint.name
                    ),
                };
            }
            match crate::vision::capture_screenshot_data_url(
                tab,
                self.config
                    .screenshot_max_dimension
                    .unwrap_or(crate::vision::DEFAULT_MAX_DIMENSION),
            ) {
                Ok(data_url) => Some(data_url),
                Err(e) => {
                    return StepResult {
                        name: "[assert]".into(),
                        status: StepStatus::Failed,
                        message: format!("screenshot capture failed: {e}"),
                    };
                }
            }
        } else {
            None
        };

        if let Some(def_name) = definition {
            if let Some(def) = self.definitions.get(def_name) {
                return self.run_assert_def(
                    def,
                    &page_content,
                    image.as_deref(),
                    step_endpoint,
                    test_endpoint,
                    tab,
                );
            }
            return StepResult {
                name: format!("[assert] {def_name}"),
                status: StepStatus::Failed,
                message: format!("definition '{def_name}' not found"),
            };
        }

        if let Some(preset_name) = preset {
            // Deterministic DOM layout scan — runs JS in the browser and
            // never calls the LLM (free, fast, no pixel budget).
            if preset_name == "layout_no_issues" {
                return self.run_layout_preset(tab);
            }
            return self.run_preset(
                preset_name,
                assert_text,
                &page_content,
                image.as_deref(),
                step_endpoint,
                test_endpoint,
            );
        }

        if let Some(prompt_text) = prompt {
            return self.run_custom(
                prompt_text,
                &page_content,
                image.as_deref(),
                step_endpoint,
                test_endpoint,
            );
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
        image: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
        tab: &Tab,
    ) -> StepResult {
        // Agent-based definition: delegate to an A2A agent
        if let Some(ref agent) = def.agent {
            if image.is_some() {
                return StepResult {
                    name: format!("[assert] {}", def.name),
                    status: StepStatus::Failed,
                    message: "agent-backed assertions do not support screenshots".into(),
                };
            }
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
                image,
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
                    |prompt| {
                        self.run_custom(prompt, page_content, image, step_endpoint, test_endpoint)
                    },
                )
            },
            |preset_name| {
                if preset_name == "layout_no_issues" {
                    return self.run_layout_preset(tab);
                }
                self.run_preset(
                    preset_name,
                    def.assert_text.as_deref(),
                    page_content,
                    image,
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
        image: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        let user_prompt = template
            .replace("{url}", &page_content.url)
            .replace("{title}", &page_content.title)
            .replace("{content}", &page_content.body_text)
            .replace("{expected_text}", assert_text.unwrap_or(""))
            .replace("{description}", "");

        // Custom preset definitions frequently forget the {content}
        // placeholder — without it the LLM has no page to evaluate and
        // answers "I can't determine that without seeing the page". Always
        // append the page context unless the template already references it.
        let user_prompt = if template.contains("{content}") {
            user_prompt
        } else {
            format!(
                "{user_prompt}\n\nPage URL: {url}\nPage Title: {title}\n\nPage Content:\n{content}",
                url = page_content.url,
                title = page_content.title,
                content = page_content.body_text,
            )
        };

        eprintln!("      assert: {name} (custom preset)");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = system.to_owned();
        let image = image.map(str::to_owned);

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let call = async {
                match image.as_deref() {
                    Some(img) => llm_chat_vision_with_usage(&llm, &sys, &user_prompt, img).await,
                    None => llm_chat_with_usage(&llm, &sys, &user_prompt).await,
                }
            };
            rt.block_on(call)
        })
        .join()
        .unwrap();

        response.map_or_else(
            |e| StepResult {
                name: format!("[assert] {name}"),
                status: StepStatus::Failed,
                message: format!("LLM assertion call failed: {e}"),
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
        image: Option<&str>,
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
        if preset_name.starts_with("visual_") && image.is_none() {
            return StepResult {
                name: format!("[assert] {preset_name}"),
                status: StepStatus::Failed,
                message: format!(
                    "preset '{preset_name}' evaluates the page screenshot — set screenshot = true on the assert step and point it at a vision endpoint (vision = true)"
                ),
            };
        }

        let user_prompt = preset
            .user_template
            .replace("{url}", &page_content.url)
            .replace("{title}", &page_content.title)
            .replace("{content}", &page_content.body_text)
            .replace("{expected_text}", assert_text.unwrap_or(""))
            .replace("{description}", "");

        // Same safety net as custom presets: never let the LLM answer with
        // no page context at all.
        let user_prompt = if preset.user_template.contains("{content}") {
            user_prompt
        } else {
            format!(
                "{user_prompt}\n\nPage URL: {url}\nPage Title: {title}\n\nPage Content:\n{content}",
                url = page_content.url,
                title = page_content.title,
                content = page_content.body_text,
            )
        };

        eprintln!("      assert: {preset_name}");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = preset.system.to_owned();
        let image = image.map(str::to_owned);

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let call = async {
                match image.as_deref() {
                    Some(img) => llm_chat_vision_with_usage(&llm, &sys, &user_prompt, img).await,
                    None => llm_chat_with_usage(&llm, &sys, &user_prompt).await,
                }
            };
            rt.block_on(call)
        })
        .join()
        .unwrap();

        response.map_or_else(
            |e| StepResult {
                name: format!("[assert] {preset_name}"),
                status: StepStatus::Failed,
                message: format!("LLM assertion call failed: {e}"),
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

    /// Runs the deterministic DOM layout scan (`layout_no_issues`).
    ///
    /// Evaluates the layout-scan JS in the page and fails with the list of
    /// detected issues: horizontal page overflow, visible elements sticking
    /// out of the viewport, text clipped by `overflow: hidden` containers,
    /// and interactive elements covered by other elements. No LLM call —
    /// checks are geometry-based so the check is free, deterministic, and
    /// safe to run on every page × viewport variant.
    fn run_layout_preset(&self, tab: &Tab) -> StepResult {
        let _ = self;
        let name = "[assert] layout_no_issues".to_owned();
        eprintln!("      assert: layout_no_issues (DOM layout scan)");
        let result = tab.evaluate(LAYOUT_SCAN_JS, false);
        let json_str = match result {
            Ok(r) => r
                .value
                .as_ref()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| "[]".to_owned()),
            Err(e) => {
                return StepResult {
                    name,
                    status: StepStatus::Failed,
                    message: format!("layout scan JS failed: {e}"),
                };
            }
        };
        let issues: Vec<LayoutIssue> = serde_json::from_str(&json_str).unwrap_or_default();
        if issues.is_empty() {
            return StepResult {
                name,
                status: StepStatus::Passed,
                message: "PASS — no layout defects detected".into(),
            };
        }
        let mut lines: Vec<String> = issues
            .iter()
            .take(10)
            .map(|i| {
                format!(
                    "- [{type_}] {element}: {detail}",
                    type_ = i.issue_type,
                    element = i.element,
                    detail = i.detail
                )
            })
            .collect();
        if issues.len() > 10 {
            lines.push(format!("- … and {} more", issues.len() - 10));
        }
        StepResult {
            name,
            status: StepStatus::Failed,
            message: format!(
                "FAIL — {} layout defect(s) detected:\n{}",
                issues.len(),
                lines.join("\n")
            ),
        }
    }

    fn run_custom(
        &self,
        prompt: &str,
        page_content: &PageContent,
        image: Option<&str>,
        step_endpoint: Option<&str>,
        test_endpoint: Option<&str>,
    ) -> StepResult {
        let system = "You are a QA tester evaluating a web page. Respond with exactly \"PASS\" if the assertion holds, or \"FAIL: <reason>\" if it does not.";

        let mut user = format!(
            "Page URL: {url}\nPage Title: {title}\n\nPage Content:\n{content}\n\nAssertion: {prompt}",
            url = page_content.url,
            title = page_content.title,
            content = page_content.body_text,
        );
        if image.is_some() {
            user.push_str(
                "\n\nA screenshot of the page is attached — inspect it for visual evidence when answering.",
            );
        }

        eprintln!("      custom assert");

        let endpoint = self
            .endpoints
            .resolve(step_endpoint.or(test_endpoint), TaskType::Assertion);
        let llm = self.build_llm_for_endpoint(endpoint);
        let usage = Arc::clone(&self.usage);
        let endpoint_name = endpoint.name.clone();
        let sys = system.to_owned();
        let image = image.map(str::to_owned);

        let response = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let call = async {
                match image.as_deref() {
                    Some(img) => llm_chat_vision_with_usage(&llm, &sys, &user, img).await,
                    None => llm_chat_with_usage(&llm, &sys, &user).await,
                }
            };
            rt.block_on(call)
        })
        .join()
        .unwrap();

        response.map_or_else(
            |e| StepResult {
                name: "[assert] custom".into(),
                status: StepStatus::Failed,
                message: format!("LLM assertion call failed: {e}"),
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
    ///
    /// LLM responses are sanitized and verified against the live page: a
    /// response that is not a selector (empty, `:not(*)`, `null`, …) fails
    /// immediately with the raw LLM output, and a selector that matches
    /// nothing triggers one retry with feedback before failing.
    #[allow(clippy::too_many_lines)]
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

        let retry_user = format!(
            "Your previous answer was not usable. {}\n\nPage URL: {}\nPage Title: {}\n\nPage body text (first 4000 chars):\n{}\n\nInteractive elements:\n{}\n\nFind the CSS selector for: {}\nReturn ONLY a single CSS selector that matches an existing element. No explanations.",
            "The selector must match at least one element currently present on the page.",
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

        let call_llm = |prompt: &str| {
            let llm = llm.clone();
            let sys = sys.clone();
            let prompt = prompt.to_owned();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(llm_chat_with_usage(&llm, &sys, &prompt))
            })
            .join()
            .unwrap()
        };

        let first = call_llm(&user);
        let lr = match first {
            Ok(lr) => lr,
            Err(e) => {
                return Err(format!("LLM element targeting failed: {e}"));
            }
        };
        usage.record_llm_call(
            &endpoint_name,
            &endpoint_clone,
            lr.usage.prompt_tokens,
            lr.usage.completion_tokens,
        );
        let clean = sanitize_selector(&lr.content);
        eprintln!("      resolved selector: {clean}");

        if selector_is_useless(&clean) {
            return Err(format!(
                "LLM element targeting failed: the LLM did not return a usable selector for {target:?} (got {raw:?}). Check the page state in the diagnostics above.",
                raw = lr.content.trim(),
            ));
        }
        if let Err(reason) = validate_selector(&clean) {
            return Err(format!(
                "LLM element targeting failed: invalid selector for {target:?}: {reason} (LLM response: {raw:?})",
                raw = lr.content.trim(),
            ));
        }
        if !selector_matches(tab, &clean).unwrap_or(false) {
            // One retry with feedback: flaky models occasionally invent a
            // selector that does not exist on the page.
            eprintln!(
                "      selector {clean} matches nothing — retrying LLM targeting with feedback"
            );
            let second = call_llm(&retry_user);
            let lr2 = match second {
                Ok(lr2) => lr2,
                Err(e) => {
                    return Err(format!(
                        "LLM element targeting failed: first answer {clean:?} matched nothing, retry also failed: {e}"
                    ));
                }
            };
            usage.record_llm_call(
                &endpoint_name,
                &endpoint_clone,
                lr2.usage.prompt_tokens,
                lr2.usage.completion_tokens,
            );
            let clean2 = sanitize_selector(&lr2.content);
            eprintln!("      resolved selector (retry): {clean2}");
            if selector_is_useless(&clean2) {
                return Err(format!(
                    "LLM element targeting failed: selector {clean:?} matched nothing; the retry returned no usable selector for {target:?} (got {raw:?}). Page excerpt: {excerpt}",
                    raw = lr2.content.trim(),
                    excerpt = truncate(&page_content.body_text, 300),
                ));
            }
            if !selector_matches(tab, &clean2).unwrap_or(false) {
                return Err(format!(
                    "LLM element targeting failed: selector {clean2:?} does not match any element on the page for {target:?}. Verify the page state in the diagnostics; the login/SPA may not have rendered."
                ));
            }
            return Ok(clean2);
        }

        Ok(clean)
    }
}

/// Evaluates a JS expression that is expected to return a boolean.
fn eval_bool(tab: &Tab, js: &str) -> Result<bool, String> {
    tab.evaluate(js, false)
        .map_err(|e| format!("evaluate failed: {e}"))?
        .value
        .and_then(|v| v.as_bool())
        .ok_or_else(|| "evaluate returned non-boolean".to_owned())
}

/// Checks whether a CSS selector matches at least one current element.
fn selector_matches(tab: &Tab, selector: &str) -> Result<bool, String> {
    eval_bool(tab, &crate::selectors::selector_matches_js(selector))
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
        .evaluate(
            "document.body ? document.body.innerText : document.documentElement.innerText",
            false,
        )
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

/// Human-readable label for a step, used when steps are skipped after an
/// earlier failure.
fn step_label(step: &TestStep) -> String {
    match step {
        TestStep::Navigate { url, .. } => format!("[navigate] {url}"),
        TestStep::Click { target, .. } => format!("[click] {target}"),
        TestStep::Type { target, .. } => format!("[type] {target}"),
        TestStep::Wait { target, .. } => format!("[wait] {target}"),
        TestStep::Assert {
            definition,
            preset,
            prompt,
            ..
        } => definition.as_ref().map_or_else(
            || {
                preset.as_ref().map_or_else(
                    || {
                        prompt.as_ref().map_or_else(
                            || "[assert]".to_owned(),
                            |pr| format!("[assert] custom ({})", truncate(pr, 60)),
                        )
                    },
                    |p| format!("[assert] {p}"),
                )
            },
            |d| format!("[assert] {d}"),
        ),
        TestStep::Screenshot { .. } => "[screenshot]".to_owned(),
        TestStep::Agent { agent, .. } => format!("[agent] {agent}"),
        TestStep::Mcp { server, tool, .. } => format!("[mcp] {server}:{tool}"),
    }
}

/// Short kind word for artifact file names (e.g. `click`, `wait`, `assert`).
#[must_use]
const fn step_kind_label(step: &TestStep) -> &'static str {
    match step {
        TestStep::Navigate { .. } => "navigate",
        TestStep::Click { .. } => "click",
        TestStep::Type { .. } => "type",
        TestStep::Wait { .. } => "wait",
        TestStep::Assert { .. } => "assert",
        TestStep::Screenshot { .. } => "screenshot",
        TestStep::Agent { .. } => "agent",
        TestStep::Mcp { .. } => "mcp",
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
