//! Typed run events emitted by the runner.
//!
//! The runner no longer prints directly: every interesting point in a run
//! (test start/end, step start/end, LLM call, budget warning) is emitted as a
//! [`TestEvent`]. Sinks attached to the [`crate::reporting::Reporter`] render
//! these events for humans (console), machines (NDJSON log file, JUnit XML),
//! CI systems (GitHub workflow commands) and profilers (Perfetto trace).
//!
//! The event stream doubles as the run's source of truth: JUnit and trace
//! files are derived from it, and a `--log-file` NDJSON capture alone is
//! enough to replay or analyze a run.

use serde::Serialize;

/// Outcome of a single step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    /// Step executed successfully and all assertions passed.
    Passed,
    /// Step execution or assertion failed.
    Failed,
    /// Step was skipped.
    Skipped,
}

/// One event in the run's lifecycle.
///
/// Serialized as a flat JSON object with a `type` discriminator, e.g.
/// `{"type":"step_finished","test":"login","status":"passed",...}`. The
/// reporter adds a `ts` (epoch milliseconds) field on emission.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TestEvent {
    /// The run is about to execute `total_tests` tests.
    RunStarted {
        /// Number of tests in the run.
        total_tests: u32,
    },
    /// A test started.
    TestStarted {
        /// Test name (includes viewport-matrix suffix when expanded).
        test: String,
    },
    /// A step inside a test started.
    StepStarted {
        /// Enclosing test name.
        test: String,
        /// Zero-based step index within the test.
        index: u32,
        /// Human-readable label, e.g. `[click] the sign-in button`.
        label: String,
    },
    /// A step finished.
    StepFinished {
        /// Enclosing test name.
        test: String,
        /// Zero-based step index within the test.
        index: u32,
        /// Human-readable label, e.g. `[click] the sign-in button`.
        label: String,
        /// Outcome of the step.
        status: StepStatus,
        /// Wall-clock duration of the step.
        duration_ms: u64,
        /// Human-readable result message (includes the page-state excerpt
        /// and the failure message for failed steps).
        message: String,
        /// Multi-line failure diagnostics (URL, title, visible text, alerts)
        /// — `None` for non-failed steps.
        diagnostics: Option<String>,
        /// Path of the failure screenshot, if one was written.
        screenshot: Option<String>,
    },
    /// An LLM call started (one per attempted endpoint chain).
    LlmCallStarted {
        /// Enclosing test name.
        test: String,
        /// Zero-based step index within the test.
        index: u32,
        /// Endpoint name (or chain entry) being called.
        endpoint: String,
        /// Model name sent in the request.
        model: String,
        /// Call purpose: `targeting` or `assertion`.
        purpose: String,
    },
    /// An LLM call finished (success or exhaustion of all attempts).
    LlmCallFinished {
        /// Enclosing test name.
        test: String,
        /// Zero-based step index within the test.
        index: u32,
        /// Endpoint that answered (or the last one tried on failure).
        endpoint: String,
        /// Model name used.
        model: String,
        /// Call purpose: `targeting` or `assertion`.
        purpose: String,
        /// Whether an answer was obtained.
        ok: bool,
        /// Wall-clock duration of the call including retries.
        duration_ms: u64,
        /// Input (prompt) tokens billed.
        input_tokens: u64,
        /// Output (completion) tokens billed.
        output_tokens: u64,
        /// Computed cost in USD.
        cost: f64,
        /// Error message when `ok` is false.
        error: Option<String>,
    },
    /// A test finished.
    TestFinished {
        /// Test name.
        test: String,
        /// Number of passed steps.
        passed: u32,
        /// Number of failed steps.
        failed: u32,
        /// Number of skipped steps.
        skipped: u32,
        /// Wall-clock duration of the whole test.
        duration_ms: u64,
        /// Total cost of the test in USD.
        cost: f64,
        /// Total tokens consumed by the test.
        tokens: u64,
        /// Total LLM/MCP/agent calls made by the test.
        calls: u64,
    },
    /// The whole run finished.
    RunFinished {
        /// Tests that passed.
        tests_passed: u32,
        /// Tests that failed.
        tests_failed: u32,
        /// Steps that passed.
        steps_passed: u32,
        /// Steps that failed.
        steps_failed: u32,
        /// Steps that were skipped.
        steps_skipped: u32,
        /// Total cost in USD across all tests.
        total_cost: f64,
        /// Total tokens consumed.
        total_tokens: u64,
        /// Total calls made.
        total_calls: u64,
    },
    /// A non-fatal warning (budget soft-exceeded, feature not enabled, ...).
    Warning {
        /// Human-readable warning text.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{StepStatus, TestEvent};

    #[test]
    fn test_step_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(StepStatus::Passed).unwrap(),
            serde_json::json!("passed")
        );
        assert_eq!(
            serde_json::to_value(StepStatus::Failed).unwrap(),
            serde_json::json!("failed")
        );
        assert_eq!(
            serde_json::to_value(StepStatus::Skipped).unwrap(),
            serde_json::json!("skipped")
        );
    }

    #[test]
    fn test_event_serializes_with_type_tag() {
        let event = TestEvent::StepFinished {
            test: "login".into(),
            index: 2,
            label: "[click] the button".into(),
            status: StepStatus::Failed,
            duration_ms: 1234,
            message: "element #btn not found".into(),
            diagnostics: Some("    │ url: http://x".into()),
            screenshot: Some("artifacts/x.png".into()),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "step_finished");
        assert_eq!(value["test"], "login");
        assert_eq!(value["index"], 2);
        assert_eq!(value["label"], "[click] the button");
        assert_eq!(value["status"], "failed");
        assert_eq!(value["duration_ms"], 1234);
        assert_eq!(value["message"], "element #btn not found");
        assert_eq!(value["diagnostics"], "    │ url: http://x");
        assert_eq!(value["screenshot"], "artifacts/x.png");
    }

    #[test]
    fn test_run_started_shape() {
        let value = serde_json::to_value(TestEvent::RunStarted { total_tests: 3 }).unwrap();
        assert_eq!(value["type"], "run_started");
        assert_eq!(value["total_tests"], 3);
    }

    #[test]
    fn test_llm_call_finished_shape() {
        let value = serde_json::to_value(TestEvent::LlmCallFinished {
            test: "login".into(),
            index: 0,
            endpoint: "default".into(),
            model: "deepseek".into(),
            purpose: "targeting".into(),
            ok: false,
            duration_ms: 900,
            input_tokens: 100,
            output_tokens: 0,
            cost: 0.0012,
            error: Some("HTTP 429: slow down".into()),
        })
        .unwrap();
        assert_eq!(value["type"], "llm_call_finished");
        assert_eq!(value["endpoint"], "default");
        assert_eq!(value["model"], "deepseek");
        assert_eq!(value["purpose"], "targeting");
        assert_eq!(value["ok"], false);
        assert_eq!(value["cost"], 0.0012);
        assert_eq!(value["error"], "HTTP 429: slow down");
    }

    #[test]
    fn test_run_finished_shape() {
        let value = serde_json::to_value(TestEvent::RunFinished {
            tests_passed: 2,
            tests_failed: 1,
            steps_passed: 9,
            steps_failed: 1,
            steps_skipped: 2,
            total_cost: 0.05,
            total_tokens: 5000,
            total_calls: 7,
        })
        .unwrap();
        assert_eq!(value["type"], "run_finished");
        assert_eq!(value["tests_passed"], 2);
        assert_eq!(value["steps_failed"], 1);
        assert_eq!(value["total_cost"], 0.05);
    }

    #[test]
    fn test_warning_shape() {
        let value = serde_json::to_value(TestEvent::Warning {
            message: "budget soft".into(),
        })
        .unwrap();
        assert_eq!(value["type"], "warning");
        assert_eq!(value["message"], "budget soft");
    }
}
