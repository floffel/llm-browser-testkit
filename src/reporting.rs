//! Report sinks: console, NDJSON, JUnit XML, GitHub annotations, Perfetto trace.
//!
//! The [`Reporter`] is the single place test-run output flows through. The
//! runner emits [`TestEvent`]s; the reporter fans them out to every enabled
//! sink:
//!
//! - **Console** — level-filtered, optionally colorized, ASCII-safe lines so
//!   output renders in CI logs and dumb terminals. Failures always show full
//!   diagnostics; debug events (LLM calls, selector resolution) are hidden
//!   unless `-v`/`-vv` is passed.
//! - **NDJSON** (`--log-file`) — one JSON object per event, each with a `ts`
//!   epoch-millisecond field and a `type` discriminator. Machine-readable and
//!   lossless (no truncation): `jq '. | select(.type == "step_finished" and
//!   .status == "failed")' run.jsonl` works out of the box.
//! - **JUnit XML** (`--junit`) — one `<testcase>` per test with `<failure>`
//!   entries per failed step, for Jenkins/GitLab/Azure/TeamCity.
//! - **GitHub** — `::error::` workflow commands on failed steps plus a
//!   `GITHUB_STEP_SUMMARY` markdown file, when running in Actions.
//! - **Perfetto trace** (`--trace`) — a Chrome Trace Event Format file with
//!   test/step/LLM-call spans, openable in `ui.perfetto.dev`.

use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::costs::UsageSnapshot;
use crate::events::{StepStatus, TestEvent};

/// Console verbosity level. Higher = more output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Only failures.
    Error = 0,
    /// Failures, warnings, and the run summary.
    Warn = 1,
    /// Default: plus test and step results.
    Info = 2,
    /// Plus LLM call details and step starts.
    Debug = 3,
    /// Everything.
    Trace = 4,
}

impl Level {
    /// Maps `-q`/`-v` flag counts to a level. Quiet wins over verbose.
    #[must_use]
    pub const fn from_flags(quiet: u8, verbose: u8) -> Self {
        match (quiet, verbose) {
            (0, 0) => Self::Info,
            (0, 1) => Self::Debug,
            (0, _) => Self::Trace,
            (1, _) => Self::Warn,
            _ => Self::Error,
        }
    }

    fn shows(self, level: Self) -> bool {
        self >= level
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Level {
    fn default() -> Self {
        Self::Info
    }
}

/// Color output mode for the console sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorMode {
    /// Use colors when stderr is a TTY and `NO_COLOR` is unset.
    #[default]
    Auto,
    /// Always emit ANSI colors.
    Always,
    /// Never emit ANSI colors.
    Never,
}

/// Decides whether the console sink should emit ANSI codes.
#[must_use]
pub fn detect_color(mode: ColorMode) -> bool {
    match mode {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => std::env::var_os("NO_COLOR").is_none() && io::stderr().is_terminal(),
    }
}

/// Minimal ANSI palette; every method degrades to plain text when disabled.
#[derive(Clone, Copy)]
struct Palette {
    enabled: bool,
}

impl Palette {
    fn paint(self, s: &str, code: &str) -> String {
        if self.enabled {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_owned()
        }
    }

    fn green(self, s: &str) -> String {
        self.paint(s, "32")
    }

    fn red(self, s: &str) -> String {
        self.paint(s, "31")
    }

    fn yellow(self, s: &str) -> String {
        self.paint(s, "33")
    }

    fn cyan(self, s: &str) -> String {
        self.paint(s, "36")
    }

    fn dim(self, s: &str) -> String {
        self.paint(s, "2")
    }
}

/// Aggregated run counts captured from the final `RunFinished` event.
#[derive(Clone, Copy)]
struct RunSummary {
    tests_passed: u32,
    tests_failed: u32,
    steps_passed: u32,
    steps_failed: u32,
    steps_skipped: u32,
    total_cost: f64,
    total_tokens: u64,
    total_calls: u64,
}

/// One test's `JUnit` result, accumulated while events stream in.
struct JunitCase {
    name: String,
    duration_ms: u64,
    failures: Vec<String>,
    skipped: u32,
}

impl JunitCase {
    const fn new(name: String) -> Self {
        Self {
            name,
            duration_ms: 0,
            failures: Vec::new(),
            skipped: 0,
        }
    }
}

/// An open (unfinished) span in the Perfetto trace.
struct OpenSpan {
    kind: u8,
    name: String,
    start: Instant,
}

/// Mutable reporter state, serialized behind a mutex so the reporter can be
/// shared between the runner and the CLI.
struct ReporterState {
    jsonl: Option<BufWriter<File>>,
    junit_path: Option<PathBuf>,
    junit_cases: Vec<JunitCase>,
    trace_path: Option<PathBuf>,
    trace_events: Vec<serde_json::Value>,
    trace_open: Vec<OpenSpan>,
    trace_base: Option<Instant>,
}

/// Locks a mutex, recovering from a poisoned state instead of panicking.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Fans run events out to every enabled sink.
pub struct Reporter {
    level: Level,
    palette: Palette,
    github_annotations: bool,
    github_summary: Mutex<Option<RunSummary>>,
    state: Mutex<ReporterState>,
}

impl Default for Reporter {
    fn default() -> Self {
        Self::new(Level::Info, ColorMode::Auto, None, None, None, false)
            .expect("default reporter has no files to open")
    }
}

impl Reporter {
    /// Creates a reporter with the given console settings and optional
    /// output files. Fails if a log/JUnit/trace file cannot be opened.
    ///
    /// # Errors
    ///
    /// Returns an io error when any configured output file cannot be created.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        level: Level,
        color: ColorMode,
        log_file: Option<&Path>,
        junit_file: Option<&Path>,
        trace_file: Option<&Path>,
        github_annotations: bool,
    ) -> io::Result<Self> {
        let jsonl = match log_file {
            Some(path) => Some(BufWriter::new(File::create(path)?)),
            None => None,
        };
        Ok(Self {
            level,
            palette: Palette {
                enabled: detect_color(color),
            },
            github_annotations,
            github_summary: Mutex::new(None),
            state: Mutex::new(ReporterState {
                jsonl,
                junit_path: junit_file.map(Path::to_path_buf),
                junit_cases: Vec::new(),
                trace_path: trace_file.map(Path::to_path_buf),
                trace_events: Vec::new(),
                trace_open: Vec::new(),
                trace_base: None,
            }),
        })
    }

    /// Emits one event to every enabled sink.
    ///
    /// # Errors
    ///
    /// Returns an io error when writing the NDJSON log file fails.
    pub fn emit(&self, event: &TestEvent) -> io::Result<()> {
        let (level, text) = format_event(event, self.palette);
        if self.level.shows(level) {
            let _ = writeln!(io::stderr().lock(), "{text}");
        }
        self.write_jsonl(event)?;
        self.track_trace(event);
        self.track_junit(event);
        self.track_github(event);
        Ok(())
    }

    /// Writes a plain informational line if the current level allows it.
    pub fn info(&self, msg: impl AsRef<str>) {
        self.line(Level::Info, msg.as_ref());
    }

    /// Writes a debug line if `-v` (or higher) was passed.
    pub fn debug(&self, msg: impl AsRef<str>) {
        self.line(Level::Debug, msg.as_ref());
    }

    /// Writes a warning line; always shown unless `-qq` was passed.
    pub fn warn(&self, msg: impl AsRef<str>) {
        self.line(Level::Warn, &format!("  ! {}", msg.as_ref()));
    }

    /// Writes an error line; always shown.
    pub fn error(&self, msg: impl AsRef<str>) {
        let text = self.palette.red(&format!("  ✗ {}", msg.as_ref()));
        self.line(Level::Error, &text);
    }

    fn line(&self, level: Level, text: &str) {
        if self.level.shows(level) {
            let _ = writeln!(io::stderr().lock(), "{text}");
        }
    }

    /// Flushes the `NDJSON` log and writes `JUnit`, trace, and `GitHub`
    /// summary files. Call once after the run finishes.
    ///
    /// # Errors
    ///
    /// Returns an io error when flushing the log or writing any report file
    /// fails.
    #[allow(clippy::significant_drop_tightening)]
    pub fn finish(&self) -> io::Result<()> {
        let mut state = lock(&self.state);
        if let Some(writer) = state.jsonl.as_mut() {
            writer.flush()?;
        }
        Self::write_junit(&state)?;
        Self::write_trace(&state)?;
        self.write_github_summary();
        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)]
    fn write_jsonl(&self, event: &TestEvent) -> io::Result<()> {
        let mut state = lock(&self.state);
        let Some(writer) = state.jsonl.as_mut() else {
            return Ok(());
        };
        let mut value = serde_json::to_value(event).map_err(io::Error::other)?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert("ts".to_owned(), json!(now_ms()));
        }
        serde_json::to_writer(&mut *writer, &value).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
        Ok(())
    }

    // ── Perfetto trace sink ──────────────────────────────────────────────

    #[allow(clippy::significant_drop_tightening)]
    fn track_trace(&self, event: &TestEvent) {
        let mut state = lock(&self.state);
        if state.trace_path.is_none() {
            return;
        }
        state.trace_base.get_or_insert_with(Instant::now);
        match event {
            TestEvent::TestStarted { test } => {
                state.trace_open.push(OpenSpan {
                    kind: 0,
                    name: test.clone(),
                    start: Instant::now(),
                });
            }
            TestEvent::StepStarted { label, .. } => {
                state.trace_open.push(OpenSpan {
                    kind: 1,
                    name: label.clone(),
                    start: Instant::now(),
                });
            }
            TestEvent::LlmCallStarted {
                endpoint,
                model,
                purpose,
                ..
            } => {
                state.trace_open.push(OpenSpan {
                    kind: 2,
                    name: format!("llm {purpose}: {endpoint} ({model})"),
                    start: Instant::now(),
                });
            }
            TestEvent::TestFinished {
                test,
                passed,
                failed,
                skipped,
                ..
            } => {
                close_span(
                    &mut state,
                    0,
                    &json!({"test": test, "passed": passed, "failed": failed, "skipped": skipped}),
                );
            }
            TestEvent::StepFinished { status, .. } => {
                close_span(&mut state, 1, &json!({"status": status}));
            }
            TestEvent::LlmCallFinished {
                ok,
                input_tokens,
                output_tokens,
                cost,
                error,
                ..
            } => {
                let args = json!({"ok": ok, "in_tokens": input_tokens, "out_tokens": output_tokens, "cost": cost, "error": error});
                close_span(&mut state, 2, &args);
            }
            _ => {}
        }
    }

    fn write_trace(state: &ReporterState) -> io::Result<()> {
        let Some(path) = state.trace_path.as_deref() else {
            return Ok(());
        };
        let doc = json!({
            "traceEvents": state.trace_events,
            "displayTimeUnit": "ms",
        });
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &doc).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
        writer.flush()
    }

    // ── JUnit sink ───────────────────────────────────────────────────────

    #[allow(clippy::significant_drop_tightening)]
    fn track_junit(&self, event: &TestEvent) {
        let mut state = lock(&self.state);
        if state.junit_path.is_none() {
            return;
        }
        match event {
            TestEvent::StepFinished {
                test,
                label,
                status,
                message,
                diagnostics,
                screenshot,
                ..
            } => {
                let case = locate_case(&mut state, test);
                match status {
                    StepStatus::Failed => {
                        let mut text = format!("{label}: {message}");
                        if let Some(diag) = diagnostics {
                            text.push('\n');
                            text.push_str(diag);
                        }
                        if let Some(shot) = screenshot {
                            use std::fmt::Write as _;
                            let _ = write!(text, "\nscreenshot: {shot}");
                        }
                        case.failures.push(text);
                    }
                    StepStatus::Skipped => case.skipped += 1,
                    StepStatus::Passed => {}
                }
            }
            TestEvent::TestFinished {
                test, duration_ms, ..
            } => {
                let case = locate_case(&mut state, test);
                case.duration_ms = *duration_ms;
            }
            _ => {}
        }
    }

    fn write_junit(state: &ReporterState) -> io::Result<()> {
        let Some(path) = state.junit_path.as_deref() else {
            return Ok(());
        };
        write_junit_xml(&state.junit_cases, "llm-browser-testkit", path)
    }

    // ── GitHub sink ──────────────────────────────────────────────────────

    fn track_github(&self, event: &TestEvent) {
        if !self.github_annotations {
            return;
        }
        match event {
            TestEvent::StepFinished {
                test,
                label,
                status,
                message,
                screenshot,
                ..
            } => {
                if *status == StepStatus::Failed {
                    let mut props = Vec::new();
                    if let Some(shot) = screenshot {
                        props.push(format!("file={}", escape_property(shot)));
                    }
                    props.push(format!(
                        "title={}:{}",
                        escape_property(test),
                        escape_property(label)
                    ));
                    let _ = writeln!(
                        io::stdout().lock(),
                        "::error {}::{}",
                        props.join(","),
                        escape_data(message)
                    );
                }
            }
            TestEvent::RunFinished {
                tests_passed,
                tests_failed,
                steps_passed,
                steps_failed,
                steps_skipped,
                total_cost,
                total_tokens,
                total_calls,
            } => {
                *lock(&self.github_summary) = Some(RunSummary {
                    tests_passed: *tests_passed,
                    tests_failed: *tests_failed,
                    steps_passed: *steps_passed,
                    steps_failed: *steps_failed,
                    steps_skipped: *steps_skipped,
                    total_cost: *total_cost,
                    total_tokens: *total_tokens,
                    total_calls: *total_calls,
                });
            }
            _ => {}
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    fn write_github_summary(&self) {
        let summary = lock(&self.github_summary);
        let Some(summary) = *summary else {
            return;
        };
        let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") else {
            return;
        };
        let Ok(mut file) = File::options().append(true).open(path) else {
            return;
        };
        let verdict = if summary.tests_failed == 0 {
            "✅ all tests passed"
        } else {
            "❌ some tests failed"
        };
        let _ = writeln!(file, "## llm-browser-testkit run: {verdict}\n");
        let _ = writeln!(
            file,
            "- tests: {} passed, {} failed",
            summary.tests_passed, summary.tests_failed
        );
        let _ = writeln!(
            file,
            "- steps: {} passed, {} failed, {} skipped",
            summary.steps_passed, summary.steps_failed, summary.steps_skipped
        );
        let _ = writeln!(
            file,
            "- cost: ${:.4} | tokens: {} | calls: {}",
            summary.total_cost, summary.total_tokens, summary.total_calls
        );
    }
}

/// Finds the `JUnit` case for a test, creating it on first sight.
fn locate_case<'a>(state: &'a mut ReporterState, test: &str) -> &'a mut JunitCase {
    let idx = state
        .junit_cases
        .iter()
        .rposition(|c| c.name == test)
        .unwrap_or_else(|| {
            state.junit_cases.push(JunitCase::new(test.to_owned()));
            state.junit_cases.len() - 1
        });
    &mut state.junit_cases[idx]
}

/// Closes the most recent open span of the given kind into a complete trace
/// event (`ph: "X"`).
#[allow(clippy::cast_possible_truncation)]
fn close_span(state: &mut ReporterState, kind: u8, args: &serde_json::Value) {
    let Some(idx) = state.trace_open.iter().rposition(|s| s.kind == kind) else {
        return;
    };
    let span = state.trace_open.remove(idx);
    let base = state
        .trace_base
        .expect("trace base set before any span opens");
    let start_ts = if span.start >= base {
        span.start.duration_since(base).as_micros() as u64
    } else {
        0
    };
    let dur_us = span.start.elapsed().as_micros() as u64;
    let cat = match kind {
        0 => "test",
        1 => "step",
        _ => "llm",
    };
    state.trace_events.push(json!({
        "name": span.name,
        "ph": "X",
        "ts": start_ts,
        "dur": dur_us,
        "pid": 1,
        "tid": 1,
        "cat": cat,
        "args": args,
    }));
}

/// Serializes one event into its console text and required level.
#[must_use]
#[allow(clippy::too_many_lines)]
fn format_event(event: &TestEvent, palette: Palette) -> (Level, String) {
    match event {
        TestEvent::RunStarted { total_tests } => {
            (Level::Debug, format!("run started: {total_tests} test(s)"))
        }
        TestEvent::TestStarted { test } => (Level::Info, palette.cyan(&format!("Test: {test}"))),
        TestEvent::StepStarted { label, .. } => (Level::Debug, format!("  - {label}")),
        TestEvent::StepFinished {
            label,
            message,
            duration_ms,
            status,
            diagnostics,
            screenshot,
            ..
        } => {
            let level = match status {
                StepStatus::Failed => Level::Error,
                StepStatus::Skipped => Level::Warn,
                StepStatus::Passed => Level::Info,
            };
            let icon = match status {
                StepStatus::Passed => palette.green("✓"),
                StepStatus::Failed => palette.red("✗"),
                StepStatus::Skipped => palette.yellow("–"),
            };
            let dur = palette.dim(&format_duration(*duration_ms));
            let mut lines = vec![format!("  {icon} {label} — {message} ({dur})")];
            if let Some(diag) = diagnostics {
                lines.push(diag.clone());
            }
            if let Some(shot) = screenshot {
                lines.push(format!("      screenshot: {shot}"));
            }
            (level, lines.join("\n"))
        }
        TestEvent::LlmCallStarted {
            endpoint,
            model,
            purpose,
            ..
        } => (
            Level::Debug,
            format!("      llm({purpose}): {endpoint} ({model}) ..."),
        ),
        TestEvent::LlmCallFinished {
            endpoint,
            model,
            purpose,
            ok,
            duration_ms,
            input_tokens,
            output_tokens,
            cost,
            error,
            ..
        } => {
            let status = if *ok {
                palette.green("ok")
            } else {
                palette.red("failed")
            };
            let dur = palette.dim(&format_duration(*duration_ms));
            let mut line = format!(
                "      llm({purpose}): {endpoint} ({model}) {status} {dur} | \
                 {input_tokens} in / {output_tokens} out | ${cost:.4}"
            );
            if let Some(err) = error {
                use std::fmt::Write as _;
                let _ = write!(line, " — {err}");
            }
            let level = if *ok { Level::Debug } else { Level::Warn };
            (level, line)
        }
        TestEvent::TestFinished {
            test,
            passed,
            failed,
            skipped,
            duration_ms,
            cost,
            tokens,
            calls,
        } => {
            let verdict = if *failed == 0 && *passed > 0 {
                palette.green("passed")
            } else {
                palette.red("failed")
            };
            let dur = palette.dim(&format_duration(*duration_ms));
            (
                Level::Info,
                format!(
                    "Test: {test} — {verdict} ({dur}, ${cost:.4}, {tokens} tokens, \
                     {calls} calls, {passed}+{failed}+{skipped} steps)"
                ),
            )
        }
        TestEvent::RunFinished {
            tests_passed,
            tests_failed,
            steps_passed,
            steps_failed,
            steps_skipped,
            total_cost,
            total_tokens,
            total_calls,
        } => {
            let verdict = if *tests_failed == 0 {
                palette.green("passed")
            } else {
                palette.red("failed")
            };
            (
                Level::Warn,
                format!(
                    "run {verdict}: tests {tests_passed} passed, {tests_failed} failed | \
                     steps {steps_passed} passed, {steps_failed} failed, \
                     {steps_skipped} skipped | ${total_cost:.4} | {total_tokens} tokens | \
                     {total_calls} calls"
                ),
            )
        }
        TestEvent::Warning { message } => (Level::Warn, palette.yellow(&format!("  ! {message}"))),
    }
}

/// Human-friendly duration: `431ms` or `1.2s`.
fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let secs = ms as f64 / 1000.0;
        format!("{secs:.1}s")
    }
}

/// Epoch milliseconds, for the NDJSON `ts` field.
#[allow(clippy::cast_possible_truncation)]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Escapes a string for the data section of a GitHub workflow command.
#[must_use]
fn escape_data(s: &str) -> String {
    s.replace('%', "%25")
        .replace('\r', "%0D")
        .replace('\n', "%0A")
}

/// Escapes a string for the property section of a GitHub workflow command.
#[must_use]
fn escape_property(s: &str) -> String {
    s.replace('%', "%25")
        .replace('\r', "%0D")
        .replace('\n', "%0A")
        .replace(':', "%3A")
        .replace(',', "%2C")
}

/// Escapes a string for use in an XML attribute value.
#[must_use]
fn esc_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Writes the accumulated cases as a `JUnit` XML document.
fn write_junit_xml(cases: &[JunitCase], classname: &str, path: &Path) -> io::Result<()> {
    let tests = cases.len();
    let failures: usize = cases.iter().map(|c| c.failures.len()).sum();
    let skipped: u32 = cases.iter().map(|c| c.skipped).sum();
    #[allow(clippy::cast_precision_loss)]
    let time: f64 = cases.iter().map(|c| c.duration_ms as f64 / 1000.0).sum();

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")?;
    writeln!(
        writer,
        "<testsuites tests=\"{tests}\" failures=\"{failures}\" skipped=\"{skipped}\" \
         time=\"{time:.3}\">"
    )?;
    for case in cases {
        let name = esc_attr(&case.name);
        #[allow(clippy::cast_precision_loss)]
        let case_time = case.duration_ms as f64 / 1000.0;
        if case.failures.is_empty() && case.skipped == 0 {
            writeln!(
                writer,
                "  <testcase classname=\"{classname}\" name=\"{name}\" time=\"{case_time:.3}\"/>"
            )?;
        } else {
            writeln!(
                writer,
                "  <testcase classname=\"{classname}\" name=\"{name}\" time=\"{case_time:.3}\">"
            )?;
            for _ in 0..case.skipped {
                writeln!(writer, "    <skipped/>")?;
            }
            for failure in &case.failures {
                write!(writer, "    <failure message=\"{}\">", esc_attr(failure))?;
                writer.write_all(b"<![CDATA[")?;
                writer.write_all(failure.replace("]]>", "]]]]><![CDATA[>").as_bytes())?;
                writer.write_all(b"]]></failure>\n")?;
            }
            writeln!(writer, "  </testcase>")?;
        }
    }
    writeln!(writer, "</testsuites>")?;
    writer.flush()
}

/// Prints a cost report to stderr after all tests complete.
pub fn print_report(per_test: &[(String, UsageSnapshot)], global: &UsageSnapshot) {
    if per_test.is_empty() {
        return;
    }

    eprintln!();
    eprintln!("-------------------------------");
    eprintln!("  COST REPORT");
    eprintln!("-------------------------------");

    for (test_name, snapshot) in per_test {
        eprintln!(
            "  Test: \"{test_name}\" — ${cost:.4} | {tokens} tokens | {calls} calls",
            cost = snapshot.total_cost,
            tokens = snapshot.total_tokens,
            calls = snapshot.total_calls,
        );
        for (ep_name, ep_usage) in &snapshot.endpoints {
            if ep_usage.calls == 0 {
                continue;
            }
            eprintln!(
                "    endpoint.{ep_name}:   {calls:>3} calls, {tokens:>7} tokens, ${cost:.4}",
                calls = ep_usage.calls,
                tokens = ep_usage.input_tokens + ep_usage.output_tokens,
                cost = ep_usage.cost,
            );
        }
    }

    eprintln!("-------------------------------");
    eprintln!("  GLOBAL SUMMARY");
    eprintln!("    Total cost:     ${cost:.4}", cost = global.total_cost);
    eprintln!("    Total tokens:   {tokens}", tokens = global.total_tokens);
    eprintln!("    Total calls:    {calls}", calls = global.total_calls);
    eprintln!("-------------------------------");
}

/// Prints a budget exceeded warning to stderr.
pub fn print_budget_warning(message: &str) {
    eprintln!("  ! BUDGET WARNING: {message}");
}

/// Prints a budget exceeded hard error to stderr.
pub fn print_budget_error(message: &str) {
    eprintln!("  ✗ BUDGET EXCEEDED: {message}");
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::path::PathBuf;

    use super::{
        escape_data, escape_property, format_duration, format_event, print_report, ColorMode,
        Level, Palette, Reporter,
    };
    use crate::costs::{EndpointUsage, UsageSnapshot};
    use crate::events::{StepStatus, TestEvent};
    use std::collections::HashMap;

    fn palette() -> Palette {
        Palette { enabled: false }
    }

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lbt-report-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    fn read_file(path: &PathBuf) -> String {
        let mut s = String::new();
        let mut f = std::fs::File::open(path).expect("open report file");
        f.read_to_string(&mut s).expect("read report file");
        s
    }

    fn step_event(status: StepStatus) -> TestEvent {
        TestEvent::StepFinished {
            test: "login".into(),
            index: 0,
            label: "[click] sign in".into(),
            status,
            duration_ms: 1200,
            message: if status == StepStatus::Failed {
                "element #btn not found".into()
            } else {
                "clicked #btn".into()
            },
            diagnostics: (status == StepStatus::Failed).then(|| "    | url: http://x".into()),
            screenshot: (status == StepStatus::Failed).then(|| "artifacts/login.png".into()),
        }
    }

    #[test]
    fn test_level_from_flags() {
        assert_eq!(Level::from_flags(0, 0), Level::Info);
        assert_eq!(Level::from_flags(0, 1), Level::Debug);
        assert_eq!(Level::from_flags(0, 2), Level::Trace);
        assert_eq!(Level::from_flags(0, 9), Level::Trace);
        assert_eq!(Level::from_flags(1, 0), Level::Warn);
        assert_eq!(Level::from_flags(1, 5), Level::Warn, "quiet wins");
        assert_eq!(Level::from_flags(2, 0), Level::Error);
        assert_eq!(Level::from_flags(3, 0), Level::Error);
    }

    #[test]
    fn test_format_step_finished_failed_includes_diagnostics() {
        let (level, text) = format_event(&step_event(StepStatus::Failed), palette());
        assert_eq!(level, Level::Error);
        assert!(text.contains("[click] sign in"));
        assert!(text.contains("element #btn not found"));
        assert!(text.contains("1.2s"));
        assert!(text.contains("| url: http://x"));
        assert!(text.contains("screenshot: artifacts/login.png"));
        assert!(!text.contains("\x1b["), "no ANSI codes when color disabled");
    }

    #[test]
    fn test_format_step_finished_passed_level_info() {
        let (level, text) = format_event(&step_event(StepStatus::Passed), palette());
        assert_eq!(level, Level::Info);
        assert!(text.contains("clicked #btn"));
        assert!(!text.contains("screenshot:"));
    }

    #[test]
    fn test_format_test_finished_verdict() {
        let ok = format_event(
            &TestEvent::TestFinished {
                test: "t1".into(),
                passed: 3,
                failed: 0,
                skipped: 0,
                duration_ms: 6100,
                cost: 0.0123,
                tokens: 1234,
                calls: 4,
            },
            palette(),
        );
        assert_eq!(ok.0, Level::Info);
        assert!(ok.1.contains("passed"));

        let bad = format_event(
            &TestEvent::TestFinished {
                test: "t2".into(),
                passed: 1,
                failed: 1,
                skipped: 2,
                duration_ms: 6100,
                cost: 0.0123,
                tokens: 1234,
                calls: 4,
            },
            palette(),
        );
        assert!(bad.1.contains("failed"));
    }

    #[test]
    fn test_format_llm_call_lines() {
        let started = format_event(
            &TestEvent::LlmCallStarted {
                test: "t".into(),
                index: 0,
                endpoint: "default".into(),
                model: "deepseek".into(),
                purpose: "targeting".into(),
            },
            palette(),
        );
        assert_eq!(started.0, Level::Debug);
        assert!(started.1.contains("llm(targeting): default (deepseek)"));

        let failed = format_event(
            &TestEvent::LlmCallFinished {
                test: "t".into(),
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
            },
            palette(),
        );
        assert_eq!(failed.0, Level::Warn);
        assert!(failed.1.contains("failed"));
        assert!(failed.1.contains("HTTP 429"));
    }

    #[test]
    fn test_format_run_started_is_debug() {
        let (level, text) = format_event(&TestEvent::RunStarted { total_tests: 3 }, palette());
        assert_eq!(level, Level::Debug);
        assert!(text.contains("3 test(s)"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0), "0ms");
        assert_eq!(format_duration(431), "431ms");
        assert_eq!(format_duration(1000), "1.0s");
        assert_eq!(format_duration(1234), "1.2s");
    }

    #[test]
    fn test_github_escaping() {
        assert_eq!(escape_data("a% b\nc\rd"), "a%25 b%0Ac%0Dd");
        assert_eq!(escape_property("a:b,c\n%d"), "a%3Ab%2Cc%0A%25d");
    }

    #[test]
    fn test_reporter_jsonl_emits_typed_timestamped_lines() {
        let path = temp_file("run.jsonl");
        let reporter = Reporter::new(
            Level::Debug,
            ColorMode::Never,
            Some(&path),
            None,
            None,
            false,
        )
        .expect("open jsonl");
        reporter
            .emit(&TestEvent::RunStarted { total_tests: 1 })
            .expect("emit");
        reporter
            .emit(&TestEvent::StepStarted {
                test: "login".into(),
                index: 0,
                label: "[click] sign in".into(),
            })
            .expect("emit");
        reporter
            .emit(&step_event(StepStatus::Passed))
            .expect("emit");
        reporter
            .emit(&TestEvent::TestFinished {
                test: "login".into(),
                passed: 1,
                failed: 0,
                skipped: 0,
                duration_ms: 1200,
                cost: 0.001,
                tokens: 100,
                calls: 1,
            })
            .expect("emit");
        reporter
            .emit(&TestEvent::RunFinished {
                tests_passed: 1,
                tests_failed: 0,
                steps_passed: 1,
                steps_failed: 0,
                steps_skipped: 0,
                total_cost: 0.001,
                total_tokens: 100,
                total_calls: 1,
            })
            .expect("emit");
        reporter.finish().expect("finish");

        let content = read_file(&path);
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5);
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("json line");
        assert_eq!(first["type"], "run_started");
        assert!(first["ts"].as_u64().expect("ts") > 0);
        let step: serde_json::Value = serde_json::from_str(lines[2]).expect("json line");
        assert_eq!(step["type"], "step_finished");
        assert_eq!(step["status"], "passed");
        assert_eq!(step["test"], "login");
        let finished: serde_json::Value = serde_json::from_str(lines[4]).expect("json line");
        assert_eq!(finished["type"], "run_finished");
        assert_eq!(finished["tests_passed"], 1);
    }

    #[test]
    fn test_reporter_junit_creates_failure_elements() {
        let path = temp_file("report.xml");
        let reporter = Reporter::new(
            Level::Info,
            ColorMode::Never,
            None,
            Some(&path),
            None,
            false,
        )
        .expect("open junit");
        reporter
            .emit(&step_event(StepStatus::Failed))
            .expect("emit");
        reporter
            .emit(&step_event(StepStatus::Skipped))
            .expect("emit");
        reporter
            .emit(&TestEvent::TestFinished {
                test: "login".into(),
                passed: 0,
                failed: 1,
                skipped: 1,
                duration_ms: 5000,
                cost: 0.0,
                tokens: 0,
                calls: 0,
            })
            .expect("emit");
        reporter.finish().expect("finish");

        let xml = read_file(&path);
        assert!(xml.starts_with("<?xml version=\"1.0\""));
        assert!(xml.contains("<testsuites tests=\"1\" failures=\"1\" skipped=\"1\""));
        assert!(xml.contains("name=\"login\""));
        assert!(xml.contains("<skipped/>"));
        assert!(xml.contains("<failure"));
        assert!(xml.contains("element #btn not found"));
        assert!(xml.contains("screenshot: artifacts/login.png"));
        // XML safety: a failure message with angle brackets must be escaped
        let weird = TestEvent::StepFinished {
            test: "weird".into(),
            index: 0,
            label: "[wait] <weird>".into(),
            status: StepStatus::Failed,
            duration_ms: 10,
            message: "boom <tag> & \"quote\"".into(),
            diagnostics: None,
            screenshot: None,
        };
        reporter.emit(&weird).expect("emit");
        reporter
            .emit(&TestEvent::TestFinished {
                test: "weird".into(),
                passed: 0,
                failed: 1,
                skipped: 0,
                duration_ms: 10,
                cost: 0.0,
                tokens: 0,
                calls: 0,
            })
            .expect("emit");
        reporter.finish().expect("finish");
        let xml2 = read_file(&path);
        // Attribute copies are XML-escaped...
        assert!(xml2.contains("&lt;weird&gt;") && xml2.contains("&amp; &quot;quote&quot;"));
        // ...while the CDATA body keeps the raw text.
        assert!(xml2.contains("[wait] <weird>: boom <tag> & \"quote\""));
    }

    #[test]
    fn test_reporter_trace_emits_complete_spans() {
        let path = temp_file("trace.json");
        let reporter = Reporter::new(
            Level::Info,
            ColorMode::Never,
            None,
            None,
            Some(&path),
            false,
        )
        .expect("open trace");
        reporter
            .emit(&TestEvent::TestStarted {
                test: "login".into(),
            })
            .expect("emit");
        reporter
            .emit(&TestEvent::StepStarted {
                test: "login".into(),
                index: 0,
                label: "[click] sign in".into(),
            })
            .expect("emit");
        reporter
            .emit(&step_event(StepStatus::Passed))
            .expect("emit");
        reporter
            .emit(&TestEvent::TestFinished {
                test: "login".into(),
                passed: 1,
                failed: 0,
                skipped: 0,
                duration_ms: 1200,
                cost: 0.0,
                tokens: 0,
                calls: 0,
            })
            .expect("emit");
        reporter.finish().expect("finish");

        let content = read_file(&path);
        let doc: serde_json::Value = serde_json::from_str(&content).expect("trace json");
        let events = doc["traceEvents"].as_array().expect("traceEvents array");
        assert_eq!(events.len(), 2);
        // Spans close inner-first, so the array starts with the step.
        let step = &events[0];
        assert_eq!(step["name"], "[click] sign in");
        assert_eq!(step["ph"], "X");
        assert_eq!(step["cat"], "step");
        assert_eq!(events[1]["name"], "login");
        assert_eq!(events[1]["cat"], "test");
        assert!(step["ts"].as_u64().expect("ts") > 0);
        assert_eq!(doc["displayTimeUnit"], "ms");
    }

    #[test]
    fn test_print_report_empty() {
        // Should return early, no panic
        print_report(&[], &UsageSnapshot::default());
    }

    #[test]
    fn test_print_report_single_test() {
        let per_test = vec![("test1".to_owned(), make_snapshot(0.05, 500, 3))];
        // Should not panic
        print_report(&per_test, &make_snapshot(0.05, 500, 3));
    }

    fn make_snapshot(cost: f64, tokens: u64, calls: u64) -> UsageSnapshot {
        let mut eps = HashMap::new();
        eps.insert(
            "default".to_owned(),
            EndpointUsage {
                calls,
                input_tokens: tokens / 2,
                output_tokens: tokens / 2,
                cost,
            },
        );
        UsageSnapshot::from_endpoints(&eps)
    }
}
