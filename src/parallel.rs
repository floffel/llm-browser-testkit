//! Concurrent execution of multiple scenario files.
//!
//! A single `llm-browser-testkit run a.toml b.toml c.toml` invocation runs
//! each file on its own **isolated browser** (separate Chrome process, so
//! cookies, localStorage, and other session state can never leak between
//! files). Each file's own tests always run sequentially on that file's
//! browser.
//!
//! # Concurrency control
//!
//! - `[config] concurrency_group = "name"` makes every file that declares
//!   the same group mutually exclusive: they never execute at the same
//!   time. A file without a group gets its own implicit group, so distinct
//!   files run in parallel by default.
//! - Concurrency is bounded by a [`ParallelMode`]: an exact manual count
//!   (`--parallel N`) or an **auto** mode that adapts to the machine.
//!
//! # Auto scaling
//!
//! In auto mode the orchestrator never assumes a fixed per-browser memory
//! footprint. It instead **learns** it by trial and error:
//!
//! - it probes available system memory once and keeps a running estimate of
//!   how many bytes one browser actually holds (`available_before` −
//!   `available_after` around each run), clamped to sane bounds;
//! - after each successful file it raises the concurrency limit toward
//!   `capacity = available / learned_footprint`, and otherwise ramps up one
//!   at a time;
//! - it **guards** launches: a worker will not start a new browser while
//!   less than an absolute [`MIN_HEADROOM_BYTES`] is free, or there is no
//!   room for one more browser of the learned footprint. It never uses a
//!   fraction of `total` memory, so it stays correct in VMs/containers
//!   where `total` reports the host's much larger RAM;
//! - when a launch fails with an out-of-memory-style error it **halves the
//!   limit** and **retries the file** with backoff, up to
//!   [`MAX_LAUNCH_RETRIES`].
//!
//! Bounds come from `--parallel-min` / `--parallel-max` (default `1` /
//! unlimited). When system memory cannot be measured the auto ceiling falls
//! back to a conservative [`DEFAULT_AUTO_CEILING`].

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;

use crate::costs::{EndpointUsage, UsageSnapshot};
use crate::events::TestEvent;
use crate::reporting::Reporter;
use crate::runner::{RunReport, ScenarioRunner};
use crate::scenario::{AssertDefinition, ScenarioConfig, TestGroup};

/// Worker-thread cap used when `--parallel-max` is unlimited, so we never
/// spawn an unbounded number of threads. The adaptive limit still throttles
/// the number of *running browsers* below this, and the memory guard /
/// learned footprint stop it before the machine is saturated.
const DEFAULT_THREAD_CAP: usize = 64;

/// Fallback auto ceiling when system memory cannot be measured (roughly "a
/// handful of browsers", so the default is genuinely parallel), overridable
/// with `--parallel-max`.
const DEFAULT_AUTO_CEILING: usize = 8;

/// Absolute memory floor (bytes) kept free before new browsers are gated.
/// Deliberately NOT a fraction of `total` memory: on VMs/containers `total`
/// often reports the host's RAM (hundreds of GB), so a percentage guard
/// would never fire — or a small cgroup limit would be blocked forever
/// against the huge host figure. An absolute floor + "room for one learned
/// browser" is what actually prevents Chrome from `OOM`-ing the process.
const MIN_HEADROOM_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

/// How many times a memory-style browser-launch failure is retried before
/// the file is reported as failed.
const MAX_LAUNCH_RETRIES: u32 = 3;

/// Base and per-retry step for the backoff before a retried launch.
const BACKOFF_BASE_MS: u64 = 600;
const BACKOFF_STEP_MS: u64 = 600;
const MAX_BACKOFF_MS: u64 = 4000;

/// Bounds for the learned per-browser footprint estimate (bytes), rejecting
/// noise from pages that balloon or transient dips.
const MIN_FOOTPRINT: f64 = 128.0 * 1024.0 * 1024.0; // 128 MiB
const MAX_FOOTPRINT: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0; // 2 GiB

/// How concurrency is chosen for a batch of files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelMode {
    /// Run at exactly this many files concurrently (`--parallel N`).
    Manual(usize),
    /// Adapt to available memory and learned browser footprint, never
    /// exceeding `max` (0 = unlimited) or dropping below `min`.
    Auto {
        /// Hard floor for concurrency (from `--parallel-min`).
        min: u32,
        /// Hard ceiling for concurrency; `0` = unlimited (from
        /// `--parallel-max`).
        max: u32,
    },
}

/// Builds the [`ParallelMode`] from CLI flags. `--parallel N` selects manual
/// mode; otherwise auto mode with the given min/max bounds.
#[must_use]
pub fn mode_from_cli(parallel: Option<u32>, min: u32, max: u32) -> ParallelMode {
    parallel.map_or_else(
        || ParallelMode::Auto {
            min: min.max(1),
            max,
        },
        |k| ParallelMode::Manual(k.max(1) as usize),
    )
}

/// Best-effort snapshot of total / available physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Total physical memory, bytes.
    pub total: u64,
    /// Currently available (free) memory, bytes.
    pub available: u64,
}

/// Runs one scenario file, ready for the scheduler. CLI overrides applied
/// and viewport matrix expanded by the caller.
#[derive(Clone)]
pub struct ScenarioFile {
    /// Human-readable label (typically the file path) used in logs.
    pub label: String,
    /// Effective scenario configuration for this file.
    pub config: ScenarioConfig,
    /// Reusable assertion definitions from this file.
    pub definitions: Vec<AssertDefinition>,
    /// The (viewport-expanded) tests to run.
    pub tests: Vec<TestGroup>,
}

impl ScenarioFile {
    /// The concurrency key for this file: its declared `concurrency_group`,
    /// or a unique per-file key so distinct files run in parallel by
    /// default.
    fn concurrency_key(&self, index: usize) -> String {
        self.config
            .concurrency_group
            .clone()
            .unwrap_or_else(|| format!("<file {index}>"))
    }
}

/// Options controlling a parallel batch run.
#[derive(Clone)]
pub struct RunOptions {
    /// How concurrency is chosen (manual or auto-scaling).
    pub mode: ParallelMode,
    /// Shared event sink (thread-safe: every sink is mutex-guarded).
    pub reporter: Arc<Reporter>,
    /// Best-effort memory snapshot; `None` falls back to trial-and-error.
    pub memory: Option<MemoryInfo>,
}

/// Result of a parallel batch run: the merged report plus combined usage.
#[derive(Debug, Default)]
pub struct ParallelRun {
    /// Merged test/step results across all files.
    pub report: RunReport,
    /// Per-test usage snapshots across all files.
    pub per_test: Vec<(String, UsageSnapshot)>,
    /// Combined global usage across all files.
    pub global: UsageSnapshot,
}

/// Runs every file concurrently, respecting the concurrency mode and
/// per-file groups, each on its own isolated browser. Emits one
/// `RunStarted`/`RunFinished` pair for the batch and merges cost/usage.
///
/// # Errors
///
/// Returns an error only for internal scheduling failures; a file whose
/// browser cannot be launched (after retries) is recorded as a failed
/// report rather than aborting the batch.
#[allow(clippy::cast_possible_truncation, clippy::significant_drop_tightening)]
pub fn run_scenarios(files: Vec<ScenarioFile>, opts: RunOptions) -> Result<ParallelRun> {
    let reporter = opts.reporter;
    let total_tests: u32 = files.iter().map(|f| f.tests.len() as u32).sum();

    reporter.emit(&TestEvent::RunStarted { total_tests })?;

    if files.is_empty() {
        reporter.emit(&TestEvent::RunFinished {
            tests_passed: 0,
            tests_failed: 0,
            steps_passed: 0,
            steps_failed: 0,
            steps_skipped: 0,
            total_cost: 0.0,
            total_tokens: 0,
            total_calls: 0,
        })?;
        return Ok(ParallelRun::default());
    }

    let n = files.len();
    let mut results: Vec<Option<ParallelRun>> = Vec::new();
    for _ in 0..n {
        results.push(None);
    }
    let state = Arc::new(Mutex::new(SchedulerState {
        files,
        results,
        started: vec![false; n],
        active_groups: HashSet::new(),
    }));
    let gate = Arc::new(Mutex::new(make_gate(&opts.mode, opts.memory, n)));
    let workers = gate.lock().unwrap().threads;

    let mut threads: Vec<std::thread::JoinHandle<()>> = Vec::new();
    for _ in 0..workers {
        // Shadow with per-iteration clones so each closure captures its own
        // Arc (the loop variable itself must not move into the closure).
        let state = Arc::clone(&state);
        let gate = Arc::clone(&gate);
        let reporter = Arc::clone(&reporter);
        threads.push(std::thread::spawn(move || {
            worker_loop(&state, &gate, &reporter);
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }

    // Merge per-file reports + usage while holding the scheduler lock, then
    // release it (end of the block) before emitting the batch RunFinished.
    let run = {
        let st = state.lock().unwrap();
        let mut report = RunReport::default();
        let mut per_test: Vec<(String, UsageSnapshot)> = Vec::new();
        let mut globals: Vec<UsageSnapshot> = Vec::new();
        for r in &st.results {
            if let Some(run) = r.as_ref() {
                report.tests_passed += run.report.tests_passed;
                report.tests_failed += run.report.tests_failed;
                report.passed += run.report.passed;
                report.failed += run.report.failed;
                report.skipped += run.report.skipped;
                report.details.extend(run.report.details.clone());
                per_test.extend(run.per_test.clone());
                globals.push(run.global.clone());
            }
        }
        ParallelRun {
            report,
            per_test,
            global: merge_globals(&globals),
        }
    };

    reporter.emit(&TestEvent::RunFinished {
        tests_passed: run.report.tests_passed,
        tests_failed: run.report.tests_failed,
        steps_passed: run.report.passed,
        steps_failed: run.report.failed,
        steps_skipped: run.report.skipped,
        total_cost: run.global.total_cost,
        total_tokens: run.global.total_tokens,
        total_calls: run.global.total_calls,
    })?;

    Ok(run)
}

// ── Adaptive gate ─────────────────────────────────────────────────────

/// The adaptive controller shared by every worker. All fields are guarded
/// by the `Mutex` the workers lock around it.
struct Gate {
    /// Number of browsers currently running.
    active: usize,
    /// Current concurrency limit; moves in `[lower, threads]` as the
    /// controller learns the machine's memory behaviour.
    limit: usize,
    /// Hard floor for the limit (from `--parallel-min`).
    lower: usize,
    /// Number of worker threads; also the ceiling the limit may reach.
    threads: usize,
    /// Ceiling for the limit when memory is unknown (ramp target).
    ramp_ceiling: usize,
    /// Total physical memory, if probed.
    memory_total: Option<u64>,
    /// Last known available memory, if probed.
    available: Option<u64>,
    /// Learned average per-browser footprint, bytes (0 = unknown).
    footprint: f64,
    /// Number of footprint samples folded into the average.
    footprint_count: u32,
    /// Current backoff (ms) applied before retrying a failed launch.
    retry_backoff: u64,
}

/// Computes the initial gate state from the mode and (optional) memory.
fn make_gate(mode: &ParallelMode, memory: Option<MemoryInfo>, files_len: usize) -> Gate {
    match mode {
        ParallelMode::Manual(k) => {
            let k = *k;
            let threads = k.max(1).min(files_len).max(1);
            let limit = k.max(1).min(threads);
            Gate {
                active: 0,
                limit,
                lower: limit,
                threads,
                ramp_ceiling: limit,
                memory_total: memory.map(|m| m.total),
                available: memory.map(|m| m.available),
                footprint: 0.0,
                footprint_count: 0,
                retry_backoff: 0,
            }
        }
        ParallelMode::Auto { min, max } => {
            let min = *min;
            let max = *max;
            let lower = min.max(1) as usize;
            let user_max: Option<usize> = if max == 0 { None } else { Some(max as usize) };
            let threads = user_max.unwrap_or(DEFAULT_THREAD_CAP).min(files_len).max(1);
            let ramp_ceiling = user_max.unwrap_or(DEFAULT_AUTO_CEILING).min(threads).max(1);
            Gate {
                active: 0,
                limit: lower.min(threads).max(1),
                lower: lower.min(threads).max(1),
                threads,
                ramp_ceiling,
                memory_total: memory.map(|m| m.total),
                available: memory.map(|m| m.available),
                footprint: 0.0,
                footprint_count: 0,
                retry_backoff: 0,
            }
        }
    }
}

impl Gate {
    /// Whether a new browser may be launched right now: below the
    /// concurrency limit and with enough memory headroom.
    fn can_launch(&self) -> bool {
        self.active < self.limit && !self.memory_blocked()
    }

    /// True when a new browser should not be launched yet: either less than an
    /// absolute [`MIN_HEADROOM_BYTES`] is free, or there is not room for one
    /// more browser of the learned footprint. Uses absolute free bytes (never a
    /// fraction of `total`), so it stays correct inside VMs/containers where
    /// `total` may report the host's much larger RAM.
    #[allow(
        clippy::unnecessary_unwrap,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn memory_blocked(&self) -> bool {
        let Some(available) = self.available else {
            return false;
        };
        if available < MIN_HEADROOM_BYTES {
            return true;
        }
        if self.footprint_count > 0 && self.footprint > 0.0 {
            available < self.footprint as u64
        } else {
            false
        }
    }

    /// Reserves a launch slot.
    #[allow(clippy::missing_const_for_fn)]
    fn launch_started(&mut self, before: Option<u64>) {
        if before.is_some() {
            self.available = before;
        }
        self.active += 1;
    }

    /// Folds the memory delta around one run into the learned footprint and
    /// refreshes the last-known available memory.
    #[allow(
        clippy::unnecessary_unwrap,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops
    )]
    fn launch_finished(&mut self, before: Option<u64>, after: Option<u64>) {
        if before.is_some() {
            self.available = before;
        }
        if after.is_some() {
            self.available = after;
        }
        if before.is_some() && after.is_some() {
            let delta = before.unwrap().saturating_sub(after.unwrap());
            if delta > 0 {
                let d = delta as f64;
                if self.footprint_count == 0 {
                    self.footprint = d;
                } else {
                    self.footprint = self.footprint * 0.7 + d * 0.3;
                }
                self.footprint = self.footprint.clamp(MIN_FOOTPRINT, MAX_FOOTPRINT);
                self.footprint_count = (self.footprint_count + 1).min(20);
            }
        }
    }

    /// After a successful run, try to raise the limit toward the
    /// memory-derived capacity, or ramp up one at a time when memory is
    /// unknown.
    #[allow(
        clippy::unnecessary_unwrap,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn on_success(&mut self) {
        if self.footprint_count > 0 && self.footprint > 0.0 && self.memory_total.is_some() {
            let total = self.memory_total.unwrap();
            let available = self.available.unwrap_or(total);
            if available > 0 {
                let capacity = (available as f64 / self.footprint) as usize;
                self.limit = capacity.max(self.lower).min(self.threads);
                return;
            }
        }
        self.limit = (self.limit + 1).min(self.ramp_ceiling).max(self.lower);
    }

    /// After an out-of-memory-style launch failure, back off: halve the
    /// limit (never below the floor) and increase the retry backoff.
    fn on_launch_failure(&mut self) {
        self.limit = (self.limit / 2).max(self.lower);
        self.retry_backoff = (self.retry_backoff + BACKOFF_STEP_MS).min(MAX_BACKOFF_MS);
    }

    /// Frees a launch slot and refreshes known available memory.
    #[allow(clippy::missing_const_for_fn)]
    fn release(&mut self, after: Option<u64>) {
        if after.is_some() {
            self.available = after;
        }
        self.active = self.active.saturating_sub(1);
    }
}

// ── Scheduler ─────────────────────────────────────────────────────────

/// Outcome of [`SchedulerState::claim`].
enum Claim {
    /// File `usize` is assigned to this worker.
    Take(usize),
    /// Work remains but every unstarted file is blocked by a concurrency
    /// group — retry shortly.
    Wait,
    /// Every file has been started; no more work.
    Done,
}

/// Shared scheduler state, guarded by the mutex every worker locks.
struct SchedulerState {
    files: Vec<ScenarioFile>,
    results: Vec<Option<ParallelRun>>,
    started: Vec<bool>,
    active_groups: HashSet<String>,
}

impl SchedulerState {
    /// Reserves the first unstarted file whose concurrency group is not
    /// currently running, or reports that the worker should wait / that all
    /// work is done.
    fn claim(&mut self) -> Claim {
        for i in 0..self.files.len() {
            if self.started[i] {
                continue;
            }
            let key = self.files[i].concurrency_key(i);
            if self.active_groups.contains(&key) {
                continue;
            }
            self.started[i] = true;
            self.active_groups.insert(key);
            return Claim::Take(i);
        }
        if self.started.iter().all(|b| *b) {
            Claim::Done
        } else {
            Claim::Wait
        }
    }

    /// Releases file `i`'s concurrency group so blocked files can start.
    fn complete(&mut self, i: usize) {
        let key = self.files[i].concurrency_key(i);
        self.active_groups.remove(&key);
    }
}

/// Per-file run decision, computed while holding the gate.
enum Decision {
    /// Success; return the run.
    Done(ParallelRun),
    /// Memory-style failure to retry after `u64` ms.
    Retry(u64),
    /// Non-retryable failure; record as failed.
    GiveUp(String),
}

/// Worker loop: repeatedly claim a runnable file, run it on its own
/// isolated browser (gated + retried), and record the result.
#[allow(clippy::significant_drop_tightening)]
fn worker_loop(
    state: &Arc<Mutex<SchedulerState>>,
    gate: &Arc<Mutex<Gate>>,
    reporter: &Arc<Reporter>,
) {
    loop {
        let claimed = state.lock().unwrap().claim();
        match claimed {
            Claim::Done => break,
            Claim::Wait => std::thread::sleep(Duration::from_millis(25)),
            Claim::Take(i) => {
                let file = state.lock().unwrap().files[i].clone();
                let run = run_file_with_retries(&file, gate, reporter);
                let mut st = state.lock().unwrap();
                st.results[i] = Some(run);
                st.complete(i);
            }
        }
    }
}

/// Runs one file, waiting for a launch slot, and retrying memory-style
/// launch failures with backoff.
#[allow(clippy::significant_drop_tightening)]
fn run_file_with_retries(
    file: &ScenarioFile,
    gate: &Arc<Mutex<Gate>>,
    reporter: &Arc<Reporter>,
) -> ParallelRun {
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let before = available_memory_now();

        // Wait for a launch slot (concurrency limit + memory headroom). The
        // gate guard drops at the end of the inner block, before we sleep.
        {
            loop {
                let mut g = gate.lock().unwrap();
                if g.can_launch() {
                    g.launch_started(before);
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }

        let result = run_one_file(file, reporter);
        let after = available_memory_now();

        let decision = {
            let mut g = gate.lock().unwrap();
            g.launch_finished(before, after);
            match &result {
                Ok(_) => {
                    g.on_success();
                    g.release(after);
                    Decision::Done(result.unwrap())
                }
                Err(e) => {
                    let oom = is_retryable_oom(e);
                    if oom && attempt < MAX_LAUNCH_RETRIES {
                        let backoff = g.retry_backoff.max(BACKOFF_BASE_MS);
                        g.on_launch_failure();
                        g.release(after);
                        Decision::Retry(backoff)
                    } else {
                        g.on_launch_failure();
                        g.release(after);
                        Decision::GiveUp(e.clone())
                    }
                }
            }
        };

        match decision {
            Decision::Done(run) => return run,
            Decision::Retry(ms) => {
                reporter.warn(format!(
                    "{}: browser launch failed (likely out of memory); retrying ({attempt}/{MAX_LAUNCH_RETRIES}) in {ms}ms",
                    file.label,
                ));
                std::thread::sleep(Duration::from_millis(ms));
            }
            Decision::GiveUp(e) => {
                reporter.error(format!("{}: {e}", file.label));
                return synthesized_failed_report(file);
            }
        }
    }
}

/// Runs one scenario file on a fresh isolated browser. Returns the run on
/// success, or the launch error on failure (so the caller can classify and
/// retry).
fn run_one_file(file: &ScenarioFile, reporter: &Arc<Reporter>) -> Result<ParallelRun, String> {
    let runner = ScenarioRunner::with_reporter_parallel(
        file.config.clone(),
        file.definitions.clone(),
        Arc::clone(reporter),
    );
    match runner.run(&file.tests) {
        Ok(report) => {
            let usage = runner.usage_tracker();
            Ok(ParallelRun {
                report,
                per_test: usage.per_test_snapshots(),
                global: usage.global_snapshot(),
            })
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Whether a browser-launch error looks like memory exhaustion and is worth
/// retrying after backing off.
#[must_use]
fn is_retryable_oom(err: &str) -> bool {
    const KEYWORDS: [&str; 8] = [
        "memory",
        "cannot allocate",
        "out of memory",
        "killed",
        "oom",
        "resource temporarily unavailable",
        "failed to allocate",
        "no memory",
    ];
    let e = err.to_lowercase();
    KEYWORDS.iter().any(|kw| e.contains(kw))
}

/// A report that fails every test in `file` (used when the browser cannot be
/// launched after retries).
#[allow(clippy::cast_possible_truncation)]
fn synthesized_failed_report(file: &ScenarioFile) -> ParallelRun {
    let n = file.tests.len() as u32;
    ParallelRun {
        report: RunReport {
            tests_passed: 0,
            tests_failed: n,
            passed: 0,
            failed: n,
            skipped: 0,
            details: Vec::new(),
        },
        per_test: Vec::new(),
        global: UsageSnapshot::default(),
    }
}

/// Merges a batch of per-file global usage snapshots into one, summing
/// per-endpoint counters so the cost report is accurate across files.
#[must_use]
fn merge_globals(snapshots: &[UsageSnapshot]) -> UsageSnapshot {
    let mut endpoints: HashMap<String, EndpointUsage> = HashMap::new();
    for snapshot in snapshots {
        for (name, usage) in &snapshot.endpoints {
            let acc = endpoints.entry(name.clone()).or_default();
            acc.calls += usage.calls;
            acc.input_tokens += usage.input_tokens;
            acc.output_tokens += usage.output_tokens;
            acc.cost += usage.cost;
        }
    }
    UsageSnapshot::from_endpoints(&endpoints)
}

// ── Memory probing ────────────────────────────────────────────────────

/// Best-effort probe of total/available physical memory (Linux then macOS).
/// Returns `None` when unavailable; the scheduler then relies on trial and
/// error.
#[must_use]
#[allow(clippy::unnecessary_unwrap)]
pub async fn probe_memory_async() -> Option<MemoryInfo> {
    // Linux: `free -b` prints "Mem:  <total> <used> <free> <shared> <buff> <cache> <available>".
    let total = run_sh_async("free -b | awk '/^Mem:/{print $2}'").await;
    let available = run_sh_async("free -b | awk '/^Mem:/{print $7}'").await;
    if total.is_some() && available.is_some() {
        let t = total.unwrap().trim().parse::<u64>().ok();
        let a = available.unwrap().trim().parse::<u64>().ok();
        if t.is_some() && a.is_some() {
            return Some(MemoryInfo {
                total: t.unwrap(),
                available: a.unwrap(),
            });
        }
    }

    // macOS: total bytes from sysctl; free pages × page size (from sysctl,
    // or vm_stat). `Pages free` is printed with a trailing period, so strip
    // non-digits.
    let total_mac = run_sh_async("sysctl -n hw.memsize").await;
    let page_size = run_sh_async("sysctl -n hw.pagesize").await;
    let pages =
        run_sh_async("vm_stat | awk '/^Pages free:/{gsub(/[^0-9]/, \"\", $3); print $3}'").await;
    if total_mac.is_some() && pages.is_some() && page_size.is_some() {
        let t = total_mac.unwrap().trim().parse::<u64>().ok();
        let p = pages.unwrap().trim().parse::<u64>().ok();
        let ps = page_size.unwrap().trim().parse::<u64>().ok();
        if t.is_some() && p.is_some() && ps.is_some() {
            return Some(MemoryInfo {
                total: t.unwrap(),
                available: p.unwrap() * ps.unwrap(),
            });
        }
    }
    None
}

/// Synchronous best-effort probe of currently available memory, used by
/// workers to learn the per-browser footprint and refresh the headroom
/// guard. Returns `None` when unavailable (e.g. non-Linux workers).
#[must_use]
fn available_memory_now() -> Option<u64> {
    run_sh_sync("free -b 2>/dev/null | awk '/^Mem:/{print $7}'")
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Runs a `sh -c` script asynchronously and returns trimmed stdout.
#[must_use]
async fn run_sh_async(script: &str) -> Option<String> {
    let output = tokio::process::Command::new("sh")
        .args(["-c", script])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Runs a `sh -c` script synchronously and returns the first trimmed line.
#[must_use]
fn run_sh_sync(script: &str) -> Option<String> {
    let mut child = std::process::Command::new("sh")
        .args(["-c", script])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .ok()?;
    let stdout = child.stdout.as_mut()?;
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    let _ = reader.read_line(&mut line);
    if line.is_empty() {
        None
    } else {
        Some(line.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::costs::{EndpointUsage, UsageSnapshot};
    use crate::scenario::ScenarioConfig;

    use super::{
        is_retryable_oom, make_gate, merge_globals, mode_from_cli, run_sh_sync, Claim,
        ParallelMode, RunOptions, ScenarioFile, SchedulerState,
    };

    fn file(label: &str, group: Option<&str>) -> ScenarioFile {
        ScenarioFile {
            label: label.to_owned(),
            config: ScenarioConfig {
                concurrency_group: group.map(std::borrow::ToOwned::to_owned),
                ..ScenarioConfig::default()
            },
            definitions: Vec::new(),
            tests: Vec::new(),
        }
    }

    fn state(files: Vec<ScenarioFile>) -> SchedulerState {
        let n = files.len();
        SchedulerState {
            files,
            results: Vec::new(),
            started: vec![false; n],
            active_groups: std::collections::HashSet::new(),
        }
    }

    #[test]
    fn test_distinct_groups_claim_in_parallel() {
        let mut s = state(vec![file("a", None), file("b", Some("x")), file("c", None)]);
        assert!(matches!(s.claim(), Claim::Take(0)));
        assert!(matches!(s.claim(), Claim::Take(1)));
        assert!(matches!(s.claim(), Claim::Take(2)));
        s.complete(1);
        assert!(matches!(s.claim(), Claim::Done));
    }

    #[test]
    fn test_same_group_blocks_until_completed() {
        let mut s = state(vec![
            file("a", Some("g")),
            file("b", Some("g")),
            file("c", None),
        ]);
        assert!(matches!(s.claim(), Claim::Take(0)));
        assert!(
            matches!(s.claim(), Claim::Take(2)),
            "a different group still runs while 'g' is active"
        );
        assert!(
            matches!(s.claim(), Claim::Wait),
            "b is blocked by a's group"
        );
        s.complete(0);
        assert!(
            matches!(s.claim(), Claim::Take(1)),
            "b runs after a finishes"
        );
        s.complete(1);
        assert!(matches!(s.claim(), Claim::Done));
    }

    #[test]
    fn test_no_group_means_own_group() {
        let mut s = state(vec![file("a", None), file("b", None)]);
        assert!(matches!(s.claim(), Claim::Take(0)));
        assert!(
            matches!(s.claim(), Claim::Take(1)),
            "no-group files run in parallel"
        );
    }

    #[test]
    fn test_mode_from_cli_manual_wins() {
        assert_eq!(mode_from_cli(Some(5), 1, 0), ParallelMode::Manual(5));
    }

    #[test]
    fn test_mode_from_cli_auto_with_defaults() {
        assert_eq!(
            mode_from_cli(None, 1, 0),
            ParallelMode::Auto { min: 1, max: 0 }
        );
    }

    #[test]
    fn test_gate_manual_is_fixed() {
        let gate = make_gate(&ParallelMode::Manual(3), None, 10);
        assert_eq!(gate.limit, 3);
        assert_eq!(gate.threads, 3);
        assert_eq!(gate.lower, 3);
    }

    #[test]
    fn test_gate_auto_ramps_from_min_toward_ceiling() {
        let gate = make_gate(&ParallelMode::Auto { min: 1, max: 0 }, None, 100);
        assert_eq!(gate.lower, 1);
        assert_eq!(gate.limit, 1);
        // Unknown memory: ceiling falls back to DEFAULT_AUTO_CEILING (8).
        assert_eq!(gate.ramp_ceiling, 8);
        assert_eq!(gate.threads, 64);
    }

    #[test]
    fn test_gate_auto_respects_user_max() {
        let gate = make_gate(&ParallelMode::Auto { min: 1, max: 8 }, None, 100);
        assert_eq!(gate.threads, 8);
        assert_eq!(gate.ramp_ceiling, 8);
    }

    #[test]
    fn test_gate_memory_guard_blocks_low_headroom() {
        let gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 4 },
            Some(crate::parallel::MemoryInfo {
                total: 1_000_000_000,
                available: 50_000_000, // below MIN_HEADROOM_BYTES
            }),
            4,
        );
        assert!(gate.memory_blocked());
        assert!(!gate.can_launch());
    }

    #[test]
    fn test_gate_memory_guard_allows_headroom() {
        let gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 4 },
            Some(crate::parallel::MemoryInfo {
                total: 1_000_000_000,
                available: 900_000_000,
            }),
            4,
        );
        assert!(!gate.memory_blocked());
        assert!(gate.can_launch());
    }

    #[test]
    fn test_gate_memory_guard_ignores_huge_host_total() {
        // VM/container: total reports the host's huge RAM, but available is
        // small. The guard must block based on absolute available bytes, not
        // a fraction of total.
        let gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 4 },
            Some(crate::parallel::MemoryInfo {
                total: 500_000_000_000, // 500 GB host
                available: 100_000_000, // 100 MB left in the cgroup
            }),
            4,
        );
        assert!(gate.memory_blocked(), "must block despite a 500 GB 'total'");
    }

    #[test]
    fn test_gate_memory_guard_blocks_when_no_room_for_one_footprint() {
        let mut gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 4 },
            Some(crate::parallel::MemoryInfo {
                total: 8_000_000_000,
                available: 300_000_000, // above the absolute floor...
            }),
            4,
        );
        // ...but one learned browser needs ~500 MB, so it must still block.
        gate.footprint = 500.0 * 1024.0 * 1024.0;
        gate.footprint_count = 3;
        assert!(gate.memory_blocked());
    }

    #[test]
    fn test_oom_failure_halves_limit_and_sets_backoff() {
        let mut gate = make_gate(&ParallelMode::Auto { min: 1, max: 16 }, None, 100);
        gate.limit = 16;
        gate.on_launch_failure();
        assert_eq!(gate.limit, 8);
        assert!(gate.retry_backoff > 0);
    }

    #[test]
    fn test_is_retryable_oom_matches_memory_errors() {
        assert!(is_retryable_oom("failed to launch browser: out of memory"));
        assert!(is_retryable_oom("cannot allocate memory for page"));
        assert!(!is_retryable_oom("Chrome binary not found"));
        assert!(!is_retryable_oom("invalid URL"));
    }

    #[test]
    fn test_run_options_cloneable() {
        let _ = RunOptions {
            mode: ParallelMode::Manual(2),
            reporter: Arc::new(crate::reporting::Reporter::default()),
            memory: None,
        };
    }

    #[test]
    fn test_launch_finished_learns_footprint_from_delta() {
        let mut gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 4 },
            Some(crate::parallel::MemoryInfo {
                total: 8_000_000_000,
                available: 8_000_000_000,
            }),
            4,
        );
        // Before = 1 GiB free, after = 600 MiB free → one browser ≈ 400 MiB.
        gate.launch_finished(Some(1_000_000_000), Some(600_000_000));
        assert!((gate.footprint - 400_000_000.0).abs() < 1.0);
        assert_eq!(gate.footprint_count, 1);
        assert_eq!(gate.available, Some(600_000_000));
    }

    #[test]
    fn test_on_success_raises_to_memory_capacity() {
        let mut gate = make_gate(
            &ParallelMode::Auto { min: 1, max: 0 },
            Some(crate::parallel::MemoryInfo {
                total: 8_000_000_000,
                available: 4_000_000_000,
            }),
            100,
        );
        assert_eq!(gate.limit, 1);
        // Learn a 1 GiB footprint, then capacity = 4 GiB / 1 GiB = 4.
        gate.footprint = 1_000_000_000.0;
        gate.footprint_count = 3;
        gate.on_success();
        assert_eq!(gate.limit, 4);
    }

    #[test]
    fn test_on_success_ramps_when_memory_unknown() {
        let mut gate = make_gate(&ParallelMode::Auto { min: 1, max: 0 }, None, 100);
        assert_eq!(gate.limit, 1);
        gate.on_success();
        assert_eq!(gate.limit, 2, "ramps up one at a time");
    }

    #[test]
    fn test_can_launch_respects_active_limit() {
        let mut gate = make_gate(&ParallelMode::Manual(2), None, 10);
        assert!(gate.can_launch());
        gate.launch_started(Some(1_000_000_000));
        assert!(gate.can_launch(), "one of two slots free");
        gate.launch_started(Some(1_000_000_000));
        assert!(!gate.can_launch(), "both manual slots in use");
        gate.release(Some(1_000_000_000));
        assert!(gate.can_launch(), "slot freed after release");
    }

    #[test]
    fn test_merge_globals_sums_endpoint_counters() {
        let mut snap1 = UsageSnapshot::default();
        let mut snap2 = UsageSnapshot::default();
        snap1.endpoints.insert(
            "a".to_owned(),
            EndpointUsage {
                calls: 1,
                input_tokens: 100,
                output_tokens: 50,
                cost: 0.01,
            },
        );
        snap2.endpoints.insert(
            "a".to_owned(),
            EndpointUsage {
                calls: 2,
                input_tokens: 200,
                output_tokens: 100,
                cost: 0.02,
            },
        );
        snap2.endpoints.insert(
            "b".to_owned(),
            EndpointUsage {
                calls: 1,
                input_tokens: 10,
                output_tokens: 5,
                cost: 0.001,
            },
        );
        let merged = merge_globals(&[snap1, snap2]);
        assert_eq!(merged.endpoints.len(), 2);
        let a = merged.endpoints.get("a").unwrap();
        assert_eq!(a.calls, 3);
        assert_eq!(a.input_tokens, 300);
        assert!((a.cost - 0.03).abs() < 0.0001);
        let b = merged.endpoints.get("b").unwrap();
        assert_eq!(b.calls, 1);
    }

    #[test]
    fn test_run_sh_sync_captures_output() {
        let Some(out) = run_sh_sync("echo hello") else {
            return; // sh unavailable on this platform — nothing to assert
        };
        assert_eq!(out, "hello");
    }

    /// Runs the real worker pool against two failing scenario files and
    /// asserts the batch machinery: both files' tests are accounted for, and
    /// exactly one `RunStarted` + one `RunFinished` are emitted (each file
    /// suppresses its own, so the batch owns the pair). Uses an unreachable
    /// URL so files fail fast with or without Chrome installed.
    #[test]
    fn test_run_scenarios_batch_emits_one_run_event_pair() {
        use crate::reporting::{ColorMode, Level, Reporter};
        use crate::scenario::{TestGroup, TestStep};

        let id = std::process::id();
        let log_path = std::env::temp_dir().join(format!("lbt-parallel-{id}.ndjson"));
        let reporter = Arc::new(
            Reporter::new(
                Level::Error,
                ColorMode::Never,
                Some(&log_path),
                None,
                None,
                false,
            )
            .ok()
            .unwrap(),
        );

        let mut files: Vec<ScenarioFile> = Vec::new();
        for (label, url) in [("a", "http://127.0.0.1:9/"), ("b", "http://127.0.0.1:9/")] {
            files.push(ScenarioFile {
                label: label.to_owned(),
                config: ScenarioConfig::default(),
                definitions: Vec::new(),
                tests: vec![TestGroup {
                    name: label.to_owned(),
                    start_url: None,
                    auto_navigate: None,
                    base_url: None,
                    timeout_secs: Some(5),
                    browser_headless: Some(true),
                    viewport_width: None,
                    viewport_height: None,
                    budget: None,
                    endpoint: None,
                    steps: vec![TestStep::Navigate {
                        url: url.to_owned(),
                        wait_after_ms: None,
                    }],
                }],
            });
        }

        let run = crate::parallel::run_scenarios(
            files,
            RunOptions {
                mode: ParallelMode::Manual(2),
                reporter: Arc::clone(&reporter),
                memory: None,
            },
        )
        .ok()
        .unwrap();

        // Both files' single test must be accounted for in the merged report.
        assert_eq!(
            run.report.tests_passed + run.report.tests_failed,
            2,
            "both files contributed exactly one test"
        );

        // Exactly one batch-level RunStarted and one RunFinished.
        reporter.finish().ok().unwrap();
        let text = std::fs::read_to_string(&log_path).ok().unwrap();
        let started = text
            .lines()
            .filter(|l| l.contains("\"type\":\"run_started\""))
            .count();
        let finished = text
            .lines()
            .filter(|l| l.contains("\"type\":\"run_finished\""))
            .count();
        assert_eq!(started, 1, "exactly one RunStarted for the batch");
        assert_eq!(finished, 1, "exactly one RunFinished for the batch");
    }
}
