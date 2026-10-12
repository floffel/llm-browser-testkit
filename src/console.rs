//! Browser console + uncaught-exception capture for assertions.
//!
//! The CDP `Log` domain reports console API calls (`console.error`,
//! `console.warn`), failed network requests, and CSP/security violations;
//! `Runtime.exceptionThrown` reports uncaught JS exceptions and unhandled
//! promise rejections. Assertions otherwise only see the rendered DOM text,
//! so a swallowed `console.error` (e.g. a failed GraphQL mutation) is
//! invisible. This module captures those entries and — per
//! [`ConsoleErrorsPolicy`] — discards them, feeds them to the LLM assertion,
//! or fails the step outright.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use headless_chrome::protocol::cdp::types::Event;
use headless_chrome::Tab;
use serde::Deserialize;

/// What to do with browser console errors captured during a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ConsoleErrorsPolicy {
    /// Ignore console output entirely (default; backwards compatible).
    #[default]
    Discard,
    /// Append the captured errors/warnings to the assertion payload so the
    /// LLM evaluates them, and surface them in the step log.
    Show,
    /// Fail the assertion step deterministically when any console error or
    /// uncaught exception was captured (no LLM call).
    Fail,
}

/// A single captured browser log or exception entry.
#[derive(Debug, Clone)]
pub struct ConsoleEntry {
    /// Severity (`error`, `warning`, ...).
    pub level: String,
    /// Origin channel (`javascript`, `network`, `security`, `exception`, ...).
    pub source: String,
    /// The message text.
    pub text: String,
    /// `url:line` of the source, when known.
    pub location: Option<String>,
}

impl ConsoleEntry {
    /// Whether this entry is a hard error: severity `error` or an uncaught
    /// exception.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.level.eq_ignore_ascii_case("error") || self.source == "exception"
    }
}

/// Shared, thread-safe capture buffer installed on the browser tab.
///
/// Cheap to clone (the buffer is behind an `Arc`) so the CDP event-listener
/// closure and the runner can share it.
#[derive(Debug, Clone, Default)]
pub struct ConsoleCapture {
    entries: Arc<Mutex<Vec<ConsoleEntry>>>,
}

impl ConsoleCapture {
    /// Create an empty capture.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Install the CDP listeners on `tab` so console / exception events are
    /// appended to this buffer for the rest of the run.
    ///
    /// # Errors
    ///
    /// Returns an error if the `Log` domain cannot be enabled or the listener
    /// cannot be registered.
    pub fn install(&self, tab: &Tab) -> Result<()> {
        tab.enable_log()
            .context("failed to enable the browser Log domain")?;
        let sink = Arc::clone(&self.entries);
        tab.add_event_listener(Arc::new(move |event: &Event| match event {
            Event::LogEntryAdded(ev) => {
                let entry = &ev.params.entry;
                let level = format!("{:?}", entry.level).to_lowercase();
                // Keep only actionable severities; `info`/`verbose` are noise.
                if level != "error" && level != "warning" {
                    return;
                }
                let location = entry.url.as_ref().map(|url| {
                    entry
                        .line_number
                        .map_or_else(|| url.clone(), |line| format!("{url}:{line}"))
                });
                push(
                    &sink,
                    ConsoleEntry {
                        level,
                        source: format!("{:?}", entry.source).to_lowercase(),
                        text: entry.text.clone(),
                        location,
                    },
                );
            }
            Event::RuntimeExceptionThrown(ev) => {
                let details = &ev.params.exception_details;
                let text = details
                    .exception
                    .as_ref()
                    .and_then(|remote| remote.description.clone())
                    .unwrap_or_else(|| details.text.clone());
                let location = details
                    .url
                    .clone()
                    .map(|url| format!("{url}:{}", details.line_number));
                push(
                    &sink,
                    ConsoleEntry {
                        level: "error".to_owned(),
                        source: "exception".to_owned(),
                        text,
                        location,
                    },
                );
            }
            _ => {}
        }))?;
        Ok(())
    }

    /// Drop all captured entries (call at the start of each test).
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    /// Append one entry (used by the CDP listener and by tests).
    pub fn push(&self, entry: ConsoleEntry) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.push(entry);
        }
    }

    /// Number of captured entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().map_or(0, |entries| entries.len())
    }

    /// Whether nothing was captured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether any captured entry is an error or an uncaught exception.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.entries
            .lock()
            .is_ok_and(|entries| entries.iter().any(ConsoleEntry::is_error))
    }

    /// Render the captured entries as a human-readable block (one per line).
    #[must_use]
    pub fn render(&self) -> String {
        let Ok(entries) = self.entries.lock() else {
            return String::new();
        };
        let mut out = String::new();
        for entry in entries.iter() {
            let _ = write!(out, "- [{}] {}: {}", entry.level, entry.source, entry.text);
            if let Some(location) = &entry.location {
                let _ = write!(out, " ({location})");
            }
            out.push('\n');
        }
        out
    }
}

/// Append an entry to a shared sink, ignoring a poisoned lock (the capture is
/// best-effort and must never panic the browser thread).
fn push(sink: &Arc<Mutex<Vec<ConsoleEntry>>>, entry: ConsoleEntry) {
    if let Ok(mut entries) = sink.lock() {
        entries.push(entry);
    }
}
