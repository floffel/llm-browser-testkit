//! Failure diagnostics — page-state capture and artifact writing.
//!
//! When a step fails, the runner captures the current page state (URL,
//! title, visible text, error-ish alert elements) and saves a screenshot.
//! The formatted context is appended to step messages so CI logs answer the
//! classic "why did the wait time out?" question instead of printing a bare
//! timeout message.

use std::fmt::Write as _;
use std::path::PathBuf;

use headless_chrome::{protocol::cdp::Page::CaptureScreenshotFormatOption, Tab};

use crate::truncate;

/// Maximum length of visible text kept in the in-message excerpt.
const EXCERPT_LEN: usize = 160;
/// Maximum length of visible text kept for the printed diagnostics block.
const FULL_TEXT_LEN: usize = 1500;

/// JavaScript that collects visible "alert-like" elements — error banners,
/// toast/snackbars, validation messages, error cards.
pub const DIAGNOSTIC_ALERTS_JS: &str = r#"
(() => {
  const sel = '[role="alert"], [aria-live="assertive"], .alert, .error, .error-message, .errorMessage, .error-text, [data-error], [class*="error"], [class*="Error"], mat-snack-bar-container, .mat-mdc-snack-bar-container, .toast, .notification';
  const els = document.querySelectorAll(sel);
  const out = [];
  const seen = new Set();
  els.forEach((el) => {
    if (el.offsetParent === null && !el.closest('mat-snack-bar-container')) return;
    const text = (el.innerText || el.textContent || '').trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    out.push(text.substring(0, 200));
  });
  return JSON.stringify(out.slice(0, 6));
})()
"#;

/// Snapshot of the page at the moment a step failed.
pub struct PageState {
    /// Current URL.
    pub url: String,
    /// `document.title`.
    pub title: String,
    /// Visible body text (truncated).
    pub visible_text: String,
    /// Text of visible error/alert elements (truncated, deduplicated).
    pub alerts: Vec<String>,
}

/// Captures the current page state via CDP evaluation. Never fails — every
/// extractor degrades to a default value so diagnostic capture cannot mask
/// the original step failure.
#[must_use]
pub fn capture(tab: &Tab) -> PageState {
    let url = tab.get_url();
    let title = tab
        .evaluate("document.title", false)
        .ok()
        .and_then(|r| r.value)
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".to_owned());

    let visible_text = tab
        .evaluate(
            "document.body ? document.body.innerText : document.documentElement.innerText",
            false,
        )
        .ok()
        .and_then(|r| r.value)
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default();

    let alerts = tab
        .evaluate(DIAGNOSTIC_ALERTS_JS, false)
        .ok()
        .and_then(|r| r.value)
        .and_then(|v| v.as_str().map(String::from))
        .and_then(|json| serde_json::from_str::<Vec<String>>(&json).ok())
        .unwrap_or_default();

    PageState {
        url,
        title,
        visible_text: truncate(&visible_text, FULL_TEXT_LEN),
        alerts,
    }
}

/// Compact one-line excerpt of the page state, appended to `StepResult`
/// messages so the summary line in CI logs stays self-contained.
#[must_use]
pub fn inline_excerpt(state: &PageState) -> String {
    let mut out = format!("page: {} ({})", state.url, state.title);
    let trimmed = state.visible_text.trim();
    if !trimmed.is_empty() {
        let excerpt = truncate(trimmed, EXCERPT_LEN);
        let one_line = excerpt.replace('\n', " ⏎ ");
        let _ = write!(out, " — visible: \"{one_line}\"");
    }
    if !state.alerts.is_empty() {
        let _ = write!(out, " — alerts: {}", state.alerts.join(" | "));
    }
    out
}

/// Multi-line diagnostics block printed to stderr after a step failure.
#[must_use]
pub fn full_context(state: &PageState, screenshot: Option<&str>) -> String {
    let mut lines = Vec::new();
    lines.push(format!("    │ url:      {}", state.url));
    lines.push(format!("    │ title:    {}", state.title));
    if !state.alerts.is_empty() {
        for alert in &state.alerts {
            lines.push(format!("    │ alert:    {alert}"));
        }
    }
    let trimmed = state.visible_text.trim();
    if !trimmed.is_empty() {
        lines.push(format!("    │ content:  {trimmed}"));
    }
    if let Some(path) = screenshot {
        lines.push(format!("    📸 screenshot: {path}"));
    }
    lines.join("\n")
}

/// Sanitizes a test/step name for use in artifact file names.
#[must_use]
pub fn slugify(name: &str, max_len: usize) -> String {
    let mut slug: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-');
    if slug.len() > max_len {
        // Slug is pure ASCII (letters/digits/dashes), so byte slicing is
        // safe and keeps the file name at exactly max_len characters.
        slug[..max_len].to_owned()
    } else {
        slug.to_owned()
    }
}

/// Saves a PNG screenshot of the current tab under `dir`, named after the
/// scenario/test/step. Returns the absolute-ish path, or `None` when the
/// screenshot fails (best effort — never fails the step).
#[must_use]
pub fn save_screenshot(
    tab: &Tab,
    dir: &PathBuf,
    scenario: &str,
    test: &str,
    step_index: usize,
    step_kind: &str,
) -> Option<String> {
    let data = tab
        .capture_screenshot(CaptureScreenshotFormatOption::Png, None, None, true)
        .ok()?;
    let file_name = format!(
        "{scenario}__{test}__{step_index:03}-{step_kind}.png",
        scenario = slugify(scenario, 40),
        test = slugify(test, 40),
    );
    let path = dir.join(file_name);
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return None;
        }
    }
    if std::fs::write(&path, &data).is_err() {
        return None;
    }
    Some(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{full_context, inline_excerpt, slugify};
    use crate::diagnostics::PageState;

    fn state() -> PageState {
        PageState {
            url: "http://127.0.0.1:8082/auth/login".into(),
            title: "Immosai — Sign in".into(),
            visible_text: "Email address\nPassword\nSign in\nInvalid credentials.".into(),
            alerts: vec!["Invalid credentials.".into()],
        }
    }

    #[test]
    fn test_inline_excerpt_contains_url_and_text() {
        let s = inline_excerpt(&state());
        assert!(s.contains("auth/login"));
        assert!(s.contains("visible:"));
        assert!(s.contains("alerts:"));
    }

    #[test]
    fn test_inline_excerpt_newlines_flattened() {
        let s = inline_excerpt(&state());
        assert!(!s.contains('\n'));
    }

    #[test]
    fn test_full_context_lists_alerts_and_screenshot() {
        let s = full_context(&state(), Some("artifacts/x.png"));
        assert!(s.contains("url:"));
        assert!(s.contains("alert:"));
        assert!(s.contains("screenshot: artifacts/x.png"));
    }

    #[test]
    fn test_slugify() {
        assert_eq!(
            slugify("Login and open account", 60),
            "login-and-open-account"
        );
        assert_eq!(slugify("Äpfel & Birnen!", 60), "pfel-birnen");
        assert_eq!(slugify("a".repeat(100).as_str(), 20), "a".repeat(20));
    }
}
