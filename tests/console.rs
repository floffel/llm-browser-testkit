//! Browser console-capture policy tests.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use llm_browser_testkit::console::{ConsoleCapture, ConsoleEntry, ConsoleErrorsPolicy};
use llm_browser_testkit::scenario::Scenario;

fn entry(level: &str, source: &str, text: &str) -> ConsoleEntry {
    ConsoleEntry {
        level: level.to_owned(),
        source: source.to_owned(),
        text: text.to_owned(),
        location: None,
    }
}

#[test]
fn default_policy_is_show() {
    assert_eq!(ConsoleErrorsPolicy::default(), ConsoleErrorsPolicy::Show);
}

#[test]
fn is_error_covers_error_level_and_exceptions() {
    assert!(entry("error", "javascript", "boom").is_error());
    assert!(entry("error", "network", "500").is_error());
    assert!(entry("error", "exception", "TypeError").is_error());
    assert!(!entry("warning", "javascript", "deprecated").is_error());
    assert!(!entry("info", "javascript", "hi").is_error());
}

#[test]
fn capture_push_render_clear() {
    let capture = ConsoleCapture::new();
    assert!(capture.is_empty());
    assert!(!capture.has_errors());

    capture.push(entry("warning", "javascript", "deprecated api"));
    assert_eq!(capture.len(), 1);
    assert!(!capture.has_errors());

    capture.push(entry("error", "javascript", "Uncaught TypeError"));
    assert!(capture.has_errors());

    let rendered = capture.render();
    assert!(rendered.contains("[warning] javascript: deprecated api"));
    assert!(rendered.contains("[error] javascript: Uncaught TypeError"));

    capture.clear();
    assert!(capture.is_empty());
    assert_eq!(capture.render(), "");
}

#[test]
fn clones_share_the_same_buffer() {
    let capture = ConsoleCapture::new();
    let clone = capture.clone();
    clone.push(entry("error", "exception", "TypeError: x is undefined"));
    assert_eq!(capture.len(), 1);
    assert!(capture.has_errors());
}

#[test]
fn scenario_parses_console_errors_policy() {
    let toml = r#"
[config]
console_errors = "fail"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert_eq!(scenario.config.console_errors, ConsoleErrorsPolicy::Fail);
}

#[test]
fn scenario_defaults_console_errors_to_show() {
    let toml = r#"
[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert_eq!(scenario.config.console_errors, ConsoleErrorsPolicy::Show);
}
