//! Tests for the `browser_test!` and `browser_test_inline!` macros.
//!
//! These tests verify that macros expand and the TOML parses correctly.
//! Actual browser/LLM execution is skipped when infrastructure is unavailable.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use llm_browser_testkit::browser_test;
use llm_browser_testkit::browser_test_inline;

browser_test!(smoke_from_file => "examples/smoke.toml");

browser_test_inline!(
    smoke_inline,
    r#"
[config]
base_url = "https://example.com"
start_url = "/"

[[definitions]]
name = "no_errors"
preset = "no_error_on_page"

[[test]]
name = "homepage"

[[test.steps]]
kind = "navigate"
url = "/"

[[test.steps]]
kind = "assert"
definition = "no_errors"
"#
);

#[test]
fn macros_are_exported() {
    // Compile-time check: the macro invocations above already verify
    // that browser_test! and browser_test_inline! are in scope.
}
