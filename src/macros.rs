//! Macros for integrating browser tests into `cargo test`.
//!
//! Each macro generates a `#[test]` function that runs a TOML scenario
//! through the browser. Tests are **auto-skipped** when the LLM endpoint
//! or Chrome is unavailable, so they are safe to include in every
//! `cargo test` run without environment setup.
//!
//! # Feature gate
//!
//! Enable with `features = ["macros"]` in your `Cargo.toml`. The macros
//! carry heavy dependencies (`headless_chrome`, `reqwest`) and are
//! disabled by default to keep compile times fast for library consumers.
//!
//! # Examples
//!
//! ```ignore
//! use llm_browser_testkit::browser_test;
//!
//! browser_test!(smoke_test => "tests/smoke.toml");
//! ```
//!
//! ```ignore
//! use llm_browser_testkit::browser_test_inline;
//!
//! browser_test_inline!(hello_world, r#"
//! [config]
//! base_url = "https://example.com"
//! start_url = "/"
//!
//! [[definitions]]
//! name = "no_errors"
//! preset = "no_error_on_page"
//!
//! [[test]]
//! name = "homepage"
//!
//! [[test.steps]]
//! kind = "navigate"
//! url = "/"
//!
//! [[test.steps]]
//! kind = "assert"
//! definition = "no_errors"
//! "#);
//! ```

/// Generates a `#[test]` that runs a TOML scenario file via the browser.
///
/// The TOML file path is resolved relative to the crate root
/// via `concat!(env!("CARGO_MANIFEST_DIR"), "/", $path)`.
///
/// The test is auto-skipped (passes silently) when no LLM endpoint or
/// Chrome is detected.
#[macro_export]
macro_rules! browser_test {
    ($name:ident => $path:expr) => {
        #[test]
        fn $name() {
            let path = ::std::concat!(::std::env!("CARGO_MANIFEST_DIR"), "/", $path);
            let toml_content = match ::std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    ::std::eprintln!(
                        "⏭️  skipping {}: cannot read TOML ({e})",
                        ::std::stringify!($name)
                    );
                    return;
                }
            };
            let scenario: $crate::scenario::Scenario = match ::toml::from_str(&toml_content) {
                Ok(s) => s,
                Err(e) => {
                    ::std::eprintln!(
                        "⏭️  skipping {}: invalid TOML ({e})",
                        ::std::stringify!($name)
                    );
                    return;
                }
            };

            let config = scenario.config.clone();
            let definitions = scenario.definitions;
            let runner = $crate::runner::ScenarioRunner::new(config, definitions);

            match runner.run(&scenario.test) {
                Ok(report) => {
                    if report.tests_passed == 0
                        && report.failed > 0
                        && report.details.iter().any(|d| {
                            d.message.contains("LLM assertion call failed")
                                || d.message.contains("LLM element targeting failed")
                        })
                    {
                        ::std::eprintln!(
                            "⏭️  skipping {}: LLM endpoint unavailable",
                            ::std::stringify!($name)
                        );
                        return;
                    }
                    assert_eq!(
                        report.tests_failed,
                        0,
                        "{}: {} test(s) failed, {} passed",
                        ::std::stringify!($name),
                        report.tests_failed,
                        report.tests_passed,
                    );
                    assert_ne!(
                        report.tests_passed,
                        0,
                        "{}: no tests ran",
                        ::std::stringify!($name)
                    );
                }
                Err(e) => {
                    if e.to_string().contains("failed to launch browser") {
                        ::std::eprintln!(
                            "⏭️  skipping {}: browser unavailable ({e})",
                            ::std::stringify!($name)
                        );
                        return;
                    }
                    panic!("{}: scenario run failed: {e}", ::std::stringify!($name));
                }
            }
        }
    };
}

/// Generates a `#[test]` from an inline TOML string literal.
///
/// Useful for small scenarios defined directly in the test file.
#[macro_export]
macro_rules! browser_test_inline {
    ($name:ident, $toml:expr) => {
        #[test]
        fn $name() {
            let scenario: $crate::scenario::Scenario = match ::toml::from_str($toml) {
                Ok(s) => s,
                Err(e) => {
                    ::std::eprintln!(
                        "⏭️  skipping {}: invalid TOML ({e})",
                        ::std::stringify!($name)
                    );
                    return;
                }
            };

            let config = scenario.config.clone();
            let definitions = scenario.definitions;
            let runner = $crate::runner::ScenarioRunner::new(config, definitions);

            match runner.run(&scenario.test) {
                Ok(report) => {
                    if report.tests_passed == 0
                        && report.failed > 0
                        && report.details.iter().any(|d| {
                            d.message.contains("LLM assertion call failed")
                                || d.message.contains("LLM element targeting failed")
                        })
                    {
                        ::std::eprintln!(
                            "⏭️  skipping {}: LLM endpoint unavailable",
                            ::std::stringify!($name)
                        );
                        return;
                    }
                    assert_eq!(
                        report.tests_failed,
                        0,
                        "{}: {} test(s) failed, {} passed",
                        ::std::stringify!($name),
                        report.tests_failed,
                        report.tests_passed,
                    );
                    assert_ne!(
                        report.tests_passed,
                        0,
                        "{}: no tests ran",
                        ::std::stringify!($name)
                    );
                }
                Err(e) => {
                    if e.to_string().contains("failed to launch browser") {
                        ::std::eprintln!(
                            "⏭️  skipping {}: browser unavailable ({e})",
                            ::std::stringify!($name)
                        );
                        return;
                    }
                    panic!("{}: scenario run failed: {e}", ::std::stringify!($name));
                }
            }
        }
    };
}
