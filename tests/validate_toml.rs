//! Integration tests validating TOML scenario files parse correctly.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::fs;

#[test]
fn validate_all_toml_scenarios() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/examples");
    let files = [
        "smoke.toml",
        "dashboard-smoke.toml",
        "backlog-story-crud.toml",
        "navigation-health.toml",
        "full-feature-smoke.toml",
        "basic-navigation.toml",
    ];
    for f in &files {
        let path = format!("{dir}/{f}");
        let content = fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        let s: llm_browser_testkit::scenario::Scenario =
            toml::from_str(&content).unwrap_or_else(|e| panic!("parsing {path}: {e}"));
        assert!(!s.test.is_empty(), "{path}: no tests defined");
        for t in &s.test {
            assert!(!t.name.is_empty(), "{path}: test has empty name");
        }
    }
}
