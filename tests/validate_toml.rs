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
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {dir}: {e}"));

    let mut toml_files: Vec<_> = entries
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "toml")
        })
        .collect();
    toml_files.sort_by_key(std::fs::DirEntry::file_name);
    assert!(
        !toml_files.is_empty(),
        "no TOML examples found in {dir}"
    );

    for entry in &toml_files {
        let path = entry.path();
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let s: llm_browser_testkit::scenario::Scenario =
            toml::from_str(&content)
                .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        assert!(!s.test.is_empty(), "{}: no tests defined", path.display());
        for t in &s.test {
            assert!(!t.name.is_empty(), "{}: test has empty name", path.display());
        }
    }
}
