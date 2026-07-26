//! Test that temperature, thinking, and `model_params` are parsed from TOML.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use llm_browser_testkit::scenario::Scenario;

#[test]
fn test_toml_parses_temperature_and_thinking() {
    let toml = r#"
[config]
temperature = 0.7
thinking = true

[[definitions]]
name = "no_errors"
preset = "no_error_on_page"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert!((scenario.config.temperature - 0.7).abs() < f64::EPSILON);
    assert_eq!(scenario.config.thinking, Some(true));
}

#[test]
fn test_default_temperature_and_thinking() {
    let toml = r#"
[[definitions]]
name = "no_errors"
preset = "no_error_on_page"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert!((scenario.config.temperature - 0.0).abs() < f64::EPSILON);
    assert_eq!(scenario.config.thinking, None);
}

#[test]
fn test_temperature_zero_and_thinking_false() {
    let toml = r#"
[config]
temperature = 0.0
thinking = false

[[definitions]]
name = "no_errors"
preset = "no_error_on_page"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert!((scenario.config.temperature - 0.0).abs() < f64::EPSILON);
    assert_eq!(scenario.config.thinking, Some(false));
}

#[test]
fn test_custom_assertion_definition() {
    let toml = r#"
[[definitions]]
name = "my_check"
system = "You are a friendly QA bot."
user_template = "Does the page mention {expected_text}?"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let def = &scenario.definitions[0];
    assert_eq!(def.name, "my_check");
    assert_eq!(def.system.as_deref(), Some("You are a friendly QA bot."));
    assert_eq!(
        def.user_template.as_deref(),
        Some("Does the page mention {expected_text}?")
    );
}

#[test]
fn test_model_params_toml() {
    let toml = r#"
[config]
[config.model_params]
effort = "high"
max_tokens = 2048

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let effort = scenario.config.model_params.get("effort").unwrap();
    assert_eq!(effort.as_str(), Some("high"));
    let max_t = scenario.config.model_params.get("max_tokens").unwrap();
    assert_eq!(max_t.as_i64(), Some(2048));
}
