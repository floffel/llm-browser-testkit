//! Test that temperature and thinking are parsed from TOML and
//! propagated through the ScenarioRunner to LLM calls.

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
    assert!(scenario.config.thinking);
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
    assert!(!scenario.config.thinking);
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
    assert!(!scenario.config.thinking);
}