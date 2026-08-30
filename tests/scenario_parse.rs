//! Additional scenario TOML parsing tests for edge cases.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use llm_browser_testkit::scenario::Scenario;

#[test]
fn test_empty_scenario() {
    let toml = "";
    let scenario: Scenario = toml::from_str(toml).expect("parse empty TOML");
    assert!(scenario.test.is_empty());
    assert!(scenario.definitions.is_empty());
}

#[test]
fn test_agent_definition_with_all_fields() {
    let toml = r#"
[[definitions]]
name = "full_agent"
preset = "no_error_on_page"
prompt = "check everything"
agent = "audit_agent"
task_template = "verify {url}"
assert_text = "hello"
system = "you are a QA"
user_template = "check {content}"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse full def");
    let def = &scenario.definitions[0];
    assert_eq!(def.name, "full_agent");
    assert_eq!(def.preset.as_deref(), Some("no_error_on_page"));
    assert_eq!(def.agent.as_deref(), Some("audit_agent"));
    assert_eq!(def.system.as_deref(), Some("you are a QA"));
}

#[test]
fn test_budgets_default_config_is_empty() {
    let toml = r#"
[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert!(scenario.config.budgets.global.is_none());
    assert!(scenario.config.budgets.per_test_default.is_none());
}

#[test]
fn test_budgets_max_calls() {
    let toml = r#"
[config.budgets.per_test_default]
max_calls = 25

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert_eq!(
        scenario.config.budgets.per_test_default.unwrap().max_calls,
        Some(25)
    );
}

#[test]
fn test_a2a_server_config_in_toml() {
    let toml = r#"
[config.a2a_server]
enabled = true
port = 9876

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let a2a = scenario.config.a2a_server.as_ref().unwrap();
    assert!(a2a.enabled);
    assert_eq!(a2a.port, 9876);
}

#[test]
fn test_a2a_server_disabled_by_default() {
    let toml = r#"
[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert!(scenario.config.a2a_server.is_none());
}

#[test]
fn test_endpoint_a2a_type() {
    let toml = r#"
[config.endpoints.my_agent]
type = "a2a"
url = "http://agent:9090"
pricing = { per_call = 0.05 }

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let ep = scenario.config.endpoints.get("my_agent").unwrap();
    assert_eq!(ep.url.as_deref(), Some("http://agent:9090"));
    assert!((ep.pricing.as_ref().unwrap().per_call - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_endpoint_default_for() {
    let toml = r#"
[config.endpoints.vision]
type = "llm"
url = "https://api.openai.com"
default_for = ["targeting", "assertion"]

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let ep = scenario.config.endpoints.get("vision").unwrap();
    assert_eq!(ep.default_for, vec!["targeting", "assertion"]);
}

#[test]
fn test_multiple_tests_with_budgets() {
    let toml = r#"
[config.budgets.global]
max_cost = 100.0
enforcement = "hard"

[[test]]
name = "test1"
budget = { max_cost = 10.0, enforcement = "soft" }
steps = []

[[test]]
name = "test2"
budget = { max_tokens = 5000 }
steps = []

[[test]]
name = "test3"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert_eq!(scenario.test.len(), 3);
    assert_eq!(
        scenario.test[0].budget.as_ref().unwrap().max_cost,
        Some(10.0)
    );
    assert_eq!(
        scenario.test[1].budget.as_ref().unwrap().max_tokens,
        Some(5000)
    );
    assert!(scenario.test[2].budget.is_none());
}

#[test]
fn test_endpoint_mcp_with_command() {
    let toml = r#"
[config.endpoints.db]
type = "mcp"
command = "npx"
args = ["-y", "pg-mcp"]

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let ep = scenario.config.endpoints.get("db").unwrap();
    assert_eq!(ep.command.as_deref(), Some("npx"));
    assert_eq!(ep.args, vec!["-y", "pg-mcp"]);
}

#[test]
fn test_pricing_all_fields() {
    let toml = r#"
[config.endpoints.gpt4]
type = "llm"
pricing = { input_per_1m_tokens = 2.50, output_per_1m_tokens = 10.00, per_call = 0.0 }

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let p = scenario
        .config
        .endpoints
        .get("gpt4")
        .unwrap()
        .pricing
        .as_ref()
        .unwrap();
    assert!((p.input_per_1m_tokens - 2.50).abs() < f64::EPSILON);
    assert!((p.output_per_1m_tokens - 10.00).abs() < f64::EPSILON);
    assert!((p.per_call - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_config_with_start_url_and_auto_navigate() {
    let toml = r#"
[config]
start_url = "/login"
auto_navigate = false

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert_eq!(scenario.config.start_url.as_deref(), Some("/login"));
    assert!(!scenario.config.auto_navigate);
}

#[test]
fn test_scenario_config_clone() {
    let toml = r#"
[config]
base_url = "https://example.com"
timeout_secs = 30

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let cloned = scenario.config.clone();
    // Use both original and clone to avoid redundant-clone lint
    assert!(scenario.config.base_url.is_some());
    assert_eq!(cloned.base_url.as_deref(), Some("https://example.com"));
    assert_eq!(cloned.timeout_secs, Some(30));
}

#[test]
fn test_config_all_fields() {
    let toml = r#"
[config]
base_url = "https://example.com"
llm_url = "https://llm.example.com"
llm_model = "gpt-4o"
llm_api_key = "sk-test"
llm_headers = { "X-Org" = "acme" }
browser_headless = true
timeout_secs = 120
viewport_width = 1920
viewport_height = 1080
start_url = "/home"
auto_navigate = true
temperature = 0.7
thinking = true

[config.model_params]
effort = "high"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let c = &scenario.config;
    assert_eq!(c.base_url.as_deref(), Some("https://example.com"));
    assert_eq!(c.llm_url.as_deref(), Some("https://llm.example.com"));
    assert_eq!(c.llm_model.as_deref(), Some("gpt-4o"));
    assert_eq!(c.llm_api_key.as_deref(), Some("sk-test"));
    assert_eq!(c.llm_headers.get("X-Org").map(String::as_str), Some("acme"));
    assert_eq!(c.timeout_secs, Some(120));
    assert_eq!(c.viewport_width, Some(1920));
    assert_eq!(c.viewport_height, Some(1080));
    assert!((c.temperature - 0.7).abs() < f64::EPSILON);
    assert_eq!(c.thinking, Some(true));
    assert!(c.model_params.contains_key("effort"));
}

#[test]
fn test_continue_on_failure_and_artifacts_dir_defaults() {
    let toml = r#"
[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert!(
        !scenario.config.continue_on_failure,
        "default should be fail-fast"
    );
    assert_eq!(scenario.config.artifacts_dir, None);
}

#[test]
fn test_continue_on_failure_and_artifacts_dir_override() {
    let toml = r#"
[config]
continue_on_failure = true
artifacts_dir = "ci-artifacts"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    assert!(scenario.config.continue_on_failure);
    assert_eq!(
        scenario.config.artifacts_dir.as_deref(),
        Some("ci-artifacts")
    );
}

#[test]
fn test_wait_step_text() {
    let toml = r##"
[[test]]
name = "text wait"
steps = [
    { kind = "wait", target = "the success message", text = "Welcome back", timeout_ms = 5000 },
    { kind = "wait", target = "element and text", selector = "#panel", text = "Loaded", timeout_ms = 8000 }
]
"##;
    let scenario: Scenario = toml::from_str(toml).expect("parse");
    let steps = &scenario.test[0].steps;
    match &steps[0] {
        llm_browser_testkit::scenario::TestStep::Wait {
            selector,
            text,
            timeout_ms,
            ..
        } => {
            assert!(selector.is_none());
            assert_eq!(text.as_deref(), Some("Welcome back"));
            assert_eq!(timeout_ms, &Some(5000));
        }
        _ => panic!("expected Wait step"),
    }
    match &steps[1] {
        llm_browser_testkit::scenario::TestStep::Wait {
            selector,
            text,
            timeout_ms,
            ..
        } => {
            assert_eq!(selector.as_deref(), Some("#panel"));
            assert_eq!(text.as_deref(), Some("Loaded"));
            assert_eq!(timeout_ms, &Some(8000));
        }
        _ => panic!("expected Wait step"),
    }
}

#[test]
fn test_viewport_matrix_config() {
    let toml = r#"
[config]
viewport_width = 1280
viewport_height = 720

[config.viewport_matrix]
viewports = [
    { name = "mobile", width = 390, height = 844 },
    { name = "desktop", width = 1280, height = 720 },
]

[[test]]
name = "smoke"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse matrix TOML");
    let matrix = scenario.config.viewport_matrix.expect("matrix present");
    assert_eq!(matrix.viewports.len(), 2);
    assert_eq!(matrix.viewports[0].name, "mobile");
    assert_eq!(matrix.viewports[0].width, 390);
    assert_eq!(matrix.viewports[0].height, 844);
    assert_eq!(matrix.viewports[1].name, "desktop");
}

#[test]
fn test_per_test_viewport_override() {
    let toml = r#"
[config]
viewport_width = 1280
viewport_height = 720

[[test]]
name = "mobile variant"
viewport_width = 390
viewport_height = 844
steps = []

[[test]]
name = "default viewport"
steps = []
"#;
    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let mobile = &scenario.test[0];
    assert_eq!(mobile.viewport_width, Some(390));
    assert_eq!(mobile.viewport_height, Some(844));
    let default_vp = &scenario.test[1];
    assert_eq!(default_vp.viewport_width, None);
    assert_eq!(default_vp.viewport_height, None);
}

#[test]
fn test_idempotent_steps_parse() {
    use llm_browser_testkit::scenario::TestStep;
    let toml = r##"
[[test]]
name = "flow"
steps = [
    { kind = "click", target = "the save button", idempotent = true },
    { kind = "type", target = "the email input", selector = "#email", text = "a@b.c", idempotent = true },
    { kind = "wait", target = "the shell", selector = "app-shell", timeout_ms = 5000, idempotent = true },
    { kind = "click", target = "the other button" },
]
"##;
    let scenario: Scenario = toml::from_str(toml).expect("parse idempotent steps");
    let steps = &scenario.test[0].steps;
    match &steps[0] {
        TestStep::Click { idempotent, .. } => assert!(*idempotent),
        other => panic!("expected Click, got {other:?}"),
    }
    match &steps[1] {
        TestStep::Type { idempotent, .. } => assert!(*idempotent),
        other => panic!("expected Type, got {other:?}"),
    }
    match &steps[2] {
        TestStep::Wait {
            idempotent,
            timeout_ms,
            ..
        } => {
            assert!(*idempotent);
            assert_eq!(*timeout_ms, Some(5000));
        }
        other => panic!("expected Wait, got {other:?}"),
    }
    match &steps[3] {
        TestStep::Click { idempotent, .. } => assert!(!*idempotent),
        other => panic!("expected non-idempotent Click, got {other:?}"),
    }
}
