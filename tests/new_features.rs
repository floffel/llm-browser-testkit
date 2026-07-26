//! Tests for the new scenario types: endpoints, budgets, MCP server config,
//! agent tasks, and MCP steps.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use llm_browser_testkit::scenario::Scenario;

#[test]
fn test_endpoints_config() {
    let toml = r#"
[config.endpoints.default]
type = "llm"
url = "http://localhost:8080"
model = "deepseek"
pricing = { input_per_1m_tokens = 0.15, output_per_1m_tokens = 0.60 }

[config.endpoints.my_mcp]
type = "mcp"
command = "npx"
args = ["-y", "some-mcp-server"]
pricing = { per_call = 0.001 }

[config.endpoints.my_agent]
type = "a2a"
url = "http://localhost:9090"
pricing = { per_call = 0.01 }

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let eps = &scenario.config.endpoints;
    assert_eq!(eps.len(), 3);

    let default_ep = eps.get("default").unwrap();
    assert_eq!(default_ep.url.as_deref(), Some("http://localhost:8080"));
    assert_eq!(default_ep.model.as_deref(), Some("deepseek"));
    assert!(default_ep.pricing.is_some());

    let mcp_ep = eps.get("my_mcp").unwrap();
    assert_eq!(mcp_ep.command.as_deref(), Some("npx"));
    assert_eq!(mcp_ep.args.len(), 2);

    let agent_ep = eps.get("my_agent").unwrap();
    assert_eq!(agent_ep.url.as_deref(), Some("http://localhost:9090"));
    assert!(agent_ep.pricing.as_ref().unwrap().per_call > 0.0);
}

#[test]
fn test_budgets_config() {
    let toml = r#"
[config.budgets.global]
max_cost = 5.0
max_tokens = 500_000
enforcement = "hard"

[config.budgets.per_test_default]
max_cost = 1.0
max_tokens = 100_000
enforcement = "soft"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let global = scenario.config.budgets.global.as_ref().unwrap();
    assert!((global.max_cost.unwrap() - 5.0).abs() < f64::EPSILON);
    assert_eq!(global.max_tokens, Some(500_000));
    assert_eq!(
        global.enforcement.as_ref().unwrap().to_owned(),
        llm_browser_testkit::scenario::BudgetEnforcement::Hard
    );

    let per_test = scenario.config.budgets.per_test_default.as_ref().unwrap();
    assert!((per_test.max_cost.unwrap() - 1.0).abs() < f64::EPSILON);
    assert_eq!(per_test.max_tokens, Some(100_000));
}

#[test]
fn test_per_test_budget_override() {
    let toml = r#"
[[test]]
name = "expensive"
budget = { max_cost = 2.0, max_tokens = 200_000, enforcement = "soft" }
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let test = &scenario.test[0];
    let budget = test.budget.as_ref().unwrap();
    assert!((budget.max_cost.unwrap() - 2.0).abs() < f64::EPSILON);
    assert_eq!(budget.max_tokens, Some(200_000));
}

#[test]
fn test_agent_step() {
    let toml = r#"
[[test]]
name = "agent test"
steps = [
    { kind = "agent", agent = "my_agent", task = "Check something" }
]
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let step = &scenario.test[0].steps[0];
    match step {
        llm_browser_testkit::scenario::TestStep::Agent { agent, task, .. } => {
            assert_eq!(agent, "my_agent");
            assert_eq!(task, "Check something");
        }
        _ => panic!(
            "expected Agent step, got {:?}",
            std::mem::discriminant(step)
        ),
    }
}

#[test]
fn test_mcp_step() {
    let toml = r#"
[[test]]
name = "mcp test"
steps = [
    { kind = "mcp", server = "db", tool = "query", args = { sql = "SELECT 1" } }
]
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let step = &scenario.test[0].steps[0];
    match step {
        llm_browser_testkit::scenario::TestStep::Mcp { server, tool, .. } => {
            assert_eq!(server, "db");
            assert_eq!(tool, "query");
        }
        _ => panic!("expected Mcp step"),
    }
}

#[test]
fn test_agent_definition() {
    let toml = r#"
[[definitions]]
name = "agent_check"
agent = "my_agent"
task_template = "Is {expected_text} present on {url}?"

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let def = &scenario.definitions[0];
    assert_eq!(def.agent.as_deref(), Some("my_agent"));
    assert_eq!(
        def.task_template.as_deref(),
        Some("Is {expected_text} present on {url}?")
    );
}

#[test]
fn test_mcp_server_config() {
    let toml = r#"
[config.mcp_server]
enabled = true
port = 4567

[[test]]
name = "dummy"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let mcp = scenario.config.mcp_server.as_ref().unwrap();
    assert!(mcp.enabled);
    assert_eq!(mcp.port, 4567);
}

#[test]
fn test_step_endpoint_override() {
    let toml = r#"
[[test]]
name = "endpoint test"
steps = [
    { kind = "click", target = "button", endpoint = "vision" },
    { kind = "assert", preset = "no_error_on_page", endpoint = "fast" }
]
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    let steps = &scenario.test[0].steps;
    if let llm_browser_testkit::scenario::TestStep::Click { endpoint, .. } = &steps[0] {
        assert_eq!(endpoint.as_deref(), Some("vision"));
    } else {
        panic!("expected Click step");
    }
    if let llm_browser_testkit::scenario::TestStep::Assert { endpoint, .. } = &steps[1] {
        assert_eq!(endpoint.as_deref(), Some("fast"));
    } else {
        panic!("expected Assert step");
    }
}

#[test]
fn test_test_group_endpoint() {
    let toml = r#"
[[test]]
name = "vision test"
endpoint = "vision"
steps = []
"#;

    let scenario: Scenario = toml::from_str(toml).expect("parse TOML");
    assert_eq!(scenario.test[0].endpoint.as_deref(), Some("vision"));
}
