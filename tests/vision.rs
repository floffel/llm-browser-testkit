//! Tests for vision support: endpoint `vision` flag parsing, `screenshot`
//! on assert steps, the vision chat payload shape, and the non-vision
//! endpoint guard.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use llm_browser_testkit::endpoints::{EndpointRegistry, TaskType};
use llm_browser_testkit::llm_chat_vision_with_usage;
use llm_browser_testkit::LlmConfig;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(url: &str) -> LlmConfig {
    LlmConfig {
        url: url.trim_end_matches('/').to_owned(),
        model: "vision-model".to_owned(),
        ..LlmConfig::default()
    }
}

fn ok_body() -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"content": "PASS"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2}
    })
}

#[test]
fn test_endpoint_vision_flag_parsed() {
    let toml = r#"
[config.endpoints.vision]
type = "llm"
url = "http://localhost:8080"
model = "gpt-4o"
vision = true
default_for = []

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    let ep = &scenario.config.endpoints["vision"];
    assert!(ep.vision, "vision flag must parse");

    let registry = EndpointRegistry::from_config(&scenario.config.endpoints, None);
    let resolved = registry.get("vision").expect("resolved endpoint");
    assert!(resolved.vision);
}

#[test]
fn test_endpoint_vision_default_false() {
    let toml = r#"
[config.endpoints.default]
type = "llm"
url = "http://localhost:8080"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    assert!(!scenario.config.endpoints["default"].vision);
}

#[test]
fn test_assert_step_screenshot_flag_parsed() {
    let toml = r#"
[[test]]
name = "visual"
steps = [
    { kind = "assert", preset = "visual_no_overlaps", screenshot = true, endpoint = "vision" },
    { kind = "assert", preset = "no_error_on_page" }
]
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    let steps = &scenario.test[0].steps;
    match &steps[0] {
        llm_browser_testkit::scenario::TestStep::Assert {
            preset,
            screenshot,
            endpoint,
            ..
        } => {
            assert_eq!(preset.as_deref(), Some("visual_no_overlaps"));
            assert!(screenshot, "screenshot flag must parse as true");
            assert_eq!(endpoint.as_deref(), Some("vision"));
        }
        _ => panic!("expected Assert step"),
    }
    match &steps[1] {
        llm_browser_testkit::scenario::TestStep::Assert { screenshot, .. } => {
            assert!(!screenshot, "screenshot defaults to false");
        }
        _ => panic!("expected Assert step"),
    }
}

#[test]
fn test_screenshot_max_dimension_parsed() {
    let toml = r#"
[config]
screenshot_max_dimension = 1024

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    assert_eq!(scenario.config.screenshot_max_dimension, Some(1024));
}

#[test]
fn test_screenshot_max_height_pixels_parsed() {
    let toml = r#"
[config]
screenshot_max_height = 2880

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    let Some(h) = scenario.config.screenshot_max_height else {
        panic!("screenshot_max_height must parse");
    };
    match h {
        llm_browser_testkit::scenario::ScreenshotHeight::Pixels(px) => assert_eq!(px, 2880),
        llm_browser_testkit::scenario::ScreenshotHeight::ViewportTimes(_) => {
            panic!("expected Pixels variant")
        }
    }
}

#[test]
fn test_screenshot_max_height_times_parsed() {
    let toml = r#"
[config]
screenshot_max_height = "2x"

[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    let Some(h) = scenario.config.screenshot_max_height else {
        panic!("screenshot_max_height must parse");
    };
    match h {
        llm_browser_testkit::scenario::ScreenshotHeight::ViewportTimes(m) => {
            // "2x" survives the f64 round-trip exactly.
            assert!((m - 2.0).abs() < 1e-9);
        }
        llm_browser_testkit::scenario::ScreenshotHeight::Pixels(_) => {
            panic!("expected ViewportTimes variant")
        }
    }
}

#[test]
fn test_screenshot_max_height_defaults_to_none() {
    let toml = r#"
[[test]]
name = "dummy"
steps = []
"#;
    let scenario: llm_browser_testkit::scenario::Scenario =
        toml::from_str(toml).expect("parse TOML");
    assert_eq!(scenario.config.screenshot_max_height, None);
}

#[test]
fn test_screenshot_max_height_invalid_rejected() {
    for invalid in ["\"2xp\"", "\"no\"", "\"-3x\"", "true"] {
        let toml = format!(
            r#"
[config]
screenshot_max_height = {invalid}

[[test]]
name = "dummy"
steps = []
"#
        );
        assert!(
            toml::from_str::<llm_browser_testkit::scenario::Scenario>(&toml).is_err(),
            "invalid screenshot_max_height {invalid} must be rejected"
        );
    }
}

#[test]
fn test_screenshot_max_height_resolution() {
    use llm_browser_testkit::scenario::ScreenshotHeight;
    assert_eq!(ScreenshotHeight::Pixels(600).to_px(720), 600);
    assert_eq!(ScreenshotHeight::Pixels(2880).to_px(844), 2880);
    assert_eq!(ScreenshotHeight::ViewportTimes(2.0).to_px(720), 1440);
    assert_eq!(ScreenshotHeight::ViewportTimes(1.5).to_px(844), 1266);
    assert_eq!(ScreenshotHeight::ViewportTimes(0.5).to_px(720), 360);
    assert_eq!(ScreenshotHeight::ViewportTimes(0.0).to_px(720), 1);
}

#[test]
fn test_vision_endpoint_resolves_for_assertion() {
    let mut eps = HashMap::new();
    let mut ep = llm_browser_testkit::scenario::EndpointConfig {
        endpoint_type: llm_browser_testkit::scenario::EndpointType::Llm,
        url: Some("http://vision".to_owned()),
        model: Some("gpt-4o".to_owned()),
        ..Default::default()
    };
    ep.vision = true;
    eps.insert("vision".to_owned(), ep);
    let registry = EndpointRegistry::from_config(&eps, None);
    assert!(registry.resolve(Some("vision"), TaskType::Assertion).vision);
}

#[tokio::test]
async fn test_vision_chat_sends_image_part() {
    let server = MockServer::start().await;
    let body_capture: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with({
            let body_capture = Arc::clone(&body_capture);
            move |req: &wiremock::Request| {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                    *body_capture.lock().unwrap() = Some(json);
                }
                ResponseTemplate::new(200).set_body_json(ok_body())
            }
        })
        .mount(&server)
        .await;

    let urls = vec!["data:image/jpeg;base64,/9j/4AAQSkZJRg==".to_owned()];
    let resp =
        llm_chat_vision_with_usage(&config(&server.uri()), "sys", "inspect this", Some(&urls))
            .await
            .expect("vision chat succeeds");
    assert_eq!(resp.content, "PASS");

    let body = body_capture
        .lock()
        .unwrap()
        .clone()
        .expect("request captured");
    let user_content = &body["messages"][1]["content"];
    assert!(user_content.is_array(), "user content must be a part array");
    let parts = user_content.as_array().unwrap();
    assert_eq!(parts.len(), 2, "text part + image part");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "inspect this");
    assert_eq!(parts[1]["type"], "image_url");
    let url = parts[1]["image_url"]["url"].as_str().unwrap();
    assert!(
        url.starts_with("data:image/jpeg;base64,"),
        "data URL prefix, got {url:?}"
    );
    assert!(url.len() > 30, "payload carries base64 image data");
}

#[tokio::test]
async fn test_vision_chat_sends_multiple_image_parts() {
    let server = MockServer::start().await;
    let body_capture: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with({
            let body_capture = Arc::clone(&body_capture);
            move |req: &wiremock::Request| {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                    *body_capture.lock().unwrap() = Some(json);
                }
                ResponseTemplate::new(200).set_body_json(ok_body())
            }
        })
        .mount(&server)
        .await;

    let urls = vec![
        "data:image/jpeg;base64,/9j/top-tile".to_owned(),
        "data:image/jpeg;base64,/9j/second-tile".to_owned(),
    ];
    let resp = llm_chat_vision_with_usage(
        &config(&server.uri()),
        "sys",
        "inspect the page",
        Some(&urls),
    )
    .await
    .expect("vision chat with tiles succeeds");
    assert_eq!(resp.content, "PASS");

    let body = body_capture
        .lock()
        .unwrap()
        .clone()
        .expect("request captured");
    let parts = body["messages"][1]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 3, "text part + two image parts");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(
        parts[1]["image_url"]["url"].as_str().unwrap(),
        "data:image/jpeg;base64,/9j/top-tile",
        "tiles stay ordered from the top of the page"
    );
    assert_eq!(
        parts[2]["image_url"]["url"].as_str().unwrap(),
        "data:image/jpeg;base64,/9j/second-tile"
    );
}

#[test]
fn test_tile_count() {
    use llm_browser_testkit::vision::tile_count;
    assert_eq!(tile_count(0, 720), 0);
    assert_eq!(tile_count(720, 720), 1);
    assert_eq!(tile_count(1440, 720), 2);
    assert_eq!(tile_count(1500, 720), 3);
    assert_eq!(tile_count(2880, 720), 4);
    assert_eq!(tile_count(720, 0), 1);
    let max = llm_browser_testkit::vision::MAX_TILES;
    assert_eq!(tile_count(max * 720 + 1, 720), max);
}

#[tokio::test]
async fn test_text_chat_keeps_string_content() {
    let server = MockServer::start().await;
    let body_capture: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with({
            let body_capture = Arc::clone(&body_capture);
            move |req: &wiremock::Request| {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                    *body_capture.lock().unwrap() = Some(json);
                }
                ResponseTemplate::new(200).set_body_json(ok_body())
            }
        })
        .mount(&server)
        .await;

    let resp = llm_browser_testkit::llm_chat_with_usage(&config(&server.uri()), "sys", "check")
        .await
        .expect("text chat succeeds");
    assert_eq!(resp.content, "PASS");

    let body = body_capture
        .lock()
        .unwrap()
        .clone()
        .expect("request captured");
    assert_eq!(body["messages"][1]["content"], "check");
}
