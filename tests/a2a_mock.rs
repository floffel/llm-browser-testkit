//! Mock tests for A2A client using wiremock.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::time::Duration;

use llm_browser_testkit::a2a::A2aClient;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_a2a_send_task_success() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "result": {
                "id": "task-1",
                "messages": [{
                    "parts": [{"type": "text", "text": "PASS: all good"}]
                }]
            },
            "id": 1
        })))
        .mount(&server)
        .await;

    let client = A2aClient::new(&server.uri(), Duration::from_secs(10));
    let result = client.send_task("check page").await.unwrap();
    assert_eq!(result, "PASS: all good");
}

#[tokio::test]
async fn test_a2a_send_task_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            },
            "id": 1
        })))
        .mount(&server)
        .await;

    let client = A2aClient::new(&server.uri(), Duration::from_secs(10));
    let result = client.send_task("check page").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid Request"));
}

#[tokio::test]
async fn test_a2a_send_task_fallback_full_json() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "result": {"status": "completed", "data": "some-raw-result"},
            "id": 1
        })))
        .mount(&server)
        .await;

    let client = A2aClient::new(&server.uri(), Duration::from_secs(10));
    let result = client.send_task("check page").await.unwrap();
    // Falls back to raw JSON string
    assert!(result.contains("completed"));
}

#[tokio::test]
async fn test_a2a_send_task_server_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let client = A2aClient::new(&server.uri(), Duration::from_secs(10));
    let result = client.send_task("check page").await;
    // 500 error — response parsing may fail
    assert!(result.is_err());
}

#[tokio::test]
async fn test_a2a_send_task_unreachable_server() {
    let client = A2aClient::new("http://127.0.0.1:19999", Duration::from_secs(1));
    let result = client.send_task("check page").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("request failed"));
}
