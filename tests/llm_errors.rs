//! Tests for the LLM client error reporting: HTTP status + body snippets in
//! error messages, retry behavior for transient failures, and fail-fast on
//! deterministic client errors.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::time::Duration;

use llm_browser_testkit::llm_chat_with_usage;
use llm_browser_testkit::LlmConfig;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(url: &str) -> LlmConfig {
    LlmConfig {
        url: url.trim_end_matches('/').to_owned(),
        model: "mock-model".to_owned(),
        timeout: Duration::from_secs(10),
        max_attempts: 3,
        ..LlmConfig::default()
    }
}

fn ok_body() -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"content": "PASS"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1}
    })
}

#[tokio::test]
async fn test_success_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
        .mount(&server)
        .await;

    let resp = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect("chat should succeed");
    assert_eq!(resp.content, "PASS");
    assert_eq!(resp.usage.prompt_tokens, 5);
}

#[tokio::test]
async fn test_http_error_message_includes_status_and_snippet() {
    let server = MockServer::start().await;
    let html = "<html><body>Bad Gateway — upstream offline</body></html>";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(502).set_body_string(html))
        .mount(&server)
        .await;

    let err = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect_err("502 should fail");
    assert!(err.contains("HTTP 502"), "err: {err}");
    assert!(err.contains("Bad Gateway"), "err: {err}");
    assert!(err.contains("attempt"), "err: {err}");
}

#[tokio::test]
async fn test_non_json_body_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>rate limit html</html>"))
        .mount(&server)
        .await;

    let err = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect_err("non-JSON 200 should fail");
    assert!(err.contains("non-JSON"), "err: {err}");
    assert!(err.contains("HTTP 200"), "err: {err}");
    assert!(err.contains("rate limit html"), "err: {err}");
}

#[tokio::test]
async fn test_missing_content_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"role": "assistant"}}]
        })))
        .mount(&server)
        .await;

    let err = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect_err("missing content should fail");
    assert!(err.contains("choices[0].message.content"), "err: {err}");
}

#[tokio::test]
async fn test_client_error_fails_fast() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    let err = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect_err("401 should fail");
    assert!(err.contains("HTTP 401"), "err: {err}");
    // Deterministic client errors are not retried — only 1 attempt.
    assert!(err.contains("1 attempt"), "err: {err}");
}

#[tokio::test]
async fn test_transient_then_success_recovers() {
    let server = MockServer::start().await;
    let failures = std::sync::atomic::AtomicU32::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            if failures.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                ResponseTemplate::new(429).set_body_string("slow down")
            } else {
                ResponseTemplate::new(200).set_body_json(ok_body())
            }
        })
        .mount(&server)
        .await;

    let resp = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect("retry should recover");
    assert_eq!(resp.content, "PASS");
}

#[tokio::test]
async fn test_empty_200_is_transient_and_recovers() {
    let server = MockServer::start().await;
    let failures = std::sync::atomic::AtomicU32::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            if failures.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                ResponseTemplate::new(200).set_body_string("")
            } else {
                ResponseTemplate::new(200).set_body_json(ok_body())
            }
        })
        .mount(&server)
        .await;

    let resp = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect("empty-200 retry should recover");
    assert_eq!(resp.content, "PASS");
}

#[tokio::test]
async fn test_empty_200_message_names_gateway_warmup() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .mount(&server)
        .await;

    let err = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect_err("empty 200 should fail after all attempts");
    assert!(err.contains("HTTP 200"), "err: {err}");
    assert!(err.contains("empty response"), "err: {err}");
    assert!(err.contains("warm-up"), "err: {err}");
    // Retried like a transient failure, not failed fast.
    assert!(err.contains("3 attempt"), "err: {err}");
}

#[tokio::test]
async fn test_request_payload_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body()))
        .mount(&server)
        .await;

    let resp = llm_chat_with_usage(&config(&server.uri()), "sys", "user")
        .await
        .expect("chat should succeed");
    assert_eq!(resp.content, "PASS");
}
