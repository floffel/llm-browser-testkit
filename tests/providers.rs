//! Integration tests for the `Azure` `OpenAI` and AWS Bedrock LLM providers:
//! URL construction, API-key vs. bearer auth, `Entra` ID token flows,
//! token/header commands, caching, and `SigV4` request signing.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::collections::HashMap;
use std::time::Duration;

use llm_browser_testkit::AuthConfig;
use llm_browser_testkit::AuthMode;
use llm_browser_testkit::AwsConfig;
use llm_browser_testkit::LlmConfig;
use llm_browser_testkit::Provider;
use wiremock::matchers::body_string_contains;
use wiremock::matchers::header;
use wiremock::matchers::header_exists;
use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::matchers::query_param;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;

const OPENAI_RESPONSE: &str = r#"{
    "choices": [{"message": {"content": "hello agent"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
}"#;

const CONVERSE_RESPONSE: &str = r#"{
    "output": {"message": {"role": "assistant", "content": [{"text": "bedrock says hi"}]}},
    "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
    "stopReason": "end_turn"
}"#;

fn llm_config(url: &str, provider: Provider) -> LlmConfig {
    LlmConfig {
        url: url.to_owned(),
        model: "gpt-4o".to_owned(),
        api_key: None,
        headers: HashMap::new(),
        timeout: Duration::from_secs(30),
        temperature: 0.0,
        thinking: None,
        model_params: HashMap::new(),
        max_attempts: 1,
        provider,
        deployment: None,
        api_version: None,
        auth: AuthConfig::default(),
        header_commands: HashMap::new(),
        aws: AwsConfig::default(),
    }
}

async fn mock_chat(server: &MockServer, body: &str) {
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(server)
        .await;
}

/// Reads a request header value (http 1.x `HeaderValue` has no `as_str`).
fn req_header<'h>(req: &'h wiremock::Request, name: &str) -> Option<&'h str> {
    req.headers.get(name).and_then(|v| v.to_str().ok())
}

// ---------------------------------------------------------------------------
// Azure OpenAI
// ---------------------------------------------------------------------------

#[tokio::test]
async fn azure_uses_deployment_url_and_api_key_header() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Azure);
    llm.api_key = Some("sk-azure".to_owned());
    llm.deployment = Some("my-deployment".to_owned());

    let resp = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("azure chat succeeds");
    assert_eq!(resp.content, "hello agent");
    assert_eq!(resp.usage.prompt_tokens, 10);

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(reqs.len(), 1);
    let req = &reqs[0];
    assert_eq!(
        req.url.path(),
        "/openai/deployments/my-deployment/chat/completions"
    );
    assert_eq!(
        req.url.query(),
        Some("api-version=2024-10-21"),
        "default api-version applies"
    );
    assert_eq!(
        req_header(req, "api-key"),
        Some("sk-azure"),
        "classic azure api-key header"
    );
    assert!(req.headers.get("authorization").is_none());
    let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
    assert_eq!(body["model"], "gpt-4o");
}

#[tokio::test]
async fn azure_deployment_defaults_to_model() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Azure);
    llm.api_key = Some("sk-azure".to_owned());

    let resp = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("azure chat succeeds");
    assert_eq!(resp.content, "hello agent");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(reqs.len(), 1);
    assert!(reqs[0]
        .url
        .path()
        .ends_with("/openai/deployments/gpt-4o/chat/completions"));
}

#[tokio::test]
async fn azure_honors_api_version_and_trailing_slash() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Azure);
    llm.api_key = Some("sk-azure".to_owned());
    llm.api_version = Some("2025-01-01".to_owned());

    llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("azure chat succeeds");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(
        reqs[0].url.query(),
        Some("api-version=2025-01-01"),
        "explicit api-version wins"
    );
}

// ---------------------------------------------------------------------------
// Auth modes
// ---------------------------------------------------------------------------

#[tokio::test]
#[cfg(unix)]
async fn token_command_runs_and_sets_bearer() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Openai);
    llm.auth = AuthConfig {
        mode: AuthMode::TokenCommand,
        token_command: Some("printf 'cli-token-1\\n'".to_owned()),
        ..AuthConfig::default()
    };

    let resp = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("chat with command token succeeds");
    assert_eq!(resp.content, "hello agent");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(
        req_header(&reqs[0], "authorization"),
        Some("Bearer cli-token-1")
    );
}

#[tokio::test]
async fn api_key_header_override_applies_to_openai() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Openai);
    llm.api_key = Some("sk-custom".to_owned());
    llm.auth = AuthConfig {
        api_key_header: Some("X-Api-Key".to_owned()),
        ..AuthConfig::default()
    };

    llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("chat succeeds");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(
        req_header(&reqs[0], "x-api-key"),
        Some("sk-custom"),
        "custom header receives the api key"
    );
    assert!(reqs[0].headers.get("authorization").is_none());
}

#[tokio::test]
#[cfg(unix)]
async fn static_and_command_headers_are_sent() {
    let server = MockServer::start().await;
    mock_chat(&server, OPENAI_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Openai);
    llm.headers
        .insert("X-Static".to_owned(), "static-value".to_owned());
    llm.header_commands
        .insert("X-From-Cmd".to_owned(), "printf 'cmd-value\\n'".to_owned());

    llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("chat succeeds");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(req_header(&reqs[0], "x-static"), Some("static-value"));
    assert_eq!(req_header(&reqs[0], "x-from-cmd"), Some("cmd-value"));
}

// ---------------------------------------------------------------------------
// Entra ID
// ---------------------------------------------------------------------------

#[tokio::test]
async fn entra_client_credentials_exchange_and_cache() {
    let token_server = MockServer::start().await;
    let chat_server = MockServer::start().await;
    mock_chat(&chat_server, OPENAI_RESPONSE).await;

    Mock::given(method("POST"))
        .and(path("/tenant-123/oauth2/v2.0/token"))
        .and(body_string_contains("grant_type=client_credentials"))
        .and(body_string_contains("client_id=client-456"))
        .and(body_string_contains("client_secret=super-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "entra-token-1",
            "expires_in": 3600,
            "token_type": "Bearer"
        })))
        .mount(&token_server)
        .await;

    let mut llm = llm_config(&chat_server.uri(), Provider::Openai);
    llm.auth = AuthConfig {
        mode: AuthMode::EntraClientCredentials,
        tenant_id: Some("tenant-123".to_owned()),
        client_id: Some("client-456".to_owned()),
        client_secret: Some("super-secret".to_owned()),
        token_url: Some(token_server.uri()),
        ..AuthConfig::default()
    };

    let first = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("first chat succeeds");
    let second = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("second chat succeeds");
    assert_eq!(first.content, "hello agent");
    assert_eq!(second.content, "hello agent");

    let chat_reqs = chat_server
        .received_requests()
        .await
        .expect("requests seen");
    assert_eq!(chat_reqs.len(), 2);
    for req in &chat_reqs {
        assert_eq!(
            req_header(req, "authorization"),
            Some("Bearer entra-token-1")
        );
    }
    let token_reqs = token_server
        .received_requests()
        .await
        .expect("requests seen");
    assert_eq!(token_reqs.len(), 1, "token is fetched once and cached");
}

#[tokio::test]
async fn entra_client_credentials_reports_server_errors() {
    let token_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "invalid_client",
            "error_description": "bad secret"
        })))
        .mount(&token_server)
        .await;

    let mut llm = llm_config("http://127.0.0.1:1", Provider::Openai);
    llm.auth = AuthConfig {
        mode: AuthMode::EntraClientCredentials,
        tenant_id: Some("tenant-123".to_owned()),
        client_id: Some("client-456".to_owned()),
        client_secret: Some("wrong-secret".to_owned()),
        token_url: Some(token_server.uri()),
        ..AuthConfig::default()
    };

    let err = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect_err("entra error surfaces");
    assert!(err.contains("invalid_client"), "got: {err}");
}

#[tokio::test]
async fn entra_managed_identity_uses_imds() {
    let imds_server = MockServer::start().await;
    let chat_server = MockServer::start().await;
    mock_chat(&chat_server, OPENAI_RESPONSE).await;

    let expires_on = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("post-epoch")
        .as_secs()
        + 3600;

    Mock::given(method("GET"))
        .and(path("/metadata/identity/oauth2/token"))
        .and(query_param("api-version", "2018-02-01"))
        .and(query_param(
            "resource",
            "https://cognitiveservices.azure.com/",
        ))
        .and(header("metadata", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "imds-token-1",
            "expires_on": expires_on.to_string(),
            "token_type": "Bearer"
        })))
        .mount(&imds_server)
        .await;

    let mut llm = llm_config(&chat_server.uri(), Provider::Openai);
    llm.auth = AuthConfig {
        mode: AuthMode::EntraManagedIdentity,
        token_url: Some(imds_server.uri()),
        ..AuthConfig::default()
    };

    let resp = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("managed identity chat succeeds");
    assert_eq!(resp.content, "hello agent");

    let chat_reqs = chat_server
        .received_requests()
        .await
        .expect("requests seen");
    assert_eq!(
        req_header(&chat_reqs[0], "authorization"),
        Some("Bearer imds-token-1")
    );
}

// ---------------------------------------------------------------------------
// AWS Bedrock
// ---------------------------------------------------------------------------

#[cfg(feature = "aws")]
#[tokio::test]
async fn bedrock_signed_chat_request() {
    let server = MockServer::start().await;
    mock_chat(&server, CONVERSE_RESPONSE).await;

    let mut llm = llm_config(&server.uri(), Provider::Bedrock);
    llm.model = "anthropic.claude-3-5-sonnet".to_owned();
    llm.temperature = 0.2;
    llm.thinking = Some(true);
    llm.aws = AwsConfig {
        access_key_id: Some("AKIATESTKEY".to_owned()),
        secret_access_key: Some("secrettestkey".to_owned()),
        region: Some("eu-central-1".to_owned()),
        ..AwsConfig::default()
    };

    let resp = llm_browser_testkit::llm_chat_with_usage(&llm, "you are a QA", "check page")
        .await
        .expect("bedrock chat succeeds");
    assert_eq!(resp.content, "bedrock says hi");
    assert_eq!(resp.usage.prompt_tokens, 7);
    assert_eq!(resp.usage.completion_tokens, 3);
    assert_eq!(resp.usage.total_tokens, 10);

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(reqs.len(), 1);
    let req = &reqs[0];

    let auth = req
        .headers
        .get("authorization")
        .expect("signed request carries authorization");
    let auth = auth.to_str().expect("ascii header");
    assert!(
        auth.starts_with("AWS4-HMAC-SHA256"),
        "expected SigV4 signature, got: {auth}"
    );
    assert!(
        auth.contains("/eu-central-1/bedrock/aws4_request"),
        "signature scope includes region and service: {auth}"
    );
    assert!(
        req.headers.get("x-amz-content-sha256").is_none(),
        "the payload checksum header is only sent when a service requires it (S3)"
    );
    assert!(
        req.headers.get("x-amz-date").is_some(),
        "date header attached"
    );

    let body: serde_json::Value = serde_json::from_slice(&req.body).expect("json body");
    assert_eq!(body["system"][0]["text"], "you are a QA");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"][0]["text"], "check page");
    assert_eq!(body["inferenceConfig"]["temperature"], 0.2);
    assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
    assert_eq!(body["inferenceConfig"]["thinking"]["type"], "enabled");
}

#[cfg(feature = "aws")]
#[tokio::test]
async fn bedrock_missing_region_errors_clearly() {
    let mut llm = llm_config("http://127.0.0.1:1", Provider::Bedrock);
    llm.aws = AwsConfig {
        access_key_id: Some("AKIATESTKEY".to_owned()),
        secret_access_key: Some("secrettestkey".to_owned()),
        ..AwsConfig::default()
    };

    let err = llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect_err("missing region must fail");
    assert!(err.contains("region"), "got: {err}");
}

#[cfg(feature = "aws")]
#[tokio::test]
#[cfg(unix)]
async fn bedrock_custom_headers_and_commands_are_signed_and_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header_exists("authorization"))
        .and(header("x-agent", "test-runner"))
        .respond_with(ResponseTemplate::new(200).set_body_string(CONVERSE_RESPONSE))
        .mount(&server)
        .await;

    let mut llm = llm_config(&server.uri(), Provider::Bedrock);
    llm.headers
        .insert("X-Agent".to_owned(), "test-runner".to_owned());
    llm.header_commands
        .insert("X-From-Cmd".to_owned(), "printf 'signed-cmd\\n'".to_owned());
    llm.aws = AwsConfig {
        access_key_id: Some("AKIATESTKEY".to_owned()),
        secret_access_key: Some("secrettestkey".to_owned()),
        region: Some("us-east-1".to_owned()),
        ..AwsConfig::default()
    };

    llm_browser_testkit::llm_chat_with_usage(&llm, "sys", "user")
        .await
        .expect("signed chat with custom headers succeeds");

    let reqs = server.received_requests().await.expect("requests seen");
    assert_eq!(req_header(&reqs[0], "x-from-cmd"), Some("signed-cmd"));
}
