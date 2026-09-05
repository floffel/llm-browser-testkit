//! AWS Bedrock provider — `SigV4`-signed Converse API calls.
//!
//! Enabled by the `aws` cargo feature. Uses the standard AWS credential
//! chain through `aws-config` (env vars, shared config, SSO, ECS/IMDS) with
//! optional `profile` / `region` overrides, or explicit access keys from the
//! endpoint's `[config.endpoints.<name>.aws]` table. Requests are signed
//! with `SigV4` using `aws-sigv4` and sent with the regular HTTP client, so
//! retries, budgets, and cost tracking behave exactly like every other
//! provider.
//!
//! The request/response mapping follows the Bedrock Converse API:
//! - system prompt → top-level `system` block
//! - user message → `content` blocks (`text`, plus `image` for vision)
//! - `temperature` / `maxTokens` → `inferenceConfig`
//! - usage → `inputTokens` / `outputTokens`

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex};

use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::sign;
use aws_sigv4::http_request::SignableBody;
use aws_sigv4::http_request::SignableRequest;
use aws_sigv4::http_request::SigningSettings;
use aws_sigv4::sign::v4;

use crate::auth;
use crate::costs::LlmResponse;
use crate::costs::LlmUsage;
use crate::scenario::AwsConfig;
use crate::LlmCallError;
use crate::LlmConfig;

/// AWS service name used for `SigV4` signing.
const BEDROCK_SERVICE: &str = "bedrock";

/// Resolved credentials plus the region they will be used with.
struct BedrockCredentials {
    /// Resolved AWS credentials.
    credentials: aws_credential_types::Credentials,
    /// Region for the Bedrock endpoint and signature.
    region: String,
}

impl Clone for BedrockCredentials {
    fn clone(&self) -> Self {
        Self {
            credentials: self.credentials.clone(),
            region: self.region.clone(),
        }
    }
}

/// Cached credentials per configuration fingerprint. The AWS SDK config
/// load (profile parsing, SSO handling) is expensive; resolving it once per
/// unique config keeps per-call overhead at zero.
static CREDENTIAL_CACHE: LazyLock<Mutex<Option<(u64, BedrockCredentials)>>> =
    LazyLock::new(|| Mutex::new(None));

/// Executes one Bedrock Converse chat completion against the endpoint
/// resolved from `llm` (explicit URL when set, otherwise the standard
/// `https://bedrock-runtime.<region>.amazonaws.com/model/<model>/converse`
/// endpoint).
pub async fn chat_once(
    client: &reqwest::Client,
    llm: &LlmConfig,
    system: &str,
    user: &str,
    image_data_url: Option<&str>,
) -> Result<LlmResponse, LlmCallError> {
    let creds = resolve_credentials(&llm.aws)
        .await
        .map_err(|e| LlmCallError::Auth { message: e })?;

    let url = if llm.url.trim().is_empty() {
        format!(
            "https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse",
            region = creds.region,
            model = llm.model
        )
    } else {
        llm.url.trim_end_matches('/').to_owned()
    };

    let payload = build_payload(llm, system, user, image_data_url);
    let body = serde_json::to_vec(&payload).map_err(|e| LlmCallError::Auth {
        message: format!("serializing Bedrock payload: {e}"),
    })?;

    let mut headers: Vec<(String, String)> = vec![
        ("content-type".to_owned(), "application/json".to_owned()),
        ("accept".to_owned(), "application/json".to_owned()),
    ];
    for (name, value) in &llm.headers {
        validate_header(name, value)?;
        headers.push((name.clone(), value.clone()));
    }
    for (name, command) in &llm.header_commands {
        let value = auth::run_header_command(command)
            .await
            .map_err(|e| LlmCallError::Auth {
                message: format!("header command for `{name}` failed: {e}"),
            })?;
        if value.trim().is_empty() {
            return Err(LlmCallError::Auth {
                message: format!("header command for `{name}` produced an empty value"),
            });
        }
        headers.push((name.clone(), value));
    }

    let signature_headers = sign_request(&url, &body, &headers, &creds)?;

    let mut req = client.post(&url);
    for (name, value) in &headers {
        req = req.header(name.as_str(), value.as_str());
    }
    for (name, value) in &signature_headers {
        req = req.header(name.as_str(), value.as_str());
    }
    send_and_parse(req, body).await
}

/// Applies `SigV4` to a Bedrock request and returns the signature headers
/// (`authorization`, `x-amz-date`, …) to attach on top of the regular ones.
fn sign_request(
    url: &str,
    body: &[u8],
    headers: &[(String, String)],
    creds: &BedrockCredentials,
) -> Result<Vec<(String, String)>, LlmCallError> {
    let header_refs: Vec<(&str, &str)> = headers
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_str()))
        .collect();
    let signable = SignableRequest::new(
        "POST",
        url,
        header_refs.iter().copied(),
        SignableBody::Bytes(body),
    )
    .map_err(|e| LlmCallError::Auth {
        message: format!("building signable Bedrock request: {e}"),
    })?;

    // SigV4 with the default settings — the payload hash is signed into the
    // canonical request (the AWS SDK's standard, non-S3 behavior). The
    // `x-amz-content-sha256` header is only attached when a service requires
    // it (S3); Bedrock does not.
    let settings = SigningSettings::default();
    let identity: aws_smithy_runtime_api::client::identity::Identity =
        creds.credentials.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(creds.region.as_str())
        .name(BEDROCK_SERVICE)
        .time(std::time::SystemTime::now())
        .settings(settings)
        .build()
        .map_err(|e| LlmCallError::Auth {
            message: format!("building SigV4 signing params: {e}"),
        })?
        .into();
    let (instructions, _signature) = sign(signable, &params)
        .map_err(|e| LlmCallError::Auth {
            message: format!("SigV4 signing failed: {e}"),
        })?
        .into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect())
}

/// Sends the signed request and maps the response into `LlmResponse`.
async fn send_and_parse(
    req: reqwest::RequestBuilder,
    body: Vec<u8>,
) -> Result<LlmResponse, LlmCallError> {
    let resp = req
        .body(body)
        .send()
        .await
        .map_err(|e| LlmCallError::Transport {
            message: format!("Bedrock request failed: {e}"),
        })?;
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    if !(200..300).contains(&status) {
        return Err(LlmCallError::Http { status, body });
    }
    if body.trim().is_empty() {
        return Err(LlmCallError::EmptyBody { status });
    }
    let json: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return Err(LlmCallError::InvalidJson {
                status,
                detail: e.to_string(),
                body,
            });
        }
    };
    parse_response(&json).map_err(|_message| LlmCallError::MissingContent { json: body })
}

/// Resolves credentials for a Bedrock endpoint: explicit access keys win,
/// otherwise the standard AWS chain (`AWS_*` env vars, `~/.aws/config` +
/// `~/.aws/credentials`, SSO, ECS/IMDS) with optional `profile`/`region`
/// overrides.
///
/// # Errors
///
/// Returns a description when no credentials or region can be resolved.
async fn resolve_credentials(aws: &AwsConfig) -> Result<BedrockCredentials, String> {
    let fingerprint = credential_fingerprint(aws);
    if let Some((key, cached)) = CREDENTIAL_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
    {
        if *key == fingerprint {
            return Ok(cached.clone());
        }
    }

    let resolved = if let (Some(ak), Some(sk)) = (&aws.access_key_id, &aws.secret_access_key) {
        let credentials = aws_credential_types::Credentials::new(
            ak.clone(),
            sk.clone(),
            aws.session_token.clone(),
            None,
            "llm-browser-testkit-static",
        );
        let region = aws.region.clone().ok_or_else(|| {
            "provider = \"bedrock\" with explicit access keys requires the endpoint's \
             [config.endpoints.<name>.aws] region to be set"
                .to_owned()
        })?;
        BedrockCredentials {
            credentials,
            region,
        }
    } else {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(profile) = &aws.profile {
            loader = loader.profile_name(profile.clone());
        }
        if let Some(region) = &aws.region {
            loader = loader.region(aws_config::Region::new(region.clone()));
        }
        let sdk_config = loader.load().await;
        let provider = sdk_config.credentials_provider().ok_or_else(|| {
            "no AWS credentials provider available — set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (or AWS_PROFILE), or the endpoint's \
             [config.endpoints.<name>.aws] access_key_id + secret_access_key"
                .to_owned()
        })?;
        let credentials = provider
            .provide_credentials()
            .await
            .map_err(|e| format!("AWS credential chain failed: {e}"))?;
        let region = sdk_config
            .region()
            .map(|r| r.as_ref().to_owned())
            .or_else(|| aws.region.clone())
            .ok_or_else(|| {
                "no AWS region configured — set AWS_REGION, the profile's region, or the endpoint's \
                 [config.endpoints.<name>.aws] region"
                    .to_owned()
            })?;
        BedrockCredentials {
            credentials,
            region,
        }
    };

    *CREDENTIAL_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((fingerprint, resolved.clone()));
    Ok(resolved)
}

/// Builds the Converse request body for a chat completion.
#[must_use]
fn build_payload(
    llm: &LlmConfig,
    system: &str,
    user: &str,
    image_data_url: Option<&str>,
) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "messages".to_owned(),
        serde_json::json!([{
            "role": "user",
            "content": build_user_content(user, image_data_url),
        }]),
    );
    if !system.is_empty() {
        payload.insert("system".to_owned(), serde_json::json!([{ "text": system }]));
    }
    let mut inference = serde_json::Map::new();
    inference.insert("maxTokens".to_owned(), serde_json::json!(4096));
    inference.insert("temperature".to_owned(), serde_json::json!(llm.temperature));
    match llm.thinking {
        Some(true) => {
            inference.insert(
                "thinking".to_owned(),
                serde_json::json!({ "type": "enabled", "budgetTokens": 1024 }),
            );
        }
        Some(false) => {
            inference.insert(
                "thinking".to_owned(),
                serde_json::json!({ "type": "disabled" }),
            );
        }
        None => {}
    }
    payload.insert(
        "inferenceConfig".to_owned(),
        serde_json::Value::Object(inference),
    );
    for (key, value) in &llm.model_params {
        payload.insert(key.clone(), value.clone());
    }
    serde_json::Value::Object(payload)
}

/// Builds the user content blocks; image data URLs become Converse `image`
/// blocks with raw base64 bytes.
#[must_use]
fn build_user_content(user: &str, image_data_url: Option<&str>) -> serde_json::Value {
    let Some(url) = image_data_url else {
        return serde_json::json!([{ "text": user }]);
    };
    let mut blocks = vec![serde_json::json!({ "text": user })];
    if let Some((format, bytes)) = split_data_url(url) {
        blocks.push(serde_json::json!({
            "image": {
                "format": format,
                "source": { "bytes": bytes },
            }
        }));
    }
    serde_json::Value::Array(blocks)
}

/// Splits an `image` data URL into a Bedrock image format plus the base64
/// payload; unrecognized formats degrade to `jpeg`.
#[must_use]
fn split_data_url(url: &str) -> Option<(&'static str, String)> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;
    let rest = url.strip_prefix("data:image/")?;
    let (format, rest) = rest.split_once(';')?;
    let data = rest.strip_prefix("base64,")?;
    let bytes = STANDARD.encode(STANDARD.decode(data).ok()?);
    let format = if format.eq_ignore_ascii_case("png") {
        "png"
    } else {
        "jpeg"
    };
    Some((format, bytes))
}

/// Extracts content and usage from a Converse response.
///
/// # Errors
///
/// Returns a description when the response has no usable text content.
fn parse_response(json: &serde_json::Value) -> Result<LlmResponse, String> {
    let content: String = json["output"]["message"]["content"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<&str>>()
                .join("")
        })
        .unwrap_or_default();
    if content.is_empty() {
        return Err(format!(
            "Bedrock response missing output.message.content text: {}",
            crate::truncate(&json.to_string(), 300)
        ));
    }
    let usage = &json["usage"];
    Ok(LlmResponse {
        content,
        usage: LlmUsage {
            prompt_tokens: usage["inputTokens"].as_u64().unwrap_or(0),
            completion_tokens: usage["outputTokens"].as_u64().unwrap_or(0),
            total_tokens: usage["totalTokens"].as_u64().unwrap_or(0),
        },
    })
}

/// Appends a header to the signing list with validated name/value.
fn validate_header(name: &str, value: &str) -> Result<(), LlmCallError> {
    if name.trim().is_empty() || name.contains(['\r', '\n']) {
        return Err(LlmCallError::Auth {
            message: format!("invalid header name `{name}`"),
        });
    }
    if value.trim().is_empty() || value.contains(['\r', '\n']) {
        return Err(LlmCallError::Auth {
            message: format!("invalid header value for `{name}`"),
        });
    }
    Ok(())
}

/// Stable fingerprint of the credential configuration (secrets are not
/// part of the key beyond their presence).
fn credential_fingerprint(aws: &AwsConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    (aws.profile.as_deref(), aws.region.as_deref()).hash(&mut hasher);
    (
        aws.access_key_id.is_some(),
        aws.secret_access_key.is_some(),
        aws.session_token.is_some(),
    )
        .hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{build_payload, build_user_content, parse_response, split_data_url};
    use crate::costs::LlmResponse;
    use crate::scenario::AuthConfig;
    use crate::scenario::AwsConfig;
    use crate::LlmConfig;
    use std::collections::HashMap;

    fn llm_config() -> LlmConfig {
        LlmConfig {
            url: String::new(),
            model: "anthropic.claude-3-5-sonnet".to_owned(),
            api_key: None,
            headers: HashMap::new(),
            timeout: std::time::Duration::from_secs(10),
            temperature: 0.2,
            thinking: Some(true),
            model_params: HashMap::new(),
            max_attempts: 3,
            provider: crate::scenario::Provider::Bedrock,
            deployment: None,
            api_version: None,
            auth: AuthConfig::default(),
            header_commands: HashMap::new(),
            aws: AwsConfig::default(),
        }
    }

    #[test]
    fn payload_text_only() {
        let payload = build_payload(&llm_config(), "you are a QA", "check the page", None);
        assert_eq!(payload["system"][0]["text"], "you are a QA");
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(
            payload["messages"][0]["content"][0]["text"],
            "check the page"
        );
        assert_eq!(payload["inferenceConfig"]["maxTokens"], 4096);
        assert_eq!(payload["inferenceConfig"]["temperature"], 0.2);
        assert_eq!(payload["inferenceConfig"]["thinking"]["type"], "enabled");
        assert_eq!(payload["inferenceConfig"]["thinking"]["budgetTokens"], 1024);
    }

    #[test]
    fn payload_no_system_when_empty() {
        let payload = build_payload(&llm_config(), "", "hi", None);
        assert!(payload.get("system").is_none());
    }

    #[test]
    fn content_vision_image_block() {
        let content = build_user_content("inspect", Some("data:image/png;base64,AAAA"));
        assert_eq!(content[0]["text"], "inspect");
        // base64 "AAAA" decodes to 3 zero bytes, re-encoded identically.
        assert_eq!(content[1]["image"]["format"], "png");
        assert_eq!(content[1]["image"]["source"]["bytes"], "AAAA");
    }

    #[test]
    fn data_url_split() {
        assert_eq!(
            split_data_url("data:image/jpeg;base64,SGVsbG8="),
            Some(("jpeg", String::from("SGVsbG8=")))
        );
        assert_eq!(
            split_data_url("data:image/png;base64,SGVsbG8="),
            Some(("png", String::from("SGVsbG8=")))
        );
        assert!(split_data_url("not-a-data-url").is_none());
        assert!(split_data_url("data:image/jpeg;charset=utf8,abc").is_none());
        assert!(split_data_url("data:image/jpeg;base64,%%%invalid%%%").is_none());
    }

    #[test]
    fn response_parsed() {
        let json = serde_json::json!({
            "output": {"message": {"role": "assistant", "content": [
                {"text": "PASS"},
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 3, "totalTokens": 13},
            "stopReason": "end_turn"
        });
        let resp: LlmResponse = parse_response(&json).expect("parses");
        assert_eq!(resp.content, "PASS");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 13);
    }

    #[test]
    fn response_missing_content_errors() {
        let json = serde_json::json!({ "output": { "message": { "content": [] } } });
        assert!(parse_response(&json).is_err());
    }

    #[test]
    fn response_concatenates_text_blocks() {
        let json = serde_json::json!({
            "output": {"message": {"content": [
                {"text": "PA"},
                {"text": "SS"},
            ]}},
            "usage": {}
        });
        let resp: LlmResponse = parse_response(&json).expect("parses");
        assert_eq!(resp.content, "PASS");
    }
}
