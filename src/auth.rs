//! Authentication token resolution for LLM endpoints.
//!
//! Supports Entra ID (OAuth 2.0 client-credentials grant and managed
//! identity via the IMDS endpoint) and arbitrary "token command" programs
//! (e.g. `az account get-access-token …`). Tokens are cached with a TTL
//! (server-issued expiry for Entra, configurable for commands) so a test
//! run does not re-authenticate on every LLM call.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::redact::observe_secret;
use crate::scenario::{AuthConfig, AuthMode};

/// Default scope for `Azure` Cognitive Services Entra tokens.
const DEFAULT_AZURE_COGNITIVE_SCOPE: &str = "https://cognitiveservices.azure.com/.default";
/// Default resource for the managed-identity IMDS call.
const DEFAULT_AZURE_IMDS_RESOURCE: &str = "https://cognitiveservices.azure.com/";
/// Default Entra token endpoint authority.
const DEFAULT_TOKEN_URL: &str = "https://login.microsoftonline.com";
/// Default IMDS endpoint.
const DEFAULT_IMDS_URL: &str = "http://169.254.169.254";
/// How long a command-produced token is reused before re-running the
/// command (seconds).
const DEFAULT_COMMAND_CACHE_TTL_SECS: u64 = 300;

/// A cached token together with its expiry.
struct CacheEntry {
    /// Wall-clock expiry of the token.
    expires_at: Instant,
    /// The bearer token value.
    token: String,
}

/// Process-wide token cache keyed by an auth-config fingerprint. Keys are
/// hashes so secrets (client secrets, commands) never appear in the key.
static TOKEN_CACHE: LazyLock<Mutex<HashMap<u64, CacheEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolves the bearer token for an endpoint, or `None` when the mode does
/// not produce one (e.g. `api-key` mode without a key, which callers treat
/// as "no auth header").
///
/// # Errors
///
/// Returns a description when the configured auth mode is incomplete
/// (missing tenant/client/secret/command), the token command fails, or the
/// Entra token endpoint cannot be reached or answers with an error.
pub async fn resolve_bearer_token(
    auth: &AuthConfig,
    static_key: Option<&str>,
) -> Result<Option<String>, String> {
    match auth.mode {
        AuthMode::ApiKey => Ok(static_key.map(str::to_owned)),
        AuthMode::TokenCommand => {
            let cmd = auth.token_command.as_deref().ok_or_else(|| {
                "auth.mode = \"token-command\" requires auth.token_command".to_owned()
            })?;
            let ttl = effective_cmd_ttl(auth);
            let key = cache_key(("command", cmd, ""));
            let output = run_token_command(cmd).await?;
            fetch_cached(key, ttl, async move { Ok((0, output)) })
                .await
                .map(Some)
        }
        AuthMode::EntraClientCredentials => {
            let tenant = auth.tenant_id.as_deref().ok_or_else(|| {
                "auth.mode = \"entra-client-credentials\" requires auth.tenant_id".to_owned()
            })?;
            let client_id = auth.client_id.as_deref().ok_or_else(|| {
                "auth.mode = \"entra-client-credentials\" requires auth.client_id".to_owned()
            })?;
            let secret = auth.client_secret.as_deref().ok_or_else(|| {
                "auth.mode = \"entra-client-credentials\" requires auth.client_secret".to_owned()
            })?;
            let base = auth
                .token_url
                .clone()
                .unwrap_or_else(|| DEFAULT_TOKEN_URL.to_owned());
            let scope = auth
                .scope
                .clone()
                .unwrap_or_else(|| DEFAULT_AZURE_COGNITIVE_SCOPE.to_owned());
            let url = format!("{}/{tenant}/oauth2/v2.0/token", base.trim_end_matches('/'));
            let key = cache_key(("entra-cc", tenant, client_id, &scope, &base));
            fetch_cached(key, DEFAULT_COMMAND_CACHE_TTL_SECS, async move {
                let client = crate::http_client(Duration::from_secs(30));
                let resp = client
                    .post(&url)
                    .form(&[
                        ("grant_type", "client_credentials"),
                        ("client_id", client_id),
                        ("client_secret", secret),
                        ("scope", &scope),
                    ])
                    .send()
                    .await
                    .map_err(|e| format!("Entra token request failed: {e}"))?;
                parse_token_response(resp, "Entra").await
            })
            .await
            .map(Some)
        }
        AuthMode::EntraManagedIdentity => {
            let base = auth
                .token_url
                .clone()
                .unwrap_or_else(|| DEFAULT_IMDS_URL.to_owned());
            let resource = auth
                .scope
                .clone()
                .unwrap_or_else(|| DEFAULT_AZURE_IMDS_RESOURCE.to_owned());
            let key = cache_key(("imds", &resource, &base));
            fetch_cached(key, DEFAULT_COMMAND_CACHE_TTL_SECS, async move {
                let client = crate::http_client(Duration::from_secs(10));
                let resp = client
                    .get(format!(
                        "{}/metadata/identity/oauth2/token",
                        base.trim_end_matches('/')
                    ))
                    .header("Metadata", "true")
                    .query(&[("api-version", "2018-02-01"), ("resource", &resource)])
                    .send()
                    .await
                    .map_err(|e| format!("managed-identity token request failed: {e}"))?;
                parse_token_response(resp, "managed identity").await
            })
            .await
            .map(Some)
        }
    }
}

/// Runs a token-command and returns its (first-line, trimmed) stdout.
///
/// # Errors
///
/// Returns a description when the command cannot be spawned, exits
/// non-zero, or produces empty output.
pub async fn run_token_command(cmd: &str) -> Result<String, String> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .await
        .map_err(|e| format!("token command failed to start: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "token command failed ({output_status}): {stderr}",
            output_status = output.status
        ));
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_owned();
    if line.is_empty() {
        return Err("token command produced empty stdout".to_owned());
    }
    observe_secret(&line);
    Ok(line)
}

/// Runs a header-command (for `header_commands`) and returns the first,
/// trimmed line of its stdout as the header value.
///
/// # Errors
///
/// Returns a description when the command fails or outputs nothing.
pub async fn run_header_command(cmd: &str) -> Result<String, String> {
    run_token_command(cmd).await
}

/// Shared cache probe: returns the cached token when still valid,
/// otherwise awaits `fetch` and stores its result with its reported TTL.
async fn fetch_cached(
    key: u64,
    fallback_ttl_secs: u64,
    fetch: impl std::future::Future<Output = Result<(u64, String), String>>,
) -> Result<String, String> {
    let now = Instant::now();
    if let Some(entry) = TOKEN_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
    {
        if entry.expires_at > now {
            return Ok(entry.token.clone());
        }
    }
    let (ttl, token) = fetch.await?;
    let ttl = if ttl == 0 { fallback_ttl_secs } else { ttl };
    TOKEN_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            key,
            CacheEntry {
                expires_at: Instant::now() + Duration::from_secs(ttl),
                token: token.clone(),
            },
        );
    Ok(token)
}

/// Parses an Entra / IMDS token HTTP response and extracts the access
/// token plus its server-issued TTL (seconds). The TTL is 0 when the
/// server did not report one (caller falls back to its configured TTL).
async fn parse_token_response(
    resp: reqwest::Response,
    kind: &str,
) -> Result<(u64, String), String> {
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    if status != 200 {
        return Err(format!(
            "{kind} token endpoint returned HTTP {status}: {}",
            crate::truncate(&text, 300)
        ));
    }
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        format!(
            "{kind} token endpoint returned non-JSON ({e}): {}",
            crate::truncate(&text, 300)
        )
    })?;
    let token = json["access_token"]
        .as_str()
        .ok_or_else(|| {
            format!(
                "{kind} token response missing access_token: {}",
                crate::truncate(&text, 300)
            )
        })?
        .to_owned();
    observe_secret(&token);
    let ttl = json["expires_in"]
        .as_u64()
        .or_else(|| {
            json["expires_on"]
                .as_str()
                .and_then(|s| s.parse::<u64>().ok())
        })
        .map_or(0, |s| s.saturating_sub(60).max(60));
    Ok((ttl, token))
}

/// Effective cache TTL for command-produced tokens.
fn effective_cmd_ttl(auth: &AuthConfig) -> u64 {
    auth.cache_ttl_secs
        .unwrap_or(DEFAULT_COMMAND_CACHE_TTL_SECS)
        .max(1)
}

/// Stable fingerprint for the token cache (secrets are hashed, never kept
/// in the key).
fn cache_key(parts: impl std::hash::Hash) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    parts.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::run_token_command;

    #[tokio::test]
    #[cfg(unix)]
    async fn token_command_returns_first_line() {
        let token = run_token_command("printf 'tok-123\\nignored\\n'")
            .await
            .expect("command runs");
        assert_eq!(token, "tok-123");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn token_command_failure_reported() {
        let err = run_token_command("exit 3")
            .await
            .expect_err("failing command must error");
        assert!(err.contains("token command failed"), "got: {err}");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn token_command_empty_output_reported() {
        let err = run_token_command("true")
            .await
            .expect_err("empty output must error");
        assert!(err.contains("empty stdout"), "got: {err}");
    }
}
