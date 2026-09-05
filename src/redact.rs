//! Secret redaction for every report sink.
//!
//! All run output flows through [`crate::reporting::Reporter`]. The
//! [`Redactor`] held by the reporter replaces known secret values with
//! `[REDACTED]` before any sink (console, NDJSON, JUnit, GitHub, Perfetto)
//! sees the text, so a leaked API key or token can never make it into a log
//! or CI report.
//!
//! Secrets come from three places, all funneled into one redactor:
//!
//! - **Config-derived** — `collect_secrets_from_scenario_config` gathers
//!   `llm_api_key`, per-endpoint API keys, Entra client secrets, AWS static
//!   credentials, and the values of sensitive-named headers, and the runner
//!   registers them on every [`crate::runner::ScenarioRunner`] it builds.
//! - **Runtime-obtained** — tokens fetched by
//!   [`crate::auth`](auth) (token commands, header commands, Entra
//!   client-credentials and managed identity) are pushed into a
//!   process-global observed-secret registry via [`observe_secret`], which
//!   every [`Redactor::redact`] consults. They are registered before the
//!   `LlmCallFinished` event that describes the call is emitted, so the
//!   very event that echoes a token is redacted.
//! - **Explicit extras** — callers add literal values via
//!   `Reporter::add_redaction_secret` (the CLI's `--redact` /
//!   `HARNESS_REDACT`), for secrets not present in the config (URL query
//!   tokens, scenario-embedded test data).
//!
//! Redaction is exact-match, case-sensitive substring replacement. Values
//! shorter than [`MIN_SECRET_LEN`] are skipped when derived from config or
//! runtime sources so short, common strings (e.g. `"dev"`) do not destroy
//! log readability — explicit extras always apply.
//!
//! Raw `eprintln!` sites that bypass the reporter (the `#[browser_test]`
//! run-report strings, MCP/A2A server startup banners, the cost report)
//! are intentionally out of scope: they carry no secret-bearing text today.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{LazyLock, PoisonError, RwLock};

use crate::events::TestEvent;
use crate::scenario::ScenarioConfig;

/// Replacement text for every redacted value.
const REDACTED: &str = "[REDACTED]";

/// Shortest secret worth registering from config-derived or runtime
/// sources. Explicit extras (`add_secret` with `min_len 0`) always apply.
const MIN_SECRET_LEN: usize = 6;

/// Maximum number of runtime-observed secrets kept in the global registry,
/// bounding memory in long-running MCP/A2A server processes. Oldest
/// entries are dropped first.
const OBSERVED_CAP: usize = 512;

/// Header names whose values are treated as secrets (compared lowercased).
/// Innocuous headers (`x-org`, `content-type`) are NOT registered.
const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "api-key",
    "api_key",
    "x-api-key",
    "x-auth-token",
    "authentication",
    "token",
    "cookie",
    "x-amz-security-token",
];

/// Ordered set of runtime-observed secrets: `set` enables O(1) dedupe and
/// lookup, `order` (insertion order) enables dropping the oldest entry when
/// the cap is reached. `HashSet` alone has no ordering.
#[derive(Default)]
struct ObservedRegistry {
    set: HashSet<String>,
    order: VecDeque<String>,
}

impl ObservedRegistry {
    /// Inserts a value, dropping the oldest entry at [`OBSERVED_CAP`].
    fn insert(&mut self, value: String) {
        if self.set.contains(&value) {
            return;
        }
        if self.order.len() >= OBSERVED_CAP {
            if let Some(oldest) = self.order.pop_front() {
                self.set.remove(&oldest);
            }
        }
        self.set.insert(value.clone());
        self.order.push_back(value);
    }

    /// Iterates the registered secrets in insertion order.
    fn iter(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }
}

/// Process-global registry of runtime-observed secrets, shared by every
/// reporter in the process (including macro-path default reporters).
static OBSERVED: LazyLock<RwLock<ObservedRegistry>> =
    LazyLock::new(|| RwLock::new(ObservedRegistry::default()));

/// Registers a runtime-obtained secret so every subsequent
/// [`Redactor::redact`] call replaces it: token-command output,
/// header-command output, and Entra/IMDS access tokens.
///
/// Values shorter than [`MIN_SECRET_LEN`] are ignored; duplicates are
/// deduplicated; the registry is capped at [`OBSERVED_CAP`] entries.
pub fn observe_secret(value: &str) {
    if value.len() < MIN_SECRET_LEN {
        return;
    }
    OBSERVED
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(value.to_owned());
}

/// Replaces known secret values in a string with `[REDACTED]`.
#[derive(Default)]
pub struct Redactor {
    secrets: Vec<String>,
}

impl Redactor {
    /// Creates an empty redactor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a secret value when it is at least `min_len` characters
    /// long (and non-empty). Explicit extras pass `min_len 0` so they
    /// always apply.
    pub fn add_secret(&mut self, value: &str, min_len: usize) {
        if value.is_empty() || value.len() < min_len {
            return;
        }
        if self.secrets.iter().any(|s| s == value) {
            return;
        }
        self.secrets.push(value.to_owned());
    }

    /// Records config-derived secrets, applying [`MIN_SECRET_LEN`].
    pub fn add_secret_values(&mut self, values: impl IntoIterator<Item = String>) {
        for value in values {
            self.add_secret(&value, MIN_SECRET_LEN);
        }
    }

    /// Whether no local secrets are registered (runtime-observed secrets
    /// may still apply).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.secrets.is_empty()
    }

    /// Replaces every occurrence of every registered secret (local list
    /// first, then the process-global observed-secret registry) with
    /// `[REDACTED]`. Infallible: a broken registry can never fail an emit.
    #[must_use]
    pub fn redact(&self, input: &str) -> String {
        let mut out = input.to_owned();
        for secret in &self.secrets {
            out = out.replace(secret, REDACTED);
        }
        {
            let observed = OBSERVED.read().unwrap_or_else(PoisonError::into_inner);
            for secret in observed.iter() {
                out = out.replace(secret, REDACTED);
            }
        }
        out
    }

    /// Returns a copy of the event with every `String` field redacted;
    /// numeric/enum fields are copied verbatim.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn redact_event(&self, event: &TestEvent) -> TestEvent {
        match event {
            TestEvent::RunStarted { total_tests } => TestEvent::RunStarted {
                total_tests: *total_tests,
            },
            TestEvent::TestStarted { test } => TestEvent::TestStarted {
                test: self.redact(test),
            },
            TestEvent::StepStarted { test, index, label } => TestEvent::StepStarted {
                test: self.redact(test),
                index: *index,
                label: self.redact(label),
            },
            TestEvent::StepFinished {
                test,
                index,
                label,
                status,
                duration_ms,
                message,
                diagnostics,
                screenshot,
            } => TestEvent::StepFinished {
                test: self.redact(test),
                index: *index,
                label: self.redact(label),
                status: *status,
                duration_ms: *duration_ms,
                message: self.redact(message),
                diagnostics: diagnostics.as_deref().map(|s| self.redact(s)),
                screenshot: screenshot.as_deref().map(|s| self.redact(s)),
            },
            TestEvent::LlmCallStarted {
                test,
                index,
                endpoint,
                model,
                purpose,
            } => TestEvent::LlmCallStarted {
                test: self.redact(test),
                index: *index,
                endpoint: self.redact(endpoint),
                model: self.redact(model),
                purpose: self.redact(purpose),
            },
            TestEvent::LlmCallFinished {
                test,
                index,
                endpoint,
                model,
                purpose,
                ok,
                duration_ms,
                input_tokens,
                output_tokens,
                cost,
                error,
            } => TestEvent::LlmCallFinished {
                test: self.redact(test),
                index: *index,
                endpoint: self.redact(endpoint),
                model: self.redact(model),
                purpose: self.redact(purpose),
                ok: *ok,
                duration_ms: *duration_ms,
                input_tokens: *input_tokens,
                output_tokens: *output_tokens,
                cost: *cost,
                error: error.as_deref().map(|s| self.redact(s)),
            },
            TestEvent::TestFinished {
                test,
                passed,
                failed,
                skipped,
                duration_ms,
                cost,
                tokens,
                calls,
            } => TestEvent::TestFinished {
                test: self.redact(test),
                passed: *passed,
                failed: *failed,
                skipped: *skipped,
                duration_ms: *duration_ms,
                cost: *cost,
                tokens: *tokens,
                calls: *calls,
            },
            TestEvent::RunFinished {
                tests_passed,
                tests_failed,
                steps_passed,
                steps_failed,
                steps_skipped,
                total_cost,
                total_tokens,
                total_calls,
            } => TestEvent::RunFinished {
                tests_passed: *tests_passed,
                tests_failed: *tests_failed,
                steps_passed: *steps_passed,
                steps_failed: *steps_failed,
                steps_skipped: *steps_skipped,
                total_cost: *total_cost,
                total_tokens: *total_tokens,
                total_calls: *total_calls,
            },
            TestEvent::Warning { message } => TestEvent::Warning {
                message: self.redact(message),
            },
        }
    }
}

/// Gathers the startup-known secrets of a (merged) scenario config:
/// LLM API keys, endpoint credentials, and sensitive header values.
///
/// Returns raw strings; the [`MIN_SECRET_LEN`] guard is applied at insert
/// time by [`Redactor::add_secret_values`].
#[must_use]
pub fn collect_secrets_from_scenario_config(cfg: &ScenarioConfig) -> Vec<String> {
    let mut secrets: Vec<String> = Vec::new();
    if let Some(key) = &cfg.llm_api_key {
        secrets.push(key.clone());
    }
    collect_sensitive_header_values(&cfg.llm_headers, &mut secrets);
    for ep in cfg.endpoints.values() {
        if let Some(key) = &ep.api_key {
            secrets.push(key.clone());
        }
        if let Some(secret) = &ep.auth.client_secret {
            secrets.push(secret.clone());
        }
        if let Some(key) = &ep.aws.secret_access_key {
            secrets.push(key.clone());
        }
        if let Some(token) = &ep.aws.session_token {
            secrets.push(token.clone());
        }
        collect_sensitive_header_values(&ep.headers, &mut secrets);
    }
    secrets
}

/// Pushes the values of sensitive-named headers into `out` (matched
/// case-insensitively).
fn collect_sensitive_header_values(headers: &HashMap<String, String>, out: &mut Vec<String>) {
    for (name, value) in headers {
        if SENSITIVE_HEADER_NAMES
            .iter()
            .any(|n| name.to_ascii_lowercase() == *n)
        {
            out.push(value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        collect_secrets_from_scenario_config, observe_secret, Redactor, MIN_SECRET_LEN,
        OBSERVED_CAP,
    };
    use crate::events::{StepStatus, TestEvent};
    use crate::scenario::ScenarioConfig;

    #[test]
    fn test_redact_exact_replacement() {
        let mut r = Redactor::new();
        r.add_secret("sk-secret-key", 0);
        assert_eq!(r.redact("Bearer sk-secret-key"), "Bearer [REDACTED]");
    }

    #[test]
    fn test_redact_multiple_secrets() {
        let mut r = Redactor::new();
        r.add_secret("key-one", 0);
        r.add_secret("token-two", 0);
        let out = r.redact("key-one then token-two then key-one again");
        assert_eq!(out, "[REDACTED] then [REDACTED] then [REDACTED] again");
    }

    #[test]
    fn test_redact_secret_mid_string() {
        let mut r = Redactor::new();
        r.add_secret("abc123", 0);
        assert_eq!(
            r.redact("https://app.example.com/login?token=abc123&next=/x"),
            "https://app.example.com/login?token=[REDACTED]&next=/x"
        );
    }

    #[test]
    fn test_redact_empty_redactor_unchanged() {
        let r = Redactor::new();
        let input = "plain hello world 12345";
        assert_eq!(r.redact(input), input);
    }

    #[test]
    fn test_redact_empty_secret_ignored() {
        let mut r = Redactor::new();
        r.add_secret("", 0);
        assert_eq!(r.redact("a secret-leak here"), "a secret-leak here");
    }

    #[test]
    fn test_length_guard_skips_short_config_secrets() {
        let cfg = ScenarioConfig {
            llm_api_key: Some("short".into()), // 5 chars < MIN_SECRET_LEN
            ..ScenarioConfig::default()
        };
        let mut r = Redactor::new();
        r.add_secret_values(collect_secrets_from_scenario_config(&cfg));
        assert!(r.is_empty());
        assert_eq!(r.redact("token short here"), "token short here");
    }

    #[test]
    fn test_explicit_min_len_zero_always_applies() {
        let mut r = Redactor::new();
        r.add_secret("short", 0);
        assert!(!r.is_empty());
        assert_eq!(r.redact("token short here"), "token [REDACTED] here");
    }

    #[test]
    fn test_collect_secrets_from_config() {
        let mut headers = HashMap::new();
        headers.insert("X-API-Key".to_owned(), "hdr-key-value".to_owned());
        headers.insert("x-org".to_owned(), "acme".to_owned());
        headers.insert("content-type".to_owned(), "application/json".to_owned());
        let mut endpoints = HashMap::new();
        let ep = crate::scenario::EndpointConfig {
            api_key: Some("endpoint-key".to_owned()),
            headers: HashMap::from([("Authorization".to_owned(), "Bearer ep-bearer".to_owned())]),
            auth: crate::scenario::AuthConfig {
                client_secret: Some("entra-secret".to_owned()),
                ..crate::scenario::AuthConfig::default()
            },
            aws: crate::scenario::AwsConfig {
                secret_access_key: Some("aws-secret".to_owned()),
                session_token: Some("aws-session".to_owned()),
                ..crate::scenario::AwsConfig::default()
            },
            fallbacks: vec!["fallback".to_owned()],
            ..crate::scenario::EndpointConfig::default()
        };
        endpoints.insert("primary".to_owned(), ep);
        endpoints.insert(
            "fallback".to_owned(),
            crate::scenario::EndpointConfig {
                api_key: Some("fallback-key".to_owned()),
                ..crate::scenario::EndpointConfig::default()
            },
        );
        let cfg = ScenarioConfig {
            llm_api_key: Some("global-key".to_owned()),
            llm_headers: headers,
            endpoints,
            ..ScenarioConfig::default()
        };
        let secrets = collect_secrets_from_scenario_config(&cfg);
        for expected in [
            "global-key",
            "hdr-key-value",
            "endpoint-key",
            "Bearer ep-bearer",
            "entra-secret",
            "aws-secret",
            "aws-session",
            "fallback-key",
        ] {
            assert!(secrets.iter().any(|s| s == expected), "missing {expected}");
        }
        assert!(
            secrets
                .iter()
                .all(|s| s != "acme" && s != "application/json"),
            "innocuous header values must not be registered"
        );
    }

    #[test]
    fn test_registry_observe_and_redact() {
        observe_secret("registry-token-123456");
        let r = Redactor::new();
        assert_eq!(
            r.redact("error body: registry-token-123456"),
            "error body: [REDACTED]"
        );
    }

    #[test]
    fn test_registry_deduplicates() {
        observe_secret("registry-dedup-abcdef");
        observe_secret("registry-dedup-abcdef");
        let r = Redactor::new();
        assert_eq!(r.redact("registry-dedup-abcdef"), "[REDACTED]");
    }

    #[test]
    fn test_registry_min_len_guard() {
        observe_secret("tiny");
        let r = Redactor::new();
        assert_eq!(r.redact("token tiny here"), "token tiny here");
    }

    #[test]
    fn test_registry_cap_drops_oldest() {
        let r = Redactor::new();
        for i in 0..OBSERVED_CAP + 10 {
            observe_secret(&format!("registry-cap-secret-{i:03}"));
        }
        // The first 10 entries were evicted...
        assert_eq!(
            r.redact("registry-cap-secret-000"),
            "registry-cap-secret-000"
        );
        // ...while the newest entries still redact.
        assert_eq!(r.redact("registry-cap-secret-521"), "[REDACTED]");
    }

    #[test]
    fn test_redact_event_step_finished() {
        let mut r = Redactor::new();
        r.add_secret("tok-query-secret", 0);
        let event = TestEvent::StepFinished {
            test: "login".into(),
            index: 0,
            label: "[navigate] /dashboard?token=tok-query-secret".into(),
            status: StepStatus::Failed,
            duration_ms: 100,
            message: "element not found at /x?token=tok-query-secret".into(),
            diagnostics: Some("url: /x?token=tok-query-secret".into()),
            screenshot: Some("artifacts/login.png".into()),
        };
        let redacted = r.redact_event(&event);
        let TestEvent::StepFinished {
            label,
            message,
            diagnostics,
            screenshot,
            status,
            ..
        } = redacted
        else {
            panic!("expected step_finished");
        };
        assert_eq!(label, "[navigate] /dashboard?token=[REDACTED]");
        assert_eq!(message, "element not found at /x?token=[REDACTED]");
        assert_eq!(diagnostics.as_deref(), Some("url: /x?token=[REDACTED]"));
        assert_eq!(screenshot.as_deref(), Some("artifacts/login.png"));
        assert_eq!(status, StepStatus::Failed);
    }

    #[test]
    fn test_redact_event_llm_call_error_with_registry_token() {
        observe_secret("runtime-token-987654");
        let mut r = Redactor::new();
        r.add_secret("sk-static-key", 0);
        let event = TestEvent::LlmCallFinished {
            test: "login".into(),
            index: 0,
            endpoint: "default".into(),
            model: "deepseek".into(),
            purpose: "targeting".into(),
            ok: false,
            duration_ms: 900,
            input_tokens: 0,
            output_tokens: 0,
            cost: 0.0,
            error: Some("HTTP 401: Bearer sk-static-key invalid (got runtime-token-987654)".into()),
        };
        let redacted = r.redact_event(&event);
        let TestEvent::LlmCallFinished { error, .. } = redacted else {
            panic!("expected llm_call_finished");
        };
        assert_eq!(
            error.as_deref(),
            Some("HTTP 401: Bearer [REDACTED] invalid (got [REDACTED])")
        );
    }

    #[test]
    fn test_redact_event_preserves_non_string_fields() {
        let r = Redactor::new();
        let event = TestEvent::TestFinished {
            test: "t".into(),
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 12,
            cost: 0.0042,
            tokens: 42,
            calls: 1,
        };
        let redacted = r.redact_event(&event);
        let TestEvent::TestFinished {
            test,
            passed,
            failed,
            duration_ms,
            cost,
            tokens,
            calls,
            ..
        } = redacted
        else {
            panic!("expected test_finished");
        };
        assert_eq!(test, "t");
        assert_eq!((passed, failed), (1, 0));
        assert_eq!(duration_ms, 12);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(cost, 0.0042);
        }
        assert_eq!((tokens, calls), (42, 1));
    }

    #[test]
    fn test_min_secret_len_constant_matches_guard() {
        // Sanity: the default guard applies to `add_secret_values`.
        let mut r = Redactor::new();
        r.add_secret_values(["12345".to_owned(), "123456".to_owned()]);
        assert_eq!(r.secrets.len(), 1);
        assert_eq!(r.secrets[0], "123456");
        assert!(!r.is_empty());
        assert_eq!(MIN_SECRET_LEN, 6);
    }
}
