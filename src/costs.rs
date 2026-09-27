//! Cost calculation, usage tracking, and pricing logic.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use crate::endpoints::ResolvedEndpoint;
use crate::scenario::Provider;

/// Accumulated usage for a single endpoint.
#[derive(Debug, Default, Clone)]
pub struct EndpointUsage {
    /// Number of calls made.
    pub calls: u64,
    /// Total input tokens consumed.
    pub input_tokens: u64,
    /// Total output tokens consumed.
    pub output_tokens: u64,
    /// Input tokens served from the provider's prompt cache.
    pub cached_input_tokens: u64,
    /// Input tokens written to the provider's prompt cache (cache creation).
    pub cache_creation_input_tokens: u64,
    /// Accumulated cost in USD.
    pub cost: f64,
    /// Model names observed on this endpoint, sorted and deduplicated.
    pub models: BTreeSet<String>,
}

impl EndpointUsage {
    const fn tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Aggregated usage across all endpoints for a test or scenario run.
#[derive(Debug, Default, Clone)]
pub struct UsageSnapshot {
    /// Per-endpoint usage.
    pub endpoints: HashMap<String, EndpointUsage>,
    /// Total cost across all endpoints.
    pub total_cost: f64,
    /// Total calls across all endpoints.
    pub total_calls: u64,
    /// Total tokens across all endpoints.
    pub total_tokens: u64,
    /// Total input (prompt) tokens across all endpoints.
    pub total_input_tokens: u64,
    /// Total output (completion) tokens across all endpoints.
    pub total_output_tokens: u64,
    /// Total input tokens served from provider prompt caches.
    pub total_cached_input_tokens: u64,
    /// Total input tokens written to provider prompt caches.
    pub total_cache_creation_input_tokens: u64,
    /// Model names observed across all endpoints, sorted and deduplicated.
    pub models: Vec<String>,
}

impl UsageSnapshot {
    /// Creates a snapshot from per-endpoint usage data.
    #[must_use]
    pub fn from_endpoints(endpoints: &HashMap<String, EndpointUsage>) -> Self {
        let total_cost = endpoints.values().map(|u| u.cost).sum();
        let total_calls = endpoints.values().map(|u| u.calls).sum();
        let total_tokens = endpoints.values().map(EndpointUsage::tokens).sum();
        let total_input_tokens = endpoints.values().map(|u| u.input_tokens).sum();
        let total_output_tokens = endpoints.values().map(|u| u.output_tokens).sum();
        let total_cached_input_tokens = endpoints.values().map(|u| u.cached_input_tokens).sum();
        let total_cache_creation_input_tokens = endpoints
            .values()
            .map(|u| u.cache_creation_input_tokens)
            .sum();
        let models: Vec<String> = endpoints
            .values()
            .flat_map(|u| u.models.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Self {
            endpoints: endpoints.clone(),
            total_cost,
            total_calls,
            total_tokens,
            total_input_tokens,
            total_output_tokens,
            total_cached_input_tokens,
            total_cache_creation_input_tokens,
            models,
        }
    }
}

/// Thread-safe usage tracker for the test runner.
pub struct UsageTracker {
    inner: Mutex<UsageInner>,
}

struct UsageInner {
    /// Per-endpoint usage for the current test.
    per_endpoint: HashMap<String, EndpointUsage>,
    /// Aggregated usage across all completed tests.
    global: UsageSnapshot,
    /// Per-test snapshots keyed by test name.
    per_test: Vec<(String, UsageSnapshot)>,
}

impl UsageTracker {
    /// Creates a new empty usage tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(UsageInner {
                per_endpoint: HashMap::new(),
                global: UsageSnapshot::default(),
                per_test: Vec::new(),
            }),
        }
    }

    /// Records a completed LLM call, adding usage, cost, and the answering
    /// model to the endpoint's accumulator.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record_llm_call(
        &self,
        endpoint_name: &str,
        endpoint: &ResolvedEndpoint,
        model: &str,
        usage: &LlmUsage,
    ) {
        let cost = calculate_llm_cost(endpoint, usage);
        let mut inner = self.inner.lock().unwrap();
        let eu = inner
            .per_endpoint
            .entry(endpoint_name.to_owned())
            .or_default();
        eu.calls += 1;
        eu.input_tokens += usage.prompt_tokens;
        eu.output_tokens += usage.completion_tokens;
        eu.cached_input_tokens += usage.cached_input_tokens;
        eu.cache_creation_input_tokens += usage.cache_creation_input_tokens;
        eu.cost += cost;
        if !model.is_empty() {
            eu.models.insert(model.to_owned());
        }
    }

    /// Records a flat-cost call (MCP tool, agent task).
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record_flat_call(&self, endpoint_name: &str, endpoint: &ResolvedEndpoint) {
        let mut inner = self.inner.lock().unwrap();
        let eu = inner
            .per_endpoint
            .entry(endpoint_name.to_owned())
            .or_default();
        eu.calls += 1;
        eu.cost += endpoint.per_call_price;
    }

    /// Reads current usage without locking for the full snapshot.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[must_use]
    pub fn current_test_snapshot(&self) -> UsageSnapshot {
        let inner = self.inner.lock().unwrap();
        UsageSnapshot::from_endpoints(&inner.per_endpoint)
    }

    /// Reads the global aggregated snapshot.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[must_use]
    pub fn global_snapshot(&self) -> UsageSnapshot {
        let inner = self.inner.lock().unwrap();
        inner.global.clone()
    }

    /// Reads per-test snapshots.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[must_use]
    pub fn per_test_snapshots(&self) -> Vec<(String, UsageSnapshot)> {
        let inner = self.inner.lock().unwrap();
        inner.per_test.clone()
    }

    /// Resets the per-test accumulator. Call at the start of each test.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    pub fn reset_per_test(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.per_endpoint.clear();
    }

    /// Commits the current test's usage to the global accumulator and stores
    /// it as a per-test snapshot.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    pub fn commit_test(&self, test_name: &str) {
        let mut inner = self.inner.lock().unwrap();
        let snapshot = UsageSnapshot::from_endpoints(&inner.per_endpoint);
        // Merge into global
        let ep_snapshot = inner.per_endpoint.clone();
        for (ep_name, ep_usage) in &ep_snapshot {
            let ge = inner.global.endpoints.entry(ep_name.clone()).or_default();
            ge.calls += ep_usage.calls;
            ge.input_tokens += ep_usage.input_tokens;
            ge.output_tokens += ep_usage.output_tokens;
            ge.cached_input_tokens += ep_usage.cached_input_tokens;
            ge.cache_creation_input_tokens += ep_usage.cache_creation_input_tokens;
            ge.cost += ep_usage.cost;
            ge.models.extend(ep_usage.models.iter().cloned());
        }
        inner.global.total_cost += snapshot.total_cost;
        inner.global.total_calls += snapshot.total_calls;
        inner.global.total_tokens += snapshot.total_tokens;
        inner.global.total_input_tokens += snapshot.total_input_tokens;
        inner.global.total_output_tokens += snapshot.total_output_tokens;
        inner.global.total_cached_input_tokens += snapshot.total_cached_input_tokens;
        inner.global.total_cache_creation_input_tokens +=
            snapshot.total_cache_creation_input_tokens;
        inner.global.models = inner
            .global
            .endpoints
            .values()
            .flat_map(|u| u.models.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        inner.per_test.push((test_name.to_owned(), snapshot));
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// How a provider reports cache tokens relative to its input token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAccounting {
    /// Cache tokens are a **subset** of the reported input tokens
    /// (`OpenAI`, `Azure`, `OpenRouter`, Groq, xAI, `DeepSeek`, …).
    Subset,
    /// Cache tokens are reported **in addition** to the input tokens
    /// (`Anthropic` Messages and AWS Bedrock Converse).
    Additive,
}

/// Returns the cache accounting convention for a provider.
#[must_use]
pub const fn cache_accounting(provider: Provider) -> CacheAccounting {
    match provider {
        Provider::Openai | Provider::Azure => CacheAccounting::Subset,
        Provider::Bedrock => CacheAccounting::Additive,
    }
}

/// Calculates the cost of an LLM call based on token pricing.
///
/// When [`ResolvedEndpoint::cache_pricing`] is enabled (the default), cache
/// reads and writes are billed at their cache rates; otherwise every prompt
/// token is billed at the flat input price. Cache tokens are treated as a
/// subset or as additive to the input count depending on the provider (see
/// [`CacheAccounting`]).
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
#[must_use]
pub fn calculate_llm_cost(endpoint: &ResolvedEndpoint, usage: &LlmUsage) -> f64 {
    let million = 1_000_000.0;
    let (ordinary_input, cached, cache_write) =
        if cache_accounting(endpoint.provider) == CacheAccounting::Subset {
            (
                usage
                    .prompt_tokens
                    .saturating_sub(usage.cached_input_tokens)
                    .saturating_sub(usage.cache_creation_input_tokens),
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
            )
        } else {
            (
                usage.prompt_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
            )
        };
    let input_cost = if endpoint.cache_pricing {
        (ordinary_input as f64 / million) * endpoint.input_price_per_1m
            + (cached as f64 / million) * endpoint.cached_input_price_per_1m
            + (cache_write as f64 / million) * endpoint.cache_write_price_per_1m
    } else {
        ((ordinary_input + cached + cache_write) as f64 / million) * endpoint.input_price_per_1m
    };
    let output_cost = (usage.completion_tokens as f64 / million) * endpoint.output_price_per_1m;
    input_cost + output_cost + endpoint.per_call_price
}

/// Usage info extracted from an LLM API response.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlmUsage {
    /// Number of prompt / input tokens.
    pub prompt_tokens: u64,
    /// Number of completion / output tokens.
    pub completion_tokens: u64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Input tokens served from the provider's prompt cache (cache hit).
    pub cached_input_tokens: u64,
    /// Input tokens written to the provider's prompt cache (cache creation).
    pub cache_creation_input_tokens: u64,
}

/// Result of an LLM chat call including usage data.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// The message content from the LLM.
    pub content: String,
    /// Token usage from the API response.
    pub usage: LlmUsage,
}

/// Extracts token usage from an OpenAI-compatible API response JSON.
///
/// Cache reads are read from `usage.prompt_tokens_details.cached_tokens`
/// (`OpenAI`, `Azure`, `OpenRouter`, Groq, xAI, …), falling back to the
/// `Anthropic`-compatible `usage.cache_read_input_tokens` and `DeepSeek`
/// `usage.prompt_cache_hit_tokens` spellings. Cache writes are read from
/// `usage.prompt_tokens_details.cache_write_tokens` (`OpenRouter`, and
/// `OpenAI`/`Azure` on GPT-5.6+), falling back to the `Anthropic`-compatible
/// `usage.cache_creation_input_tokens`. Providers that do not report prompt
/// caching yield `0`.
///
/// Note the provider semantics differ: for `OpenAI`-compatible responses
/// `cached_tokens`/`cache_write_tokens` are subsets of `prompt_tokens`,
/// whereas for `Anthropic`/Bedrock-style responses the cache counters are
/// reported in addition to `inputTokens`.
#[must_use]
pub fn extract_usage(value: &serde_json::Value) -> LlmUsage {
    let usage = &value["usage"];
    if usage.is_null() {
        return extract_gemini_usage(value);
    }
    LlmUsage {
        prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
        total_tokens: usage["total_tokens"].as_u64().unwrap_or(0),
        cached_input_tokens: usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .or_else(|| usage["cache_read_input_tokens"].as_u64())
            .or_else(|| usage["prompt_cache_hit_tokens"].as_u64())
            .unwrap_or(0),
        cache_creation_input_tokens: usage["prompt_tokens_details"]["cache_write_tokens"]
            .as_u64()
            .or_else(|| usage["cache_creation_input_tokens"].as_u64())
            .unwrap_or(0),
    }
}

/// Fallback extraction for a native Google Gemini `generateContent` response,
/// which reports usage under `usageMetadata` rather than `usage`:
/// `promptTokenCount` / `candidatesTokenCount` / `totalTokenCount` and the
/// prompt-cache read counter `cachedContentTokenCount`. Returns an all-zero
/// usage when neither shape is present.
#[must_use]
fn extract_gemini_usage(value: &serde_json::Value) -> LlmUsage {
    let metadata = &value["usageMetadata"];
    LlmUsage {
        prompt_tokens: metadata["promptTokenCount"].as_u64().unwrap_or(0),
        completion_tokens: metadata["candidatesTokenCount"].as_u64().unwrap_or(0),
        total_tokens: metadata["totalTokenCount"].as_u64().unwrap_or(0),
        cached_input_tokens: metadata["cachedContentTokenCount"].as_u64().unwrap_or(0),
        cache_creation_input_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use crate::costs::{calculate_llm_cost, LlmUsage, UsageTracker};
    use crate::endpoints::ResolvedEndpoint;
    use crate::scenario::EndpointType;
    use crate::scenario::Provider;

    fn make_endpoint(
        name: &str,
        input_price: f64,
        output_price: f64,
        per_call: f64,
    ) -> ResolvedEndpoint {
        ResolvedEndpoint {
            name: name.to_owned(),
            endpoint_type: EndpointType::Llm,
            url: String::new(),
            model: None,
            api_key: None,
            headers: std::collections::HashMap::new(),
            command: None,
            args: vec![],
            vision: false,
            input_price_per_1m: input_price,
            output_price_per_1m: output_price,
            cached_input_price_per_1m: input_price * 0.1,
            cache_write_price_per_1m: input_price * 1.25,
            cache_pricing: true,
            cache_markers: true,
            per_call_price: per_call,
            max_attempts: 3,
            fallbacks: vec![],
            provider: Provider::Openai,
            deployment: None,
            api_version: None,
            auth: crate::scenario::AuthConfig::default(),
            header_commands: std::collections::HashMap::new(),
            aws: crate::scenario::AwsConfig::default(),
        }
    }

    fn usage(prompt: u64, completion: u64, cached: u64) -> LlmUsage {
        LlmUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cached_input_tokens: cached,
            cache_creation_input_tokens: 0,
        }
    }

    #[test]
    fn test_calculate_llm_cost() {
        let ep = make_endpoint("test", 0.15, 0.60, 0.0);
        // 1M input tokens = $0.15, 500K output = $0.30
        let cost = calculate_llm_cost(&ep, &usage(1_000_000, 500_000, 0));
        assert!((cost - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_calculate_zero_cost() {
        let ep = make_endpoint("free", 0.0, 0.0, 0.0);
        let cost = calculate_llm_cost(&ep, &usage(1_000_000, 1_000_000, 0));
        assert!((cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_llm_cost_cache_subset() {
        // OpenAI-style: cached tokens are a subset of prompt_tokens.
        let ep = make_endpoint("gpt", 1.0, 0.0, 0.0);
        let cost = calculate_llm_cost(&ep, &usage(1_000_000, 0, 1_000_000));
        // All input is cached at 0.1x => $0.10.
        assert!((cost - 0.10).abs() < 0.001, "got {cost}");
    }

    #[test]
    fn test_calculate_llm_cost_cache_additive_bedrock() {
        // Bedrock/Anthropic-style: cache tokens are additional to input.
        let mut ep = make_endpoint("bedrock", 1.0, 0.0, 0.0);
        ep.provider = Provider::Bedrock;
        let cost = calculate_llm_cost(&ep, &usage(1_000_000, 0, 1_000_000));
        // 1M ordinary input ($1.00) + 1M cached read ($0.10).
        assert!((cost - 1.10).abs() < 0.001, "got {cost}");
    }

    #[test]
    fn test_calculate_llm_cost_cache_pricing_disabled() {
        // With cache pricing off, every prompt token is billed at input price
        // even for additive (Bedrock) accounting.
        let mut ep = make_endpoint("bedrock", 1.0, 0.0, 0.0);
        ep.provider = Provider::Bedrock;
        ep.cache_pricing = false;
        let cost = calculate_llm_cost(&ep, &usage(1_000_000, 0, 1_000_000));
        assert!((cost - 2.0).abs() < 0.001, "got {cost}");
    }

    #[test]
    fn test_usage_tracker_record_llm() {
        let tracker = UsageTracker::new();
        let ep = make_endpoint("gpt4", 2.50, 10.0, 0.0);
        tracker.record_llm_call("gpt4", &ep, "gpt-4o", &usage(1000, 500, 200));

        let snap = tracker.current_test_snapshot();
        assert_eq!(snap.total_calls, 1);
        assert_eq!(snap.total_tokens, 1500);
        assert_eq!(snap.total_input_tokens, 1000);
        assert_eq!(snap.total_output_tokens, 500);
        assert_eq!(snap.total_cached_input_tokens, 200);
        assert_eq!(snap.models, vec!["gpt-4o".to_owned()]);
        assert!(
            snap.total_cost > 0.0,
            "expected cost > 0, got {}",
            snap.total_cost
        );

        let ep_usage = snap.endpoints.get("gpt4").unwrap();
        assert_eq!(ep_usage.calls, 1);
        assert_eq!(ep_usage.input_tokens, 1000);
        assert_eq!(ep_usage.output_tokens, 500);
        assert_eq!(ep_usage.cached_input_tokens, 200);
        assert!(ep_usage.models.contains("gpt-4o"));
    }

    #[test]
    fn test_usage_tracker_record_flat() {
        let tracker = UsageTracker::new();
        let ep = make_endpoint("agent", 0.0, 0.0, 0.01);
        tracker.record_flat_call("agent", &ep);
        tracker.record_flat_call("agent", &ep);

        let snap = tracker.current_test_snapshot();
        assert_eq!(snap.total_calls, 2);
        assert!((snap.total_cost - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_tracker_multiple_endpoints() {
        let tracker = UsageTracker::new();
        let fast = make_endpoint("fast", 0.15, 0.60, 0.0);
        let slow = make_endpoint("slow", 2.50, 10.0, 0.0);

        tracker.record_llm_call("fast", &fast, "fast-model", &usage(100, 50, 0));
        tracker.record_llm_call("slow", &slow, "slow-model", &usage(200, 100, 0));

        let snap = tracker.current_test_snapshot();
        assert_eq!(snap.total_calls, 2);
        assert_eq!(snap.endpoints.len(), 2);
        assert_eq!(
            snap.models,
            vec!["fast-model".to_owned(), "slow-model".to_owned()]
        );
    }

    #[test]
    fn test_usage_tracker_reset_and_commit() {
        let tracker = UsageTracker::new();
        let ep = make_endpoint("test", 0.15, 0.60, 0.0);

        tracker.record_llm_call("test", &ep, "m1", &usage(100, 50, 0));
        tracker.commit_test("test1");
        tracker.reset_per_test();

        tracker.record_llm_call("test", &ep, "m2", &usage(200, 100, 0));
        tracker.commit_test("test2");

        let global = tracker.global_snapshot();
        assert_eq!(global.total_calls, 2);
        assert_eq!(global.total_tokens, 450);
        assert_eq!(global.models, vec!["m1".to_owned(), "m2".to_owned()]);

        let per_test = tracker.per_test_snapshots();
        assert_eq!(per_test.len(), 2);
        assert_eq!(per_test[0].0, "test1");
        assert_eq!(per_test[1].0, "test2");
    }
}
