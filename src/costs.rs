//! Cost calculation, usage tracking, and pricing logic.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::endpoints::ResolvedEndpoint;

/// Accumulated usage for a single endpoint.
#[derive(Debug, Default, Clone)]
pub struct EndpointUsage {
    /// Number of calls made.
    pub calls: u64,
    /// Total input tokens consumed.
    pub input_tokens: u64,
    /// Total output tokens consumed.
    pub output_tokens: u64,
    /// Accumulated cost in USD.
    pub cost: f64,
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
}

impl UsageSnapshot {
    /// Creates a snapshot from per-endpoint usage data.
    #[must_use]
    pub fn from_endpoints(endpoints: &HashMap<String, EndpointUsage>) -> Self {
        let total_cost = endpoints.values().map(|u| u.cost).sum();
        let total_calls = endpoints.values().map(|u| u.calls).sum();
        let total_tokens = endpoints.values().map(EndpointUsage::tokens).sum();
        Self {
            endpoints: endpoints.clone(),
            total_cost,
            total_calls,
            total_tokens,
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

    /// Records a completed call, adding usage and cost.
    ///
    /// # Panics
    ///
    /// Panics if the mutex is poisoned.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record_llm_call(
        &self,
        endpoint_name: &str,
        endpoint: &ResolvedEndpoint,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        let cost = calculate_llm_cost(endpoint, input_tokens, output_tokens);
        let mut inner = self.inner.lock().unwrap();
        let eu = inner
            .per_endpoint
            .entry(endpoint_name.to_owned())
            .or_default();
        eu.calls += 1;
        eu.input_tokens += input_tokens;
        eu.output_tokens += output_tokens;
        eu.cost += cost;
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
            ge.cost += ep_usage.cost;
        }
        inner.global.total_cost += snapshot.total_cost;
        inner.global.total_calls += snapshot.total_calls;
        inner.global.total_tokens += snapshot.total_tokens;
        inner.per_test.push((test_name.to_owned(), snapshot));
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculates the cost of an LLM call based on token pricing.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
#[must_use]
pub fn calculate_llm_cost(
    endpoint: &ResolvedEndpoint,
    input_tokens: u64,
    output_tokens: u64,
) -> f64 {
    let input_cost = (input_tokens as f64 / 1_000_000.0) * endpoint.input_price_per_1m;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * endpoint.output_price_per_1m;
    input_cost + output_cost
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
#[must_use]
pub fn extract_usage(value: &serde_json::Value) -> LlmUsage {
    let usage = &value["usage"];
    LlmUsage {
        prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
        total_tokens: usage["total_tokens"].as_u64().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use crate::costs::{calculate_llm_cost, UsageTracker};
    use crate::endpoints::ResolvedEndpoint;
    use crate::scenario::EndpointType;

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
            per_call_price: per_call,
            max_attempts: 3,
            fallbacks: vec![],
            provider: crate::scenario::Provider::Openai,
            deployment: None,
            api_version: None,
            auth: crate::scenario::AuthConfig::default(),
            header_commands: std::collections::HashMap::new(),
            aws: crate::scenario::AwsConfig::default(),
        }
    }

    #[test]
    fn test_calculate_llm_cost() {
        let ep = make_endpoint("test", 0.15, 0.60, 0.0);
        // 1M input tokens = $0.15, 500K output = $0.30
        let cost = calculate_llm_cost(&ep, 1_000_000, 500_000);
        assert!((cost - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_calculate_zero_cost() {
        let ep = make_endpoint("free", 0.0, 0.0, 0.0);
        let cost = calculate_llm_cost(&ep, 1_000_000, 1_000_000);
        assert!((cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_tracker_record_llm() {
        let tracker = UsageTracker::new();
        let ep = make_endpoint("gpt4", 2.50, 10.0, 0.0);
        tracker.record_llm_call("gpt4", &ep, 1000, 500);

        let snap = tracker.current_test_snapshot();
        assert_eq!(snap.total_calls, 1);
        assert_eq!(snap.total_tokens, 1500);
        assert!(
            snap.total_cost > 0.0,
            "expected cost > 0, got {}",
            snap.total_cost
        );

        let ep_usage = snap.endpoints.get("gpt4").unwrap();
        assert_eq!(ep_usage.calls, 1);
        assert_eq!(ep_usage.input_tokens, 1000);
        assert_eq!(ep_usage.output_tokens, 500);
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

        tracker.record_llm_call("fast", &fast, 100, 50);
        tracker.record_llm_call("slow", &slow, 200, 100);

        let snap = tracker.current_test_snapshot();
        assert_eq!(snap.total_calls, 2);
        assert_eq!(snap.endpoints.len(), 2);
    }

    #[test]
    fn test_usage_tracker_reset_and_commit() {
        let tracker = UsageTracker::new();
        let ep = make_endpoint("test", 0.15, 0.60, 0.0);

        tracker.record_llm_call("test", &ep, 100, 50);
        tracker.commit_test("test1");
        tracker.reset_per_test();

        tracker.record_llm_call("test", &ep, 200, 100);
        tracker.commit_test("test2");

        let global = tracker.global_snapshot();
        assert_eq!(global.total_calls, 2);
        assert_eq!(global.total_tokens, 450);

        let per_test = tracker.per_test_snapshots();
        assert_eq!(per_test.len(), 2);
        assert_eq!(per_test[0].0, "test1");
        assert_eq!(per_test[1].0, "test2");
    }
}
