//! Endpoint registry — resolves named endpoints and task-type routing.

use std::collections::HashMap;

use crate::scenario::EndpointConfig;
use crate::scenario::EndpointType;

/// Classification of a task for endpoint routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// LLM-based element targeting (resolving CSS selectors from natural
    /// language).
    Targeting,
    /// LLM-based assertion evaluation.
    Assertion,
}

impl TaskType {
    /// Returns the routing key string for this task type.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Targeting => "targeting",
            Self::Assertion => "assertion",
        }
    }
}

/// Resolved endpoint ready for use in calls.
#[derive(Debug, Clone)]
pub struct ResolvedEndpoint {
    /// Endpoint name.
    pub name: String,
    /// Endpoint type.
    pub endpoint_type: EndpointType,
    /// Base URL for HTTP-based endpoints.
    pub url: String,
    /// Model name (LLM endpoints only).
    pub model: Option<String>,
    /// API key / bearer token.
    pub api_key: Option<String>,
    /// Custom HTTP headers.
    pub headers: HashMap<String, String>,
    /// Command for MCP subprocess endpoints.
    pub command: Option<String>,
    /// Arguments for MCP subprocess endpoints.
    pub args: Vec<String>,
    /// Whether this endpoint accepts image parts (vision).
    pub vision: bool,
    /// Input token pricing per 1M tokens.
    pub input_price_per_1m: f64,
    /// Output token pricing per 1M tokens.
    pub output_price_per_1m: f64,
    /// Flat cost per call.
    pub per_call_price: f64,
}

impl ResolvedEndpoint {
    /// Creates a default LLM endpoint from environment variables.
    #[must_use]
    pub fn default_llm() -> Self {
        Self {
            name: "default".to_owned(),
            endpoint_type: EndpointType::Llm,
            url: crate::llm_base_url(),
            model: Some(crate::llm_model()),
            api_key: std::env::var("HARNESS_LLM_API_KEY").ok(),
            headers: crate::parse_headers_env(),
            command: None,
            args: Vec::new(),
            vision: false,
            input_price_per_1m: 0.0,
            output_price_per_1m: 0.0,
            per_call_price: 0.0,
        }
    }
}

/// Registry of all configured endpoints with routing logic.
#[derive(Debug, Clone)]
pub struct EndpointRegistry {
    endpoints: HashMap<String, ResolvedEndpoint>,
    default_for: HashMap<String, String>,
}

impl EndpointRegistry {
    /// Builds a registry from the endpoint definitions in scenario config.
    ///
    /// Falls back to a default LLM endpoint derived from `fallback_llm`
    /// (the runner's effective config — CLI arguments merged over env vars
    /// and scenario fields) when no `[config.endpoints]` are defined, so
    /// `--llm-url` / `--llm-model` / `--llm-api-key` are honored even for
    /// scenarios without an explicit endpoint table. Without a fallback,
    /// environment variables are used.
    #[must_use]
    pub fn from_config(
        endpoints: &HashMap<String, EndpointConfig>,
        fallback_llm: Option<&crate::LlmConfig>,
    ) -> Self {
        if endpoints.is_empty() {
            let default_llm =
                fallback_llm.map_or_else(ResolvedEndpoint::default_llm, |llm| ResolvedEndpoint {
                    name: "default".to_owned(),
                    endpoint_type: EndpointType::Llm,
                    url: llm.url.clone(),
                    model: Some(llm.model.clone()),
                    api_key: llm.api_key.clone(),
                    headers: llm.headers.clone(),
                    command: None,
                    args: Vec::new(),
                    vision: false,
                    input_price_per_1m: 0.0,
                    output_price_per_1m: 0.0,
                    per_call_price: 0.0,
                });
            let mut map = HashMap::new();
            let mut default_for = HashMap::new();
            for tt in &[TaskType::Targeting, TaskType::Assertion] {
                default_for.insert(tt.as_str().to_owned(), "default".to_owned());
            }
            map.insert("default".to_owned(), default_llm);
            return Self {
                endpoints: map,
                default_for,
            };
        }

        let mut resolved: HashMap<String, ResolvedEndpoint> = HashMap::new();
        let mut default_for: HashMap<String, String> = HashMap::new();

        for (name, ec) in endpoints {
            let re = ResolvedEndpoint {
                name: name.clone(),
                endpoint_type: ec.endpoint_type.clone(),
                url: ec
                    .url
                    .clone()
                    .unwrap_or_else(|| match ec.endpoint_type {
                        EndpointType::Llm => crate::llm_base_url(),
                        EndpointType::A2a | EndpointType::Mcp => String::new(),
                    })
                    .trim_end_matches('/')
                    .to_owned(),
                model: ec.model.clone(),
                api_key: ec.api_key.clone(),
                headers: ec.headers.clone(),
                command: ec.command.clone(),
                args: ec.args.clone(),
                vision: ec.vision,
                input_price_per_1m: ec.pricing.as_ref().map_or(0.0, |p| p.input_per_1m_tokens),
                output_price_per_1m: ec.pricing.as_ref().map_or(0.0, |p| p.output_per_1m_tokens),
                per_call_price: ec.pricing.as_ref().map_or(0.0, |p| p.per_call),
            };

            for df in &ec.default_for {
                default_for.insert(df.clone(), name.clone());
            }

            resolved.insert(name.clone(), re);
        }

        Self {
            endpoints: resolved,
            default_for,
        }
    }

    /// Resolves an endpoint by explicit name.
    ///
    /// Returns `None` if no endpoint with the given name exists.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ResolvedEndpoint> {
        self.endpoints.get(name)
    }

    /// Resolves the best endpoint for a given task type.
    ///
    /// Checks for a `default_for` mapping first, then falls back to any LLM
    /// endpoint, then panics (config error).
    #[must_use]
    pub fn resolve_for_task(&self, task: TaskType) -> &ResolvedEndpoint {
        let key = task.as_str();
        if let Some(name) = self.default_for.get(key) {
            if let Some(ep) = self.endpoints.get(name) {
                return ep;
            }
        }
        // Fallback: first LLM endpoint
        self.endpoints
            .values()
            .find(|ep| ep.endpoint_type == EndpointType::Llm)
            .unwrap_or_else(|| panic!("no LLM endpoint configured for task {key}"))
    }

    /// Resolves an endpoint: explicit name takes priority, then task-type
    /// routing, then first LLM endpoint.
    #[must_use]
    pub fn resolve(&self, name: Option<&str>, task: TaskType) -> &ResolvedEndpoint {
        if let Some(n) = name {
            if let Some(ep) = self.endpoints.get(n) {
                return ep;
            }
        }
        self.resolve_for_task(task)
    }

    /// Returns the number of configured endpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// Returns true if no endpoints are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_task_type_as_str() {
        assert_eq!(TaskType::Targeting.as_str(), "targeting");
        assert_eq!(TaskType::Assertion.as_str(), "assertion");
    }

    #[test]
    fn test_registry_empty_config() {
        let endpoints = HashMap::new();
        let registry = EndpointRegistry::from_config(&endpoints, None);
        assert_eq!(registry.len(), 1);
        let ep = registry.get("default").unwrap();
        assert_eq!(ep.endpoint_type, EndpointType::Llm);
    }

    #[test]
    fn test_registry_resolve_by_name() {
        let mut endpoints = HashMap::new();
        endpoints.insert(
            "vision".to_owned(),
            EndpointConfig {
                endpoint_type: EndpointType::Llm,
                url: Some("https://api.openai.com".into()),
                model: Some("gpt-4o".into()),
                ..Default::default()
            },
        );

        let registry = EndpointRegistry::from_config(&endpoints, None);
        let ep = registry.get("vision");
        assert!(ep.is_some());
        assert_eq!(ep.unwrap().model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn test_resolve_for_task_with_default() {
        let mut endpoints = HashMap::new();
        let ec = EndpointConfig {
            endpoint_type: EndpointType::Llm,
            url: Some("http://localhost:8080".into()),
            model: Some("deepseek".into()),
            default_for: vec!["targeting".to_owned()],
            ..Default::default()
        };
        endpoints.insert("main".to_owned(), ec);

        let registry = EndpointRegistry::from_config(&endpoints, None);
        let ep = registry.resolve_for_task(TaskType::Targeting);
        assert_eq!(ep.name, "main");
    }

    #[test]
    fn test_resolve_explicit_overrides_task() {
        let mut endpoints = HashMap::new();
        endpoints.insert(
            "default".to_owned(),
            EndpointConfig {
                endpoint_type: EndpointType::Llm,
                url: Some("http://default".into()),
                default_for: vec!["targeting".to_owned()],
                ..Default::default()
            },
        );
        endpoints.insert(
            "fast".to_owned(),
            EndpointConfig {
                endpoint_type: EndpointType::Llm,
                url: Some("http://fast".into()),
                ..Default::default()
            },
        );

        let registry = EndpointRegistry::from_config(&endpoints, None);
        let ep = registry.resolve(Some("fast"), TaskType::Targeting);
        assert_eq!(ep.name, "fast");
    }

    #[test]
    fn test_default_llm_has_env_values() {
        let ep = ResolvedEndpoint::default_llm();
        assert_eq!(ep.endpoint_type, EndpointType::Llm);
        assert!(ep.model.is_some());
        assert!(!ep.url.is_empty());
    }
}
