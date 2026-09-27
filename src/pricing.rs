//! Automatic pricing lookup from providers that expose exact, public
//! per-token prices.
//!
//! Today only `OpenRouter` qualifies: its `GET /api/v1/models` response
//! carries exact per-token `prompt`, `completion`, `input_cache_read` and
//! `input_cache_write` prices with no authentication. Other providers either
//! expose no public price API (`OpenAI`, Groq, xAI, `DeepSeek`, Google) or
//! expose prices that cannot be mapped to a model exactly without heuristics
//! (AWS Bedrock Price List, Azure Retail Prices).

use std::collections::HashMap;

use crate::scenario::EndpointConfig;

/// Exact per-1M-token prices for a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPricing {
    /// Input (prompt) price per 1M tokens.
    pub input_per_1m: f64,
    /// Output (completion) price per 1M tokens.
    pub output_per_1m: f64,
    /// Cached-input (cache read) price per 1M tokens, when offered.
    pub cached_input_per_1m: Option<f64>,
    /// Cache-write (cache creation) price per 1M tokens, when offered.
    pub cache_write_per_1m: Option<f64>,
}

/// Parses an `OpenRouter` `/api/v1/models` response for `model`.
///
/// Matches the full model id case-insensitively first, then falls back to the
/// leaf after the last `/` (so `claude-3.5-sonnet` matches
/// `anthropic/claude-3.5-sonnet`). Returns `None` when the model is absent or
/// its pricing is not a usable positive number.
#[must_use]
pub fn parse_openrouter_models(json: &serde_json::Value, model: &str) -> Option<ModelPricing> {
    let wanted = model.to_ascii_lowercase();
    let leaf = wanted.rsplit('/').next().unwrap_or(wanted.as_str());
    let data = json["data"].as_array()?;
    let entry = data.iter().find(|m| {
        let id = m["id"].as_str().unwrap_or_default().to_ascii_lowercase();
        id == wanted || id.rsplit('/').next().unwrap_or_default() == leaf
    })?;
    let pricing = &entry["pricing"];
    Some(ModelPricing {
        input_per_1m: per_million(pricing, "prompt")?,
        output_per_1m: per_million(pricing, "completion")?,
        cached_input_per_1m: per_million(pricing, "input_cache_read"),
        cache_write_per_1m: per_million(pricing, "input_cache_write"),
    })
}

/// Reads a per-token price (string or number) and converts it to per-1M
/// tokens. Negative values (`OpenRouter` uses `-1` for dynamic/auto prices)
/// and missing keys yield `None`.
fn per_million(pricing: &serde_json::Value, key: &str) -> Option<f64> {
    let value = match &pricing[key] {
        serde_json::Value::String(s) => s.parse::<f64>().ok()?,
        serde_json::Value::Number(n) => n.as_f64()?,
        _ => return None,
    };
    (value >= 0.0).then_some(value * 1_000_000.0)
}

/// Fetches exact `OpenRouter` pricing for `model`.
///
/// # Errors
///
/// Returns a description when the request fails, the response is not JSON, or
/// the model is not present in the catalog.
pub async fn fetch_openrouter_pricing(
    client: &reqwest::Client,
    model: &str,
) -> Result<ModelPricing, String> {
    let resp = client
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await
        .map_err(|e| format!("OpenRouter models request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "OpenRouter models request returned HTTP {}",
            resp.status()
        ));
    }
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("OpenRouter models response was not JSON: {e}"))?;
    parse_openrouter_models(&json, model)
        .ok_or_else(|| format!("model `{model}` not found in OpenRouter pricing"))
}

/// Whether an endpoint opts into automatic pricing, and from where.
#[must_use]
fn wants_openrouter_pricing(ec: &EndpointConfig) -> bool {
    match ec.pricing_source.as_deref() {
        Some("openrouter") => true,
        Some("auto") => ec
            .url
            .as_deref()
            .is_some_and(|url| url.to_ascii_lowercase().contains("openrouter.ai")),
        _ => false,
    }
}

/// Fills unset pricing fields on `ec` from fetched `pricing`; explicit values
/// win.
pub fn apply_pricing(ec: &mut EndpointConfig, pricing: &ModelPricing) {
    let p = ec.pricing.get_or_insert_with(Default::default);
    if p.input_per_1m_tokens == 0.0 {
        p.input_per_1m_tokens = pricing.input_per_1m;
    }
    if p.output_per_1m_tokens == 0.0 {
        p.output_per_1m_tokens = pricing.output_per_1m;
    }
    if p.cached_input_per_1m_tokens.is_none() {
        p.cached_input_per_1m_tokens = pricing.cached_input_per_1m;
    }
    if p.cache_write_per_1m_tokens.is_none() {
        p.cache_write_per_1m_tokens = pricing.cache_write_per_1m;
    }
}

/// Applies automatic pricing to every endpoint that opts in.
///
/// Returns the number of endpoints priced. Endpoints without a model, without
/// a supported source, or whose lookup fails are left untouched. The first
/// failure is returned as an error string while the remaining endpoints are
/// still processed, so a single unknown model never blocks a run.
///
/// # Errors
///
/// Returns the first lookup error when **no** endpoint could be priced; a
/// partial success returns `Ok` with the number priced.
#[allow(clippy::implicit_hasher)]
pub async fn apply_auto_pricing(
    endpoints: &mut HashMap<String, EndpointConfig>,
    client: &reqwest::Client,
) -> Result<usize, String> {
    let mut priced = 0;
    let mut first_error: Option<String> = None;
    for ec in endpoints.values_mut() {
        if !wants_openrouter_pricing(ec) {
            continue;
        }
        let Some(model) = ec.model.clone() else {
            continue;
        };
        match fetch_openrouter_pricing(client, &model).await {
            Ok(pricing) => {
                apply_pricing(ec, &pricing);
                priced += 1;
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        };
    }
    match first_error {
        Some(e) if priced == 0 => Err(e),
        _ => Ok(priced),
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_pricing, parse_openrouter_models};
    use crate::scenario::{EndpointConfig, PricingConfig};

    fn catalog() -> serde_json::Value {
        serde_json::json!({
            "data": [
                {"id": "anthropic/claude-3.5-sonnet", "pricing": {
                    "prompt": "0.000003", "completion": "0.000015",
                    "input_cache_read": "0.0000003", "input_cache_write": "0.00000375"
                }},
                {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}
            ]
        })
    }

    #[test]
    fn parses_exact_match_with_cache_prices() {
        let p = parse_openrouter_models(&catalog(), "anthropic/claude-3.5-sonnet").unwrap();
        assert!((p.input_per_1m - 3.0).abs() < 1e-9);
        assert!((p.output_per_1m - 15.0).abs() < 1e-9);
        assert!((p.cached_input_per_1m.unwrap() - 0.3).abs() < 1e-9);
        assert!((p.cache_write_per_1m.unwrap() - 3.75).abs() < 1e-9);
    }

    #[test]
    fn parses_by_leaf_when_namespaced_omitted() {
        let p = parse_openrouter_models(&catalog(), "claude-3.5-sonnet").unwrap();
        assert!((p.input_per_1m - 3.0).abs() < 1e-9);
    }

    #[test]
    fn dynamic_negative_prices_are_not_usable() {
        assert!(parse_openrouter_models(&catalog(), "openrouter/auto").is_none());
    }

    #[test]
    fn unknown_model_is_none() {
        assert!(parse_openrouter_models(&catalog(), "does/not-exist").is_none());
    }

    #[test]
    fn apply_pricing_keeps_explicit_values() {
        let mut ec = EndpointConfig {
            model: Some("anthropic/claude-3.5-sonnet".to_owned()),
            pricing: Some(PricingConfig {
                input_per_1m_tokens: 9.0,
                output_per_1m_tokens: 0.0,
                ..PricingConfig::default()
            }),
            ..EndpointConfig::default()
        };
        let p = parse_openrouter_models(&catalog(), "anthropic/claude-3.5-sonnet").unwrap();
        apply_pricing(&mut ec, &p);
        let pricing = ec.pricing.unwrap();
        assert!(
            (pricing.input_per_1m_tokens - 9.0).abs() < 1e-9,
            "explicit wins"
        );
        assert!(
            (pricing.output_per_1m_tokens - 15.0).abs() < 1e-9,
            "fetched fills gap"
        );
        assert!(pricing.cached_input_per_1m_tokens.is_some());
        assert!(pricing.cache_write_per_1m_tokens.is_some());
    }
}
