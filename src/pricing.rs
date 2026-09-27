//! Automatic pricing lookup from providers that expose exact, public
//! per-token prices.
//!
//! Two sources are supported:
//!
//! - `OpenRouter`: `GET /api/v1/models` carries exact per-token `prompt`,
//!   `completion`, `input_cache_read` and `input_cache_write` prices with no
//!   authentication.
//! - AWS Bedrock: the public AWS Price List offers `AmazonBedrock` (Nova,
//!   Llama, Mistral, …) and `AmazonBedrockFoundationModels` (Anthropic
//!   Claude, Cohere, …) publish exact per-model input/output/cache-read/
//!   cache-write prices per region. Both are fetched at startup and mapped to
//!   the configured model id.
//!
//! Other providers either expose no public price API (`OpenAI`, Groq, xAI,
//! `DeepSeek`, Google) or expose prices that cannot be mapped to a model
//! exactly without heuristics (Azure Retail Prices).

use std::collections::HashMap;

use crate::scenario::{EndpointConfig, Provider};

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

/// Which exact-pricing source an endpoint opts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// `OpenRouter` public models API.
    OpenRouter,
    /// AWS Bedrock Price List (`AmazonBedrock` + `AmazonBedrockFoundationModels`).
    Bedrock,
}

/// Resolves the pricing source for an endpoint.
///
/// `pricing_source` is `"openrouter"`, `"bedrock"` or `"auto"`. `"auto"` uses
/// Bedrock for Bedrock endpoints and `OpenRouter` when the URL host is
/// `openrouter.ai`.
#[must_use]
fn source_for(ec: &EndpointConfig) -> Option<Source> {
    match ec.pricing_source.as_deref() {
        Some("openrouter") => Some(Source::OpenRouter),
        Some("bedrock") => Some(Source::Bedrock),
        Some("auto") => {
            if ec.provider == Provider::Bedrock {
                Some(Source::Bedrock)
            } else if ec
                .url
                .as_deref()
                .is_some_and(|url| url.to_ascii_lowercase().contains("openrouter.ai"))
            {
                Some(Source::OpenRouter)
            } else {
                None
            }
        }
        _ => None,
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
#[allow(clippy::implicit_hasher, clippy::too_many_lines)]
pub async fn apply_auto_pricing(
    endpoints: &mut HashMap<String, EndpointConfig>,
    client: &reqwest::Client,
) -> Result<usize, String> {
    let mut priced = 0;
    let mut first_error: Option<String> = None;
    // Fetched once per region, reused across Bedrock endpoints.
    let mut bedrock_catalogs: HashMap<String, HashMap<String, ModelPricing>> = HashMap::new();
    for ec in endpoints.values_mut() {
        let Some(model) = ec.model.clone() else {
            continue;
        };
        let result = match source_for(ec) {
            Some(Source::OpenRouter) => fetch_openrouter_pricing(client, &model).await,
            Some(Source::Bedrock) => {
                let region = bedrock_region(ec);
                if !bedrock_catalogs.contains_key(&region) {
                    match fetch_bedrock_catalog(client, &region).await {
                        Ok(catalog) => {
                            bedrock_catalogs.insert(region.clone(), catalog);
                        }
                        Err(e) => {
                            first_error.get_or_insert(e);
                            continue;
                        }
                    }
                }
                let catalog = &bedrock_catalogs[&region];
                lookup_bedrock(catalog, &model)
                    .ok_or_else(|| format!("model `{model}` not found in Bedrock pricing"))
            }
            None => continue,
        };
        match result {
            Ok(pricing) => {
                apply_pricing(ec, &pricing);
                priced += 1;
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    match first_error {
        Some(e) if priced == 0 => Err(e),
        _ => Ok(priced),
    }
}

/// Resolves the region whose Bedrock price list applies to an endpoint:
/// the endpoint's explicit `aws.region`, else `AWS_REGION`/`AWS_DEFAULT_REGION`,
/// else `us-east-1`.
#[must_use]
fn bedrock_region(ec: &EndpointConfig) -> String {
    ec.aws
        .region
        .clone()
        .or_else(|| std::env::var("AWS_REGION").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .filter(|r| !r.trim().is_empty())
        .unwrap_or_else(|| "us-east-1".to_owned())
}

/// Base host of the public AWS Price List (always `us-east-1`, global).
const PRICING_HOST: &str = "https://pricing.us-east-1.amazonaws.com";

/// AWS Price List offers that together cover all Bedrock models: AWS-published
/// models (Nova, Llama, Mistral, …) and Marketplace models (Anthropic, …).
const BEDROCK_OFFERS: [&str; 2] = ["AmazonBedrock", "AmazonBedrockFoundationModels"];

/// Fetches and merges the Bedrock price catalogs for `region` into a map keyed
/// by the normalized model name.
///
/// # Errors
///
/// Returns a description when either offer cannot be fetched or the region is
/// absent from its region index.
async fn fetch_bedrock_catalog(
    client: &reqwest::Client,
    region: &str,
) -> Result<HashMap<String, ModelPricing>, String> {
    let mut acc: HashMap<String, Partial> = HashMap::new();
    for offer in BEDROCK_OFFERS {
        let doc = fetch_offer_region(client, offer, region).await?;
        // The FoundationModels (Marketplace) offer wins on conflicts: it is
        // the current source for Anthropic and carries cache write prices.
        let overwrite = offer == "AmazonBedrockFoundationModels";
        parse_bedrock_offer(&doc, &mut acc, overwrite);
    }
    Ok(acc
        .into_iter()
        .filter_map(|(key, partial)| partial.finish().map(|p| (key, p)))
        .collect())
}

/// Fetches one offer's price file for `region` by following its region index.
async fn fetch_offer_region(
    client: &reqwest::Client,
    offer: &str,
    region: &str,
) -> Result<serde_json::Value, String> {
    let index_url = format!("{PRICING_HOST}/offers/v1.0/aws/{offer}/current/region_index.json");
    let index: serde_json::Value = client
        .get(&index_url)
        .send()
        .await
        .map_err(|e| format!("{offer} region index request failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("{offer} region index was not JSON: {e}"))?;
    let path = index["regions"][region]["currentVersionUrl"]
        .as_str()
        .ok_or_else(|| format!("region `{region}` not present in {offer} price list"))?;
    let url = format!("{PRICING_HOST}{path}");
    client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("{offer} price list request failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("{offer} price list was not JSON: {e}"))
}

/// Role of a price entry within a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Ordinary input (prompt) tokens.
    Input,
    /// Output (completion) tokens.
    Output,
    /// Prompt cache read (cached input) tokens.
    Read,
    /// Prompt cache write (cache creation) tokens.
    Write,
}

/// Per-model accumulator while merging the two offers.
#[derive(Debug, Default, Clone, Copy)]
struct Partial {
    input: Option<f64>,
    output: Option<f64>,
    read: Option<f64>,
    write: Option<f64>,
}

impl Partial {
    const fn set(&mut self, role: Role, value: f64, overwrite: bool) {
        let slot = match role {
            Role::Input => &mut self.input,
            Role::Output => &mut self.output,
            Role::Read => &mut self.read,
            Role::Write => &mut self.write,
        };
        if overwrite || slot.is_none() {
            *slot = Some(value);
        }
    }

    fn finish(self) -> Option<ModelPricing> {
        Some(ModelPricing {
            input_per_1m: self.input?,
            output_per_1m: self.output?,
            cached_input_per_1m: self.read,
            cache_write_per_1m: self.write,
        })
    }
}

/// Parses one offer document into `acc`, keyed by normalized model name.
fn parse_bedrock_offer(
    doc: &serde_json::Value,
    acc: &mut HashMap<String, Partial>,
    overwrite: bool,
) {
    let Some(products) = doc["products"].as_object() else {
        return;
    };
    for (sku, product) in products {
        let a = &product["attributes"];
        let Some((model, role)) = bedrock_entry(a) else {
            continue;
        };
        let Some((usd, unit)) = on_demand_price(doc, sku) else {
            continue;
        };
        let Some(per_million) = to_per_million(usd, &unit) else {
            continue;
        };
        acc.entry(compact_key(model))
            .or_default()
            .set(role, per_million, overwrite);
    }
}

/// Extracts the model name and price role from a product's attributes, for
/// either offer shape. Returns `None` for non-standard tiers (batch, flex,
/// priority, global, latency-optimized, provisioned throughput).
fn bedrock_entry(attrs: &serde_json::Value) -> Option<(&str, Role)> {
    // Offer 1 (`AmazonBedrock`): a non-empty `model` attribute, role in
    // `inferenceType`, restricted to on-demand inference.
    if let Some(model) = attrs["model"].as_str().filter(|m| !m.is_empty()) {
        if attrs["feature"].as_str() != Some("On-demand Inference") {
            return None;
        }
        if attrs["batch"].as_str().is_some_and(|b| !b.is_empty()) {
            return None;
        }
        let role = offer1_role(attrs["inferenceType"].as_str().unwrap_or_default())?;
        return Some((model, role));
    }
    // Offer 2 (`AmazonBedrockFoundationModels`): model in `servicename`
    // (always suffixed `(Amazon Bedrock Edition)`), role in `usagetype`.
    if let Some(servicename) = attrs["servicename"].as_str() {
        if let Some(role) = offer2_role(attrs["usagetype"].as_str().unwrap_or_default()) {
            let model = servicename
                .strip_suffix(" (Amazon Bedrock Edition)")
                .unwrap_or(servicename);
            return Some((model, role));
        }
    }
    None
}

/// Lowercases and drops separators so hyphen/underscore/camel spellings match.
fn normalize_token(value: &str) -> String {
    value
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Maps an `AmazonBedrock` `inferenceType` to a role, excluding tier variants.
fn offer1_role(inference_type: &str) -> Option<Role> {
    let s = normalize_token(inference_type);
    if ["priority", "flex", "batch", "global", "latency"]
        .iter()
        .any(|t| s.contains(t))
    {
        return None;
    }
    if s.contains("cacheread") {
        Some(Role::Read)
    } else if s.contains("cachewrite") {
        Some(Role::Write)
    } else if s.contains("input")
        && !s.contains("image")
        && !s.contains("video")
        && !s.contains("audio")
    {
        Some(Role::Input)
    } else if s.contains("output") && !s.contains("image") && !s.contains("video") {
        Some(Role::Output)
    } else {
        None
    }
}

/// Maps an `AmazonBedrockFoundationModels` `usagetype` to a role, excluding
/// tier/TTL variants.
fn offer2_role(usagetype: &str) -> Option<Role> {
    let s = normalize_token(usagetype);
    if [
        "global", "batch", "priority", "flex", "latency", "1h", "30m", "custom",
    ]
    .iter()
    .any(|t| s.contains(t))
    {
        return None;
    }
    if s.contains("cacheread") {
        Some(Role::Read)
    } else if s.contains("cachewrite") {
        Some(Role::Write)
    } else if s.contains("inputtoken") {
        Some(Role::Input)
    } else if s.contains("outputtoken") {
        Some(Role::Output)
    } else {
        None
    }
}

/// Reads the first `OnDemand` price and its unit for a SKU.
fn on_demand_price(doc: &serde_json::Value, sku: &str) -> Option<(f64, String)> {
    let offers = doc["terms"]["OnDemand"][sku].as_object()?;
    let offer = offers.values().next()?;
    let dimension = offer["priceDimensions"].as_object()?.values().next()?;
    let usd = dimension["pricePerUnit"]["USD"]
        .as_str()?
        .parse::<f64>()
        .ok()?;
    let unit = dimension["unit"].as_str().unwrap_or_default().to_owned();
    Some((usd, unit))
}

/// Converts a price to USD per 1M tokens from its unit string.
fn to_per_million(price: f64, unit: &str) -> Option<f64> {
    let unit = unit.to_ascii_lowercase();
    if unit.contains("1m") || unit.contains("million") {
        Some(price)
    } else if unit.contains("1k") || unit.contains("thousand") {
        Some(price * 1_000.0)
    } else if unit.contains("token") {
        Some(price * 1_000_000.0)
    } else {
        None
    }
}

/// Inference-profile prefixes on Bedrock model ids (`us.`, `global.`, …).
const MODEL_PREFIXES: [&str; 12] = [
    "us", "eu", "apac", "global", "us-gov", "ca", "sa", "me", "af", "il", "ap", "gov",
];

/// Provider prefixes on Bedrock model ids (`anthropic.`, `amazon.`, …).
const MODEL_PROVIDERS: [&str; 21] = [
    "anthropic",
    "amazon",
    "meta",
    "mistral",
    "cohere",
    "ai21",
    "stability",
    "deepseek",
    "openai",
    "google",
    "qwen",
    "writer",
    "nvidia",
    "minimax",
    "moonshot",
    "moonshotai",
    "zai",
    "xai",
    "kimi",
    "twelvelabs",
    "luma",
];

/// Normalizes a model id or display name to a compact comparison key.
///
/// Strips inference-profile prefixes (`us.`, `global.`, …) and provider
/// prefixes (`anthropic.`, `amazon.`, …), the `:0` revision suffix, 8-digit
/// date tokens, and all separators, so `us.anthropic.claude-3-5-sonnet-
/// 20241022-v2:0` and `Claude 3.5 Sonnet v2` both become `claude35sonnetv2`.
#[must_use]
fn model_key(model: &str) -> String {
    let lower = model.to_ascii_lowercase();
    let no_revision = lower.split(':').next().unwrap_or(&lower);
    let kept: Vec<&str> = no_revision
        .split('.')
        .filter(|s| !MODEL_PREFIXES.contains(s) && !MODEL_PROVIDERS.contains(s))
        .collect();
    let joined = if kept.is_empty() {
        no_revision
    } else {
        &kept.join("-")
    };
    compact_key(joined)
}

/// Removes separators and 8-digit date tokens, lowercasing the result.
#[must_use]
fn compact_key(value: &str) -> String {
    let mut out = String::new();
    for token in value.split(['-', '_', ' ']).filter(|t| !t.is_empty()) {
        if token.len() == 8 && token.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        out.push_str(token);
    }
    out.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Finds the catalog entry matching `model`: exact key first, then the closest
/// prefix match, then a legacy fallback that drops a trailing `vN`.
#[must_use]
fn lookup_bedrock(catalog: &HashMap<String, ModelPricing>, model: &str) -> Option<ModelPricing> {
    let key = model_key(model);
    if let Some(p) = catalog.get(&key) {
        return Some(*p);
    }
    if let Some(p) = prefix_lookup(catalog, &key) {
        return Some(p);
    }
    if let Some(stripped) = strip_trailing_version(&key) {
        if let Some(p) = catalog.get(&stripped) {
            return Some(*p);
        }
        if let Some(p) = prefix_lookup(catalog, &stripped) {
            return Some(p);
        }
    }
    None
}

/// Closest catalog key where one key is a prefix of the other.
fn prefix_lookup(catalog: &HashMap<String, ModelPricing>, key: &str) -> Option<ModelPricing> {
    let mut keys: Vec<&String> = catalog.keys().collect();
    keys.sort();
    let mut best: Option<(&String, usize)> = None;
    for candidate in keys {
        if key.starts_with(candidate.as_str()) || candidate.starts_with(key) {
            let diff = candidate.len().abs_diff(key.len());
            if best.as_ref().is_none_or(|(_, d)| diff < *d) {
                best = Some((candidate, diff));
            }
        }
    }
    best.and_then(|(k, _)| catalog.get(k).copied())
}

/// Drops a trailing `v<digits>` from a compact key (`claudev2` → `claude`).
fn strip_trailing_version(key: &str) -> Option<String> {
    let idx = key.rfind('v')?;
    let rest = &key[idx + 1..];
    if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
        Some(key[..idx].to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_pricing, lookup_bedrock, model_key, parse_bedrock_offer, parse_openrouter_models,
        to_per_million, Partial,
    };
    use crate::scenario::{EndpointConfig, PricingConfig};
    use std::collections::HashMap;

    /// Builds a Bedrock catalog from synthetic offer documents.
    fn bedrock_catalog() -> HashMap<String, super::ModelPricing> {
        let offer1 = serde_json::json!({
            "products": {
                "s1": {"attributes": {"model": "Nova Lite", "feature": "On-demand Inference",
                    "inferenceType": "Input tokens", "batch": ""}},
                "s2": {"attributes": {"model": "Nova Lite", "feature": "On-demand Inference",
                    "inferenceType": "Output tokens", "batch": ""}},
                "s3": {"attributes": {"model": "Nova Lite", "feature": "On-demand Inference",
                    "inferenceType": "Input tokens flex", "batch": ""}},
                "s4": {"attributes": {"model": "Nova Lite", "feature": "Batch Inference",
                    "inferenceType": "Input tokens", "batch": "true"}}
            },
            "terms": {"OnDemand": {
                "s1": {"o": {"priceDimensions": {"d": {"unit": "1K tokens",
                    "pricePerUnit": {"USD": "0.00006"}}}}},
                "s2": {"o": {"priceDimensions": {"d": {"unit": "1K tokens",
                    "pricePerUnit": {"USD": "0.00024"}}}}},
                "s3": {"o": {"priceDimensions": {"d": {"unit": "1K tokens",
                    "pricePerUnit": {"USD": "0.00005"}}}}},
                "s4": {"o": {"priceDimensions": {"d": {"unit": "1K tokens",
                    "pricePerUnit": {"USD": "0.00003"}}}}}
            }}
        });
        let offer2 = serde_json::json!({
            "products": {
                "c1": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_InputTokenCount-Units"}},
                "c2": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_OutputTokenCount-Units"}},
                "c3": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_CacheReadInputTokenCount-Units"}},
                "c4": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_CacheWriteInputTokenCount-Units"}},
                "c5": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_CacheWrite1hInputTokenCount-Units"}},
                "c6": {"attributes": {"servicename": "Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)",
                    "usagetype": "USE1-MP:USE1_InputTokenCount_Global-Units"}}
            },
            "terms": {"OnDemand": {
                "c1": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "3.0"}}}}},
                "c2": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "15.0"}}}}},
                "c3": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "0.3"}}}}},
                "c4": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "3.75"}}}}},
                "c5": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "6.0"}}}}},
                "c6": {"o": {"priceDimensions": {"d": {"unit": "1M tokens",
                    "pricePerUnit": {"USD": "99.0"}}}}}
            }}
        });
        let mut acc: HashMap<String, Partial> = HashMap::new();
        parse_bedrock_offer(&offer1, &mut acc, false);
        parse_bedrock_offer(&offer2, &mut acc, true);
        acc.into_iter()
            .filter_map(|(k, p)| p.finish().map(|mp| (k, mp)))
            .collect()
    }

    #[test]
    fn bedrock_model_key_normalizes_profiles_dates_and_revisions() {
        assert_eq!(
            model_key("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "claude35sonnetv2"
        );
        assert_eq!(model_key("amazon.nova-lite-v1:0"), "novalitev1");
        assert_eq!(model_key("Claude 3.5 Sonnet v2"), "claude35sonnetv2");
        assert_eq!(
            model_key("anthropic.claude-3-haiku-20240307-v1:0"),
            "claude3haikuv1"
        );
    }

    #[test]
    fn bedrock_units_convert_to_per_million() {
        assert!((to_per_million(0.003, "1K tokens").unwrap() - 3.0).abs() < 1e-9);
        assert!((to_per_million(3.0, "1M tokens").unwrap() - 3.0).abs() < 1e-9);
        assert!((to_per_million(0.000_003, "tokens").unwrap() - 3.0).abs() < 1e-9);
        assert!(to_per_million(176.0, "hour").is_none());
    }

    #[test]
    fn bedrock_offer1_skips_tier_variants() {
        let catalog = bedrock_catalog();
        let nova = catalog.get("novalite").unwrap();
        // Standard on-demand input/output only; the flex variant must not win.
        assert!(
            (nova.input_per_1m - 0.06).abs() < 1e-9,
            "got {}",
            nova.input_per_1m
        );
        assert!((nova.output_per_1m - 0.24).abs() < 1e-9);
    }

    #[test]
    fn bedrock_offer2_includes_cache_and_skips_global_and_1h() {
        let catalog = bedrock_catalog();
        let claude = catalog.get("claude35sonnetv2").unwrap();
        assert!((claude.input_per_1m - 3.0).abs() < 1e-9);
        assert!((claude.output_per_1m - 15.0).abs() < 1e-9);
        assert!((claude.cached_input_per_1m.unwrap() - 0.3).abs() < 1e-9);
        // The 1h TTL variant must not overwrite the default (5m) cache write.
        assert!((claude.cache_write_per_1m.unwrap() - 3.75).abs() < 1e-9);
    }

    #[test]
    fn bedrock_lookup_matches_full_ids_and_prefixes() {
        let catalog = bedrock_catalog();
        assert!(lookup_bedrock(&catalog, "us.anthropic.claude-3-5-sonnet-20241022-v2:0").is_some());
        assert!(lookup_bedrock(&catalog, "anthropic.claude-3-5-sonnet-20240620-v1:0").is_some());
        assert!(lookup_bedrock(&catalog, "amazon.nova-lite-v1:0").is_some());
        assert!(lookup_bedrock(&catalog, "amazon.nova-pro-v1:0").is_none());
    }

    /// Live check against the real AWS Price List (network). Run with:
    /// `cargo test --all-features bedrock_live_prices -- --ignored`
    #[tokio::test]
    #[ignore = "hits the live AWS Price List"]
    async fn bedrock_live_prices() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        let catalog = super::fetch_bedrock_catalog(&client, "us-east-1")
            .await
            .expect("fetch live catalog");
        let claude = lookup_bedrock(&catalog, "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
            .expect("claude 3.5 sonnet v2");
        assert!((claude.input_per_1m - 3.0).abs() < 1e-6, "{claude:?}");
        assert!((claude.output_per_1m - 15.0).abs() < 1e-6, "{claude:?}");
        assert!(
            (claude.cached_input_per_1m.unwrap() - 0.3).abs() < 1e-6,
            "{claude:?}"
        );
        assert!(
            (claude.cache_write_per_1m.unwrap() - 3.75).abs() < 1e-6,
            "{claude:?}"
        );
        let nova = lookup_bedrock(&catalog, "amazon.nova-lite-v1:0").expect("nova lite");
        assert!(nova.input_per_1m > 0.0, "{nova:?}");
    }

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
