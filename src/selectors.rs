//! CSS selector sanitization and validation for LLM-generated selectors.
//!
//! The LLM sometimes returns garbage instead of a CSS selector (explanations,
//! JSON objects, `:not(*)`, `null`, stray quotes or code fences). These
//! helpers normalize the response and reject values that can never match an
//! element, so step errors show the LLM's actual output instead of a cryptic
//! "element :not(*) not found".

/// Strips common LLM response noise from a raw selector string:
/// surrounding whitespace, code fences (with optional language tag), and
/// outer single/double/backtick quotes.
///
/// Returns an empty string when nothing meaningful remains.
#[must_use]
pub fn sanitize_selector(raw: &str) -> String {
    let mut s = raw.trim().to_owned();
    // ```css ... ``` / ``` ... ``` code fences
    if let Some(rest) = s.strip_prefix("```") {
        let rest = rest.strip_suffix("```").unwrap_or(rest);
        s = rest.trim().to_owned();
        if let Some((_, after_lang)) = s.split_once('\n') {
            s = after_lang.trim().to_owned();
        }
    }
    // Backtick-wrapped: `selector`
    if s.len() >= 2 && s.starts_with('`') && s.ends_with('`') {
        s = s[1..s.len() - 1].to_owned();
    }
    // Quote-wrapped: "selector" or 'selector'
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        s = s[1..s.len() - 1].to_owned();
    }
    s.trim().to_owned()
}

/// Returns `true` when a sanitized selector can never match a real element,
/// i.e. the LLM clearly did not produce a selector at all.
///
/// Used to fail fast with a readable error instead of handing the garbage to
/// the browser's `querySelector`.
#[must_use]
pub fn selector_is_useless(sel: &str) -> bool {
    let s = sel.trim();
    if s.is_empty() {
        return true;
    }
    let lower = s.to_lowercase();
    matches!(
        lower.as_str(),
        "*" | ":not(*)" | "null" | "undefined" | "none" | "n/a" | "na" | "nil"
    ) || lower.starts_with(":not(")
        || lower.starts_with('{')
        || lower.starts_with("json")
        || s.contains('\n')
}

/// Basic syntactic sanity check for a selector before it is executed.
///
/// Returns a human-readable reason when the selector is obviously malformed
/// (unbalanced brackets/parens/quotes), or `Ok` otherwise. This is a cheap
/// pre-flight only — the browser's `querySelector` remains the authority.
///
/// # Errors
///
/// Returns a description of the malformed part when the selector has
/// unbalanced brackets, parentheses, or quotes.
pub fn validate_selector(sel: &str) -> Result<(), String> {
    let mut parens = 0i32;
    let mut brackets = 0i32;
    let mut in_quote: Option<char> = None;
    for c in sel.chars() {
        if let Some(q) = in_quote {
            if c == q {
                in_quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => in_quote = Some(c),
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            _ => {}
        }
        if parens < 0 || brackets < 0 {
            return Err(format!("unbalanced '{c}' in selector: {sel:?}"));
        }
    }
    if in_quote.is_some() {
        return Err(format!("unterminated quote in selector: {sel:?}"));
    }
    if parens != 0 || brackets != 0 {
        return Err(format!("unbalanced brackets in selector: {sel:?}"));
    }
    Ok(())
}

/// Builds the JavaScript snippet that checks whether a selector matches at
/// least one element on the current page.
#[must_use]
pub fn selector_matches_js(selector: &str) -> String {
    let escaped = selector.replace('\\', "\\\\").replace('\'', "\\'");
    format!("document.querySelector('{escaped}') !== null")
}

#[cfg(test)]
mod tests {
    use super::{sanitize_selector, selector_is_useless, validate_selector};

    #[test]
    fn test_sanitize_trims_whitespace() {
        assert_eq!(sanitize_selector("  #login  "), "#login");
    }

    #[test]
    fn test_sanitize_strips_code_fence() {
        assert_eq!(
            sanitize_selector("```css\nbutton.btn--primary\n```"),
            "button.btn--primary"
        );
    }

    #[test]
    fn test_sanitize_strips_quotes() {
        assert_eq!(sanitize_selector("\"button#send\""), "button#send");
        assert_eq!(
            sanitize_selector("'input[name=email]'"),
            "input[name=email]"
        );
        assert_eq!(sanitize_selector("`a.login-link`"), "a.login-link");
    }

    #[test]
    fn test_sanitize_empty() {
        assert_eq!(sanitize_selector("```\n\n```"), "");
        assert_eq!(sanitize_selector("   "), "");
    }

    #[test]
    fn test_sanitize_quoted_fenced() {
        assert_eq!(sanitize_selector("```css\n\"div.card\"\n```"), "div.card");
    }

    #[test]
    fn test_useless_selectors() {
        for bad in [
            "",
            "   ",
            "*",
            ":not(*)",
            ":not(anything)",
            "null",
            "undefined",
            "none",
            "N/A",
            "{",
            "{\"a\":1}",
            "json {\"a\":1}",
        ] {
            assert!(selector_is_useless(bad), "expected {bad:?} to be useless");
        }
    }

    #[test]
    fn test_useful_selectors() {
        for good in [
            "#id",
            "button.btn",
            "input[name=email]",
            ":is(a, button)",
            "a > span.x",
            "div[aria-label='New chat']",
        ] {
            assert!(!selector_is_useless(good), "expected {good:?} to be useful");
        }
    }

    #[test]
    fn test_validate_balanced() {
        assert!(validate_selector("div[aria-label='x']:nth-child(2)").is_ok());
        assert!(validate_selector("a:has(> span)").is_ok());
    }

    #[test]
    fn test_validate_unbalanced() {
        assert!(validate_selector("div[aria-label").is_err());
        assert!(validate_selector("div:nth-child(2").is_err());
        assert!(validate_selector("div:not(").is_err());
        assert!(validate_selector("div[aria-label='x]").is_err());
    }
}
