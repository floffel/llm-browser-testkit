//! Cost and token report printer.

use crate::costs::UsageSnapshot;

/// Prints a cost report to stderr after all tests complete.
pub fn print_report(per_test: &[(String, UsageSnapshot)], global: &UsageSnapshot) {
    if per_test.is_empty() {
        return;
    }

    eprintln!();
    eprintln!("═══════════════════════════════════════════════");
    eprintln!("  COST REPORT");
    eprintln!("═══════════════════════════════════════════════");

    for (test_name, snapshot) in per_test {
        eprintln!(
            "  Test: \"{test_name}\" — ${cost:.4} | {tokens} tokens | {calls} calls",
            cost = snapshot.total_cost,
            tokens = snapshot.total_tokens,
            calls = snapshot.total_calls,
        );
        for (ep_name, ep_usage) in &snapshot.endpoints {
            if ep_usage.calls == 0 {
                continue;
            }
            eprintln!(
                "    endpoint.{ep_name}:   {calls:>3} calls, {tokens:>7} tokens, ${cost:.4}",
                calls = ep_usage.calls,
                tokens = ep_usage.input_tokens + ep_usage.output_tokens,
                cost = ep_usage.cost,
            );
        }
    }

    eprintln!("───────────────────────────────────────────────");
    eprintln!("  GLOBAL SUMMARY");
    eprintln!("    Total cost:     ${cost:.4}", cost = global.total_cost);
    eprintln!("    Total tokens:   {tokens}", tokens = global.total_tokens);
    eprintln!("    Total calls:    {calls}", calls = global.total_calls);
    eprintln!("═══════════════════════════════════════════════");
}

/// Prints a budget exceeded warning to stderr.
pub fn print_budget_warning(message: &str) {
    eprintln!("  ⚠️  BUDGET WARNING: {message}");
}

/// Prints a budget exceeded hard error to stderr.
pub fn print_budget_error(message: &str) {
    eprintln!("  🛑 BUDGET EXCEEDED: {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::costs::{EndpointUsage, UsageSnapshot};
    use std::collections::HashMap;

    fn make_snapshot(cost: f64, tokens: u64, calls: u64) -> UsageSnapshot {
        let mut eps = HashMap::new();
        eps.insert(
            "default".to_owned(),
            EndpointUsage {
                calls,
                input_tokens: tokens / 2,
                output_tokens: tokens / 2,
                cost,
            },
        );
        UsageSnapshot::from_endpoints(&eps)
    }

    #[test]
    fn test_print_report_empty() {
        // Should return early, no panic
        print_report(&[], &UsageSnapshot::default());
    }

    #[test]
    fn test_print_report_single_test() {
        let per_test = vec![("test1".to_owned(), make_snapshot(0.05, 500, 3))];
        // Should not panic
        print_report(&per_test, &make_snapshot(0.05, 500, 3));
    }

    #[test]
    fn test_print_report_zero_cost() {
        let per_test = vec![("free".to_owned(), make_snapshot(0.0, 0, 0))];
        print_report(&per_test, &make_snapshot(0.0, 0, 0));
    }

    #[test]
    fn test_print_budget_warning_no_panic() {
        print_budget_warning("cost limit $1.00 exceeded");
    }

    #[test]
    fn test_print_budget_error_no_panic() {
        print_budget_error("cost limit $5.00 exceeded");
    }
}
