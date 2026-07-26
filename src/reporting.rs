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
