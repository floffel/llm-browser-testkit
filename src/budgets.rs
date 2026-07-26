//! Budget tracking and enforcement.

use crate::costs::UsageSnapshot;
use crate::scenario::{BudgetDef, BudgetEnforcement, BudgetsConfig};

/// Result of a budget check before or after a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetStatus {
    /// Budget is within limits.
    Ok,
    /// Budget exceeded, enforcement is soft — log but continue.
    SoftExceeded {
        /// Which budget was exceeded.
        budget: String,
        /// Human-readable message.
        message: String,
    },
    /// Budget exceeded, enforcement is hard — abort execution.
    HardExceeded {
        /// Which budget was exceeded.
        budget: String,
        /// Human-readable message.
        message: String,
    },
}

/// Tracks overall and per-test budgets across a scenario run.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    global: Option<ResolvedBudget>,
    per_test_default: Option<ResolvedBudget>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedBudget {
    max_cost: Option<f64>,
    max_tokens: Option<u64>,
    max_calls: Option<u64>,
    enforcement: BudgetEnforcement,
}

impl ResolvedBudget {
    fn from_def(def: &BudgetDef) -> Self {
        Self {
            max_cost: def.max_cost,
            max_tokens: def.max_tokens,
            max_calls: def.max_calls,
            enforcement: def.enforcement.clone().unwrap_or(BudgetEnforcement::Hard),
        }
    }
}

impl BudgetTracker {
    /// Creates a new budget tracker from the scenario's budget config.
    #[must_use]
    pub fn from_config(budgets: &BudgetsConfig) -> Self {
        Self {
            global: budgets.global.as_ref().map(ResolvedBudget::from_def),
            per_test_default: budgets
                .per_test_default
                .as_ref()
                .map(ResolvedBudget::from_def),
        }
    }

    /// Checks global budgets against the global usage snapshot.
    #[must_use]
    pub fn check_global(&self, usage: &UsageSnapshot) -> BudgetStatus {
        let Some(global) = &self.global else {
            return BudgetStatus::Ok;
        };
        Self::check_budget("global", global, usage)
    }

    /// Checks per-test budgets against a test's usage snapshot.
    ///
    /// `override_budget` allows the test to specify tighter (or looser)
    /// limits.
    #[must_use]
    pub fn check_per_test(
        &self,
        test_name: &str,
        usage: &UsageSnapshot,
        override_budget: Option<&BudgetDef>,
    ) -> BudgetStatus {
        let budget = match override_budget {
            Some(def) => ResolvedBudget::from_def(def),
            None => match &self.per_test_default {
                Some(def) => def.clone(),
                None => return BudgetStatus::Ok,
            },
        };
        Self::check_budget(test_name, &budget, usage)
    }

    /// Estimates the cost of a pending LLM call to check if it would exceed
    /// the budget.
    ///
    /// Uses `max_tokens` from the request as a worst-case estimate for
    /// output tokens, plus estimated input tokens.
    #[must_use]
    #[allow(dead_code)]
    #[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    pub(crate) fn check_pre_flight_llm(
        budget: &ResolvedBudget,
        usage: &UsageSnapshot,
        estimated_input_tokens: u64,
        max_tokens: u64,
        input_price_per_1m: f64,
        output_price_per_1m: f64,
    ) -> BudgetStatus {
        let estimated_cost = (estimated_input_tokens as f64 / 1_000_000.0) * input_price_per_1m
            + (max_tokens as f64 / 1_000_000.0) * output_price_per_1m;
        let estimated_total_tokens = usage.total_tokens + estimated_input_tokens + max_tokens;

        Self::check_limits(
            budget,
            "pre-flight",
            usage,
            estimated_cost,
            estimated_total_tokens,
            1,
        )
    }

    /// Checks the budget for a flat-cost call (MCP, agent).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn check_pre_flight_flat(
        budget: &ResolvedBudget,
        usage: &UsageSnapshot,
        per_call_price: f64,
    ) -> BudgetStatus {
        Self::check_limits(budget, "pre-flight", usage, per_call_price, 0, 1)
    }

    fn check_budget(name: &str, budget: &ResolvedBudget, usage: &UsageSnapshot) -> BudgetStatus {
        Self::check_limits(budget, name, usage, usage.total_cost, usage.total_tokens, 0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn check_limits(
        budget: &ResolvedBudget,
        name: &str,
        usage: &UsageSnapshot,
        additional_cost: f64,
        additional_tokens: u64,
        additional_calls: u64,
    ) -> BudgetStatus {
        let projected_cost = usage.total_cost + additional_cost;
        let projected_tokens = usage.total_tokens + additional_tokens;
        let projected_calls = usage.total_calls + additional_calls;

        let exceeded = |limit_name: &str, current: f64, limit: f64| -> Option<BudgetStatus> {
            if current > limit {
                let msg =
                    format!("{limit_name} budget exceeded for '{name}': {current:.6} > {limit:.6}");
                Some(match budget.enforcement {
                    BudgetEnforcement::Hard => BudgetStatus::HardExceeded {
                        budget: name.to_owned(),
                        message: msg,
                    },
                    BudgetEnforcement::Soft => BudgetStatus::SoftExceeded {
                        budget: name.to_owned(),
                        message: msg,
                    },
                })
            } else {
                None
            }
        };

        if let Some(max) = budget.max_cost {
            if let Some(status) = exceeded("Cost", projected_cost, max) {
                return status;
            }
        }
        if let Some(max) = budget.max_tokens {
            if let Some(status) = exceeded("Token", projected_tokens as f64, max as f64) {
                return status;
            }
        }
        if let Some(max) = budget.max_calls {
            if let Some(status) = exceeded("Call", projected_calls as f64, max as f64) {
                return status;
            }
        }

        BudgetStatus::Ok
    }

    /// Checks both per-test and global budgets. Returns the most severe
    /// violation.
    #[must_use]
    pub fn check_all(
        &self,
        test_name: &str,
        test_usage: &UsageSnapshot,
        global_usage: &UsageSnapshot,
        test_budget_override: Option<&BudgetDef>,
    ) -> BudgetStatus {
        let per_test = self.check_per_test(test_name, test_usage, test_budget_override);
        if matches!(per_test, BudgetStatus::HardExceeded { .. }) {
            return per_test;
        }
        let global = self.check_global(global_usage);
        if matches!(global, BudgetStatus::HardExceeded { .. }) {
            return global;
        }
        if per_test != BudgetStatus::Ok {
            return per_test;
        }
        global
    }
}
