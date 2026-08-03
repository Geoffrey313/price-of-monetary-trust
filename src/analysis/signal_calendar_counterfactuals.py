"""Portfolio counterfactuals for the French Bonds-or-Gold paper.

This module changes neither the thirteen-market universe nor any headline H1,
H2, or H3 return.  It asks a narrower question: does the observed calendar of
the binary bond-gold signal add information relative to rules that retain the
same assets and cost model but remove that calendar?

Outputs
-------
results/complete-sample-rerun-2026-07-26/
    signal_counterfactual_rules.csv
    signal_circular_shift_summary.csv
    signal_circular_shift_draws.csv
    signal_regime_permutation_summary.csv
    signal_regime_permutation_draws.csv
    signal_counterfactual_audit.csv

The circular-shift test is exact over every possible displacement, including
the observed displacement zero.  It preserves the complete binary sequence,
state frequency, run lengths, and number of switches on a circle.

The regime-duration test is Monte Carlo.  It independently permutes the
observed run lengths within the bond state and within the gold state, while
retaining the observed alternating state order, state counts, number of runs,
and total sample length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import engine
from analysis import run_full_sample as rerun


MARKETS = rerun.MARKETS
NAMES = rerun.NAMES
OUT = rerun.OUT
COST_ONEWAY = engine.COST_ONEWAY
REGIME_DRAWS = 4999
SEED = 20260726
ASSET_ORDER = ("eq", "bond", "bill", "gold")


def portfolio_returns(
    returns: np.ndarray,
    targets: np.ndarray,
    rebalance: str = "M",
    cost_oneway: float = COST_ONEWAY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the engine's drift, rebalance, and one-way cost conventions."""
    if returns.shape != targets.shape:
        raise ValueError(
            f"returns {returns.shape} and targets {targets.shape} differ"
        )
    if rebalance not in {"M", "Q"}:
        raise ValueError(f"unsupported rebalance frequency: {rebalance}")
    n = len(returns)
    out = np.empty(n)
    turnover = np.empty(n)
    costs = np.empty(n)
    previous_weights: np.ndarray | None = None

    for position in range(n):
        target = targets[position]
        if previous_weights is None:
            weights = target
            traded = float(np.abs(weights).sum())
        else:
            drifted = previous_weights * (1 + returns[position - 1])
            drifted = drifted / drifted.sum()
            month = position_months[position].month
            unchanged = np.array_equal(
                targets[position], targets[position - 1]
            )
            if (
                rebalance == "Q"
                and month not in (3, 6, 9, 12)
                and unchanged
            ):
                weights = drifted
                traded = 0.0
            else:
                weights = target
                traded = float(np.abs(weights - drifted).sum())
        cost = cost_oneway * traded
        out[position] = float(weights @ returns[position] - cost)
        turnover[position] = traded
        costs[position] = cost
        previous_weights = weights
    return out, turnover, costs


# The engine's quarterly decision depends on the actual calendar month.  This
# variable is rebound for each market immediately before portfolio_returns.
position_months: pd.PeriodIndex


def targets_from_signal(signal: np.ndarray) -> np.ndarray:
    """Four paper weights from an already-lagged binary/fractional signal."""
    signal = np.asarray(signal, dtype=float)
    return np.column_stack(
        [
            np.full(len(signal), 0.25),
            0.25 + 0.25 * signal,
            np.full(len(signal), 0.25),
            0.25 * (1 - signal),
        ]
    )


def constant_targets(n: int, signal_fraction: float) -> np.ndarray:
    return targets_from_signal(np.full(n, signal_fraction))


def monthly_returns_for_signals(
    returns: np.ndarray,
    signal_paths: np.ndarray,
    cost_oneway: float = COST_ONEWAY,
) -> np.ndarray:
    """Vectorized copy of the monthly engine for many signal calendars."""
    signals = np.asarray(signal_paths, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != len(returns):
        raise ValueError(
            "signal_paths must have shape (calendars, return months)"
        )
    path_count, month_count = signals.shape
    out = np.empty((path_count, month_count))
    previous_weights: np.ndarray | None = None

    for position in range(month_count):
        signal = signals[:, position]
        weights = np.column_stack(
            [
                np.full(path_count, 0.25),
                0.25 + 0.25 * signal,
                np.full(path_count, 0.25),
                0.25 * (1 - signal),
            ]
        )
        if previous_weights is None:
            traded = np.abs(weights).sum(axis=1)
        else:
            drifted = previous_weights * (1 + returns[position - 1])
            drifted = drifted / drifted.sum(axis=1, keepdims=True)
            traded = np.abs(weights - drifted).sum(axis=1)
        out[:, position] = (
            np.einsum("ij,j->i", weights, returns[position])
            - cost_oneway * traded
        )
        previous_weights = weights
    return out


def annualized_sharpes(
    portfolio_returns_matrix: np.ndarray,
    bill: np.ndarray,
) -> np.ndarray:
    """Annualized excess-return Sharpe for each row of a return matrix."""
    excess = portfolio_returns_matrix - bill[None, :]
    return excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(12)


def annualized_metrics(
    returns: np.ndarray,
    bill: np.ndarray,
    turnover: np.ndarray,
    costs: np.ndarray,
) -> dict[str, float | int]:
    excess = returns - bill
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    return {
        "months": len(returns),
        "annual_return": float(
            wealth[-1] ** (12 / len(returns)) - 1
        ),
        "annual_volatility": float(excess.std(ddof=1) * np.sqrt(12)),
        "sharpe": float(
            excess.mean() / excess.std(ddof=1) * np.sqrt(12)
        ),
        "max_drawdown": float((wealth / peak - 1).min()),
        "terminal_wealth": float(wealth[-1]),
        "annual_turnover": float(turnover.sum() * 12 / len(returns)),
        "cumulative_cost": float(costs.sum()),
    }


def run_lengths(signal: np.ndarray) -> list[tuple[int, int]]:
    """Return the observed alternating binary runs as (state, length)."""
    states = np.asarray(signal, dtype=int)
    boundaries = np.r_[0, np.flatnonzero(np.diff(states) != 0) + 1, len(states)]
    return [
        (int(states[start]), int(end - start))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def permute_run_durations(
    runs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute run lengths within state, preserving every conditioning total."""
    lengths = {
        state: rng.permutation(
            [length for run_state, length in runs if run_state == state]
        ).tolist()
        for state in (0, 1)
    }
    cursor = {0: 0, 1: 0}
    pieces: list[np.ndarray] = []
    for state, _ in runs:
        length = lengths[state][cursor[state]]
        cursor[state] += 1
        pieces.append(np.full(length, state, dtype=float))
    return np.concatenate(pieces)


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    sorted_values = values.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * sorted_values
    )
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return pd.Series(adjusted, index=values.index)


def main() -> int:
    global position_months

    OUT.mkdir(parents=True, exist_ok=True)
    rerun.wire_everything()
    engine.run_market.return_series = True
    rng = np.random.default_rng(SEED)

    rule_rows: list[dict[str, object]] = []
    circular_rows: list[dict[str, object]] = []
    circular_draw_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    regime_draw_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for market in MARKETS:
        result = engine.run_market(
            market,
            real_gate=False,
            bootstrap_draws=0,
        )
        series = result["series"]
        index = series["strat"].index
        position_months = pd.PeriodIndex(index, freq="M")
        frame = pd.DataFrame(
            {asset: series[asset].reindex(index) for asset in ASSET_ORDER}
        )
        returns = frame.to_numpy(float)
        bill = frame["bill"].to_numpy(float)
        observed_signal = (
            series["signal"].reindex(index).fillna(1.0).to_numpy(float)
        )

        observed, observed_turnover, observed_costs = portfolio_returns(
            returns, targets_from_signal(observed_signal), rebalance="M"
        )
        pp_rebuilt, pp_turnover, pp_costs = portfolio_returns(
            returns, constant_targets(len(index), 0.0), rebalance="Q"
        )
        matched_share = float(result["matched_bond_state_share"])
        matched_rebuilt, matched_turnover, matched_costs = portfolio_returns(
            returns,
            constant_targets(len(index), matched_share),
            rebalance="Q",
        )
        always_bonds = portfolio_returns(
            returns,
            constant_targets(len(index), 1.0),
            rebalance="M",
        )
        always_gold = portfolio_returns(
            returns,
            constant_targets(len(index), 0.0),
            rebalance="M",
        )
        fixed_half = portfolio_returns(
            returns,
            constant_targets(len(index), 0.5),
            rebalance="M",
        )

        observed_error = float(
            np.max(np.abs(observed - series["strat"].to_numpy(float)))
        )
        observed_batch_error = float(
            np.max(
                np.abs(
                    monthly_returns_for_signals(
                        returns, observed_signal[None, :]
                    )[0]
                    - observed
                )
            )
        )
        pp_error = float(
            np.max(np.abs(pp_rebuilt - series["pp"].to_numpy(float)))
        )
        matched_error = float(
            np.max(np.abs(matched_rebuilt - series["matched"].to_numpy(float)))
        )
        always_gold_permanent_error = float(
            np.max(np.abs(always_gold[0] - pp_rebuilt))
        )
        if max(
            observed_error,
            observed_batch_error,
            pp_error,
            matched_error,
        ) > 1e-12:
            raise RuntimeError(
                f"{market}: portfolio replication failed "
                f"({observed_error=}, {observed_batch_error=}, "
                f"{pp_error=}, {matched_error=})"
            )

        audit_rows.append(
            {
                "market": market,
                "country": NAMES[market],
                "signal_window_months": engine.WINDOW,
                "signal_lag_months": 1,
                "cost_oneway": COST_ONEWAY,
                "observed_max_abs_return_error": observed_error,
                "observed_batch_max_abs_return_error": observed_batch_error,
                "permanent_max_abs_return_error": pp_error,
                "matched_max_abs_return_error": matched_error,
                "always_gold_permanent_max_abs_return_difference": (
                    always_gold_permanent_error
                ),
                "always_gold_equals_permanent": (
                    always_gold_permanent_error <= 1e-12
                ),
                "reason_not_equal": (
                    "always-gold counterfactual rebalances monthly; "
                    "permanent benchmark rebalances quarterly"
                ),
            }
        )

        rules = {
            "observed_switch_monthly": (
                observed,
                observed_turnover,
                observed_costs,
                observed_signal,
                "monthly",
            ),
            "always_bonds_monthly": (
                *always_bonds,
                np.ones(len(index)),
                "monthly",
            ),
            "always_gold_monthly": (
                *always_gold,
                np.zeros(len(index)),
                "monthly",
            ),
            "fixed_half_monthly": (
                *fixed_half,
                np.full(len(index), 0.5),
                "monthly",
            ),
            "permanent_quarterly": (
                pp_rebuilt,
                pp_turnover,
                pp_costs,
                np.zeros(len(index)),
                "quarterly",
            ),
            "matched_static_quarterly": (
                matched_rebuilt,
                matched_turnover,
                matched_costs,
                np.full(len(index), matched_share),
                "quarterly",
            ),
        }
        for rule, (rule_returns, turnover, costs, signal, rebalance) in rules.items():
            rule_rows.append(
                {
                    "market": market,
                    "country": NAMES[market],
                    "rule": rule,
                    "start": str(index.min()),
                    "end": str(index.max()),
                    "rebalance": rebalance,
                    "cost_oneway": COST_ONEWAY,
                    "mean_bond_state_fraction": float(np.mean(signal)),
                    "state_changes": int(
                        np.count_nonzero(np.diff(signal) != 0)
                    ),
                    **annualized_metrics(
                        rule_returns, bill, turnover, costs
                    ),
                }
            )

        observed_sharpe = annualized_metrics(
            observed, bill, observed_turnover, observed_costs
        )["sharpe"]
        shifted_signals = np.vstack(
            [
                np.roll(observed_signal, shift)
                for shift in range(len(index))
            ]
        )
        shifted_returns = monthly_returns_for_signals(
            returns, shifted_signals
        )
        shifted_sharpes = annualized_sharpes(shifted_returns, bill)
        for shift, shifted_sharpe in enumerate(shifted_sharpes):
            circular_draw_rows.append(
                {
                    "market": market,
                    "shift_months": shift,
                    "is_observed_calendar": shift == 0,
                    "sharpe": shifted_sharpe,
                    "difference_from_observed": shifted_sharpe - observed_sharpe,
                }
            )
        circular_p = float(
            np.count_nonzero(shifted_sharpes >= observed_sharpe)
            / len(shifted_sharpes)
        )
        circular_rows.append(
            {
                "market": market,
                "country": NAMES[market],
                "calendar_count_including_observed": len(shifted_sharpes),
                "cost_oneway": COST_ONEWAY,
                "observed_sharpe": observed_sharpe,
                "shifted_mean_sharpe": float(shifted_sharpes.mean()),
                "shifted_median_sharpe": float(np.median(shifted_sharpes)),
                "shifted_p05_sharpe": float(
                    np.quantile(shifted_sharpes, 0.05)
                ),
                "shifted_p95_sharpe": float(
                    np.quantile(shifted_sharpes, 0.95)
                ),
                "calendars_at_least_observed": int(
                    np.count_nonzero(shifted_sharpes >= observed_sharpe)
                ),
                "p_exact_observed_not_superior": circular_p,
                "observed_percentile": float(
                    np.count_nonzero(shifted_sharpes <= observed_sharpe)
                    / len(shifted_sharpes)
                ),
            }
        )

        runs = run_lengths(observed_signal)
        regime_signals = np.vstack(
            [
                permute_run_durations(runs, rng)
                for _ in range(REGIME_DRAWS)
            ]
        )
        regime_returns = monthly_returns_for_signals(
            returns, regime_signals
        )
        regime_sharpes = annualized_sharpes(regime_returns, bill)
        for draw, permuted_sharpe in enumerate(regime_sharpes):
            regime_draw_rows.append(
                {
                    "market": market,
                    "draw": draw + 1,
                    "sharpe": permuted_sharpe,
                    "difference_from_observed": permuted_sharpe - observed_sharpe,
                }
            )
        regime_p = float(
            (
                1
                + np.count_nonzero(regime_sharpes >= observed_sharpe)
            )
            / (REGIME_DRAWS + 1)
        )
        regime_rows.append(
            {
                "market": market,
                "country": NAMES[market],
                "draws": REGIME_DRAWS,
                "random_seed": SEED,
                "cost_oneway": COST_ONEWAY,
                "runs": len(runs),
                "state_changes": len(runs) - 1,
                "observed_sharpe": observed_sharpe,
                "permuted_mean_sharpe": float(regime_sharpes.mean()),
                "permuted_median_sharpe": float(
                    np.median(regime_sharpes)
                ),
                "permuted_p05_sharpe": float(
                    np.quantile(regime_sharpes, 0.05)
                ),
                "permuted_p95_sharpe": float(
                    np.quantile(regime_sharpes, 0.95)
                ),
                "p_monte_carlo_observed_not_superior": regime_p,
                "observed_percentile": float(
                    np.count_nonzero(regime_sharpes <= observed_sharpe)
                    / REGIME_DRAWS
                ),
            }
        )

    rules_frame = pd.DataFrame(rule_rows)
    circular_frame = pd.DataFrame(circular_rows)
    circular_frame["p_holm_13"] = holm_adjust(
        circular_frame["p_exact_observed_not_superior"]
    )
    circular_frame["nominal_5pct"] = (
        circular_frame["p_exact_observed_not_superior"] < 0.05
    )
    circular_frame["holm_5pct"] = circular_frame["p_holm_13"] < 0.05
    regime_frame = pd.DataFrame(regime_rows)
    regime_frame["p_holm_13"] = holm_adjust(
        regime_frame["p_monte_carlo_observed_not_superior"]
    )
    regime_frame["nominal_5pct"] = (
        regime_frame["p_monte_carlo_observed_not_superior"] < 0.05
    )
    regime_frame["holm_5pct"] = regime_frame["p_holm_13"] < 0.05
    audit_frame = pd.DataFrame(audit_rows)

    rules_frame.to_csv(OUT / "signal_counterfactual_rules.csv", index=False)
    circular_frame.to_csv(
        OUT / "signal_circular_shift_summary.csv", index=False
    )
    pd.DataFrame(circular_draw_rows).to_csv(
        OUT / "signal_circular_shift_draws.csv", index=False
    )
    regime_frame.to_csv(
        OUT / "signal_regime_permutation_summary.csv", index=False
    )
    pd.DataFrame(regime_draw_rows).to_csv(
        OUT / "signal_regime_permutation_draws.csv", index=False
    )
    audit_frame.to_csv(OUT / "signal_counterfactual_audit.csv", index=False)

    print("Portfolio replication max errors:")
    print(
        audit_frame[
            [
                "observed_max_abs_return_error",
                "observed_batch_max_abs_return_error",
                "permanent_max_abs_return_error",
                "matched_max_abs_return_error",
            ]
        ].max()
    )
    print("\nCircular shifts:")
    print(
        circular_frame[
            [
                "market",
                "observed_sharpe",
                "shifted_median_sharpe",
                "p_exact_observed_not_superior",
                "p_holm_13",
            ]
        ].to_string(index=False)
    )
    print("\nRegime-duration permutations:")
    print(
        regime_frame[
            [
                "market",
                "observed_sharpe",
                "permuted_median_sharpe",
                "p_monte_carlo_observed_not_superior",
                "p_holm_13",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
