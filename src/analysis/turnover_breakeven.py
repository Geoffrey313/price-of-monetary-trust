"""Implementation economics: realized turnover and per-market breakeven costs.

Inputs: the wired H1 engine (``engine.engine``) and the H3 graded-variant
builders (``analysis.h3_variant``), on the configured snapshots.
Outputs: ``results/full-sample/turnover_breakeven.csv``,
one row per market plus a cross-market median row.
Purpose: referee comment M11. (1) Annualized one-way turnover, defined as the
mean monthly traded weight sum(|w - w_drift|) times 12, in percent, for the
binary switch, the graded H3 variant, the local 60/40, and the permanent
portfolio. The initial position-establishment month (traded weight 1.0 by
construction) is excluded from the mean; it is a one-off, not recurring
turnover. (2) The one-way cost at which the H1 advantage over the harder
benchmark crosses zero, interpolated linearly on the published cost grid
{0, 20, 40, 60, 80, 100 bp} rerun per market with the same machinery as
``analysis.run_full_sample.run_sensitivity``.

The traded-weight bookkeeping replicates the engine's own cost leg
(``engine.engine.run_market``'s inner ``portfolio``) and is reconciled per
market: the net return rebuilt as gross minus cost times traded weight must
match the engine's published net series to 1e-12 (and ``h3_variant.run_variant``
for the graded arm). Deterministic: no bootstrap draws, no random state.
Run: ``FOUR_QUADRANT_FROM_DERIVED=1 PYTHONPATH=src python -m analysis.turnover_breakeven``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.h3_variant import run_variant, series_for
from analysis.run_full_sample import wire_everything
from common.names import MARKETS
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

COST_GRID = (0.0, 0.002, 0.004, 0.006, 0.008, 0.010)   # the published sensitivity grid
RECON_TOL = 1e-12


def simulate(
    R: pd.DataFrame,
    weight_fn,
    cost: float,
    rebalance: str = "M",
) -> tuple[pd.Series, pd.Series]:
    """Replicate the engine's portfolio loop, returning (net return, traded weight).

    Mirrors ``engine.engine.run_market``'s inner ``portfolio`` exactly: monthly
    drift of last month's weights by realized returns, renormalisation, then
    either a rebalance back to target (traded weight sum(|w - w_drift|), charged
    at ``cost`` one-way) or, under quarterly rebalancing with an unchanged
    target, a free hold of the drifted weights.
    """
    w_prev = None
    net: dict = {}
    traded: dict = {}
    for t in R.index:
        tgt = weight_fn(t)
        if w_prev is None:
            w = tgt
            traded[t] = sum(abs(x) for x in w.values())
        else:
            drift = {a: w_prev[a] * (1 + R.loc[t_prev, a]) for a in w_prev}
            tot = sum(drift.values())
            drift = {a: v / tot for a, v in drift.items()}
            if rebalance == "Q" and t.month not in (3, 6, 9, 12) \
                    and weight_fn(t) == weight_fn(t_prev):
                w = drift
                traded[t] = 0.0
            else:
                w = tgt
                traded[t] = sum(abs(w[a] - drift.get(a, 0)) for a in w)
        net[t] = sum(w[a] * R.loc[t, a] for a in w) - cost * traded[t]
        w_prev, t_prev = w, t
    return pd.Series(net), pd.Series(traded)


def graded_traded(R: pd.DataFrame, frac: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Traded weight of the graded H3 variant, mirroring ``h3_variant.run_variant``."""
    f = frac.shift(1).ffill()

    def weights(t):
        fv = f.loc[t]
        fv = 1.0 if np.isnan(fv) else float(fv)
        return {"eq": .25, "bond": .25 + .25 * fv, "bill": .25, "gold": .25 * (1 - fv)}

    return simulate(R[["eq", "bond", "gold", "bill"]], weights, E.COST_ONEWAY)


def annualized_turnover_pct(traded: pd.Series) -> float:
    """Mean monthly traded weight x 12, in percent, excluding the establishment month."""
    return float(traded.iloc[1:].mean() * 12 * 100)


def market_turnover(mkt: str) -> dict[str, float]:
    """Turnover of the four legs for one market, with 1e-12 net-return reconciliation."""
    E.run_market.return_series = True
    res = E.run_market(mkt, real_gate=False, bootstrap_draws=0)
    s = res["series"]
    cost = res["cost_oneway"]
    R = pd.DataFrame({"eq": s["eq"], "bond": s["bond"], "gold": s["gold"], "bill": s["bill"]})
    sigl = s["signal"].reindex(R.index)

    def wf_switch(t):                                  # engine's wf_h1, verbatim
        v = sigl.loc[t]
        state = int(v) if not np.isnan(v) else 1
        fourth = "bond" if state else "gold"
        w = {"eq": .25, "bond": .25, "bill": .25, "gold": 0.0}
        w[fourth] = w.get(fourth, 0) + .25
        return w

    checks = {
        "switch": (simulate(R, wf_switch, cost), s["strat"]),
        "6040": (simulate(R, lambda t: {"eq": .60, "bond": .40}, cost, "Q"), s["b6040"]),
        "pp": (
            simulate(R, lambda t: {"eq": .25, "bond": .25, "gold": .25, "bill": .25}, cost, "Q"),
            s["pp"],
        ),
    }
    Rg, _, frac = series_for(mkt)
    checks["graded"] = (graded_traded(Rg, frac), run_variant(Rg, frac))

    out = {}
    for leg, ((net, traded), published) in checks.items():
        gap = float((net - published).abs().max())
        if not gap <= RECON_TOL:
            raise RuntimeError(
                f"{mkt} {leg}: rebuilt net return misses the published series "
                f"by {gap:.3e} (> {RECON_TOL:.0e})"
            )
        out[leg] = annualized_turnover_pct(traded)
    return out


def breakeven_bp(advantages: list[float]) -> tuple[str, float]:
    """Zero crossing of the advantage on COST_GRID, as (label, censored value in bp).

    The censored value maps '>100' to +inf and 'never positive' to -inf so the
    cross-market median stays well defined.
    """
    grid_bp = [c * 1e4 for c in COST_GRID]
    if advantages[0] <= 0:
        return "<0 (never positive)", -np.inf
    if advantages[-1] > 0:
        return ">100", np.inf
    for i in range(len(advantages) - 1):
        a0, a1 = advantages[i], advantages[i + 1]
        if a0 > 0 >= a1:
            c = grid_bp[i] + (grid_bp[i + 1] - grid_bp[i]) * a0 / (a0 - a1)
            return f"{c:.1f}", c
    raise RuntimeError("no sign change located despite endpoint signs")     # unreachable


def market_advantage_grid(mkt: str) -> list[float]:
    """H1 advantage vs the harder benchmark at each grid cost (run_sensitivity's machinery)."""
    E.run_market.return_series = False
    return [
        E.run_market(mkt, real_gate=False, cost_oneway=c, bootstrap_draws=0)["advantage"]
        for c in COST_GRID
    ]


def main() -> int:
    wire_everything()
    rows = []
    grid_by_market = {}
    for mkt in MARKETS:
        print(f"[{mkt}]", flush=True)
        turn = market_turnover(mkt)
        adv = market_advantage_grid(mkt)
        grid_by_market[mkt] = adv
        label, censored = breakeven_bp(adv)
        rows.append(
            {
                "market": mkt,
                "turnover_switch_pct": round(turn["switch"], 2),
                "turnover_graded_pct": round(turn["graded"], 2),
                "turnover_6040_pct": round(turn["6040"], 2),
                "turnover_pp_pct": round(turn["pp"], 2),
                "breakeven_bp": label,
                "_breakeven_censored": censored,
            }
        )

    published = pd.read_csv(FULL_SAMPLE_RESULTS / "h1_cost_sensitivity.csv")
    for i, cost in enumerate(COST_GRID):
        med = float(np.median([grid_by_market[m][i] for m in MARKETS]))
        ref = float(published.loc[published["cost_oneway"] == cost, "median_advantage"].iloc[0])
        flag = "" if abs(med - ref) <= 1e-9 else "  MISMATCH vs published grid"
        print(f"  grid check cost={cost:.3f}: median advantage {med:.3f} "
              f"(published {ref:.3f}){flag}", flush=True)

    med_censored = float(np.median([r["_breakeven_censored"] for r in rows]))
    median_row = {
        "market": "MEDIAN",
        "turnover_switch_pct": round(float(np.median([r["turnover_switch_pct"] for r in rows])), 2),
        "turnover_graded_pct": round(float(np.median([r["turnover_graded_pct"] for r in rows])), 2),
        "turnover_6040_pct": round(float(np.median([r["turnover_6040_pct"] for r in rows])), 2),
        "turnover_pp_pct": round(float(np.median([r["turnover_pp_pct"] for r in rows])), 2),
        "breakeven_bp": ">100" if np.isinf(med_censored) else f"{med_censored:.1f}",
    }
    table = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_breakeven_censored"} for r in rows]
        + [median_row]
    )
    out = FULL_SAMPLE_RESULTS / "turnover_breakeven.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
