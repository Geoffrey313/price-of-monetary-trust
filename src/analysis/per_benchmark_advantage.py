"""Report the H1 advantage against each benchmark separately, per market.

Inputs: the in-memory monthly series from the H1 engine
(``engine.engine.run_market`` with the derived data layer wired by
``analysis.run_full_sample.wire_everything``) and the published
``results/complete-sample-rerun-2026-07-26/h1_per_market.csv`` used as a
cross-check.
Outputs: ``results/complete-sample-rerun-2026-07-26/
h1_per_benchmark_advantage.csv`` and a printed table.
Purpose: referee point M8.  The paper's headline advantage is measured
against the binding benchmark, max(SR_6040, SR_permanent), which conceals how
the strategy fares against each benchmark on its own.  For each of the
thirteen markets, on the unchanged full evaluation window, this module
reports the net Sharpe of the switch strategy, the 60/40 benchmark, and the
permanent portfolio, the advantage versus each benchmark separately, which
benchmark binds, and the binding advantage (the minimum of the two, which by
construction equals SR_switch minus the maximum benchmark Sharpe).  The
binding advantage is verified against the published per-market ``advantage``
column to 1e-9 before any output is written.  A final row reports
cross-market medians.
"""
from __future__ import annotations

import pandas as pd

from analysis import run_full_sample as R
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

TOLERANCE = 1e-9


def main() -> int:
    R.wire_everything()
    E.run_market.return_series = True

    rows = []
    for mkt in R.MARKETS:
        series = E.run_market(mkt, real_gate=False, bootstrap_draws=0)["series"]
        bill = series["bill"]
        sharpe_strat = R.sharpe(series["strat"], bill)
        sharpe_6040 = R.sharpe(series["b6040"], bill)
        sharpe_pp = R.sharpe(series["pp"], bill)
        adv_vs_6040 = sharpe_strat - sharpe_6040
        adv_vs_pp = sharpe_strat - sharpe_pp
        adv_binding = min(adv_vs_6040, adv_vs_pp)
        sanity = sharpe_strat - max(sharpe_6040, sharpe_pp)
        if abs(adv_binding - sanity) > TOLERANCE:
            print(
                f"{mkt}: min-of-advantages {adv_binding!r} differs from "
                f"SR_strat - max benchmark {sanity!r}"
            )
            return 1
        rows.append(
            {
                "market": mkt,
                "country": R.NAMES[mkt],
                "sharpe_strat": sharpe_strat,
                "sharpe_6040": sharpe_6040,
                "sharpe_pp": sharpe_pp,
                "adv_vs_6040": adv_vs_6040,
                "adv_vs_pp": adv_vs_pp,
                "binding": "60/40" if sharpe_6040 >= sharpe_pp else "permanent",
                "adv_binding": adv_binding,
            }
        )
    table = pd.DataFrame(rows)

    published = pd.read_csv(FULL_SAMPLE_RESULTS / "h1_per_market.csv")
    merged = table.merge(
        published[["market", "advantage"]], on="market", validate="one_to_one"
    )
    gap = (merged["adv_binding"] - merged["advantage"]).abs()
    if (gap > TOLERANCE).any():
        offending = merged.loc[gap > TOLERANCE, ["market", "adv_binding", "advantage"]]
        print("cross-check against published h1_per_market.csv FAILED:")
        print(offending.to_string(index=False))
        return 1

    table = table.sort_values("adv_binding", ascending=False).reset_index(drop=True)
    numeric = [
        "sharpe_strat",
        "sharpe_6040",
        "sharpe_pp",
        "adv_vs_6040",
        "adv_vs_pp",
        "adv_binding",
    ]
    median_row = {
        "market": "MEDIAN",
        "country": "médiane inter-marchés",
        "binding": "",
        **{column: float(table[column].median()) for column in numeric},
    }
    table = pd.concat([table, pd.DataFrame([median_row])], ignore_index=True)

    out = FULL_SAMPLE_RESULTS / "h1_per_benchmark_advantage.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\ncross-check against published advantages passed at {TOLERANCE:g}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
