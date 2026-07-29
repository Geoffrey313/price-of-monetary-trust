"""Re-evaluate the US H1 advantage outside the administered-gold-parity era.

Inputs: the in-memory USA monthly series from the H1 engine
(``engine.engine.run_market`` with the derived data layer wired by
``analysis.run_full_sample.wire_everything``).
Outputs: ``results/complete-sample-rerun-2026-07-26/
h1_us_gold_parity_robustness.csv`` and a printed table.
Purpose: referee point M5(ii).  The paper's US evaluation window starts in
1960:4, inside the Bretton Woods era when the dollar gold parity was
administered rather than market-determined.  This module re-evaluates the US
H1 advantage on two restricted windows that exclude that era: months from
1972-01 (after the August 1971 suspension of convertibility) and months from
1975-01 (after US private gold ownership was re-legalised and gold traded
freely).  All four series (switch strategy, 60/40, permanent portfolio, bill)
are sliced to the window, net Sharpe ratios are recomputed on the window, the
binding benchmark is re-selected as max(SR_6040, SR_permanent) on the
restricted window, and the advantage is SR_switch minus that maximum.  The
full-window advantage computed identically is reported for comparison, along
with each window's month count and its bond-state share (the mean of the
applied lagged signal over the window).
"""
from __future__ import annotations

import pandas as pd

from analysis import run_full_sample as R
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

WINDOWS: tuple[tuple[str, str | None], ...] = (
    ("full", None),
    (">=1972-01", "1972-01"),
    (">=1975-01", "1975-01"),
)


def evaluate_window(series: dict[str, pd.Series], start: str | None) -> dict:
    """Recompute the H1 comparison on the months at or after ``start``."""
    if start is None:
        window = dict(series)
    else:
        cutoff = pd.Period(start, freq="M")
        window = {key: value[value.index >= cutoff] for key, value in series.items()}
    bill = window["bill"]
    sharpe_strat = R.sharpe(window["strat"], bill)
    sharpe_6040 = R.sharpe(window["b6040"], bill)
    sharpe_pp = R.sharpe(window["pp"], bill)
    binding = "60/40" if sharpe_6040 >= sharpe_pp else "permanent"
    # The applied signal treats the pre-warm-up NaN months as the bond state,
    # exactly as the engine does when it allocates.
    bond_state_share = float(window["signal"].fillna(1.0).mean())
    return {
        "months": int(len(window["strat"])),
        "sharpe_strat": sharpe_strat,
        "sharpe_6040": sharpe_6040,
        "sharpe_pp": sharpe_pp,
        "binding": binding,
        "advantage": sharpe_strat - max(sharpe_6040, sharpe_pp),
        "bond_state_share": bond_state_share,
    }


def main() -> int:
    R.wire_everything()
    E.run_market.return_series = True
    result = E.run_market("USA", real_gate=False, bootstrap_draws=0)
    series = result["series"]

    rows = [
        {"window": label, **evaluate_window(series, start)}
        for label, start in WINDOWS
    ]
    table = pd.DataFrame(rows)
    out = FULL_SAMPLE_RESULTS / "h1_us_gold_parity_robustness.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
