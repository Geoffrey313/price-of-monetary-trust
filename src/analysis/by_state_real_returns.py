"""Referee check M10, part 2: real bond and gold returns by governing state.

Inputs: the wired thirteen-market engine (pocket return series and CPI levels
from the derived layer).
Outputs: ``results/full-sample/by_state_real_returns.csv``.
Purpose: show directly whether the state variable sorts months in which bonds
beat gold in real terms from months in which gold beats bonds.  For every
market the engine's nominal bond and gold pocket returns
(``run_market(...)['series']['bond'|'gold']`` with ``real_gate=False``) are
deflated by monthly CPI inflation and split by the LAGGED state that governs
the portfolio in that month (the engine's applied signal series,
s_{t-1} = 1 bond state, 0 gold state; the engine's convention of treating a
missing first applied signal as the bond state is kept).

Deflation convention (identical for both assets): exact ratio deflation,
``real = (1 + nominal) / (1 + monthly CPI inflation) - 1``, the same
convention the engine uses for its real-return companion series [S7], with
monthly inflation ``E.cpi_index(market).pct_change()``.  Months without a CPI
observation are dropped.  The table reports the arithmetic mean monthly real
return annualised as ``mean * 12 * 100`` in each state, per market and pooled
over all market-months.
"""
from __future__ import annotations

import pandas as pd

from analysis import run_full_sample as R
from common.names import MARKETS
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

OUT = FULL_SAMPLE_RESULTS
OUT_FILE = OUT / "by_state_real_returns.csv"

STATE_LABELS = {1: "bond_state", 0: "gold_state"}


def market_panel(market: str) -> pd.DataFrame:
    """Real bond and gold returns with the governing state, one market."""
    series = E.run_market(market, real_gate=False, bootstrap_draws=0)["series"]
    state = series["signal"].fillna(1.0)  # engine treats a missing applied
    # signal as the bond state (run_market wf_h1)
    inflation = E.cpi_index(market).pct_change()
    frame = pd.DataFrame(
        {
            "real_bond": (1 + series["bond"])
            / (1 + inflation.reindex(series["bond"].index))
            - 1,
            "real_gold": (1 + series["gold"])
            / (1 + inflation.reindex(series["gold"].index))
            - 1,
            "state": state.astype(int),
        }
    ).dropna()
    return frame.assign(market=market)


def state_rows(name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    """Annualised mean real returns in each state for one (or all) markets."""
    rows: list[dict[str, object]] = []
    for value in (1, 0):
        block = frame[frame["state"] == value]
        rows.append(
            {
                "market": name,
                "state": STATE_LABELS[value],
                "months": len(block),
                "real_bond_ann_pct": float(block["real_bond"].mean() * 12 * 100),
                "real_gold_ann_pct": float(block["real_gold"].mean() * 12 * 100),
            }
        )
    return rows


def main() -> int:
    R.wire_everything()
    E.run_market.return_series = True

    print("[M10] real bond and gold returns by governing state", flush=True)
    panels: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for market in MARKETS:
        print(f"  {market}", flush=True)
        frame = market_panel(market)
        panels.append(frame)
        rows.extend(state_rows(market, frame))
    pooled = pd.concat(panels, ignore_index=True)
    rows.extend(state_rows("POOLED", pooled))

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_FILE, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"wrote {OUT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
