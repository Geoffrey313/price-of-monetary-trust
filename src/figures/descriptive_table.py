"""Build the per-market descriptive asset-return table.

Inputs: the frozen engine's equity, yield, gold, and bill series.
Outputs: a LaTeX-ready table block on standard output.
Purpose: report the sample spans and annualized moments underlying H1--H3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import rdd_1999 as R_
from common.names import ENGLISH_NAMES as NAMES
from data.treasury_total_return import construct_monthly_tr
from engine import engine as E


def ann(r):
    return float(r.mean() * 12 * 100), float(r.std() * np.sqrt(12) * 100)


def main() -> int:
    from analysis.run_full_sample import wire_everything

    wire_everything()
    rows = []
    for m in R_.GROUP:
        try:
            eq = E.equity_tr(m)
            y = E.long_yield(m)
            bond = construct_monthly_tr(y.to_timestamp().rename("y"))
            bond_tr = pd.Series(bond["tr"].values, index=y.index)
            bond_w = pd.Series(bond["wealth"].values, index=y.index)
            gold = E.gold_local(m).pct_change()
            bill = E.short_rate(m) / 100 / 12
            ratio = (bond_w / E.gold_local(m)).dropna()
            sig = (ratio >= ratio.rolling(E.WINDOW).mean()).astype(int)
            sig = sig[ratio.rolling(E.WINDOW).count() >= E.WINDOW]
            common = eq.index.intersection(bond_tr.index).intersection(gold.index)\
                .intersection(bill.index).intersection(sig.index)
        except Exception as e:
            print(f"  [skip] {m}: {str(e)[:70]}"); continue
        eqm, eqs = ann(eq.reindex(common))
        bdm, bds = ann(bond_tr.reindex(common))
        gdm, gds = ann(gold.reindex(common))
        blm, _ = ann(bill.reindex(common))
        rows.append((NAMES[m], str(common.min()), str(common.max()), len(common),
                     eqm, eqs, bdm, bds, gdm, gds, blm))
    rows.sort(key=lambda r: r[0])
    print("\n%% Table 1 rows: market & start & end & N & eq mu & eq sd & bond mu & bond sd & gold mu & gold sd & bill mu")
    for r in rows:
        print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]:.1f} & {r[5]:.1f} & "
              f"{r[6]:.1f} & {r[7]:.1f} & {r[8]:.1f} & {r[9]:.1f} & {r[10]:.1f} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
