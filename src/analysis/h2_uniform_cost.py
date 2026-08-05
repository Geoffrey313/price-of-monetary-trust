"""Referee comment M9, part 1: H2 augmentation advantage at a uniform 40 bp cost.

Inputs: the same derived/reconstructed inputs as ``analysis.run_full_sample``
and the published ``h2_per_market.csv`` used for reconciliation.
Outputs: ``results/full-sample/h2_uniform_cost.csv``.
Purpose: rerun the exact published H2 computation with the energy pocket
charged the uniform 40 bp one-way cost instead of the published 50 bp, and
report the per-market augmentation advantage under both costs.

``run_full_sample.portfolio`` exposes ``energy_cost`` as a keyword (default
0.005), but ``run_full_sample.run_h2_market`` does not forward it.  To keep
``run_full_sample.py`` byte-identical, the 40 bp arm calls the unmodified
``run_h2_market`` while ``run_full_sample.portfolio`` is temporarily replaced
by a thin wrapper that injects ``energy_cost=0.004`` and delegates every other
argument to the original function.  The 50 bp arm runs the published code path
untouched and is reconciled against the published CSV to 1e-9 before the
alternative cost is trusted.

Run: ``FOUR_QUADRANT_FROM_DERIVED=1 PYTHONPATH=src python -m analysis.h2_uniform_cost``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

from analysis import run_full_sample as F
from common.names import MARKETS, NAMES
from common.paths import FULL_SAMPLE_RESULTS

TOLERANCE = 1e-9
UNIFORM_ENERGY_COST = 0.004


@contextmanager
def energy_cost_override(energy_cost: float):
    """Temporarily force ``portfolio``'s energy cost inside ``run_h2_market``."""
    original = F.portfolio

    def wrapped(returns, weight_fn, rebalance="M", cost=0.004, **_ignored):
        return original(
            returns, weight_fn, rebalance=rebalance, cost=cost, energy_cost=energy_cost
        )

    F.portfolio = wrapped
    try:
        yield
    finally:
        F.portfolio = original


def main() -> int:
    F.wire_everything()
    published = pd.read_csv(FULL_SAMPLE_RESULTS / "h2_per_market.csv").set_index(
        "market"
    )
    keys = set() if os.environ.get("FOUR_QUADRANT_FROM_DERIVED") == "1" else F.energy_keys()

    rows = []
    for mkt in MARKETS:
        print(f"[h2 uniform cost] {mkt}", flush=True)
        energy, _coverage = F.market_energy(mkt, keys)

        result_50 = F.run_h2_market(mkt, energy)
        advantage_50 = (
            result_50["sharpe_currency_energy"] - result_50["sharpe_currency_only"]
        )
        reference = float(published.loc[mkt, "augmentation_advantage"])
        gap = abs(advantage_50 - reference)
        if gap > TOLERANCE:
            raise RuntimeError(
                f"{mkt}: 50 bp rerun does not reconcile with the published "
                f"h2_per_market.csv ({advantage_50!r} vs {reference!r}, gap {gap:.3e})"
            )

        with energy_cost_override(UNIFORM_ENERGY_COST):
            result_40 = F.run_h2_market(mkt, energy)
        advantage_40 = (
            result_40["sharpe_currency_energy"] - result_40["sharpe_currency_only"]
        )

        rows.append(
            {
                "market": mkt,
                "country": NAMES[mkt],
                "months": result_50["months"],
                "augmentation_advantage_50bp": advantage_50,
                "augmentation_advantage_uniform_40bp": advantage_40,
                "delta_40bp_minus_50bp": advantage_40 - advantage_50,
            }
        )

    table = pd.DataFrame(rows)
    numeric = [
        "augmentation_advantage_50bp",
        "augmentation_advantage_uniform_40bp",
        "delta_40bp_minus_50bp",
    ]
    for label, reducer in (("MEDIAN", "median"), ("MEAN", "mean")):
        rows.append(
            {
                "market": label,
                "country": label.lower() + " across 13 markets",
                "months": "",
                **{column: getattr(table[column], reducer)() for column in numeric},
            }
        )
    out = pd.DataFrame(rows)
    out_path = FULL_SAMPLE_RESULTS / "h2_uniform_cost.csv"
    out.to_csv(out_path, index=False)

    negative_50 = int((table["augmentation_advantage_50bp"] < 0).sum())
    negative_40 = int((table["augmentation_advantage_uniform_40bp"] < 0).sum())
    print(f"\nreconciled 13/13 markets against h2_per_market.csv to {TOLERANCE:g}")
    print(
        f"50 bp energy cost : {negative_50}/13 negative, "
        f"median {table['augmentation_advantage_50bp'].median():+.4f}"
    )
    print(
        f"uniform 40 bp cost: {negative_40}/13 negative, "
        f"median {table['augmentation_advantage_uniform_40bp'].median():+.4f}"
    )
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
