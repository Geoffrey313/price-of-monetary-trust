"""Referee comment M9, part 2: a graded-mapping sensitivity family for H3.

Inputs: the same in-memory market series as ``analysis.h3_variant`` and the
published ``h3_per_market.csv`` used for the sanity reconciliation.
Outputs: ``results/full-sample/h3_mapping_family.csv``.
Purpose: stress-test the paper's H3 reading (the information is in the sign of
the monetary signal, not its amplitude) by running four alternative graded
mappings from the frozen amplitude score z2 to a bond fraction in [0, 1].

The engine pieces are reused from ``analysis.h3_variant`` (``series_for``,
``run_variant``, ``lw_test``); that module returns the published mapping's
fraction but not z2 itself, so the frozen z2 definition (expanding z-score of
the ratio's deviation from its 84-month mean, defined only once 84 deviations
exist) is recomputed here.  A sanity gate reruns the PUBLISHED mapping
frac = clip(0.5 + z2/4, 0, 1) through this harness and requires per-market
agreement with ``h3_per_market.csv`` to 1e-9 before any alternative mapping is
evaluated.  Months before z2 exists always fall back to the binary signal,
exactly as in ``h3_variant``.

Mappings:
  published : frac = clip(0.5 + z2/4, 0, 1)              (sanity only)
  steep     : frac = clip(0.5 + z2/2, 0, 1)
  shallow   : frac = clip(0.5 + z2/8, 0, 1)
  rank      : frac = expanding percentile in (0, 1] of the deviation among its
              own past values, same 84-month availability rule as z2
  band      : frac = 1 if z2 > +0.25, 0 if z2 < -0.25, previous month's frac
              inside the band (seeded by the binary signal)

The pooled difference per mapping reuses the published pooling machinery of
``analysis.run_full_sample.run_h3``: excess-return series are stacked across
markets and passed to the Ledoit-Wolf test against a zero bill.

Run: ``FOUR_QUADRANT_FROM_DERIVED=1 PYTHONPATH=src python -m analysis.h3_mapping_family``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.h3_variant import (
    lw_test,
    pooled_calendar_hac,
    run_variant,
    series_for,
)
from analysis.run_full_sample import wire_everything
from common.names import MARKETS
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

TOLERANCE = 1e-9
BAND_HALF_WIDTH = 0.25


def amplitude_ingredients(mkt: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (d, z2, binary signal) under the frozen H3' definition.

    Mirrors ``h3_variant.series_for`` up to the mapping step, because that
    function returns the mapped fraction only.  The sanity gate in ``main``
    verifies this recomputation against the published output to 1e-9.
    """
    y = E.long_yield(mkt)
    bond = E.construct_monthly_tr(y.to_timestamp().rename("y"))
    bond_w = pd.Series(bond["wealth"].values, index=y.index)
    gold = E.gold_local(mkt)
    ratio = (bond_w / gold).dropna()
    ma = ratio.rolling(E.WINDOW).mean()
    cnt = ratio.rolling(E.WINDOW).count()
    d = (ratio - ma)[cnt >= E.WINDOW].dropna()
    mu = d.expanding().mean()
    sd = d.expanding().std()
    z2 = (d - mu) / sd
    z2[d.expanding().count() < E.WINDOW] = np.nan
    sig_bin = (d >= 0).astype(float)
    return d, z2, sig_bin


def rank_fraction(d: pd.Series) -> pd.Series:
    """Expanding percentile in (0, 1] of d among its own history, 84m rule."""
    frac = d.expanding().rank(method="average") / d.expanding().count()
    frac[d.expanding().count() < E.WINDOW] = np.nan
    return frac


def band_fraction(z2: pd.Series, sig_bin: pd.Series) -> pd.Series:
    """Three-state mapping with a neutral band that persists the prior state."""
    values = {}
    previous = None
    for t in z2.index:
        z = z2.loc[t]
        if np.isnan(z):
            value = float(sig_bin.loc[t])
        elif z > BAND_HALF_WIDTH:
            value = 1.0
        elif z < -BAND_HALF_WIDTH:
            value = 0.0
        else:
            value = previous if previous is not None else float(sig_bin.loc[t])
        values[t] = value
        previous = value
    return pd.Series(values)


def mapping_family(
    d: pd.Series, z2: pd.Series, sig_bin: pd.Series
) -> dict[str, pd.Series]:
    """All graded fractions on d's index, binary-filled before z2 exists."""
    family = {
        "published": (0.5 + z2 / 4).clip(0, 1),
        "steep": (0.5 + z2 / 2).clip(0, 1),
        "shallow": (0.5 + z2 / 8).clip(0, 1),
        "rank": rank_fraction(d),
        "band": band_fraction(z2, sig_bin),
    }
    return {name: frac.fillna(sig_bin) for name, frac in family.items()}


def net_sharpe(x: pd.Series, bill: pd.Series) -> float:
    e = (x - bill).dropna()
    return float(e.mean() / e.std() * np.sqrt(12))


def main() -> int:
    wire_everything()
    published = pd.read_csv(FULL_SAMPLE_RESULTS / "h3_per_market.csv").set_index(
        "market"
    )

    mapping_names = ["published", "steep", "shallow", "rank", "band"]
    rows = []
    pooled_variant: dict[str, list[pd.Series]] = {m: [] for m in mapping_names}
    pooled_binary: list[pd.Series] = []

    for mkt in MARKETS:
        print(f"[h3 mapping family] {mkt}", flush=True)
        returns, sig_bin_common, frac_published = series_for(mkt)
        d, z2, sig_bin = amplitude_ingredients(mkt)
        family = {
            name: frac.reindex(returns.index)
            for name, frac in mapping_family(d, z2, sig_bin).items()
        }
        if not np.allclose(
            family["published"], frac_published, atol=TOLERANCE, equal_nan=True
        ):
            raise RuntimeError(f"{mkt}: recomputed published fraction drifts from series_for")

        binary = run_variant(returns, sig_bin_common)
        binary_excess = (binary - returns["bill"]).dropna()
        pooled_binary.append(binary_excess)
        sharpe_binary = net_sharpe(binary, returns["bill"])

        for name in mapping_names:
            variant = run_variant(returns, family[name])
            diff, p_value, n = lw_test(variant, binary, returns["bill"])
            variant_excess = (variant - returns["bill"]).reindex(binary_excess.index)
            pooled_variant[name].append(variant_excess)
            rows.append(
                {
                    "mapping": name,
                    "market": mkt,
                    "months": n,
                    "sharpe_variant": net_sharpe(variant, returns["bill"]),
                    "sharpe_binary": sharpe_binary,
                    "diff": diff,
                    "p_variant_greater": p_value,
                }
            )

        gap_diff = abs(
            rows[-len(mapping_names)]["diff"]
            - float(published.loc[mkt, "difference_amplitude_minus_binary"])
        )
        gap_sharpe = abs(
            rows[-len(mapping_names)]["sharpe_variant"]
            - float(published.loc[mkt, "sharpe_amplitude"])
        )
        gap_binary = abs(
            sharpe_binary - float(published.loc[mkt, "sharpe_binary"])
        )
        if max(gap_diff, gap_sharpe, gap_binary) > TOLERANCE:
            raise RuntimeError(
                f"{mkt}: published-mapping sanity reconciliation failed "
                f"(diff gap {gap_diff:.3e}, sharpe gap {gap_sharpe:.3e}, "
                f"binary gap {gap_binary:.3e})"
            )

    print("sanity: published mapping reconciled 13/13 against h3_per_market.csv")

    table = pd.DataFrame(rows)
    summaries = []
    for name in mapping_names:
        sub = table[table["mapping"] == name]
        amp_parts, bin_parts, month_parts = [], [], []
        for binary_excess, variant_excess in zip(
            pooled_binary, pooled_variant[name]
        ):
            aligned = pd.DataFrame(
                {"amp": variant_excess, "bin": binary_excess}
            ).dropna()
            amp_parts.append(aligned["amp"].to_numpy(float))
            bin_parts.append(aligned["bin"].to_numpy(float))
            month_parts.append(aligned.index.astype(str).to_numpy())
        pooled_diff, _, _, pooled_p, _, _, pooled_n, _ = pooled_calendar_hac(
            np.concatenate(amp_parts),
            np.concatenate(bin_parts),
            np.concatenate(month_parts),
            lags=12,
        )
        summaries.append(
            {
                "mapping": name,
                "market": "SUMMARY",
                "months": pooled_n,
                "sharpe_variant": np.nan,
                "sharpe_binary": np.nan,
                "diff": np.nan,
                "p_variant_greater": np.nan,
                "markets_variant_wins": int((sub["diff"] > 0).sum()),
                "median_diff": float(sub["diff"].median()),
                "mean_diff": float(sub["diff"].mean()),
                "pooled_diff_calendar_months": pooled_diff,
                "pooled_p_variant_greater": pooled_p,
            }
        )

    out = pd.concat([table, pd.DataFrame(summaries)], ignore_index=True)
    out_path = FULL_SAMPLE_RESULTS / "h3_mapping_family.csv"
    out.to_csv(out_path, index=False)

    print("\nmapping    wins/13  median_diff  mean_diff  pooled_diff  pooled_p")
    for s in summaries:
        print(
            f"{s['mapping']:<9} {s['markets_variant_wins']:>6}   "
            f"{s['median_diff']:+.4f}      {s['mean_diff']:+.4f}    "
            f"{s['pooled_diff_calendar_months']:+.4f}      "
            f"{s['pooled_p_variant_greater']:.3f}"
        )
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
