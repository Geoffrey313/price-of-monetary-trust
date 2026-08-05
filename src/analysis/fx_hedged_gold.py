"""Quantify the FX content of the gold pocket and of the monetary signal (referee M7).

Inputs: the wired engine inputs (derived layer with ``FOUR_QUADRANT_FROM_DERIVED=1``)
and ``results/full-sample/h1_per_market.csv`` for the parity check.
Outputs: ``results/full-sample/h1_fx_hedged_gold.csv``.
Purpose: for the twelve non-US markets, gold in local currency is dollar gold times
the local exchange rate, so both the fourth-pocket P&L and the bond/gold signal embed
an implicit long-dollar position. This driver reruns H1 through the unmodified engine
under two variants:

  * ``hedged_pocket`` — the gold pocket earns the DOLLAR gold return (FX-hedged),
    while the signal stays on gold in LOCAL currency. Implemented by injecting
    dollar gold as the market's gold series and wrapping the bond total-return
    constructor so the wealth index is divided by the exchange rate: the ratio
    (W/FX)/G_usd equals W/G_local, the local signal, exactly. The wealth index
    feeds nothing but the signal ratio inside ``run_market``.
  * ``hedged_both`` — dollar gold both in the pocket and in the signal ratio.

Dollar gold is restricted to each market's local-gold coverage so the evaluation
sample is identical across variants. The US market is its own control: its local
currency is the dollar, so all three variants coincide. No engine module is
modified; the overrides are installed and restored around each run.

Run: ``FOUR_QUADRANT_FROM_DERIVED=1 PYTHONPATH=src python -m analysis.fx_hedged_gold``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import run_full_sample as RFS
from common.names import MARKETS, NAMES
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

OUT = FULL_SAMPLE_RESULTS
VARIANTS = ("baseline", "hedged_pocket", "hedged_both")
PARITY_TOLERANCE = 1e-9


def dollar_gold_and_fx(mkt: str) -> tuple[pd.Series, pd.Series]:
    """Dollar gold restricted to the market's local-gold coverage, and the FX level.

    FX is defined as gold_local / gold_usd (units of local currency per dollar up to
    the common gold quote), the exact implicit dollar leg of the local gold series.
    """
    local = E.gold_local(mkt)
    dollar = E.gold_local("USA").reindex(local.index)
    if dollar.isna().any():
        missing = dollar[dollar.isna()].index.tolist()
        raise ValueError(f"{mkt}: dollar gold missing at {missing[:5]}")
    return dollar, (local / dollar)


def run_variant(mkt: str, variant: str) -> dict:
    """Run H1 for one market under one variant through the unmodified engine."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    original_gold_local = E.gold_local
    original_construct = E.construct_monthly_tr
    try:
        if variant != "baseline" and mkt != "USA":
            dollar, fx = dollar_gold_and_fx(mkt)

            def hedged_gold_local(market: str, _mkt=mkt, _dollar=dollar):
                if market != _mkt:
                    raise RuntimeError(f"unexpected gold_local call for {market}")
                return _dollar

            E.gold_local = hedged_gold_local
            if variant == "hedged_pocket":
                # Divide the bond wealth index by FX so the signal ratio
                # (W/FX)/G_usd reproduces the local ratio W/G_local. Inside
                # run_market the wealth column feeds only the signal ratio.
                def local_signal_construct(
                    yields_pct_monthly: pd.Series,
                    include_convexity: bool | None = None,
                    _fx=fx,
                    _original=original_construct,
                ) -> pd.DataFrame:
                    out = _original(yields_pct_monthly, include_convexity).copy()
                    periods = pd.PeriodIndex(out.index, freq="M")
                    out["wealth"] = out["wealth"].to_numpy() / _fx.reindex(periods).to_numpy()
                    return out

                E.construct_monthly_tr = local_signal_construct
        return E.run_market(mkt, real_gate=False, bootstrap_draws=0)
    finally:
        E.gold_local = original_gold_local
        E.construct_monthly_tr = original_construct


def summarize(mkt: str, variant: str, result: dict, baseline: dict | None) -> dict:
    """One output row, with Sharpe ratios recomputed at full precision.

    Benchmarks are the paper's fixed local yardstick: the 60/40 holds no gold, and
    the permanent portfolio's 25 percent gold sleeve stays in LOCAL currency under
    every variant, so both benchmark Sharpes and the binding selection are taken
    from the baseline run. The permanent portfolio recomputed with hedged gold is
    reported separately as ``sharpe_pp_hedged`` for information only.
    """
    s = result["series"]
    bench = baseline["series"] if baseline is not None else s
    binding, best = RFS.harder_benchmark(bench)
    row = {
        "market": mkt,
        "country": NAMES[mkt],
        "variant": variant,
        "start": str(s["strat"].index.min()),
        "end": str(s["strat"].index.max()),
        "months": len(s["strat"]),
        "sharpe_strat": RFS.sharpe(s["strat"], s["bill"]),
        "sharpe_6040": RFS.sharpe(bench["b6040"], bench["bill"]),
        "sharpe_pp": RFS.sharpe(bench["pp"], bench["bill"]),
        "sharpe_pp_hedged": (
            RFS.sharpe(s["pp"], s["bill"]) if baseline is not None else np.nan
        ),
        "binding": binding,
        "advantage": RFS.sharpe(s["strat"], s["bill"]) - RFS.sharpe(best, bench["bill"]),
        "signal_diff_share": np.nan,
    }
    if baseline is not None:
        base = baseline["series"]
        if not s["strat"].index.equals(base["strat"].index):
            raise AssertionError(f"{mkt} {variant}: evaluation sample moved")
        if float((s["bill"] - base["bill"]).abs().max()) != 0.0:
            raise AssertionError(f"{mkt} {variant}: bill series moved")
        applied = s["signal"].reindex(s["strat"].index).fillna(1.0)
        applied_base = base["signal"].reindex(base["strat"].index).fillna(1.0)
        diff_share = float((applied != applied_base).mean())
        if variant == "hedged_pocket" and diff_share != 0.0:
            raise AssertionError(f"{mkt}: hedged_pocket signal deviates from local signal")
        if variant == "hedged_both":
            row["signal_diff_share"] = diff_share
    return row


def check_baseline_parity(rows: list[dict]) -> None:
    """Reconcile the unmodified rerun with the committed per-market H1 file."""
    reference = pd.read_csv(OUT / "h1_per_market.csv").set_index("market")
    for row in rows:
        ref = reference.loc[row["market"]]
        for ours, theirs in (
            ("advantage", "advantage"),
            ("sharpe_strat", "sharpe_switch"),
            ("sharpe_6040", "sharpe_6040"),
            ("sharpe_pp", "sharpe_permanent"),
        ):
            gap = abs(row[ours] - float(ref[theirs]))
            if gap > PARITY_TOLERANCE:
                raise AssertionError(
                    f"baseline parity FAILED for {row['market']} {ours}: "
                    f"{row[ours]!r} vs committed {float(ref[theirs])!r} (gap {gap:.3e})"
                )
        if int(row["months"]) != int(ref["months"]):
            raise AssertionError(f"baseline parity FAILED for {row['market']}: months differ")


def main() -> int:
    RFS.wire_everything()
    E.run_market.return_series = True

    baseline_results = {mkt: run_variant(mkt, "baseline") for mkt in MARKETS}
    baseline_rows = [summarize(mkt, "baseline", baseline_results[mkt], None) for mkt in MARKETS]
    check_baseline_parity(baseline_rows)
    print("baseline parity vs h1_per_market.csv: OK (13 markets, tol 1e-9)", flush=True)

    rows: list[dict] = []
    for mkt in MARKETS:
        rows.append(next(r for r in baseline_rows if r["market"] == mkt))
        for variant in ("hedged_pocket", "hedged_both"):
            result = run_variant(mkt, variant)
            rows.append(summarize(mkt, variant, result, baseline_results[mkt]))
        print(f"  {mkt} done", flush=True)

    table = pd.DataFrame(rows)
    for variant in VARIANTS:
        sub = table[table["variant"] == variant]
        for label, keep in (("MEDIAN_13", sub), ("MEDIAN_12_EX_USA", sub[sub["market"] != "USA"])):
            rows.append(
                {
                    "market": label,
                    "country": "",
                    "variant": variant,
                    "start": "",
                    "end": "",
                    "months": int(keep["months"].sum()),
                    "sharpe_strat": float(keep["sharpe_strat"].median()),
                    "sharpe_6040": float(keep["sharpe_6040"].median()),
                    "sharpe_pp": float(keep["sharpe_pp"].median()),
                    "sharpe_pp_hedged": (
                        float(keep["sharpe_pp_hedged"].median())
                        if variant != "baseline"
                        else np.nan
                    ),
                    "binding": "",
                    "advantage": float(keep["advantage"].median()),
                    "signal_diff_share": (
                        float(keep["signal_diff_share"].median())
                        if variant == "hedged_both"
                        else np.nan
                    ),
                }
            )
    out_path = OUT / "h1_fx_hedged_gold.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"written {out_path}", flush=True)

    summary = pd.DataFrame(rows)
    for variant in VARIANTS:
        sub = summary[
            (summary["variant"] == variant) & summary["market"].isin(MARKETS)
        ]
        med = float(sub["advantage"].median())
        pos = int((sub["advantage"] > 0).sum())
        print(f"{variant}: median advantage {med:+.3f}, positive {pos}/13", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
