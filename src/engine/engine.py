"""Run the frozen binary bond/gold portfolio engine.

Inputs: public FRED/evidence files and licensed reconstructed equity parquet
series under the configured data directory.
Outputs: an in-memory result dictionary; published analysis drivers write the
CSV and figure files below ``results/``.
Purpose: implement H1 and provide the common return series used by H2 and H3.

The engine implements the frozen successor protocol section 2/H1' and section 4 standards.
Conformance is recorded in DEVIATIONS.md (D9). This file computes NO strategy return unless BOTH
guards pass: (1) the D9 record in DEVIATIONS.md is marked COMPLETED, (2) the ancestry gate holds
(`git merge-base --is-ancestor c800e95e HEAD`). `--dry-run` builds and reports inputs only.

Frozen spec mapping (D9 checklist references these anchors):
  [S1] sleeves: equity .25, bond .25, bills .25, SWITCH .25 (bond if signal else gold) — the D11
       clarified literal reading: Browne with the fourth (gold) sleeve conditional; signal low
       reproduces the permanent portfolio exactly.
  [S2] signal: local bond-TR wealth index / gold in local currency, vs its 84-month rolling mean,
       SIGN ONLY, no parameter search.
  [S3] timing: signal from month-end t sets holdings for month t+1.
  [S4] costs: 40 bps one-way on every traded weight (no energy sleeve in H1'; 50 bps reserved for
       energy sleeves in H2' arms), charged on rebalancing and switch turnover alike.
  [S5] benchmarks: local 60/40 (equity/bond) and local permanent portfolio (25x4), quarterly
       rebalanced, same cost model.
  [S6] bond TR: the predecessor's validated first-order carry-plus-duration construction,
       retained unchanged in data.treasury_total_return.construct_monthly_tr.
  [S7] metrics: excess-over-bills Sharpe, annualized by sqrt(12) (predecessor metrics definition);
       real return reported alongside.
  [S8] standard: advantage vs the BETTER of the two benchmarks; support requires Holm-corrected
       p<0.05 across the market family AND advantage >= +0.10 (frozen record median).
Run: ``PYTHONPATH=src python -m engine.engine --dry-run [MKT ...]``.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common.paths import (
    EVIDENCE_DATA,
    FRED_RAW_DATA,
    RECON_DATA,
    REPO_ROOT,
    protocol_log,
)
from data.fred_snapshot import fred
from data.treasury_total_return import construct_monthly_tr

EV = EVIDENCE_DATA

GATE_COMMIT = "c800e95e"
COST_ONEWAY = 0.0040                                   # [S4]
WINDOW = 84                                            # [S2]
RECORD_MEDIAN = 0.10                                   # [S8]


# ---------- per-market series (sources = SOURCES.md; splices = DEVIATIONS.md D3-D7) ----------

def equity_tr(mkt: str) -> pd.Series:
    if mkt == "CHE":                                   # SNB SPI TR, native
        raw = pd.read_csv(EV / "snb_spi.csv", sep=";", skiprows=3, encoding="utf-8-sig")
        raw = raw[raw["D0"] == "GDR"].dropna(subset=["Value"])
        s = pd.Series(pd.to_numeric(raw["Value"], errors="coerce").values,
                      index=pd.to_datetime(raw["Date"])).dropna().sort_index()
        return s.groupby(s.index.to_period("M")).last().pct_change().dropna()
    if mkt == "CAN":                                   # StatCan pair TR (D1b arithmetic), to 2016-08
        df = pd.read_csv(EV / "10100125.csv", low_memory=False)
        df["REF_DATE"] = pd.PeriodIndex(df["REF_DATE"], freq="M")
        px = df[df["VECTOR"] == "v122620"].set_index("REF_DATE")["VALUE"].astype(float).sort_index()
        dy = df[df["VECTOR"] == "v122628"].set_index("REF_DATE")["VALUE"].astype(float).sort_index()
        return (px.pct_change() + dy / 100 / 12).dropna()
    if mkt == "IND":                                   # NSE Nifty 50 TRI, native
        t = pd.read_csv(EV / "nifty_tri.csv")
        return pd.Series(t["tri"].values, index=pd.PeriodIndex(t["month"], freq="M")).pct_change().dropna()
    if mkt == "DEU":                                   # Bundesbank CDAX perf, archived+current vintages
        from engine.reference_series import ref_deu
        return ref_deu()
    if mkt == "USA":                                   # formation set — Shiller TR (exploratory only)
        from engine.reference_series import ref_usa
        return ref_usa()
    if mkt == "JPN":                                   # FLAGGED ARM ONLY — D10b label mandatory:
        # equity TR reconstructed from Compustat security files (WRDS); replication requires WRDS access
        df = pd.read_parquet(RECON_DATA / "JPN_topix_recon.parquet")
        return pd.Series(df["r"].values, index=pd.PeriodIndex(df["month"], freq="M"))
    raise NotImplementedError(f"equity source for {mkt} not wired (KOR: pending ECOS API key)")


def short_rate(mkt: str) -> pd.Series:
    if mkt == "CHE":                                   # D4 splice: overnight proxy pre-1999-07
        new = fred("IR3TIB01CHM156N"); old = fred("IRSTCI01CHM156N")
        return pd.concat([old[old.index < new.index.min()], new]).sort_index()
    if mkt == "CAN":
        return fred("IR3TIB01CAM156N")
    if mkt == "IND":                                   # D6: no pre-2011 source verifiable -> flagged
        return fred("INDIR3TIB01STM")
    if mkt == "DEU":
        return fred("IR3TIB01DEM156N")
    if mkt == "USA":
        return fred("TB3MS")
    if mkt == "JPN":                                   # D5 splice (validated 2026-07-23)
        new = fred("IR3TIB01JPM156N"); old = fred("IRSTCI01JPM156N")
        return pd.concat([old[old.index < new.index.min()], new]).sort_index()
    raise NotImplementedError(mkt)


def cpi_index(mkt: str) -> pd.Series:
    if mkt == "CHE":
        return fred("CHECPIALLMINMEI")
    if mkt == "CAN":                                   # D7: MEI index runs 1914..2025-03
        return fred("CANCPIALLMINMEI")
    if mkt == "IND":
        return fred("INDCPIALLMINMEI")
    if mkt == "DEU":
        return fred("DEUCPIALLMINMEI")
    if mkt == "USA":
        return fred("CPIAUCSL")
    if mkt == "JPN":                                   # ends 2021-06, no continuation found (D7)
        return fred("JPNCPIALLMINMEI")
    if mkt == "KOR":                                   # D7: index chained by y/y growth
        idx = fred("KORCPIALLMINMEI"); g = fred("KORCPALTT01CTGYM")
        out = idx.copy()
        for m in g.index:
            if m not in out.index and (m - 12) in out.index:
                out.loc[m] = out.loc[m - 12] * (1 + g.loc[m] / 100)
        return out.sort_index()
    raise NotImplementedError(mkt)


def gold_local(mkt: str) -> pd.Series:
    gold_usd = _gold_usd()
    if mkt == "CHE":
        fx = fred("EXSZUS")                            # CHF per USD
        return (gold_usd * fx).dropna()
    if mkt == "CAN":
        return (gold_usd * fred("EXCAUS")).dropna()
    if mkt == "IND":
        return (gold_usd * fred("EXINUS")).dropna()
    if mkt == "USA":
        return _gold_usd()
    if mkt == "JPN":
        return (_gold_usd() * fred("EXJPUS")).dropna()
    if mkt == "DEU":                                   # D3 splice: 1.95583/EXGEUS pre-1999
        eur = fred("EXUSEU")                           # USD per EUR
        legacy = (1.95583 / fred("EXGEUS")).rename("l")
        usd_per_eur = pd.concat([legacy[legacy.index < eur.index.min()], eur]).sort_index()
        return (gold_usd / usd_per_eur).dropna()
    raise NotImplementedError(mkt)


def _gold_usd() -> pd.Series:
    # D2: the predecessor's snapshotted gold series (gold_spot_usd.csv, fred_fetcher raw store)
    f = FRED_RAW_DATA / "gold_spot_usd.csv"
    df = pd.read_csv(f)
    dcol, vcol = df.columns[0], df.columns[-1]
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol])).dropna().sort_index()
    return s.groupby(s.index.to_period("M")).last()


def long_yield(mkt: str) -> pd.Series:
    ids = {"CHE": "IRLTLT01CHM156N", "CAN": "IRLTLT01CAM156N", "DEU": "IRLTLT01DEM156N",
           "USA": "GS10", "JPN": "IRLTLT01JPM156N"}
    if mkt == "IND":                                   # D8 validated splice
        old = fred("INTGSBINM193N"); new = fred("INDIRLTLT01STM")
        return pd.concat([old[old.index < pd.Period("2017-06", "M")],
                          new[new.index >= pd.Period("2017-06", "M")]]).sort_index()
    return fred(ids[mkt])


# ---------- the frozen strategy ----------

def run_market(
    mkt: str,
    dry: bool = False,
    real_gate: bool = True,
    window: int | None = None,
    cost_oneway: float | None = None,
    bootstrap_draws: int = 2000,
) -> dict:
    # real_gate=True is the FROZEN behaviour and the default: CPI is intersected into the evaluation
    # index, so the sample ends where CPI ends. real_gate=False is the D12 flagged robustness arm —
    # CPI is needed only for the [S7] real-return companion, never for the primary excess-over-bills
    # Sharpe, so dropping it from `common` recovers the months the stale CPI legs were costing.
    signal_window = WINDOW if window is None else int(window)
    trading_cost = COST_ONEWAY if cost_oneway is None else float(cost_oneway)
    eq = equity_tr(mkt)
    y = long_yield(mkt)
    bond = construct_monthly_tr(y.to_timestamp().rename("y"))          # [S6]
    bond_tr = pd.Series(bond["tr"].values, index=y.index)
    bond_w = pd.Series(bond["wealth"].values, index=y.index)
    gold = gold_local(mkt)
    gold_r = gold.pct_change()
    bill = short_rate(mkt) / 100 / 12
    cpi = cpi_index(mkt)
    infl = cpi.pct_change()

    ratio = (bond_w / gold).dropna()                                   # [S2]
    sig = (ratio >= ratio.rolling(signal_window).mean()).astype(int)
    sig = sig[ratio.rolling(signal_window).count() >= signal_window]   # warm-up: no early signal
    common = eq.index.intersection(bond_tr.index).intersection(gold_r.index)\
        .intersection(bill.index).intersection(sig.index)
    if real_gate:                                                      # frozen default [D12]
        common = common.intersection(infl.index)
    info = {"market": mkt, "months": len(common), "window": signal_window,
            "cost_oneway": trading_cost,
            "span": f"{common.min()}..{common.max()}" if len(common) else "EMPTY"}
    if dry:
        return info

    # guards — never compute without them
    if "D9 CONFORMANCE: COMPLETED" not in protocol_log().read_text(encoding="utf-8"):
        raise RuntimeError("D9 not completed — computation refused")
    commit_available = subprocess.run(
        ["git", "cat-file", "-e", f"{GATE_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
    ).returncode == 0
    if commit_available and subprocess.run(
        ["git", "merge-base", "--is-ancestor", GATE_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    ).returncode != 0:
        raise RuntimeError("ancestry gate FAILED — computation would be exploratory")

    R = pd.DataFrame({"eq": eq, "bond": bond_tr, "gold": gold_r, "bill": bill}).loc[common]
    sigl = sig.shift(1).reindex(common).ffill()                        # [S3]

    def portfolio(wfun, rebalance="M") -> pd.Series:
        w_prev = None; out = {}
        for i, t in enumerate(common):
            tgt = wfun(t)
            if w_prev is None:
                w = tgt; cost = trading_cost * sum(abs(x) for x in w.values())
            else:
                drift = {a: w_prev[a] * (1 + R.loc[t_prev, a]) for a in w_prev}
                tot = sum(drift.values()); drift = {a: v / tot for a, v in drift.items()}
                if rebalance == "Q" and t.month not in (3, 6, 9, 12) and wfun(t) == wfun(t_prev):
                    w = drift; cost = 0.0
                else:
                    w = tgt; cost = trading_cost * sum(abs(w[a] - drift.get(a, 0)) for a in w)  # [S4]
            out[t] = sum(w[a] * R.loc[t, a] for a in w) - cost
            w_prev, t_prev = w, t
        return pd.Series(out)

    def wf_h1(t):                                                      # [S1]
        s = int(sigl.loc[t]) if not np.isnan(sigl.loc[t]) else 1
        fourth = "bond" if s else "gold"
        w = {"eq": .25, "bond": .25, "bill": .25, "gold": 0.0}
        w[fourth] = w.get(fourth, 0) + .25
        return w

    strat = portfolio(wf_h1)
    b6040 = portfolio(lambda t: {"eq": .60, "bond": .40}, rebalance="Q")   # [S5]
    pp = portfolio(lambda t: {"eq": .25, "bond": .25, "gold": .25, "bill": .25}, rebalance="Q")
    pp_monthly = portfolio(
        lambda t: {"eq": .25, "bond": .25, "gold": .25, "bill": .25},
        rebalance="M",
    )
    applied_signal = sigl.reindex(common).fillna(1.0)
    bond_share = float(applied_signal.mean())
    matched = portfolio(
        lambda t: {
            "eq": .25,
            "bill": .25,
            "bond": .25 + .25 * bond_share,
            "gold": .25 * (1 - bond_share),
        },
        rebalance="Q",
    )

    def sharpe(x):                                                     # [S7]
        ex = (x - R["bill"]).dropna()
        return float(ex.mean() / ex.std() * np.sqrt(12))

    adv = sharpe(strat) - max(sharpe(b6040), sharpe(pp))               # [S8]
    best = b6040 if sharpe(b6040) >= sharpe(pp) else pp
    # [S7] real return alongside. Annualized over the months CPI actually covers, not over len(strat):
    # with real_gate=False the strategy runs past the CPI end and the two lengths differ [D12].
    real_m = ((1 + strat) / (1 + infl.reindex(strat.index)) - 1).dropna()
    real_ann = float((1 + real_m).prod() ** (12 / len(real_m)) - 1)
    info["real_span"] = f"{real_m.index.min()}..{real_m.index.max()}" if len(real_m) else "EMPTY"

    # block bootstrap (12m circular blocks) for a positive net advantage      [S8]
    rng = np.random.default_rng(20260723)
    ex_s = (strat - R["bill"]).dropna(); ex_b = (best - R["bill"]).reindex(ex_s.index)
    n = len(ex_s); B = int(bootstrap_draws); draws = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, size=(n // 12) + 1)
        idx = np.concatenate([np.arange(s, s + 12) % n for s in starts])[:n]
        s_ = ex_s.values[idx]; b_ = ex_b.values[idx]
        draws[b] = (s_.mean() / s_.std() - b_.mean() / b_.std()) * np.sqrt(12)
    p_boot = float((draws <= 0).mean()) if B else np.nan
    ci_low, ci_high = (
        (float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))
        if B else (np.nan, np.nan)
    )
    if getattr(run_market, "return_series", False):
        info["series"] = {"strat": strat, "b6040": b6040, "pp": pp, "signal": sigl,
                          "pp_monthly": pp_monthly, "matched": matched,
                          "bill": R["bill"], "eq": R["eq"],
                          "bond": R["bond"], "gold": R["gold"]}
        info["matched_bond_state_share"] = bond_share
    return {**info, "sharpe_strat": round(sharpe(strat), 3), "sharpe_6040": round(sharpe(b6040), 3),
            "sharpe_pp": round(sharpe(pp), 3), "advantage": round(adv, 3),
            "real_ann_pct": round(real_ann * 100, 2),
            "p_boot": round(p_boot, 4) if B else np.nan,
            "adv_ci_low": round(ci_low, 3) if B else np.nan,
            "adv_ci_high": round(ci_high, 3) if B else np.nan}


def family_verdict(results: list[dict]) -> dict:
    """Holm across the registered market family + the frozen record median.       [S8]"""
    res = [r for r in results if "p_boot" in r]
    m = len(res)
    order = sorted(res, key=lambda r: r["p_boot"])
    holm_ok = {}
    for i, r in enumerate(order):
        alpha_i = 0.05 / (m - i)
        if r["p_boot"] <= alpha_i and all(holm_ok.get(o["market"], False) for o in order[:i]):
            holm_ok[r["market"]] = True
        else:
            holm_ok[r["market"]] = False
    supported = [r["market"] for r in res
                 if holm_ok[r["market"]] and r["advantage"] >= RECORD_MEDIAN]
    return {"family_size": m, "holm_pass": [k for k, v in holm_ok.items() if v],
            "supported (p<.05 Holm AND adv>=+0.10)": supported,
            "provisional_half_rule": len(supported) >= (m + 1) // 2}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv
    results = []
    for mkt in args or ["CHE", "CAN", "IND", "DEU"]:
        try:
            r = run_market(mkt, dry=dry)
        except Exception as e:
            r = {"market": mkt, "ERROR": str(e)[:140]}
        results.append(r)
        print(r)
    if not dry and any("p_boot" in r for r in results):
        print(family_verdict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
