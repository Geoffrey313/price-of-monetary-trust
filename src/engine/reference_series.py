"""Reconstruct and validate the equity reference series.

Inputs: public validation files and licensed WRDS/CRSP parquet snapshots.
Outputs: reconstructed parquet inputs and printed D10a validation diagnostics.
Purpose: supply the equity sleeve used by H1--H3 and retain the frozen
reconstruction-versus-reference gate.

D10a validation gate — all testable markets (criterion frozen in DEVIATIONS.md D10a: per market,
monthly corr >= 0.98 AND |annualized TR gap| <= 60 bps; ALL testable markets must pass).

Markets and references:
  CHE  Compustat Global all-share (CHF)            vs SNB SPI TR (data/evidence/snb_spi.csv)     [PASS 26.8bps/0.9839]
  DEU  Compustat Global all-share (DEM->EUR fixed) vs Bundesbank CDAX perf (wu018a 2011+current)
  CAN  Compustat NA all-share, fic=CAN (CAD)       vs StatCan pair TR v122620 + v122628/12 (10100125.csv)
  IND  Compustat Global top-50 by lagged cap (INR) vs NSE Nifty 50 TRI (fetched separately)
  USA  Compustat NA top-500 by lagged cap (USD)    vs Shiller S&P TR (engine provider)

Construction identical to build_recon_tr.py (month-end g_secd/secd rows; TRI=prccd/ajexdi*trfd;
value weight = lagged cap; annual-cshoi fallback for missing cshoc; DEM converted at the irrevocable
1.95583 so the 1999 boundary return is clean, per D3).

Gate note: validation-set work; no strategy return computed.
Run: ``PYTHONPATH=src python -m engine.reference_series DEU CAN``.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import EVIDENCE_DATA, FRED_RAW_DATA, RECON_DATA
from data.wrds import WRDSClient

EV = EVIDENCE_DATA
CACHE = RECON_DATA

CRIT_CORR, CRIT_GAP_BPS = 0.98, 60.0


# ---------------- reconstruction ----------------

def pull_market(db, fic: str, lib: str) -> pd.DataFrame:
    cache = CACHE / f"{fic}_monthend.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    table = "comp_global_daily.g_secd" if lib == "global" else "comp_na_daily_all.secd"
    sectab = "comp_global_daily.g_security" if lib == "global" else "comp_na_daily_all.security"
    quote_unit = "d.qunit" if lib == "global" else "1.0::double precision"
    q = f"""
        SELECT t.gvkey, t.iid, t.datadate, t.prccd, t.ajexdi, t.trfd, t.cshoc, t.curcdd, t.qunit
        FROM (
          SELECT d.gvkey, d.iid, d.datadate, d.prccd, d.ajexdi, d.trfd, d.cshoc, d.curcdd,
                 {quote_unit} AS qunit,
                 row_number() OVER (PARTITION BY d.gvkey, d.iid, date_trunc('month', d.datadate)
                                    ORDER BY d.datadate DESC) AS rn
          FROM {table} d
          WHERE d.fic = '{fic}' AND d.prccd IS NOT NULL AND d.ajexdi IS NOT NULL AND d.trfd IS NOT NULL
        ) t
        JOIN {sectab} s ON s.gvkey = t.gvkey AND s.iid = t.iid
        WHERE t.rn = 1 AND s.tpci = '0' AND s.excntry = '{fic}'
    """
    df = db.raw_sql(q)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def annual_shares(db, fic: str, lib: str) -> pd.DataFrame:
    cache = CACHE / f"{fic}_csho.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if lib == "global":
        q = (f"SELECT gvkey, datadate, cshoi AS csho FROM comp_global_daily.g_funda "
             f"WHERE fic='{fic}' AND cshoi IS NOT NULL AND indfmt='INDL' "
             f"AND datafmt='HIST_STD' AND consol='C'")
    else:
        q = (f"SELECT gvkey, datadate, csho FROM comp_na_daily_all.funda "
             f"WHERE fic='{fic}' AND csho IS NOT NULL AND indfmt='INDL' "
             f"AND datafmt='STD' AND consol='C' AND popsrc='D'")
    df = db.raw_sql(q)
    df.to_parquet(cache)
    return df


def build_index(sec: pd.DataFrame, csho: pd.DataFrame, currencies: dict[str, float],
                top_n: int | None = None, min_names: int = 20) -> pd.Series:
    sec = sec[sec["curcdd"].isin(currencies)].copy()
    conv = sec["curcdd"].map(currencies)
    quote_unit = (
        pd.to_numeric(sec["qunit"], errors="coerce")
        if "qunit" in sec
        else pd.Series(1.0, index=sec.index)
    )
    if quote_unit.isna().any() or bool((quote_unit <= 0).any()):
        raise ValueError("unité de cotation absente, nulle ou négative")
    # Prix par action dans la monnaie principale. QUNIT vaut notamment 1 000
    # pour certaines cotations historiques brésiliennes.
    sec["prccd"] = sec["prccd"] / conv / quote_unit
    sec["datadate"] = pd.to_datetime(sec["datadate"])
    sec["month"] = sec["datadate"].dt.to_period("M")
    sec = sec.sort_values(["gvkey", "iid", "datadate"]).drop_duplicates(
        ["gvkey", "iid", "month"], keep="last")
    sec["tri"] = sec["prccd"] / sec["ajexdi"] * sec["trfd"]

    csho = csho.copy()
    csho["datadate"] = pd.to_datetime(csho["datadate"])
    csho["month"] = csho["datadate"].dt.to_period("M")
    csho = csho.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "month"], keep="last")
    sec = sec.merge(csho[["gvkey", "month", "csho"]], on=["gvkey", "month"], how="left")
    sec = sec.sort_values(["gvkey", "iid", "month"])
    sec["csho_ff"] = sec.groupby(["gvkey", "iid"])["csho"].ffill()
    sec["shares"] = sec["cshoc"].fillna(sec["csho_ff"] * 1e6)
    sec = sec.dropna(subset=["shares"])
    sec["cap"] = sec["prccd"] * sec["shares"]

    g = sec.groupby(["gvkey", "iid"])
    sec["r"] = g["tri"].pct_change()
    sec["gap"] = g["month"].diff().apply(lambda d: getattr(d, "n", None))
    sec.loc[sec["gap"] != 1, "r"] = np.nan
    sec["w"] = g["cap"].shift(1)
    sec = sec.dropna(subset=["r", "w"])
    sec = sec[(sec["r"] > -0.99) & (sec["r"] < 10)]
    if top_n:
        sec["rank"] = sec.groupby("month")["w"].rank(ascending=False)
        sec = sec[sec["rank"] <= top_n]
    n = sec.groupby("month").size()
    idx = sec.groupby("month").apply(
        lambda x: np.average(x["r"], weights=x["w"]), include_groups=False)
    idx = idx[n >= min_names]
    if idx.empty:
        print(
            f"   recon vide, months <{min_names} names dropped: "
            f"{int((n < min_names).sum())}"
        )
    else:
        print(
            f"   recon months {idx.index.min()}..{idx.index.max()}, "
            f"median names/month {int(n.median())}, "
            f"months <{min_names} names dropped: {int((n < min_names).sum())}"
        )
    return idx


# ---------------- references ----------------

def _bbk_csv(path: Path) -> pd.Series:
    vals = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^(\d{4}-\d{2});([0-9.,]+)", line)
        if m:
            vals[m.group(1)] = float(m.group(2).replace(".", "").replace(",", "."))
    s = pd.Series(vals)
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s.sort_index()


def ref_deu() -> pd.Series:
    old = _bbk_csv(EV / "wu018a_2011.csv")          # 1970-01 .. 2011-11 vintage
    new = _bbk_csv(EV / "wu018a.csv")               # 1994-06 .. current vintage
    ov = pd.concat([old.rename("o"), new.rename("n")], axis=1).dropna()
    rel = (ov["o"] / ov["n"] - 1).abs().max()
    print(f"   CDAX vintage overlap {ov.index.min()}..{ov.index.max()}: max level mismatch {rel:.2%}")
    lvl = pd.concat([old[old.index < new.index.min()], new])
    return lvl.pct_change().dropna()


def ref_can() -> pd.Series:
    df = pd.read_csv(EV / "10100125.csv", low_memory=False)
    df["REF_DATE"] = pd.PeriodIndex(df["REF_DATE"], freq="M")
    px = df[df["VECTOR"] == "v122620"].set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    dy = df[df["VECTOR"] == "v122628"].set_index("REF_DATE")["VALUE"].astype(float).sort_index()
    r = px.pct_change() + dy / 100.0 / 12.0        # D1b-validated arithmetic
    return r.dropna()


def ref_che() -> pd.Series:
    raw = pd.read_csv(EV / "snb_spi.csv", sep=";", skiprows=3, encoding="utf-8-sig")
    raw = raw[raw["D0"] == "GDR"].dropna(subset=["Value"])
    s = pd.Series(pd.to_numeric(raw["Value"], errors="coerce").values,
                  index=pd.to_datetime(raw["Date"])).dropna().sort_index()
    return s.groupby(s.index.to_period("M")).last().pct_change().dropna()


def ref_ind() -> pd.Series:
    f = EV / "nifty_tri.csv"                        # fetched by fetch_nifty_tri.sh
    if not f.exists():
        raise FileNotFoundError("run fetch_nifty_tri.sh first (NSE TRI download)")
    df = pd.read_csv(f)
    s = pd.Series(df["tri"].values, index=pd.PeriodIndex(df["month"], freq="M")).sort_index()
    return s.pct_change().dropna()


def ref_usa() -> pd.Series:
    """Rebuild Shiller's monthly S&P total-return wealth index."""
    workbook = (FRED_RAW_DATA / "sp500_total_return.xls").read_bytes()
    sh = pd.read_excel(
        io.BytesIO(workbook),
        sheet_name="Data",
        header=7,
        engine="xlrd",
    )
    sh = sh.rename(columns=lambda column: str(column).strip())
    raw_date = pd.to_numeric(sh["Date"], errors="coerce")
    years = raw_date.apply(lambda value: int(value) if pd.notna(value) else None)
    months = (
        (
            raw_date
            - raw_date.apply(lambda value: int(value) if pd.notna(value) else 0)
        )
        * 100
    ).round()
    dates = []
    for year, month in zip(years, months):
        if year is None or pd.isna(month):
            dates.append(pd.NaT)
        else:
            dates.append(pd.Timestamp(year=int(year), month=int(month), day=1))
    sh["_date"] = pd.Series(dates, index=sh.index)
    sh = sh.dropna(subset=["_date"]).reset_index(drop=True)
    price = pd.to_numeric(sh["P"], errors="coerce")
    dividend = pd.to_numeric(sh["D"], errors="coerce")
    monthly_return = (price + dividend / 12.0) / price.shift(1) - 1.0
    wealth = 100.0 * (1.0 + monthly_return.fillna(0.0)).cumprod()
    wealth[price.isna()] = pd.NA
    tr = pd.Series(wealth.values, index=sh["_date"]).dropna()
    tr.index = tr.index.to_period("M")
    return tr.pct_change().dropna()


def recon_usa_crsp(db) -> pd.Series:
    """US top-500 value-weighted from CRSP msf (ret includes dividends; cap = |prc| * shrout).
    Universe: common shares (shrcd 10/11), NYSE/AMEX/NASDAQ (exchcd 1-3)."""
    cache = CACHE / "USA_crsp_msf.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = db.raw_sql(
            "SELECT m.permno, m.date, m.ret, abs(m.prc) * m.shrout AS cap "
            "FROM crsp.msf m JOIN crsp.msenames n ON n.permno = m.permno "
            "AND m.date BETWEEN n.namedt AND n.nameendt "
            "WHERE n.shrcd IN (10,11) AND n.exchcd IN (1,2,3) "
            "AND m.ret IS NOT NULL AND m.prc IS NOT NULL AND m.shrout IS NOT NULL")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    df = df.sort_values(["permno", "month"])
    df["w"] = df.groupby("permno")["cap"].shift(1)
    df = df.dropna(subset=["w", "ret"])
    df["rank"] = df.groupby("month")["w"].rank(ascending=False)
    df = df[df["rank"] <= 500]
    n = df.groupby("month").size()
    idx = df.groupby("month").apply(
        lambda x: np.average(x["ret"], weights=x["w"]), include_groups=False)
    print(f"   recon (CRSP top-500) months {idx.index.min()}..{idx.index.max()}, "
          f"median names/month {int(n.median())}")
    return idx


MK = {
    "CHE": dict(lib="global", cur={"CHF": 1.0}, top_n=None, ref=ref_che, ref_name="SNB SPI TR"),
    "DEU": dict(lib="global", cur={"DEM": 1.95583, "EUR": 1.0}, top_n=None, ref=ref_deu,
                ref_name="Bundesbank CDAX performance (2011 vintage + current, spliced)"),
    "CAN": dict(lib="na", cur={"CAD": 1.0}, top_n=300, ref=ref_can,
                ref_name="StatCan v122620 + v122628/12 pair TR"),
    # top_n=300: the reference is the TSE 300 (300 names, 1977-2002) then S&P/TSX Composite
    # (~250); the first all-share run FAILed on 1984-era microcap noise alone - method revision
    # logged in DEVIATIONS.md, criterion untouched.
    "IND": dict(lib="global", cur={"INR": 1.0}, top_n=50, ref=ref_ind, ref_name="NSE Nifty 50 TRI"),
    "USA": dict(lib="crsp", cur={"USD": 1.0}, top_n=500, ref=ref_usa, ref_name="Shiller S&P TR"),
}


def run(fic: str) -> bool:
    cfg = MK[fic]
    if fic == "USA":
        with WRDSClient() as db:
            recon = recon_usa_crsp(db)
        print(f"\n=== USA (CRSP) ===")
    else:
        with WRDSClient() as db:
            sec = pull_market(db, fic, cfg["lib"])
            csho = annual_shares(db, fic, cfg["lib"])
        print(f"\n=== {fic}: {len(sec)} month-end rows, {len(csho)} annual share rows ===")
        recon = build_index(sec, csho, cfg["cur"], top_n=cfg["top_n"])
    ref = cfg["ref"]()
    df = pd.concat([recon.rename("recon"), ref.rename("ref")], axis=1).dropna()
    ann_r = (1 + df["recon"]).prod() ** (12 / len(df)) - 1
    ann_f = (1 + df["ref"]).prod() ** (12 / len(df)) - 1
    gap = abs(ann_r - ann_f) * 1e4
    corr = float(df["recon"].corr(df["ref"]))
    print(f"== D10a gate, {fic} (recon vs {cfg['ref_name']}) ==")
    print(f"   overlap {df.index.min()}..{df.index.max()} ({len(df)} months)")
    print(f"   annualized: recon {ann_r:+.3%}  ref {ann_f:+.3%}  |gap| {gap:.1f} bps (<= {CRIT_GAP_BPS:.0f})")
    print(f"   monthly corr {corr:.4f} (>= {CRIT_CORR})")
    ok = gap <= CRIT_GAP_BPS and corr >= CRIT_CORR
    print(f"   VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        d = df["recon"] - df["ref"]
        print("   diagnostics (bps by 5y era):")
        print((d.groupby(d.index.year // 5 * 5).agg(["mean", "std"]) * 1e4).round(1).to_string())
    return ok


def main() -> int:
    fics = sys.argv[1:] or ["DEU", "CAN"]
    results = {f: run(f) for f in fics}
    print("\n==== summary:", {f: ("PASS" if v else "FAIL") for f, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
