"""Build and wire the registered WRDS equity reconstructions.

Inputs: licensed WRDS observations or cached parquet reconstructions.
Outputs: parquet return series below the configured reconstruction directory.
Purpose: supply identically constructed equity sleeves for the H1--H3
thirteen-market sample; the D10b provenance label remains binding.

FLAGGED BATCH — six WRDS-reconstruction arms (registered before execution).
D10b label on every number: equity TR reconstructed from Compustat security files (WRDS);
replication requires WRDS access. None enters the family, Holm, or the half rule.
Run: ``PYTHONPATH=src python -m engine.equity_reconstruction``.
"""
import numpy as np, pandas as pd
from common.paths import RECON_DATA
from data.wrds import WRDSClient
from engine import engine as E
from engine import reference_series as G

CACHE = RECON_DATA
CFG = {
    "GBR": dict(cur={"GBP": 1.0}, top_n=None, yid="IRLTLT01GBM156N", sid="IR3TIB01GBM156N",
                cid="GBRCPIALLMINMEI", fx=("EXUSUK", "div")),      # USD per GBP -> gold/fx
    "ZAF": dict(cur={"ZAR": 1.0}, top_n=None, yid="INTGSBZAM193N", sid="INTGSTZAM193N",
                cid="ZAFCPIALLMINMEI", fx=("EXSFUS", "mul")),
    "NOR": dict(cur={"NOK": 1.0}, top_n=None, yid="IRLTLT01NOM156N", sid="IR3TIB01NOM156N",
                cid="NORCPIALLMINMEI", fx=("EXNOUS", "mul")),
    "AUS": dict(cur={"AUD": 1.0}, top_n=500, yid="IRLTLT01AUM156N", sid="IR3TIB01AUM156N",
                cid="AUSCPIALLQINMEI", fx=("EXUSAL", "div"), cpi_q=True),
    "FRA": dict(cur={"FRF": 6.55957, "EUR": 1.0}, top_n=None, yid="IRLTLT01FRM156N",
                sid="IR3TIB01FRM156N", cid="FRACPIALLMINMEI", fx=("FR_EUR", "div")),
    "MEX": dict(cur={"MXN": 1.0}, top_n=None, yid="IRLTLT01MXM156N", sid="IR3TIB01MXM156N",
                cid="MEXCPIALLMINMEI", fx=("EXMXUS", "mul"), members="150039"),
}


def build_recon(db, mkt, cfg):
    f = CACHE / f"{mkt}_recon.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        return pd.Series(df["r"].values, index=pd.PeriodIndex(df["month"], freq="M"))
    sec = G.pull_market(db, mkt, "global")
    csho = G.annual_shares(db, mkt, "global")
    s = sec[sec["curcdd"].isin(cfg["cur"])].copy()
    s["prccd"] = s["prccd"] / s["curcdd"].map(cfg["cur"])
    s["datadate"] = pd.to_datetime(s["datadate"]); s["month"] = s["datadate"].dt.to_period("M")
    s = s.sort_values(["gvkey", "iid", "datadate"]).drop_duplicates(["gvkey", "iid", "month"], keep="last")
    s["tri"] = s["prccd"] / s["ajexdi"] * s["trfd"]
    c = csho.copy(); c["datadate"] = pd.to_datetime(c["datadate"]); c["month"] = c["datadate"].dt.to_period("M")
    c = c.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "month"], keep="last")
    s = s.merge(c[["gvkey", "month", "csho"]], on=["gvkey", "month"], how="left").sort_values(["gvkey", "iid", "month"])
    s["csho_ff"] = s.groupby(["gvkey", "iid"])["csho"].ffill()
    s["shares"] = s["cshoc"].fillna(s["csho_ff"] * 1e6); s = s.dropna(subset=["shares"])
    s["cap"] = s["prccd"] * s["shares"]
    g = s.groupby(["gvkey", "iid"]); s["r"] = g["tri"].pct_change()
    s["gp"] = g["month"].diff().apply(lambda d: getattr(d, "n", None)); s.loc[s["gp"] != 1, "r"] = np.nan
    s["w"] = g["cap"].shift(1); s = s.dropna(subset=["r", "w"]); s = s[(s["r"] > -0.99) & (s["r"] < 10)]
    if cfg.get("members"):
        mem = db.raw_sql(f"SELECT gvkey, iid, \"from\" AS f, thru FROM comp_global_daily.g_idxcst_his WHERE gvkeyx='{cfg['members']}'")
        mem["f"] = pd.to_datetime(mem["f"]); mem["thru"] = pd.to_datetime(mem["thru"]).fillna(pd.Timestamp("2030-01-01"))
        s = s.merge(mem, on=["gvkey", "iid"], how="inner")
        ms = s["month"].dt.start_time
        s = s[(ms >= s["f"].values.astype("datetime64[M]").astype("datetime64[ns]")) & (ms <= s["thru"])]
    if cfg.get("top_n"):
        s["rank"] = s.groupby("month")["w"].rank(ascending=False); s = s[s["rank"] <= cfg["top_n"]]
    n = s.groupby("month").size()
    rec = s.groupby("month").apply(lambda x: np.average(x["r"], weights=x["w"]), include_groups=False)
    rec = rec[n >= max(20, (n.median() * 0.25))]
    pd.DataFrame({"month": rec.index.astype(str), "r": rec.values}).to_parquet(f)
    print(f"{mkt} recon: {rec.index.min()}..{rec.index.max()}, median names {int(n.median())}")
    return rec


def wire(mkt, cfg, rec):
    _eq, _sr, _cpi, _gl, _ly = E.equity_tr, E.short_rate, E.cpi_index, E.gold_local, E.long_yield
    E.equity_tr = lambda m, _o=_eq: rec if m == mkt else _o(m)
    E.short_rate = lambda m, _o=_sr: E.fred(cfg["sid"]) if m == mkt else _o(m)

    def cpi(m, _o=_cpi):
        if m != mkt:
            return _o(m)
        s = E.fred(cfg["cid"])
        if cfg.get("cpi_q"):     # quarterly -> monthly by log-linear interpolation (deviation noted)
            full = pd.period_range(s.index.min(), s.index.max(), freq="M")
            s = np.exp(np.log(s.astype(float)).reindex(full).interpolate())
        return s
    E.cpi_index = cpi

    def gl(m, _o=_gl):
        if m != mkt:
            return _o(m)
        g = E._gold_usd()
        if cfg["fx"][0] == "FR_EUR":   # D3: 6.55957/EXFRUS pre-1999, EXUSEU after (USD per EUR-equiv)
            eur = E.fred("EXUSEU"); legacy = (6.55957 / E.fred("EXFRUS"))
            ov = pd.concat([legacy.rename("o"), eur.rename("n")], axis=1).dropna()
            print(f"   [D3 FR] overlap {len(ov)}m mean gap {(ov['o']-ov['n']).mean():+.4f} corr {ov['o'].corr(ov['n']):.3f}")
            usd_per_eur = pd.concat([legacy[legacy.index < eur.index.min()], eur]).sort_index()
            return (g / usd_per_eur).dropna()      # gold in EUR (FRF converted at the irrevocable rate)
        fx = E.fred(cfg["fx"][0])
        return (g * fx).dropna() if cfg["fx"][1] == "mul" else (g / fx).dropna()
    E.gold_local = gl
    E.long_yield = lambda m, _o=_ly: E.fred(cfg["yid"]) if m == mkt else _o(m)


def main():
    with WRDSClient() as db:
        recs = {m: build_recon(db, m, c) for m, c in CFG.items()}
    print("\n[D10b] every result: equity TR reconstructed from Compustat security files (WRDS); "
          "replication requires WRDS access — FLAGGED, outside the family/Holm/half rule\n")
    for m, c in CFG.items():
        try:
            wire(m, c, recs[m])
            print(E.run_market(m))
        except Exception as e:
            print({"market": m, "ERROR": str(e)[:140]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
