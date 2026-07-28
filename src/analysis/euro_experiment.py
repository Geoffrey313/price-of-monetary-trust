"""Wire euro-area and control-market inputs for the full-sample engine.

Inputs: cached WRDS parquet reconstructions and public FRED snapshots.
Outputs: no files when imported; the optional diagnostic CLI prints pre/post
1999 advantages.
Purpose: provide euro legacy-currency splices and reconstructed equity series
used by the published H1--H3 sample; the natural-experiment interpretation
remains exploratory.

Euro natural experiment with a control group (registered before execution).
Treated (lost own monetary policy in 1999): eurozone DE/FR/ES/NL/BE.
Peg control (kept currency, pegs hard to EUR -> imports ECB policy): Denmark.
Float controls (kept independent policy): Switzerland, UK, Sweden, Norway.
Prediction: post-1999 advantage collapses for treated + peg, persists for independent floats.
Equity Compustat-reconstructed except CHE (public SNB). D10b on all reconstructed inputs.
Run: python euro_experiment.py
"""
import numpy as np, pandas as pd
from common.paths import RECON_DATA
from data.wrds import WRDSClient
from engine import engine as E
from engine import equity_reconstruction as F
from engine import reference_series as G

RATES = {"DEM": 1.95583, "FRF": 6.55957, "ESP": 166.386, "NLG": 2.20371, "BEF": 40.3399,
         "ATS": 13.7603, "PTE": 200.482, "LUF": 40.3399, "IEP": 0.787564, "EUR": 1.0}
# market -> (yield, short, cpi, usd-fx, group, legacy_euro_rate or None)
EURO_CONFIG = {
    "DEU": ("IRLTLT01DEM156N", "IR3TIB01DEM156N", "DEUCPIALLMINMEI", "EXGEUS", "euro", 1.95583),
    "FRA": ("IRLTLT01FRM156N", "IR3TIB01FRM156N", "FRACPIALLMINMEI", "EXFRUS", "euro", 6.55957),
    "ESP": ("IRLTLT01ESM156N", "IR3TIB01ESM156N", "ESPCPIALLMINMEI", "EXSPUS", "euro", 166.386),
    "NLD": ("IRLTLT01NLM156N", "IR3TIB01NLM156N", "NLDCPIALLMINMEI", "EXNEUS", "euro", 2.20371),
    "BEL": ("IRLTLT01BEM156N", "IR3TIB01BEM156N", "BELCPIALLMINMEI", "EXBEUS", "euro", 40.3399),
}
CTRL = {
    "DNK": ("IRLTLT01DKM156N", "IR3TIB01DKM156N", "DNKCPIALLMINMEI", "EXDNUS", "peg-euro", None),
    "SWE": ("IRLTLT01SEM156N", "IR3TIB01SEM156N", "SWECPIALLMINMEI", "EXSDUS", "float", None),
    "CHE": (None, None, None, None, "float", None),   # public, fully in h1_engine
    "GBR": ("IRLTLT01GBM156N", "IR3TIB01GBM156N", "GBRCPIALLMINMEI", "EXUSUK", "float", None),
    "NOR": ("IRLTLT01NOM156N", "IR3TIB01NOM156N", "NORCPIALLMINMEI", "EXNOUS", "float", None),
}


def equity_recon(db, fic, euro_rate):
    cache = RECON_DATA / f"{fic}_recon.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return pd.Series(df["r"].values, index=pd.PeriodIndex(df["month"], freq="M"))
    sec = G.pull_market(db, fic, "global"); csho = G.annual_shares(db, fic, "global")
    curs = ({cc: RATES[cc] for cc in sec["curcdd"].dropna().unique() if cc in RATES}
            if euro_rate else {sec["curcdd"].mode()[0]: 1.0})
    rec = G.build_index(sec, csho, curs, top_n=None)
    pd.DataFrame({"month": rec.index.astype(str), "r": rec.values}).to_parquet(cache)
    return rec


def wire(mkt, cfg, recon):
    y, s, c, fx, grp, rate = cfg
    if y is None:                                     # fully public in h1_engine (e.g. CHE) — no override
        return
    _eq, _sr, _gl, _ly, _cpi = E.equity_tr, E.short_rate, E.gold_local, E.long_yield, E.cpi_index
    if recon is not None:
        E.equity_tr = lambda mm, r=recon, m0=mkt, o=_eq: r if mm == m0 else o(mm)
    E.short_rate = lambda mm, sid=s, m0=mkt, o=_sr: E.fred(sid) if mm == m0 else o(mm)
    E.long_yield = lambda mm, yid=y, m0=mkt, o=_ly: E.fred(yid) if mm == m0 else o(mm)
    E.cpi_index = lambda mm, cid=c, m0=mkt, o=_cpi: E.fred(cid) if mm == m0 else o(mm)

    def gl(mm, fxid=fx, rt=rate, m0=mkt, o=_gl):
        if mm != m0:
            return o(mm)
        g = E._gold_usd()
        if rt:                                        # euro member: legacy splice at irrevocable rate
            eur = E.fred("EXUSEU"); legacy = (rt / E.fred(fxid))
            upe = pd.concat([legacy[legacy.index < eur.index.min()], eur]).sort_index()
            return (g / upe).dropna()
        fxs = E.fred(fxid)                            # control: plain USD fx (fx is ccy/USD or USD/ccy)
        return (g * fxs).dropna() if fxid in ("EXDNUS", "EXSDUS", "EXNOUS") else (g / fxs).dropna()
    E.gold_local = gl


def adv_split(mkt):
    E.run_market.return_series = True
    ser = E.run_market(mkt)["series"]; idx = ser["strat"].index
    out = {}
    for tag, msk in [("pre", idx < pd.Period("1999-01", "M")), ("post", idx >= pd.Period("1999-01", "M"))]:
        m = pd.Series(msk, index=idx)
        def sh(x):
            ex = (x - ser["bill"])[m].dropna()
            return ex.mean() / ex.std() * np.sqrt(12) if len(ex) > 24 else np.nan
        out[tag] = (sh(ser["strat"]) - max(sh(ser["b6040"]), sh(ser["pp"])), int(m.sum()))
    return out


def main():
    with WRDSClient() as db:
        for mkt, cfg in EURO_CONFIG.items():
            wire(mkt, cfg, equity_recon(db, mkt, cfg[5]))
        for mkt, cfg in CTRL.items():
            rec = None if mkt == "CHE" else equity_recon(db, mkt, None)
            wire(mkt, cfg, rec)
    print(f"{'market':5s} {'group':10s} {'pre-1999':>16s} {'post-1999':>16s}")
    post = {"euro": [], "peg-euro": [], "float": []}
    for mkt, cfg in {**EURO_CONFIG, **CTRL}.items():
        r = adv_split(mkt)
        post[cfg[4]].append(r["post"][0])
        print(f"{mkt:5s} {cfg[4]:10s}  {r['pre'][0]:+.3f} ({r['pre'][1]:>3d}m)  {r['post'][0]:+.3f} ({r['post'][1]:>3d}m)")
    print("\nPOST-1999 mean advantage by group:")
    for g, v in post.items():
        print(f"  {g:10s}: {np.nanmean(v):+.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
