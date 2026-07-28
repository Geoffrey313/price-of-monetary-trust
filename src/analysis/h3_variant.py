"""Compare the binary monetary switch with its graded H3 variant.

Inputs: in-memory H1 market series built from the configured snapshots.
Outputs: diagnostics when run directly and per-market objects consumed by
``analysis.run_full_sample`` for the published H3 CSV files.
Purpose: test H3, the functional form of the monetary signal.

H3' is the registered confirmatory test: binary sign rule vs the frozen amplitude variant.
Variant (frozen §2): switch-sleeve bond fraction = min(1, max(0, 0.5 + z2/4)); z2 = expanding-window
z-score of the ratio's deviation from its 84m mean, only once 84 months of deviation history exist
(binary before). Standard (frozen §4): Ledoit-Wolf test on the Sharpe difference, per market and
pooled; prediction upheld if the amplitude variant FAILS to beat the binary at 0.05.
Run: ``PYTHONPATH=src python -m analysis.h3_variant``.
"""
import numpy as np, pandas as pd
from engine import engine as E

def series_for(mkt):
    eq = E.equity_tr(mkt); y = E.long_yield(mkt)
    bond = E.construct_monthly_tr(y.to_timestamp().rename("y"))
    bond_tr = pd.Series(bond["tr"].values, index=y.index)
    bond_w = pd.Series(bond["wealth"].values, index=y.index)
    gold = E.gold_local(mkt); gold_r = gold.pct_change()
    bill = E.short_rate(mkt) / 100 / 12
    ratio = (bond_w / gold).dropna()
    ma = ratio.rolling(E.WINDOW).mean(); cnt = ratio.rolling(E.WINDOW).count()
    d = (ratio - ma)[cnt >= E.WINDOW].dropna()
    mu = d.expanding().mean(); sd = d.expanding().std()
    z2 = ((d - mu) / sd)
    z2[d.expanding().count() < E.WINDOW] = np.nan          # amplitude only after 84m of deviations
    sig_bin = (d >= 0).astype(float)
    frac = (0.5 + z2 / 4).clip(0, 1)
    frac = frac.fillna(sig_bin)                            # binary before z2 exists (both arms equal)
    common = eq.index.intersection(bond_tr.index).intersection(gold_r.index)\
        .intersection(bill.index).intersection(sig_bin.index)
    R = pd.DataFrame({"eq": eq, "bond": bond_tr, "gold": gold_r, "bill": bill}).loc[common]
    return R, sig_bin.reindex(common), frac.reindex(common)

def run_variant(R, f_series):
    f = f_series.shift(1).ffill()
    w_prev = None; out = {}
    for t in R.index:
        fv = f.loc[t]; fv = 1.0 if np.isnan(fv) else float(fv)
        w = {"eq": .25, "bond": .25 + .25 * fv, "bill": .25, "gold": .25 * (1 - fv)}
        if w_prev is None:
            cost = E.COST_ONEWAY * sum(w.values())
        else:
            drift = {a: w_prev[a] * (1 + R.loc[t_prev, a]) for a in w_prev}
            tot = sum(drift.values()); drift = {a: v / tot for a, v in drift.items()}
            cost = E.COST_ONEWAY * sum(abs(w[a] - drift.get(a, 0)) for a in w)
        out[t] = sum(w[a] * R.loc[t, a] for a in w) - cost
        w_prev, t_prev = w, t
    return pd.Series(out)

def lw_test(x_amp, x_bin, bill, lags=12):
    """Ledoit-Wolf style HAC test on Sharpe difference (amp - bin), one-sided p for amp > bin."""
    e1 = (x_amp - bill).dropna(); e2 = (x_bin - bill).reindex(e1.index)
    n = len(e1)
    def sh(e): return e.mean() / e.std()
    diff = (sh(e1) - sh(e2)) * np.sqrt(12)
    # delta method on (mu1,mu2,g1,g2), g=E[e^2]; HAC (Bartlett) covariance of moments
    Y = np.column_stack([e1, e2, e1**2, e2**2]); Yc = Y - Y.mean(0)
    S = Yc.T @ Yc / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Yc[L:].T @ Yc[:-L] / n
        S += w * (G + G.T)
    m1, m2, g1, g2 = Y.mean(0)
    v1, v2 = g1 - m1**2, g2 - m2**2
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -0.5 * m1 / v1**1.5, 0.5 * m2 / v2**1.5])
    se = np.sqrt(grad @ S @ grad / n) * np.sqrt(12)
    from scipy.stats import norm
    return diff, float(1 - norm.cdf(diff / se)), n

def main():
    rows, pool_a, pool_b, pool_bill = [], [], [], []
    for mkt in ["CHE", "CAN", "DEU", "IND"]:
        R, sig_bin, frac = series_for(mkt)
        x_bin = run_variant(R, sig_bin); x_amp = run_variant(R, frac)
        d, p, n = lw_test(x_amp, x_bin, R["bill"])
        sb = ((x_bin - R["bill"]).mean() / (x_bin - R["bill"]).std()) * np.sqrt(12)
        sa = ((x_amp - R["bill"]).mean() / (x_amp - R["bill"]).std()) * np.sqrt(12)
        rows.append((mkt, n, round(sb, 3), round(sa, 3), round(d, 3), round(p, 3)))
        pool_a.append(x_amp); pool_b.append(x_bin); pool_bill.append(R["bill"])
    idx = pool_a[0].index
    for s in pool_a[1:]: idx = idx.intersection(s.index)
    pa = sum(s.reindex(idx) for s in pool_a) / 4
    pb = sum(s.reindex(idx) for s in pool_b) / 4
    pbill = sum(s.reindex(idx) for s in pool_bill) / 4
    d, p, n = lw_test(pa, pb, pbill)
    print("mkt  n  Sharpe_bin  Sharpe_amp  diff(amp-bin)  p(amp>bin)")
    for r in rows: print(r)
    print(("POOLED", n, "-", "-", round(d, 3), round(p, 3)))
    verdict = "UPHELD (amplitude fails to beat binary)" if all(r[5] > 0.05 for r in rows) and p > 0.05 \
        else "FALSIFIED in at least one arm"
    print("H3' registered verdict:", verdict)

if __name__ == "__main__":
    main()
