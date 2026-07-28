"""Construct the constant-duration government-bond total-return series.

Input: a monthly CSV-derived yield series in percent.
Output: an in-memory frame containing monthly return and wealth; the calling
analysis writes the published result CSV files.
Purpose: supply the bond sleeve and bond/gold monetary signal used by H1--H3.

FRED gives *yields*, not total returns. This builds a monthly TR from a monthly
constant-maturity yield the way the frozen protocol specifies:

    TR_t = carry + price-return
         = y_{t-1}/12  +  (-MD·Δy + ½·Convexity·Δy²)

where MD (modified duration) and convexity are those of a par bond at the
prior-month yield, maturity and coupon frequency from the shared protocol.
Duration/convexity are computed numerically from the par-bond price function, so
no closed-form is hand-coded. All parameters come from config — nothing hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.protocol import TREASURY_TOTAL_RETURN

_BUMP = 1e-6  # yield bump (decimal) for numerical derivatives


def _par_bond_price(reprice_yield: float, coupon_rate: float, maturity_years: float,
                    coupons_per_year: int, face: float = 100.0) -> float:
    """Price of a bond (coupon = `coupon_rate`) discounted at `reprice_yield`."""
    m = coupons_per_year
    n_periods = int(round(maturity_years * m))
    i = reprice_yield / m
    cpn = coupon_rate / m * face
    k = np.arange(1, n_periods + 1)
    return float((cpn / (1 + i) ** k).sum() + face / (1 + i) ** n_periods)


def duration_convexity(par_yield: float, maturity_years: float, coupons_per_year: int
                       ) -> tuple[float, float]:
    """Modified duration and convexity of a par bond at `par_yield` (all decimal)."""
    y, mat, m = par_yield, maturity_years, coupons_per_year
    p0 = _par_bond_price(y, y, mat, m)          # par by construction => ~face
    p_up = _par_bond_price(y + _BUMP, y, mat, m)
    p_dn = _par_bond_price(y - _BUMP, y, mat, m)
    dP = (p_up - p_dn) / (2 * _BUMP)
    d2P = (p_up - 2 * p0 + p_dn) / (_BUMP ** 2)
    modified_duration = -dP / p0
    convexity = d2P / p0
    return modified_duration, convexity


def construct_monthly_tr(yields_pct_monthly: pd.Series,
                         include_convexity: bool | None = None) -> pd.DataFrame:
    """Monthly total return + wealth index from a monthly yield series (in percent).

    Index must be monthly-sorted dates. Returns columns: yield_pct, tr (monthly
    total return), wealth (index, base 1.0 at first valid month). ``include_convexity``
    defaults to the frozen shared protocol (convexity is off).
    """
    crit = TREASURY_TOTAL_RETURN
    use_convexity = crit.include_convexity if include_convexity is None else include_convexity
    y = yields_pct_monthly.sort_index().astype(float) / 100.0  # -> decimal
    prev = y.shift(1)
    dy = y - prev

    md = prev.apply(lambda v: duration_convexity(v, crit.maturity_years, crit.coupons_per_year)[0]
                    if pd.notna(v) else np.nan)
    cx = prev.apply(lambda v: duration_convexity(v, crit.maturity_years, crit.coupons_per_year)[1]
                    if pd.notna(v) else np.nan)

    carry = prev / 12.0
    price_return = -md * dy + (0.5 * cx * dy ** 2 if use_convexity else 0.0)
    tr = carry + price_return

    out = pd.DataFrame({"yield_pct": yields_pct_monthly.sort_index(), "tr": tr})
    valid = out["tr"].notna()
    out["wealth"] = np.nan
    out.loc[valid, "wealth"] = (1.0 + out.loc[valid, "tr"]).cumprod()
    return out


def annual_total_return(monthly_tr: pd.Series) -> pd.Series:
    """Compound monthly TR into calendar-year total returns (complete years only)."""
    s = monthly_tr.dropna()
    by_year = (1.0 + s).groupby(s.index.year).prod() - 1.0
    counts = s.groupby(s.index.year).size()
    complete = counts[counts == 12].index
    return by_year.loc[by_year.index.isin(complete)]
