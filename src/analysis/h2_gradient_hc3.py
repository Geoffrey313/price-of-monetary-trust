"""HC3 inference for the H2 exposure gradient (published-number source).

Inputs: the published per-market H2 file ``h2_per_market.csv``.
Outputs: ``h2_gradient_hc3.csv`` in the complete-sample results directory.
Purpose: give the manuscript's HC3 one-sided p-value on the exposure-gradient
slope a generated counterpart. The paper prints this number in the H2 gradient
table and text; before this module it had no source in ``src/`` (the robust
regression CSV holds only the Huber arm). The specification mirrors the OLS
arm of the published gradient: augmentation advantage on the mean exposure
rank, heteroskedasticity-consistent HC3 covariance, one-sided normal p-value
in the predicted (positive) direction.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common.paths import FULL_SAMPLE_RESULTS


def main() -> int:
    per_market = pd.read_csv(FULL_SAMPLE_RESULTS / "h2_per_market.csv")
    outcome = per_market["augmentation_advantage"]
    regressors = sm.add_constant(per_market["exposure_mean_rank"])
    fit = sm.OLS(outcome, regressors).fit(cov_type="HC3")
    slope = float(fit.params["exposure_mean_rank"])
    slope_se = float(fit.bse["exposure_mean_rank"])
    slope_t = float(fit.tvalues["exposure_mean_rank"])
    p_one_sided = float(1 - stats.norm.cdf(slope_t))
    frame = pd.DataFrame(
        [
            {
                "method": "OLS_HC3",
                "intercept": float(fit.params["const"]),
                "slope": slope,
                "slope_se": slope_se,
                "slope_t": slope_t,
                "slope_p_one_sided_normal": p_one_sided,
            }
        ]
    )
    path = FULL_SAMPLE_RESULTS / "h2_gradient_hc3.csv"
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
