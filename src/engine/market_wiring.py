"""Wire every published market series into the portfolio engine.

Inputs: cached parquet equity reconstructions and public FRED snapshots.
Outputs: no files; functions install the registered market-specific loaders on
the in-memory engine.
Purpose: provide the uniform thirteen-market input layer used by H1--H3 without
retaining the exploratory plotting code that originally contained this logic.
"""
from __future__ import annotations

import pandas as pd

from common.names import EURO, MARKETS
from engine import engine as E
from engine import equity_reconstruction as F


def wire_all() -> None:
    """Install the base (non-euro) reconstructed equity and macro series.

    This is the engine-layer helper. Euro markets are wired separately, on top of
    this, by ``run_full_sample.wire_everything()``, the single complete entry
    point. Callers should use ``wire_everything()``, not this function directly.
    """
    for market, config in F.CFG.items():
        if market not in MARKETS or market in EURO:
            continue
        frame = pd.read_parquet(F.CACHE / f"{market}_recon.parquet")
        returns = pd.Series(
            frame["r"].values,
            index=pd.PeriodIndex(frame["month"], freq="M"),
        )
        F.wire(market, config, returns)

    for market, series in [
        (
            "JPN",
            ("IR3TIB01JPM156N", "IRLTLT01JPM156N", "JPNCPIALLMINMEI", "EXJPUS"),
        ),
    ]:
        path = (
            F.CACHE / f"{market}_recon.parquet"
            if market == "KOR"
            else F.CACHE / "JPN_topix_recon.parquet"
        )
        frame = pd.read_parquet(path)
        returns = pd.Series(
            frame["r"].values,
            index=pd.PeriodIndex(frame["month"], freq="M"),
        )
        short_id, yield_id, _cpi_id, fx_id = series
        old_equity = E.equity_tr
        old_short = E.short_rate
        old_gold = E.gold_local
        old_yield = E.long_yield
        E.equity_tr = (
            lambda candidate, value=returns, target=market, previous=old_equity:
            value if candidate == target else previous(candidate)
        )

        def short_rate(
            candidate: str,
            sid: str = short_id,
            target: str = market,
            previous=old_short,
        ):
            if candidate != target:
                return previous(candidate)
            if target == "JPN":
                new = E.fred("IR3TIB01JPM156N")
                old = E.fred("IRSTCI01JPM156N")
                return pd.concat([old[old.index < new.index.min()], new]).sort_index()
            return E.fred(sid)

        E.short_rate = short_rate
        E.gold_local = (
            lambda candidate, fx=fx_id, target=market, previous=old_gold:
            (E._gold_usd() * E.fred(fx)).dropna()
            if candidate == target
            else previous(candidate)
        )
        E.long_yield = (
            lambda candidate, sid=yield_id, target=market, previous=old_yield:
            E.fred(sid) if candidate == target else previous(candidate)
        )
