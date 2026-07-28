"""Reproduce the thirteen-market inputs from the committed derived layer.

Inputs: ``data/derived/{market}.parquet`` (one file per market, monthly).
Outputs: none; installs the per-market loaders on the in-memory engine.
Purpose: let the full pipeline reproduce H1-H3, the era analyses and the figures
without any licensed WRDS access.

Each derived file holds five monthly series: the aggregated cap-weighted equity
total return (``eq``), the long government yield, gold in local currency, the
short rate, and the consumer price index. Only ``eq`` is downstream of WRDS, and
it is a market-level aggregate, never firm-level records. The other four are
public (FRED and a public gold series). Running from this layer bypasses
``equity_reconstruction`` entirely.

Regenerate the layer with ``PYTHONPATH=src python -m engine.derived_inputs``.
"""
from __future__ import annotations

import os

import pandas as pd

from common.names import MARKETS
from common.paths import DATA_ROOT
from engine import engine as E

DERIVED = DATA_ROOT / "derived"
COLUMNS = ("eq", "long_yield", "gold_local", "short_rate", "cpi_index")


def available() -> bool:
    """True when every market's derived file is present."""
    return DERIVED.exists() and all((DERIVED / f"{m}.parquet").exists() for m in MARKETS)


def missing_files() -> list[str]:
    """Return the expected derived files that are still absent."""
    return [f"{market}.parquet" for market in MARKETS if not (DERIVED / f"{market}.parquet").exists()]


def _load() -> dict[str, pd.DataFrame]:
    frames = {}
    for market in MARKETS:
        frame = pd.read_parquet(DERIVED / f"{market}.parquet")
        missing = [column for column in ("month", *COLUMNS) if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{DERIVED / f'{market}.parquet'} is missing required columns: {', '.join(missing)}"
            )
        frame.index = pd.PeriodIndex(frame["month"], freq="M")
        frames[market] = frame
    return frames


def wire_from_derived() -> None:
    """Install per-market series from the derived layer, bypassing WRDS."""
    frames = _load()

    def column(name: str):
        return lambda market: frames[market][name].dropna()

    E.equity_tr = column("eq")
    E.long_yield = column("long_yield")
    E.gold_local = column("gold_local")
    E.short_rate = column("short_rate")
    E.cpi_index = column("cpi_index")


def export() -> None:
    """Rebuild the derived layer from the full WRDS wiring (author use only)."""
    from analysis import run_full_sample as RFS

    prior = os.environ.pop("FOUR_QUADRANT_FROM_DERIVED", None)
    try:
        RFS.wire_everything()
    finally:
        if prior is not None:
            os.environ["FOUR_QUADRANT_FROM_DERIVED"] = prior
    DERIVED.mkdir(parents=True, exist_ok=True)
    for market in MARKETS:
        series = {
            "eq": E.equity_tr(market),
            "long_yield": E.long_yield(market),
            "gold_local": E.gold_local(market),
            "short_rate": E.short_rate(market),
            "cpi_index": E.cpi_index(market),
        }
        frame = pd.DataFrame(series)
        frame.index = frame.index.astype(str)
        frame.index.name = "month"
        frame.reset_index().to_parquet(DERIVED / f"{market}.parquet", index=False)


if __name__ == "__main__":
    export()
    print(f"exported {len(MARKETS)} derived market files to {DERIVED}")
