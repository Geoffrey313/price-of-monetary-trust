"""Language switch for the figure and table generators.

The manuscript ships in two languages. The numbers are identical across them;
only the on-figure text differs. A single environment variable selects the
language of every label, legend, axis title and generated-table caption:

    FOUR_QUADRANT_FIG_LANG=fr   (default) French labels, historical file names
    FOUR_QUADRANT_FIG_LANG=en             English labels, ``_en`` file names

Call sites wrap each user-visible string in ``L(french, english)`` and each
output path in ``figpath(path)``. Nothing else changes: the data, the styling
and the layout are shared. Keeping both languages behind one switch avoids a
second, drifting copy of the plotting code.
"""
from __future__ import annotations

import os
from pathlib import Path

LANG = os.environ.get("FOUR_QUADRANT_FIG_LANG", "fr").strip().lower()


def L(fr: str, en: str) -> str:
    """Return the English string under the English build, the French otherwise."""
    return en if LANG == "en" else fr


# English market display names, keyed by ISO 3166 alpha-3 code. The generators
# hold French names (in ``run_full_sample.NAMES`` and the ``country`` column of
# the per-market CSVs); under the English build the axis labels use these.
MARKET_NAMES_EN: dict[str, str] = {
    "AUS": "Australia",
    "BEL": "Belgium",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "DEU": "Germany",
    "ESP": "Spain",
    "FRA": "France",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "USA": "United States",
    "ZAF": "South Africa",
}


def market_name(code: str, fallback: str) -> str:
    """Return the English market name under the English build, else the French one.

    ``code`` is the ISO alpha-3 market code; ``fallback`` is the French display
    name the generator already holds, used verbatim under the French build and as
    a safety net for any code missing from the map.
    """
    if LANG == "en":
        return MARKET_NAMES_EN.get(code, fallback)
    return fallback


def figpath(path: str | Path) -> Path:
    """Map an output path to its language variant.

    Under the French build the path is returned unchanged, preserving the
    historical file names. Under the English build a trailing ``_fr`` is dropped
    and ``_en`` is appended, so the two suites never overwrite each other:
    ``h1_wealth_13_markets.png``  -> ``h1_wealth_13_markets_en.png`` and
    ``allocation_methods_overview_fr.png`` -> ``allocation_methods_overview_en.png``.
    """
    p = Path(path)
    if LANG != "en":
        return p
    stem = p.stem
    if stem.endswith("_fr"):
        stem = stem[:-3]
    return p.with_name(f"{stem}_en{p.suffix}")
