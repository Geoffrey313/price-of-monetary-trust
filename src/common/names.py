"""Canonical market sets and labels used throughout the paper.

Inputs: none.
Outputs: none; this module only exposes stable constants.
Purpose: keep the thirteen-market H1--H3 universe and its labels identical
across the engine, analyses, tables, and figures.
"""
from __future__ import annotations

MARKETS: tuple[str, ...] = (
    "AUS",
    "BEL",
    "CAN",
    "FRA",
    "DEU",
    "JPN",
    "NLD",
    "NOR",
    "ZAF",
    "ESP",
    "CHE",
    "GBR",
    "USA",
)

EURO: frozenset[str] = frozenset({"BEL", "FRA", "DEU", "NLD", "ESP"})
NON_EURO: tuple[str, ...] = tuple(market for market in MARKETS if market not in EURO)

NAMES: dict[str, str] = {
    "AUS": "Australie",
    "BEL": "Belgique",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Allemagne",
    "JPN": "Japon",
    "NLD": "Pays-Bas",
    "NOR": "Norvège",
    "ZAF": "Afrique du Sud",
    "ESP": "Espagne",
    "CHE": "Suisse",
    "GBR": "Royaume-Uni",
    "USA": "États-Unis",
}

ENGLISH_NAMES: dict[str, str] = {
    "AUS": "Australia",
    "BEL": "Belgium",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Germany",
    "JPN": "Japan",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "ZAF": "South Africa",
    "ESP": "Spain",
    "CHE": "Switzerland",
    "GBR": "United Kingdom",
    "USA": "United States",
}
