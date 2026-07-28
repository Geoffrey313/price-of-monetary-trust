"""Prepare and validate the input layer for the published reproduction.

Inputs: public snapshots in ``data/raw`` and ``data/fred/raw`` plus licensed
WRDS/CRSP parquet reconstructions in ``data/recon``.
Outputs: none in offline mode; with ``--refresh-fred`` it refreshes public FRED
JSON snapshots and their manifest.
Purpose: implement the data-fetch/preflight stage of ``reproduce.py`` while
failing explicitly when licensed inputs are unavailable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.paths import EVIDENCE_DATA, FRED_RAW_DATA, RAW_DATA, RECON_DATA
from data import fetch_fred

REQUIRED_EVIDENCE = (
    "10100125.csv",
    "snb_spi.csv",
    "wu018a.csv",
    "wu018a_2011.csv",
)
REQUIRED_FRED_FILES = (
    "gold_spot_usd.csv",
    "sp500_total_return.xls",
)
REQUIRED_RECONSTRUCTIONS = (
    "AUS_recon.parquet",
    "BEL_recon.parquet",
    "DEU_recon.parquet",
    "ESP_recon.parquet",
    "FRA_recon.parquet",
    "GBR_recon.parquet",
    "JPN_topix_recon.parquet",
    "NLD_recon.parquet",
    "NOR_recon.parquet",
    "ZAF_recon.parquet",
    "USA_energy_crsp.parquet",
    "sector_map.parquet",
)


def _missing(directory: Path, names: tuple[str, ...]) -> list[Path]:
    return [directory / name for name in names if not (directory / name).exists()]


def validate_inputs() -> None:
    """Fail with a complete list of missing public or licensed input files."""
    missing = [
        *_missing(EVIDENCE_DATA, REQUIRED_EVIDENCE),
        *_missing(FRED_RAW_DATA, REQUIRED_FRED_FILES),
        *_missing(RECON_DATA, REQUIRED_RECONSTRUCTIONS),
    ]
    if not RAW_DATA.exists() or not any(RAW_DATA.glob("*.json")):
        missing.append(RAW_DATA / "<FRED JSON snapshots>")
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The reproduction input layer is incomplete. Missing:\n" + listing
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-fred",
        action="store_true",
        help="refresh the registered public FRED snapshots before validation",
    )
    args = parser.parse_args(argv)
    if args.refresh_fred:
        fetch_fred.main()
    validate_inputs()
    print(
        "input preflight passed: public snapshots, evidence, and licensed "
        "reconstructions are available"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
