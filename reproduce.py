#!/usr/bin/env python3
"""Reproduce the paper's complete results, figures, controls, and appendices.

Inputs: public and licensed files described in ``data/README.md`` and optional
credentials from the environment or ``.env.local``.
Outputs: CSV/JSON/PNG files below ``results/`` and LaTeX fragments below
``paper/_gen``.
Purpose: provide one deterministic entry point for H1--H3, temporal analyses,
main/state figures, macro controls, and appendix tables.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from common.paths import PAPER_GENERATED, RESULTS_ROOT


def file_hashes() -> dict[Path, str]:
    """Hash existing published outputs before or after a reproduction."""
    roots = (RESULTS_ROOT, PAPER_GENERATED)
    return {
        path.relative_to(REPO_ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


def compare_existing(
    before: dict[Path, str],
    after: dict[Path, str],
) -> list[Path]:
    """Return pre-existing outputs whose content changed during the rerun."""
    return sorted(path for path, digest in before.items() if after.get(path) != digest)


def run_step(label: str, module: str, *arguments: str) -> None:
    """Run one stage in a fresh process so engine wiring cannot leak."""
    print(f"\n=== {label} ===", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC_ROOT)
    environment.setdefault("MPLCONFIGDIR", "/tmp/four-quadrant-matplotlib")
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-fred",
        action="store_true",
        help="refresh public FRED snapshots (requires FRED_API_KEY)",
    )
    parser.add_argument(
        "--no-hash-check",
        action="store_true",
        help="do not compare regenerated files with the pre-existing outputs",
    )
    args = parser.parse_args(argv)

    before = file_hashes()
    refresh = ("--refresh-fred",) if args.refresh_fred else ()
    run_step("data fetch and input preflight", "data.fetch_inputs", *refresh)
    run_step("full-sample H1--H3", "analysis.run_full_sample")
    run_step("era and state analysis", "analysis.era_state")
    run_step("post-2000 panel", "analysis.post_2000_panel")
    run_step("era timing", "analysis.era_timing")
    run_step("main paper figures", "figures.figures_main")
    run_step("state figures", "figures.figures_state")
    run_step("macroeconomic controls", "analysis.macro_controls")
    run_step("appendix tables", "figures.appendix_tables")

    after = file_hashes()
    changed = [] if args.no_hash_check else compare_existing(before, after)
    if changed:
        print("\nReproduction hash check failed; pre-existing outputs changed:")
        for path in changed:
            print(f"  - {path}")
        return 1
    print(
        "\nReproduction complete. All pre-existing output hashes are unchanged; "
        f"{len(after) - len(before)} new output file(s) were created."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
