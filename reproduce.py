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
import importlib.util
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


def file_mtimes() -> dict[Path, float]:
    """Record the modification time of every existing published output."""
    roots = (RESULTS_ROOT, PAPER_GENERATED)
    return {
        path.relative_to(REPO_ROOT): path.stat().st_mtime
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


NUMERIC_OUTPUT_SUFFIXES = (".csv", ".json", ".tex")


def not_regenerated(
    before: dict[Path, float],
    after: dict[Path, float],
) -> list[Path]:
    """Pre-existing numeric outputs the rerun never rewrote (candidate orphans).

    A data file whose modification time is unchanged was not produced by any
    step, so its published values are not backed by the reproduction chain. The
    check is limited to numeric outputs; figures and working notes under
    ``results/`` are not part of the deterministic chain.
    """
    return sorted(
        path
        for path, mtime in before.items()
        if path.suffix in NUMERIC_OUTPUT_SUFFIXES
        and path in after
        and after[path] == mtime
    )


def run_step(
    label: str,
    module: str,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> None:
    """Run one stage in a fresh process so engine wiring cannot leak."""
    print(f"\n=== {label} ===", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC_ROOT)
    environment.setdefault("MPLCONFIGDIR", "/tmp/four-quadrant-matplotlib")
    if env:
        environment.update(env)
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


# Steps whose output text and file names depend on the manuscript language. Run
# once per language so both the French and the English generated fragments and
# figures are reproduced. run_full_sample and era_state are re-run under the
# English build for their language-variant figures only; the numbers they write
# are identical to the French build, so the CSV outputs are unchanged.
LANGUAGE_VARIANT_STEPS = (
    ("full-sample H1--H3 figures", "analysis.run_full_sample"),
    ("H1 forest figure (post-reselection)", "figures.figure_h1_forest"),
    ("era and state figures", "analysis.era_state"),
    ("main paper figures", "figures.figures_main"),
    ("state figures", "figures.figures_state"),
    ("inflation-state figure", "figures.figure_signal_inflation"),
    ("H3 relative-wealth figure", "figures.figure_h3_wealth"),
    ("appendix tables", "figures.appendix_tables"),
    ("provenance table", "figures.provenance_table"),
    ("allocation overview", "figures.figure_allocation_overview"),
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
    parser.add_argument(
        "--from-derived",
        action="store_true",
        help="reproduce from the committed data/derived/ layer, without WRDS access",
    )
    args = parser.parse_args(argv)

    if args.from_derived:
        os.environ["FOUR_QUADRANT_FROM_DERIVED"] = "1"

    before = file_hashes()
    before_mtimes = file_mtimes()
    refresh = ("--refresh-fred",) if args.refresh_fred else ()
    if not args.from_derived:
        if importlib.util.find_spec("data.fetch_inputs") is None:
            print(
                "The full WRDS/FRED acquisition layer (data.fetch_inputs and the "
                "data/*.py fetchers) is not part of the published repository. "
                "Reproduce offline from the shipped derived layer instead:\n"
                "    python reproduce.py --from-derived",
                file=sys.stderr,
            )
            return 2
        run_step("data fetch and input preflight", "data.fetch_inputs", *refresh)
    run_step("full-sample H1--H3", "analysis.run_full_sample")
    run_step("inference corrections (Holm, joint dependence, H3 reverse, RDD)", "analysis.inference_corrections")
    # Producer chain that must precede era_state: the macro controls write
    # h1_alpha_controls_comparison, which the post-2000 panel reconciles against;
    # the panel in turn writes post_2000_h1_market_means, which era_state reads
    # via load_alpha_data(). Both depend only on run_full_sample and the joint
    # dependence inference produced just above, so they run here.
    run_step("macroeconomic controls", "analysis.macro_controls")
    run_step("post-2000 panel", "analysis.post_2000_panel")
    run_step("era and state analysis", "analysis.era_state")
    # Reads the monthly signal states written by era_state; must precede the
    # inflation-state figure and the predictive mechanism check, which read its
    # inflation joint-inference output.
    run_step("supplementary inference (direct Sharpe, static era, reselection)", "analysis.supplementary_inference")
    # The H1 forest figure overlays the reference-reselection intervals written
    # by supplementary_inference, so it is rendered here rather than inside
    # run_full_sample. It runs after the audit in both language builds (this
    # French pass and the English language pass), so both figures are identical.
    run_step("H1 forest figure (post-reselection)", "figures.figure_h1_forest")
    run_step("deflated Sharpe and HLZ multiple-testing haircut", "analysis.deflated_sharpe")
    run_step("signal calendar counterfactuals", "analysis.signal_calendar_counterfactuals")
    run_step("era timing", "analysis.era_timing")
    run_step("main paper figures", "figures.figures_main")
    run_step("state figures", "figures.figures_state")
    run_step("inflation-state figure", "figures.figure_signal_inflation")
    run_step("H3 relative-wealth figure", "figures.figure_h3_wealth")
    run_step("appendix tables", "figures.appendix_tables")
    run_step("provenance table", "figures.provenance_table")
    run_step("supply-shock IRF (US complement)", "analysis.supply_shock_irf")
    # The allocation-overview figure places this paper's H1/H3 net Sharpe ratios
    # beside three constants quoted from the previous four-quadrant paper. It is
    # contextual and outside this paper's inference, but it is reproduced (here in
    # French, and in English via the language pass) so both manuscripts build from
    # a clean run.
    run_step("allocation overview (contextual)", "figures.figure_allocation_overview")
    # Referee-report robustness suite (report 1, block 1).
    run_step("US gold-parity robustness", "analysis.us_gold_parity_robustness")
    run_step("per-benchmark advantages", "analysis.per_benchmark_advantage")
    run_step("FX-hedged gold robustness", "analysis.fx_hedged_gold")
    run_step("H2 uniform-cost robustness", "analysis.h2_uniform_cost")
    run_step("H2 robust regression and benchmark selection", "analysis.h2_robustness")
    run_step("H2 gradient HC3 inference", "analysis.h2_gradient_hc3")
    run_step("H3 mapping family", "analysis.h3_mapping_family")
    run_step("predictive mechanism check", "analysis.mechanism_predictive")
    run_step("by-state real returns", "analysis.by_state_real_returns")
    run_step("turnover and breakeven", "analysis.turnover_breakeven")

    # The steps above build the French figures and fragments (the default
    # language). Regenerate their English variants so the English manuscript is
    # reproduced too; the underlying numbers are identical, only the on-figure and
    # caption text differ.
    for label, module in LANGUAGE_VARIANT_STEPS:
        run_step(f"{label} [en]", module, env={"FOUR_QUADRANT_FIG_LANG": "en"})

    after = file_hashes()
    stale = not_regenerated(before_mtimes, file_mtimes())
    if stale:
        print(
            "\nWarning: pre-existing outputs not rewritten by this run "
            "(no generating step; verify they are not orphaned):"
        )
        for path in stale:
            print(f"  - {path}")
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
