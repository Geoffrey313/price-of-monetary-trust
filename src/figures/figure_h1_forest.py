"""H1 forest figure, rendered after the reference-reselection audit.

Inputs: ``h1_per_market.csv`` and ``h1_reference_reselection_bootstrap.csv`` from
    the complete-sample result directory. The reselection file is produced by
    ``analysis.supplementary_inference``, which must therefore run first.
Outputs: ``h1_advantage_13_markets.png`` (French build) or the ``_en`` variant,
    in the complete-sample result directory.
Purpose: render the H1 forest once, after reselection, so the French and English
    figures are identical and stable across reruns. The rendering itself lives in
    ``analysis.run_full_sample.render_h1_forest``; this module only sequences it
    after the audit step.
"""
from __future__ import annotations

from analysis.run_full_sample import render_h1_forest


def main() -> int:
    render_h1_forest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
