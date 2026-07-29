"""Build the per-market data-provenance appendix table (referee comment M6).

Inputs: the committed derived layer ``data/derived/{market}.parquet`` (columns
``month``, ``eq``, ``long_yield``, ``gold_local``, ``short_rate``) and the
in-memory engine for the cross-check.
Outputs: ``paper/_gen/appendix_provenance_fr.tex`` (or ``_en`` under
``FOUR_QUADRANT_FIG_LANG=en``) and the machine-readable
``results/complete-sample-rerun-2026-07-26/provenance_per_market.csv``.
Purpose: document, for each of the thirteen markets, the equity source, the
first and last usable month of the four raw series, the month the bond/gold
signal completes its 84-month warm-up, the evaluation window, and which series
binds each end of that window.

The window is reconstructed exactly as the engine forms it: the bond total
return needs two consecutive yields, so its first month is one month after the
first yield; the 84-month warm-up runs on the bond-wealth/gold ratio only; the
evaluation index is the intersection of equities, bond returns, gold returns,
the short rate and the warmed-up signal. Every reconstructed span is asserted
against the engine's own dry-run span before anything is written.

Run twice for the two languages:
``FOUR_QUADRANT_FROM_DERIVED=1 PYTHONPATH=src python -m figures.provenance_table``
and the same with ``FOUR_QUADRANT_FIG_LANG=en``.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from analysis import run_full_sample as rerun
from common.names import MARKETS, NAMES
from common.paths import DATA_ROOT, FULL_SAMPLE_RESULTS, PAPER_GENERATED, REPO_ROOT
from data.treasury_total_return import construct_monthly_tr
from engine import engine as E
from figures.i18n import L, figpath, market_name

WARMUP = 84  # months of the bond/gold ratio consumed by the signal, engine [S2]

EQUITY_SOURCES = {
    "CHE": ("SNB SPI (natif)", "SNB SPI (native)"),
    "USA": ("Shiller (natif)", "Shiller (native)"),
    "CAN": ("StatCan (natif, DY fin 2016:8)", "StatCan (native, DY ends 2016:8)"),
}
COMPUSTAT = ("reconstruction Compustat", "Compustat reconstruction")

# Binding-series labels, keyed by the CSV (English) value.
BINDERS = {
    "signal warm-up": ("amorçage du signal", "signal warm-up"),
    "equities": ("actions", "equities"),
    "yields": ("rendements", "yields"),
    "gold": ("or", "gold"),
    "short rate": ("taux court", "short rate"),
    "several": ("plusieurs", "several"),
}


def month_label(period: pd.Period) -> str:
    """Format a month as in the manuscript's tables, e.g. ``1986:1``."""
    return f"{period.year}:{period.month}"


def binding(candidates: dict[str, pd.Period], bound: pd.Period) -> str:
    """Name the series whose availability equals the binding constraint."""
    hits = [name for name, month in candidates.items() if month == bound]
    if not hits:
        raise AssertionError(f"no candidate matches the bound {bound}: {candidates}")
    return hits[0] if len(hits) == 1 else "several"


def market_provenance(mkt: str) -> dict:
    """Reconstruct one market's series spans and evaluation window."""
    frame = pd.read_parquet(DATA_ROOT / "derived" / f"{mkt}.parquet")
    frame.index = pd.PeriodIndex(frame["month"], freq="M")
    eq = frame["eq"].dropna()
    y = frame["long_yield"].dropna()
    bond = construct_monthly_tr(y.to_timestamp().rename("y"))
    bond_tr = pd.Series(bond["tr"].values, index=y.index)
    bond_w = pd.Series(bond["wealth"].values, index=y.index)
    gold = frame["gold_local"].dropna()
    bill = frame["short_rate"].dropna()

    # Engine [S2]: the warm-up counts observations of the bond-wealth/gold ratio only.
    ratio = (bond_w / gold).dropna()
    if len(ratio) < WARMUP:
        raise AssertionError(f"{mkt}: ratio has fewer than {WARMUP} observations")
    ratio_ready = ratio.index[WARMUP - 1]
    signal_index = ratio.index[WARMUP - 1:]

    # Engine evaluation index: intersection of the four series and the signal.
    # ``bond_tr`` and ``gold.pct_change()`` keep their first (NaN) month in the
    # index exactly as the engine does; the warm-up starts far later anyway.
    common = (
        eq.index.intersection(bond_tr.index)
        .intersection(gold.pct_change().index)
        .intersection(bill.index)
        .intersection(signal_index)
    )
    start, end = common.min(), common.max()

    start_candidates = {
        "signal warm-up": ratio_ready,
        "equities": eq.index.min(),
        "short rate": bill.index.min(),
    }
    end_candidates = {
        "equities": eq.index.max(),
        "yields": y.index.max(),
        "gold": gold.index.max(),
        "short rate": bill.index.max(),
    }
    if max(start_candidates.values()) != start or min(end_candidates.values()) != end:
        raise AssertionError(f"{mkt}: window {start}..{end} not explained by the candidates")

    return {
        "market": mkt,
        "equity_source": EQUITY_SOURCES.get(mkt, COMPUSTAT),
        "equities_first": eq.index.min(),
        "equities_last": eq.index.max(),
        "bonds_first": bond_tr.dropna().index.min(),  # two consecutive yields needed
        "bonds_last": y.index.max(),
        "gold_first": gold.index.min(),
        "gold_last": gold.index.max(),
        "short_rate_first": bill.index.min(),
        "short_rate_last": bill.index.max(),
        "ratio_ready": ratio_ready,
        "window_start": start,
        "window_end": end,
        "window_months": int(len(common)),
        "binds_start": binding(start_candidates, start),
        "binds_end": binding(end_candidates, end),
    }


def reconcile_with_engine(rows: list[dict]) -> None:
    """Require every reconstructed window to equal the engine's dry-run span."""
    for row in rows:
        info = E.run_market(row["market"], real_gate=False, bootstrap_draws=0, dry=True)
        span = f"{row['window_start']}..{row['window_end']}"
        if span != info["span"] or row["window_months"] != info["months"]:
            raise AssertionError(
                f"{row['market']}: reconstructed {span} ({row['window_months']} months) "
                f"disagrees with engine {info['span']} ({info['months']} months)"
            )
    fixed = {row["market"]: row for row in rows}
    checks = (
        ("USA", "window_start", pd.Period("1960-04", "M")),
        ("USA", "window_end", pd.Period("2024-09", "M")),
        ("USA", "window_months", 774),
        ("DEU", "window_start", pd.Period("1986-01", "M")),
        ("CAN", "window_end", pd.Period("2016-08", "M")),
    )
    for market, field, expected in checks:
        if fixed[market][field] != expected:
            raise AssertionError(f"{market}.{field}={fixed[market][field]} != {expected}")


def write_csv(rows: list[dict]) -> Path:
    """Write the language-independent machine-readable audit (English labels)."""
    records = []
    for row in rows:
        record = dict(row)
        record["equity_source"] = row["equity_source"][1]
        for key, value in record.items():
            if isinstance(value, pd.Period):
                record[key] = str(value)
        records.append(record)
    path = FULL_SAMPLE_RESULTS / "provenance_per_market.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def write_tex(rows: list[dict]) -> Path:
    """Write the LaTeX fragment (booktabs tabular inside a \\resizebox)."""
    path = figpath(PAPER_GENERATED / "appendix_provenance_fr.tex")

    def span(first: pd.Period, last: pd.Period) -> str:
        return f"{month_label(first)}--{month_label(last)}"

    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        L(
            (
                r" & & \multicolumn{4}{c}{Disponibilité (premier--dernier mois)} & & "
                r"\multicolumn{2}{c}{Fenêtre d'évaluation} & "
                r"\multicolumn{2}{c}{Série contraignante} \\"
            ),
            (
                r" & & \multicolumn{4}{c}{Availability (first--last month)} & & "
                r"\multicolumn{2}{c}{Evaluation window} & "
                r"\multicolumn{2}{c}{Binding series} \\"
            ),
        ),
        r"\cmidrule(lr){3-6}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        L(
            (
                r"Marché & Source actions & Actions & Obligations & Or & Taux court "
                r"& Signal prêt & Début & Fin & Début & Fin \\"
            ),
            (
                r"Market & Equity source & Equities & Bonds & Gold & Short rate "
                r"& Ratio ready & Start & End & Start & End \\"
            ),
        ),
        r"\midrule",
    ]
    for row in rows:
        cells = [
            market_name(row["market"], NAMES[row["market"]]),
            L(*row["equity_source"]),
            span(row["equities_first"], row["equities_last"]),
            span(row["bonds_first"], row["bonds_last"]),
            span(row["gold_first"], row["gold_last"]),
            span(row["short_rate_first"], row["short_rate_last"]),
            month_label(row["ratio_ready"]),
            month_label(row["window_start"]),
            month_label(row["window_end"]),
            L(*BINDERS[row["binds_start"]]),
            L(*BINDERS[row["binds_end"]]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\par\vspace{4pt}",
            r"\begin{minipage}{0.95\textwidth}\footnotesize",
            L(
                (
                    "Note : les rendements à dix ans et le taux court proviennent de FRED "
                    "(OCDE/IFS) pour chaque marché ; l'or est le prix USD converti au taux de "
                    "change du marché. Les obligations débutent un mois après le premier "
                    "rendement (deux rendements consécutifs sont requis pour le rendement "
                    "total). « Signal prêt » est le premier mois disposant de 84 observations "
                    "du rapport richesse obligataire/or (amorçage du signal). La fenêtre "
                    "d'évaluation commence au plus tardif de l'amorçage du signal, des actions "
                    "et du taux court, et finit à la plus courte des séries ; « plusieurs » "
                    "signifie que les rendements, l'or et le taux court s'arrêtent le même "
                    "mois (fin de la photographie FRED)."
                ),
                (
                    "Note: the ten-year yields and the short rate come from FRED (OECD/IFS) "
                    "for every market; gold is the USD price converted at the market's "
                    "exchange rate. Bonds start one month after the first yield (two "
                    "consecutive yields are required for the total return). \"Ratio ready\" "
                    "is the first month with 84 observations of the bond-wealth/gold ratio "
                    "(the signal warm-up). The evaluation window starts at the latest of the "
                    "signal warm-up, equities and the short rate, and ends at the shortest "
                    "series; \"several\" means yields, gold and the short rate stop in the "
                    "same month (the end of the FRED snapshot)."
                ),
            ),
            r"\end{minipage}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    rerun.wire_everything()
    rows = [market_provenance(mkt) for mkt in MARKETS]
    # Manuscript row order: French names, accent-insensitive (États-Unis after Espagne).
    rows.sort(
        key=lambda row: unicodedata.normalize("NFKD", NAMES[row["market"]])
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    reconcile_with_engine(rows)

    PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(rows)
    tex_path = write_tex(rows)

    for row in rows:
        print(
            f"{row['market']}: {row['window_start']}..{row['window_end']} "
            f"({row['window_months']} months) start={row['binds_start']} "
            f"end={row['binds_end']}"
        )
    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {tex_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
