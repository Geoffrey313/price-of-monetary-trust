"""Build and audit the paper's appendix descriptive and notation tables.

Inputs: frozen engine inputs plus published primary and macro-control CSV files.
Outputs: descriptive/notation CSV audits and LaTeX fragments in ``paper/_gen``.
Purpose: make the appendix variable definitions and descriptive statistics
fully reproducible from the same data as H1--H3.

The descriptive table is rebuilt from the same frozen 13-market engine and
published result panels as the paper.  Existing N/mean/std/min/median/max
values are asserted before P25 and P75 are written.
"""
from __future__ import annotations
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

from analysis import macro_controls as macro
from analysis import run_full_sample as rerun
from figures.i18n import L, LANG, figpath
from common.paths import (
    FULL_SAMPLE_RESULTS,
    MACRO_RESULTS,
    PAPER_GENERATED,
    REPO_ROOT,
)

RESULTS = FULL_SAMPLE_RESULTS
CONTROLS = MACRO_RESULTS
GENERATED = PAPER_GENERATED


PANELS = {
    "A": L(
        "Rendements mensuels bruts utilisés, en pour cent",
        "Raw monthly returns used, in percent",
    ),
    "B": L(
        "Rendements mensuels des portefeuilles calculés, en pour cent",
        "Computed monthly portfolio returns, in percent",
    ),
    "C": L("Variables calculées des tests", "Computed test variables"),
    "D": L(
        "Variables macroéconomiques exploratoires du panel H1",
        "Exploratory macroeconomic variables of the H1 panel",
    ),
}

VARIABLES = [
    ("A", "equity", L("Actions", "Equities")),
    ("A", "bond", L("Obligations", "Bonds")),
    ("A", "gold", L("Or en monnaie locale", "Gold in local currency")),
    ("A", "bill", L("Taux court", "Short rate")),
    ("B", "strategy", L("Bascule binaire", "Binary switch")),
    ("B", "6040", L("60/40 local", "Local 60/40")),
    ("B", "permanent", L("Portefeuille permanent", "Permanent portfolio")),
    ("B", "matched", L("Statique comparable", "Matched static")),
    (
        "C",
        "d",
        L(
            r"$d_{i,t}$, écart de rendement, en points de pourcentage",
            r"$d_{i,t}$, return gap, in percentage points",
        ),
    ),
    ("C", "z", L(r"$z_{i,t}$, écart ajusté du risque", r"$z_{i,t}$, risk-adjusted gap")),
    (
        "C",
        "energy_rank",
        L(r"$E_i$, rang d'exposition énergétique", r"$E_i$, energy exposure rank"),
    ),
    ("C", "h2_advantage", L(r"$A_i^{E}$, écart de Sharpe H2", r"$A_i^{E}$, H2 Sharpe advantage")),
    (
        "C",
        "h3_delta",
        L(
            r"$\Delta_i^{\mathrm{H3}}$, graduée moins binaire",
            r"$\Delta_i^{\mathrm{H3}}$, graded minus binary",
        ),
    ),
    (
        "D",
        "oil_score",
        L(
            r"Position pétrolière nette, transformation $\operatorname{asinh}$",
            r"Net oil position, $\operatorname{asinh}$ transformation",
        ),
    ),
    ("D", "petroleum_surplus", L("Surplus pétrolier, indicatrice", "Petroleum surplus, indicator")),
    ("D", "balance_sheet", L("Expansion du bilan, points de PIB", "Balance-sheet expansion, points of GDP")),
    ("D", "broad_money", L("Croissance de la monnaie large, en pour cent", "Broad-money growth, in percent")),
]

# Values printed in the manuscript before adding the quartiles.  H3's maximum
# was deliberately printed to four decimals because it is close to zero.
PUBLISHED = {
    "equity": (6264, 0.855, 4.700, -38.547, 1.222, 25.564),
    "bond": (6264, 0.492, 1.751, -10.216, 0.510, 10.183),
    "gold": (6264, 0.632, 3.880, -16.692, 0.362, 48.420),
    "bill": (6264, 0.347, 0.334, -0.078, 0.289, 1.834),
    "strategy": (6264, 0.633, 1.604, -12.102, 0.653, 14.726),
    "6040": (6264, 0.710, 2.976, -25.389, 0.908, 16.887),
    "permanent": (6264, 0.583, 1.592, -10.282, 0.591, 15.821),
    "matched": (6264, 0.566, 1.485, -11.565, 0.600, 7.662),
    "d": (6264, 0.024, 0.980, -6.837, 0.003, 11.400),
    "z": (5796, 0.025, 0.508, -3.432, 0.003, 6.808),
    "energy_rank": (13, 7.000, 3.518, 1.000, 7.000, 11.500),
    "h2_advantage": (13, -0.149, 0.178, -0.686, -0.107, 0.012),
    "h3_delta": (13, -0.043, 0.022, -0.067, -0.050, -0.0003),
    "oil_score": (5486, -0.386, 0.981, -0.881, -0.861, 3.413),
    "petroleum_surplus": (5486, 0.174, 0.379, 0.000, 0.000, 1.000),
    "balance_sheet": (3960, 1.154, 5.782, -30.113, 0.438, 27.872),
    "broad_money": (4119, 5.617, 3.908, -4.745, 5.549, 24.098),
}


def pooled_h1_series() -> dict[str, pd.Series]:
    """Rebuild the eight pooled asset/portfolio return series without bootstrap."""
    rerun.wire_everything()
    rerun.E.run_market.return_series = True
    frames: dict[str, list[pd.Series]] = {
        key: []
        for key in (
            "equity",
            "bond",
            "gold",
            "bill",
            "strategy",
            "6040",
            "permanent",
            "matched",
        )
    }
    engine_names = {
        "equity": "eq",
        "bond": "bond",
        "gold": "gold",
        "bill": "bill",
        "strategy": "strat",
        "6040": "b6040",
        "permanent": "pp",
        "matched": "matched",
    }
    for market in rerun.MARKETS:
        result = rerun.E.run_market(
            market,
            real_gate=False,
            bootstrap_draws=0,
        )
        series = result["series"]
        for key, engine_name in engine_names.items():
            values = series[engine_name].copy()
            values.index = pd.MultiIndex.from_arrays(
                [
                    np.repeat(market, len(values)),
                    values.index.astype(str),
                ],
                names=["market", "month"],
            )
            frames[key].append(values)
    pooled = {
        key: pd.concat(parts).sort_index().astype(float) * 100
        for key, parts in frames.items()
    }
    if any(len(values) != 6264 for values in pooled.values()):
        counts = {key: len(values) for key, values in pooled.items()}
        raise RuntimeError(f"Unexpected pooled H1 counts: {counts}")
    return pooled


def test_and_macro_series() -> dict[str, pd.Series]:
    """Load the published test variables and rebuild the attached macro controls."""
    h1_return = pd.read_csv(RESULTS / "h1_return_panel.csv")
    h1_risk = pd.read_csv(RESULTS / "h1_risk_adjusted_panel.csv")
    h2 = pd.read_csv(RESULTS / "h2_per_market.csv")
    h3 = pd.read_csv(RESULTS / "h3_per_market.csv")
    controlled = macro.attach_controls(h1_risk)
    return {
        "d": h1_return["d"].astype(float) * 100,
        "z": h1_risk["z"].astype(float),
        "energy_rank": h2["exposure_mean_rank"].astype(float),
        "h2_advantage": h2["augmentation_advantage"].astype(float),
        "h3_delta": h3["difference_amplitude_minus_binary"].astype(float),
        "oil_score": controlled["oil_score"].dropna().astype(float),
        "petroleum_surplus": controlled["petroleum_surplus"].dropna().astype(float),
        "balance_sheet": controlled["balance_sheet_change_pp"].dropna().astype(float),
        "broad_money": controlled["broad_money_growth_lag2"].dropna().astype(float),
    }


def describe(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    return {
        "observations": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def reconcile(frame: pd.DataFrame) -> None:
    """Require the rebuilt table to reproduce every previously printed statistic."""
    for row in frame.itertuples(index=False):
        expected = PUBLISHED[row.variable_key]
        actual = (
            row.observations,
            row.mean,
            row.std,
            row.min,
            row.median,
            row.max,
        )
        if actual[0] != expected[0]:
            raise AssertionError(
                f"{row.variable_key}: N={actual[0]} instead of {expected[0]}"
            )
        for name, observed, reference in zip(
            ("mean", "std", "min", "median", "max"),
            actual[1:],
            expected[1:],
        ):
            tolerance = (
                0.00005
                if row.variable_key == "h3_delta" and name == "max"
                else 0.0005
            )
            if abs(float(observed) - float(reference)) > tolerance + 1e-12:
                raise AssertionError(
                    f"{row.variable_key}/{name}: {observed} does not reconcile "
                    f"with published {reference}"
                )


def format_n(value: int) -> str:
    grouped = f"{value:,}"
    # English uses the comma thousands separator; French the thin non-breaking space.
    return grouped if LANG == "en" else grouped.replace(",", "~")


def format_number(value: float) -> str:
    rounded = 0.0 if abs(value) < 0.0005 else value
    if LANG == "en":
        # English decimal point; negatives in math mode for a proper minus sign.
        text = f"{rounded:.3f}"
        return f"${text}$" if rounded < 0 else text
    text = f"{rounded:.3f}".replace(".", ",")
    return f"${text.replace(',', '{,}')}$" if rounded < 0 else text


def write_descriptive_tex(frame: pd.DataFrame) -> Path:
    path = figpath(GENERATED / "appendix_variable_descriptives_fr.tex")
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        L(
            (
                r"Panneau & Variable & Obs. & Moyenne & Écart-type & Minimum "
                r"& P25 & Médiane & P75 & Maximum \\"
            ),
            (
                r"Panel & Variable & Obs. & Mean & Std. dev. & Minimum "
                r"& P25 & Median & P75 & Maximum \\"
            ),
        ),
        r"\midrule",
    ]
    for panel, title in PANELS.items():
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{panel}. {title}}}}} \\")
        for row in frame[frame["panel"] == panel].itertuples(index=False):
            numbers = " & ".join(
                [
                    format_n(row.observations),
                    format_number(row.mean),
                    format_number(row.std),
                    format_number(row.min),
                    format_number(row.p25),
                    format_number(row.median),
                    format_number(row.p75),
                    format_number(row.max),
                ]
            )
            lines.append(rf"& {row.variable_tex} & {numbers} \\")
        if panel != "D":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def notation_rows() -> list[dict[str, str]]:
    """Return the source mapping required by the appendix notation table."""
    return [
        {
            "name_tex": L(r"Richesse obligataire, $W^b_{i,t}$", r"Bond wealth, $W^b_{i,t}$"),
            "symbol_tex": r"W^b_{i,t}",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": r"$W^b_{i,t}=W^b_{i,t-1}(1+r^b_{i,t})$",
            "definition_fr": L(
                (
                    "Indice de richesse obtenu en composant les rendements totaux "
                    "des obligations d'État locales à dix ans."
                ),
                (
                    "Wealth index obtained by compounding the total returns of "
                    "ten-year local government bonds."
                ),
            ),
            "source_fr": L(
                "FRED/OCDE et FRED/IFS; calcul des auteurs.",
                "FRED/OECD and FRED/IFS; authors' calculation.",
            ),
        },
        {
            "name_tex": L(r"Or en dollars, $G^{\$}_t$", r"Gold in dollars, $G^{\$}_t$"),
            "symbol_tex": r"G^{\$}_t",
            "field_tex": r"\texttt{gold\_spot\_usd}",
            "formula_tex": L(r"Niveau observé $G^{\$}_t$", r"Observed level $G^{\$}_t$"),
            "definition_fr": L(
                "Prix mensuel d'une once d'or exprimé en dollars.",
                "Monthly price of one ounce of gold expressed in dollars.",
            ),
            "source_fr": L(
                "Photographie Datahub, série dérivée de la LBMA, 1833--2026.",
                "Datahub snapshot, series derived from the LBMA, 1833--2026.",
            ),
        },
        {
            "name_tex": L(r"Taux de change, $\mathrm{FX}_{i,t}$", r"Exchange rate, $\mathrm{FX}_{i,t}$"),
            "symbol_tex": r"\mathrm{FX}_{i,t}",
            "field_tex": r"\texttt{EX*US}, \texttt{EXUSEU}",
            "formula_tex": L(
                r"Unités de monnaie $i$ pour un dollar",
                r"Units of currency $i$ per dollar",
            ),
            "definition_fr": L(
                (
                    "Conversion mensuelle du dollar en monnaie locale, avec raccords "
                    "officiels des monnaies prédécesseures de l'euro."
                ),
                (
                    "Monthly conversion of the dollar into local currency, with "
                    "official splices of the euro's predecessor currencies."
                ),
            ),
            "source_fr": "Federal Reserve Board via FRED.",
        },
        {
            "name_tex": L(r"Or local, $G_{i,t}$", r"Local gold, $G_{i,t}$"),
            "symbol_tex": r"G_{i,t}",
            "field_tex": L(
                r"\texttt{gold\_spot\_usd} et \texttt{EX*US}",
                r"\texttt{gold\_spot\_usd} and \texttt{EX*US}",
            ),
            "formula_tex": r"$G_{i,t}=G^{\$}_t\mathrm{FX}_{i,t}$",
            "definition_fr": L(
                "Prix de l'or exprimé dans la monnaie du marché.",
                "Gold price expressed in the market's currency.",
            ),
            "source_fr": L(
                "Calcul des auteurs à partir de l'or USD et du change.",
                "Authors' calculation from USD gold and the exchange rate.",
            ),
        },
        {
            "name_tex": L(
                r"Taux long et duration, $y_{i,t}$ et $D(y)$",
                r"Long rate and duration, $y_{i,t}$ and $D(y)$",
            ),
            "symbol_tex": r"y_{i,t}, D(y)",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": L(
                (
                    r"Champ source divisé par 100; duration d'une "
                    r"obligation au pair à dix ans"
                ),
                r"Source field divided by 100; duration of a ten-year par bond",
            ),
            "definition_fr": L(
                (
                    "Rendement à dix ans en fraction annuelle dans les équations et "
                    "sensibilité de premier ordre du prix obligataire, en années."
                ),
                (
                    "Ten-year yield as an annual fraction in the equations and "
                    "first-order sensitivity of the bond price, in years."
                ),
            ),
            "source_fr": L(
                "FRED/OCDE et FRED/IFS; calcul des auteurs.",
                "FRED/OECD and FRED/IFS; authors' calculation.",
            ),
        },
        {
            "name_tex": L(r"Taux sans risque, $r^f_{i,t}$", r"Risk-free rate, $r^f_{i,t}$"),
            "symbol_tex": r"r^f_{i,t}",
            "field_tex": (
                r"\texttt{IR3TIB01*M156N}, \texttt{TB3MS}, "
                r"\texttt{INTGSTZAM193N}"
            ),
            "formula_tex": L(
                r"Taux annuel publié divisé par 1\,200",
                r"Published annual rate divided by 1\,200",
            ),
            "definition_fr": L(
                "Rendement mensuel du taux court local publié en pourcentage annuel.",
                "Monthly return of the local short rate published in annual percent.",
            ),
            "source_fr": L("FRED/OCDE et FRED/IFS.", "FRED/OECD and FRED/IFS."),
        },
        {
            "name_tex": L(r"Rendement obligataire, $r^b_{i,t}$", r"Bond return, $r^b_{i,t}$"),
            "symbol_tex": r"r^b_{i,t}",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": L(r"Équation~\eqref{eq:bondtr}", r"Equation~\eqref{eq:bondtr}"),
            "definition_fr": L(
                (
                    "Rendement total mensuel approché par le portage et la variation "
                    "de prix à duration modifiée."
                ),
                (
                    "Monthly total return approximated by carry and the "
                    "modified-duration price change."
                ),
            ),
            "source_fr": L(
                "FRED/OCDE et FRED/IFS; calcul des auteurs.",
                "FRED/OECD and FRED/IFS; authors' calculation.",
            ),
        },
        {
            "name_tex": L(r"Rendement de stratégie, $r^s_{i,t}$", r"Strategy return, $r^s_{i,t}$"),
            "symbol_tex": r"r^s_{i,t}",
            "field_tex": r"\texttt{eq}, \texttt{bond}, \texttt{gold}, \texttt{bill}",
            "formula_tex": L(r"Équation~\eqref{eq:netret}", r"Equation~\eqref{eq:netret}"),
            "definition_fr": L(
                "Rendement mensuel net de la règle binaire, après coûts de rotation.",
                "Monthly return of the binary rule, net of turnover costs.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Rendement d'actif, $r^a_{i,t}$", r"Asset return, $r^a_{i,t}$"),
            "symbol_tex": r"r^a_{i,t}",
            "field_tex": r"\texttt{eq}, \texttt{bond}, \texttt{gold}, \texttt{bill}, \texttt{energy}",
            "formula_tex": L(
                r"Variation mensuelle simple de l'actif $a$",
                r"Simple monthly change of asset $a$",
            ),
            "definition_fr": L(
                "Rendement mensuel de la poche actions, obligations, or, taux ou énergie.",
                "Monthly return of the equity, bond, gold, bill or energy pocket.",
            ),
            "source_fr": L(
                "Sources de marché décrites dans la section des données.",
                "Market sources described in the data section.",
            ),
        },
        {
            "name_tex": L(r"Rapport monétaire, $R_{i,t}$", r"Monetary ratio, $R_{i,t}$"),
            "symbol_tex": r"R_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": r"$R_{i,t}=W^b_{i,t}/G_{i,t}$",
            "definition_fr": L(
                "Richesse obligataire relativement au prix local de l'or.",
                "Bond wealth relative to the local gold price.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Signal binaire, $s_{i,t}$", r"Binary signal, $s_{i,t}$"),
            "symbol_tex": r"s_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": L(r"Équation~\eqref{eq:signal}", r"Equation~\eqref{eq:signal}"),
            "definition_fr": L(
                (
                    "Indicatrice valant un lorsque le rapport monétaire atteint ou "
                    "dépasse sa moyenne mobile de 84 mois."
                ),
                (
                    "Indicator equal to one when the monetary ratio reaches or "
                    "exceeds its 84-month moving average."
                ),
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Poids de portefeuille, $w^a_{i,t}$", r"Portfolio weight, $w^a_{i,t}$"),
            "symbol_tex": r"w^a_{i,t}",
            "field_tex": r"\texttt{signal}",
            "formula_tex": L(r"Équation~\eqref{eq:weights}", r"Equation~\eqref{eq:weights}"),
            "definition_fr": L(
                "Poids cible de l'actif $a$, appliqué avec un mois de retard.",
                "Target weight of asset $a$, applied with a one-month lag.",
            ),
            "source_fr": L("Règle de portefeuille préspécifiée.", "Pre-specified portfolio rule."),
        },
        {
            "name_tex": L(
                r"Poids avant transaction, $\widetilde w^a_{i,t}$",
                r"Pre-trade weight, $\widetilde w^a_{i,t}$",
            ),
            "symbol_tex": r"\widetilde w^a_{i,t}",
            "field_tex": r"\texttt{previous\_weight}, \texttt{asset\_return}",
            "formula_tex": (
                r"$\widetilde w^a_{i,t}=w^a_{i,t-1}(1+r^a_{i,t-1})/"
                r"\sum_jw^j_{i,t-1}(1+r^j_{i,t-1})$"
            ),
            "definition_fr": L(
                "Poids auquel la poche dérive avant le rééquilibrage du mois.",
                "Weight to which the pocket drifts before the month's rebalancing.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Coût unitaire, $c$", r"Unit cost, $c$"),
            "symbol_tex": r"c",
            "field_tex": r"\texttt{COST\_ONEWAY}",
            "formula_tex": L(r"$c=0{,}004$", r"$c=0.004$"),
            "definition_fr": L(
                "Coût proportionnel aller simple appliqué au poids traité.",
                "Proportional one-way cost applied to the traded weight.",
            ),
            "source_fr": L(
                "Hypothèse préspécifiée de 40 points de base.",
                "Pre-specified assumption of 40 basis points.",
            ),
        },
        {
            "name_tex": L(r"Ratio de Sharpe, $\mathrm{SR}_i$", r"Sharpe ratio, $\mathrm{SR}_i$"),
            "symbol_tex": r"\mathrm{SR}_i",
            "field_tex": r"\texttt{return}, \texttt{bill}",
            "formula_tex": L(r"Équation~\eqref{eq:sharpe}", r"Equation~\eqref{eq:sharpe}"),
            "definition_fr": L(
                "Performance annualisée par unité de risque excédentaire.",
                "Annualized performance per unit of excess risk.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Référence contraignante, $b_i^{*}$", r"Binding benchmark, $b_i^{*}$"),
            "symbol_tex": r"b_i^{*}",
            "field_tex": r"\texttt{b6040}, \texttt{pp}",
            "formula_tex": r"$b_i^{*}=\arg\max_{b\in\{60/40,pp\}}\mathrm{SR}^b_i$",
            "definition_fr": L(
                "Référence statique ayant le Sharpe net le plus élevé dans le marché.",
                "Static benchmark with the highest net Sharpe in the market.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Avantage H1, $A_i$", r"H1 advantage, $A_i$"),
            "symbol_tex": r"A_i",
            "field_tex": r"\texttt{strat}, \texttt{b6040}, \texttt{pp}",
            "formula_tex": L(r"Équation~\eqref{eq:advantage}", r"Equation~\eqref{eq:advantage}"),
            "definition_fr": L(
                "Écart de Sharpe entre la stratégie et sa référence contraignante.",
                "Sharpe difference between the strategy and its binding benchmark.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Écart de rendement, $d_{i,t}$", r"Return gap, $d_{i,t}$"),
            "symbol_tex": r"d_{i,t}",
            "field_tex": r"\texttt{d}",
            "formula_tex": L(r"Équation~\eqref{eq:dit}", r"Equation~\eqref{eq:dit}"),
            "definition_fr": L(
                "Écart mensuel de rendement excédentaire entre stratégie et référence.",
                "Monthly excess-return gap between strategy and benchmark.",
            ),
            "source_fr": r"\texttt{h1\_return\_panel.csv}.",
        },
        {
            "name_tex": L(
                (
                    r"Écart et volatilité retardée, $z_{i,t}$ et "
                    r"$\widehat\sigma^x_{i,t-1}$"
                ),
                (
                    r"Gap and lagged volatility, $z_{i,t}$ and "
                    r"$\widehat\sigma^x_{i,t-1}$"
                ),
            ),
            "symbol_tex": r"z_{i,t}, \widehat\sigma^x_{i,t-1}",
            "field_tex": r"\texttt{z}",
            "formula_tex": L(r"Équation~\eqref{eq:zit}", r"Equation~\eqref{eq:zit}"),
            "definition_fr": L(
                (
                    "Différence des rendements excédentaires divisés par leurs "
                    "écarts-types sur les 36 mois achevés en $t-1$."
                ),
                (
                    "Difference of excess returns divided by their standard "
                    "deviations over the 36 months ending at $t-1$."
                ),
            ),
            "source_fr": r"\texttt{h1\_risk\_adjusted\_panel.csv}.",
        },
        {
            "name_tex": L(
                (
                    r"Constantes et résidus groupés, $\alpha_d$, $\alpha_z$, "
                    r"$\varepsilon^d_{i,t}$ et $\varepsilon^z_{i,t}$"
                ),
                (
                    r"Pooled constants and residuals, $\alpha_d$, $\alpha_z$, "
                    r"$\varepsilon^d_{i,t}$ and $\varepsilon^z_{i,t}$"
                ),
            ),
            "symbol_tex": (
                r"\alpha_d, \alpha_z, \varepsilon^d_{i,t}, "
                r"\varepsilon^z_{i,t}"
            ),
            "field_tex": L(r"\texttt{d} ou \texttt{z}", r"\texttt{d} or \texttt{z}"),
            "formula_tex": L(r"Équation~\eqref{eq:panel}", r"Equation~\eqref{eq:panel}"),
            "definition_fr": L(
                (
                    "Moyennes groupées des écarts de rendement et de risque, avec "
                    "leurs termes non expliqués respectifs."
                ),
                (
                    "Pooled means of the return and risk gaps, with their "
                    "respective unexplained terms."
                ),
            ),
            "source_fr": L(
                "Estimation des auteurs, erreurs groupées par mois.",
                "Authors' estimation, errors clustered by month.",
            ),
        },
        {
            "name_tex": L(r"Exposition énergétique, $E_i$", r"Energy exposure, $E_i$"),
            "symbol_tex": r"E_i",
            "field_tex": r"\texttt{EG.EGY.PRIM.PP.KD}, \texttt{TX.VAL.FUEL.ZS.UN}",
            "formula_tex": L(
                r"$E_i=(\operatorname{rang}_{1i}+\operatorname{rang}_{2i})/2$",
                r"$E_i=(\operatorname{rank}_{1i}+\operatorname{rank}_{2i})/2$",
            ),
            "definition_fr": L(
                (
                    "Moyenne des rangs 2021 d'intensité énergétique et de part des "
                    "combustibles dans les exportations."
                ),
                (
                    "Average of the 2021 ranks of energy intensity and of the fuel "
                    "share in exports."
                ),
            ),
            "source_fr": L(
                "Indicateurs du développement mondial, Banque mondiale.",
                "World Development Indicators, World Bank.",
            ),
        },
        {
            "name_tex": L(
                r"Avantage H2, $A_i^E$",
                r"H2 advantage, $A_i^E$",
            ),
            "symbol_tex": r"A_i^E",
            "field_tex": r"\texttt{augmentation\_advantage}",
            "formula_tex": L(r"Équation~\eqref{eq:h2advantage}", r"Equation~\eqref{eq:h2advantage}"),
            "definition_fr": L(
                "Écart de Sharpe produit par la substitution énergétique.",
                "Sharpe difference produced by the energy substitution.",
            ),
            "source_fr": r"\texttt{h2\_per\_market.csv}.",
        },
        {
            "name_tex": L(
                (
                    r"Rendements et poids H2, $r^E_{i,t}$, "
                    r"$r^{\mathrm{cur}}_{i,t}$, $w^E_{i,t}$ et "
                    r"$w^{\mathrm{cur}}_{i,t}$"
                ),
                (
                    r"H2 returns and weights, $r^E_{i,t}$, "
                    r"$r^{\mathrm{cur}}_{i,t}$, $w^E_{i,t}$ and "
                    r"$w^{\mathrm{cur}}_{i,t}$"
                ),
            ),
            "symbol_tex": (
                r"r^E_{i,t}, r^{\mathrm{cur}}_{i,t}, "
                r"w^E_{i,t}, w^{\mathrm{cur}}_{i,t}"
            ),
            "field_tex": L(
                r"\texttt{energy}, \texttt{signal}, rendements des quatre actifs",
                r"\texttt{energy}, \texttt{signal}, returns of the four assets",
            ),
            "formula_tex": L(r"Équation~\eqref{eq:h2weights}", r"Equation~\eqref{eq:h2weights}"),
            "definition_fr": L(
                (
                    "Vecteurs de poids et rendements des règles énergétique et "
                    "purement monétaire."
                ),
                (
                    "Weight and return vectors of the energy rule and the purely "
                    "monetary rule."
                ),
            ),
            "source_fr": L("Règles de portefeuille préspécifiées.", "Pre-specified portfolio rules."),
        },
        {
            "name_tex": L(
                (
                    r"Constante, pente et résidu H2, $\alpha_E$, $\beta_E$ et "
                    r"$\varepsilon_i^E$"
                ),
                (
                    r"H2 constant, slope and residual, $\alpha_E$, $\beta_E$ and "
                    r"$\varepsilon_i^E$"
                ),
            ),
            "symbol_tex": r"\alpha_E, \beta_E, \varepsilon_i^E",
            "field_tex": r"\texttt{exposure\_mean\_rank}",
            "formula_tex": L(r"Équation~\eqref{eq:h2}", r"Equation~\eqref{eq:h2}"),
            "definition_fr": L(
                (
                    "Ordonnée à l'origine, pente par rang d'exposition et résidu "
                    "de la coupe transversale H2."
                ),
                (
                    "Intercept, slope per exposure rank and residual of the H2 "
                    "cross-section."
                ),
            ),
            "source_fr": L("Estimation des auteurs.", "Authors' estimation."),
        },
        {
            "name_tex": L(r"Fraction graduée, $f_{i,t}$", r"Graded fraction, $f_{i,t}$"),
            "symbol_tex": r"f_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": L(r"Équation~\eqref{eq:amplitude}", r"Equation~\eqref{eq:amplitude}"),
            "definition_fr": L(
                "Fraction obligataire continue de la poche commutée dans la variante H3.",
                "Continuous bond fraction of the switched pocket in the H3 variant.",
            ),
            "source_fr": L("Calcul des auteurs.", "Authors' calculation."),
        },
        {
            "name_tex": L(r"Force standardisée, $u_{i,t}$", r"Standardized strength, $u_{i,t}$"),
            "symbol_tex": r"u_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": L(
                (
                    r"Score en fenêtre croissante de $R_{i,t}$ moins sa moyenne "
                    r"mobile de 84 mois"
                ),
                r"Expanding-window score of $R_{i,t}$ minus its 84-month moving average",
            ),
            "definition_fr": L(
                "Intensité standardisée du signal employée uniquement par la variante H3.",
                "Standardized signal strength used only by the H3 variant.",
            ),
            "source_fr": L(
                "Calcul des auteurs après 84 écarts disponibles.",
                "Authors' calculation after 84 available gaps.",
            ),
        },
        {
            "name_tex": L(r"Écart H3, $\Delta_i^{\mathrm{H3}}$", r"H3 gap, $\Delta_i^{\mathrm{H3}}$"),
            "symbol_tex": r"\Delta_i^{\mathrm{H3}}",
            "field_tex": r"\texttt{difference\_amplitude\_minus\_binary}",
            "formula_tex": r"$\Delta_i^{\mathrm{H3}}=\mathrm{SR}^{\mathrm{amp}}_i-\mathrm{SR}^{\mathrm{bin}}_i$",
            "definition_fr": L(
                "Différence de Sharpe entre la règle graduée et la règle binaire.",
                "Sharpe difference between the graded rule and the binary rule.",
            ),
            "source_fr": r"\texttt{h3\_per\_market.csv}.",
        },
        {
            "name_tex": L(
                (
                    r"Panel macroéconomique, $z^h_{i,t}$, $X_{i,t}$, "
                    r"$\alpha_i^h$, $\lambda_t^h$, $\beta_X^h$ et "
                    r"$\varepsilon^h_{i,t}$"
                ),
                (
                    r"Macroeconomic panel, $z^h_{i,t}$, $X_{i,t}$, "
                    r"$\alpha_i^h$, $\lambda_t^h$, $\beta_X^h$ and "
                    r"$\varepsilon^h_{i,t}$"
                ),
            ),
            "symbol_tex": (
                r"z^h_{i,t}, X_{i,t}, \alpha_i^h, \lambda_t^h, "
                r"\beta_X^h, \varepsilon^h_{i,t}"
            ),
            "field_tex": (
                r"\texttt{oil\_score}, \texttt{petroleum\_surplus}, "
                r"\texttt{balance\_sheet\_change\_pp}, "
                r"\texttt{broad\_money\_growth\_lag2}, "
                r"\texttt{market}, \texttt{month}"
            ),
            "formula_tex": L(r"Équation~\eqref{eq:macrocontrols}", r"Equation~\eqref{eq:macrocontrols}"),
            "definition_fr": L(
                (
                    "Écart ajusté du risque H1 ou H2, contrôle pétrole, bilan ou "
                    "monnaie large, effets fixes, pente et résidu."
                ),
                (
                    "H1 or H2 risk-adjusted gap, oil control, balance sheet or "
                    "broad money, fixed effects, slope and residual."
                ),
            ),
            "source_fr": L(
                "EIA, BRI et OCDE; estimation des auteurs.",
                "EIA, BIS and OECD; authors' estimation.",
            ),
        },
    ]


def write_notation(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    csv_path = figpath(RESULTS / "appendix_notation.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    tex_path = figpath(GENERATED / "appendix_notation_fr.tex")
    header_row = L(
        r"Nom et symbole & Champ(s) de base & Formule ou équation & Définition & Source \\",
        r"Name and symbol & Base field(s) & Formula or equation & Definition & Source \\",
    )
    lines = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{longtable}{@{}L{0.15\linewidth}L{0.18\linewidth}L{0.20\linewidth}L{0.27\linewidth}L{0.14\linewidth}@{}}",
        L(
            r"\caption{Notation, champs de base et provenance}\label{tab:notation}\\",
            r"\caption{Notation, base fields and provenance}\label{tab:notation}\\",
        ),
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        L(
            r"\multicolumn{5}{l}{\tablename\ \thetable\ -- suite} \\",
            r"\multicolumn{5}{l}{\tablename\ \thetable\ -- continued} \\",
        ),
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        L(
            r"\multicolumn{5}{r}{Suite page suivante} \\",
            r"\multicolumn{5}{r}{Continued on next page} \\",
        ),
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    def break_identifiers(value: str) -> str:
        return value.replace(r"\_", r"\_\allowbreak ").replace(
            ".",
            r".\allowbreak ",
        )

    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["name_tex"],
                    break_identifiers(row["field_tex"]),
                    row["formula_tex"],
                    row["definition_fr"],
                    break_identifiers(row["source_fr"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", r"\end{landscape}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, tex_path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    all_series = {**pooled_h1_series(), **test_and_macro_series()}
    records = []
    for panel, key, label in VARIABLES:
        record = {
            "panel": panel,
            "panel_title_fr": PANELS[panel],
            "variable_key": key,
            "variable_tex": label,
            **describe(all_series[key]),
        }
        records.append(record)
    descriptions = pd.DataFrame(records)
    reconcile(descriptions)

    descriptive_csv = figpath(RESULTS / "appendix_variable_descriptives.csv")
    descriptions.to_csv(descriptive_csv, index=False)
    descriptive_tex = write_descriptive_tex(descriptions)
    notation_csv, notation_tex = write_notation(notation_rows())

    outputs = [descriptive_csv, notation_csv, descriptive_tex, notation_tex]
    manifest = pd.DataFrame(
        [
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ]
    )
    manifest_path = figpath(RESULTS / "appendix_tables_sha256.csv")
    manifest.to_csv(manifest_path, index=False)

    print(descriptions.to_string(index=False))
    print(f"\nWrote {descriptive_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {notation_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {descriptive_tex.relative_to(REPO_ROOT)}")
    print(f"Wrote {notation_tex.relative_to(REPO_ROOT)}")
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
