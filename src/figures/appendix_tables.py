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

import hashlib

import numpy as np
import pandas as pd

from analysis import macro_controls as macro
from analysis import run_full_sample as rerun
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
    "A": "Rendements mensuels bruts utilisés, en pour cent",
    "B": "Rendements mensuels des portefeuilles calculés, en pour cent",
    "C": "Variables calculées des tests",
    "D": "Variables macroéconomiques exploratoires du panel H1",
}

VARIABLES = [
    ("A", "equity", "Actions"),
    ("A", "bond", "Obligations"),
    ("A", "gold", "Or en monnaie locale"),
    ("A", "bill", "Taux court"),
    ("B", "strategy", "Bascule binaire"),
    ("B", "6040", "60/40 local"),
    ("B", "permanent", "Portefeuille permanent"),
    ("B", "matched", "Statique comparable"),
    ("C", "d", r"$d_{i,t}$, écart de rendement, en points de pourcentage"),
    ("C", "z", r"$z_{i,t}$, écart ajusté du risque"),
    ("C", "energy_rank", r"$E_i$, rang d'exposition énergétique"),
    ("C", "h2_advantage", r"$A_i^{E}$, écart de Sharpe H2"),
    ("C", "h3_delta", r"$\Delta_i^{\mathrm{H3}}$, graduée moins binaire"),
    (
        "D",
        "oil_score",
        r"Position pétrolière nette, transformation $\operatorname{asinh}$",
    ),
    ("D", "petroleum_surplus", "Surplus pétrolier, indicatrice"),
    ("D", "balance_sheet", "Expansion du bilan, points de PIB"),
    ("D", "broad_money", "Croissance de la monnaie large, en pour cent"),
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
    return f"{value:,}".replace(",", "~")


def format_number(value: float) -> str:
    rounded = 0.0 if abs(value) < 0.0005 else value
    text = f"{rounded:.3f}".replace(".", ",")
    return f"${text.replace(',', '{,}')}$" if rounded < 0 else text


def write_descriptive_tex(frame: pd.DataFrame) -> Path:
    path = GENERATED / "appendix_variable_descriptives_fr.tex"
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (
            r"Panneau & Variable & Obs. & Moyenne & Écart-type & Minimum "
            r"& P25 & Médiane & P75 & Maximum \\"
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
            "name_tex": r"Richesse obligataire, $W^b_{i,t}$",
            "symbol_tex": r"W^b_{i,t}",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": r"$W^b_{i,t}=W^b_{i,t-1}(1+r^b_{i,t})$",
            "definition_fr": (
                "Indice de richesse obtenu en composant les rendements totaux "
                "des obligations d'État locales à dix ans."
            ),
            "source_fr": "FRED/OCDE et FRED/IFS; calcul des auteurs.",
        },
        {
            "name_tex": r"Or en dollars, $G^{\$}_t$",
            "symbol_tex": r"G^{\$}_t",
            "field_tex": r"\texttt{gold\_spot\_usd}",
            "formula_tex": r"Niveau observé $G^{\$}_t$",
            "definition_fr": "Prix mensuel d'une once d'or exprimé en dollars.",
            "source_fr": "Photographie Datahub, série dérivée de la LBMA, 1833--2026.",
        },
        {
            "name_tex": r"Taux de change, $\mathrm{FX}_{i,t}$",
            "symbol_tex": r"\mathrm{FX}_{i,t}",
            "field_tex": r"\texttt{EX*US}, \texttt{EXUSEU}",
            "formula_tex": r"Unités de monnaie $i$ pour un dollar",
            "definition_fr": (
                "Conversion mensuelle du dollar en monnaie locale, avec raccords "
                "officiels des monnaies prédécesseures de l'euro."
            ),
            "source_fr": "Federal Reserve Board via FRED.",
        },
        {
            "name_tex": r"Or local, $G_{i,t}$",
            "symbol_tex": r"G_{i,t}",
            "field_tex": r"\texttt{gold\_spot\_usd} et \texttt{EX*US}",
            "formula_tex": r"$G_{i,t}=G^{\$}_t\mathrm{FX}_{i,t}$",
            "definition_fr": "Prix de l'or exprimé dans la monnaie du marché.",
            "source_fr": "Calcul des auteurs à partir de l'or USD et du change.",
        },
        {
            "name_tex": r"Taux long et duration, $y_{i,t}$ et $D(y)$",
            "symbol_tex": r"y_{i,t}, D(y)",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": (
                r"Champ source divisé par 100; duration d'une "
                r"obligation au pair à dix ans"
            ),
            "definition_fr": (
                "Rendement à dix ans en fraction annuelle dans les équations et "
                "sensibilité de premier ordre du prix obligataire, en années."
            ),
            "source_fr": "FRED/OCDE et FRED/IFS; calcul des auteurs.",
        },
        {
            "name_tex": r"Taux sans risque, $r^f_{i,t}$",
            "symbol_tex": r"r^f_{i,t}",
            "field_tex": (
                r"\texttt{IR3TIB01*M156N}, \texttt{TB3MS}, "
                r"\texttt{INTGSTZAM193N}"
            ),
            "formula_tex": r"Taux annuel publié divisé par 1\,200",
            "definition_fr": "Rendement mensuel du taux court local publié en pourcentage annuel.",
            "source_fr": "FRED/OCDE et FRED/IFS.",
        },
        {
            "name_tex": r"Rendement obligataire, $r^b_{i,t}$",
            "symbol_tex": r"r^b_{i,t}",
            "field_tex": (
                r"\texttt{IRLTLT01*M156N}, \texttt{GS10}, "
                r"\texttt{INTGSBZAM193N}"
            ),
            "formula_tex": r"Équation~\eqref{eq:bondtr}",
            "definition_fr": (
                "Rendement total mensuel approché par le portage et la variation "
                "de prix à duration modifiée."
            ),
            "source_fr": "FRED/OCDE et FRED/IFS; calcul des auteurs.",
        },
        {
            "name_tex": r"Rendement de stratégie, $r^s_{i,t}$",
            "symbol_tex": r"r^s_{i,t}",
            "field_tex": r"\texttt{eq}, \texttt{bond}, \texttt{gold}, \texttt{bill}",
            "formula_tex": r"Équation~\eqref{eq:netret}",
            "definition_fr": "Rendement mensuel net de la règle binaire, après coûts de rotation.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Rendement d'actif, $r^a_{i,t}$",
            "symbol_tex": r"r^a_{i,t}",
            "field_tex": r"\texttt{eq}, \texttt{bond}, \texttt{gold}, \texttt{bill}, \texttt{energy}",
            "formula_tex": r"Variation mensuelle simple de l'actif $a$",
            "definition_fr": "Rendement mensuel de la poche actions, obligations, or, taux ou énergie.",
            "source_fr": "Sources de marché décrites dans la section des données.",
        },
        {
            "name_tex": r"Rapport monétaire, $R_{i,t}$",
            "symbol_tex": r"R_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": r"$R_{i,t}=W^b_{i,t}/G_{i,t}$",
            "definition_fr": "Richesse obligataire relativement au prix local de l'or.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Signal binaire, $s_{i,t}$",
            "symbol_tex": r"s_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": r"Équation~\eqref{eq:signal}",
            "definition_fr": (
                "Indicatrice valant un lorsque le rapport monétaire atteint ou "
                "dépasse sa moyenne mobile de 84 mois."
            ),
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Poids de portefeuille, $w^a_{i,t}$",
            "symbol_tex": r"w^a_{i,t}",
            "field_tex": r"\texttt{signal}",
            "formula_tex": r"Équation~\eqref{eq:weights}",
            "definition_fr": "Poids cible de l'actif $a$, appliqué avec un mois de retard.",
            "source_fr": "Règle de portefeuille préspécifiée.",
        },
        {
            "name_tex": r"Poids avant transaction, $\widetilde w^a_{i,t}$",
            "symbol_tex": r"\widetilde w^a_{i,t}",
            "field_tex": r"\texttt{previous\_weight}, \texttt{asset\_return}",
            "formula_tex": (
                r"$\widetilde w^a_{i,t}=w^a_{i,t-1}(1+r^a_{i,t-1})/"
                r"\sum_jw^j_{i,t-1}(1+r^j_{i,t-1})$"
            ),
            "definition_fr": "Poids auquel la poche dérive avant le rééquilibrage du mois.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Coût unitaire, $c$",
            "symbol_tex": r"c",
            "field_tex": r"\texttt{COST\_ONEWAY}",
            "formula_tex": r"$c=0{,}004$",
            "definition_fr": "Coût proportionnel aller simple appliqué au poids traité.",
            "source_fr": "Hypothèse préspécifiée de 40 points de base.",
        },
        {
            "name_tex": r"Ratio de Sharpe, $\mathrm{SR}_i$",
            "symbol_tex": r"\mathrm{SR}_i",
            "field_tex": r"\texttt{return}, \texttt{bill}",
            "formula_tex": r"Équation~\eqref{eq:sharpe}",
            "definition_fr": "Performance annualisée par unité de risque excédentaire.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Référence contraignante, $b_i^\star$",
            "symbol_tex": r"b_i^\star",
            "field_tex": r"\texttt{b6040}, \texttt{pp}",
            "formula_tex": r"$b_i^\star=\arg\max_{b\in\{60/40,pp\}}\mathrm{SR}^b_i$",
            "definition_fr": "Référence statique ayant le Sharpe net le plus élevé dans le marché.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Avantage H1, $A_i$",
            "symbol_tex": r"A_i",
            "field_tex": r"\texttt{strat}, \texttt{b6040}, \texttt{pp}",
            "formula_tex": r"Équation~\eqref{eq:advantage}",
            "definition_fr": "Écart de Sharpe entre la stratégie et sa référence contraignante.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Écart de rendement, $d_{i,t}$",
            "symbol_tex": r"d_{i,t}",
            "field_tex": r"\texttt{d}",
            "formula_tex": r"Équation~\eqref{eq:dit}",
            "definition_fr": "Écart mensuel de rendement excédentaire entre stratégie et référence.",
            "source_fr": r"\texttt{h1\_return\_panel.csv}.",
        },
        {
            "name_tex": (
                r"Écart et volatilité retardée, $z_{i,t}$ et "
                r"$\widehat\sigma^x_{i,t-1}$"
            ),
            "symbol_tex": r"z_{i,t}, \widehat\sigma^x_{i,t-1}",
            "field_tex": r"\texttt{z}",
            "formula_tex": r"Équation~\eqref{eq:zit}",
            "definition_fr": (
                "Différence des rendements excédentaires divisés par leurs "
                "écarts-types sur les 36 mois achevés en $t-1$."
            ),
            "source_fr": r"\texttt{h1\_risk\_adjusted\_panel.csv}.",
        },
        {
            "name_tex": (
                r"Constantes et résidus groupés, $\alpha_d$, $\alpha_z$, "
                r"$\varepsilon^d_{i,t}$ et $\varepsilon^z_{i,t}$"
            ),
            "symbol_tex": (
                r"\alpha_d, \alpha_z, \varepsilon^d_{i,t}, "
                r"\varepsilon^z_{i,t}"
            ),
            "field_tex": r"\texttt{d} ou \texttt{z}",
            "formula_tex": r"Équation~\eqref{eq:panel}",
            "definition_fr": (
                "Moyennes groupées des écarts de rendement et de risque, avec "
                "leurs termes non expliqués respectifs."
            ),
            "source_fr": "Estimation des auteurs, erreurs groupées par mois.",
        },
        {
            "name_tex": r"Exposition énergétique, $E_i$",
            "symbol_tex": r"E_i",
            "field_tex": r"\texttt{EG.EGY.PRIM.PP.KD}, \texttt{TX.VAL.FUEL.ZS.UN}",
            "formula_tex": r"$E_i=(\operatorname{rang}_{1i}+\operatorname{rang}_{2i})/2$",
            "definition_fr": (
                "Moyenne des rangs 2021 d'intensité énergétique et de part des "
                "combustibles dans les exportations."
            ),
            "source_fr": "Indicateurs du développement mondial, Banque mondiale.",
        },
        {
            "name_tex": r"Avantage H2, $A_i^{\mathrm{aug}}\equiv A_i^E$",
            "symbol_tex": r"A_i^{\mathrm{aug}}\equiv A_i^E",
            "field_tex": r"\texttt{augmentation\_advantage}",
            "formula_tex": r"Équation~\eqref{eq:h2advantage}",
            "definition_fr": "Écart de Sharpe produit par la substitution énergétique.",
            "source_fr": r"\texttt{h2\_per\_market.csv}.",
        },
        {
            "name_tex": (
                r"Rendements et poids H2, $r^E_{i,t}$, "
                r"$r^{\mathrm{cur}}_{i,t}$, $w^E_{i,t}$ et "
                r"$w^{\mathrm{cur}}_{i,t}$"
            ),
            "symbol_tex": (
                r"r^E_{i,t}, r^{\mathrm{cur}}_{i,t}, "
                r"w^E_{i,t}, w^{\mathrm{cur}}_{i,t}"
            ),
            "field_tex": r"\texttt{energy}, \texttt{signal}, rendements des quatre actifs",
            "formula_tex": r"Équation~\eqref{eq:h2weights}",
            "definition_fr": (
                "Vecteurs de poids et rendements des règles énergétique et "
                "purement monétaire."
            ),
            "source_fr": "Règles de portefeuille préspécifiées.",
        },
        {
            "name_tex": (
                r"Constante, pente et résidu H2, $\alpha_E$, $\beta_E$ et "
                r"$\varepsilon_i^E$"
            ),
            "symbol_tex": r"\alpha_E, \beta_E, \varepsilon_i^E",
            "field_tex": r"\texttt{exposure\_mean\_rank}",
            "formula_tex": r"Équation~\eqref{eq:h2}",
            "definition_fr": (
                "Ordonnée à l'origine, pente par rang d'exposition et résidu "
                "de la coupe transversale H2."
            ),
            "source_fr": "Estimation des auteurs.",
        },
        {
            "name_tex": r"Fraction graduée, $f_{i,t}$",
            "symbol_tex": r"f_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": r"Équation~\eqref{eq:amplitude}",
            "definition_fr": "Fraction obligataire continue de la poche commutée dans la variante H3.",
            "source_fr": "Calcul des auteurs.",
        },
        {
            "name_tex": r"Force standardisée, $z^{(2)}_{i,t}$",
            "symbol_tex": r"z^{(2)}_{i,t}",
            "field_tex": r"\texttt{bond\_wealth}, \texttt{gold\_local}",
            "formula_tex": (
                r"Score en fenêtre croissante de $R_{i,t}$ moins sa moyenne "
                r"mobile de 84 mois"
            ),
            "definition_fr": "Intensité standardisée du signal employée uniquement par la variante H3.",
            "source_fr": "Calcul des auteurs après 84 écarts disponibles.",
        },
        {
            "name_tex": r"Écart H3, $\Delta_i^{\mathrm{H3}}$",
            "symbol_tex": r"\Delta_i^{\mathrm{H3}}",
            "field_tex": r"\texttt{difference\_amplitude\_minus\_binary}",
            "formula_tex": r"$\Delta_i^{\mathrm{H3}}=\mathrm{SR}^{\mathrm{amp}}_i-\mathrm{SR}^{\mathrm{bin}}_i$",
            "definition_fr": "Différence de Sharpe entre la règle graduée et la règle binaire.",
            "source_fr": r"\texttt{h3\_per\_market.csv}.",
        },
        {
            "name_tex": (
                r"Panel macroéconomique, $z^h_{i,t}$, $X_{i,t}$, "
                r"$\alpha_i^h$, $\lambda_t^h$, $\beta_X^h$ et "
                r"$\varepsilon^h_{i,t}$"
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
            "formula_tex": r"Équation~\eqref{eq:macrocontrols}",
            "definition_fr": (
                "Écart ajusté du risque H1 ou H2, contrôle pétrole, bilan ou "
                "monnaie large, effets fixes, pente et résidu."
            ),
            "source_fr": "EIA, BRI et OCDE; estimation des auteurs.",
        },
    ]


def write_notation(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    csv_path = RESULTS / "appendix_notation.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    tex_path = GENERATED / "appendix_notation_fr.tex"
    lines = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\singlespacing",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{longtable}{@{}L{0.15\linewidth}L{0.18\linewidth}L{0.20\linewidth}L{0.27\linewidth}L{0.14\linewidth}@{}}",
        r"\caption{Notation, champs de base et provenance}\label{tab:notation}\\",
        r"\toprule",
        r"Nom et symbole & Champ(s) de base & Formule ou équation & Définition & Source \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\tablename\ \thetable\ -- suite} \\",
        r"\toprule",
        r"Nom et symbole & Champ(s) de base & Formule ou équation & Définition & Source \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Suite page suivante} \\",
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

    descriptive_csv = RESULTS / "appendix_variable_descriptives.csv"
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
    manifest_path = RESULTS / "appendix_tables_sha256_2026-07-27.csv"
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
