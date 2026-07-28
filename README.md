# Bonds or Gold: The Price of Monetary Trust

This repository holds a research paper and the code that reproduces it. The
manuscript is in French: `paper/bonds-or-gold-fr.tex`.

The study starts from a simple ratio, long government bonds over gold, and asks
whether it carries usable information about the monetary regime. It also asks
what is left of the practitioner "four quadrants" allocation framework once that
framework is taken apart and each piece is tested on its own.

## The question

The four-quadrant framework splits a portfolio across four macroeconomic states,
read from two market-price signals. One axis is energy, taken from the ratio of
an equity index to the oil price. The other is monetary, taken from the ratio of
a bond total-return index to gold. An earlier study on United States data found
the composite rule real but modest, and found the two axes to be uneven. Here we
test the two axes separately, on thirteen markets estimated together, and ask
what survives.

The monetary ratio, bond wealth over gold in local currency, reads as the
relative price of trust in the currency. When bonds outrun gold, holding the
currency is rewarded. When gold outruns bonds, investors leave it.

## What we test

Three hypotheses, one universe of thirteen markets, the same rule parameters and
cost assumptions everywhere.

* **H1, the monetary axis.** A rule that holds government bonds when the local
  bond-gold ratio sits above its seven-year average, and gold otherwise, beats
  the better of two static reference portfolios.
* **H2, the energy axis.** Replacing the fixed bond pocket with a pocket of
  listed energy equities improves performance, and the improvement grows with a
  market's energy exposure.
* **H3, the functional form.** The binary sign rule is not worse than a graded
  variant that scales the switched pocket by the strength of the signal.

## Main results

* The monetary axis survives. The rule beats the binding benchmark in twelve of
  thirteen markets, with a median net Sharpe advantage of 0.143. The pooled
  risk-adjusted estimate is 0.086, significant at the one-sided ten percent level
  under a covariance that treats cross-market and serial dependence together.
* The energy substitution lowers the net Sharpe in eleven of thirteen markets.
  The energy axis does not carry the information the framework claims for it.
* The surviving rule is binary. The graded variant loses in all thirteen markets.
* The advantage is concentrated before 2000. The pooled alpha moves from 0.317
  before 2000 to minus 0.019 after. This split is descriptive and carries no
  causal claim.

Put together, the four-quadrant framework reduces to a single binary monetary
switch between bonds and gold, held inside an otherwise static equal-weight
portfolio.

## Repository layout

```
src/common/    market constants, frozen protocol values, and portable paths
src/data/      public snapshots, WRDS access, and bond-return construction
src/engine/    the portfolio engine and equity reference reconstruction
src/analysis/  H1--H3, era analyses, and macroeconomic controls
src/figures/   paper figures and generated appendix tables
protocol/      frozen source, deviation, and trial records
paper/         manuscript, compiled PDF, and generated appendix tables
results/       figures and tables used by the paper
reproduce.py   deterministic end-to-end entry point
```

## Reproducing the results

1. Copy `.env.example` to `.env.local` and fill in the API credentials (see Data
   and credentials below).
2. Install the dependencies: `pip install -r requirements.txt`.
3. Put licensed WRDS/CRSP reconstructions and downloaded public inputs in the
   layout documented by `data/README.md`, or set `FOUR_QUADRANT_DATA_DIR`.
4. Run `python reproduce.py`. It validates the input layer, rebuilds H1--H3,
   the era analyses, figures, macro controls, and `paper/_gen`, then checks that
   every pre-existing output retains the same SHA-256 digest.

Use `python reproduce.py --refresh-fred` only when intentionally refreshing the
public FRED snapshots. Direct module commands use `PYTHONPATH=src`, for example
`PYTHONPATH=src python -m analysis.run_full_sample`.

## Data and credentials

The study draws on public and licensed sources: the OECD Main Economic
Indicators through FRED for rates, exchange rates and prices; S&P Compustat
Global and CRSP through WRDS for the equity reconstructions and the energy
pockets; a dollar gold series converted into local currency; and Shiller, the
Swiss National Bank, Statistics Canada and Damodaran for validation. All
credentials are read from the process environment or the repository
`.env.local` and are never written into code. Licensed and raw inputs are
excluded from version control.

## The paper

Compile with `latexmk -cd -pdf paper/bonds-or-gold-fr.tex` so paths to
`results/` are resolved from the manuscript directory. The compiled PDF sits
in `paper/`. The appendix variable and notation tables are generated into
`paper/_gen/` from the sample outputs.

## Disclaimer

This is a research paper about an illustrative allocation rule. Nothing here is
investment advice, and the rule is not a security.
