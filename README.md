# Bonds or Gold: The Price of Monetary Trust

Replication package for:

> **Bonds or Gold: The Price of Monetary Trust**
> Geoffrey Ducournau, Jinliang Li

This repository holds only the code that reproduces the results of the paper
*Bonds or Gold: The Price of Monetary Trust*. The manuscript and the data are not
distributed here; the code below rebuilds every result from source.

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

* The monetary axis survives in the full sample. The rule beats the binding
  benchmark in twelve of thirteen markets, with a median net Sharpe advantage of
  0.143. The pooled risk-adjusted estimate is 0.086, significant at the one-sided
  ten percent level under a covariance that treats cross-market and serial
  dependence together; the direct difference of conventional Sharpe ratios is 0.122.
* The energy substitution lowers the net Sharpe in eleven of thirteen markets.
  The energy axis does not carry the information the framework claims for it.
* The surviving rule is binary. The graded variant loses in all thirteen markets.
* The advantage is concentrated before 2000. The pooled alpha moves from 0.317
  before 2000 to minus 0.019 after. This split is descriptive and carries no
  causal claim. In the modern sample, with a freely floating and investable gold,
  the pooled alpha is essentially zero, so the rule's advantage is a dated feature
  of the record, sitting in the pre-float, less investable gold regime.

Put together, the four-quadrant framework reduces to a single binary monetary
switch between bonds and gold, real in the full sample but carried by the
pre-float gold regime, held inside an otherwise static equal-weight portfolio.

## Repository layout

```
src/common/    market constants, frozen protocol values, and portable paths
src/data/      readers for the shipped derived layer and bond-return construction
src/engine/    the portfolio engine and equity reference reconstruction
src/analysis/  H1--H3, era analyses, and macroeconomic controls
src/figures/   figures and generated appendix tables
reproduce.py   deterministic end-to-end entry point
data/derived/  the shipped WRDS-free monthly inputs, one Parquet per market
data/vendor/   public vendored macro inputs (oil VAR, industry, production)
data/README.md documents the shipped data and its licensing
```

## Reproducing the results

The repository ships the transformed inputs the offline run reads: the derived
layer (`data/derived/`, one Parquet per market) and the public vendored macro
files (`data/vendor/`). Install the dependencies and run the offline entry
point:

```
pip install -r requirements.txt
python reproduce.py --from-derived
```

This reads the shipped inputs and rebuilds the currency results and figures
locally. Neither the generated outputs nor the manuscript are stored in this
repository; the code is the single source of truth. Direct module commands use
`PYTHONPATH=src`, for example `PYTHONPATH=src python -m analysis.run_full_sample`.

The H2 energy test and the full build reconstruct listed-energy and equity total
returns from S&P Compustat Global through WRDS. Those licensed inputs are not
part of this repository and are available on request (see Contact). Given WRDS
and FRED access, `PYTHONPATH=src python -m engine.derived_inputs` regenerates the
shipped derived layer.

## Data and credentials

The study draws on public and licensed sources: the OECD Main Economic
Indicators through FRED for rates, exchange rates and prices; S&P Compustat
Global and CRSP through WRDS for the equity reconstructions and the energy
pockets; a dollar gold series converted into local currency; and Shiller, the
Swiss National Bank, Statistics Canada and Damodaran for validation. All
credentials are read from the process environment or the repository
`.env.local` and are never written into code. Licensed and raw inputs, and the
scripts that acquire them, are excluded from version control; the transformed
inputs the offline run needs are shipped under `data/`.

## Contact

Questions about the code, or requests for the licensed reconstruction inputs:
Geoffrey Ducournau, G.ducournau.voisin@gmail.com.

## The paper

The manuscript itself is not distributed in this repository. This repository
contains only the code that reproduces its results.

## Disclaimer

This code accompanies a research paper about an illustrative allocation rule.
Nothing here is investment advice, and the rule is not a security.
