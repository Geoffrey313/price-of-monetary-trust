# Derived market inputs (WRDS-free reproduction)

One Parquet file per market, `{market}.parquet`, holding the monthly series the
engine consumes:

| column       | meaning                                       | source                          |
|--------------|-----------------------------------------------|---------------------------------|
| `month`      | calendar month                                | --                              |
| `eq`         | aggregated cap-weighted equity total return   | derived (see the note below)    |
| `long_yield` | long government bond yield                     | FRED (public)                   |
| `gold_local` | gold in local currency                        | public gold series              |
| `short_rate` | short rate                                     | FRED (public)                   |
| `cpi_index`  | consumer price index                          | FRED (public)                   |

From these five series the pipeline reproduces H1, H3, the era analyses and the
figures without any WRDS access. The firm-level reconstruction step is skipped:

    PYTHONPATH=src python reproduce.py --from-derived

## Energy sublayer (`energy/{market}.parquet`)

The H2 energy test needs each market's listed-energy total return. That series is
frozen here, one file per market, so H2 also reproduces offline:

| column         | meaning                                            |
|----------------|----------------------------------------------------|
| `month`        | calendar month                                     |
| `return`       | value-weighted listed-energy total return          |
| `n_securities` | securities in the sleeve that month                |
| `n_firms`      | distinct firms in the sleeve that month            |
| `hhi`          | Herfindahl concentration of the sleeve weights     |
| `effective_n`  | effective number of names (`1 / hhi`)              |

These are market-level aggregates, not firm-level records. A full run rebuilds
and refreshes this sublayer automatically; `--from-derived` reads it.

The raw firm-level records (`recon/*_monthend.parquet`, `*_csho.parquet`,
`sector_map.parquet`) are licensed and are not part of this repository.

Regenerate this layer from the full WRDS wiring (author use only):

    PYTHONPATH=src python -m engine.derived_inputs

## Note on licensing

`eq` is a market-level aggregate, a cap-weighted index total return, not
firm-level Compustat records. For ten markets it is reconstructed from S&P
Compustat Global through WRDS; for the United States, Switzerland and Canada it
comes from public native series (Shiller, the Swiss National Bank, Statistics
Canada). Publishing aggregated derived series of this kind is generally
permitted, but the repository owner is responsible for confirming it against the
applicable WRDS and index-provider terms before release.
