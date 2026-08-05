# Local reproduction data

Place licensed and downloaded inputs below this directory, or set
`FOUR_QUADRANT_DATA_DIR` to an equivalent directory:

```
data/
├── raw/             FRED JSON snapshots
├── fred/raw/        normalized public series and source workbooks
├── evidence/        public validation files
├── recon/           licensed WRDS/CRSP reconstructions
├── derived/         built locally: one {market}.parquet per market (see below)
└── macro_controls/2026-07-27/
```

The source tree contains no credentials and no licensed raw data. A clean clone
should use the layout above.

## The derived layer and `--from-derived`

`data/derived/` holds one small monthly parquet per market with five series
(aggregated equity total return, long yield, gold in local currency, short rate,
consumer price index). It lets `reproduce.py --from-derived` rebuild every result
without WRDS access.

It is not shipped: the equity series is downstream of licensed WRDS/CRSP/Compustat
reconstructions, and all of `data/` is git-ignored. Build it once, with WRDS
credentials in place, then run offline afterwards:

```
PYTHONPATH=src python -m engine.derived_inputs   # writes data/derived/*.parquet
python reproduce.py --from-derived               # rebuilds results without WRDS
```
