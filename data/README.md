# Local reproduction data

Place licensed and downloaded inputs below this directory, or set
`FOUR_QUADRANT_DATA_DIR` to an equivalent directory:

```
data/
├── raw/             FRED JSON snapshots
├── fred/raw/        normalized public series and source workbooks
├── evidence/        public validation files
├── recon/           licensed WRDS/CRSP reconstructions
└── macro_controls/2026-07-27/
```

The source tree contains no credentials and no licensed raw data. During the
local transition from the research workspace, the path resolver can read the
same directories under `tooling/`; a clean clone should use the layout above.
