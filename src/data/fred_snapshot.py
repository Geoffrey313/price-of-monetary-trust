"""Load snapshotted FRED data and validate the registered historical splices.

Inputs: JSON observations under the configured ``data/raw`` directory and,
only when a snapshot is missing, ``FRED_API_KEY``.
Outputs: missing JSON snapshots and printed splice diagnostics; no portfolio
return is written.
Purpose: provide vintage-stable macro inputs for the H1--H3 engine and audit
the D3--D7 splice assumptions.

D3-D7 splice validations for the headline markets run before any computation.
Each splice fetches both segments from FRED and
report overlap stats (mean level gap, max abs gap, level corr, change corr, change vols).

Splices covered here:
  D4  CH bills:  IR3TIB01CHM156N (1999-07+)  <- earlier segment candidates (IFS)
  D6  IN bills:  INDIR3TIB01STM (2011-11+)   <- INTGSTINM193N (IFS T-bill)
  D7  KR CPI:    KORCPIALLMINMEI (ends 2023-11) -> continuation candidate
  D7  CA CPI:    CPALTT01CAM657N (ends 2024-02) -> continuation candidate
  D3  DE FX:     EXUSEU (1999+) <- EXGEUS legacy, boundary via irrevocable 1.95583 DEM/EUR

Gate note: compares raw series only; no portfolio, no strategy return.
Run: ``PYTHONPATH=src python -m data.fred_snapshot``.
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd

from common.paths import RAW_DATA, read_env_value

SNAP = RAW_DATA


def fred(sid: str) -> pd.Series | None:
    f = SNAP / f"{sid}.json"
    if f.exists():
        j = json.loads(f.read_text())
    else:
        key = read_env_value("FRED_API_KEY", required=True)
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
               f"&api_key={key}&file_type=json")
        out = subprocess.run(["curl", "-s", "--max-time", "30", url],
                             capture_output=True, text=True).stdout
        try:
            j = json.loads(out)
        except Exception:
            return None
        if "observations" not in j:
            return None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(out)                      # snapshot new pulls alongside the manifest set
    s = pd.Series({o["date"]: float(o["value"]) for o in j["observations"] if o["value"] != "."})
    if s.empty:
        return None
    s.index = pd.PeriodIndex(pd.to_datetime(s.index), freq="M")
    return s.groupby(level=0).last().sort_index()


def validate(name: str, old: pd.Series, new: pd.Series) -> None:
    ov = pd.concat([old.rename("o"), new.rename("n")], axis=1).dropna()
    if len(ov) < 6:
        print(f"{name}: OVERLAP TOO SHORT ({len(ov)} months) — splice NOT validated")
        return
    d = ov["o"] - ov["n"]
    print(f"{name}: overlap {ov.index.min()}..{ov.index.max()} ({len(ov)}m) | "
          f"mean gap {d.mean():+.3f}, max|gap| {d.abs().max():.3f}, level corr {ov['o'].corr(ov['n']):.3f}, "
          f"chg corr {ov['o'].diff().corr(ov['n'].diff()):.2f}, "
          f"chg vol {ov['o'].diff().std():.3f} vs {ov['n'].diff().std():.3f}")


def main() -> int:
    # D4 - Switzerland short rate
    ch_new = fred("IR3TIB01CHM156N")
    for cand in ["INTGSTCHM193N", "IR3TCD01CHM156N", "INTDSRCHM193N", "IRSTCI01CHM156N"]:
        ch_old = fred(cand)
        if ch_old is not None and ch_old.index.min() < pd.Period("1990-01", "M"):
            print(f"  [D4 CH] candidate {cand}: {ch_old.index.min()}..{ch_old.index.max()}")
            validate(f"D4 CH short ({cand} -> IR3TIB01CHM156N)", ch_old, ch_new)

    # D6 - India short rate
    in_new = fred("INDIR3TIB01STM")
    in_old = fred("INTGSTINM193N")
    if in_old is not None:
        print(f"  [D6 IN] INTGSTINM193N: {in_old.index.min()}..{in_old.index.max()}")
        validate("D6 IN short (INTGSTINM193N -> INDIR3TIB01STM)", in_old, in_new)
    else:
        print("  [D6 IN] INTGSTINM193N NOT on FRED")

    # D7 - CPI continuations (index-level candidates; legacy MEI series are 2015=100 indexes)
    kr_old = fred("KORCPIALLMINMEI")
    ca_old = fred("CPALTT01CAM657N")           # NOTE: this one is a monthly % change series
    for cand in ["KORCPALTT01IXNBM", "KORCPALTT01IXOBM", "CPALTT01KRM661N", "KORCPIALLMINMEI"]:
        s = fred(cand)
        if s is not None and cand != "KORCPIALLMINMEI" and s.index.max() > pd.Period("2025-06", "M"):
            print(f"  [D7 KR] candidate {cand}: {s.index.min()}..{s.index.max()}")
            validate(f"D7 KR CPI ({'KORCPIALLMINMEI'} -> {cand})", kr_old, s)
    for cand in ["CPALCY01CAM661N", "CPALTT01CAM661N", "CANCPIALLMINMEI", "CPALTT01CAM659N"]:
        s = fred(cand)
        if s is not None and s.index.max() > pd.Period("2025-06", "M"):
            print(f"  [D7 CA] candidate {cand}: {s.index.min()}..{s.index.max()}")
            # CA legacy is % change m/m; compare like-for-like where candidate is an index
            if cand.endswith(("661N", "659N")):
                cand_chg = s.pct_change() * 100.0
                validate(f"D7 CA CPI (CPALTT01CAM657N %chg -> {cand} %chg)", ca_old, cand_chg)

    # D3 - Germany FX pre-euro
    de_new = fred("EXUSEU")                    # USD per EUR, 1999-01+
    de_old = fred("EXGEUS")                    # DEM per USD, legacy
    if de_old is not None and de_new is not None:
        # convert legacy to USD-per-EUR at the irrevocable rate: EUR/USD = 1.95583 / (DEM/USD)
        de_old_conv = (1.95583 / de_old).rename("o")
        print(f"  [D3 DE] EXGEUS: {de_old.index.min()}..{de_old.index.max()} (converted at 1.95583)")
        validate("D3 DE FX (1.95583/EXGEUS -> EXUSEU)", de_old_conv, de_new)
    else:
        print("  [D3 DE] EXGEUS or EXUSEU missing")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
