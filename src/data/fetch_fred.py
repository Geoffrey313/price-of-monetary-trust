"""Fetch and verify the registered public FRED inputs.

Inputs: ``FRED_API_KEY`` from the environment or repository ``.env.local``.
Outputs: immutable JSON observations and a SHA-256 manifest below the
configured raw-data directory.
Purpose: populate the public macroeconomic layer used by H1--H3; this module
does not compute a portfolio return.

The successor protocol snapshots, for each market, the four FRED-verifiable series
(long yield, short rate, CPI, USD exchange rate) plus the shared gold price, snapshots raw JSON with
sha256 hashes, and prints the availability matrix. Equity total return is NOT on FRED for these
markets; its status is reported for the deviations log, and NO portfolio return is computed here.

Gate note: this script fetches and verifies series; it computes no strategy return. The ancestry
gate commit is c800e95e.
Run: python snapshot_markets.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time

from common.paths import RAW_DATA, read_env_value

SNAP = RAW_DATA

# per-market FRED series (verified spans for yields; the rest verified here)
MK = {
    "Canada":        {"yield": "IRLTLT01CAM156N", "short": "IR3TIB01CAM156N", "cpi": "CPALTT01CAM657N", "fx": "EXCAUS"},
    "South Africa":  {"yield": "INTGSBZAM193N",   "short": "INTGSTZAM193N",   "cpi": "ZAFCPIALLMINMEI", "fx": "EXSFUS"},
    "Norway":        {"yield": "IRLTLT01NOM156N", "short": "IR3TIB01NOM156N", "cpi": "NORCPIALLMINMEI", "fx": "EXNOUS"},
    "Australia":     {"yield": "IRLTLT01AUM156N", "short": "IR3TIB01AUM156N", "cpi": "AUSCPIALLQINMEI", "fx": "EXUSAL"},
    "India":         {"yield": "INTGSBINM193N",   "short": "INDIR3TIB01STM",  "cpi": "INDCPIALLMINMEI", "fx": "EXINUS"},
    "India(post17)": {"yield": "INDIRLTLT01STM"},
    "Korea":         {"yield": "IRLTLT01KRM156N", "short": "IR3TIB01KRM156N", "cpi": "KORCPIALLMINMEI", "fx": "EXKOUS"},
    "Mexico":        {"yield": "IRLTLT01MXM156N", "short": "IR3TIB01MXM156N", "cpi": "MEXCPIALLMINMEI", "fx": "EXMXUS"},
    "Switzerland":   {"yield": "IRLTLT01CHM156N", "short": "IR3TIB01CHM156N", "cpi": "CHECPIALLMINMEI", "fx": "EXSZUS"},
    "Japan":         {"yield": "IRLTLT01JPM156N", "short": "IR3TIB01JPM156N", "cpi": "JPNCPIALLMINMEI", "fx": "EXJPUS"},
    "France":        {"yield": "IRLTLT01FRM156N", "short": "IR3TIB01FRM156N", "cpi": "FRACPIALLMINMEI", "fx": "EXUSEU"},
    "United Kingdom":{"yield": "IRLTLT01GBM156N", "short": "IR3TIB01GBM156N", "cpi": "GBRCPIALLMINMEI", "fx": "EXUSUK"},
    "Germany":       {"yield": "IRLTLT01DEM156N", "short": "IR3TIB01DEM156N", "cpi": "DEUCPIALLMINMEI", "fx": "EXUSEU"},
}
GOLD = "GOLDPMGBD228NLBM"   # LBMA gold PM fix, USD


def fetch(sid: str):
    key = read_env_value("FRED_API_KEY", required=True)
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
           f"&api_key={key}&file_type=json")
    out = subprocess.run(["curl", "-s", "--max-time", "30", url], capture_output=True, text=True).stdout
    try:
        j = json.loads(out)
    except Exception:
        return None, None, None
    obs = [o for o in j.get("observations", []) if o.get("value") not in (".", None)]
    if not obs:
        return None, None, None
    return out, obs[0]["date"], obs[-1]["date"]


def main() -> int:
    SNAP.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"{'market':16s} {'series':7s} {'id':20s} {'span':26s}")
    for mkt, series in MK.items():
        for kind, sid in series.items():
            raw, first, last = fetch(sid)
            time.sleep(0.3)
            status = f"{first} .. {last}" if first else "NOT AVAILABLE"
            print(f"{mkt:16s} {kind:7s} {sid:20s} {status}")
            if raw:
                f = SNAP / f"{sid}.json"
                f.write_text(raw)
                manifest.append({"market": mkt, "kind": kind, "series": sid,
                                 "first": first, "last": last,
                                 "sha256": hashlib.sha256(raw.encode()).hexdigest()})
    raw, first, last = fetch(GOLD)
    if raw:
        (SNAP / f"{GOLD}.json").write_text(raw)
        manifest.append({"market": "GLOBAL", "kind": "gold", "series": GOLD,
                         "first": first, "last": last,
                         "sha256": hashlib.sha256(raw.encode()).hexdigest()})
        print(f"{'GLOBAL':16s} {'gold':7s} {GOLD:20s} {first} .. {last}")
    mf = SNAP.parent / "snapshot_manifest.json"
    mf.write_text(json.dumps(manifest, indent=1))
    print(f"\nsnapshotted {len(manifest)} series -> {SNAP}  (manifest with sha256: {mf})")
    print("Equity total return: not available on FRED for these markets - deviations-log item.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
