"""Connect to WRDS for licensed equity and energy reconstructions.

Inputs: ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` from the environment or the
repository ``.env.local``.
Outputs: data frames returned in memory; reconstruction modules write parquet
inputs beneath the configured data directory.
Purpose: provide licensed market inputs for H1 and H2 without storing secrets
or licensed raw observations in the published repository.
Pattern ported from ``THU_NUCLEAR/wrds_fetcher`` (the connection idiom used by
every working script there), generalised for reuse.

Usage
-----
    from data.wrds import WRDSClient

    with WRDSClient() as db:                     # reads arcane/.env.local
        one = db.get_table("comp_na_daily_all", "funda", obs=1)
        df  = db.raw_sql("SELECT ... FROM ...")

Network note: WRDS pgdata (165.123.60.0/24 :9737) must be reachable. On this
host it sits behind a Tailscale exit node — run ``wrds_bypass_test.sh`` (sudo)
first to route that subnet locally.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from common.paths import env_file


def load_wrds_credentials(env_path: Optional[os.PathLike | str] = None) -> dict:
    """Parse WRDS_USERNAME / WRDS_PASSWORD out of a dotenv-style file.

    No external dependency (a hand-rolled parser mirroring the THU scripts) so
    this works with only ``wrds`` + ``pandas`` installed.
    """
    path = Path(os.fspath(env_path)) if env_path else env_file()
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")

    creds: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip().strip('"').strip("'")

    missing = [k for k in ("WRDS_USERNAME", "WRDS_PASSWORD") if not creds.get(k)]
    if missing:
        raise KeyError(f"missing {missing} in {path}")
    return {"username": creds["WRDS_USERNAME"], "password": creds["WRDS_PASSWORD"]}


class WRDSClient:
    """Thin wrapper over ``wrds.Connection`` with env-based password auth."""

    def __init__(
        self,
        env_path: Optional[os.PathLike | str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if username and password:
            self._user, self._pw = username, password
        else:
            c = load_wrds_credentials(env_path)
            self._user, self._pw = c["username"], c["password"]
        self.db = None  # type: ignore[assignment]

    def connect(self) -> "WRDSClient":
        import wrds  # imported lazily so `import client` works without the driver

        self.db = wrds.Connection(
            wrds_username=self._user,
            wrds_password=self._pw,
            autoconnect=False,
        )
        self.db.connect()
        self.db.load_library_list()
        return self

    # --- passthroughs -----------------------------------------------------
    def raw_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        return self.db.raw_sql(sql, **kwargs)

    def get_table(self, library: str, table: str, obs: int = 1, **kwargs) -> pd.DataFrame:
        return self.db.get_table(library=library, table=table, obs=obs, **kwargs)

    def list_libraries(self):
        return self.db.list_libraries()

    def list_tables(self, library: str):
        return self.db.list_tables(library=library)

    def describe_table(self, library: str, table: str):
        return self.db.describe_table(library=library, table=table)

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            finally:
                self.db = None

    # --- context manager --------------------------------------------------
    def __enter__(self) -> "WRDSClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()
