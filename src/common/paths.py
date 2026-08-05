"""Resolve repository, credential, input-data, result, and paper paths.

Inputs: optional environment variables and a local ``.env.local`` file.
Outputs: no files; callers receive machine-independent :class:`pathlib.Path`
objects and credential values.
Purpose: make the complete-sample engine and every published analysis portable
without embedding a workstation path or a secret in source code.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"
FULL_SAMPLE_RESULTS = RESULTS_ROOT / "full-sample"
MACRO_RESULTS = RESULTS_ROOT / "controls-13-markets"
PAPER_ROOT = REPO_ROOT / "paper"
PAPER_GENERATED = PAPER_ROOT / "_gen"

_LEGACY_SUCCESSOR_DATA = REPO_ROOT / "tooling" / "successor" / "data"
_LEGACY_FRED_DATA = REPO_ROOT / "tooling" / "fred_fetcher" / "data"


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


DATA_ROOT = _configured_path("FOUR_QUADRANT_DATA_DIR", REPO_ROOT / "data")


def _data_subdir(name: str) -> Path:
    """Prefer the publication layout, with a local transition fallback."""
    canonical = DATA_ROOT / name
    if canonical.exists():
        return canonical
    return _LEGACY_SUCCESSOR_DATA / name


RAW_DATA = _data_subdir("raw")
RECON_DATA = _data_subdir("recon")
EVIDENCE_DATA = _data_subdir("evidence")
MACRO_DATA = _data_subdir("macro_controls") / "2026-07-27"

_canonical_fred = DATA_ROOT / "fred"
FRED_DATA = _configured_path(
    "FOUR_QUADRANT_FRED_DATA_DIR",
    _canonical_fred if _canonical_fred.exists() else _LEGACY_FRED_DATA,
)
FRED_RAW_DATA = FRED_DATA / "raw"


def env_file() -> Path:
    """Return the configured dotenv file or ``.env.local`` at repository root."""
    configured = os.environ.get("FOUR_QUADRANT_ENV_FILE")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT / ".env.local"
    )


def read_env_value(name: str, *, required: bool = False) -> str | None:
    """Read one setting from the process first, then the local dotenv file."""
    value = os.environ.get(name)
    if value:
        return value
    path = env_file()
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, candidate = line.split("=", 1)
            if key.strip() == name:
                value = candidate.strip().strip('"').strip("'")
                if value:
                    return value
    if required:
        raise RuntimeError(
            f"{name} is required; set it in the environment or in {path}"
        )
    return None


def protocol_log() -> Path:
    """Locate the frozen conformance record in publication or legacy layout."""
    canonical = REPO_ROOT / "protocol" / "DEVIATIONS.md"
    if canonical.exists():
        return canonical
    return REPO_ROOT / "tooling" / "successor" / "DEVIATIONS.md"
