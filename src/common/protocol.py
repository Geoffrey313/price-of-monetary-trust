"""Frozen numerical protocol constants used by the published engine.

Inputs: none.
Outputs: none.
Purpose: expose the preregistered bond total-return assumptions from one
auditable location rather than repeating numerical literals in data code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreasuryTotalReturnProtocol:
    """Constant-duration government-bond total-return construction."""

    maturity_years: int = 10
    coupons_per_year: int = 2
    include_convexity: bool = False


TREASURY_TOTAL_RETURN = TreasuryTotalReturnProtocol()
