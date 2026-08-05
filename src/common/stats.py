"""Shared statistical helpers used across analysis modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm step-down adjusted p-values in the original row order."""
    order = np.argsort(p_values.to_numpy())
    sorted_p = p_values.to_numpy()[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(sorted_p) - np.arange(len(sorted_p))) * sorted_p
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(len(sorted_p))
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)
