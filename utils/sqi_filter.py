"""
SQI-based filtering utilities for EEG datasets.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def apply_sqi_filter(df: pd.DataFrame, mode: str, threshold: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply SQI filtering according to the specified mode.

    Parameters
    ----------
    df:
        EEG dataframe containing an ``SQI`` column.
    mode:
        Filtering mode - ``None`` (no filtering), ``>=`` (keep SQI >= threshold),
        or ``<=`` (keep SQI <= threshold).
    threshold:
        SQI threshold (0-100).
    """
    if "SQI" not in df.columns:
        raise ValueError("Dataframe must include 'SQI' column for SQI filtering.")

    mode = (mode or "None").strip().lower()
    original_count = len(df)

    if mode in {"none", "no filter"}:
        filtered = df.copy()
    elif mode in {"keep sqi >= threshold", ">=", "ge"}:
        filtered = df[df["SQI"] >= threshold].copy()
    elif mode in {"keep sqi <= threshold", "<=", "le"}:
        filtered = df[df["SQI"] <= threshold].copy()
    else:
        raise ValueError(f"Unsupported SQI filter mode: {mode}")

    filtered_count = len(filtered)
    removed = original_count - filtered_count
    stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": removed,
        "percent_removed": (removed / original_count * 100) if original_count else 0.0,
    }
    return filtered, stats


