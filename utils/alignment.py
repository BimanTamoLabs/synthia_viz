"""
Timestamp alignment helpers for EEG and HRV datasets.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def validate_timestamp_alignment(eeg_df: pd.DataFrame, hrv_df: pd.DataFrame) -> dict:
    """
    Validate that EEG timestamps envelop HRV timestamps.
    """
    if eeg_df.empty or hrv_df.empty:
        raise ValueError("Both eeg_df and hrv_df must be non-empty.")

    if "Timestamp" not in eeg_df.columns or "timestamp" not in hrv_df.columns:
        raise ValueError("EEG dataframe must have 'Timestamp'; HRV dataframe must have 'timestamp'.")

    eeg_start = eeg_df["Timestamp"].min()
    eeg_end = eeg_df["Timestamp"].max()
    hrv_start = hrv_df["timestamp"].min()
    hrv_end = hrv_df["timestamp"].max()

    report = {
        "eeg_start": eeg_start,
        "eeg_end": eeg_end,
        "hrv_start": hrv_start,
        "hrv_end": hrv_end,
        "start_aligned": eeg_start <= hrv_start,
        "end_aligned": eeg_end >= hrv_end,
        "start_delta": (eeg_start - hrv_start) if eeg_start > hrv_start else (hrv_start - eeg_start),
        "end_delta": (hrv_end - eeg_end) if eeg_end < hrv_end else (eeg_end - hrv_end),
    }
    report["is_valid"] = report["start_aligned"] and report["end_aligned"]
    return report


def assign_hrv_testTypes_to_eeg(
    eeg_df: pd.DataFrame,
    hrv_timings_df: pd.DataFrame,
    phase_column: str = "testType",
    default_label: str = "OUT",
) -> pd.DataFrame:
    """
    Label EEG samples with HRV phase windows.
    """
    if eeg_df.empty or hrv_timings_df.empty:
        raise ValueError("Both eeg_df and hrv_timings_df must be non-empty.")

    if "Timestamp" not in eeg_df.columns:
        raise ValueError("eeg_df must contain 'Timestamp' column.")
    if not {"start_time", "end_time"}.issubset(hrv_timings_df.columns):
        raise ValueError("hrv_timings_df must contain 'start_time' and 'end_time'.")

    labeled = eeg_df.copy()
    labeled[phase_column] = default_label

    for phase_name, row in hrv_timings_df.iterrows():
        mask = (labeled["Timestamp"] >= row["start_time"]) & (labeled["Timestamp"] <= row["end_time"])
        labeled.loc[mask, phase_column] = phase_name

    return labeled


