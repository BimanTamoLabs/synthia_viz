"""
Data processing utilities for EEG and HRV inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import json

import numpy as np
import pandas as pd


PERCENTAGE_COLUMNS = ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"]
CONSCIOUSNESS_COLUMNS = ["CSI", "EMG", "SQI", "BS"]


def process_eeg_data(
    df: pd.DataFrame,
    eeg_prefix: str = "EEG",
    drop_original_channels: bool = True,
) -> pd.DataFrame:
    """
    Clean and augment an EEG dataframe.
    """
    frame = df.copy()

    frame = frame.sort_values("Timestamp").reset_index(drop=True)

    eeg_cols = [col for col in frame.columns if col.startswith(eeg_prefix)]
    if eeg_cols:
        frame["EEG_Values"] = frame[eeg_cols].apply(lambda row: row.values.tolist(), axis=1)
        frame["EEG_Count"] = frame["EEG_Values"].apply(len)
        frame["EEG_Min"] = frame["EEG_Values"].apply(lambda values: min(values) if values else np.nan)
        frame["EEG_Max"] = frame["EEG_Values"].apply(lambda values: max(values) if values else np.nan)
        if drop_original_channels:
            frame = frame.drop(columns=eeg_cols)

    if all(col in frame.columns for col in PERCENTAGE_COLUMNS):
        frame = frame.dropna(subset=PERCENTAGE_COLUMNS)

    return frame


def get_column_statistics(df: pd.DataFrame, exclude_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns excluding provided names.
    """
    if df.empty:
        return pd.DataFrame()

    if exclude_columns:
        include = [col for col in df.columns if col not in exclude_columns]
    else:
        include = list(df.columns)

    numeric_cols = df[include].select_dtypes(include="number")
    return numeric_cols.describe().transpose()


def validate_eeg_data(df: pd.DataFrame) -> dict:
    """
    Perform validation checks against an EEG dataframe.
    """
    if df.empty or "Timestamp" not in df.columns:
        raise ValueError("EEG dataframe must be non-empty and include 'Timestamp'.")

    start_time = df["Timestamp"].min()
    end_time = df["Timestamp"].max()
    duration = end_time - start_time

    results = {
        "start_time": start_time,
        "end_time": end_time,
        "total_duration": duration,
        "valid_data_points": int(len(df)),
        "estimated_sampling_rate_hz": None,
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "anomalies": {},
    }

    if duration.total_seconds() > 0 and len(df) > 1:
        results["estimated_sampling_rate_hz"] = round((len(df) - 1) / duration.total_seconds(), 2)

    anomalies = {}
    for col in PERCENTAGE_COLUMNS:
        if col in df.columns:
            count_neg = int((df[col] < 0).sum())
            if count_neg:
                anomalies[f"negative_values_in_{col}"] = count_neg
    results["anomalies"] = anomalies

    if all(col in df.columns for col in PERCENTAGE_COLUMNS):
        percentage_sum = df[PERCENTAGE_COLUMNS].sum(axis=1)
        violations = int((~percentage_sum.between(99, 101)).sum())
        results["percentage_sum_check_failed_rows"] = violations
    else:
        results["percentage_sum_check_failed_rows"] = None

    return results


def load_hrv_json_to_df(json_source: Union[str, Path, dict, bytes]) -> pd.DataFrame:
    """
    Flatten a VibeScience HRV JSON file into a tidy dataframe.
    """
    if isinstance(json_source, (str, Path)):
        with open(Path(json_source), "r") as fp:
            payload = json.load(fp)
    elif isinstance(json_source, bytes):
        payload = json.loads(json_source.decode("utf-8"))
    elif isinstance(json_source, dict):
        payload = json_source
    else:
        raise ValueError("Unsupported json_source type.")

    records: List[dict] = []

    def _coerce_timestamp(value):
        if value is None:
            return pd.NaT
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return pd.NaT
            if stripped.isdigit():
                value = int(stripped)
            else:
                return pd.to_datetime(stripped, errors="coerce")
        if isinstance(value, (int, float)):
            magnitude = abs(int(value))
            if magnitude >= 10**15:
                unit = "ns"
            elif magnitude >= 10**12:
                unit = "ms"
            elif magnitude >= 10**9:
                unit = "s"
            elif magnitude >= 10**6:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(value, unit=unit, errors="coerce")
        return pd.to_datetime(value, errors="coerce")

    for detail in payload.get("details", []):
        phase = detail.get("testType", "unknown")
        for metric in detail.get("metrics", []):
            raw = metric.get("raw")
            if not raw:
                continue
            raw_entries = raw if isinstance(raw, list) else [raw]
            start_reference = None
            for key in ("startTime", "start_time", "startTimestamp", "start_datetime", "startDateTime"):
                if key in detail:
                    start_reference = _coerce_timestamp(detail.get(key))
                    if not pd.isna(start_reference):
                        break

            for entry in raw_entries:
                row = dict(entry)
                row["testType"] = phase

                timestamp_value = None
                for container in (metric, entry):
                    if not isinstance(container, dict):
                        continue
                    for key in ("on", "timestamp", "Timestamp", "timeStamp", "time", "Time", "datetime", "dateTime"):
                        candidate = container.get(key)
                        if candidate not in (None, ""):
                            timestamp_value = candidate
                            break
                    if timestamp_value is not None:
                        break
                timestamp_raw = timestamp_value
                timestamp = _coerce_timestamp(timestamp_value)
                if timestamp_value not in (None, ""):
                    if (pd.isna(timestamp) or (isinstance(timestamp, pd.Timestamp) and timestamp.year <= 1971)):
                        try:
                            numeric_val = float(timestamp_value)
                        except (ValueError, TypeError):
                            numeric_val = None
                        if numeric_val is not None:
                            if start_reference is not None and not pd.isna(start_reference):
                                unit = "ms" if abs(numeric_val) >= 1000 else "s"
                                timestamp = start_reference + pd.to_timedelta(numeric_val, unit=unit)
                            else:
                                timestamp = pd.NaT
                row["timestamp_raw"] = timestamp_raw
                row["timestamp"] = timestamp
                records.append(row)

    if not records:
        return pd.DataFrame(columns=["timestamp", "testType"])

    frame = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return frame


def get_testType_timings(hrv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-phase start, end, and duration derived from HRV dataframe.
    """
    if hrv_df.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "duration"])

    if not {"timestamp", "testType"}.issubset(hrv_df.columns):
        raise ValueError("hrv_df must contain 'timestamp' and 'testType' columns.")

    timings = (
        hrv_df.groupby("testType")["timestamp"]
        .agg(start_time="min", end_time="max")
        .assign(duration=lambda d: d["end_time"] - d["start_time"])
        .sort_values("start_time")
    )
    return timings


def process_hrv_json(
    json_source: Union[str, Path, dict, bytes],
    save_summary_csv: bool = False,
    output_prefix: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Parse HRV JSON and compute per-phase and overall summaries.
    """
    df = load_hrv_json_to_df(json_source)
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary_phase = df.groupby("testType")[numeric_cols].mean().round(2)
    summary_overall = df[numeric_cols].mean().to_frame("overall_mean").round(2)

    pre_mask = df["testType"].str.contains("preRun", case=False, na=False)
    post_mask = df["testType"].str.contains("postRun", case=False, na=False)
    delta = None
    if pre_mask.any() and post_mask.any():
        pre_mean = df.loc[pre_mask, numeric_cols].mean()
        post_mean = df.loc[post_mask, numeric_cols].mean()
        delta = (post_mean - pre_mean).to_frame("post_minus_pre").round(2)

    if save_summary_csv:
        if output_prefix is None:
            raise ValueError("output_prefix is required when save_summary_csv=True.")
        prefix = Path(output_prefix)
        summary_phase.to_csv(prefix.with_name(f"{prefix.stem}_hrv_summary_phase.csv"))
        summary_overall.to_csv(prefix.with_name(f"{prefix.stem}_hrv_summary_overall.csv"))
        if delta is not None:
            delta.to_csv(prefix.with_name(f"{prefix.stem}_hrv_summary_delta.csv"))

    return df, summary_phase, summary_overall, delta


