"""
Statistical aggregations for EEG and HRV datasets.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def calculate_eeg_statistics(
    eeg_labeled_df: pd.DataFrame,
    metrics: Sequence[str],
    phase_column: str = "testType",
    exclude_phase: str = "OUT",
) -> pd.DataFrame:
    """
    Compute per-phase statistics (mean, min, max, std) for EEG metrics.
    """
    if eeg_labeled_df.empty:
        return pd.DataFrame()

    required_cols = set(metrics) | {phase_column, "Timestamp"}
    missing = required_cols.difference(eeg_labeled_df.columns)
    if missing:
        raise ValueError(f"Dataframe missing required columns: {sorted(missing)}")

    df = eeg_labeled_df[eeg_labeled_df[phase_column] != exclude_phase].copy()
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    stats = df.groupby(phase_column)[metrics].agg(["mean", "min", "max", "std"]).round(3)
    ordering = df.groupby(phase_column)["Timestamp"].min().sort_values().index
    stats = stats.reindex(ordering)
    return stats


def calculate_phase_ratios(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate phase comparison ratios based on mean statistics.
    """
    if stats_df.empty:
        return pd.DataFrame()
    if "mean" not in stats_df.columns.get_level_values(1):
        raise ValueError("stats_df must contain mean statistics.")

    mean_stats = stats_df.xs("mean", level=1, axis=1)

    def _get_phase(name: str) -> pd.Series:
        if name not in mean_stats.index:
            return pd.Series(np.nan, index=mean_stats.columns)
        return mean_stats.loc[name]

    pre1 = _get_phase("preRun1")
    pre2 = _get_phase("preRun2")
    post1 = _get_phase("postRun1")
    post2 = _get_phase("postRun2")
    live = _get_phase("liveRun")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = pd.DataFrame(
            {
                "pre1_div_post1": pre1 / post1,
                "pre2_div_post2": pre2 / post2,
                "pre_all_div_post_all": (pre1 + pre2) / (post1 + post2),
                "pre_all_div_live": (pre1 + pre2) / live,
                "live_div_post_all": live / (post1 + post2),
            }
        ).transpose()

    return ratios.replace([np.inf, -np.inf], np.nan).round(3)


def calculate_hrv_statistics_by_phase(hrv_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate per-phase HRV statistics and deltas.
    """
    if hrv_df.empty:
        return {"per_phase": pd.DataFrame(), "delta": pd.DataFrame(), "overall": pd.DataFrame()}

    numeric_cols = hrv_df.select_dtypes(include="number").columns.tolist()
    per_phase = hrv_df.groupby("testType")[numeric_cols].agg(["mean", "std", "min", "max"]).round(3)

    pre_mask = hrv_df["testType"].str.contains("preRun", case=False, na=False)
    post_mask = hrv_df["testType"].str.contains("postRun", case=False, na=False)
    delta = pd.DataFrame()
    if pre_mask.any() and post_mask.any():
        pre_mean = hrv_df.loc[pre_mask, numeric_cols].mean()
        post_mean = hrv_df.loc[post_mask, numeric_cols].mean()
        delta = (post_mean - pre_mean).to_frame("post_minus_pre").round(3)

    overall = hrv_df[numeric_cols].agg(["mean", "std", "min", "max"]).round(3)
    return {"per_phase": per_phase, "delta": delta, "overall": overall}


