"""
Utility helpers for loading participant metadata, EEG CSV files,
and hypothesis reference tables.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd


EEG_EXTENSIONS = {".csv"}
HRV_EXTENSIONS = {".json"}


def _normalise_path(path: Union[str, Path]) -> Path:
    """Return a resolved Path instance."""
    return Path(path).expanduser().resolve()


def get_needed_participants(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the participant lookup CSV and return rows where ``Needed?`` == ``Y``.

    Parameters
    ----------
    file_path:
        Path to ``LIVE_participant_ID_name_lookup.csv`` (or equivalent).

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least ``Date`` and ``KeyCode`` columns.
    """
    path = _normalise_path(file_path)
    df = pd.read_csv(path)

    required_cols = {"Date", "KeyCode", "Needed?"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Participant lookup missing columns: {sorted(missing)}")

    filtered = (
        df.assign(**{"Needed?": df["Needed?"].fillna("N")})
        .loc[lambda d: d["Needed?"].astype(str).str.strip().str.upper() == "Y", ["Date", "KeyCode"]]
        .reset_index(drop=True)
    )
    return filtered


def get_keycode_dict_by_date(participants_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group participant ``KeyCode`` entries by formatted date (``DD_MM_YY``).
    """
    if participants_df.empty:
        return {}

    if not {"Date", "KeyCode"}.issubset(participants_df.columns):
        raise ValueError("participants_df must contain 'Date' and 'KeyCode' columns.")

    df = participants_df.copy()
    df["datetime"] = pd.to_datetime(df["Date"], format="mixed")
    df["formatted_date"] = df["datetime"].dt.strftime("%d_%m_%y")
    return df.groupby("formatted_date")["KeyCode"].apply(list).to_dict()


def get_full_paths_for_date(
    date_keycode_map: Dict[str, Sequence[str]],
    date_key: str,
    base_path: Union[str, Path],
) -> List[Path]:
    """
    Construct full folder paths for a specific date key.
    """
    keycodes = date_keycode_map.get(date_key, [])
    base = _normalise_path(base_path)
    paths: List[Path] = []

    for raw_key in keycodes:
        try:
            keycode_int = int(float(raw_key))
        except (TypeError, ValueError):
            continue
        path = base / date_key / str(keycode_int)
        if path not in paths:
            paths.append(path)
    return paths


def find_files(
    main_folder_location: Union[str, Path],
    patient_id_folder: Optional[str] = None,
    file_type: str = "EEG",
) -> List[Path]:
    """
    Discover EEG CSV or HRV JSON files within a participant folder structure.
    """
    base = _normalise_path(main_folder_location)
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base}")

    if file_type == "EEG":
        extensions = EEG_EXTENSIONS
        keyword = "EEG"
    elif file_type == "HRV":
        extensions = HRV_EXTENSIONS
        keyword = "HRV"
    else:
        raise ValueError("file_type must be either 'EEG' or 'HRV'.")

    candidates: List[Path] = []

    def _scan(folder: Path) -> None:
        for child in folder.iterdir():
            if child.is_file() and child.suffix.lower() in extensions and keyword in child.name:
                candidates.append(child)

    if patient_id_folder:
        target = base / patient_id_folder
        if target.is_dir():
            _scan(target)
        else:
            raise FileNotFoundError(f"Participant directory not found: {target}")
    else:
        for subfolder in base.iterdir():
            if subfolder.is_dir():
                _scan(subfolder)

    return candidates


def load_eeg_file(
    file_source: Union[str, Path, BytesIO, StringIO],
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load an EEG CSV file and ensure the Timestamp column exists.
    """
    if isinstance(file_source, (str, Path)):
        df = pd.read_csv(_normalise_path(file_source))
    else:
        df = pd.read_csv(file_source)

    first_col = df.columns[0]
    if first_col != "Timestamp":
        df = df.rename(columns={first_col: "Timestamp"})

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format=timestamp_format,
        errors="coerce",
    )
    if df["Timestamp"].isna().all():
        raise ValueError("Failed to parse any timestamps in EEG file.")

    if "SQI" not in df.columns:
        raise ValueError("EEG file missing required 'SQI' column.")

    return df


def load_hypothesis_table(file_source: Union[str, Path, BytesIO, StringIO]) -> pd.DataFrame:
    """
    Load the hypothesis reference table and validate required columns.
    """
    if isinstance(file_source, (str, Path)):
        df = pd.read_csv(_normalise_path(file_source))
    else:
        df = pd.read_csv(file_source)

    required_cols = {
        "Hypothesis_ID",
        "Category",
        "Metric_Name",
        "Input_Pre",
        "Input_Post",
        "Equation",
        "Benchmark_Min",
        "Benchmark_Max",
        "Direction",
        "Interpretation",
        "Meaning",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Hypothesis table missing columns: {sorted(missing)}")

    if df["Hypothesis_ID"].duplicated().any():
        duplicates = df[df["Hypothesis_ID"].duplicated()]["Hypothesis_ID"].unique()
        raise ValueError(f"Duplicate Hypothesis_ID values found: {duplicates}")

    return df


