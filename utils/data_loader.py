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
        raw_text = Path(file_source).read_text(encoding="utf-8")
    elif isinstance(file_source, BytesIO):
        raw_text = file_source.getvalue().decode("utf-8")
        file_source.seek(0)
    elif isinstance(file_source, StringIO):
        raw_text = file_source.getvalue()
        file_source.seek(0)
    else:
        raw_text = str(file_source)

    lines = [line.rstrip("\n") for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("EEG file appears to be empty.")

    header = [col.strip() for col in lines[0].lstrip("\t").replace("\t", "").split(",")]
    data_rows: List[List[str]] = []
    expected_cols = len(header)

    for line in lines[1:]:
        cleaned = line.lstrip("\t").replace("\t", "")
        parts = [part.strip() for part in cleaned.split(",")]
        if not parts:
            continue
        if len(parts) >= expected_cols:
            base = parts[: expected_cols - 1]
            eeg_part = ",".join(parts[expected_cols - 1 :])
            base.append(eeg_part)
            parts = base
        else:
            parts.extend([""] * (expected_cols - len(parts)))
        data_rows.append(parts[:expected_cols])

    df = pd.DataFrame(data_rows, columns=header)

    # Normalise column labels to make matching easier (e.g. "时间", " Timestamp ")
    df.columns = [str(col).strip() for col in df.columns]

    candidate_names = [name for name in ("Timestamp", "时间", "Time", "time", "Datetime", "datetime") if name in df.columns]
    if candidate_names:
        timestamp_col = candidate_names[0]
    else:
        timestamp_col = df.columns[0]

    if timestamp_col != "Timestamp":
        df = df.rename(columns={timestamp_col: "Timestamp"})
    print(f"[EEG Loader] Timestamp column resolved: '{timestamp_col}' -> 'Timestamp'")

    def _clean_timestamp_cell(value):
        if value is None:
            return ""
        text = str(value).strip()
        text = text.replace("\t", " ")
        if "," in text:
            text = text.split(",", 1)[0]
        parts = text.split()
        if len(parts) >= 2:
            text = f"{parts[0]} {parts[1]}"
        elif parts:
            text = parts[0]
        else:
            text = ""
        return text

    df["Timestamp"] = df["Timestamp"].apply(_clean_timestamp_cell)

    parsed = pd.to_datetime(
        df["Timestamp"],
        format=timestamp_format,
        errors="coerce",
        infer_datetime_format=True,
    )
    if parsed.notna().any():
        df["Timestamp"] = parsed
        print("[EEG Loader] Parsed timestamps (first 5 rows):")
        print(df[["Timestamp"]].head())
    else:
        sample = df["Timestamp"].head(5).tolist()
        raise ValueError(f"Failed to parse any timestamps in EEG file. Sample values: {sample}")

    for col in df.columns:
        if col not in {"Timestamp", "EEG"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


