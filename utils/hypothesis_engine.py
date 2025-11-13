"""
Hypothesis evaluation engine for EEG/HRV phase-based metrics.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PHASE_ALIASES: Mapping[str, Sequence[str]] = {
    "pre": ["preRun1", "preRun2"],
    "post": ["postRun1", "postRun2"],
    "live": ["liveRun"],
    "prerun1": ["preRun1"],
    "prerun2": ["preRun2"],
    "postrun1": ["postRun1"],
    "postrun2": ["postRun2"],
}

VARIABLE_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
ALLOWED_SYMBOLS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/().% ")


def sanitize_equation(equation: str) -> str:
    """
    Ensure equation contains only allowed characters/operators.
    """
    if not equation or not isinstance(equation, str):
        raise ValueError("Equation must be a non-empty string.")

    cleaned = equation.replace("^", "**")
    illegal = set(cleaned) - ALLOWED_SYMBOLS
    if illegal:
        raise ValueError(f"Equation contains illegal characters: {illegal}")
    if "__" in cleaned:
        raise ValueError("Equation must not contain double underscores.")
    return cleaned.strip()


def extract_variables_from_equation(equation: str) -> List[str]:
    """
    Extract variable identifiers from an equation string.
    """
    sanitized = sanitize_equation(equation)
    tokens = VARIABLE_PATTERN.findall(sanitized)
    python_keywords = {"and", "or", "not"}
    return sorted({token for token in tokens if token.lower() not in python_keywords})


def _sanitize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _create_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {_sanitize_label(col): col for col in df.columns}


def _split_variable(variable: str) -> Tuple[str, Optional[str]]:
    if "_" not in variable:
        return variable, None
    metric, suffix = variable.rsplit("_", 1)
    return metric, suffix.lower()


def _resolve_column(df: pd.DataFrame, metric_name: str) -> str:
    columns = _create_column_lookup(df)
    key = _sanitize_label(metric_name)
    if key in columns:
        return columns[key]
    raise KeyError(f"Could not resolve metric '{metric_name}' to dataframe columns.")


def _phase_rows(df: pd.DataFrame, suffix: Optional[str], phase_column: str) -> pd.Series:
    if suffix is None:
        return pd.Series(True, index=df.index)
    phases = PHASE_ALIASES.get(suffix)
    if not phases:
        raise KeyError(f"Unsupported phase suffix '{suffix}' in equation.")
    return df[phase_column].isin(phases)


def _aggregated_value(df: pd.DataFrame, column: str, mask: pd.Series) -> float:
    series = pd.to_numeric(df.loc[mask, column], errors="coerce")
    if series.empty:
        return math.nan
    return float(series.mean())


def _build_aggregated_context(
    df: pd.DataFrame,
    variables: Sequence[str],
    phase_column: str,
) -> Dict[str, float]:
    context: Dict[str, float] = {}
    for variable in variables:
        metric, suffix = _split_variable(variable)
        column = _resolve_column(df, metric)
        mask = _phase_rows(df, suffix, phase_column)
        context[variable] = _aggregated_value(df, column, mask)
    return context


def _build_per_sample_context(
    df: pd.DataFrame,
    variables: Sequence[str],
    phase_column: str,
) -> pd.DataFrame:
    context = pd.DataFrame(index=df.index)

    aggregated_cache: Dict[Tuple[str, Optional[str]], float] = {}

    for variable in variables:
        metric, suffix = _split_variable(variable)
        column = _resolve_column(df, metric)
        series = pd.Series(np.nan, index=df.index, dtype=float)

        if suffix is None:
            series[:] = pd.to_numeric(df[column], errors="coerce")
        else:
            phases = PHASE_ALIASES.get(suffix)
            if not phases:
                raise KeyError(f"Unsupported phase suffix '{suffix}' in equation.")
            mask = df[phase_column].isin(phases)
            if suffix == "post" or suffix == "live" or suffix.startswith("postrun"):
                series.loc[mask] = pd.to_numeric(df.loc[mask, column], errors="coerce")
            else:
                key = (column, suffix)
                if key not in aggregated_cache:
                    aggregated_cache[key] = _aggregated_value(df, column, mask)
                series[:] = aggregated_cache[key]
        context[variable] = series

    return context


def compute_metric_aggregated(
    df: pd.DataFrame,
    equation: str,
    phase_column: str = "testType",
) -> float:
    """
    Evaluate equation on aggregated phase statistics.
    """
    variables = extract_variables_from_equation(equation)
    context = _build_aggregated_context(df, variables, phase_column)
    sanitized = sanitize_equation(equation)
    result = pd.eval(sanitized, local_dict=context, engine="python")
    return float(result)


def compute_metric_per_sample(
    df: pd.DataFrame,
    equation: str,
    phase_column: str = "testType",
) -> pd.Series:
    """
    Evaluate equation for each sample (where possible).
    """
    variables = extract_variables_from_equation(equation)
    context_df = _build_per_sample_context(df, variables, phase_column)
    sanitized = sanitize_equation(equation)
    result = context_df.eval(sanitized, engine="python")
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def evaluate_benchmark(values: pd.Series, benchmark_min: float, benchmark_max: float) -> Dict[str, float]:
    """
    Compare metric series against benchmark bounds.
    """
    valid = values.dropna()
    within = valid.between(benchmark_min, benchmark_max)
    percent_in_range = float(within.mean() * 100) if not valid.empty else 0.0
    return {
        "valid_samples": int(valid.count()),
        "within_benchmark_count": int(within.sum()),
        "within_benchmark_percent": percent_in_range,
        "passes_benchmark": percent_in_range >= 80.0,
    }


def evaluate_hypothesis(
    df: pd.DataFrame,
    hypothesis_row: Mapping[str, object],
    phase_column: str = "testType",
    mode: str = "aggregated",
) -> Dict[str, object]:
    """
    Evaluate a single hypothesis definition against the dataframe.
    """
    equation = str(hypothesis_row["Equation"])
    benchmark_min = float(hypothesis_row["Benchmark_Min"])
    benchmark_max = float(hypothesis_row["Benchmark_Max"])

    if mode == "aggregated":
        metric_value = compute_metric_aggregated(df, equation, phase_column=phase_column)
        metric_series = pd.Series([metric_value], name=hypothesis_row["Hypothesis_ID"])
    elif mode == "per-sample":
        metric_series = compute_metric_per_sample(df, equation, phase_column=phase_column)
    else:
        raise ValueError("mode must be 'aggregated' or 'per-sample'.")

    stats = {
        "metric_mean": float(metric_series.mean()) if not metric_series.empty else math.nan,
        "metric_std": float(metric_series.std(ddof=0)) if len(metric_series) > 1 else math.nan,
        "metric_median": float(metric_series.median()) if not metric_series.empty else math.nan,
        "metric_min": float(metric_series.min()) if not metric_series.empty else math.nan,
        "metric_max": float(metric_series.max()) if not metric_series.empty else math.nan,
    }

    benchmark = evaluate_benchmark(metric_series, benchmark_min, benchmark_max)

    return {
        "hypothesis_id": hypothesis_row["Hypothesis_ID"],
        "category": hypothesis_row["Category"],
        "metric_name": hypothesis_row["Metric_Name"],
        "equation": equation,
        "metric_values": metric_series,
        **stats,
        **benchmark,
        "benchmark_min": benchmark_min,
        "benchmark_max": benchmark_max,
        "direction": hypothesis_row.get("Direction"),
        "interpretation": hypothesis_row.get("Interpretation"),
        "meaning": hypothesis_row.get("Meaning"),
    }


def batch_evaluate_hypotheses(
    df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
    hypothesis_ids: Optional[Sequence[str]] = None,
    phase_column: str = "testType",
    mode: str = "aggregated",
) -> List[Dict[str, object]]:
    """
    Evaluate multiple hypotheses and return list of result dicts.
    """
    if hypothesis_ids:
        hdf = hypothesis_df[hypothesis_df["Hypothesis_ID"].isin(hypothesis_ids)]
    else:
        hdf = hypothesis_df

    results: List[Dict[str, object]] = []
    for _, row in hdf.iterrows():
        results.append(evaluate_hypothesis(df, row, phase_column=phase_column, mode=mode))
    return results


