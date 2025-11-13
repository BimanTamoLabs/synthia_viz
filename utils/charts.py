"""
Plotly-based visualisations with consistent phase highlighting.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHASE_COLORS = {
    "preRun1": "#a2d5ab",
    "preRun2": "#6aa84f",
    "liveRun": "#6fa8dc",
    "postRun1": "#f4cccc",
    "postRun2": "#e06666",
    "OUT": "#eeeeee",
}


def _phase_windows(timings: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    windows = []
    for phase_name, row in timings.iterrows():
        windows.append((row["start_time"], row["end_time"], phase_name))
    return windows


def _add_phase_shapes(fig: go.Figure, windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> None:
    for start, end, name in windows:
        color = PHASE_COLORS.get(name, "#cccccc")
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.15,
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=start, line_color=color, opacity=0.3, line_width=1)
        fig.add_vline(x=end, line_color=color, opacity=0.3, line_width=1)


def plot_phase_timeline(timings: pd.DataFrame) -> go.Figure:
    data = []
    for phase_name, row in timings.iterrows():
        data.append(dict(Task="Phase", Start=row["start_time"], Finish=row["end_time"], Phase=phase_name))
    fig = px.timeline(data, x_start="Start", x_end="Finish", y="Task", color="Phase", color_discrete_map=PHASE_COLORS)
    fig.update_layout(showlegend=True, yaxis_title="", xaxis_title="Time", bargap=0.1)
    return fig


def plot_eeg_timeseries_phases(
    df: pd.DataFrame,
    metrics: Sequence[str],
    phase_column: str,
    timings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    color_cycle = px.colors.qualitative.Set2
    for idx, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[metric],
                mode="lines",
                name=metric,
                line=dict(color=color_cycle[idx % len(color_cycle)]),
            )
        )
    if timings is not None and not timings.empty:
        _add_phase_shapes(fig, _phase_windows(timings))
    fig.update_layout(title="EEG Metrics Over Time", xaxis_title="Timestamp", yaxis_title="Value", hovermode="x unified")
    return fig


def plot_brainwave_stacked(
    df: pd.DataFrame,
    phase_column: str,
    timings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    bands = ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"]
    palette = {
        "Delta(%)": "#5b21b6",
        "Theta(%)": "#3b82f6",
        "Alpha(%)": "#10b981",
        "Beta(%)": "#f97316",
    }
    fig = go.Figure()
    for band in bands:
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[band],
                mode="lines",
                line=dict(color=palette.get(band, "#888888")),
                stackgroup="one",
                name=band,
            )
        )
    if timings is not None and not timings.empty:
        _add_phase_shapes(fig, _phase_windows(timings))
    fig.update_layout(title="Brainwave Distribution", xaxis_title="Timestamp", yaxis_title="Percentage")
    return fig


def plot_consciousness_metrics(
    df: pd.DataFrame,
    weights: dict,
    phase_column: str,
    timings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    required = ["CSI", "EMG", "SQI", "BS"]
    if not all(col in df.columns for col in required):
        raise ValueError("Dataframe missing consciousness metrics.")

    score_series = sum(df[col] * weights.get(col, 0) for col in required)
    metrics = required + ["Consciousness Score"]
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=metrics)

    for idx, metric in enumerate(metrics, start=1):
        series = df[metric] if metric in df.columns else score_series
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=series,
                mode="lines",
                name=metric,
            ),
            row=idx,
            col=1,
        )
        if timings is not None and not timings.empty:
            for start, end, name in _phase_windows(timings):
                color = PHASE_COLORS.get(name, "#cccccc")
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    row=idx,
                    col=1,
                )
                fig.add_vline(x=start, line_color=color, opacity=0.2, line_width=1, row=idx, col=1)
                fig.add_vline(x=end, line_color=color, opacity=0.2, line_width=1, row=idx, col=1)

    fig.update_layout(
        title="Consciousness Metrics",
        height=250 * len(metrics),
        showlegend=False,
        xaxis_title="Timestamp",
    )
    return fig


def plot_hrv_trends_phases(
    df: pd.DataFrame,
    metrics: Sequence[str],
    timings: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.D3
    for idx, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[metric],
                mode="lines",
                name=metric,
                line=dict(color=colors[idx % len(colors)]),
            )
        )
    if timings is not None and not timings.empty:
        _add_phase_shapes(fig, _phase_windows(timings))
    fig.update_layout(title="HRV Metrics Over Time", xaxis_title="Timestamp", yaxis_title="Value", hovermode="x unified")
    return fig


def plot_phase_comparison_bars(stats_df: pd.DataFrame, metrics: Sequence[str]) -> go.Figure:
    mean_df = stats_df.xs("mean", level=1, axis=1)
    mean_df = mean_df[metrics]
    mean_df.index.name = mean_df.index.name or "Phase"
    melted = mean_df.reset_index().melt(id_vars=mean_df.index.name, value_vars=metrics, var_name="Metric", value_name="Mean")
    fig = px.bar(melted, x=mean_df.index.name, y="Mean", color="Metric", barmode="group", text="Mean")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(title="Phase Comparison (Mean)", xaxis_title="Phase", yaxis_title="Mean Value", bargap=0.2)
    return fig


def plot_phase_comparison_heatmap(stats_df: pd.DataFrame, metrics: Sequence[str]) -> go.Figure:
    mean_df = stats_df.xs("mean", level=1, axis=1)[metrics]
    zmin = float(np.nanmin(mean_df.values))
    zmax = float(np.nanmax(mean_df.values))
    fig = go.Figure(data=go.Heatmap(z=mean_df.values, x=mean_df.columns, y=mean_df.index, colorscale="RdBu", zmin=zmin, zmax=zmax))
    fig.update_layout(title="Phase Comparison Heatmap", xaxis_title="Metric", yaxis_title="Phase")
    return fig


def plot_phase_count_bar(counts: pd.Series, title: str = "Samples per Phase") -> go.Figure:
    if isinstance(counts, pd.Series):
        phases = counts.index.tolist()
        values = counts.values.astype(int)
    else:
        phases = list(counts.keys())
        values = list(counts.values())
    colors = [PHASE_COLORS.get(phase, "#636EFA") for phase in phases]
    fig = go.Figure(
        data=go.Bar(
            x=phases,
            y=values,
            text=[f"{int(v)}" for v in values],
            textposition="outside",
            marker_color=colors,
        )
    )
    fig.update_layout(title=title, xaxis_title="Phase", yaxis_title="Samples", bargap=0.4)
    return fig


def plot_metric_histogram(values: pd.Series, hypothesis_row: Mapping[str, object]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=30, name="Metric"))
    min_value = hypothesis_row["Benchmark_Min"]
    max_value = hypothesis_row["Benchmark_Max"]
    fig.add_vline(x=min_value, line_color="green", line_dash="dash", annotation_text="Benchmark Min")
    fig.add_vline(x=max_value, line_color="red", line_dash="dash", annotation_text="Benchmark Max")
    fig.update_layout(title=f"Distribution: {hypothesis_row['Metric_Name']}", xaxis_title="Value", yaxis_title="Frequency")
    return fig


def plot_sqi_distribution(before_df: pd.DataFrame, after_df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=before_df["SQI"], nbinsx=30, name="Before", opacity=0.6))
    fig.add_trace(go.Histogram(x=after_df["SQI"], nbinsx=30, name="After", opacity=0.6))
    fig.add_vline(x=threshold, line_color="orange", line_dash="dash", annotation_text=f"Threshold {threshold}")
    fig.update_layout(barmode="overlay", title="SQI Distribution Before/After Filtering", xaxis_title="SQI", yaxis_title="Count")
    return fig


def plot_ratio_heatmap(ratio_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=ratio_df.values,
            x=ratio_df.columns,
            y=ratio_df.index,
            colorscale="RdBu",
            zmid=1.0,
        )
    )
    fig.update_layout(title="Phase Ratio Heatmap", xaxis_title="Metric", yaxis_title="Ratio Type")
    return fig


