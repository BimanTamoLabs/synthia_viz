from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from utils import alignment, data_loader, data_processor, hypothesis_engine, sqi_filter, statistics
from utils import charts

PHASE_COLUMN = "testType"
PHASE_ORDER = ["preRun1", "preRun2", "liveRun", "postRun1", "postRun2", "OUT"]


def initialise_session_state() -> None:
    defaults = {
        "eeg_df_raw": None,
        "hrv_df": None,
        "phase_timings": None,
        "eeg_df_labeled": None,
        "eeg_df_filtered": None,
        "filter_stats": None,
        "hypothesis_table_df": None,
        "hypothesis_mode": "aggregated",
        "use_filtered_for_hypothesis": True,
        "consciousness_weights": {"CSI": 0.4, "EMG": 0.2, "SQI": 0.2, "BS": 0.2},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def load_eeg_from_upload(file, timestamp_format: Optional[str] = None) -> pd.DataFrame:
    return data_loader.load_eeg_file(file, timestamp_format=timestamp_format)


@st.cache_data(show_spinner=False)
def load_hypothesis_table_from_upload(file) -> pd.DataFrame:
    return data_loader.load_hypothesis_table(file)


def sidebar_participant_section():
    st.sidebar.header("Participant Selection")
    data_mode = st.sidebar.radio("Data Input Mode", ["Individual Files", "Participant Lookup"], key="data_mode")
    lookup_df = None
    date_keycode_map: Dict[str, List[str]] = {}
    selected_date = None
    selected_participant = None

    if data_mode == "Participant Lookup":
        lookup_file = st.sidebar.file_uploader("Upload Participant Lookup CSV", type=["csv"])
        if lookup_file is not None:
            lookup_df = data_loader.get_needed_participants(lookup_file)
            date_keycode_map = data_loader.get_keycode_dict_by_date(lookup_df)
            if date_keycode_map:
                selected_date = st.sidebar.selectbox("Select Session Date", sorted(date_keycode_map.keys()))
                participants = date_keycode_map.get(selected_date, [])
                selected_participant = st.sidebar.selectbox("Select Participant", participants)
                st.sidebar.caption(f"Participants available: {len(participants)}")

    return data_mode, lookup_df, date_keycode_map, selected_date, selected_participant


def sidebar_data_loading_section(data_mode: str, selected_participant: Optional[str]):
    st.sidebar.header("Data Loading")
    eeg_file = None
    hrv_file = None

    if data_mode == "Individual Files":
        eeg_file = st.sidebar.file_uploader("Upload EEG CSV", type=["csv"], key="eeg_individual")
        hrv_file = st.sidebar.file_uploader("Upload HRV JSON", type=["json"], key="hrv_individual")
    else:
        st.sidebar.caption("Upload EEG and HRV files for batch analysis. Files are matched to participants by filename.")
        uploaded_eeg_files = st.sidebar.file_uploader(
            "Upload EEG CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="eeg_batch",
        )
        uploaded_hrv_files = st.sidebar.file_uploader(
            "Upload HRV JSON files",
            type=["json"],
            accept_multiple_files=True,
            key="hrv_batch",
        )
        if selected_participant:
            try:
                participant_key = str(int(float(selected_participant)))
            except (TypeError, ValueError):
                participant_key = str(selected_participant).strip()
            if uploaded_eeg_files:
                eeg_matches = [f for f in uploaded_eeg_files if participant_key in f.name]
                if eeg_matches:
                    eeg_file = eeg_matches[0]
                    st.sidebar.success(f"Matched EEG file: {eeg_file.name}")
                else:
                    st.sidebar.warning("No EEG file matched the selected participant.")
            if uploaded_hrv_files:
                hrv_matches = [f for f in uploaded_hrv_files if participant_key in f.name]
                if hrv_matches:
                    hrv_file = hrv_matches[0]
                    st.sidebar.success(f"Matched HRV file: {hrv_file.name}")
                else:
                    st.sidebar.warning("No HRV file matched the selected participant.")

    load_eeg = st.sidebar.button("Load EEG Data", key="load_eeg_button", disabled=eeg_file is None)
    load_hrv = st.sidebar.button("Load HRV Data", key="load_hrv_button", disabled=hrv_file is None)

    if load_eeg and eeg_file is not None:
        with st.spinner("Loading EEG data..."):
            eeg_df = load_eeg_from_upload(eeg_file)
            eeg_df = data_processor.process_eeg_data(eeg_df)
            st.session_state["eeg_df_raw"] = eeg_df
        st.sidebar.success(f"EEG loaded: {len(st.session_state['eeg_df_raw'])} samples")

    if load_hrv and hrv_file is not None:
        with st.spinner("Loading HRV data..."):
            if hasattr(hrv_file, "getvalue"):
                payload = json.loads(hrv_file.getvalue().decode("utf-8"))
            else:
                payload = json.load(hrv_file)
            hrv_df = data_processor.load_hrv_json_to_df(payload)
            st.session_state["hrv_df"] = hrv_df
            st.sidebar.success(f"HRV loaded: {hrv_df[PHASE_COLUMN].nunique()} phases")


def sidebar_alignment_section():
    st.sidebar.header("Alignment")
    eeg_df = st.session_state.get("eeg_df_raw")
    hrv_df = st.session_state.get("hrv_df")
    can_align = eeg_df is not None and hrv_df is not None
    align_button = st.sidebar.button("Align EEG with HRV Phases", disabled=not can_align)

    if align_button and can_align:
        with st.spinner("Validating timestamps and assigning phases..."):
            timings = data_processor.get_testType_timings(hrv_df)
            st.session_state["phase_timings"] = timings

            alignment_report = alignment.validate_timestamp_alignment(eeg_df, hrv_df)
            st.session_state["alignment_report"] = alignment_report
            if alignment_report["is_valid"]:
                st.sidebar.success("Timestamp coverage verified.")
            else:
                st.sidebar.warning("Timestamp ranges do not fully overlap. Please verify data coverage.")
            labeled = alignment.assign_hrv_testTypes_to_eeg(eeg_df, timings, phase_column=PHASE_COLUMN)
            st.session_state["eeg_df_labeled"] = labeled
        st.sidebar.success("Phase alignment complete.")


def sidebar_sqi_section():
    st.sidebar.header("Signal Quality Filtering")
    mode = st.sidebar.selectbox(
        "Filter Mode",
        ["None", "Keep SQI >= threshold", "Keep SQI <= threshold"],
        index=1,
        key="sqi_mode",
    )
    threshold = st.sidebar.slider("SQI Threshold", 0, 100, 70, key="sqi_threshold")
    apply_filter = st.sidebar.button("Apply Filter", key="apply_sqi_button", disabled=st.session_state.get("eeg_df_labeled") is None)

    if apply_filter:
        labeled = st.session_state.get("eeg_df_labeled")
        filtered_df, stats = sqi_filter.apply_sqi_filter(labeled, mode, threshold)
        st.session_state["eeg_df_filtered"] = filtered_df
        st.session_state["filter_stats"] = stats
        st.sidebar.success("SQI filter applied.")
        st.sidebar.json(stats)


def sidebar_analysis_configuration():
    st.sidebar.header("Analysis Settings")
    use_filtered = st.sidebar.checkbox("Use filtered data for hypothesis testing", value=True)
    st.session_state["use_filtered_for_hypothesis"] = use_filtered

    with st.sidebar.expander("Adjust Consciousness Weights"):
        weights = {}
        total = 0.0
        for key in ["CSI", "EMG", "SQI", "BS"]:
            default = st.session_state["consciousness_weights"].get(key, 0.25)
            weight = st.slider(f"{key} weight", 0.0, 1.0, float(default), 0.05, key=f"weight_{key}")
            weights[key] = weight
            total += weight
        if abs(total - 1.0) > 1e-6:
            st.warning("Weights must sum to 1.0; values will be normalised automatically.")
            weights = {k: v / total if total else 0.0 for k, v in weights.items()}
        st.session_state["consciousness_weights"] = weights

    hypothesis_mode = st.sidebar.radio("Hypothesis Evaluation Mode", ["aggregated", "per-sample"], key="hypothesis_mode_radio")
    st.session_state["hypothesis_mode"] = hypothesis_mode

    hypothesis_file = st.sidebar.file_uploader("Upload Hypothesis Table", type=["csv"], key="hypothesis_table")
    if hypothesis_file is not None:
        st.session_state["hypothesis_table_df"] = load_hypothesis_table_from_upload(hypothesis_file)
        st.sidebar.success("Hypothesis table loaded.")


def summarise_data_overview():
    eeg_df = st.session_state.get("eeg_df_labeled")
    filtered_df = st.session_state.get("eeg_df_filtered")
    hrv_df = st.session_state.get("hrv_df")
    timings = st.session_state.get("phase_timings")

    st.subheader("Data Overview")
    cols = st.columns(4)

    active_df = filtered_df if filtered_df is not None else eeg_df

    if active_df is not None and not active_df.empty:
        sample_delta = None
        if filtered_df is not None and eeg_df is not None:
            sample_delta = len(filtered_df) - len(eeg_df)
        cols[0].metric("EEG Samples (usable)", len(active_df), sample_delta)
        duration = data_processor.validate_eeg_data(active_df)["total_duration"]
        cols[1].metric("EEG Duration", str(duration) if duration else "N/A")
        cols[2].metric("Average SQI", round(active_df["SQI"].mean(), 2))
        phase_counts = active_df[PHASE_COLUMN].value_counts().reindex(PHASE_ORDER, fill_value=0)
        cols[3].metric("Samples per Phase", ", ".join(f"{phase}: {int(count)}" for phase, count in phase_counts.items() if int(count) > 0))
    else:
        if filtered_df is None and st.session_state.get("eeg_df_raw") is not None:
            cols[0].info("Run 'Align EEG with HRV Phases' to prepare data for analysis.")
        else:
            cols[0].info("EEG data not loaded.")

    if hrv_df is not None:
        st.caption(f"HRV samples: {len(hrv_df)} • Phases detected: {hrv_df[PHASE_COLUMN].nunique()}")
    if timings is not None and not timings.empty:
        st.plotly_chart(charts.plot_phase_timeline(timings), use_container_width=True)

    report = st.session_state.get("alignment_report")
    if report:
        status = "OK" if report["is_valid"] else "WARNING"
        st.caption(
            f"[{status}] Alignment: EEG [{report['eeg_start']} -> {report['eeg_end']}] | HRV [{report['hrv_start']} -> {report['hrv_end']}]"
        )
    if filtered_df is not None and st.session_state.get("filter_stats"):
        stats = st.session_state["filter_stats"]
        st.caption(
            f"SQI filter applied: {stats['filtered_count']} of {stats['original_count']} samples retained ({stats['percent_removed']:.1f}% removed)."
        )
    if active_df is not None and not active_df.empty:
        phase_counts = active_df[PHASE_COLUMN].value_counts().reindex(PHASE_ORDER, fill_value=0)
        if phase_counts.sum() > 0:
            st.plotly_chart(charts.plot_phase_count_bar(phase_counts), use_container_width=True)


def eeg_analysis_tab():
    eeg_df = st.session_state.get("eeg_df_labeled")
    timings = st.session_state.get("phase_timings")
    filtered = st.session_state.get("eeg_df_filtered")

    if eeg_df is None:
        st.info("Load and align EEG data to view this tab.")
        return

    active_df = filtered if filtered is not None else eeg_df

    st.subheader("EEG Analysis")
    if filtered is not None:
        st.caption(f"Displaying SQI-filtered dataset ({len(filtered)} samples).")

    with st.expander("Data Preview", expanded=True):
        columns = st.multiselect("Select columns to display", active_df.columns.tolist(), default=active_df.columns.tolist()[:6])
        st.dataframe(active_df[columns].head(50))

    metrics = [col for col in ["CSI", "EMG", "SQI", "BS", "Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"] if col in active_df.columns]
    default_metrics = [metric for metric in ["SQI", "CSI"] if metric in metrics] or metrics[:2]
    selected_metrics = st.multiselect("Select time series metrics", metrics, default=default_metrics)
    if selected_metrics:
        fig = charts.plot_eeg_timeseries_phases(active_df, selected_metrics, PHASE_COLUMN, timings)
        st.plotly_chart(fig, use_container_width=True)

    if all(band in active_df.columns for band in ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"]):
        stacked_fig = charts.plot_brainwave_stacked(active_df, PHASE_COLUMN, timings)
        st.plotly_chart(stacked_fig, use_container_width=True)

    weights = st.session_state["consciousness_weights"]
    try:
        consciousness_fig = charts.plot_consciousness_metrics(active_df, weights, PHASE_COLUMN, timings)
        st.plotly_chart(consciousness_fig, use_container_width=True)
    except ValueError as exc:
        st.warning(str(exc))

    stats = statistics.calculate_eeg_statistics(active_df, ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"], phase_column=PHASE_COLUMN)
    if not stats.empty:
        st.dataframe(stats)
        ratio_df = statistics.calculate_phase_ratios(stats)
        if not ratio_df.empty:
            st.plotly_chart(charts.plot_ratio_heatmap(ratio_df), use_container_width=True)

    if filtered is not None:
        st.markdown("#### SQI Filtering Impact")
        threshold = st.session_state.get("sqi_threshold", 70)
        st.plotly_chart(charts.plot_sqi_distribution(eeg_df, filtered, threshold), use_container_width=True)


def hrv_analysis_tab():
    hrv_df = st.session_state.get("hrv_df")
    timings = st.session_state.get("phase_timings")
    if hrv_df is None or hrv_df.empty:
        st.info("Load HRV data to view this tab.")
        return

    st.subheader("HRV Analysis")
    hrv_available_metrics = [col for col in hrv_df.columns if col not in {"timestamp", PHASE_COLUMN}]
    preview_defaults = [m for m in ["rmssd", "meanHR", "sdnn"] if m in hrv_available_metrics] or hrv_available_metrics[:2]
    with st.expander("HRV Data Preview", expanded=True):
        metrics = st.multiselect("Select HRV metrics", hrv_available_metrics, default=preview_defaults)
        st.dataframe(hrv_df[["timestamp", PHASE_COLUMN] + metrics].head(50))

    trend_options = metrics if metrics else hrv_available_metrics
    trend_defaults = [m for m in ["rmssd", "meanHR", "sdnn"] if m in trend_options] or trend_options[:1]
    trend_metrics = st.multiselect("Trend metrics", trend_options, default=trend_defaults)
    if trend_metrics:
        st.plotly_chart(charts.plot_hrv_trends_phases(hrv_df, trend_metrics, timings), use_container_width=True)

    stats = statistics.calculate_hrv_statistics_by_phase(hrv_df)
    if not stats["per_phase"].empty:
        st.dataframe(stats["per_phase"])
    if not stats["delta"].empty:
        st.dataframe(stats["delta"])


def phase_comparison_tab():
    eeg_df = st.session_state.get("eeg_df_labeled")
    filtered = st.session_state.get("eeg_df_filtered")
    active_df = filtered if filtered is not None else eeg_df
    hrv_df = st.session_state.get("hrv_df")
    if active_df is None or hrv_df is None:
        st.info("Load EEG and HRV data for comparisons.")
        return

    st.subheader("Phase Comparisons")
    eeg_stats = statistics.calculate_eeg_statistics(active_df, ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"], phase_column=PHASE_COLUMN)
    if not eeg_stats.empty:
        st.plotly_chart(charts.plot_phase_comparison_bars(eeg_stats, ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"]), use_container_width=True)
        st.plotly_chart(charts.plot_phase_comparison_heatmap(eeg_stats, ["Delta(%)", "Theta(%)", "Alpha(%)", "Beta(%)"]), use_container_width=True)


def hypothesis_testing_tab():
    eeg_df = st.session_state.get("eeg_df_labeled")
    filtered = st.session_state.get("eeg_df_filtered")
    hypothesis_table = st.session_state.get("hypothesis_table_df")
    timings = st.session_state.get("phase_timings")

    if eeg_df is None or hypothesis_table is None:
        st.info("Upload hypothesis table and ensure EEG data is loaded.")
        return

    st.subheader("Hypothesis Testing")
    data_source = filtered if st.session_state["use_filtered_for_hypothesis"] and filtered is not None else eeg_df
    st.caption(f"Using {'filtered' if data_source is filtered else 'raw'} EEG dataset for evaluation.")

    categories = hypothesis_table["Category"].unique().tolist()
    selected_category = st.selectbox("Select category", categories)
    subset = hypothesis_table[hypothesis_table["Category"] == selected_category]
    selected_hypothesis_id = st.selectbox(
        "Select hypothesis",
        subset["Hypothesis_ID"] + " – " + subset["Metric_Name"],
        format_func=lambda x: x,
    )
    selected_row = subset[subset["Hypothesis_ID"].str.cat(subset["Metric_Name"], sep=" – ") == selected_hypothesis_id].iloc[0]

    st.write(f"**Equation:** `{selected_row['Equation']}`")
    st.write(f"**Benchmark:** {selected_row['Benchmark_Min']} to {selected_row['Benchmark_Max']} ({selected_row['Direction']})")
    st.write(selected_row["Interpretation"])

    if st.button("Run Hypothesis Test"):
        with st.spinner("Evaluating hypothesis..."):
            result = hypothesis_engine.evaluate_hypothesis(
                data_source,
                selected_row,
                phase_column=PHASE_COLUMN,
                mode=st.session_state["hypothesis_mode"],
            )
            st.json({k: v for k, v in result.items() if k not in {"metric_values"}})
            metric_values = result["metric_values"]
            if isinstance(metric_values, pd.Series) and not metric_values.empty:
                st.plotly_chart(charts.plot_metric_histogram(metric_values, selected_row), use_container_width=True)

    with st.expander("Batch Evaluation"):
        selected_categories = st.multiselect("Select categories", categories)
        if st.button("Run Batch Evaluation"):
            with st.spinner("Evaluating hypotheses in batch..."):
                ids = hypothesis_table[hypothesis_table["Category"].isin(selected_categories)]["Hypothesis_ID"].tolist()
                batch_results = hypothesis_engine.batch_evaluate_hypotheses(
                    data_source,
                    hypothesis_table,
                    hypothesis_ids=ids,
                    phase_column=PHASE_COLUMN,
                    mode=st.session_state["hypothesis_mode"],
                )
                summary_df = pd.DataFrame(
                    [
                        {
                            "Hypothesis_ID": r["hypothesis_id"],
                            "Metric_Name": r["metric_name"],
                            "Category": r["category"],
                            "Passes": r["passes_benchmark"],
                            "% Within Benchmark": r["within_benchmark_percent"],
                        }
                        for r in batch_results
                    ]
                )
                st.dataframe(summary_df)


def export_tab():
    st.subheader("Export & Reports")
    eeg_df = st.session_state.get("eeg_df_labeled")
    filtered = st.session_state.get("eeg_df_filtered")
    hrv_df = st.session_state.get("hrv_df")

    if eeg_df is not None:
        st.download_button(
            "Download Labeled EEG (CSV)",
            data=eeg_df.to_csv(index=False).encode("utf-8"),
            file_name="eeg_labeled.csv",
            mime="text/csv",
        )
    if filtered is not None:
        st.download_button(
            "Download Filtered EEG (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="eeg_filtered.csv",
            mime="text/csv",
        )
    if hrv_df is not None:
        stats = statistics.calculate_hrv_statistics_by_phase(hrv_df)
        st.download_button(
            "Download HRV Phase Summary (CSV)",
            data=stats["per_phase"].to_csv().encode("utf-8"),
            file_name="hrv_phase_summary.csv",
            mime="text/csv",
        )


def main():
    st.set_page_config(page_title="EEG & HRV Analysis", layout="wide")
    initialise_session_state()
    st.title("EEG & HRV Phase-Based Analysis Dashboard")

    data_mode, lookup_df, date_keycode_map, selected_date, selected_participant = sidebar_participant_section()
    sidebar_data_loading_section(data_mode, selected_participant)
    sidebar_alignment_section()
    sidebar_sqi_section()
    sidebar_analysis_configuration()

    summarise_data_overview()

    tabs = st.tabs(["EEG Analysis", "HRV Analysis", "Phase Comparisons", "Hypothesis Testing", "Export & Reports"])
    with tabs[0]:
        eeg_analysis_tab()
    with tabs[1]:
        hrv_analysis_tab()
    with tabs[2]:
        phase_comparison_tab()
    with tabs[3]:
        hypothesis_testing_tab()
    with tabs[4]:
        export_tab()


if __name__ == "__main__":
    main()

