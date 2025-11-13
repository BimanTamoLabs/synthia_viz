# synthia_viz

Streamlit application for synchronized EEG and HRV analysis with phase-based comparisons, hypothesis testing, and rich visualisations for the Synthia project.

## Structure

- `app.py` – Streamlit entry point
- `utils/` – Shared utility modules (data loading, preprocessing, alignment, statistics, visualisations, hypothesis engine)
- `hypothesis_reference_table.csv` – Hypothesis definitions

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Development Notes

- Supports both individual file uploads and batch participant discovery.
- Visualisations highlight experiment phases (preRun1, preRun2, liveRun, postRun1, postRun2) with combined shading and boundaries.
- Hypothesis engine evaluates equations against aggregated means or per-sample distributions.

