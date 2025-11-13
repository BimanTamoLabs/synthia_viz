# **EEG & HRV Analysis Application \- Requirements Document**

## **Project Overview**

Build a production-ready Streamlit web application for synchronized EEG and HRV data analysis with phase-based comparison (preRun1, preRun2, liveRun, postRun1, postRun2). The application enables researchers to analyze individual participant data across experimental phases, apply quality filtering, test hypotheses, and visualize multi-modal physiological data with phase alignment.

---

## **1\. Core Data Requirements**

### **1.1 EEG Data Structure**

**Required Columns:**

* `Timestamp` \- DateTime format for temporal alignment  
* `SQI` \- Signal Quality Index (0-100 scale)

**Brainwave Frequency Bands:**

* `Delta(%)` \- 0.5-4 Hz slow wave sleep  
* `Theta(%)` \- 4-8 Hz drowsiness/meditation  
* `Alpha(%)` \- 8-13 Hz relaxed awareness  
* `Beta(%)` \- 13-30 Hz active thinking

**Consciousness Metrics:**

* `CSI` \- Consciousness State Index  
* `EMG` \- Electromyography signal  
* `BS` \- Brain State metric  
* `SQI` \- Signal Quality Index

**Phase Labeling:**

* `testType` \- Phase identifier populated from HRV timing alignment  
  * Values: "preRun1", "preRun2", "liveRun", "postRun1", "postRun2", "OUT"  
  * "OUT" indicates timestamps outside any defined HRV phase

**Optional Columns:**

* `EEG_Values` \- Raw multi-channel EEG signal arrays  
* `EEG_Count`, `EEG_Min`, `EEG_Max` \- Channel statistics

### **1.2 HRV Data Structure (JSON Format)**

**JSON Schema:**

```

{
  "details": {
    "testType": {
      "preRun1": {
        "metrics": {
          "raw": [
            {
              "time": timestamp_ms,
              "AVNN": value,
              "RMSSD": value,
              "SDNN": value,
              "pNN50": value,
              "LF": value,
              "HF": value,
              "LF_HF_ratio": value,
              ...
            }
          ]
        }
      },
      "preRun2": {...},
      "liveRun": {...},
      "postRun1": {...},
      "postRun2": {...}
    }
  }
}

```

**HRV Metrics:**

* **Time Domain:**  
  * `AVNN` \- Average NN interval  
  * `SDNN` \- Standard deviation of NN intervals  
  * `RMSSD` \- Root mean square of successive differences  
  * `pNN50` \- Percentage of successive NN differences \> 50ms  
* **Frequency Domain:**  
  * `LF` \- Low frequency power (0.04-0.15 Hz)  
  * `HF` \- High frequency power (0.15-0.4 Hz)  
  * `LF_HF_ratio` \- Sympathetic/parasympathetic balance  
* **Phase Timing:**  
  * `start_time` \- Phase start timestamp  
  * `end_time` \- Phase end timestamp  
  * `duration` \- Phase duration

### **1.3 Hypothesis Reference Table**

**Structure:** CSV file with columns defining testable hypotheses

**Required Columns:**

* `Hypothesis_ID` \- Unique identifier (e.g., "H1.1", "H1.2")  
* `Category` \- Grouping (e.g., "Sleep Propensity", "Cognitive Load")  
* `Metric_Name` \- Display name  
* `Input_Pre` \- Pre-intervention column name(s)  
* `Input_Post` \- Post-intervention column name(s)  
* `Equation` \- Calculation formula using column names  
* `Benchmark` \- Human-readable threshold (e.g., "\> \-30%")  
* `Benchmark_Min` \- Minimum acceptable value  
* `Benchmark_Max` \- Maximum acceptable value  
* `Direction` \- "increase" or "decrease"  
* `Interpretation` \- Result meaning  
* `Meaning` \- Scientific context

**Example Hypotheses:**

* H1.1: Beta Power Ratio \- Expects \>30% decrease indicating brain slowing  
* H1.2: Theta Power Rise \- Expects \>15% increase indicating sleep transition  
* H1.3: Theta/Beta Ratio \- Expects \>25% increase indicating drowsiness  
* H1.4: Delta/Beta Ratio \- Expects \>15% increase indicating heavy drowsiness

---

## **2\. Functional Requirements**

### **2.1 Participant Data Management**

#### **FR-2.1.1: Participant Selection**

**Purpose:** Enable selection of individual participant data for analysis

**Requirements:**

* Load participant lookup CSV containing:  
  * `Date` \- Session date  
  * `KeyCode` \- Participant identifier  
  * `Needed?` \- Flag indicating data completeness ("Y" or "N")  
* Filter participants where `Needed? = "Y"`  
* Display available dates and participant counts  
* Allow single participant selection per analysis session  
* Display participant metadata (ID, date, folder path)

#### **FR-2.1.2: Automated File Discovery**

**Purpose:** Locate EEG and HRV files for selected participant

**Requirements:**

* Search directory structure: `{base_path}/{date_formatted}/{participant_id}/`  
* Discover EEG CSV files containing "EEG" in filename  
* Discover HRV JSON files containing "HRV" in filename  
* Validate file existence and readability  
* Report missing files with clear error messages  
* Handle multiple files per participant (if applicable)

### **2.2 EEG Data Processing**

#### **FR-2.2.1: EEG Data Loading**

**Purpose:** Load and validate EEG CSV files

**Requirements:**

* Parse CSV with automatic delimiter detection  
* Identify Timestamp column (first column if unnamed)  
* Convert timestamps to datetime objects  
* Validate SQI column presence  
* Check minimum row count (≥100 rows for statistical validity)  
* Handle missing values appropriately  
* Cache loaded data for performance

#### **FR-2.2.2: EEG Preprocessing**

**Purpose:** Clean and transform raw EEG data

**Requirements:**

* **Timestamp Standardization:**  
  * Parse various datetime formats automatically  
  * Sort data chronologically  
  * Detect and report timestamp gaps \>10 seconds  
* **EEG Channel Consolidation:**  
  * Identify columns starting with "EEG" prefix  
  * Consolidate multiple EEG channels into `EEG_Values` list column  
  * Calculate `EEG_Count`, `EEG_Min`, `EEG_Max` per sample  
  * Drop original individual channel columns to reduce memory  
* **Brainwave Validation:**  
  * Check Delta \+ Theta \+ Alpha \+ Beta ≈ 100% (±1% tolerance)  
  * Flag rows with negative percentage values  
  * Report percentage sum violations

#### **FR-2.2.3: SQI-Based Filtering**

**Purpose:** Remove low-quality EEG data based on Signal Quality Index

**Filter Modes:**

* **None** \- No filtering (analyze all data)  
* **Keep SQI ≥ threshold** \- Retain high-quality data only  
* **Keep SQI ≤ threshold** \- Isolate low-quality data for inspection

**Requirements:**

* Apply selected filter mode with user-defined threshold (0-100 slider)  
* Calculate and display filtering statistics:  
  * Original row count  
  * Filtered row count  
  * Removed row count  
  * Percentage removed  
* Preserve original data for comparison  
* Allow filter re-application with different parameters  
* Visualize SQI distribution before/after filtering

#### **FR-2.2.4: EEG Validation**

**Purpose:** Comprehensive data quality assessment

**Requirements:**

* **Session Metadata:**  
  * Extract session start time (earliest timestamp)  
  * Extract session end time (latest timestamp)  
  * Calculate total duration  
  * Estimate sampling rate (samples / duration)  
* **Data Quality Checks:**  
  * Count missing values per column  
  * Identify columns with \>50% null values (flag as problematic)  
  * Detect outliers (values beyond 3 standard deviations)  
  * Report anomalies in consciousness metrics  
* **Output:**  
  * Return validation report dictionary  
  * Display summary in UI  
  * Highlight critical issues requiring attention

### **2.3 HRV Data Processing**

#### **FR-2.3.1: HRV JSON Loading**

**Purpose:** Parse HRV JSON files and extract phase-based metrics

**Requirements:**

* Parse VibeScience-specific JSON schema  
* Navigate nested structure: `details → testType → metrics → raw`  
* Extract all time-domain and frequency-domain metrics  
* Preserve phase labels (preRun1, preRun2, liveRun, postRun1, postRun2)  
* Convert timestamps from milliseconds to datetime objects  
* Handle missing phases gracefully

#### **FR-2.3.2: HRV Phase Timing Extraction**

**Purpose:** Determine start/end times for each experimental phase

**Requirements:**

* Group HRV data by `testType` (phase)  
* For each phase, calculate:  
  * `start_time` \- Minimum timestamp in phase  
  * `end_time` \- Maximum timestamp in phase  
  * `duration` \- end\_time \- start\_time  
* Sort phases chronologically  
* Return DataFrame indexed by phase name  
* Validate phase sequence (preRun1 before preRun2, etc.)

#### **FR-2.3.3: HRV Statistical Summaries**

**Purpose:** Compute per-phase and overall HRV statistics

**Requirements:**

* **Per-Phase Summary:**  
  * Calculate mean, std, min, max for all HRV metrics within each phase  
  * Group by testType  
  * Return multi-index DataFrame (phase × statistic)  
* **Overall Summary:**  
  * Aggregate statistics across entire session  
  * Calculate global mean, std, min, max  
  * Compare to normative values (if available)  
* **Phase Deltas:**  
  * Compute differences: postRun1 \- preRun1, postRun2 \- preRun2  
  * Calculate percentage changes  
  * Identify significant shifts (\>20% change)

#### **FR-2.3.4: HRV Trend Visualization**

**Purpose:** Display HRV metrics over time with phase boundaries

**Requirements:**

* Plot time series for selected HRV metrics  
* Color-code by phase (different color per testType)  
* Draw vertical lines at phase transitions  
* Label each phase region  
* Support metric selection (dropdown or multiselect)  
* Optional: Save plot as PNG file

### **2.4 EEG-HRV Time Alignment**

#### **FR-2.4.1: Timestamp Validation**

**Purpose:** Ensure EEG data covers complete HRV recording period

**Requirements:**

* Compare EEG timestamp range vs HRV timestamp range  
* Validate: EEG start ≤ HRV start  
* Validate: EEG end ≥ HRV end  
* Display validation report:  
  * HRV Start: {datetime}  
  * EEG Start: {datetime}  
  * Status: ✅ EEG starts before HRV / ❌ Gap detected  
  * HRV Finish: {datetime}  
  * EEG Finish: {datetime}  
  * Status: ✅ EEG ends after HRV / ❌ Gap detected  
* Calculate time deltas for any gaps  
* Warn if gaps exceed 1 minute

#### **FR-2.4.2: Phase Label Assignment to EEG**

**Purpose:** Annotate each EEG sample with corresponding experimental phase

**Requirements:**

* Use HRV phase timings as ground truth  
* For each EEG timestamp:  
  * Check if it falls within any HRV phase window  
  * Assign phase label (preRun1, preRun2, liveRun, postRun1, postRun2)  
  * If outside all phases, label as "OUT"  
* Add `testType` column to EEG DataFrame  
* Preserve original EEG data (create labeled copy)  
* Display progress during labeling (can take time for large datasets)  
* Validate: Check that each phase has EEG samples assigned  
* Report count of samples per phase

### **2.5 Phase-Based Statistical Analysis**

#### **FR-2.5.1: EEG Statistics by Phase**

**Purpose:** Compute EEG metrics grouped by experimental phase

**Requirements:**

* Group EEG data by `testType` column  
* Exclude "OUT" phase from analysis  
* For each phase, calculate per metric:  
  * Mean  
  * Standard deviation  
  * Minimum  
  * Maximum  
  * Median (25th, 50th, 75th percentiles)  
* Return multi-index DataFrame: (phase, statistic) × metrics  
* Support column filtering (analyze subset of metrics)

#### **FR-2.5.2: Phase Ratio Calculations**

**Purpose:** Compare metrics across phases using ratios

**Ratio Types:**

* **Simple Pre/Post Ratios:**  
  * preRun1 / postRun1 (immediate effect)  
  * preRun2 / postRun2 (delayed effect)  
* **Combined Pre/Post Ratios:**  
  * (preRun1 \+ preRun2) / (postRun1 \+ postRun2) (overall effect)  
* **Pre/Live Ratios:**  
  * (preRun1 \+ preRun2) / liveRun (baseline vs intervention)  
* **Live/Post Ratios:**  
  * liveRun / (postRun1 \+ postRun2) (intervention vs outcome)  
* **Derived Brainwave Ratios (per phase):**  
  * Theta(%) / Beta(%) \- Drowsiness indicator  
  * Delta(%) / Beta(%) \- Deep sleep propensity  
  * Delta(%) / Alpha(%) \- Consciousness reduction  
  * Delta(%) / Theta(%) \- Sleep stage transition  
  * Alpha(%) / Beta(%) \- Relaxation vs alertness  
  * (Theta(%) \+ Alpha(%)) / Beta(%) \- Combined relaxation

**Requirements:**

* Calculate all ratio types using phase statistics (means)  
* Handle division by zero (return NaN or inf marker)  
* Return transposed DataFrame: ratio names × metrics  
* Support hypothesis-specific ratio selection

#### **FR-2.5.3: HRV Statistics by Phase**

**Purpose:** Compute HRV metrics grouped by experimental phase

**Requirements:**

* Already implemented in `process_hrv_json()` function  
* Return per-phase summary for all HRV time/frequency metrics  
* Calculate phase deltas (post \- pre changes)  
* Support export to CSV

### **2.6 Hypothesis Testing Engine**

#### **FR-2.6.1: Hypothesis Table Loading**

**Purpose:** Load and validate hypothesis reference table

**Requirements:**

* Read hypothesis\_reference\_table.csv  
* Validate required columns present  
* Check for duplicate Hypothesis\_IDs  
* Parse equations for syntax errors (basic validation)  
* Cache table for session duration  
* Support hot-reload if table updated

#### **FR-2.6.2: Equation Evaluation**

**Purpose:** Safely compute hypothesis metrics from equations

**Requirements:**

* **Equation Sanitization:**  
  * Replace `^` with `**` for exponentiation  
  * Validate only allowed operators: `+`, `-`, `*`, `/`, `**`, `%`, `(`, `)`  
  * Reject function calls, imports, attribute access  
  * Strip whitespace  
* **Variable Extraction:**  
  * Parse equation to identify column names referenced  
  * Use regex to find variable patterns  
  * Return list of required columns  
* **Column Mapping:**  
  * Map equation variables to actual DataFrame columns  
  * Support suffix handling (\_pre, \_post) for phase-based hypotheses  
  * Example: "Beta\_pre" maps to "Beta(%)" data filtered to preRun phases  
* **Safe Evaluation:**  
  * Use pandas `eval()` method for vectorized computation  
  * Validate all referenced columns exist before evaluation  
  * Handle division by zero gracefully (replace inf with NaN)  
  * Return Series of computed metric values per sample

#### **FR-2.6.3: Benchmark Comparison**

**Purpose:** Evaluate hypothesis metric against acceptance criteria

**Requirements:**

* Extract Benchmark\_Min and Benchmark\_Max from hypothesis row  
* For each computed metric value, check if within range:  
  * `within_benchmark = (value >= Benchmark_Min) & (value <= Benchmark_Max)`  
* Calculate percentage within benchmark:  
  * `percent_in_range = (within_benchmark.sum() / valid_samples) * 100`  
* Determine pass/fail status:  
  * Pass if percent\_in\_range ≥ threshold (default 80%)  
  * Fail otherwise  
* Return comparison results

#### **FR-2.6.4: Hypothesis Result Generation**

**Purpose:** Compile comprehensive hypothesis evaluation report

**Output Dictionary:**

```

Copy
{
    'hypothesis_id': str,
    'category': str,
    'metric_name': str,
    'equation': str,
    'metric_values': pd.Series,
    'valid_samples': int,
    'metric_mean': float,
    'metric_std': float,
    'metric_median': float,
    'metric_min': float,
    'metric_max': float,
    'benchmark_min': float,
    'benchmark_max': float,
    'within_benchmark_count': int,
    'within_benchmark_percent': float,
    'passes_benchmark': bool,
    'direction': str,
    'interpretation': str,
    'meaning': str
}

```

**Requirements:**

* Compute all descriptive statistics  
* Apply benchmark comparison  
* Format results for display  
* Include metadata for interpretation

#### **FR-2.6.5: Batch Hypothesis Evaluation**

**Purpose:** Test multiple hypotheses simultaneously for comparison

**Requirements:**

* Accept list of hypothesis IDs or categories  
* Evaluate each hypothesis sequentially  
* Compile results into summary DataFrame  
* Display comparative results table  
* Highlight passing vs failing hypotheses  
* Support export of batch results

### **2.7 Visualization Requirements**

#### **FR-2.7.1: EEG Time Series by Phase**

**Purpose:** Display EEG metrics over time with phase color-coding

**Requirements:**

* Multi-panel subplot layout (one subplot per metric)  
* X-axis: Timestamp with HH:MM:SS formatting  
* Y-axis: Metric value (auto-scaled per subplot)  
* Color-code data points or regions by testType:  
  * preRun1: Light blue  
  * preRun2: Medium blue  
  * liveRun: Green  
  * postRun1: Orange  
  * postRun2: Red  
  * OUT: Gray  
* Draw vertical lines at phase transitions  
* Label each phase region with text annotation  
* Support column selection (choose which metrics to plot)  
* Interactive zoom/pan (Plotly)  
* Hover tooltips showing exact values and phase

#### **FR-2.7.2: Brainwave Frequency Stacked Area Chart**

**Purpose:** Visualize brainwave distribution over time by phase

**Requirements:**

* Stacked area chart: Delta \+ Theta \+ Alpha \+ Beta \= 100%  
* X-axis: Timestamp  
* Y-axis: Percentage (0-100%)  
* Color scheme:  
  * Delta: Dark purple (deep sleep)  
  * Theta: Light blue (drowsiness)  
  * Alpha: Green (relaxation)  
  * Beta: Orange (alertness)  
* Phase boundaries marked with vertical lines  
* Phase regions shaded or annotated  
* Legend identifying each brainwave band  
* Smooth interpolation between samples

#### **FR-2.7.3: Consciousness Metrics Multi-Panel**

**Purpose:** Display consciousness-related metrics with weighted score

**Requirements:**

* 5 subplots stacked vertically:  
  * CSI (Consciousness State Index)  
  * EMG (Muscle activity)  
  * SQI (Signal quality)  
  * BS (Brain state)  
  * **Consciousness Score** (weighted combination)  
* Consciousness Score formula:  
  * `Score = w1*CSI + w2*EMG + w3*SQI + w4*BS`  
  * Default weights: {CSI: 0.4, EMG: 0.2, SQI: 0.2, BS: 0.2}  
  * User can adjust weights via sliders (must sum to 1.0)  
* Phase color-coding applied to all subplots  
* Shared X-axis (timestamp) aligned across panels  
* 2-minute tick intervals  
* Filled area under curves for visual clarity

#### **FR-2.7.4: EEG Heatmap with Aligned Metrics**

**Purpose:** Display raw EEG signal as heatmap with synchronized metrics

**Requirements:**

* Layout: GridSpec with 2 sections  
  * **Top (60% height):** EEG heatmap  
    * X-axis: Time  
    * Y-axis: EEG channel/sample index  
    * Color: EEG amplitude (diverging colormap: blue-white-red)  
    * Requires `EEG_Values` column (list of arrays)  
  * **Bottom (40% height):** Aligned line plots for CSI, EMG, SQI, BS  
    * 4 subplots sharing X-axis  
    * Phase color-coding  
* Time range selector: Allow user to specify start\_time and end\_time  
* Synchronized zooming across all panels  
* Phase boundary markers on all panels

#### **FR-2.7.5: HRV Trend Plots**

**Purpose:** Display HRV metrics over time with phase segmentation

**Requirements:**

* Line plot for selected HRV metric(s)  
* X-axis: Timestamp  
* Y-axis: HRV metric value (e.g., RMSSD, LF/HF ratio)  
* Color-code by phase (same scheme as EEG)  
* Draw vertical lines at phase transitions  
* Label phase regions  
* Support multiple metrics on same plot (different line styles)  
* Display mean per phase as horizontal line segments within each region  
* Show std as shaded confidence band

#### **FR-2.7.6: Phase Comparison Bar Charts**

**Purpose:** Compare mean values across phases for selected metrics

**Chart Types:**

* **Grouped Bar Chart:**  
  * X-axis: Phases (preRun1, preRun2, liveRun, postRun1, postRun2)  
  * Y-axis: Mean metric value  
  * Multiple bars per phase (one per metric, grouped)  
  * Error bars showing ±1 std  
  * Legend identifying metrics  
* **Heatmap Table:**  
  * Rows: Metrics  
  * Columns: Phases  
  * Cell color: Mean value intensity (gradient)  
  * Cell text: Mean ± std formatted  
  * Benchmark overlay (highlight cells outside expected range)

**Requirements:**

* Support both EEG and HRV metrics  
* Allow metric selection via multiselect  
* Interactive tooltips with exact values  
* Export chart as image

#### **FR-2.7.7: Hypothesis Metric Distribution**

**Purpose:** Show distribution of computed hypothesis metric with benchmark

**Requirements:**

* Histogram with configurable bins (default 30\)  
* X-axis: Metric value  
* Y-axis: Frequency count or density  
* Color-code bars:  
  * Green: Values within benchmark range  
  * Red: Values outside benchmark range  
* Vertical lines marking Benchmark\_Min and Benchmark\_Max  
* Annotation showing percentage within benchmark  
* Overlay normal distribution curve (if appropriate)  
* Display mean, median, std as reference lines

#### **FR-2.7.8: SQI Distribution Comparison**

**Purpose:** Visualize impact of SQI filtering

**Requirements:**

* Side-by-side or overlaid histograms  
* Before filtering: Blue histogram (raw data SQI)  
* After filtering: Green histogram (cleaned data SQI)  
* Vertical line showing threshold value  
* Statistics overlay:  
  * Original mean SQI  
  * Filtered mean SQI  
  * Improvement percentage  
* Legend clearly labeling "Before" and "After"

#### **FR-2.7.9: Ratio Comparison Heatmap**

**Purpose:** Display calculated phase ratios in matrix form

**Requirements:**

* Heatmap with:  
  * Rows: Ratio types (pre1/post1, pre2/post2, combined, etc.)  
  * Columns: Metrics (Delta, Theta, Alpha, Beta, ratios)  
  * Cell color: Ratio value intensity (diverging colormap)  
  * Cell text: Ratio value formatted (2 decimal places)  
* Highlight cells where ratio indicates significant change (\>20% deviation)  
* Interactive tooltips explaining ratio interpretation  
* Export as image or CSV

---

## **3\. User Interface Requirements**

### **3.1 Application Layout**

#### **Layout Structure:**

```

[SIDEBAR]                    [MAIN CONTENT AREA]
- Participant Selection      - Page Title & Instructions
- File Upload/Discovery      - Participant Info Panel
- SQI Filtering Controls     - Data Overview Metrics
- Analysis Configuration     - Phase Timeline Visualization
- Hypothesis Selection       - Tabbed Content Area
                               ├─ Tab 1: EEG Analysis
                               ├─ Tab 2: HRV Analysis
                               ├─ Tab 3: Phase Comparisons
                               ├─ Tab 4: Hypothesis Testing
                               └─ Tab 5: Export & Reports

```

### **3.2 Sidebar Components**

#### **UI-3.2.1: Participant Selection Panel**

**Components:**

* **Section Header:** "📂 Participant Selection"  
* **Date Selector:**  
  * Widget: Selectbox  
  * Label: "Select Session Date"  
  * Options: Dynamically loaded from participant lookup CSV (formatted: "DD\_MM\_YY")  
  * Help text: "Choose the date of the recording session"  
* **Participant Selector:**  
  * Widget: Selectbox  
  * Label: "Select Participant"  
  * Options: Filtered by selected date (display: "Participant {KeyCode}")  
  * Help text: "Choose participant for analysis"  
* **File Discovery Button:**  
  * Label: "🔍 Find Data Files"  
  * Action: Search for EEG CSV and HRV JSON in participant folder  
  * Display results:  
    * ✅ EEG file found: {filename}  
    * ✅ HRV file found: {filename}  
    * ❌ Missing files with suggestions

#### **UI-3.2.2: Data Loading Panel**

**Components:**

* **Section Header:** "📥 Data Loading"  
* **Load EEG Button:**  
  * Label: "Load EEG Data"  
  * Enabled only after EEG file discovered  
  * Shows spinner during loading  
  * Success message: "✅ EEG loaded: {row\_count} samples"  
* **Load HRV Button:**  
  * Label: "Load HRV Data"  
  * Enabled only after HRV file discovered  
  * Success message: "✅ HRV loaded: {phase\_count} phases"  
* **Align Data Button:**  
  * Label: "🔗 Align EEG with HRV Phases"  
  * Enabled only after both datasets loaded  
  * Action: Validate timestamps and assign phases to EEG  
  * Progress bar during processing  
  * Success message: "✅ Phase alignment complete"

#### **UI-3.2.3: SQI Filtering Panel**

**Components:**

* **Section Header:** "🔍 Signal Quality Filtering"  
* **Filter Mode Selector:**  
  * Widget: Selectbox  
  * Options: \["None", "Keep SQI \>= threshold", "Keep SQI \<= threshold"\]  
  * Default: "Keep SQI \>= threshold"  
* **Threshold Slider:**  
  * Widget: Slider  
  * Range: 0-100  
  * Default: 70  
  * Step: 1  
  * Label: "SQI Threshold: {value}"  
  * Help text: "Higher threshold \= higher quality but less data"  
* **Apply Filter Button:**  
  * Label: "Apply Filter"  
  * Displays filtering statistics after application

#### **UI-3.2.4: Analysis Configuration Panel**

**Components:**

* **Section Header:** "⚙️ Analysis Settings"  
* **Data Source Toggle:**  
  * Widget: Checkbox  
  * Label: "Use filtered data for hypothesis testing"  
  * Default: Checked  
* **Consciousness Score Weights:**  
  * Expander: "Adjust Consciousness Weights"  
  * Four sliders (must sum to 1.0):  
    * CSI weight (0-1, default 0.4)  
    * EMG weight (0-1, default 0.2)  
    * SQI weight (0-1, default 0.2)  
    * BS weight (0-1, default 0.2)  
  * Validation: Display error if sum ≠ 1.0

### **3.3 Main Content Area Components**

#### **UI-3.3.1: Page Header**

**Components:**

* **Title:** "EEG & HRV Phase-Based Analysis Dashboard"  
* **Participant Info Card:**  
  * Display when participant selected:  
    * Participant ID: {KeyCode}  
    * Session Date: {Date}  
    * Data Folder: {path}  
    * EEG Status: Loaded/Not Loaded  
    * HRV Status: Loaded/Not Loaded  
    * Alignment Status: Complete/Pending

#### **UI-3.3.2: Data Overview Metrics**

**Components:**

* **Layout:** 4 columns using st.columns(4)  
* **EEG Metrics:**  
  * Total EEG Samples  
  * EEG Duration (HH:MM:SS)  
  * Average SQI  
  * Samples per Phase (expandable detail)  
* **HRV Metrics:**  
  * Total HRV Samples  
  * HRV Duration  
  * Phases Detected (count)  
  * Average RMSSD (example metric)

#### **UI-3.3.3: Phase Timeline Visualization**

**Purpose:** Show temporal structure of experiment

**Components:**

* Horizontal timeline bar  
* Color-coded segments for each phase (proportional to duration)  
* Phase labels with start/end times  
* Hover: Show phase duration  
* Legend: Phase colors

#### **UI-3.3.4: Tab 1 \- EEG Analysis**

**Contents:**

**Expander 1: Data Preview**

* First 50 rows of EEG data (with testType column)  
* Column selector: Choose columns to display  
* Download button: Export visible data as CSV

**Expander 2: EEG Time Series by Phase**

* Metric selector: Multiselect for columns to plot  
* Plot: Call `visualize_eeg_trends_advanced()`  
* Show statistics overlay toggle  
* Download plot button

**Expander 3: Brainwave Frequency Distribution**

* Plot: Stacked area chart for Delta, Theta, Alpha, Beta  
* Phase boundaries marked  
* Option: Normalize to percentage or show absolute values

**Expander 4: Consciousness Metrics**

* Plot: Multi-panel consciousness metrics  
* Adjustable weights (linked to sidebar sliders)  
* Display weighted consciousness score

**Expander 5: EEG Statistics by Phase**

* Table: Statistics DataFrame (phase × metric)  
* Columns: mean, std, min, max per phase  
* Download table button

#### **UI-3.3.5: Tab 2 \- HRV Analysis**

**Contents:**

**Expander 1: HRV Data Preview**

* Table: HRV raw data with timestamp and testType  
* Metric selector: Choose HRV metrics to display  
* Download button

**Expander 2: HRV Phase Timings**

* Table: Phase timing DataFrame  
  * Columns: testType, start\_time, end\_time, duration  
* Validation status: Show alignment check results

**Expander 3: HRV Trend Plots**

* Metric selector: Choose HRV metrics (AVNN, RMSSD, LF, HF, etc.)  
* Plot: Time series with phase color-coding  
* Display per-phase means as horizontal lines  
* Download plot button

**Expander 4: HRV Statistics by Phase**

* Two tables:  
  * Per-phase summary (mean, std per metric)  
  * Phase deltas (post \- pre changes)  
* Highlight significant changes (\>20%)  
* Download tables button

#### **UI-3.3.6: Tab 3 \- Phase Comparisons**

**Contents:**

**Expander 1: EEG Phase Comparison**

* Metric selector: Choose EEG metrics  
* Chart type selector: \["Grouped Bar", "Heatmap"\]  
* Plot: Mean values across phases with error bars  
* Download chart button

**Expander 2: HRV Phase Comparison**

* Same structure as EEG comparison  
* Use HRV metrics

**Expander 3: Phase Ratio Analysis**

* Display calculated ratios table (from `calculate_phase_ratios()`)  
* Heatmap visualization of ratios  
* Highlight ratios indicating significant changes  
* Interpretation guide: Explain what ratios mean

**Expander 4: Pre vs Post Summary**

* Focused comparison: (preRun1+preRun2) vs (postRun1+postRun2)  
* Side-by-side metrics table  
* Percentage change column  
* Color-code: Green for expected direction, red for unexpected

#### **UI-3.3.7: Tab 4 \- Hypothesis Testing**

**Contents:**

**Expander 1: Hypothesis Selection**

* **Category Selector:**  
  * Widget: Selectbox  
  * Options: Unique categories from hypothesis table  
* **Hypothesis Selector:**  
  * Widget: Selectbox  
  * Options: Hypotheses filtered by category  
  * Display format: "{Hypothesis\_ID}: {Metric\_Name}"  
* **Hypothesis Details Card:**  
  * Display selected hypothesis metadata  
  * Equation in code block  
  * Benchmark range highlighted  
  * Interpretation and meaning text

**Expander 2: Apply Hypothesis**

* **Data Source Reminder:**  
  * Show which dataset will be used (raw or filtered)  
* **Apply Button:**  
  * Label: "🧪 Run Hypothesis Test"  
  * Progress spinner during computation

**Expander 3: Hypothesis Results**

* **Summary Statistics Card:**  
  * Valid samples count  
  * Metric mean ± std  
  * Metric median  
  * Min to max range  
* **Benchmark Comparison:**  
  * Large pass/fail icon (✅ or ❌)  
  * Progress bar: Percentage within benchmark  
  * Color-coded: Green (\>80%), yellow (50-80%), red (\<50%)  
* **Interpretation Display:**  
  * Hypothesis interpretation text  
  * Scientific meaning (expandable)  
  * Contextual explanation based on results

**Expander 4: Metric Distribution Visualization**

* Plot: Histogram with benchmark overlay  
* Color-coded bars (green/red for within/outside benchmark)  
* Download plot button

**Expander 5: Batch Hypothesis Testing**

* **Category Batch Selector:**  
  * Widget: Multiselect  
  * Options: Select multiple categories  
* **Run Batch Button:**  
  * Evaluates all hypotheses in selected categories  
* **Results Table:**  
  * Columns: Hypothesis\_ID, Category, Metric\_Name, Pass/Fail, % Within Benchmark  
  * Sortable and filterable  
  * Download results button

#### **UI-3.3.8: Tab 5 \- Export & Reports**

**Contents:**

**Expander 1: Data Export**

* **Export Filtered EEG:**  
  * Button: Download cleaned EEG with phase labels as CSV  
* **Export HRV Summary:**  
  * Button: Download HRV phase statistics as CSV  
* **Export Phase Ratios:**  
  * Button: Download ratio analysis as CSV

**Expander 2: Visualization Export**

* List of all generated plots with individual download buttons  
* **Batch Export:**  
  * Button: Download all visualizations as ZIP file

**Expander 3: Analysis Report**

* **Generate Report Button:**  
  * Creates comprehensive analysis summary  
  * Includes:  
    * Participant metadata  
    * Data quality summary  
    * Phase timing information  
    * Key statistics per phase  
    * Hypothesis test results  
    * All visualizations embedded  
  * Format options: PDF or HTML  
  * Download button

**Expander 4: Session State**

* Display current analysis configuration  
* Option to save session state as JSON  
* Option to load previous session state

---

## **4\. Workflow and User Journey**

### **4.1 Standard Analysis Workflow**

**Step 1: Participant Selection (Sidebar)**

* User selects session date from dropdown  
* User selects participant from filtered list  
* User clicks "Find Data Files"  
* System discovers EEG CSV and HRV JSON files  
* System displays file paths and confirmation

**Step 2: Data Loading (Sidebar)**

* User clicks "Load EEG Data"  
  * System loads, validates, preprocesses EEG  
  * Data overview metrics populate  
* User clicks "Load HRV Data"  
  * System parses JSON, extracts phases and metrics  
  * HRV overview metrics populate  
* User clicks "Align EEG with HRV Phases"  
  * System validates timestamp coverage  
  * System assigns testType to each EEG sample  
  * Phase timeline visualization appears  
  * Success message confirms alignment

**Step 3: Data Quality Review (Tab 1 & 2\)**

* User navigates to **Tab 1: EEG Analysis**  
* Expands "Data Preview" to view raw data  
* Expands "EEG Statistics by Phase" to check distributions  
* User navigates to **Tab 2: HRV Analysis**  
* Expands "HRV Phase Timings" to verify phase structure  
* Reviews phase alignment validation report

**Step 4: SQI Filtering (Sidebar)**

* User selects filter mode (typically "Keep SQI \>= threshold")  
* User adjusts threshold slider (e.g., 70\)  
* User clicks "Apply Filter"  
* System displays filtering statistics:  
  * Rows removed  
  * Percentage removed per phase  
* User reviews SQI distribution comparison (before/after)  
* User may adjust threshold and re-apply if needed

**Step 5: Phase Comparison Analysis (Tab 3\)**

* User navigates to **Tab 3: Phase Comparisons**  
* Expands "EEG Phase Comparison"  
* Selects metrics of interest (e.g., Beta, Theta, Alpha)  
* Views grouped bar chart showing mean values per phase  
* Expands "Phase Ratio Analysis"  
* Reviews calculated ratios (pre/post, live comparisons)  
* Identifies significant changes (color-coded cells)  
* Repeats for HRV metrics in "HRV Phase Comparison"

**Step 6: Hypothesis Testing (Tab 4\)**

* User navigates to **Tab 4: Hypothesis Testing**  
* Expands "Hypothesis Selection"  
* Selects category (e.g., "Sleep Propensity")  
* Selects specific hypothesis (e.g., "H1.1: Beta Power Ratio")  
* Reviews hypothesis details (equation, benchmark, interpretation)  
* Expands "Apply Hypothesis"  
* Ensures correct data source selected (filtered or raw)  
* Clicks "Run Hypothesis Test"  
* Reviews results:  
  * Summary statistics  
  * Pass/fail status with percentage within benchmark  
  * Interpretation text  
* Expands "Metric Distribution Visualization" to see histogram  
* Optionally tests additional hypotheses or runs batch test

**Step 7: Export Results (Tab 5\)**

* User navigates to **Tab 5: Export & Reports**  
* Downloads desired datasets (filtered EEG, HRV summary, ratios)  
* Downloads individual visualizations or batch ZIP  
* Generates comprehensive analysis report (PDF or HTML)  
* Downloads report for archival or sharing

---

## **5\. Technical Implementation Notes**

### **5.1 Module Organization**

**utils/data\_loader.py**

* `get_needed_participants(file_path)` \- Load participant lookup  
* `get_keycode_dict_by_date(participants_df)` \- Group participants by date  
* `get_full_paths_for_date(date_keycode_map, date_key, base_path)` \- Build file paths  
* `find_files(main_folder_location, patient_id_folder, file_type)` \- Discover EEG/HRV files  
* `load_eeg_file(file_path)` \- Load and validate EEG CSV  
* `load_hypothesis_table(file_path)` \- Load hypothesis reference

**utils/data\_processor.py**

* `process_eeg_data(file_path)` \- EEG preprocessing (from provided script)  
* `validate_eeg_data(df)` \- Comprehensive EEG validation  
* `get_column_statistics(df, exclude_columns)` \- Descriptive statistics  
* `process_hrv_json(json_path, ...)` \- HRV JSON parsing  
* `load_hrv_json_to_df(json_path)` \- Flatten HRV to DataFrame  
* `get_testType_timings(hrv_df)` \- Extract phase timing information

**utils/alignment.py** (NEW)

* `validate_timestamp_alignment(eeg_df, hrv_df)` \- Check timestamp coverage  
* `assign_hrv_testTypes_to_eeg(eeg_df, hrv_timings_df)` \- Label EEG with phases

**utils/sqi\_filter.py**

* `apply_sqi_filter(df, mode, threshold)` \- SQI-based filtering

**utils/statistics.py**

* `calculate_eeg_statistics(eeg_labeled_df, columns)` \- Phase-grouped EEG stats  
* `calculate_phase_ratios(stats_df)` \- Compute phase ratios  
* (Add HRV equivalents if needed)

**utils/hypothesis\_engine.py**

* `extract_variables_from_equation(equation)` \- Parse equation variables  
* `sanitize_equation(equation)` \- Prepare equation for safe eval  
* `compute_metric(df, equation, column_mapping)` \- Evaluate equation  
* `evaluate_hypothesis(df, hypothesis_row)` \- Full hypothesis evaluation  
* `batch_evaluate_hypotheses(df, hypothesis_list)` \- Batch testing

**utils/charts.py**

* `plot_eeg_timeseries(df, columns, phase_column)` \- Time series with phase coloring  
* `plot_brainwave_bands(df)` \- Stacked area chart  
* `plot_consciousness_metrics(df, weights)` \- Multi-panel consciousness  
* `plot_combined_metrics_eeg_heatmap_aligned(df, start, end)` \- EEG heatmap  
* `visualize_eeg_trends_advanced(df, columns, ...)` \- Advanced EEG trends  
* `plot_hrv_trends(hrv_df, metrics)` \- HRV time series with phases  
* `plot_phase_comparison_bars(stats_df, metrics)` \- Grouped bar chart  
* `plot_phase_comparison_heatmap(stats_df, metrics)` \- Heatmap table  
* `plot_metric_histogram(metric_series, hypothesis_row)` \- Hypothesis distribution  
* `plot_sqi_distribution(df_raw, df_clean, threshold)` \- SQI before/after  
* `plot_ratio_heatmap(ratio_df)` \- Phase ratio heatmap

### **5.2 Caching Strategy**

**Use Streamlit @st.cache\_data for:**

* Participant lookup loading  
* EEG file loading (keyed by file path and modification time)  
* HRV JSON loading (keyed by file path)  
* Hypothesis table loading  
* Phase timing extraction (keyed by HRV data hash)  
* Phase statistics calculation (keyed by data hash \+ phase column)

**Do NOT cache:**

* SQI filtering (parameters change frequently)  
* Hypothesis evaluation (depends on user selections)  
* Visualizations (depend on dynamic parameters)

### **5.3 Performance Considerations**

**Large EEG Datasets:**

* EEG files can exceed 100,000 rows  
* Use efficient pandas operations (vectorized, no loops)  
* Downsample time series visualizations if \>50,000 points  
* Consider chunked processing for extreme sizes

**Phase Assignment Performance:**

* Labeling each EEG row with testType can be slow  
* Use vectorized timestamp comparisons instead of row iteration  
* Display progress bar for user feedback

**Memory Management:**

* Drop unused columns early (e.g., individual EEG channels after consolidation)  
* Use appropriate dtypes (float32 vs float64)  
* Clear cached data when switching participants

### **5.4 Error Handling Priorities**

**Critical Errors (Block Analysis):**

* Participant data files not found  
* EEG missing SQI column  
* HRV JSON parsing failure  
* Timestamp alignment validation failure

**Non-Critical Errors (Warn but Continue):**

* Missing optional EEG columns (e.g., EEG\_Values)  
* Percentage sum violations in brainwave bands  
* SQI column contains some null values  
* Hypothesis equation references missing columns

**User-Friendly Messages:**

* Always explain what went wrong  
* Suggest corrective actions  
* Provide examples of correct formats  
* Display available alternatives (e.g., "Available columns: ...")

---

## **6\. Success Criteria**

**The application is considered complete when:**

### **6.1 Functional Completeness**

* ✅ Can load participant lookup and discover data files  
* ✅ Can load and validate EEG CSV files with all expected columns  
* ✅ Can parse HRV JSON files and extract phase timings  
* ✅ Can validate EEG-HRV timestamp alignment  
* ✅ Can assign phase labels (testType) to all EEG samples  
* ✅ Can apply SQI filtering with all three modes  
* ✅ Can compute phase-grouped statistics for both EEG and HRV  
* ✅ Can calculate all defined phase ratios  
* ✅ Can evaluate hypotheses from reference table  
* ✅ Can generate all required visualizations with phase color-coding

### **6.2 Phase Comparison Requirements**

* ✅ Clear visualization of all 5 phases (preRun1, preRun2, liveRun, postRun1, postRun2)  
* ✅ Side-by-side comparison of pre vs post metrics  
* ✅ Ratio analysis showing intervention effects  
* ✅ Statistical significance indicators for phase differences

### **6.3 Usability**

* ✅ Intuitive participant selection workflow  
* ✅ Clear phase timeline visualization  
* ✅ All visualizations have phase color-coding and labels  
* ✅ Hypothesis results include clear pass/fail indicators  
* ✅ Export functionality for all data and visualizations

### **6.4 Performance**

* ✅ Loads typical participant data (\<10,000 EEG rows) in \<10 seconds  
* ✅ Phase alignment completes in \<30 seconds  
* ✅ Visualizations render in \<5 seconds  
* ✅ No application freezing or crashes

### **6.5 Documentation**

* ✅ README with installation and usage instructions  
* ✅ In-app instructions for workflow  
* ✅ Code comments explaining phase alignment logic  
* ✅ Example participant data provided

---

## **7\. Priority Development Order**

### **Phase 1: Core Data Pipeline (HIGHEST PRIORITY)**

* Implement participant selection and file discovery  
* Implement EEG and HRV loading  
* Implement timestamp alignment and phase labeling  
* Test with real participant data

### **Phase 2: Phase-Based Analysis**

* Implement phase-grouped statistics for EEG  
* Implement phase-grouped statistics for HRV  
* Implement phase ratio calculations  
* Test phase comparison logic

### **Phase 3: Visualization Layer**

* Implement phase timeline visualization  
* Implement EEG time series with phase coloring  
* Implement HRV trend plots with phases  
* Implement phase comparison bar charts  
* Implement other supporting visualizations

### **Phase 4: Hypothesis Testing**

* Implement hypothesis engine with equation evaluation  
* Implement benchmark comparison  
* Integrate with phase-labeled EEG data  
* Implement batch hypothesis testing

### **Phase 5: UI Assembly**

* Build sidebar with participant selection and controls  
* Build tabbed main content area  
* Integrate all visualizations into tabs  
* Add export functionality

### **Phase 6: Polish and Testing**

* Comprehensive testing with multiple participants  
* Error handling refinement  
* Performance optimization  
* Documentation completion

