# ED-Context-AI: Emergency Department Clinical Decision Support System

## A Context-Aware Deep Learning System for ED Admission Prediction

---

## 1. Project Overview

**ED-Context-AI** is an end-to-end machine learning pipeline that predicts whether Emergency Department (ED) patients will require hospital admission, using temporal vital sign patterns, clinical context, and uncertainty-aware deep learning. The system is built on **MIMIC-IV-ED v2.2** — a large-scale, de-identified dataset of real ED encounters from Beth Israel Deaconess Medical Center.

The system goes beyond simple binary prediction: it quantifies prediction uncertainty via Monte Carlo Dropout, classifies patient physiological states, generates human-readable clinical explanations, and produces structured decision recommendations with urgency levels and recommended diagnostic tests.

### Core Problem

Emergency departments face a critical triage challenge: identifying which patients will deteriorate and require admission before it becomes clinically obvious. Late recognition leads to delayed interventions, while over-triage wastes limited hospital resources. This system aims to augment clinician decision-making by providing early, uncertainty-aware risk assessments based on evolving vital sign trajectories.

### Key Results

| Metric | GRU Model | Logistic Regression Baseline |
|--------|-----------|------------------------------|
| **AUROC** | **0.779** | 0.661 |
| **AUPRC** | **0.593** | 0.510 |
| **F1 Score** | **0.603** | 0.542 |
| **Brier Score** | **0.194** | 0.231 |
| **ECE** | 0.140 | — |
| **N Samples** | 321,888 | 532,836 |

The GRU model achieves an **11.8 percentage point improvement** in AUROC over the logistic regression baseline, demonstrating that temporal sequence modeling and engineered clinical context features provide meaningful predictive value.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

The system follows a five-phase pipeline, orchestrated through `main.py`:

```
Phase 1: Preprocessing    →  Raw MIMIC-IV-ED CSVs → Cleaned PatientRecords
Phase 2: Context Building  →  PatientRecords → ContextObjects (72-dim vectors)
Phase 3: Model Training    →  ContextObjects → Trained MC-Dropout GRU
Phase 4: Evaluation        →  Test set → Metrics, Plots, Baseline Comparison
Phase 5: Demo/Inference    →  Single patient → Risk + Explanation + Decision
```

Each phase can be run independently via:
```bash
python main.py --phase preprocess
python main.py --phase context
python main.py --phase train
python main.py --phase evaluate
python main.py --phase demo
python main.py               # runs all phases sequentially
```

### 2.2 Directory Structure

```
ed-context-ai/
├── main.py                          # CLI entry point
├── config.yaml                      # All hyperparameters and paths
├── src/
│   ├── preprocessing/               # Phase 1: Data ingestion and cleaning
│   │   ├── loader.py                #   Raw CSV loading (6 tables)
│   │   ├── cleaner.py               #   Vital clipping, outcome derivation, sepsis flags
│   │   ├── windower.py              #   60-min temporal window aggregation
│   │   ├── missingness.py           #   Critical feature missingness tracking
│   │   └── pipeline.py              #   Orchestrator → PatientRecord objects
│   ├── context/                     # Phase 2: Clinical context construction
│   │   ├── builder.py               #   Orchestrator → ContextObject objects
│   │   ├── trend_engine.py          #   Vital sign slope computation and classification
│   │   ├── state_classifier.py      #   Physiological state and data confidence
│   │   └── vector_builder.py        #   72-dimensional context vector assembly
│   ├── model/                       # Phase 3: Model architecture and training
│   │   ├── mc_dropout.py            #   MLP with MC Dropout (alternative model)
│   │   ├── mc_dropout_gru.py        #   GRU with attention + MC Dropout (primary model)
│   │   ├── dataset.py               #   Per-window flat dataset (for MLP)
│   │   ├── sequence_dataset.py      #   Per-stay sequence dataset (for GRU)
│   │   ├── trainer.py               #   Training loop with early stopping
│   │   └── inference.py             #   Model loading and prediction API
│   ├── evaluation/                  # Phase 4: Evaluation and baselines
│   │   ├── eval_runner.py           #   Full evaluation orchestrator
│   │   ├── metrics.py               #   AUROC, AUPRC, F1, Brier, ECE computation
│   │   ├── baseline.py              #   Logistic regression baseline
│   │   ├── calibration.py           #   Temperature scaling
│   │   └── plots.py                 #   ROC, calibration, uncertainty plots
│   ├── decision/                    # Phase 5a: Rule-based decision engine
│   │   ├── engine.py                #   Decision matrix lookup
│   │   └── rules.py                 #   Risk levels and decision matrix definition
│   ├── explanation/                 # Phase 5b: Natural language explanations
│   │   ├── pipeline.py              #   Full inference → decision → explanation pipeline
│   │   ├── generator.py             #   Template-based explanation generator
│   │   └── shap_explainer.py        #   SHAP-based feature attribution
│   └── utils/                       # Shared utilities
│       ├── schema.py                #   Data classes: TimeWindow, PatientRecord, ContextObject, SystemOutput
│       ├── config_loader.py         #   YAML config singleton
│       ├── logger.py                #   Logging setup
│       └── device.py                #   GPU/CPU device detection
├── tests/                           # Unit tests
│   ├── test_preprocessing.py        #   Vital clipping and windowing tests
│   ├── test_context.py              #   Trend slope computation tests
│   └── test_model.py                #   MC Dropout output shape/range tests
├── scripts/
│   └── verify_preprocessing.py      #   Preprocessing QA script
├── data/
│   ├── raw/                         #   MIMIC-IV-ED CSV files (gitignored)
│   ├── processed/                   #   PatientRecord pickles (gitignored)
│   ├── context/                     #   ContextObject pickles (gitignored)
│   └── splits/                      #   Train/val/test split JSON
├── outputs/
│   ├── models/                      #   Saved model checkpoints
│   ├── results/                     #   Evaluation metrics JSONs
│   └── plots/                       #   Generated evaluation plots
└── config.yaml                      #   Centralized configuration
```

---

## 3. Data Source: MIMIC-IV-ED v2.2

The system uses six tables from the MIMIC-IV-ED dataset:

| Table | File | Size | Description |
|-------|------|------|-------------|
| **edstays** | `edstays.csv` | 38 MB | Core ED stay records (subject_id, stay_id, intime, outtime, hadm_id, disposition) |
| **triage** | `triage.csv` | 37 MB | Triage assessments (acuity, chief complaint, initial vitals) |
| **vitalsign** | `vitalsign.csv` | 115 MB | Time-stamped vital signs during ED stay |
| **diagnosis** | `diagnosis.csv` | 50 MB | ICD diagnosis codes assigned during visit |
| **medrecon** | `medrecon.csv` | 360 MB | Medication reconciliation records |
| **pyxis** | `pyxis.csv` | 106 MB | Automated dispensing cabinet medication records |

**Total raw data: ~706 MB**

### Outcome Definition

The prediction target is **hospital admission**:
- `outcome_binary = 1` (admitted): `hadm_id IS NOT NULL` in edstays — patient was admitted to hospital
- `outcome_binary = 0` (discharged): `hadm_id IS NULL` — patient discharged home

This is a standard, clinically validated approach for ED admission prediction using MIMIC-IV-ED.

---

## 4. Phase 1: Data Preprocessing (`src/preprocessing/`)

### 4.1 Data Loading (`loader.py`)

Six CSV tables are loaded with appropriate date parsing:
- `edstays.csv`: `intime` and `outtime` parsed as datetime
- `vitalsign.csv`: `charttime` parsed as datetime
- `medrecon.csv`, `pyxis.csv`: `charttime` parsed as datetime

### 4.2 Data Cleaning (`cleaner.py`)

**Encounter filtering:**
- Encounters shorter than 2 hours are dropped (configurable via `min_encounter_hours`)
- This removes visits too brief for meaningful vital sign trajectories

**Outcome derivation:**
- Binary admission label derived from `hadm_id` presence in edstays

**Vital sign cleaning:**
- Non-numeric values are coerced via regex extraction (`[-+]?\d*\.?\d+`)
- Temperature normalization: values > 60 are assumed Fahrenheit and converted to Celsius
- Physiological clipping ranges applied:

| Vital | Min | Max |
|-------|-----|-----|
| Heart Rate | 20 | 280 bpm |
| Respiratory Rate | 4 | 60 breaths/min |
| SpO2 | 50 | 100% |
| Systolic BP | 40 | 260 mmHg |
| Diastolic BP | 20 | 160 mmHg |
| Temperature | 34 | 43 °C |
| Pain | 0 | 10 |

- Duplicate vital readings at the same timestamp are deduplicated (keeping last)
- Vital signs are time-referenced relative to ED arrival (`t_minutes = charttime - intime`)
- Only observations within 0–1440 minutes (24 hours) of arrival are retained

**Triage cleaning:**
- Acuity values clipped to 1–5 range, missing filled with 3.0 (median ESI)
- Chief complaint lowercased and stripped

**Sepsis flag derivation:**
- ICD diagnosis titles are searched for keywords: `sepsis`, `infection`, `pneumonia`, `bacteremia`, `cellulitis`
- Binary flag: 1 if any diagnosis matches, 0 otherwise
- Count of total diagnoses also captured

**Medication count:**
- Unique medications counted per stay (using `etc_rn == 1` to avoid duplicates)

### 4.3 Temporal Windowing (`windower.py`)

Vital signs are aggregated into **60-minute non-overlapping windows** from ED arrival:

For each window and each of 7 vital features:
- **4 statistical aggregates**: mean, last, min, max → 28 features per window
- **Missingness mask**: binary flag (1 = observed, 0 = missing) for each vital → 7 flags
- **Observation density**: fraction of vitals observed (0.0–1.0)
- **Observation count**: raw number of measurements in the window

A rolling window function is also provided for computing features over configurable horizons (1h, 3h, 6h).

### 4.4 Missingness Analysis (`missingness.py`)

- **Critical feature tracking**: SBP and heart rate are designated as critical vitals — their absence reduces data confidence
- **Consecutive missingness**: counts how many sequential windows lack a given feature
- **Triage completeness**: fraction of vital features present at triage time
- **Patient-level missingness summary**: average observation rate per feature across all windows

### 4.5 Pipeline Orchestration (`pipeline.py`)

The full preprocessing pipeline:

1. Load all 6 raw tables
2. Filter encounters (≥ 2 hours)
3. Derive outcome labels
4. Clean vitalsign, triage, diagnosis, medrecon tables
5. For each stay:
   - Build temporal windows from vital signs
   - Extract triage vitals
   - Look up sepsis flag, diagnosis count, medication count
   - Assemble into a `PatientRecord` dataclass
6. Save as `patient_records.pkl` (1.9 GB)
7. Save sample JSON for inspection
8. Create patient-level train/val/test splits (70/15/15)

**Data leakage prevention:** Splits are performed at the **patient level** (by `subject_id`), not by stay. This ensures that multiple visits from the same patient are all assigned to the same split, preventing information leakage across train/val/test boundaries.

### 4.6 Data Schemas (`utils/schema.py`)

Four core data classes define the pipeline's data contracts:

```python
@dataclass
class TimeWindow:
    window_index: int
    window_start_min: float     # e.g. 0.0
    window_end_min: float       # e.g. 60.0
    features: Dict[str, float]  # 28 vital aggregates
    missingness_mask: Dict[str, int]  # 7 binary flags
    observation_density: float
    n_observations: int

@dataclass
class PatientRecord:
    patient_id: str
    stay_id: str
    triage_vitals: Dict[str, float]  # 7 initial vitals
    acuity: float                     # ESI 1-5
    chiefcomplaint: str
    sepsis_flag: int
    n_diagnoses: int
    n_medications: int
    time_windows: List[TimeWindow]
    outcome_label: str               # "admitted" or "discharge"
    outcome_binary: int              # 1 or 0
    duration_hours: float

@dataclass
class ContextObject:
    patient_id: str
    stay_id: str
    window_index: int
    physiological_state: str          # stable/concerning/deteriorating/critical
    trends: Dict[str, str]            # per-vital trend labels
    slopes: Dict[str, float]          # per-vital slopes (units/hour)
    data_confidence: str              # high/medium/low
    observation_density: float
    missing_critical: List[str]
    deterioration_flag: bool
    acuity: float
    sepsis_flag: int
    n_medications: int
    context_vector: List[float]       # 72-dim numeric vector

@dataclass
class SystemOutput:
    patient_id: str
    stay_id: str
    window_index: int
    risk_probability: float
    uncertainty_score: float
    confidence_level: str             # High/Medium/Low
    decision_category: str            # e.g. ESCALATE_IMMEDIATELY
    urgency: str                      # immediate/urgent/routine
    recommended_tests: List[str]
    explanation: str
    context_summary: Dict
```

---

## 5. Phase 2: Context Construction (`src/context/`)

The context layer transforms raw `PatientRecord` objects into clinically enriched `ContextObject` instances — one per patient per time window. This is the system's core innovation: encoding not just current vital values, but temporal trends, physiological state assessments, data quality metrics, and derived clinical markers into a fixed-length 72-dimensional vector.

### 5.1 Trend Engine (`trend_engine.py`)

**Slope computation:**
- For each vital feature, a least-squares linear regression slope is computed over the window history
- Times are converted to hours; values with NaN are excluded
- Minimum 2 valid data points required for slope calculation

**Slope classification thresholds** (configurable in `config.yaml`):

| Slope (units/hour) | Classification |
|---------------------|----------------|
| > 5.0 | `rising_fast` |
| > 2.0 | `rising` |
| -2.0 to 2.0 | `stable` |
| < -2.0 | `falling` |
| < -5.0 | `falling_fast` |
| NaN (insufficient data) | `unknown` |

**Deterioration flag:**
- Set to `true` when heart rate is rising AND systolic BP is falling — a classic hemodynamic instability pattern

### 5.2 State Classifier (`state_classifier.py`)

**Physiological state classification** (4 levels):

| State | Criteria |
|-------|----------|
| `critical` | Deterioration flag + any critical vital changing fast |
| `deteriorating` | Deterioration flag active, OR ≥ 2 vitals changing fast |
| `concerning` | 1 vital changing fast, OR ≥ 2 vitals changing mildly |
| `stable` | None of the above |

**Data confidence classification** (3 levels):

| Confidence | Criteria |
|------------|----------|
| `high` | Density ≥ 0.70 AND no critical vitals missing |
| `medium` | Density ≥ 0.30 OR (density ≥ 0.70 with critical vitals missing) |
| `low` | Density < 0.30 OR critical vitals missing with low density |

### 5.3 Context Vector Construction (`vector_builder.py`)

Each context object is encoded as a **72-dimensional numeric vector** with the following structure:

| Index Range | Dimensions | Content |
|-------------|------------|---------|
| [0, 28) | 28 | Current window vital aggregates (7 vitals × 4 stats: mean, last, min, max) |
| [28, 35) | 7 | Trend encodings (integer: -2 to +2) |
| [35, 42) | 7 | Slope values (clipped to [-20, 20]) |
| [42] | 1 | Physiological state (0=stable, 1=concerning, 2=deteriorating, 3=critical) |
| [43] | 1 | Data confidence (0.0=low, 0.5=medium, 1.0=high) |
| [44] | 1 | Observation density (0.0–1.0) |
| [45, 52) | 7 | Missingness flags (binary per vital) |
| [52, 59) | 7 | Triage vitals (7 baseline vital values from initial assessment) |
| [59] | 1 | Acuity (ESI score normalized: (5 - acuity) / 4) |
| [60] | 1 | Sepsis flag (binary) |
| [61] | 1 | Medication count (clipped to 30, normalized to [0, 1]) |
| [62] | 1 | Missing critical flag (binary: any critical vital missing) |
| [63] | 1 | **Shock index** (HR / SBP, clipped to [0, 3]) |
| [64] | 1 | **Pulse pressure** ((SBP - DBP) / 100, clipped to [-1, 2]) |
| [65, 72) | 7 | **Vital deviations from triage** (relative change from baseline) |
| **Total** | **72** | |

Notable engineered features:
- **Shock index** (index 63): Heart rate divided by systolic blood pressure. Clinically, values > 0.7 indicate hemodynamic stress; > 1.0 suggests shock.
- **Pulse pressure** (index 64): SBP − DBP. Narrow pulse pressure suggests cardiogenic shock; wide suggests aortic regurgitation or atherosclerosis.
- **Vital deviations from triage** (indices 65–71): Relative change `(current - triage) / |triage|` for each vital, capturing how much the patient has changed from their baseline.

NaN handling: All NaN values are replaced with 0.0 via the `_safe()` function before vector construction.

### 5.4 Builder Orchestration (`builder.py`)

For each patient record:
1. Iterate over each time window
2. Compute trends and slopes using all history up to current window
3. Identify missing critical features
4. Check deterioration flag
5. Classify physiological state
6. Classify data confidence
7. Build 72-dim context vector
8. Assemble `ContextObject`

Output: `context_objects.pkl` (1.8 GB) containing ~2.66 million context objects from ~162K patients.

---

## 6. Phase 3: Model Training (`src/model/`)

### 6.1 Model Architecture: MC-Dropout GRU (`mc_dropout_gru.py`)

The primary model is a **Gated Recurrent Unit (GRU)** with attention pooling and Monte Carlo Dropout for uncertainty estimation.

```
Input: (batch, seq_len, 72)  — variable-length sequences of context vectors
  │
  ▼
GRU (2 layers, hidden_dim=128, inter-layer dropout=0.35)
  │
  ▼
Attention Pooling (learned attention weights over timesteps)
  │
  ▼
LayerNorm(128)
  │
  ▼
Dropout(0.35) → Linear(128, 64) → LayerNorm(64) → ReLU
  │
  ▼
Dropout(0.35) → Linear(64, 32) → ReLU
  │
  ▼
Dropout(0.35) → Linear(32, 1)  → raw logit output
```

**Attention mechanism:**
- A learned linear projection (`attn_w`) computes scalar attention scores for each timestep
- Scores are masked for padded positions (set to -inf before softmax)
- Softmax-normalized weights produce a weighted sum over GRU hidden states
- This allows the model to focus on the most informative time windows rather than just the last one

**Why GRU over LSTM:**
- Fewer parameters (no separate cell state)
- Comparable performance on clinical time series
- Faster training with similar expressiveness

### 6.2 Alternative Model: MC-Dropout MLP (`mc_dropout.py`)

A simpler feedforward network is also available:

```
Input: (batch, 72)
  │
  ▼
Linear(72, 256) → BatchNorm → ReLU → Dropout(0.35)
  │
  ▼
Linear(256, 128) → BatchNorm → ReLU → Dropout(0.35)
  │
  ▼
Linear(128, 64) → BatchNorm → ReLU → Dropout(0.35)
  │
  ▼
Linear(64, 1)  → raw logit output
```

The MLP uses only the last window per stay (losing temporal information), while the GRU processes the full sequence.

### 6.3 Monte Carlo Dropout for Uncertainty Estimation

Both models implement `predict_with_uncertainty()`:
1. Set model to eval mode (freezes BatchNorm/LayerNorm running statistics)
2. **Re-enable only Dropout layers** in train mode
3. Run `n_samples` (default: 50) forward passes through the stochastic network
4. Apply sigmoid to get probabilities
5. Return:
   - **Mean probability**: average across MC samples → point risk estimate
   - **Variance**: variance across MC samples → uncertainty score
   - **Raw predictions**: all MC samples for downstream analysis

**Uncertainty classification thresholds:**
| Variance | Confidence Level |
|----------|-----------------|
| < 0.01 | High confidence |
| 0.01 – 0.05 | Medium confidence |
| ≥ 0.05 | Low confidence |

### 6.4 Datasets

**EDSequenceDataset** (for GRU):
- Groups context objects by `(patient_id, stay_id)`
- Sorts windows within each stay by `window_index`
- Produces variable-length sequences of 72-dim vectors
- Skips entire stays if any window contains NaN values
- Custom `sequence_collate_fn` pads sequences to equal length within each batch

**EDContextDataset** (for MLP):
- Keeps only the last window per stay (highest `window_index`)
- Produces flat 72-dim vectors
- Skips contexts with NaN values

### 6.5 Training Loop (`trainer.py`)

**Data splitting:**
- 80/20 patient-level split for train/validation
- Deterministic via `np.random.seed(42)`

**Feature standardization:**
- Mean and standard deviation computed on training set only
- Applied to all data (train + val)
- Saved with model checkpoint for inference-time normalization

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight_decay=5e-3) |
| Base learning rate | 0.0005 |
| Batch size | 512 |
| Max epochs | 200 |
| Early stopping patience | 30 epochs |
| Gradient clipping | max_norm=1.0 |
| Label smoothing | 0.05 |
| LR warmup | 5 epochs (linear) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |

**Class balancing:**
- `pos_weight` computed as `n_neg / n_pos` for BCEWithLogitsLoss
- Addresses class imbalance (~31% admission rate)

**Label smoothing:**
- Targets transformed: `y = y * (1 - 0.05) + 0.5 * 0.05`
- Prevents overconfident predictions and improves calibration

**Model saving:**
- Best model saved based on validation AUROC
- Checkpoint includes: model state dict, model type, input dimension, feature mean/std

**GPU optimization:**
- `torch.backends.cudnn.benchmark = True`
- `torch.set_float32_matmul_precision("high")`
- Data loaded directly to GPU when available
- DataLoader workers set to 0 for GPU (data already on device)

### 6.6 Inference API (`inference.py`)

Provides three interfaces:

1. **`predict(context)`**: Single context object prediction (works for both MLP and GRU)
2. **`predict_sequence(context_list)`**: Sequence prediction for GRU (multiple windows)
3. **`load_model()`**: Lazy model loading with caching

All predictions apply the saved feature normalization before inference.

---

## 7. Phase 4: Evaluation (`src/evaluation/`)

### 7.1 Evaluation Pipeline (`eval_runner.py`)

The evaluation process:

1. **Load data**: Context objects (1.8 GB, 2.66M objects) and label map (162K patients)
2. **Split into test/validation sets** using saved `splits.json` (patient-level)
3. **Validation inference**: Run MC Dropout inference on val set
4. **Fit temperature scaling** on validation logits (post-hoc calibration)
5. **Test inference**: Run MC Dropout inference on test set (321,888 stays)
6. **Compute metrics**: Raw and temperature-calibrated
7. **Generate plots**: ROC curve, calibration curve, uncertainty distribution
8. **Run baseline**: Logistic regression for comparison

### 7.2 Metrics (`metrics.py`)

| Metric | Description | Value |
|--------|-------------|-------|
| **AUROC** | Area Under ROC Curve — overall discrimination | 0.779 |
| **AUPRC** | Area Under Precision-Recall Curve — performance on positive class | 0.593 |
| **F1 Score** | Harmonic mean of precision and recall (threshold=0.5) | 0.603 |
| **Brier Score** | Mean squared prediction error — calibration + discrimination | 0.194 |
| **ECE** | Expected Calibration Error (10 bins) | 0.140 |
| **Prevalence** | Fraction of positive cases in test set | 31.3% |

**Uncertainty-stratified AUROC:**
- High confidence (variance < 0.01): AUROC = 0.779
- Medium confidence (0.01 ≤ variance < 0.05): AUROC = 0.615
- Low confidence (variance ≥ 0.05): insufficient samples

### 7.3 Temperature Scaling (`calibration.py`)

Post-hoc calibration technique that learns a single scalar temperature `T` on the validation set:

```
calibrated_probability = sigmoid(logit / T)
```

- Implemented as a PyTorch module with a single learnable parameter
- Optimized using L-BFGS (200 iterations) to minimize BCEWithLogitsLoss on validation set
- Does not change model weights — only rescales the decision boundary
- Addresses the systematic overconfidence visible in the calibration curve

### 7.4 Baseline: Logistic Regression (`baseline.py`)

A standardized logistic regression baseline:
- Uses the same 72-dim context vectors as features
- StandardScaler → LogisticRegression(max_iter=1000, class_weight="balanced")
- 80/20 random split (not patient-level — simpler baseline)
- AUROC: **0.661** — substantially lower than the GRU's 0.779

### 7.5 Evaluation Plots

**ROC Curve** (`outputs/plots/roc_curve.png`):
- Smooth concave curve well above the random baseline diagonal
- AUC = 0.779
- At FPR ≈ 0.2, the model captures ~55-60% of true positives
- At FPR ≈ 0.4, captures ~80% of true positives

**Calibration Curve** (`outputs/plots/calibration.png`):
- Model's calibration curve lies consistently below the perfect calibration diagonal
- Indicates systematic overconfidence: predicted probabilities exceed observed frequencies
- At predicted probability 0.5, actual positive rate is ~27%
- At predicted probability 0.8, actual positive rate is ~60%
- Temperature scaling improves but does not fully resolve this

**Uncertainty Distribution** (`outputs/plots/uncertainty_dist.png`):
- Both outcomes (admitted/discharged) have heavily right-skewed uncertainty distributions
- Vast majority of predictions have very low uncertainty (< 0.005)
- No adverse event cases dominate by count (~2:1 ratio)
- Limited separation between outcome groups' uncertainty distributions

---

## 8. Phase 5: Decision Support (`src/decision/` + `src/explanation/`)

### 8.1 Rule-Based Decision Engine (`decision/`)

The decision engine maps three inputs to structured clinical recommendations:

**Inputs:**
1. **Risk level**: high (≥ 0.70), moderate (0.40–0.70), low (< 0.40)
2. **Confidence level**: High, Medium, Low (from MC Dropout variance)
3. **Trend state**: worsening (deteriorating/critical) or stable

**Decision matrix** (12+ entries covering key combinations):

| Risk | Confidence | Trend | Decision | Urgency | Reassessment |
|------|-----------|-------|----------|---------|--------------|
| High | High | Worsening | ESCALATE_IMMEDIATELY | Immediate | 15 min |
| High | High | Stable | MONITOR_AND_PREPARE | Urgent | 30 min |
| High | Medium | Worsening | ORDER_ADDITIONAL_TESTS | Urgent | 30 min |
| High | Low | Worsening | ORDER_ADDITIONAL_TESTS | Urgent | 20 min |
| Moderate | High | Worsening | ORDER_ADDITIONAL_TESTS | Routine | 30 min |
| Moderate | High | Stable | MONITOR_CLOSELY | Routine | 60 min |
| Low | High | Stable | CONTINUE_MONITORING | Routine | 60 min |
| Low | Low | Stable | GATHER_MORE_DATA | Routine | 60 min |

**Recommended tests** are context-dependent:
- High-risk + low confidence: lactate repeat, blood culture, chest X-ray
- High-risk + medium confidence: lactate repeat, CBC
- Moderate-risk + worsening: lactate, CBC

**Reasoning tags** are generated for audit trail: `risk_high`, `confidence_low`, `missing_sbp`, `heartrate_rising_fast`, etc.

### 8.2 Explanation Generator (`explanation/generator.py`)

Generates human-readable clinical explanations using template-based natural language generation:

**Sentence structure:**
1. **Risk drivers**: "The elevated risk prediction is driven by [feature trends]"
2. **Vital trends**: "Over the observation window, [vital] is [trend]"
3. **Physiological state**: "The overall trajectory is [state assessment]"
4. **Uncertainty caveat** (if Medium/Low): "Confidence is [level] because [reason]"
5. **Action recommendation**: "[Clinical action] is recommended"

**Example output:**
```
The elevated risk prediction is driven by heart rate (rapidly rising)
and systolic blood pressure (falling rapidly). Over the observation window,
heart rate is rapidly rising and systolic blood pressure is falling rapidly.
The overall trajectory indicates deterioration. Confidence is low because
systolic blood pressure and heart rate data is missing, limiting trend certainty.
Immediate clinical escalation is recommended.
```

### 8.3 SHAP Explainer (`explanation/shap_explainer.py`)

Optional SHAP (SHapley Additive exPlanations) integration:
- Uses `shap.KernelExplainer` with 50 background samples
- Computes per-feature attribution values (100 SHAP samples)
- Returns features ranked by absolute SHAP value
- Currently passed as `None` in the pipeline (template-based explanations used by default)

### 8.4 Full Pipeline (`explanation/pipeline.py`)

The `run_full_pipeline(context)` function chains:
1. **Inference**: MC Dropout prediction → risk probability + uncertainty
2. **Decision**: Rule engine → category, urgency, recommended tests
3. **Explanation**: NLG generator → human-readable text
4. **Assembly**: `SystemOutput` dataclass with all results

---

## 9. Configuration (`config.yaml`)

All hyperparameters and paths are centralized in a single YAML file:

```yaml
paths:
  raw_data: data/raw/
  processed_data: data/processed/
  context_data: data/context/
  splits: data/splits/
  model_output: outputs/models/
  results: outputs/results/
  plots: outputs/plots/

preprocessing:
  min_encounter_hours: 2
  max_forward_fill_minutes: 60
  vital_features: [heartrate, resprate, o2sat, sbp, dbp, temperature, pain]
  vital_clipping_ranges: {heartrate: [20, 280], resprate: [4, 60], ...}
  sepsis_icd_keywords: [sepsis, infection, pneumonia, bacteremia, cellulitis]

context:
  slope_stable_threshold: 2.0    # units/hour
  slope_fast_threshold: 5.0      # units/hour
  density_high_threshold: 0.70
  density_low_threshold: 0.30
  critical_missing_features: [sbp, heartrate]

model:
  model_type: "gru"
  input_dim: 72
  hidden_dims: [256, 128, 64]    # MLP layers
  dropout_rate: 0.35
  mc_dropout_samples: 50
  learning_rate: 0.0005
  batch_size: 512
  max_epochs: 200
  early_stopping_patience: 30
  label_smoothing: 0.05
  warmup_epochs: 5
  gru_hidden_dim: 128
  gru_num_layers: 2
  uncertainty_thresholds: {high: 0.05, medium: 0.01}

decision:
  risk_high_threshold: 0.70
  risk_moderate_threshold: 0.40

evaluation:
  n_bootstrap: 1000
  high_missingness_threshold: 0.30
  cv_folds: 5
```

The config is loaded as a singleton via `get_config()` — parsed once and cached globally.

---

## 10. Testing (`tests/`)

Three test modules covering core functionality:

**`test_preprocessing.py`:**
- `test_clip_vital_ranges()`: Verifies vital sign clipping respects configured bounds
- `test_missingness_mask()`: Confirms empty windows produce all-zero missingness masks

**`test_context.py`:**
- `test_rising_fast()`: Validates slope computation and "rising_fast" classification for rapidly increasing heart rate
- `test_stable()`: Validates "stable" classification for flat vital trajectories

**`test_model.py`:**
- `test_mc_dropout_output()`: Verifies MC Dropout produces valid probability [0,1], non-negative variance, and correct number of MC samples

---

## 11. Data Flow Summary

```
MIMIC-IV-ED CSVs (706 MB)
    │
    ▼  [loader.py]
6 DataFrames (edstays, triage, vitalsign, diagnosis, medrecon, pyxis)
    │
    ▼  [cleaner.py + windower.py]
PatientRecords with TimeWindows (1.9 GB pickle)
    │
    ▼  [builder.py + trend_engine.py + vector_builder.py]
ContextObjects with 72-dim vectors (1.8 GB pickle, ~2.66M objects)
    │
    ▼  [sequence_dataset.py + trainer.py]
MC-Dropout GRU Model (704 KB checkpoint)
    │
    ▼  [eval_runner.py + inference.py]
Evaluation Metrics + Plots
    │
    ▼  [decision/engine.py + explanation/generator.py]
SystemOutput: Risk + Uncertainty + Decision + Explanation
```

---

## 12. Key Design Decisions and Technical Highlights

### 12.1 Temporal Modeling via GRU

Unlike approaches that flatten all features into a single vector, this system preserves temporal ordering by feeding sequences of context vectors into a GRU. This allows the model to learn patterns like "heart rate rising across windows while BP drops" — patterns invisible to a single-timepoint model. The attention mechanism further allows the model to weight important time windows more heavily.

### 12.2 Uncertainty-Aware Predictions

MC Dropout provides a principled Bayesian approximation for prediction uncertainty without requiring ensemble training. The system runs 50 stochastic forward passes and uses the variance of predictions as an uncertainty estimate. This uncertainty directly influences clinical decisions — low-confidence predictions trigger additional testing rather than immediate escalation.

### 12.3 Clinical Context Engineering

The 72-dimensional context vector encodes domain knowledge that raw vitals alone cannot capture:
- **Shock index** (HR/SBP) is a validated hemodynamic marker
- **Pulse pressure** indicates cardiovascular pathology patterns
- **Deviations from triage baseline** capture individual-specific deterioration
- **Trend encodings** compress temporal patterns into actionable categories
- **Data confidence scoring** ensures the system knows when it doesn't know

### 12.4 Post-Hoc Temperature Scaling

Temperature scaling addresses the calibration gap evident in the calibration curve. By fitting a single scalar on the validation set, the system produces better-calibrated probabilities without retraining, maintaining discriminative performance (AUROC unchanged).

### 12.5 Patient-Level Data Splitting

All train/val/test splits are performed at the patient level. Patients with multiple ED visits have all visits in the same split, preventing the model from memorizing patient-specific patterns during training and then exploiting them at test time.

### 12.6 Structured Decision Support

Rather than outputting only a probability, the system produces structured clinical recommendations (escalate, monitor, test, gather data) with urgency levels and reassessment intervals. The decision matrix considers the interaction of risk level, prediction confidence, and physiological trajectory — matching how clinicians actually reason.

---

## 13. Scale and Performance

| Metric | Value |
|--------|-------|
| Raw data size | 706 MB (6 CSVs) |
| Processed records | 1.9 GB (PatientRecords) |
| Context objects | 1.8 GB (~2.66M objects, ~162K patients) |
| Test set stays | 321,888 |
| Model size | 704 KB |
| GRU inference | ~1,258 batches × 50 MC samples ≈ 57 seconds on RTX 5070 |
| Context vector dimension | 72 |
| GRU hidden dimension | 128 (2 layers) |
| Total trainable parameters | ~150K (estimated) |

---

## 14. Dependencies and Environment

Core dependencies:
- **PyTorch** — model training and inference
- **scikit-learn** — metrics, logistic regression baseline, calibration curves
- **pandas / numpy** — data manipulation
- **matplotlib** — evaluation plots
- **SHAP** — feature attribution (optional)
- **tqdm** — progress bars
- **PyYAML** — configuration
- **CUDA** — GPU acceleration (tested on NVIDIA RTX 5070)

---

## 15. Limitations and Future Work

### Current Limitations
1. **Calibration gap**: ECE of 0.140 shows the model systematically overestimates risk. Temperature scaling helps but doesn't fully resolve this.
2. **Uncertainty distribution**: MC Dropout uncertainty scores are concentrated near zero with limited separation between outcomes, suggesting the uncertainty estimates may not be fully informative for clinical routing.
3. **Missing data handling**: NaN values are replaced with 0.0, which may introduce bias. More sophisticated imputation (e.g., carry-forward, learned imputation) could improve performance.
4. **Deterioration flag**: Currently checks for `heart_rate` and `systolic_bp` keys in the trends dictionary, but the context vectors use `heartrate` and `sbp` — a naming inconsistency that means the deterioration flag may not trigger correctly.
5. **Baseline fairness**: The logistic regression baseline uses a simple random split rather than patient-level splitting, making the comparison slightly unfavorable to the baseline.

### Potential Improvements
- Platt scaling or isotonic regression as alternative calibration methods
- Transformer-based temporal modeling for longer sequences
- Multi-task learning (predict admission + length of stay + ICU transfer)
- Integration of lab results and imaging orders as additional features
- Prospective validation on held-out time periods
- Fairness analysis across demographic subgroups
