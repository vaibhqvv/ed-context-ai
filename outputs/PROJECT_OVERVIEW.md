# ED-Context-AI — Project Overview

Emergency Department Clinical Decision Support System using Uncertainty-Aware MC-Dropout GRU on MIMIC-IV-ED v2.2

---

## 1. File Structure

```
ed-context-ai/
├── main.py                          (59 lines)   CLI entry point
│                                                  --phase preprocess|context|train|evaluate|demo|all
├── config.yaml                      (71 lines)   All hyperparameters, paths, thresholds
├── requirements.txt                               Python dependencies
├── PROJECT_DOCUMENT.md                            Full project documentation
├── CLAUDE.md                                      Dev instructions
├── README.md
│
├── src/                                           Source code — 2,598 lines across 30 modules
│   │
│   ├── preprocessing/                             PHASE 1: Raw CSV → PatientRecords
│   │   ├── loader.py                (63 lines)   CSV file loading with date parsing
│   │   ├── cleaner.py              (166 lines)   Vital clipping, unit conversion,
│   │   │                                          deduplication, outcome derivation
│   │   ├── windower.py              (92 lines)   60-min temporal window aggregation
│   │   │                                          (mean/last/min/max per vital)
│   │   ├── missingness.py           (43 lines)   Critical feature missing detection,
│   │   │                                          consecutive missingness tracking
│   │   └── pipeline.py             (210 lines)   End-to-end preprocessing orchestrator
│   │
│   ├── context/                                   PHASE 2: PatientRecords → 72-dim ContextObjects
│   │   ├── builder.py               (94 lines)   Main context construction loop
│   │   ├── trend_engine.py          (46 lines)   Least-squares slope computation &
│   │   │                                          trend classification (stable→rising_fast)
│   │   ├── state_classifier.py      (33 lines)   Physiological state assessment
│   │   │                                          (stable/concerning/deteriorating/critical)
│   │   └── vector_builder.py       (117 lines)   72-dimensional context vector assembly
│   │                                              with derived hemodynamic markers
│   │
│   ├── model/                                     PHASE 3: Training & Inference
│   │   ├── mc_dropout_gru.py        (95 lines)   2-layer GRU + attention + MC Dropout
│   │   ├── mc_dropout.py            (44 lines)   MLP with MC Dropout (alternative)
│   │   ├── dataset.py               (52 lines)   EDContextDataset — flat, last window (MLP)
│   │   ├── sequence_dataset.py      (64 lines)   EDSequenceDataset — variable-length (GRU)
│   │   ├── trainer.py              (245 lines)   Training loop: AdamW, early stopping,
│   │   │                                          warmup LR, label smoothing, class weights
│   │   └── inference.py             (94 lines)   Model loading & prediction API
│   │
│   ├── evaluation/                                PHASE 4: Metrics, Calibration & Plots
│   │   ├── eval_runner.py          (316 lines)   Full evaluation pipeline orchestrator
│   │   ├── metrics.py               (64 lines)   AUROC, AUPRC, F1, Brier, ECE computation
│   │   ├── calibration.py           (69 lines)   Temperature scaling (post-hoc)
│   │   ├── baseline.py              (58 lines)   Logistic regression baseline
│   │   └── plots.py                 (54 lines)   ROC curve, calibration, uncertainty dist
│   │
│   ├── decision/                                  PHASE 5a: Rule-Based Clinical Decisions
│   │   ├── rules.py                (107 lines)   16-rule decision matrix definition
│   │   │                                          (risk × confidence × trend → action)
│   │   └── engine.py                (32 lines)   Risk classifier + matrix lookup
│   │
│   ├── explanation/                               PHASE 5b: Natural Language Explanations
│   │   ├── generator.py            (100 lines)   Template-based NLG (5 sentences max)
│   │   ├── shap_explainer.py        (41 lines)   SHAP KernelExplainer feature attribution
│   │   └── pipeline.py              (48 lines)   Inference → Decision → Explanation chain
│   │
│   └── utils/                                     Shared Utilities
│       ├── schema.py                (76 lines)   Dataclasses: TimeWindow, PatientRecord,
│       │                                          ContextObject, SystemOutput
│       ├── config_loader.py         (25 lines)   Singleton YAML config via get_config()
│       ├── logger.py                (16 lines)   Logging setup via get_logger()
│       └── device.py                (24 lines)   CUDA/CPU auto-detection
│
├── data/
│   ├── raw/                         (706 MB)     6 MIMIC-IV-ED CSV files (gitignored)
│   │   ├── edstays.csv              (38 MB)
│   │   ├── triage.csv               (37 MB)
│   │   ├── vitalsign.csv           (115 MB)
│   │   ├── diagnosis.csv            (50 MB)
│   │   ├── medrecon.csv            (360 MB)
│   │   └── pyxis.csv              (106 MB)
│   ├── processed/                   (1.9 GB)     patient_records.pkl (gitignored)
│   │   └── sample_records.json                   Sample for inspection
│   └── context/                     (1.8 GB)     context_objects.pkl (gitignored)
│       └── sample_contexts.json                  Sample for inspection
│
├── outputs/
│   ├── models/
│   │   └── best_model.pt            (704 KB)     Trained GRU checkpoint + normalization stats
│   ├── plots/
│   │   ├── roc_curve.png                          ROC curve (AUC = 0.779)
│   │   ├── calibration.png                        Calibration curve (10 bins)
│   │   └── uncertainty_dist.png                   MC Dropout uncertainty distribution
│   └── results/
│       ├── main_model_metrics.json                GRU model metrics
│       ├── baseline_lr_metrics.json               Logistic regression baseline metrics
│       └── comparison_metrics.json                Side-by-side comparison
│
├── tests/                                         pytest unit tests
│   ├── test_preprocessing.py        (20 lines)   Vital clipping, windowing tests
│   ├── test_context.py              (18 lines)   Trend slope computation tests
│   └── test_model.py                (13 lines)   MC Dropout output shape/range tests
│
├── scripts/
│   └── verify_preprocessing.py                    Preprocessing QA script
│
└── splits/                                        Train/val/test split JSON
```

---

## 2. Full Tech Stack

### Core Runtime

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.12.9 (pyenv) | Core runtime |
| Deep Learning | PyTorch | ≥ 2.0.0 | GRU model, MC Dropout, GPU training |
| GPU Acceleration | CUDA | 12.8 | NVIDIA RTX 5070 |
| Numerical | NumPy | ≥ 1.24.0, < 2.0 | Array operations, statistics |
| Data Manipulation | pandas | ≥ 2.0.0 | CSV loading, table joins, filtering |
| Scientific Computing | SciPy | ≥ 1.11.0 | L-BFGS optimization for temperature scaling |

### Machine Learning

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Classical ML | scikit-learn | ≥ 1.3.0 | LogisticRegression baseline, StandardScaler, all metrics |
| Explainability | SHAP | ≥ 0.44.0 | KernelExplainer for feature attribution |

### Visualization & Reporting

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Plotting | matplotlib | ≥ 3.7.0 | ROC, calibration, uncertainty plots |
| Statistical Viz | seaborn | ≥ 0.12.0 | Distribution overlays |

### Infrastructure

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Configuration | PyYAML | ≥ 6.0 | config.yaml singleton parsing |
| Progress Bars | tqdm | ≥ 4.65.0 | Training/inference progress tracking |
| Testing | pytest | ≥ 7.4.0 | Unit tests |
| Notebooks | Jupyter | ≥ 1.0 | Interactive exploration |
| Serialization | pickle (stdlib) | — | PatientRecords, ContextObjects persistence |
| Version Control | Git | — | Source control |

### Data Source

| Component | Details |
|-----------|---------|
| Dataset | MIMIC-IV-ED v2.2 |
| Provider | PhysioNet / MIT |
| Hospital | Beth Israel Deaconess Medical Center |
| Time Period | 2008–2019 |
| Access | Credentialed (requires CITI training + DUA) |

### Model Architecture Summary

| Property | Value |
|----------|-------|
| Architecture | 2-Layer GRU + Attention Pooling + FC Head |
| Input | Variable-length sequences of 72-dim vectors |
| GRU Hidden | 128 per layer |
| Classification Head | 128 → 64 → 32 → 1 (with LayerNorm + ReLU + Dropout) |
| Dropout Rate | 0.35 (used for both regularization and MC uncertainty) |
| MC Samples | 50 stochastic forward passes at inference |
| Optimizer | AdamW (lr=5e-4, weight_decay=5e-3) |
| Scheduler | 5-epoch linear warmup → ReduceLROnPlateau |
| Loss | BCEWithLogitsLoss + class weighting + label smoothing (0.05) |
| Early Stopping | Patience 30 on validation AUROC |
| Model Size | 704 KB (~150K parameters) |

---

## 3. Dataset Analysis

### 3.1 Raw Data Overview

| Table | Rows | Columns | Size | Key Fields |
|-------|-----:|---------|-----:|------------|
| edstays | 425,087 | 9 | 38 MB | subject_id, stay_id, hadm_id, intime, outtime, gender, race, disposition |
| triage | 425,087 | 11 | 37 MB | temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint |
| vitalsign | 1,564,610 | 11 | 115 MB | charttime, 7 vital signs, rhythm |
| diagnosis | 899,050 | 6 | 50 MB | icd_code, icd_version, icd_title |
| medrecon | 2,987,342 | 9 | 360 MB | charttime, medication name, GSN codes |
| pyxis | 1,586,053 | 7 | 106 MB | charttime, dispensed medication name |
| **TOTAL** | **7,886,229** | — | **706 MB** | — |

### 3.2 Population Demographics

| Metric | Value |
|--------|-------|
| Total ED stays | 425,087 |
| Unique patients | 205,504 |
| Visits per patient (mean / max) | 2.07 / 321 |
| Gender — Female / Male | 54.1% / 45.9% |
| Time span | 2008–2019 |

### 3.3 ED Disposition (Target Variable)

| Disposition | Count | Percent |
|-------------|------:|--------:|
| HOME (discharged) | 241,632 | 56.8% |
| ADMITTED | 158,010 | 37.2% |
| TRANSFER | 7,025 | 1.7% |
| LEFT WITHOUT BEING SEEN | 6,155 | 1.4% |
| ELOPED | 5,710 | 1.3% |
| OTHER | 4,297 | 1.0% |
| LEFT AGAINST MEDICAL ADVICE | 1,881 | 0.4% |
| EXPIRED | 377 | 0.1% |

**Prediction target**: Binary admission via `hadm_id IS NOT NULL`
- Raw admission rate: **47.8%** (203,016 admitted)
- After preprocessing (filtering short stays, missing data): **31.3%** in test set

### 3.4 Length of Stay Distribution

| Range | Count | Percent |
|-------|------:|--------:|
| < 2 hours | 32,406 | 7.6% |
| 2–6 hours | 205,405 | 48.3% |
| 6–12 hours | 135,301 | 31.8% |
| 12–24 hours | 40,049 | 9.4% |
| > 24 hours | 11,926 | 2.8% |

- **Mean**: 7.2 hours
- **Median**: 5.5 hours
- **Std**: 6.6 hours
- Stays < 2 hours are **excluded** during preprocessing (insufficient temporal data)

### 3.5 Triage Vitals at Admission

| Vital | Mean | Std | Median | 1–99th Percentile | Missing % |
|-------|-----:|----:|-------:|-------------------:|----------:|
| Heart Rate (bpm) | 85.1 | 18.0 | 84.0 | [49, 133] | 4.0% |
| Respiratory Rate | 17.6 | 5.5 | 18.0 | [12, 30] | 4.8% |
| SpO2 (%) | 98.5 | 17.0 | 99.0 | [91, 100] | 4.8% |
| Systolic BP (mmHg) | 135.4 | 241.0 | 133.0 | [83, 192] | 4.3% |
| Diastolic BP (mmHg) | 81.3 | 1057.2 | 77.0 | [40, 111] | 4.5% |
| Temperature (°F) | 98.0 | 4.0 | 98.0 | [96, 102] | 5.5% |
| Pain (0–10) | 4.4 | 4.1 | 5.0 | [0, 10] | 6.6% |

> High standard deviations in SBP (241.0) and DBP (1057.2) reflect extreme outliers from data entry errors (e.g., max SBP = 151,103). These are removed by physiological clipping during preprocessing (SBP clipped to [40, 260], DBP to [20, 160]).

### 3.6 Acuity (Emergency Severity Index) Distribution

| ESI Level | Meaning | Count | Percent |
|-----------|---------|------:|--------:|
| 1 — Resuscitation | Immediate life-saving intervention | 24,019 | 5.7% |
| 2 — Emergent | High risk / severe pain | 139,411 | 32.8% |
| 3 — Urgent | Multiple resources needed | 225,066 | 52.9% |
| 4 — Less urgent | Single resource | 28,504 | 6.7% |
| 5 — Non-urgent | No resources needed | 1,100 | 0.3% |
| Missing | — | 6,987 | 1.6% |

> ESI 2–3 dominate (85.7%), consistent with a tertiary care academic hospital.

### 3.7 Vital Sign Time Series (During Stay)

| Vital | Total Measurements | Mean | Std | Missing % | Notes |
|-------|-------------------:|-----:|----:|----------:|-------|
| Heart Rate | 1,494,900 | 81.2 | 18.0 | 4.5% | Reliable, frequently measured |
| Respiratory Rate | 1,475,217 | 17.8 | 78.4 | 5.7% | High std from outliers |
| SpO2 | 1,428,774 | 97.9 | 14.7 | 8.7% | Usually near-normal |
| SBP | 1,483,354 | 128.4 | 22.9 | 5.2% | Critical vital |
| DBP | 1,483,354 | 74.4 | 180.0 | 5.2% | High std from outliers |
| Temperature | 999,642 | 98.0 | 8.2 | **36.1%** | Infrequently re-measured |
| Pain | 1,002,175 | 2.5 | 3.3 | **35.9%** | Subjective, often skipped |

**Measurements per stay**: mean 3.8, median 3, max 109

> Temperature and pain have ~36% missingness — they are less frequently re-measured during stays. SBP and heart rate (designated as "critical features" in the system) are available ~95% of the time, making them reliable signals for deterioration detection.

### 3.8 Cardiac Rhythm Distribution

| Rhythm | Count |
|--------|------:|
| Sinus Rhythm | 18,894 |
| Normal Sinus Rhythm | 12,316 |
| Atrial Fibrillation | 5,238 |
| Sinus Tachycardia | 5,133 |
| Sinus Bradycardia | 3,246 |
| Paced Rhythm | 1,777 |

> Rhythm data is sparse (most vital sign rows have null rhythm). Not used as a model feature.

### 3.9 Diagnosis Patterns

| Metric | Value |
|--------|-------|
| Total diagnoses | 899,050 |
| Unique ICD codes | 13,199 |
| Diagnoses per stay (mean) | 2.1 |
| ICD-10 / ICD-9 split | 50.7% / 49.3% |

**Top 15 Diagnoses**:

| Rank | Diagnosis | Count |
|-----:|-----------|------:|
| 1 | Hypertension NOS | 26,816 |
| 2 | Essential (primary) hypertension | 21,264 |
| 3 | Chest pain, unspecified | 13,016 |
| 4 | Chest pain NOS | 12,398 |
| 5 | Diabetes uncomplicated adult | 12,026 |
| 6 | Unspecified abdominal pain | 10,504 |
| 7 | Type 2 diabetes without complications | 8,801 |
| 8 | Abdominal pain other specified | 8,704 |
| 9 | Fall on same level | 7,817 |
| 10 | Unspecified fall | 7,081 |
| 11 | Dyspnea, unspecified | 7,080 |
| 12 | Urinary tract infection NOS | 6,877 |
| 13 | Alcohol abuse with intoxication | 6,871 |
| 14 | Headache | 6,841 |
| 15 | Hypercholesterolemia | 6,662 |

**Sepsis-related diagnoses** (matching keywords: sepsis, infection, pneumonia, bacteremia, cellulitis):
- 44,773 diagnosis entries (5.0% of all diagnoses)
- 41,757 unique stays (9.8% of all stays)
- Used to derive a binary sepsis flag feature in the context vector

### 3.10 Medication Data

**Medication Reconciliation (medrecon)** — top 10 home medications:

| Medication | Count (sample) |
|------------|------:|
| Aspirin | 3,623 |
| Lisinopril | 1,796 |
| Omeprazole | 1,789 |
| Lorazepam | 1,350 |
| Gabapentin | 1,285 |
| Clonazepam | 1,254 |
| Atorvastatin | 1,236 |
| Metoprolol succinate | 1,133 |
| Levothyroxine | 1,093 |
| Simvastatin | 1,066 |

**Pyxis Dispensing** — top 10 ED-administered medications:

| Medication | Count (sample) |
|------------|------:|
| Ondansetron | 5,472 |
| Lorazepam | 2,929 |
| Acetaminophen | 2,903 |
| Hydromorphone (Dilaudid) | 2,597 |
| Morphine Sulfate | 2,498 |
| Vancomycin | 1,626 |

> Medication counts (unique per stay) are normalized and included in the context vector as a single feature. Top dispensed meds (ondansetron, analgesics) reflect common ED symptom management.

### 3.11 Data Pipeline Transformation

| Stage | Size | Records | Format |
|-------|-----:|---------|--------|
| Raw CSVs | 706 MB | 7.9M rows across 6 tables | CSV |
| PatientRecords | 1.9 GB | ~162K patients, ~425K stays | pickle (dataclass) |
| ContextObjects | 1.8 GB | ~2.66M vectors (1 per window per stay) | pickle (dataclass) |
| Train set | — | 70% of patients | Standardized tensors |
| Validation set | — | 15% of patients | Standardized tensors |
| Test set | — | 15% of patients (321,888 stays) | Standardized tensors |
| Trained model | 704 KB | ~150K parameters | PyTorch checkpoint |

**Splitting strategy**: Patient-level by `subject_id` — all visits from one patient go to the same partition, preventing data leakage.

### 3.12 Model Performance Summary

| Metric | MC-Dropout GRU | Logistic Regression | Delta |
|--------|---------------:|--------------------:|------:|
| AUROC | **0.7786** | 0.6608 | +11.8 pp |
| AUPRC | **0.5928** | 0.5102 | +8.3 pp |
| F1 Score | **0.6027** | 0.5417 | +6.1 pp |
| Brier Score | **0.1935** | 0.2306 | −0.037 (better) |
| ECE | 0.1395 | — | — |
| Test Samples | 321,888 | 532,836 | — |
| Prevalence | 31.3% | 37.0% | — |

**Uncertainty-Stratified AUROC**:

| Confidence | Variance Range | AUROC | Interpretation |
|------------|---------------|------:|----------------|
| High | < 0.01 | 0.779 | Best discrimination when model is confident |
| Medium | 0.01–0.05 | 0.615 | 16.4pp degradation signals genuine uncertainty |
| Low | ≥ 0.05 | — | Too few samples (thresholds may need tuning) |

### 3.13 Key Preprocessing Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Min encounter duration | 2 hours | Need ≥ 2 time windows for trend computation |
| Window size | 60 minutes | Balances granularity with data density |
| Vital clipping | Physiological ranges | Removes data entry errors (e.g., SBP > 150K) |
| Temperature normalization | F → C if > 60 | Mixed units in source data |
| Missingness handling | NaN → 0.0 + explicit mask | Model sees both imputed value and flag |
| Critical features | SBP, Heart Rate | Absence reduces diagnostic confidence |
| Sepsis keywords | sepsis, infection, pneumonia, bacteremia, cellulitis | Binary flag from ICD diagnosis titles |
