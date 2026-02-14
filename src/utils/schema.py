from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The 7 vital features — exact column names from vitalsign.csv
VITAL_FEATURES = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain"]
# All features tracked in windowing and context (vitals only — no labs in MIMIC-IV-ED)
ALL_FEATURES = VITAL_FEATURES


@dataclass
class TimeWindow:
    """One 60-minute observation window for a single patient."""

    window_index: int
    window_start_min: float
    window_end_min: float
    # Keys: {feat}_mean, {feat}_last, {feat}_min, {feat}_max for each vital
    features: Dict[str, float]
    # 1 = observed in this window, 0 = missing
    missingness_mask: Dict[str, int]
    # Fraction of expected vitals that were recorded
    observation_density: float
    # Number of raw vital sign rows in this window
    n_observations: int


@dataclass
class PatientRecord:
    """Complete preprocessed record for one patient encounter."""

    patient_id: str  # subject_id from MIMIC
    stay_id: str  # stay_id from edstays
    # Vitals recorded at triage time (T=0)
    triage_vitals: Dict[str, float]
    # Triage metadata
    acuity: float  # 1 (most severe) to 5 (least severe)
    chiefcomplaint: str
    # Derived from diagnosis table (ICD codes)
    sepsis_flag: int  # 1 if any infection/sepsis ICD code present
    n_diagnoses: int  # Total number of ICD codes
    # Derived from medrecon table
    n_medications: int  # Number of pre-admission medications
    # Sequential time windows from vitalsign table
    time_windows: List[TimeWindow]
    # Outcome: 0 = discharged home, 1 = admitted to hospital
    outcome_label: str  # 'discharge' or 'admitted'
    outcome_binary: int  # 0 or 1
    # Total encounter duration in hours
    duration_hours: float


@dataclass
class ContextObject:
    """MCP context packet for one patient at one time window."""

    patient_id: str
    stay_id: str
    window_index: int
    physiological_state: str  # stable/concerning/deteriorating/critical
    trends: Dict[str, str]  # feature -> trend label
    slopes: Dict[str, float]  # feature -> slope (units/hour)
    data_confidence: str  # high/medium/low
    observation_density: float
    missing_critical: List[str]  # names of missing critical features
    deterioration_flag: bool
    # Clinical context from other tables
    acuity: float
    sepsis_flag: int
    n_medications: int
    # Flattened numeric vector for model input
    context_vector: List[float]


@dataclass
class SystemOutput:
    """Final pipeline output for one patient window."""

    patient_id: str
    stay_id: str
    window_index: int
    risk_probability: float
    uncertainty_score: float
    confidence_level: str
    decision_category: str
    urgency: str
    recommended_tests: List[str]
    explanation: str
    context_summary: Dict
