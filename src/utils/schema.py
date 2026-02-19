from dataclasses import dataclass, field
from typing import Dict, List, Optional

VITAL_FEATURES = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain"]
ALL_FEATURES = VITAL_FEATURES


@dataclass
class TimeWindow:
    window_index: int
    window_start_min: float
    window_end_min: float

    features: Dict[str, float]
    missingness_mask: Dict[str, int]
    observation_density: float
    n_observations: int


@dataclass
class PatientRecord:
    """Complete preprocessed record for one patient encounter."""

    patient_id: str
    stay_id: str
    # vitals recorded at triage time (T=0)
    triage_vitals: Dict[str, float]
    # triage metadata
    acuity: float
    chiefcomplaint: str
    sepsis_flag: int
    n_diagnoses: int
    n_medications: int

    time_windows: List[TimeWindow]
    outcome_label: str
    outcome_binary: int
    duration_hours: float


@dataclass
class ContextObject:
    """MCP context packet for one patient at one time window."""

    patient_id: str
    stay_id: str
    window_index: int
    physiological_state: str
    trends: Dict[str, str]
    slopes: Dict[str, float]
    data_confidence: str  # high/medium/low
    observation_density: float
    missing_critical: List[str]
    deterioration_flag: bool
    acuity: float
    sepsis_flag: int
    n_medications: int
    # flattened numeric vector for model input
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
