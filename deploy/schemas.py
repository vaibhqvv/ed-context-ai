"""Pydantic request/response schemas for the ED-Context-AI inference API."""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class VectorInferenceRequest(BaseModel):
    """Raw 72-d context vector(s). Use this when the client already has the
    engineered context vector(s) and only needs risk + uncertainty.

    - Provide `vector` for a single window (treated as seq_len=1 for GRU).
    - Provide `sequence` for multiple consecutive windows of one stay.
    """

    vector: Optional[List[float]] = Field(default=None, description="Single 72-d context vector")
    sequence: Optional[List[List[float]]] = Field(
        default=None, description="List of 72-d vectors for one stay, oldest to newest"
    )


class RiskResponse(BaseModel):
    risk_probability: float
    uncertainty_score: float
    confidence_level: str


class ContextObjectDTO(BaseModel):
    """Full ContextObject payload (mirrors src.utils.schema.ContextObject).
    Needed when requesting the full decision + explanation pipeline.
    """

    patient_id: str
    stay_id: str
    window_index: int
    physiological_state: str
    trends: Dict[str, str]
    slopes: Dict[str, float]
    data_confidence: str
    observation_density: float
    missing_critical: List[str]
    deterioration_flag: bool
    acuity: float
    sepsis_flag: int
    n_medications: int
    context_vector: List[float]


class DecisionResponse(BaseModel):
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


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    device: str
    model_type: str
    input_dim: int
