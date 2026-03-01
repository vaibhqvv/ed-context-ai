import numpy as np
from src.utils.config_loader import get_config
from src.utils.schema import VITAL_FEATURES

cfg = get_config()
ALL_FEATURES = VITAL_FEATURES

TREND_ENC = {
    "rising_fast": 2,
    "rising": 1,
    "stable": 0,
    "falling": -1,
    "falling_fast": -2,
    "unknown": 0,
}
STATE_ENC = {"stable": 0, "concerning": 1, "deteriorating": 2, "critical": 3}
CONF_ENC = {"high": 1.0, "medium": 0.5, "low": 0.0}


def _safe(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


def build_context_vector(
    window,
    trends: dict,
    slopes: dict,
    physiological_state: str,
    data_confidence: str,
    missing_critical: list,
    triage_vitals: dict,  # from PatientRecord.triage_vitals
    acuity: float,  # from PatientRecord.acuity
    sepsis_flag: int,  # from PatientRecord.sepsis_flag
    n_medications: int,  # from PatientRecord.n_medications
) -> list:
    """
    builds the fixed-length numeric context vector for the ML model.

    Vector structure:
    [0  : 28)  — Current window vital aggregates (7 vitals x 4 stats)
    [28 : 35)  — Trend encodings (7 vitals)
    [35 : 42)  — Slope values, clipped (7 vitals)
    [42)       — Physiological state (0-3)
    [43)       — Data confidence (0.0-1.0)
    [44)       — Observation density
    [45 : 52)  — Missingness flags (7 vitals)
    [52 : 59)  — Triage vitals (7 features)
    [59)       — Acuity (1-5, normalized to 0-1)
    [60)       — Sepsis flag (0 or 1)
    [61)       — n_medications (clipped, normalized)
    [62)       — Deterioration flag (missing_critical count > 0)
    [63)       — Shock index (HR / SBP)
    [64)       — Pulse pressure (SBP - DBP)
    [65 : 72)  — Vital deviations from triage (7 vitals)
    Total: 72 dimensions
    """
    vec = []

    for feat in ALL_FEATURES:
        vec.append(_safe(window.features.get(f"{feat}_mean")))
        vec.append(_safe(window.features.get(f"{feat}_last")))
        vec.append(_safe(window.features.get(f"{feat}_min")))
        vec.append(_safe(window.features.get(f"{feat}_max")))

    for feat in ALL_FEATURES:
        vec.append(TREND_ENC.get(trends.get(feat, "unknown"), 0))

    for feat in ALL_FEATURES:
        slope = slopes.get(feat, 0.0)
        vec.append(float(np.clip(np.nan_to_num(slope, nan=0.0), -20, 20)))

    vec.append(STATE_ENC.get(physiological_state, 0))

    vec.append(CONF_ENC.get(data_confidence, 0.5))

    vec.append(_safe(window.observation_density))

    for feat in ALL_FEATURES:
        vec.append(float(window.missingness_mask.get(feat, 0)))

    for feat in ALL_FEATURES:
        vec.append(_safe(triage_vitals.get(feat)))

    vec.append(_safe((5.0 - acuity) / 4.0))

    vec.append(float(sepsis_flag))

    vec.append(float(min(n_medications, 30)) / 30.0)

    vec.append(1.0 if len(missing_critical) > 0 else 0.0)

    # --- New features (indices 63-71) ---

    # Shock index: HR / SBP (clinically important hemodynamic marker)
    hr = _safe(window.features.get("heartrate_mean"))
    sbp = _safe(window.features.get("sbp_mean"))
    shock_index = hr / sbp if sbp > 0 else 0.0
    vec.append(float(np.clip(shock_index, 0, 3)))

    # Pulse pressure: SBP - DBP (narrow = shock, wide = stiff arteries)
    dbp = _safe(window.features.get("dbp_mean"))
    pulse_pressure = (sbp - dbp) / 100.0  # normalized
    vec.append(float(np.clip(pulse_pressure, -1, 2)))

    # Vital deviations from triage (how much has patient changed from baseline)
    for feat in ALL_FEATURES:
        current = _safe(window.features.get(f"{feat}_mean"))
        triage_val = _safe(triage_vitals.get(feat))
        if triage_val != 0 and current != 0:
            deviation = (current - triage_val) / max(abs(triage_val), 1.0)
        else:
            deviation = 0.0
        vec.append(float(np.clip(deviation, -5, 5)))

    return vec
