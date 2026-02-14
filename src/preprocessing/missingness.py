from typing import Dict, List
from src.utils.config_loader import get_config
from src.utils.schema import TimeWindow

cfg = get_config()
CRITICAL = cfg["context"]["critical_missing_features"]
# = ['sbp', 'heartrate'] — the two most critical vitals


def get_missing_critical_features(window: TimeWindow) -> List[str]:
    return [f for f in CRITICAL if window.missingness_mask.get(f, 0) == 0]


def compute_consecutive_missingness(windows: List[TimeWindow], feature: str) -> int:
    count = 0
    for w in reversed(windows):
        if w.missingness_mask.get(feature, 0) == 0:
            count += 1
        else:
            break
    return count


def check_triage_completeness(triage_vitals: Dict[str, float]) -> float:
    from src.utils.schema import VITAL_FEATURES
    import math

    observed = sum(
        1
        for f in VITAL_FEATURES
        if f in triage_vitals and not math.isnan(triage_vitals.get(f, float("nan")))
    )
    return observed / len(VITAL_FEATURES) if VITAL_FEATURES else 0.0


def summarize_patient_missingness(windows: List[TimeWindow]) -> Dict[str, float]:
    if not windows:
        return {}
    all_feats = list(windows[0].missingness_mask.keys())
    return {
        f: sum(w.missingness_mask.get(f, 0) for w in windows) / len(windows)
        for f in all_feats
    }
