import numpy as np
from typing import List, Dict, Tuple
from src.utils.config_loader import get_config

cfg = get_config()
STABLE = cfg["context"]["slope_stable_threshold"]
FAST = cfg["context"]["slope_fast_threshold"]


def compute_slope(times: List[float], values: List[float]) -> float:
    clean = [(t, v) for t, v in zip(times, values) if not np.isnan(v)]
    if len(clean) < 2:
        return np.nan
    t = np.array([c[0] for c in clean]) / 60.0  #
    v = np.array([c[1] for c in clean])
    denom = np.sum((t - t.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((t - t.mean()) * (v - v.mean())) / denom)


def classify_slope(slope: float) -> str:
    if np.isnan(slope):
        return "unknown"
    if slope > FAST:
        return "rising_fast"
    if slope > STABLE:
        return "rising"
    if slope < -FAST:
        return "falling_fast"
    if slope < -STABLE:
        return "falling"
    return "stable"


def compute_all_trends(window_history, feature) -> Tuple[float, str]:
    times = [w.window_end_min for w in window_history]
    values = [w.features.get(f"{feature}_last", np.nan) for w in window_history]
    slope = compute_slope(times, values)
    return slope, classify_slope(slope)


def check_deterioration_flag(trends: Dict[str, str]) -> bool:
    hr_bad = trends.get("heart_rate", "stable") in ("rising", "rising_fast")
    bp_bad = trends.get("systolic_bp", "stable") in ("falling", "falling_fast")
    return hr_bad and bp_bad
