from src.utils.config_loader import get_config

cfg = get_config()
D_HIGH = cfg["context"]["density_high_threshold"]
D_LOW = cfg["context"]["density_low_threshold"]
FAST_TRENDS = {"rising_fast", "falling_fast"}
MILD_TRENDS = {"rising", "falling"}
CRITICAL_VITALS = {"heart_rate", "systolic_bp", "respiratory_rate"}


def classify_physiological_state(trends, deterioration_flag) -> str:
    if deterioration_flag:
        fast_critical = any(
            trends.get(v, "stable") in FAST_TRENDS for v in CRITICAL_VITALS
        )
        return "critical" if fast_critical else "deteriorating"
    n_fast = sum(1 for t in trends.values() if t in FAST_TRENDS)
    n_mild = sum(1 for t in trends.values() if t in MILD_TRENDS)
    if n_fast >= 2:
        return "deteriorating"
    if n_fast == 1 or n_mild >= 2:
        return "concerning"
    return "stable"


def classify_data_confidence(observation_density, missing_critical) -> str:
    if missing_critical:
        return "low" if observation_density < D_HIGH else "medium"
    if observation_density >= D_HIGH:
        return "high"
    if observation_density >= D_LOW:
        return "medium"
    return "low"
