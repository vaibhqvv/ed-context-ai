import numpy as np
from typing import List, Dict, Tuple
from src.utils.config_loader import get_config
from src.utils.schema import TimeWindow

cfg = get_config()

VITAL_FEATURES = cfg["preprocessing"]["vital_features"]
# = ['heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'temperature', 'pain']


def build_windows_for_patient(
    stay_id: str, vitals_df, window_size_min: int = 60
) -> List[TimeWindow]:
    """
    Build sequential time windows for one patient encounter.

    Each window covers window_size_min minutes.
    Windows are non-overlapping and chronological.
    Returns a list of TimeWindow objects.
    """
    if len(vitals_df) == 0:
        return []

    vitals_df = vitals_df.sort_values("t_minutes").copy()
    max_t = vitals_df["t_minutes"].max()

    windows = []
    window_idx = 0

    for end_min in range(
        window_size_min, int(max_t) + window_size_min, window_size_min
    ):
        start_min = end_min - window_size_min

        window_rows = vitals_df[
            (vitals_df["t_minutes"] >= start_min) & (vitals_df["t_minutes"] < end_min)
        ]

        features, mask = aggregate_vital_window(window_rows)
        density = compute_density(mask)
        n_obs = len(window_rows)

        windows.append(
            TimeWindow(
                window_index=window_idx,
                window_start_min=float(start_min),
                window_end_min=float(end_min),
                features=features,
                missingness_mask=mask,
                observation_density=density,
                n_observations=n_obs,
            )
        )
        window_idx += 1

    return windows


def aggregate_vital_window(window_rows) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    For each vital feature in this window,:
      {feature}_mean  — mean of all observations
      {feature}_last  — most recent observation (last charttime row)
      {feature}_min   — minimum
      {feature}_max   — maximum

    Missingness mask: 1 if at least one non-NaN value exists, 0 otherwise.
    """
    features = {}
    mask = {}

    for feat in VITAL_FEATURES:
        if feat in window_rows.columns:
            col_data = window_rows[feat].dropna()
        else:
            col_data = []

        if len(col_data) > 0:
            features[f"{feat}_mean"] = float(col_data.mean())
            features[f"{feat}_last"] = float(col_data.iloc[-1])
            features[f"{feat}_min"] = float(col_data.min())
            features[f"{feat}_max"] = float(col_data.max())
            mask[feat] = 1
        else:
            features[f"{feat}_mean"] = np.nan
            features[f"{feat}_last"] = np.nan
            features[f"{feat}_min"] = np.nan
            features[f"{feat}_max"] = np.nan
            mask[feat] = 0

    return features, mask


def compute_density(mask: Dict[str, int]) -> float:
    """Fraction of vital features observed (not missing) in this window."""
    if not mask:
        return 0.0
    return float(sum(mask.values())) / len(mask)


def build_rolling_window_features(
    vitals_df, current_t_min: float, horizon_hours: int
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Build features for a ROLLING window ending at current_t_min,
    looking back horizon_hours into the past.
    Used by the context layer to build multi-scale trend features.
    """
    start = current_t_min - (horizon_hours * 60)
    window_rows = vitals_df[
        (vitals_df["t_minutes"] >= start) & (vitals_df["t_minutes"] <= current_t_min)
    ]
    return aggregate_vital_window(window_rows)
