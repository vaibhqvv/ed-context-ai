import pickle, json
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm
from src.utils.schema import ContextObject, VITAL_FEATURES
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.context.trend_engine import compute_all_trends, check_deterioration_flag
from src.context.state_classifier import (
    classify_physiological_state,
    classify_data_confidence,
)
from src.context.vector_builder import build_context_vector
from src.preprocessing.missingness import get_missing_critical_features

log = get_logger(__name__)
cfg = get_config()
ALL_FEATURES = VITAL_FEATURES


def build_context_for_patient(record) -> list:
    contexts = []
    for i, window in enumerate(record.time_windows):
        history = record.time_windows[: i + 1]

        trends, slopes = {}, {}
        for feat in ALL_FEATURES:
            slope, label = compute_all_trends(history, feat)
            trends[feat] = label
            slopes[feat] = slope

        missing_critical = get_missing_critical_features(window)
        deterioration = check_deterioration_flag(trends)
        state = classify_physiological_state(trends, deterioration)
        confidence = classify_data_confidence(
            window.observation_density, missing_critical
        )

        ctx_vec = build_context_vector(
            window=window,
            trends=trends,
            slopes=slopes,
            physiological_state=state,
            data_confidence=confidence,
            missing_critical=missing_critical,
            triage_vitals=record.triage_vitals,
            acuity=record.acuity,
            sepsis_flag=record.sepsis_flag,
            n_medications=record.n_medications,
        )

        contexts.append(
            ContextObject(
                patient_id=record.patient_id,
                stay_id=record.stay_id,
                window_index=i,
                physiological_state=state,
                trends=trends,
                slopes=slopes,
                data_confidence=confidence,
                observation_density=window.observation_density,
                missing_critical=missing_critical,
                deterioration_flag=deterioration,
                acuity=record.acuity,
                sepsis_flag=record.sepsis_flag,
                n_medications=record.n_medications,
                context_vector=ctx_vec,
            )
        )
    return contexts


def run_context_construction():
    log.info("═══ MCP Context Construction ═══")
    out = Path(cfg["paths"]["context_data"])
    out.mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["paths"]["processed_data"]) / "patient_records.pkl", "rb") as f:
        records = pickle.load(f)
    all_ctxs = []
    for r in tqdm(records, desc="Building contexts"):
        all_ctxs.extend(build_context_for_patient(r))
    with open(out / "context_objects.pkl", "wb") as f:
        pickle.dump(all_ctxs, f)
    with open(out / "sample_contexts.json", "w") as f:
        json.dump([asdict(c) for c in all_ctxs[:3]], f, indent=2, default=str)
    log.info(f"Built {len(all_ctxs)} context objects from {len(records)} patients")
    log.info(
        f"Context vector dimension: {len(all_ctxs[0].context_vector)} (should be 63)"
    )
    return all_ctxs


if __name__ == "__main__":
    run_context_construction()
