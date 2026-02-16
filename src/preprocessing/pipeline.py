import pickle, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm

from src.preprocessing.loader import load_all
from src.preprocessing.cleaner import (
    filter_encounters,
    derive_outcome_label,
    clean_vitalsign,
    clean_triage,
    derive_sepsis_flag,
    derive_medication_count,
)
from src.preprocessing.windower import build_windows_for_patient
from src.preprocessing.missingness import get_missing_critical_features
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.schema import PatientRecord, VITAL_FEATURES

log = get_logger(__name__)
cfg = get_config()


def extract_triage_vitals(triage_row) -> dict:
    vitals = {}
    for feat in VITAL_FEATURES:
        val = triage_row.get(feat, np.nan)
        try:
            vitals[feat] = float(val) if not pd.isna(val) else float("nan")
        except (TypeError, ValueError):
            vitals[feat] = float("nan")
    return vitals


def run_preprocessing() -> list:
    log.info("═══ Starting Preprocessing Pipeline (MIMIC-IV-ED v2.2) ═══")
    out_path = Path(cfg["paths"]["processed_data"])
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Step 1: Loading all tables...")
    tables = load_all()
    edstays = tables["edstays"]
    triage = tables["triage"]
    vitalsign = tables["vitalsign"]
    diagnosis = tables["diagnosis"]
    medrecon = tables["medrecon"]

    log.info("Step 2: Filtering encounters...")
    edstays = filter_encounters(edstays)
    edstays = derive_outcome_label(edstays)
    valid_stay_ids = set(edstays["stay_id"].unique())
    log.info(f"Valid stays after filtering: {len(valid_stay_ids)}")

    log.info("Step 3: Cleaning vitalsign table...")
    vitals_clean = clean_vitalsign(vitalsign, edstays)

    log.info("Step 4: Cleaning triage table...")
    triage_clean = clean_triage(triage, valid_stay_ids)
    triage_idx = triage_clean.set_index("stay_id")

    log.info("Step 5: Deriving sepsis flags from diagnosis...")
    sepsis_df = derive_sepsis_flag(diagnosis, valid_stay_ids)
    sepsis_idx = sepsis_df.set_index("stay_id")

    log.info("Step 6: Counting pre-admission medications...")
    med_df = derive_medication_count(medrecon, valid_stay_ids)
    med_idx = med_df.set_index("stay_id")

    log.info("Step 7: Building patient records...")
    records = []
    n_failed = 0
    n_no_vitals = 0

    for _, stay_row in tqdm(
        edstays.iterrows(), total=len(edstays), desc="Processing stays"
    ):
        stay_id = stay_row["stay_id"]

        try:
            stay_vitals = vitals_clean[vitals_clean["stay_id"] == stay_id].copy()

            if len(stay_vitals) < 2:
                n_no_vitals += 1
                continue

            # build time windows
            windows = build_windows_for_patient(str(stay_id), stay_vitals)
            if not windows:
                n_no_vitals += 1
                continue

            # get triage info
            if stay_id in triage_idx.index:
                triage_row = triage_idx.loc[stay_id]
                triage_dict = (
                    triage_row.to_dict()
                    if hasattr(triage_row, "to_dict")
                    else dict(triage_row)
                )
                triage_vitals = extract_triage_vitals(triage_dict)
                acuity = float(triage_dict.get("acuity", 3.0))
                chiefcomplaint = str(triage_dict.get("chiefcomplaint", "")).strip()
            else:
                triage_vitals = {f: float("nan") for f in VITAL_FEATURES}
                acuity = 3.0
                chiefcomplaint = ""

            if stay_id in sepsis_idx.index:
                sepsis_flag = int(sepsis_idx.loc[stay_id, "sepsis_flag"])
                n_diag = int(sepsis_idx.loc[stay_id, "n_diagnoses"])
            else:
                sepsis_flag = 0
                n_diag = 0

            if stay_id in med_idx.index:
                n_meds = int(med_idx.loc[stay_id, "n_medications"])
            else:
                n_meds = 0

            record = PatientRecord(
                patient_id=str(stay_row["subject_id"]),
                stay_id=str(stay_id),
                triage_vitals=triage_vitals,
                acuity=acuity,
                chiefcomplaint=chiefcomplaint,
                sepsis_flag=sepsis_flag,
                n_diagnoses=n_diag,
                n_medications=n_meds,
                time_windows=windows,
                outcome_label=stay_row["outcome_label"],
                outcome_binary=int(stay_row["outcome_binary"]),
                duration_hours=float(stay_row["duration_hours"]),
            )
            records.append(record)

        except Exception as e:
            log.warning(f"Failed stay_id={stay_id}: {type(e).__name__}: {e}")
            n_failed += 1

    log.info(
        f"Done: {len(records)} records | {n_no_vitals} skipped (no vitals) | {n_failed} errors"
    )

    # Statistics
    n_admitted = sum(r.outcome_binary for r in records)
    n_discharged = len(records) - n_admitted
    avg_windows = sum(len(r.time_windows) for r in records) / max(len(records), 1)
    log.info(
        f"Admitted: {n_admitted} | Discharged: {n_discharged} | Avg windows/patient: {avg_windows:.1f}"
    )

    pkl_path = out_path / "patient_records.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    log.info(f"Saved: {pkl_path}")

    sample_path = out_path / "sample_records.json"
    with open(sample_path, "w") as f:
        json.dump([asdict(r) for r in records[:5]], f, indent=2, default=str)
    log.info(f"Sample saved: {sample_path}")

    _save_splits(records)

    return records


def _save_splits(records: list, train=0.70, val=0.15, test=0.15):
    """
    Split by unique patient (subject_id) — NOT by stay — to prevent
    data leakage (same patient appearing in train and test).
    """
    import random

    split_path = Path(cfg["paths"]["splits"])
    split_path.mkdir(parents=True, exist_ok=True)
    unique_patients = list(set(r.patient_id for r in records))
    random.seed(42)
    random.shuffle(unique_patients)

    n = len(unique_patients)
    n_tr = int(n * train)
    n_val = int(n * val)

    train_pids = set(unique_patients[:n_tr])
    val_pids = set(unique_patients[n_tr : n_tr + n_val])
    test_pids = set(unique_patients[n_tr + n_val :])

    splits = {
        "train": [r.stay_id for r in records if r.patient_id in train_pids],
        "val": [r.stay_id for r in records if r.patient_id in val_pids],
        "test": [r.stay_id for r in records if r.patient_id in test_pids],
    }

    with open(split_path / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    log.info(
        f'Splits: train={len(splits["train"])} | val={len(splits["val"])} | test={len(splits["test"])}'
    )
    log.info("Split is by PATIENT (subject_id) to prevent data leakage")


if __name__ == "__main__":
    import sys

    records = run_preprocessing()
    if records:
        r = records[0]
        print(
            f"Sample record: stay_id={r.stay_id}, windows={len(r.time_windows)}, outcome={r.outcome_label}"
        )
        print(f"First window density: {r.time_windows[0].observation_density:.2f}")
