import pandas as pd
import numpy as np
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def filter_encounters(edstays: pd.DataFrame) -> pd.DataFrame:

    cfg = get_config()
    edstays = edstays.copy()
    edstays["duration_hours"] = (
        edstays["outtime"] - edstays["intime"]
    ).dt.total_seconds() / 3600
    min_h = cfg["preprocessing"]["min_encounter_hours"]
    before = len(edstays)
    edstays = edstays[edstays["duration_hours"] >= min_h].copy()
    log.info(f"Encounter filter: {before} -> {len(edstays)} stays (>= {min_h}h)")
    return edstays


def derive_outcome_label(edstays: pd.DataFrame) -> pd.DataFrame:
    """
    MIMIC-IV-ED outcome strategy:
    - hadm_id NOT null -> patient admitted to hospital -> outcome_binary = 1
    - hadm_id IS null  -> patient discharged home     -> outcome_binary = 0

    This is the most reliable and commonly used labeling approach for MIMIC-IV-ED.
    """
    edstays = edstays.copy()
    edstays["outcome_binary"] = edstays["hadm_id"].notna().astype(int)
    edstays["outcome_label"] = edstays["outcome_binary"].map(
        {0: "discharge", 1: "admitted"}
    )
    n_admitted = edstays["outcome_binary"].sum()
    n_discharged = len(edstays) - n_admitted
    log.info(f"Outcomes: {n_admitted} admitted, {n_discharged} discharged")
    log.info(f"Admission rate: {n_admitted/len(edstays):.1%}")
    return edstays


VITAL_COLS = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain"]


def clean_vitalsign(vitals: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:

    cfg = get_config()
    ranges = cfg["preprocessing"]["vital_clipping_ranges"]

    valid_stays = set(edstays["stay_id"].unique())
    vitals = vitals[vitals["stay_id"].isin(valid_stays)].copy()
    log.info(f"Vitals after stay filter: {len(vitals)} rows")

    for col in VITAL_COLS:
        if col in vitals.columns and col in ranges:
            lo, hi = ranges[col]
            out_of_range = ((vitals[col] < lo) | (vitals[col] > hi)) & vitals[
                col
            ].notna()
            if out_of_range.sum() > 0:
                log.info(f"Clipping {out_of_range.sum()} out-of-range {col} values")
            vitals[col] = vitals[col].clip(lo, hi)

    vitals = vitals.sort_values(["stay_id", "charttime"])
    vitals = vitals.drop_duplicates(subset=["stay_id", "charttime"], keep="last")

    ref_times = edstays[["stay_id", "intime"]].drop_duplicates("stay_id")
    vitals = vitals.merge(ref_times, on="stay_id", how="left")
    vitals["t_minutes"] = (
        vitals["charttime"] - vitals["intime"]
    ).dt.total_seconds() / 60

    vitals = vitals[
        (vitals["t_minutes"] >= 0) & (vitals["t_minutes"] <= 1440)  # Max 24h
    ].copy()
    log.info(f"Vitals after time filter: {len(vitals)} rows")
    return vitals


def clean_triage(triage: pd.DataFrame, valid_stay_ids: set) -> pd.DataFrame:
    """
    Fill missing acuity with 3 (middle severity — conservative default).
    Fill missing chiefcomplaint with empty string.
    """
    triage = triage[triage["stay_id"].isin(valid_stay_ids)].copy()

    triage["acuity"] = triage["acuity"].clip(1, 5)

    triage["acuity"] = triage["acuity"].fillna(3.0)

    triage["chiefcomplaint"] = (
        triage["chiefcomplaint"].fillna("").str.strip().str.lower()
    )

    cfg = get_config()
    ranges = cfg["preprocessing"]["vital_clipping_ranges"]
    for col in VITAL_COLS:
        if col in triage.columns and col in ranges:
            lo, hi = ranges[col]
            triage[col] = triage[col].clip(lo, hi)

    log.info(f"Triage cleaned: {len(triage)} rows")
    return triage


def derive_sepsis_flag(diagnosis: pd.DataFrame, valid_stay_ids: set) -> pd.DataFrame:

    cfg = get_config()
    keywords = cfg["preprocessing"]["sepsis_icd_keywords"]

    diag = diagnosis[diagnosis["stay_id"].isin(valid_stay_ids)].copy()
    diag["icd_title_lower"] = diag["icd_title"].str.lower().fillna("")

    pattern = "|".join(keywords)
    diag["is_sepsis_related"] = diag["icd_title_lower"].str.contains(pattern, na=False)

    agg = (
        diag.groupby("stay_id")
        .agg(
            sepsis_flag=("is_sepsis_related", "max"), n_diagnoses=("icd_code", "count")
        )
        .reset_index()
    )
    agg["sepsis_flag"] = agg["sepsis_flag"].astype(int)
    log.info(f'Sepsis flag: {agg["sepsis_flag"].sum()} stays flagged of {len(agg)}')
    return agg


def derive_medication_count(
    medrecon: pd.DataFrame, valid_stay_ids: set
) -> pd.DataFrame:

    med = medrecon[medrecon["stay_id"].isin(valid_stay_ids)].copy()

    unique_meds = med[med["etc_rn"] == 1]
    counts = unique_meds.groupby("stay_id").size().reset_index(name="n_medications")
    log.info(f"Medication counts: {len(counts)} stays with med data")
    return counts
