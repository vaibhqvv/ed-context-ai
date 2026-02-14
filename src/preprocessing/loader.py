import pandas as pd
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _raw(filename: str) -> Path:
    return Path(get_config()["paths"]["raw_data"]) / filename


def load_edstays() -> pd.DataFrame:
    """
    Columns: subject_id, hadm_id, stay_id, intime, outtime
    hadm_id is NaN for patients discharged home.
    hadm_id is populated for patients admitted to hospital.
    """
    df = pd.read_csv(_raw("edstays.csv"), parse_dates=["intime", "outtime"])
    log.info(f'edstays: {len(df)} rows, {df["stay_id"].nunique()} unique stays')
    return df


# 2. triage — one row per encounter, vitals at T=0
def load_triage() -> pd.DataFrame:
    """
    Columns: subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint
    Note: no charttime column — these are recorded at triage (T=0).
    """
    df = pd.read_csv(_raw("triage.csv"))
    log.info(f"triage: {len(df)} rows")
    return df


# 3. vitalsign — repeated vitals during stay
def load_vitalsign() -> pd.DataFrame:
    """
    Columns: subject_id, stay_id, charttime, temperature, heartrate, resprate, o2sat, sbp, dbp, rhythm, pain
    Multiple rows per stay_id, each with a charttime timestamp.
    """
    df = pd.read_csv(_raw("vitalsign.csv"), parse_dates=["charttime"])
    log.info(f"vitalsign: {len(df)} rows")
    return df


# 4. diagnosis — ICD codes (post-discharge)
def load_diagnosis() -> pd.DataFrame:
    """
    Columns: subject_id, stay_id, seq_num, icd_code, icd_version, icd_title
    Up to 9 codes per stay. seq_num=1 is most relevant.
    Used to derive sepsis_flag and comorbidity signals.
    """
    df = pd.read_csv(_raw("diagnosis.csv"))
    log.info(f"diagnosis: {len(df)} rows")
    return df


# 5. medrecon — pre-admission medications
def load_medrecon() -> pd.DataFrame:
    """
    Columns: subject_id, stay_id, charttime, name, gsn, ndc, etc_rn, etccode, etcdescription
    Multiple rows per medication due to drug class ontology.
    Use etc_rn == 1 to get unique medication count.
    """
    df = pd.read_csv(_raw("medrecon.csv"), parse_dates=["charttime"])
    log.info(f"medrecon: {len(df)} rows")
    return df


# 6. pyxis — ED medication dispensing (optional)
def load_pyxis() -> pd.DataFrame:
    """
    Columns: subject_id, stay_id, charttime, med_rn, name, gsn_rn, gsn
    Medications dispensed during the ED stay. Useful as a proxy for intervention intensity (e.g., IV antibiotics dispensed = higher acuity).
    """
    df = pd.read_csv(_raw("pyxis.csv"), parse_dates=["charttime"])
    log.info(f"pyxis: {len(df)} rows")
    return df


# Load all tables at once
def load_all() -> dict:
    """Load all 6 tables and return as a dict."""
    return {
        "edstays": load_edstays(),
        "triage": load_triage(),
        "vitalsign": load_vitalsign(),
        "diagnosis": load_diagnosis(),
        "medrecon": load_medrecon(),
        "pyxis": load_pyxis(),
    }


if __name__ == "__main__":
    tables = load_all()
    for name, df in tables.items():
        print(f"{name}: {df.shape}  columns: {list(df.columns)}")
