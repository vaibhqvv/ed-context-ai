import pandas as pd
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _raw(filename: str) -> Path:
    return Path(get_config()["paths"]["raw_data"]) / filename


def load_edstays() -> pd.DataFrame:
    df = pd.read_csv(_raw("edstays.csv"), parse_dates=["intime", "outtime"])
    log.info(f'edstays: {len(df)} rows, {df["stay_id"].nunique()} unique stays')
    return df


def load_triage() -> pd.DataFrame:
    df = pd.read_csv(_raw("triage.csv"))
    log.info(f"triage: {len(df)} rows")
    return df


def load_vitalsign() -> pd.DataFrame:
    df = pd.read_csv(_raw("vitalsign.csv"), parse_dates=["charttime"])
    log.info(f"vitalsign: {len(df)} rows")
    return df


def load_diagnosis() -> pd.DataFrame:
    df = pd.read_csv(_raw("diagnosis.csv"))
    log.info(f"diagnosis: {len(df)} rows")
    return df


def load_medrecon() -> pd.DataFrame:
    df = pd.read_csv(_raw("medrecon.csv"), parse_dates=["charttime"])
    log.info(f"medrecon: {len(df)} rows")
    return df


def load_pyxis() -> pd.DataFrame:
    df = pd.read_csv(_raw("pyxis.csv"), parse_dates=["charttime"])
    log.info(f"pyxis: {len(df)} rows")
    return df


def load_all() -> dict:
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
