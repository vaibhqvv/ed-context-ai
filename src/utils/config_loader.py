import yaml
from pathlib import Path

_CONFIG = None


def get_config(config_path=None):
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if config_path is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
    with open(config_path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


if __name__ == "__main__":
    cfg = get_config()
    print("Config OK:", cfg["model"]["dropout_rate"])
