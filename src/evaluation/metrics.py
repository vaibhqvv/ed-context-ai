import numpy as np, json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


def compute_all_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }
    log.info(f'AUROC: {m["auroc"]:.3f} | AUPRC: {m["auprc"]:.3f} | F1: {m["f1"]:.3f}')
    return m


def compute_ece(y_true, y_prob, n_bins=10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def uncertainty_stratified_auroc(y_true, y_prob, uncertainties) -> dict:
    t = cfg["model"]["uncertainty_thresholds"]
    results = {}
    for label, (lo, hi) in [
        ("High", (0, t["medium"])),
        ("Medium", (t["medium"], t["high"])),
        ("Low", (t["high"], 1e9)),
    ]:
        mask = (uncertainties >= lo) & (uncertainties < hi)
        if mask.sum() < 10:
            results[label] = None
            continue
        try:
            results[label] = float(roc_auc_score(y_true[mask], y_prob[mask]))
        except:
            results[label] = None
    return results


def save_metrics(metrics, name):
    out = Path(cfg["paths"]["results"]) / f"{name}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
