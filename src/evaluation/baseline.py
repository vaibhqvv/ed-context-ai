import numpy as np, pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.evaluation.metrics import compute_all_metrics, save_metrics
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


def run_logistic_regression_baseline():
    log.info("Running LR baseline...")
    with open(Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb") as f:
        ctxs = pickle.load(f)
    with open(Path(cfg["paths"]["processed_data"]) / "patient_records.pkl", "rb") as f:
        records = pickle.load(f)
    label_map = {r.patient_id: r.outcome_binary for r in records}
    X, y = [], []
    for ctx in ctxs:
        label = label_map.get(ctx.patient_id)
        if label is None:
            continue
        if any(np.isnan(v) for v in ctx.context_vector):
            continue
        X.append(ctx.context_vector)
        y.append(float(label))
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X[:split], y[:split])
    y_prob = pipe.predict_proba(X[split:])[:, 1]
    m = compute_all_metrics(y[split:], y_prob)
    save_metrics(m, "baseline_lr")
    log.info(f'LR Baseline AUROC: {m["auroc"]:.3f}')
    return m
