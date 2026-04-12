import json, numpy as np, pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.evaluation.metrics import compute_all_metrics, save_metrics
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


def _load_stay_splits():
    """Load train/val/test stay_id sets from splits.json."""
    splits_path = Path(cfg["paths"].get("splits", "data/splits")) / "splits.json"
    if not splits_path.exists():
        return None
    with open(splits_path) as f:
        splits = json.load(f)
    return {
        "train": set(splits.get("train", [])),
        "val": set(splits.get("val", [])),
        "test": set(splits.get("test", [])),
    }


def _group_last_window_per_stay(ctxs, label_map):
    """Group context objects by (patient_id, stay_id), keep last window per stay."""
    best = {}
    for ctx in tqdm(ctxs, desc="Grouping last window per stay", unit="ctx"):
        label = label_map.get(ctx.patient_id)
        if label is None:
            continue
        if any(np.isnan(v) for v in ctx.context_vector):
            continue
        key = (ctx.patient_id, ctx.stay_id)
        if key not in best or ctx.window_index > best[key][0]:
            best[key] = (ctx.window_index, ctx.context_vector, float(label))
    return best


def _prepare_baseline_data():
    """Load data and split into train/test using the same splits as GRU.

    Returns (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    log.info("Loading context objects for baselines...")
    with open(Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb") as f:
        ctxs = pickle.load(f)

    log.info("Loading patient records...")
    with open(Path(cfg["paths"]["processed_data"]) / "patient_records.pkl", "rb") as f:
        records = pickle.load(f)

    label_map = {r.patient_id: r.outcome_binary for r in records}
    del records

    best = _group_last_window_per_stay(ctxs, label_map)
    del ctxs
    log.info(f"Total stays with valid data: {len(best)}")

    splits = _load_stay_splits()
    if splits is not None:
        log.info("Using saved splits from splits.json (patient-level)")
        train_stay_ids = splits["train"]
        test_stay_ids = splits["test"]

        X_train, y_train = [], []
        X_test, y_test = [], []
        for (pid, sid), (_widx, vec, label) in best.items():
            if sid in train_stay_ids:
                X_train.append(vec)
                y_train.append(label)
            elif sid in test_stay_ids:
                X_test.append(vec)
                y_test.append(label)
    else:
        log.info("No splits.json — creating patient-level 70/15/15 split")
        all_pids = list(set(pid for (pid, _) in best.keys()))
        np.random.seed(42)
        np.random.shuffle(all_pids)
        n = len(all_pids)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_pids = set(all_pids[:n_train])
        test_pids = set(all_pids[n_train + n_val:])

        X_train, y_train = [], []
        X_test, y_test = [], []
        for (pid, sid), (_widx, vec, label) in best.items():
            if pid in train_pids:
                X_train.append(vec)
                y_train.append(label)
            elif pid in test_pids:
                X_test.append(vec)
                y_test.append(label)

    del best
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    log.info(f"Baseline train: {len(X_train)}, test: {len(X_test)}")
    log.info(f"Baseline test prevalence: {y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test


def _run_single_baseline(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a single baseline model."""
    log.info(f"Fitting {name}...")
    model.fit(X_train, y_train)

    log.info(f"Predicting {name} on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    m = compute_all_metrics(y_test, y_prob)
    save_metrics(m, f"baseline_{name}")
    log.info(f'{name} AUROC: {m["auroc"]:.3f} | AUPRC: {m["auprc"]:.3f} | '
             f'F1: {m["f1"]:.3f} | Brier: {m["brier_score"]:.3f}')
    return m


def run_logistic_regression_baseline(X_train=None, y_train=None,
                                     X_test=None, y_test=None):
    """Logistic Regression baseline with StandardScaler."""
    if X_train is None:
        X_train, y_train, X_test, y_test = _prepare_baseline_data()

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return _run_single_baseline("lr", pipe, X_train, y_train, X_test, y_test)


def run_random_forest_baseline(X_train, y_train, X_test, y_test):
    """Random Forest baseline."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return _run_single_baseline("rf", rf, X_train, y_train, X_test, y_test)


def run_xgboost_baseline(X_train, y_train, X_test, y_test):
    """XGBoost baseline."""
    from xgboost import XGBClassifier

    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return _run_single_baseline("xgb", xgb, X_train, y_train, X_test, y_test)


def run_all_baselines():
    """Run all baselines on the same data split. Returns dict of metrics."""
    X_train, y_train, X_test, y_test = _prepare_baseline_data()

    # Standardize for LR (RF and XGB don't need it but it doesn't hurt)
    results = {}
    results["lr"] = run_logistic_regression_baseline(
        X_train, y_train, X_test, y_test
    )
    results["rf"] = run_random_forest_baseline(
        X_train, y_train, X_test, y_test
    )
    results["xgb"] = run_xgboost_baseline(
        X_train, y_train, X_test, y_test
    )
    return results
