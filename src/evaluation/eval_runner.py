import gc, torch, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm
from src.model.inference import load_model, get_model_type, predict_sequence
from src.model.dataset import EDContextDataset, load_label_map
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_ece,
    uncertainty_stratified_auroc,
    save_metrics,
)
from src.evaluation.plots import plot_roc, plot_calibration, plot_uncertainty_hist
from src.evaluation.baseline import run_logistic_regression_baseline
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.device import get_device

log = get_logger(__name__)
cfg = get_config()


def _load_contexts():
    """Load context objects from pickle with a progress bar."""
    pkl_path = Path(cfg["paths"]["context_data"]) / "context_objects.pkl"
    file_size = pkl_path.stat().st_size
    log.info(f"Loading context objects ({file_size / 1e6:.0f} MB)...")

    with open(pkl_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Loading contexts") as pbar:

            class _TrackedReader:
                def __init__(self, fh, pbar):
                    self._fh, self._pbar = fh, pbar

                def read(self, n=-1):
                    data = self._fh.read(n)
                    self._pbar.update(len(data))
                    return data

                def readline(self):
                    line = self._fh.readline()
                    self._pbar.update(len(line))
                    return line

            ctxs = pickle.load(_TrackedReader(f, pbar))
    return ctxs


def _eval_mlp(ctxs, label_map, model):
    """Per-context evaluation for MLP with batched GPU inference."""
    device = next(model.parameters()).device
    log.info("Filtering valid contexts for MLP evaluation...")

    # Collect valid contexts
    valid_x, valid_labels = [], []
    for ctx in tqdm(ctxs, desc="Filtering contexts", unit="ctx"):
        label = label_map.get(ctx.patient_id)
        if label is None:
            continue
        if any(np.isnan(v) for v in ctx.context_vector):
            continue
        valid_x.append(np.array(ctx.context_vector, dtype=np.float32))
        valid_labels.append(float(label))

    log.info(f"Valid contexts: {len(valid_x)}")
    y_true = np.array(valid_labels)

    # Batch inference on GPU for speed
    batch_size = 512
    y_prob_list, y_unc_list = [], []
    n_batches = (len(valid_x) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="MC Dropout inference", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, len(valid_x))
        x_batch = torch.tensor(np.stack(valid_x[start:end]), dtype=torch.float32).to(device)
        mean, var, _ = model.predict_with_uncertainty(x_batch)
        y_prob_list.append(mean.cpu().numpy())
        y_unc_list.append(var.cpu().numpy())

    y_prob = np.concatenate(y_prob_list)
    y_unc = np.concatenate(y_unc_list)
    return y_true, y_prob, y_unc


def _eval_gru(ctxs, label_map):
    """Sequence-level evaluation for GRU — one prediction per stay."""
    log.info("Grouping contexts by (patient_id, stay_id)...")
    stays = {}
    labels_map = {}
    for ctx in tqdm(ctxs, desc="Grouping contexts", unit="ctx"):
        label = label_map.get(ctx.patient_id)
        if label is None:
            continue
        if any(np.isnan(v) for v in ctx.context_vector):
            stays.pop((ctx.patient_id, ctx.stay_id), None)
            labels_map.pop((ctx.patient_id, ctx.stay_id), None)
            continue
        key = (ctx.patient_id, ctx.stay_id)
        if key not in stays:
            stays[key] = []
            labels_map[key] = label
        stays[key].append(ctx)

    log.info(f"Total stays to evaluate: {len(stays)}")

    y_true, y_prob, y_unc = [], [], []
    for key, ctx_list in tqdm(stays.items(), desc="GRU inference", unit="stay"):
        ctx_list.sort(key=lambda c: c.window_index)
        result = predict_sequence(ctx_list)
        y_true.append(float(labels_map[key]))
        y_prob.append(result["risk_probability"])
        y_unc.append(result["uncertainty_score"])

    return np.array(y_true), np.array(y_prob), np.array(y_unc)


def run_evaluation():
    device = get_device()
    log.info(f"=== Running Evaluation on {device} ===")

    # --- Load data ---
    ctxs = _load_contexts()
    log.info(f"Loaded {len(ctxs)} context objects")

    log.info("Loading label map...")
    label_map = load_label_map()
    log.info(f"Label map: {len(label_map)} patients")

    log.info("Loading model...")
    model = load_model()
    model_type = get_model_type()
    log.info(f"Evaluating {model_type.upper()} model on {device}")

    # --- Model inference ---
    if model_type == "gru":
        y_true, y_prob, y_unc = _eval_gru(ctxs, label_map)
    else:
        y_true, y_prob, y_unc = _eval_mlp(ctxs, label_map, model)

    # Free memory before baseline
    del ctxs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info(f"Evaluation samples: {len(y_true)}")

    # --- Metrics ---
    log.info("Computing metrics...")
    metrics = compute_all_metrics(y_true, y_prob)
    metrics["ece"] = compute_ece(y_true, y_prob)
    metrics["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob, y_unc
    )
    save_metrics(metrics, "main_model")
    log.info("Main model metrics saved")

    # --- Plots ---
    log.info("Generating plots...")
    plot_roc(y_true, y_prob)
    log.info("  ✓ ROC curve")
    plot_calibration(y_true, y_prob)
    log.info("  ✓ Calibration curve")
    plot_uncertainty_hist(y_unc, y_true)
    log.info("  ✓ Uncertainty histogram")

    # --- Baseline ---
    log.info("Running logistic regression baseline...")
    baseline_m = run_logistic_regression_baseline()
    comparison = {"model": metrics, "baseline_lr": baseline_m}
    save_metrics(comparison, "comparison")

    log.info("=== Evaluation Complete ===")
    log.info(f'  Model AUROC:    {metrics["auroc"]:.3f}')
    log.info(f'  Baseline AUROC: {baseline_m["auroc"]:.3f}')
    log.info(f'  ECE:            {metrics["ece"]:.4f}')
    return metrics


if __name__ == "__main__":
    run_evaluation()
