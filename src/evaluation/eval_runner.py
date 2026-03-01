import gc, json, torch, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from src.model.inference import load_model, get_model_type, get_feature_stats
from src.model.dataset import load_label_map
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_ece,
    uncertainty_stratified_auroc,
    save_metrics,
)
from src.evaluation.plots import plot_roc, plot_calibration, plot_uncertainty_hist
from src.evaluation.baseline import run_logistic_regression_baseline
from src.evaluation.calibration import fit_temperature, calibrate_probabilities
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


def _get_test_patient_ids(ctxs):
    """Get test-set patient IDs from splits.json or create a split."""
    splits_path = Path(cfg["paths"].get("splits", "data/splits")) / "splits.json"
    if splits_path.exists():
        log.info(f"Using saved splits from {splits_path}")
        with open(splits_path) as f:
            splits = json.load(f)
        # splits.json has stay_ids; we need patient_ids. Load the mapping.
        records_path = Path(cfg["paths"]["processed_data"]) / "patient_records.pkl"
        with open(records_path, "rb") as f:
            records = pickle.load(f)
        stay_to_patient = {r.stay_id: r.patient_id for r in records}
        del records
        test_stay_ids = set(splits.get("test", []))
        test_pids = {stay_to_patient[sid] for sid in test_stay_ids if sid in stay_to_patient}
        val_stay_ids = set(splits.get("val", []))
        val_pids = {stay_to_patient[sid] for sid in val_stay_ids if sid in stay_to_patient}
        log.info(f"Test patients: {len(test_pids)}, Val patients: {len(val_pids)}")
        return test_pids, val_pids

    # Fallback: create our own 70/15/15 patient-level split
    log.info("No splits.json found — creating patient-level 70/15/15 split")
    all_pids = list(set(ctx.patient_id for ctx in ctxs))
    np.random.seed(42)
    np.random.shuffle(all_pids)
    n = len(all_pids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    val_pids = set(all_pids[n_train:n_train + n_val])
    test_pids = set(all_pids[n_train + n_val:])
    log.info(f"Test patients: {len(test_pids)}, Val patients: {len(val_pids)}")
    return test_pids, val_pids


def _eval_mlp(ctxs, label_map, model):
    """Per-context evaluation for MLP with batched GPU inference."""
    device = next(model.parameters()).device
    log.info("Filtering valid contexts for MLP evaluation...")

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

    batch_size = 512
    y_prob_list, y_unc_list, y_logit_list = [], [], []
    n_batches = (len(valid_x) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="MC Dropout inference", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, len(valid_x))
        x_batch = torch.tensor(np.stack(valid_x[start:end]), dtype=torch.float32).to(device)
        mean, var, _ = model.predict_with_uncertainty(x_batch)
        # Convert mean probs back to logits for temperature scaling
        mean_np = mean.cpu().numpy()
        logits = np.log(mean_np / (1 - np.clip(mean_np, 1e-7, 1 - 1e-7)))
        y_logit_list.append(logits)
        y_prob_list.append(mean_np)
        y_unc_list.append(var.cpu().numpy())

    y_prob = np.concatenate(y_prob_list)
    y_unc = np.concatenate(y_unc_list)
    y_logits = np.concatenate(y_logit_list)
    return y_true, y_prob, y_unc, y_logits


def _eval_gru(ctxs, label_map):
    """Sequence-level evaluation for GRU with batched padded GPU inference."""
    device = get_device()
    model = load_model()
    feat_mean, feat_std = get_feature_stats()

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

    log.info("Building sequence tensors...")
    all_seqs = []
    all_labels = []
    for key, ctx_list in tqdm(stays.items(), desc="Building sequences", unit="stay"):
        ctx_list.sort(key=lambda c: c.window_index)
        vecs = [np.array(c.context_vector, dtype=np.float32) for c in ctx_list]
        seq = torch.tensor(np.stack(vecs), dtype=torch.float32)
        all_seqs.append(seq)
        all_labels.append(float(labels_map[key]))

    y_true = np.array(all_labels)
    del stays, labels_map
    gc.collect()

    batch_size = 256
    n_total = len(all_seqs)
    n_batches = (n_total + batch_size - 1) // batch_size
    log.info(f"Running batched GRU inference: {n_total} stays, batch_size={batch_size}")

    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    n_mc = cfg["model"]["mc_dropout_samples"]
    y_prob_list, y_unc_list, y_logit_list = [], [], []

    for i in tqdm(range(n_batches), desc="Batched GRU inference", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, n_total)
        batch_seqs = all_seqs[start:end]

        lengths = torch.tensor([s.shape[0] for s in batch_seqs], dtype=torch.long)
        padded = pad_sequence(batch_seqs, batch_first=True, padding_value=0.0).to(device)
        padded = (padded - feat_mean) / feat_std

        with torch.no_grad():
            logits_stack = torch.stack(
                [model(padded, lengths) for _ in range(n_mc)], dim=0
            )
            preds = torch.sigmoid(logits_stack)
            mean = preds.mean(dim=0)
            var = preds.var(dim=0)
            # Mean logits for temperature scaling
            mean_logits = logits_stack.mean(dim=0)

        mean_np = mean.cpu().numpy()
        y_prob_list.append(mean_np)
        y_unc_list.append(var.cpu().numpy())
        y_logit_list.append(mean_logits.cpu().numpy())

    y_prob = np.concatenate(y_prob_list)
    y_unc = np.concatenate(y_unc_list)
    y_logits = np.concatenate(y_logit_list)
    return y_true, y_prob, y_unc, y_logits


def run_evaluation():
    device = get_device()
    log.info(f"=== Running Evaluation on {device} ===")

    # --- Load data ---
    ctxs = _load_contexts()
    log.info(f"Loaded {len(ctxs)} context objects")

    log.info("Loading label map...")
    label_map = load_label_map()
    log.info(f"Label map: {len(label_map)} patients")

    # --- Get test/val splits ---
    test_pids, val_pids = _get_test_patient_ids(ctxs)

    # Filter to test-set contexts only
    test_ctxs = [c for c in tqdm(ctxs, desc="Filtering test set") if c.patient_id in test_pids]
    val_ctxs = [c for c in ctxs if c.patient_id in val_pids]
    log.info(f"Test contexts: {len(test_ctxs)}, Val contexts: {len(val_ctxs)}")

    # Free full set
    del ctxs
    gc.collect()

    log.info("Loading model...")
    model = load_model()
    model_type = get_model_type()
    log.info(f"Evaluating {model_type.upper()} model on {device}")

    # --- Validation set inference (for temperature scaling) ---
    log.info("--- Running inference on validation set for temperature calibration ---")
    if model_type == "gru":
        val_true, val_prob, val_unc, val_logits = _eval_gru(val_ctxs, label_map)
    else:
        val_true, val_prob, val_unc, val_logits = _eval_mlp(val_ctxs, label_map, model)

    del val_ctxs
    gc.collect()

    # --- Fit temperature scaling ---
    log.info("Fitting temperature scaling on validation set...")
    temperature = fit_temperature(val_logits, val_true)
    del val_true, val_prob, val_unc, val_logits
    gc.collect()

    # --- Test set inference ---
    log.info("--- Running inference on test set ---")
    if model_type == "gru":
        y_true, y_prob_raw, y_unc, y_logits = _eval_gru(test_ctxs, label_map)
    else:
        y_true, y_prob_raw, y_unc, y_logits = _eval_mlp(test_ctxs, label_map, model)

    del test_ctxs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info(f"Test samples: {len(y_true)}")

    # --- Calibrate probabilities ---
    y_prob_calibrated = calibrate_probabilities(y_logits, temperature)

    # --- Metrics (raw) ---
    log.info("Computing metrics (raw probabilities)...")
    metrics_raw = compute_all_metrics(y_true, y_prob_raw)
    metrics_raw["ece"] = compute_ece(y_true, y_prob_raw)
    metrics_raw["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob_raw, y_unc
    )
    save_metrics(metrics_raw, "main_model_raw")

    # --- Metrics (calibrated) ---
    log.info("Computing metrics (temperature-scaled)...")
    metrics_cal = compute_all_metrics(y_true, y_prob_calibrated)
    metrics_cal["ece"] = compute_ece(y_true, y_prob_calibrated)
    metrics_cal["temperature"] = temperature
    metrics_cal["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob_calibrated, y_unc
    )
    save_metrics(metrics_cal, "main_model")

    # --- Plots (calibrated) ---
    log.info("Generating plots...")
    plot_roc(y_true, y_prob_calibrated)
    log.info("  ✓ ROC curve")
    plot_calibration(y_true, y_prob_calibrated)
    log.info("  ✓ Calibration curve")
    plot_uncertainty_hist(y_unc, y_true)
    log.info("  ✓ Uncertainty histogram")

    # --- Baseline ---
    log.info("Running logistic regression baseline...")
    baseline_m = run_logistic_regression_baseline()
    comparison = {
        "model_raw": metrics_raw,
        "model_calibrated": metrics_cal,
        "baseline_lr": baseline_m,
    }
    save_metrics(comparison, "comparison")

    log.info("=== Evaluation Complete ===")
    log.info(f'  Raw AUROC:        {metrics_raw["auroc"]:.3f} (ECE: {metrics_raw["ece"]:.4f})')
    log.info(f'  Calibrated AUROC: {metrics_cal["auroc"]:.3f} (ECE: {metrics_cal["ece"]:.4f})')
    log.info(f'  Baseline AUROC:   {baseline_m["auroc"]:.3f}')
    log.info(f'  Temperature:      {temperature:.4f}')
    return metrics_cal


if __name__ == "__main__":
    run_evaluation()
