import torch, pickle, numpy as np
from pathlib import Path
from src.model.inference import load_model
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

log = get_logger(__name__)
cfg = get_config()


def run_evaluation():
    log.info("=== Running Evaluation ===")
    with open(Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb") as f:
        ctxs = pickle.load(f)
    label_map = load_label_map()
    model = load_model()

    device = next(model.parameters()).device
    y_true, y_prob, y_unc = [], [], []
    for ctx in ctxs:
        label = label_map.get(ctx.patient_id)
        if label is None:
            continue
        if any(np.isnan(v) for v in ctx.context_vector):
            continue
        x = torch.tensor(np.array(ctx.context_vector, dtype=np.float32)).unsqueeze(0).to(device)
        mean, var, _ = model.predict_with_uncertainty(x)
        y_true.append(float(label))
        y_prob.append(float(mean.item()))
        y_unc.append(float(var.item()))
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_unc = np.array(y_unc)

    metrics = compute_all_metrics(y_true, y_prob)
    metrics["ece"] = compute_ece(y_true, y_prob)
    metrics["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob, y_unc
    )
    save_metrics(metrics, "main_model")

    plot_roc(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    plot_uncertainty_hist(y_unc, y_true)

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
