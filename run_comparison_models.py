"""
Train and evaluate LSTM+Attention and Transformer encoder as comparison
baselines against the existing MC-Dropout GRU.

- Uses the SAME splits.json, context_objects.pkl, and patient_records.pkl
  as the main pipeline — no data recomputation.
- Fully GPU-accelerated (CUDA, cudnn.benchmark, TF32, pin_memory).
- Outputs go to a separate tree so existing GRU results are untouched:

    outputs/comparison/
    ├── lstm_attention/
    │   ├── models/best_model.pt
    │   ├── plots/{roc_curve,calibration,uncertainty_dist}.png
    │   └── results/{main_model,main_model_raw}_metrics.json
    ├── transformer/
    │   └── (same layout)
    └── comparison_summary.json

Usage:
    pyenv shell 3.12.9 && python run_comparison_models.py
    pyenv shell 3.12.9 && python run_comparison_models.py --only lstm
    pyenv shell 3.12.9 && python run_comparison_models.py --only transformer
"""

import argparse
import gc
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.evaluation.calibration import calibrate_probabilities, fit_temperature
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_ece,
    uncertainty_stratified_auroc,
)
from src.model.dataset import load_label_map
from src.model.mc_dropout_lstm import MCDropoutLSTM
from src.model.mc_dropout_transformer import MCDropoutTransformer
from src.model.sequence_dataset import EDSequenceDataset, sequence_collate_fn
from src.utils.config_loader import get_config
from src.utils.device import get_device
from src.utils.logger import get_logger

log = get_logger("comparison")
cfg = get_config()

COMPARISON_ROOT = Path("outputs/comparison")

# ────────────────────────────────────────────────────────────────────────
# RTX 3090 optimizations (scoped to this script only — config.yaml untouched)
# ────────────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 1024          # up from 512 — 3090's 24GB VRAM handles this
MC_INFERENCE_BATCH = 1024        # up from 256
USE_TORCH_COMPILE = True         # PyTorch 2.x graph compilation
USE_AMP = True                   # mixed precision
AMP_DTYPE = torch.bfloat16       # 3090 supports bf16 natively, more stable than fp16

# Model-specific configuration overrides (keeps config.yaml intact)
# These are only injected into the new LSTM/Transformer models, not the GRU.
MODEL_OVERRIDES = {
    "lstm_attention": {
        # keep hidden dims comparable to GRU for a fair direct architectural comparison
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 2,
    },
    "transformer": {
        # bigger transformer — 3090 can afford it
        "transformer_d_model": 256,
        "transformer_nhead": 8,
        "transformer_num_layers": 4,
        "transformer_ff_dim": 512,
    },
}

MODEL_REGISTRY = {
    "lstm_attention": {
        "class": MCDropoutLSTM,
        "label": "LSTM+Attention",
    },
    "transformer": {
        "class": MCDropoutTransformer,
        "label": "Transformer",
    },
}


def _apply_model_overrides(model_key):
    """Inject per-model config overrides into the shared cfg dict.

    Restores original values when the caller is done (via the returned dict).
    This is safe because only this script touches these keys, and the new
    models read them at __init__ time.
    """
    saved = {}
    for k, v in MODEL_OVERRIDES.get(model_key, {}).items():
        saved[k] = cfg["model"].get(k)  # may be None
        cfg["model"][k] = v
    return saved


def _restore_model_overrides(saved):
    for k, v in saved.items():
        if v is None:
            cfg["model"].pop(k, None)
        else:
            cfg["model"][k] = v


# ────────────────────────────────────────────────────────────────────────
# Data / splits
# ────────────────────────────────────────────────────────────────────────

def _load_contexts():
    pkl_path = Path(cfg["paths"]["context_data"]) / "context_objects.pkl"
    file_size = pkl_path.stat().st_size
    log.info(f"Loading context objects ({file_size / 1e6:.0f} MB)...")
    with open(pkl_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Loading pkl") as pbar:
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
            return pickle.load(_TrackedReader(f, pbar))


def _load_splits():
    splits_path = Path(cfg["paths"].get("splits", "data/splits")) / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found. Run preprocessing first."
        )
    with open(splits_path) as f:
        splits = json.load(f)
    return {
        "train": set(splits.get("train", [])),
        "val": set(splits.get("val", [])),
        "test": set(splits.get("test", [])),
    }


def _split_dataset(dataset, splits):
    train_idx, val_idx, test_idx = [], [], []
    for i, sid in enumerate(dataset.stay_ids):
        if sid in splits["train"]:
            train_idx.append(i)
        elif sid in splits["val"]:
            val_idx.append(i)
        elif sid in splits["test"]:
            test_idx.append(i)
    log.info(
        f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


# ────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────

def train_model(model_key, dataset, train_idx, val_idx, device, out_dir):
    info = MODEL_REGISTRY[model_key]
    log.info(f"=== Training {info['label']} ===")

    # Inject model-specific config overrides for this run
    saved_cfg = _apply_model_overrides(model_key)

    # Standardize with train stats only
    log.info("Computing feature standardization stats on train...")
    train_vecs = torch.cat([dataset.sequences[i] for i in train_idx], dim=0)
    mean = train_vecs.mean(dim=0)
    std = train_vecs.std(dim=0)
    std[std < 1e-8] = 1.0
    del train_vecs

    normalized = [(s - mean) / std for s in dataset.sequences]

    # Wrap into a lightweight indexable dataset view
    class _View(torch.utils.data.Dataset):
        def __init__(self, seqs, labels):
            self.seqs = seqs
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.seqs[idx], self.labels[idx]

    view = _View(normalized, dataset.labels)
    train_ds = Subset(view, train_idx)
    val_ds = Subset(view, val_idx)

    # 3090 optimization: bigger batch, more DataLoader workers (CPU→GPU overlap)
    batch_size = TRAIN_BATCH_SIZE
    n_workers = min(os.cpu_count() or 1, 4) if device.type == "cuda" else 2
    log.info(f"Batch size: {batch_size} | Workers: {n_workers}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, collate_fn=sequence_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, collate_fn=sequence_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )

    model_cls = info["class"]
    model = model_cls(input_dim=normalized[0].shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"{info['label']} parameters: {n_params:,}")

    # PyTorch 2.x graph compilation for speed on Ampere
    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info(f"{info['label']}: torch.compile enabled (reduce-overhead)")
        except Exception as e:
            log.warning(f"torch.compile failed ({e}); continuing without compile")

    base_lr = cfg["model"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-3)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    warmup_epochs = cfg["model"].get("warmup_epochs", 5)

    # Class weight
    train_labels = torch.tensor([dataset.labels[i] for i in train_idx])
    n_pos = int(train_labels.sum().item())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log.info(f"Class balance — pos:{n_pos} neg:{n_neg} pos_weight:{pos_weight.item():.2f}")

    label_smoothing = cfg["model"].get("label_smoothing", 0.0)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss()

    # AMP with bfloat16 — 3090 has native bf16 tensor cores, more numerically
    # stable than fp16 so GradScaler is not needed with bf16
    use_amp = USE_AMP and device.type == "cuda"
    use_bf16 = use_amp and AMP_DTYPE == torch.bfloat16
    # GradScaler only needed for fp16, not bf16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))
    if use_amp:
        log.info(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")

    best_auroc = 0.0
    patience_ctr = 0
    patience = cfg["model"]["early_stopping_patience"]
    max_epochs = cfg["model"]["max_epochs"]

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "best_model.pt"

    t0 = time.time()
    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        t_losses = []
        for X, y, lengths in tqdm(
            train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False
        ):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if label_smoothing > 0:
                y = y * (1 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                logits = model(X, lengths)
                loss = train_criterion(logits, y)

            if use_bf16:
                # bf16 has same dynamic range as fp32 → skip scaler
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            t_losses.append(loss.item())

        # Validation
        model.eval()
        v_losses, all_logits, all_labels = [], [], []
        with torch.no_grad():
            for X, y, lengths in tqdm(
                val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False
            ):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                    logits = model(X, lengths)
                v_losses.append(val_criterion(logits, y).item())
                all_logits.append(logits.float().cpu())
                all_labels.append(y.cpu())

        v_loss = float(np.mean(v_losses))
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-all_logits))
        val_auroc = roc_auc_score(all_labels, val_probs)
        val_auprc = average_precision_score(all_labels, val_probs)

        if epoch >= warmup_epochs:
            plateau_scheduler.step(val_auroc)

        log.info(
            f"[{info['label']}] Epoch {epoch:3d} | "
            f"Train: {np.mean(t_losses):.4f} | Val: {v_loss:.4f} | "
            f"AUROC: {val_auroc:.3f} | AUPRC: {val_auprc:.3f}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_ctr = 0
            # torch.compile wraps the model — use _orig_mod if present so the
            # checkpoint loads cleanly in the eval step (which runs uncompiled)
            base_model = getattr(model, "_orig_mod", model)
            torch.save({
                "model_state": {k: v.cpu() for k, v in base_model.state_dict().items()},
                "model_key": model_key,
                "input_dim": normalized[0].shape[-1],
                "feature_mean": mean.cpu(),
                "feature_std": std.cpu(),
                "model_overrides": MODEL_OVERRIDES.get(model_key, {}),
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    elapsed = time.time() - t0
    log.info(f"{info['label']} training done in {elapsed / 60:.1f} min | best val AUROC: {best_auroc:.4f}")

    # Restore original config values (so other models aren't affected)
    _restore_model_overrides(saved_cfg)

    # Cleanup
    del normalized, train_ds, val_ds, train_loader, val_loader, model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ckpt_path, mean, std


# ────────────────────────────────────────────────────────────────────────
# Evaluation (GRU-compatible, batched padded GPU inference with MC-Dropout)
# ────────────────────────────────────────────────────────────────────────

def _batched_mc_inference(model, sequences, indices, mean, std, device, n_mc, batch_size=MC_INFERENCE_BATCH):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    n_total = len(indices)
    y_prob_list, y_unc_list, y_logit_list = [], [], []

    for start in tqdm(range(0, n_total, batch_size), desc="MC inference", unit="batch"):
        end = min(start + batch_size, n_total)
        batch_idx = indices[start:end]
        batch_seqs = [sequences[i] for i in batch_idx]
        lengths = torch.tensor([s.shape[0] for s in batch_seqs], dtype=torch.long)
        padded = pad_sequence(batch_seqs, batch_first=True, padding_value=0.0).to(device)
        padded = (padded - mean.to(device)) / std.to(device)

        with torch.no_grad():
            logits_stack = torch.stack(
                [model(padded, lengths) for _ in range(n_mc)], dim=0
            )
            preds = torch.sigmoid(logits_stack)
            mean_preds = preds.mean(dim=0)
            var_preds = preds.var(dim=0)
            mean_logits = logits_stack.mean(dim=0)

        y_prob_list.append(mean_preds.cpu().numpy())
        y_unc_list.append(var_preds.cpu().numpy())
        y_logit_list.append(mean_logits.cpu().numpy())

    return (
        np.concatenate(y_prob_list),
        np.concatenate(y_unc_list),
        np.concatenate(y_logit_list),
    )


def evaluate_model(model_key, dataset, val_idx, test_idx, ckpt_path, mean, std, device, out_dir):
    info = MODEL_REGISTRY[model_key]
    log.info(f"=== Evaluating {info['label']} ===")

    # Re-apply overrides so the model instantiates with the same architecture
    # that was trained (important for Transformer d_model, nhead, etc.)
    saved_cfg = _apply_model_overrides(model_key)
    try:
        model_cls = info["class"]
        model = model_cls(input_dim=dataset.sequences[0].shape[-1]).to(device)
    finally:
        _restore_model_overrides(saved_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Loaded checkpoint from {ckpt_path}")

    n_mc = cfg["model"]["mc_dropout_samples"]

    # Validation set for temperature scaling
    log.info("Running MC-Dropout on validation set (for temperature scaling)...")
    val_true = np.array([dataset.labels[i].item() for i in val_idx])
    _, _, val_logits = _batched_mc_inference(
        model, dataset.sequences, val_idx, mean, std, device, n_mc
    )
    temperature = fit_temperature(val_logits, val_true)
    log.info(f"Fitted temperature: {temperature:.4f}")

    # Test set
    log.info("Running MC-Dropout on test set...")
    y_true = np.array([dataset.labels[i].item() for i in test_idx])
    y_prob_raw, y_unc, y_logits = _batched_mc_inference(
        model, dataset.sequences, test_idx, mean, std, device, n_mc
    )
    y_prob_cal = calibrate_probabilities(y_logits, temperature)

    results_dir = out_dir / "results"
    plots_dir = out_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Raw metrics
    metrics_raw = compute_all_metrics(y_true, y_prob_raw)
    metrics_raw["ece"] = compute_ece(y_true, y_prob_raw)
    metrics_raw["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob_raw, y_unc
    )
    with open(results_dir / "main_model_raw_metrics.json", "w") as f:
        json.dump(metrics_raw, f, indent=2)

    # Calibrated metrics
    metrics_cal = compute_all_metrics(y_true, y_prob_cal)
    metrics_cal["ece"] = compute_ece(y_true, y_prob_cal)
    metrics_cal["temperature"] = float(temperature)
    metrics_cal["uncertainty_stratified_auroc"] = uncertainty_stratified_auroc(
        y_true, y_prob_cal, y_unc
    )
    with open(results_dir / "main_model_metrics.json", "w") as f:
        json.dump(metrics_cal, f, indent=2)

    log.info(
        f"[{info['label']}] AUROC: {metrics_cal['auroc']:.3f} | "
        f"AUPRC: {metrics_cal['auprc']:.3f} | F1: {metrics_cal['f1']:.3f} | "
        f"Brier: {metrics_cal['brier_score']:.3f} | ECE: {metrics_cal['ece']:.4f}"
    )

    # Plots (scoped to this model's plots_dir)
    _plot_roc(y_true, y_prob_cal, plots_dir / "roc_curve.png", info["label"])
    _plot_calibration(y_true, y_prob_cal, plots_dir / "calibration.png", info["label"])
    _plot_uncertainty(y_unc, y_true, plots_dir / "uncertainty_dist.png", info["label"])

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics_cal


# ────────────────────────────────────────────────────────────────────────
# Plots (scoped to arbitrary output directories)
# ────────────────────────────────────────────────────────────────────────

def _plot_roc(y_true, y_prob, path, title_suffix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2C5F8A", lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve — {title_suffix}")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_calibration(y_true, y_prob, path, title_suffix, n_bins=10):
    fp, mp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot(mp, fp, "s-", color="#2C5F8A", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean Predicted Prob"); plt.ylabel("Fraction Positives")
    plt.title(f"Calibration — {title_suffix}")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_uncertainty(unc, y_true, path, title_suffix):
    plt.figure(figsize=(8, 5))
    for label, color in [(0, "#2A6B3A"), (1, "#C05C1A")]:
        plt.hist(unc[y_true == label], bins=40, alpha=0.6, color=color,
                 label="Discharged" if label == 0 else "Admitted")
    plt.xlabel("Uncertainty Score"); plt.ylabel("Count")
    plt.title(f"Uncertainty Distribution — {title_suffix}")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────

def run(model_keys):
    # CUDA speed knobs
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)

    # Load data once — all models share the same cached sequences
    log.info("Loading context data + labels...")
    contexts = _load_contexts()
    label_map = load_label_map()
    dataset = EDSequenceDataset(contexts, label_map, device=None)
    del contexts, label_map
    gc.collect()
    log.info(f"Dataset: {len(dataset)} stays")

    splits = _load_splits()
    train_idx, val_idx, test_idx = _split_dataset(dataset, splits)

    summary = {}
    for key in model_keys:
        out_dir = COMPARISON_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n{'='*60}\n{MODEL_REGISTRY[key]['label']} → {out_dir}\n{'='*60}")

        ckpt_path, mean, std = train_model(
            key, dataset, train_idx, val_idx, device, out_dir
        )
        metrics = evaluate_model(
            key, dataset, val_idx, test_idx, ckpt_path, mean, std, device, out_dir
        )
        summary[key] = {
            "label": MODEL_REGISTRY[key]["label"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "n_test": metrics["n_samples"],
        }

    # Include existing GRU metrics in the summary for easy comparison
    gru_metrics_path = Path(cfg["paths"]["results"]) / "main_model_metrics.json"
    if gru_metrics_path.exists():
        with open(gru_metrics_path) as f:
            gru = json.load(f)
        summary["gru"] = {
            "label": "MC-Dropout GRU",
            "auroc": gru.get("auroc"),
            "auprc": gru.get("auprc"),
            "f1": gru.get("f1"),
            "brier_score": gru.get("brier_score"),
            "ece": gru.get("ece"),
            "n_test": gru.get("n_samples"),
        }

    summary_path = COMPARISON_ROOT / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print human-readable summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Brier':>7} {'ECE':>7} {'N':>10}")
    print("-" * 80)
    for key, m in summary.items():
        n = m.get("n_test", "")
        print(f"{m['label']:<20} "
              f"{(m['auroc'] or 0):>7.3f} {(m['auprc'] or 0):>7.3f} "
              f"{(m['f1'] or 0):>7.3f} {(m['brier_score'] or 0):>7.3f} "
              f"{(m.get('ece') or 0):>7.4f} {str(n):>10}")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Run only one model (default: all)",
    )
    args = parser.parse_args()

    keys = list(MODEL_REGISTRY.keys()) if args.only == "all" else [args.only]
    run(keys)
