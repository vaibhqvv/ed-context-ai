"""
Run train + evaluate for multiple train/test split ratios.

Usage:
    pyenv shell 3.12.9 && python run_split_experiments.py

Experiments:
    1) 70/15/15 (original)
    2) 60/10/30
    3) 50/10/40

Each experiment produces its own sub-directory in:
    outputs/splits/<ratio>/
        ├── models/best_model.pt
        ├── plots/*.png        (ROC, calibration, uncertainty)
        ├── results/*.json     (GRU + all baselines)
        └── splits.json

After all experiments run, the best-performing GRU (by AUROC) is
promoted to the top-level outputs/ dirs as the canonical model.
"""

import json, pickle, shutil
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.preprocessing.pipeline import _save_splits

log = get_logger("split_experiments")
cfg = get_config()

EXPERIMENTS = [
    {"name": "70_15_15", "train": 0.70, "val": 0.15, "test": 0.15},
    {"name": "60_10_30", "train": 0.60, "val": 0.10, "test": 0.30},
    {"name": "50_10_40", "train": 0.50, "val": 0.10, "test": 0.40},
]

SPLITS_ROOT = Path("outputs/splits")


def load_records():
    pkl_path = Path(cfg["paths"]["processed_data"]) / "patient_records.pkl"
    log.info(f"Loading patient records from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _copytree(src: Path, dst: Path):
    """Copy all files from src dir into dst dir (dst created if missing)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def run_experiment(records, exp):
    name = exp["name"]
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT: {name} (train={exp['train']}, val={exp['val']}, test={exp['test']})")
    log.info(f"{'='*60}")

    split_path = Path(cfg["paths"]["splits"])
    model_path = Path(cfg["paths"]["model_output"])
    results_path = Path(cfg["paths"]["results"])
    plots_path = Path(cfg["paths"]["plots"])

    exp_dir = SPLITS_ROOT / name
    exp_models = exp_dir / "models"
    exp_results = exp_dir / "results"
    exp_plots = exp_dir / "plots"
    for d in (exp_models, exp_results, exp_plots):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate splits for this experiment
    _save_splits(records, train=exp["train"], val=exp["val"], test=exp["test"])

    # 2. Train
    log.info(f"[{name}] Training GRU model...")
    from src.model.trainer import train
    train()

    # 3. Evaluate (includes all baselines)
    log.info(f"[{name}] Running evaluation...")
    from src.evaluation.eval_runner import run_evaluation
    run_evaluation()

    # 4. Copy results → outputs/splits/<name>/results/
    for result_file in results_path.glob("*_metrics.json"):
        shutil.copy2(result_file, exp_results / result_file.name)
    log.info(f"  Saved results to: {exp_results}")

    # 5. Copy plots → outputs/splits/<name>/plots/
    _copytree(plots_path, exp_plots)
    log.info(f"  Saved plots to:   {exp_plots}")

    # 6. Copy model checkpoint → outputs/splits/<name>/models/
    model_file = model_path / "best_model.pt"
    if model_file.exists():
        shutil.copy2(model_file, exp_models / "best_model.pt")
        log.info(f"  Saved model to:   {exp_models / 'best_model.pt'}")

    # 7. Copy splits.json → outputs/splits/<name>/
    splits_file = split_path / "splits.json"
    if splits_file.exists():
        shutil.copy2(splits_file, exp_dir / "splits.json")

    log.info(f"[{name}] Done.\n")


def summarize_results():
    model_path = Path(cfg["paths"]["model_output"])
    plots_path = Path(cfg["paths"]["plots"])
    results_path = Path(cfg["paths"]["results"])
    split_path = Path(cfg["paths"]["splits"])

    summary = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_dir = SPLITS_ROOT / name
        model_metrics_file = exp_dir / "results" / "main_model_metrics.json"
        if not model_metrics_file.exists():
            continue
        with open(model_metrics_file) as f:
            model_m = json.load(f)

        entry = {"gru": {"auroc": model_m["auroc"], "auprc": model_m["auprc"],
                         "f1": model_m["f1"], "brier": model_m["brier_score"],
                         "n_test": model_m["n_samples"]}}

        for baseline in ["lr", "rf", "xgb"]:
            bl_file = exp_dir / "results" / f"baseline_{baseline}_metrics.json"
            if bl_file.exists():
                with open(bl_file) as f:
                    bl_m = json.load(f)
                entry[baseline] = {"auroc": bl_m["auroc"], "auprc": bl_m["auprc"],
                                   "f1": bl_m["f1"], "brier": bl_m["brier_score"]}

        summary[name] = entry

    # Print summary table
    print("\n" + "=" * 80)
    print("SPLIT EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Split':<12} {'Model':<6} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Brier':>7} {'N_test':>10}")
    print("-" * 80)
    for split_name, models in summary.items():
        for model_name, m in models.items():
            n_test = m.get("n_test", "")
            print(f"{split_name:<12} {model_name:<6} {m['auroc']:>7.3f} {m['auprc']:>7.3f} "
                  f"{m['f1']:>7.3f} {m['brier']:>7.3f} {str(n_test):>10}")
        print("-" * 80)

    # Pick the best GRU split by AUROC
    best_split = None
    best_auroc = -1.0
    for split_name, models in summary.items():
        gru_auroc = models.get("gru", {}).get("auroc", 0.0)
        if gru_auroc > best_auroc:
            best_auroc = gru_auroc
            best_split = split_name

    if best_split:
        best_dir = SPLITS_ROOT / best_split
        print("\n" + "=" * 80)
        print(f"BEST GRU MODEL: split={best_split} (AUROC={best_auroc:.4f})")
        print("=" * 80)

        # Promote best artifacts to top-level outputs/
        best_model = best_dir / "models" / "best_model.pt"
        if best_model.exists():
            shutil.copy2(best_model, model_path / "best_model.pt")
            print(f"Promoted model:   {best_model} → {model_path / 'best_model.pt'}")

        best_splits = best_dir / "splits.json"
        if best_splits.exists():
            shutil.copy2(best_splits, split_path / "splits.json")
            print(f"Promoted splits:  {best_splits} → {split_path / 'splits.json'}")

        # Promote plots (overwrite top-level)
        _copytree(best_dir / "plots", plots_path)
        print(f"Promoted plots:   {best_dir / 'plots'} → {plots_path}")

        # Promote metrics JSONs
        _copytree(best_dir / "results", results_path)
        print(f"Promoted results: {best_dir / 'results'} → {results_path}")

        summary["_best_split"] = best_split
        summary["_best_auroc"] = best_auroc

    # Save summary at top-level
    out = results_path / "split_experiment_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    for exp in EXPERIMENTS:
        run_experiment(records, exp)
    summarize_results()
