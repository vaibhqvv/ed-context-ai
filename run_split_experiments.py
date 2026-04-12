"""
Run train + evaluate for multiple train/test split ratios.

Usage:
    pyenv shell 3.12.9 && python run_split_experiments.py

Experiments:
    1) 70/15/15 (original)
    2) 60/10/30
    3) 50/10/40

Each experiment:
    - Creates a new splits.json with the given ratio
    - Trains a fresh GRU model
    - Evaluates on the test set
    - Runs all baselines (LR, RF, XGBoost)
    - Saves results to outputs/results/split_experiment_<name>_*.json
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


def load_records():
    pkl_path = Path(cfg["paths"]["processed_data"]) / "patient_records.pkl"
    log.info(f"Loading patient records from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_experiment(records, exp):
    name = exp["name"]
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT: {name} (train={exp['train']}, val={exp['val']}, test={exp['test']})")
    log.info(f"{'='*60}")

    split_path = Path(cfg["paths"]["splits"])
    model_path = Path(cfg["paths"]["model_output"])
    results_path = Path(cfg["paths"]["results"])

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

    # 4. Copy results with experiment prefix
    for result_file in results_path.glob("*_metrics.json"):
        dest = results_path / f"split_{name}_{result_file.name}"
        shutil.copy2(result_file, dest)
        log.info(f"  Saved: {dest.name}")

    # 5. Copy splits with experiment prefix
    splits_file = split_path / "splits.json"
    if splits_file.exists():
        dest = split_path / f"splits_{name}.json"
        shutil.copy2(splits_file, dest)

    # 6. Copy model checkpoint
    model_file = model_path / "best_model.pt"
    if model_file.exists():
        dest = model_path / f"best_model_{name}.pt"
        shutil.copy2(model_file, dest)
        log.info(f"  Model saved: {dest.name}")

    log.info(f"[{name}] Done.\n")


def summarize_results():
    results_path = Path(cfg["paths"]["results"])
    summary = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        model_file = results_path / f"split_{name}_main_model_metrics.json"
        if not model_file.exists():
            continue
        with open(model_file) as f:
            model_m = json.load(f)

        entry = {"gru": {"auroc": model_m["auroc"], "auprc": model_m["auprc"],
                         "f1": model_m["f1"], "brier": model_m["brier_score"],
                         "n_test": model_m["n_samples"]}}

        for baseline in ["lr", "rf", "xgb"]:
            bl_file = results_path / f"split_{name}_baseline_{baseline}_metrics.json"
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

    # Save summary
    out = results_path / "split_experiment_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    records = load_records()
    for exp in EXPERIMENTS:
        run_experiment(records, exp)
    summarize_results()
