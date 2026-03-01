import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from src.utils.config_loader import get_config

PLOT_DIR = Path(get_config()["paths"]["plots"])


def plot_roc(y_true, y_prob, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2C5F8A", lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


def plot_calibration(y_true, y_prob, n_bins=10, filename="calibration.png"):
    fp, mp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot(mp, fp, "s-", color="#2C5F8A", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean Predicted Prob")
    plt.ylabel("Fraction Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


def plot_uncertainty_hist(uncertainties, y_true, filename="uncertainty_dist.png"):
    plt.figure(figsize=(8, 5))
    for label, color in [(0, "#2A6B3A"), (1, "#C05C1A")]:
        plt.hist(
            uncertainties[y_true == label],
            bins=40,
            alpha=0.6,
            color=color,
            label="No adverse event" if label == 0 else "Adverse event",
        )
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Count")
    plt.title("Uncertainty Distribution by Outcome")
    plt.legend()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()
