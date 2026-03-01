"""Post-hoc temperature scaling for model calibration.

Fits a single scalar temperature T on a validation set such that
sigmoid(logit / T) is well-calibrated.  This does not change the model
weights — only the temperature is learned.
"""
import torch
import torch.nn as nn
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter to scale logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 200) -> float:
    """Fit temperature on validation logits and labels.

    Args:
        val_logits: Raw logits (before sigmoid) from the model, shape (N,).
        val_labels: Binary ground truth labels, shape (N,).
        max_iter: Optimization iterations.

    Returns:
        Optimal temperature value (float).
    """
    scaler = TemperatureScaler()
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = scaler.temperature.item()
    log.info(f"Temperature scaling: T = {T:.4f}")
    return T


def calibrate_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities.

    Args:
        logits: Raw logits array, shape (N,).
        temperature: Fitted temperature value.

    Returns:
        Calibrated probabilities, shape (N,).
    """
    scaled = logits / temperature
    return 1.0 / (1.0 + np.exp(-scaled))
