import torch
import torch.nn as nn
from src.utils.config_loader import get_config

cfg = get_config()


class MCDropoutNet(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        input_dim = input_dim or cfg["model"]["input_dim"]
        hidden_dims = cfg["model"]["hidden_dims"]
        p = cfg["model"]["dropout_rate"]
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            in_d = h
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_with_uncertainty(self, x, n_samples=None):
        n = n_samples or cfg["model"]["mc_dropout_samples"]
        self.train()
        with torch.no_grad():
            preds = torch.stack([self.forward(x) for _ in range(n)], dim=0)
        return preds.mean(0), preds.var(0), preds

    def classify_uncertainty(self, variance):
        t = cfg["model"]["uncertainty_thresholds"]
        if variance < t["medium"]:
            return "High"
        if variance < t["high"]:
            return "Medium"
        return "Low"
