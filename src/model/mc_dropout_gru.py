import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.utils.config_loader import get_config

cfg = get_config()


class MCDropoutGRU(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        input_dim = input_dim or cfg["model"]["input_dim"]
        hidden_dim = cfg["model"]["gru_hidden_dim"]
        num_layers = cfg["model"]["gru_num_layers"]
        p = cfg["model"]["dropout_rate"]

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths=None):
        """Return raw logits.

        Args:
            x: (batch, seq_len, input_dim) padded sequences
            lengths: (batch,) actual lengths for packing, or None
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        # h_n: (num_layers, batch, hidden) — take the last layer
        last_hidden = h_n[-1]  # (batch, hidden)
        return self.head(last_hidden).squeeze(-1)

    def predict_with_uncertainty(self, x, n_samples=None, lengths=None):
        n = n_samples or cfg["model"]["mc_dropout_samples"]
        self.eval()
        # Enable only dropout layers for MC sampling (keep GRU/BN in eval)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        with torch.no_grad():
            logits = torch.stack(
                [self.forward(x, lengths) for _ in range(n)], dim=0
            )
            preds = torch.sigmoid(logits)
        return preds.mean(0), preds.var(0), preds

    def classify_uncertainty(self, variance):
        t = cfg["model"]["uncertainty_thresholds"]
        if variance < t["medium"]:
            return "High"
        if variance < t["high"]:
            return "Medium"
        return "Low"
