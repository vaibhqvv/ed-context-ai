"""MC-Dropout LSTM with learned attention pooling.

Direct architectural alternative to MCDropoutGRU — swaps GRU cells for
LSTM cells. Same attention pooling and classification head. Used for
apples-to-apples comparison with the main MC-Dropout GRU model.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.utils.config_loader import get_config

cfg = get_config()


class MCDropoutLSTM(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        input_dim = input_dim or cfg["model"]["input_dim"]
        hidden_dim = cfg["model"].get("lstm_hidden_dim", cfg["model"].get("gru_hidden_dim", 128))
        num_layers = cfg["model"].get("lstm_num_layers", cfg["model"].get("gru_num_layers", 2))
        p = cfg["model"]["dropout_rate"]

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p if num_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 1),
        )

    def _attend(self, output, lengths=None):
        scores = self.attn_w(output).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) >= lengths.to(output.device).unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (output * weights).sum(dim=1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(x)

        context = self._attend(output, lengths)
        context = self.layer_norm(context)
        return self.head(context).squeeze(-1)

    def predict_with_uncertainty(self, x, n_samples=None, lengths=None):
        n = n_samples or cfg["model"]["mc_dropout_samples"]
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        with torch.no_grad():
            logits = torch.stack(
                [self.forward(x, lengths) for _ in range(n)], dim=0
            )
            preds = torch.sigmoid(logits)
        return preds.mean(0), preds.var(0), preds
