"""MC-Dropout Transformer encoder with learned attention pooling.

Pure attention-based alternative to the MC-Dropout GRU. Uses a stack of
Transformer encoder layers (self-attention + feed-forward) followed by
attention pooling over timesteps. Dropout is kept active at inference
time for MC-Dropout uncertainty estimation.
"""

import math
import torch
import torch.nn as nn
from src.utils.config_loader import get_config

cfg = get_config()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. 2017."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MCDropoutTransformer(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        input_dim = input_dim or cfg["model"]["input_dim"]
        d_model = cfg["model"].get("transformer_d_model", 128)
        nhead = cfg["model"].get("transformer_nhead", 4)
        num_layers = cfg["model"].get("transformer_num_layers", 2)
        dim_feedforward = cfg["model"].get("transformer_ff_dim", 256)
        p = cfg["model"]["dropout_rate"]

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=p,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_w = nn.Linear(d_model, 1, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 1),
        )

    def _build_padding_mask(self, lengths, seq_len, device):
        if lengths is None:
            return None
        return torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.to(device).unsqueeze(1)

    def _attend(self, output, pad_mask=None):
        scores = self.attn_w(output).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (output * weights).sum(dim=1)

    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.input_dropout(x)

        pad_mask = self._build_padding_mask(lengths, x.size(1), x.device)
        output = self.encoder(x, src_key_padding_mask=pad_mask)

        context = self._attend(output, pad_mask)
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
