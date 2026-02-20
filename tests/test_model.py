# tests/test_model.py
import torch
from src.model.mc_dropout import MCDropoutNet


def test_mc_dropout_output():
    model = MCDropoutNet(input_dim=64)
    x = torch.randn(1, 64)
    mean, var, preds = model.predict_with_uncertainty(x, n_samples=10)
    assert 0 <= float(mean.item()) <= 1, "Risk must be in [0,1]"
    assert float(var.item()) >= 0, "Variance must be non-negative"
    assert preds.shape[0] == 10, "Must have 10 samples"
    print(f"PASS: risk={float(mean):.3f}, var={float(var):.5f}")
