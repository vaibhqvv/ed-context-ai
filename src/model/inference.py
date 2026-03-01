import torch, numpy as np
from pathlib import Path
from src.model.mc_dropout import MCDropoutNet
from src.model.mc_dropout_gru import MCDropoutGRU
from src.utils.config_loader import get_config
from src.utils.device import get_device

cfg = get_config()
_MODEL = None
_FEATURE_MEAN = None
_FEATURE_STD = None
_MODEL_TYPE = None


def load_model():
    global _MODEL, _FEATURE_MEAN, _FEATURE_STD, _MODEL_TYPE
    if _MODEL is not None:
        return _MODEL
    device = get_device()
    ckpt = torch.load(
        Path(cfg["paths"]["model_output"]) / "best_model.pt", map_location=device
    )
    model_type = ckpt.get("model_type", "mlp")
    _MODEL_TYPE = model_type
    if model_type == "gru":
        model = MCDropoutGRU(input_dim=ckpt["input_dim"])
    else:
        model = MCDropoutNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    _FEATURE_MEAN = ckpt["feature_mean"].to(device)
    _FEATURE_STD = ckpt["feature_std"].to(device)
    _MODEL = model
    return model


def get_model_type():
    load_model()
    return _MODEL_TYPE


def predict(context) -> dict:
    """Predict for a single context object (works for both MLP and GRU)."""
    model = load_model()
    device = next(model.parameters()).device
    x = torch.tensor(np.array(context.context_vector, dtype=np.float32)).unsqueeze(0).to(device)
    x = (x - _FEATURE_MEAN) / _FEATURE_STD
    if _MODEL_TYPE == "gru":
        # GRU expects (batch, seq_len, input_dim); treat single context as length-1 sequence
        x = x.unsqueeze(1)  # (1, 1, input_dim)
        lengths = torch.tensor([1], dtype=torch.long)
        mean, var, _ = model.predict_with_uncertainty(x, lengths=lengths)
    else:
        mean, var, _ = model.predict_with_uncertainty(x)
    risk = float(mean.item())
    unc = float(var.item())
    return {
        "risk_probability": risk,
        "uncertainty_score": unc,
        "confidence_level": model.classify_uncertainty(unc),
    }


def predict_sequence(context_list) -> dict:
    """Predict for a sequence of context objects (GRU path).

    Args:
        context_list: List of context objects for one stay, sorted by window_index.

    Returns:
        dict with risk_probability, uncertainty_score, confidence_level.
    """
    model = load_model()
    device = next(model.parameters()).device
    vecs = [np.array(ctx.context_vector, dtype=np.float32) for ctx in context_list]
    seq = torch.tensor(np.stack(vecs), dtype=torch.float32).to(device)
    seq = (seq - _FEATURE_MEAN) / _FEATURE_STD
    # Add batch dimension: (1, seq_len, feat_dim)
    seq = seq.unsqueeze(0)
    lengths = torch.tensor([seq.shape[1]], dtype=torch.long)
    mean, var, _ = model.predict_with_uncertainty(seq, lengths=lengths)
    risk = float(mean.item())
    unc = float(var.item())
    return {
        "risk_probability": risk,
        "uncertainty_score": unc,
        "confidence_level": model.classify_uncertainty(unc),
    }
