import torch, numpy as np
from pathlib import Path
from src.model.mc_dropout import MCDropoutNet
from src.utils.config_loader import get_config
from src.utils.device import get_device

cfg = get_config()
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    device = get_device()
    ckpt = torch.load(
        Path(cfg["paths"]["model_output"]) / "best_model.pt", map_location=device
    )
    model = MCDropoutNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    _MODEL = model
    return model


def predict(context) -> dict:
    model = load_model()
    device = next(model.parameters()).device
    x = torch.tensor(np.array(context.context_vector, dtype=np.float32)).unsqueeze(0).to(device)
    mean, var, _ = model.predict_with_uncertainty(x)
    risk = float(mean.item())
    unc = float(var.item())
    return {
        "risk_probability": risk,
        "uncertainty_score": unc,
        "confidence_level": model.classify_uncertainty(unc),
    }
