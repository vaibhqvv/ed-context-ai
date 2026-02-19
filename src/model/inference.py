import torch, numpy as np
from pathlib import Path
from src.model.mc_dropout import MCDropoutNet
from src.utils.config_loader import get_config

cfg = get_config()
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    ckpt = torch.load(
        Path(cfg["paths"]["model_output"]) / "best_model.pt", map_location="cpu"
    )
    model = MCDropoutNet(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    _MODEL = model
    return model


def predict(context) -> dict:
    model = load_model()
    x = torch.tensor(np.array(context.context_vector, dtype=np.float32)).unsqueeze(0)
    mean, var, _ = model.predict_with_uncertainty(x)
    risk = float(mean.item())
    unc = float(var.item())
    return {
        "risk_probability": risk,
        "uncertainty_score": unc,
        "confidence_level": model.classify_uncertainty(unc),
    }
