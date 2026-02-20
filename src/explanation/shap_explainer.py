import shap, torch, numpy as np, pickle
from pathlib import Path
from src.model.inference import load_model
from src.utils.config_loader import get_config

cfg = get_config()
_EXPLAINER = None


def get_explainer(n_background=100):
    global _EXPLAINER
    if _EXPLAINER is not None:
        return _EXPLAINER
    model = load_model()
    model.eval()
    with open(Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb") as f:
        ctxs = pickle.load(f)
    bg = np.array(
        [
            c.context_vector
            for c in ctxs[:n_background]
            if not any(np.isnan(v) for v in c.context_vector)
        ],
        dtype=np.float32,
    )

    def predict(x_np):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(x_np, dtype=torch.float32)).numpy()

    _EXPLAINER = shap.KernelExplainer(predict, shap.sample(bg, 50))
    return _EXPLAINER


def compute_shap_values(context_vector, feature_names) -> dict:
    explainer = get_explainer()
    x = np.array([context_vector], dtype=np.float32)
    vals = explainer.shap_values(x, nsamples=100, silent=True)[0]
    result = {name: float(v) for name, v in zip(feature_names, vals)}
    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
