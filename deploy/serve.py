"""FastAPI inference server for ED-Context-AI.

Exposes HTTP endpoints for:
  - /health      liveness
  - /ready       readiness (model + device)
  - /v1/infer    pure inference from 72-d context vector(s)
  - /v1/decide   full pipeline: risk + decision + NLG explanation

Designed for GPU-container clouds where only a shell + open port is available.
Start with:  python -m deploy.serve
Or:          uvicorn deploy.serve:app --host 0.0.0.0 --port $PORT --workers 1
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import JSONResponse

# Make src/ importable regardless of launch directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.schema import ContextObject  # noqa: E402
from src.model.inference import (  # noqa: E402
    load_model,
    get_model_type,
    get_feature_stats,
)
from src.decision.engine import run_decision_engine  # noqa: E402
from src.explanation.generator import generate_explanation  # noqa: E402

from deploy.schemas import (  # noqa: E402
    ContextObjectDTO,
    DecisionResponse,
    HealthResponse,
    ReadyResponse,
    RiskResponse,
    VectorInferenceRequest,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ed-context-ai.serve")

API_KEY = os.environ.get("API_KEY")  # optional; if set, X-API-Key required


def _check_auth(x_api_key: str | None) -> None:
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up model...")
    model = load_model()
    device = next(model.parameters()).device
    log.info("Model ready on device=%s type=%s", device, get_model_type())
    yield
    log.info("Shutdown")


app = FastAPI(
    title="ED-Context-AI Inference API",
    version="1.0.0",
    description="Uncertainty-aware ED admission prediction with MC-Dropout GRU.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    try:
        model = load_model()
        device = str(next(model.parameters()).device)
        mean, _ = get_feature_stats()
        return ReadyResponse(
            ready=True,
            device=device,
            model_type=get_model_type(),
            input_dim=int(mean.numel()),
        )
    except Exception as exc:  # pragma: no cover - surfaced at /ready
        log.exception("readiness check failed")
        return JSONResponse(status_code=503, content={"ready": False, "error": str(exc)})


def _infer_sequence(vectors: List[List[float]]) -> RiskResponse:
    model = load_model()
    device = next(model.parameters()).device
    mean, std = get_feature_stats()

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected 2-D [seq_len, feat_dim] array")
    if arr.shape[1] != mean.numel():
        raise HTTPException(
            status_code=400,
            detail=f"Feature dim mismatch: got {arr.shape[1]}, model expects {mean.numel()}",
        )

    x = torch.from_numpy(arr).to(device)
    x = (x - mean) / std
    x = x.unsqueeze(0)  # (1, seq_len, feat_dim)
    lengths = torch.tensor([x.shape[1]], dtype=torch.long)

    if get_model_type() == "gru":
        risk_mean, risk_var, _ = model.predict_with_uncertainty(x, lengths=lengths)
    else:
        # MLP path: aggregate by taking the last window
        risk_mean, risk_var, _ = model.predict_with_uncertainty(x.squeeze(0)[-1:])

    unc = float(risk_var.item())
    return RiskResponse(
        risk_probability=float(risk_mean.item()),
        uncertainty_score=unc,
        confidence_level=model.classify_uncertainty(unc),
    )


@app.post("/v1/infer", response_model=RiskResponse)
def infer(req: VectorInferenceRequest, x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)
    if req.sequence is None and req.vector is None:
        raise HTTPException(status_code=400, detail="Provide 'vector' or 'sequence'")
    if req.sequence is not None and req.vector is not None:
        raise HTTPException(status_code=400, detail="Provide only one of 'vector' or 'sequence'")

    vectors = req.sequence if req.sequence is not None else [req.vector]
    return _infer_sequence(vectors)


@app.post("/v1/decide", response_model=DecisionResponse)
def decide(req: ContextObjectDTO, x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)

    ctx = ContextObject(
        patient_id=req.patient_id,
        stay_id=req.stay_id,
        window_index=req.window_index,
        physiological_state=req.physiological_state,
        trends=req.trends,
        slopes=req.slopes,
        data_confidence=req.data_confidence,
        observation_density=req.observation_density,
        missing_critical=req.missing_critical,
        deterioration_flag=req.deterioration_flag,
        acuity=req.acuity,
        sepsis_flag=req.sepsis_flag,
        n_medications=req.n_medications,
        context_vector=req.context_vector,
    )

    risk_resp = _infer_sequence([ctx.context_vector])
    dec = run_decision_engine(ctx, risk_resp.risk_probability, risk_resp.confidence_level)
    expl = generate_explanation(
        context=ctx,
        risk_probability=risk_resp.risk_probability,
        confidence_level=risk_resp.confidence_level,
        uncertainty_score=risk_resp.uncertainty_score,
        decision_category=dec["decision_category"],
        reasoning_tags=dec["reasoning_tags"],
        shap_values=None,
    )

    return DecisionResponse(
        patient_id=ctx.patient_id,
        stay_id=ctx.stay_id,
        window_index=ctx.window_index,
        risk_probability=risk_resp.risk_probability,
        uncertainty_score=risk_resp.uncertainty_score,
        confidence_level=risk_resp.confidence_level,
        decision_category=dec["decision_category"],
        urgency=dec["urgency"],
        recommended_tests=dec["recommended_tests"],
        explanation=expl,
        context_summary={
            "physiological_state": ctx.physiological_state,
            "dominant_trends": [
                f for f, t in ctx.trends.items()
                if t in ("rising_fast", "falling_fast", "rising", "falling")
            ][:5],
            "data_confidence": ctx.data_confidence,
            "missing_critical": ctx.missing_critical,
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", "1"))  # keep 1 for GPU to avoid duplicate model loads
    uvicorn.run("deploy.serve:app", host=host, port=port, workers=workers, log_level="info")
