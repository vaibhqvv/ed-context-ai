# ED-Context-AI — Cloud GPU Container Deployment

Minimal serving stack for GPU clouds that give you **only a shell inside a container** (RunPod, Lambda Cloud, Vast.ai, Paperspace Gradient, Modal-exec, etc.). No Docker-in-Docker, no Kubernetes — just `python` and an open port.

## Endpoints

| Method | Path         | Purpose                                               |
|--------|--------------|-------------------------------------------------------|
| GET    | `/health`    | Liveness probe                                        |
| GET    | `/ready`     | Readiness (model loaded, device, feature dim)         |
| POST   | `/v1/infer`  | Pure inference from 72-d context vector(s)            |
| POST   | `/v1/decide` | Full pipeline: risk + decision category + NLG explain |
| GET    | `/docs`      | Interactive OpenAPI (Swagger) UI                      |

## Deployment (copy-paste)

Once you've shelled into your GPU container:

```bash
# 1. Get the code
git clone <your-repo> ed-context-ai
cd ed-context-ai

# 2. Upload the trained model to outputs/models/best_model.pt
#    (scp / rsync / HTTP download — whatever your cloud supports)

# 3. One-shot bootstrap + launch
bash deploy/start.sh
```

The script:
1. `pip install -r deploy/requirements-serve.txt`
2. Verifies the checkpoint is present
3. Verifies CUDA is visible
4. Launches `uvicorn` on `0.0.0.0:$PORT` (default 8000)

Then point your cloud's HTTP tunnel / public URL at the container port.

## Environment variables

| Var                    | Default    | Notes                                               |
|------------------------|------------|-----------------------------------------------------|
| `PORT`                 | `8000`     | HTTP port to bind                                   |
| `HOST`                 | `0.0.0.0`  | Bind address (keep 0.0.0.0 in containers)           |
| `WORKERS`              | `1`        | Keep at 1 on GPU to avoid duplicate model copies    |
| `CUDA_VISIBLE_DEVICES` | `0`        | Which GPU to use                                    |
| `API_KEY`              | _(unset)_  | If set, clients must send `X-API-Key: <value>`      |
| `LOG_LEVEL`            | `INFO`     | Python logging level                                |

## Sample requests

### Pure inference (client already built the 72-d vector)

```bash
curl -X POST http://<host>:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"vector": [0.1, 0.2, ..., 0.3]}'   # 72 floats
```

Response:
```json
{
  "risk_probability": 0.72,
  "uncertainty_score": 0.004,
  "confidence_level": "high"
}
```

### Sequence inference (GRU consumes full temporal window sequence)

```bash
curl -X POST http://<host>:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"sequence": [[...72 floats...], [...72 floats...], [...72 floats...]]}'
```

### Full decision pipeline

```bash
curl -X POST http://<host>:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d @sample_context.json
```

Where `sample_context.json` matches `ContextObjectDTO` (see `deploy/schemas.py`).

Response includes the risk, decision category (`Escalate_Immediately`, `Monitor_and_Prepare`, etc.), urgency, recommended tests, and a natural-language explanation.

## With authentication

```bash
export API_KEY="$(openssl rand -hex 24)"
bash deploy/start.sh
```

Clients then send:
```bash
curl -H "X-API-Key: $API_KEY" ...
```

## Resource footprint

| Component            | Memory | Notes                                     |
|----------------------|--------|-------------------------------------------|
| GRU (FP32) on GPU    | ~4 MB  | 50-pass MC-Dropout still fits easily      |
| Context scalers/std  | <1 KB  | Held on device                            |
| FastAPI + uvicorn    | ~150 MB| Single process                            |
| **Total**            | ~200 MB| Fits on any GPU SKU (T4, L4, A10, 3090)   |

## Provider-specific notes

- **RunPod**: set "Container Start Command" to `bash deploy/start.sh`, expose TCP port 8000.
- **Lambda Cloud / Vast.ai**: same, use SSH + `nohup bash deploy/start.sh &` so it survives the session.
- **Modal / Beam / Replicate**: these are container-orchestrated platforms; if you have one of these, use their SDK instead of this script (they want a function decorator, not a bootstrap script).

## Health-check curl for load balancer

```bash
curl -fsS http://<host>:8000/ready
```
Returns HTTP 200 with `{"ready": true, "device": "cuda:0", ...}` when the model is loaded.

## Troubleshooting

| Symptom                                  | Fix                                                             |
|------------------------------------------|-----------------------------------------------------------------|
| `CUDA not available`                     | Container not launched with `--gpus all`; check provider docs   |
| `best_model.pt not found`                | Upload checkpoint before running `start.sh`                     |
| `ModuleNotFoundError: src...`            | Run from repo root (`start.sh` handles `PYTHONPATH`)            |
| OOM with many concurrent requests        | Drop `WORKERS=1`; scale out by spawning more containers         |
| Slow first request                       | Normal — MC-Dropout warm-up. Subsequent calls are fast          |
