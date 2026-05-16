"""
FastAPI inference service for the Email Timing Response DQN agent.

Endpoints:
  GET  /health              — liveness probe
  POST /predict             — run agent inference on a single email
  GET  /model/version       — list saved model versions
  POST /model/version/{tag} — hot-swap to a different saved model
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.api.response_routes import router as response_router
from app.schemas import (
    EmailRequest,
    HealthResponse,
    ModelVersionResponse,
    PredictionResponse,
)
from config import Config
from data.email_data import Email
from monitoring.logging_config import get_logger
from monitoring.metrics import MetricsTracker

# Ensure project root is on the path when running from inside app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = get_logger(__name__)
metrics = MetricsTracker()

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Timing Response — Inference API",
    description="DQN-based email action recommender.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include response generation routes ────────────────────────────────────────
app.include_router(response_router)

# ── State shared across requests ─────────────────────────────────────────────

_start_time: float = time.time()
_agent = None
_model_version: str = "unknown"

ACTION_LABELS = {
    0: "reply_now",
    1: "delay_reply",
    2: "mark_important",
    3: "archive",
}


def _load_agent(weights_path: Optional[str] = None, pkl_path: Optional[str] = None):
    """Load (or reload) the DQN agent. Imports deferred to avoid circular deps."""
    from agent.dqn import DQNAgent, QNetwork  # noqa: F401 — needed for pickle

    wp = weights_path or Config.DQN_WEIGHTS_PATH

    if os.path.exists(wp):
        ckpt = torch.load(wp, map_location="cpu", weights_only=False)
        agent = DQNAgent(
            state_size=ckpt.get("state_size", Config.STATE_SIZE),
            action_size=ckpt.get("action_size", Config.ACTION_SIZE),
        )
        agent._policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent._policy_net.eval()
        version = ckpt.get("version", os.path.getmtime(wp))
        logger.info("Agent loaded from weights: %s", wp)
        return agent, str(version)

    raise FileNotFoundError(
        f"No model weights found at {wp}. "
        "Train the model first (train_local_dqn.py or Colab notebook)."
    )


# ── Startup / shutdown ────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    global _agent, _model_version
    try:
        _agent, _model_version = _load_agent()
        logger.info("Inference service started. Model version: %s", _model_version)
    except FileNotFoundError as exc:
        logger.warning("Could not load model on startup: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok" if _agent is not None else "degraded",
        model_loaded=_agent is not None,
        model_version=_model_version,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: EmailRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    email = Email(
        subject=request.subject,
        sender=request.sender,
        priority=request.priority,
        sender_importance=request.sender_importance,
        waiting_time=request.waiting_time,
        workload=request.workload,
        time_of_day=request.time_of_day,
    )

    state = email.to_state_vector()

    # Greedy action (epsilon=0 for inference)
    with torch.no_grad():
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = _agent._policy_net(t).squeeze(0).numpy()

    action_id = int(np.argmax(q_values))
    # Softmax over Q-values as a proxy confidence score
    exp_q = np.exp(q_values - q_values.max())
    confidence = float(exp_q[action_id] / exp_q.sum())

    metrics.record_prediction(action_id, confidence)
    logger.info(
        "Prediction: email=%r action=%s confidence=%.3f",
        request.subject[:40],
        ACTION_LABELS[action_id],
        confidence,
    )

    return PredictionResponse(
        action_id=action_id,
        action_label=ACTION_LABELS[action_id],
        confidence=round(confidence, 4),
        state_vector=state.tolist(),
        model_version=_model_version,
    )


@app.get("/model/version", response_model=ModelVersionResponse, tags=["ops"])
def model_version():
    models_dir = Config.MODEL_DIR
    versions = []
    if os.path.isdir(models_dir):
        versions = sorted(
            [f for f in os.listdir(models_dir) if f.endswith((".pt", ".pkl"))],
        )
    return ModelVersionResponse(
        current_version=_model_version,
        available_versions=versions,
        weights_path=Config.DQN_WEIGHTS_PATH,
        pkl_path=Config.DQN_MODEL_PATH,
    )


@app.post("/model/version/{filename}", tags=["ops"])
def swap_model(filename: str):
    """Hot-swap the active model to a different checkpoint file."""
    global _agent, _model_version
    path = os.path.join(Config.MODEL_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {path}")

    try:
        if filename.endswith(".pt"):
            _agent, _model_version = _load_agent(weights_path=path)
        else:
            _agent, _model_version = _load_agent(pkl_path=path)
        logger.info("Model swapped to: %s (version=%s)", filename, _model_version)
        return {"status": "ok", "loaded": filename, "version": _model_version}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
