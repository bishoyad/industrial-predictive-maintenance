"""
FastAPI application — Predictive Maintenance REST API.

Endpoints:
  GET  /                  → welcome
  GET  /health            → liveness probe
  POST /predict/rul       → predict Remaining Useful Life
  POST /predict/anomaly   → detect anomalies in sensor readings
  GET  /metrics           → latest model evaluation metrics

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_PATH = Path("models/metrics.json")

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "REST API for real-time Remaining Useful Life (RUL) prediction "
        "and anomaly detection on turbofan engine sensor data (NASA C-MAPSS).\n\n"
        "**GitHub:** https://github.com/bishoyad/industrial-predictive-maintenance"
    ),
    version="1.0.0",
    contact={"name": "Beshoi Gaber", "url": "https://github.com/bishoyad"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Industrial Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metrics", tags=["Model Metrics"])
def get_metrics():
    """Return the latest evaluation metrics for both models."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run the training pipeline first: "
                   "python -m src.models.trainer --synthetic",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)
