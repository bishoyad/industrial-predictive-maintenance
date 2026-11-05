"""
Prediction endpoints — /predict/rul and /predict/anomaly.

Models are loaded lazily at first request and cached in module state.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import RULRequest, RULResponse, AnomalyRequest, AnomalyResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MODELS_DIR = Path("models")

# --- Lazy-loaded singletons ------------------------------------------------
_preprocessor = None
_rul_model = None
_anomaly_detector = None


def _load_models():
    global _preprocessor, _rul_model, _anomaly_detector

    # Preprocessor
    prep_candidates = list(MODELS_DIR.glob("preprocessor_*.pkl"))
    if prep_candidates and _preprocessor is None:
        with open(prep_candidates[0], "rb") as f:
            _preprocessor = pickle.load(f)
        logger.info("Preprocessor loaded.")

    # RUL model — prefer XGBoost, fall back to LSTM
    if _rul_model is None:
        from src.models.rul import XGBoostRUL, LSTMRul
        xgb_candidates = list(MODELS_DIR.glob("rul_xgboost_*.joblib"))
        lstm_candidates = list(MODELS_DIR.glob("rul_lstm_*.pt"))
        if xgb_candidates:
            _rul_model = XGBoostRUL.load(xgb_candidates[0])
            logger.info("XGBoost RUL model loaded.")
        elif lstm_candidates:
            _rul_model = LSTMRul.load(lstm_candidates[0])
            logger.info("LSTM RUL model loaded.")

    # Anomaly detector
    if _anomaly_detector is None:
        from src.models.anomaly import AnomalyDetector
        anom_candidates = list(MODELS_DIR.glob("anomaly_*.joblib"))
        if anom_candidates:
            _anomaly_detector = AnomalyDetector.load(anom_candidates[0])
            logger.info("Anomaly detector loaded.")


def _readings_to_df(readings) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in readings])


def _health_label(rul: float) -> str:
    if rul > 80:
        return "HEALTHY"
    if rul > 30:
        return "WARNING"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# POST /predict/rul
# ---------------------------------------------------------------------------

@router.post("/rul", response_model=RULResponse)
def predict_rul(request: RULRequest):
    """
    Predict Remaining Useful Life (RUL) for an engine given its recent
    sensor history.

    **Minimum 1 reading required.** For best accuracy provide at least
    30 consecutive cycles (matching the model's training window).
    """
    _load_models()

    if _preprocessor is None or _rul_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run the training pipeline first: "
                   "python -m src.models.trainer --synthetic",
        )

    df = _readings_to_df(request.readings)
    unit_id = int(df["unit_id"].iloc[0])

    try:
        df_proc = _preprocessor.transform(df)
        X = _preprocessor.build_last_sequences(df_proc)
        rul_raw = float(_rul_model.predict(X)[0])
        rul = round(max(0.0, rul_raw), 1)
    except Exception as exc:
        logger.exception("RUL prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return RULResponse(
        unit_id=unit_id,
        predicted_rul=rul,
        health_status=_health_label(rul),
        confidence_pct=None,  # extend with conformal prediction later
    )


# ---------------------------------------------------------------------------
# POST /predict/anomaly
# ---------------------------------------------------------------------------

@router.post("/anomaly", response_model=list[AnomalyResponse])
def predict_anomaly(request: AnomalyRequest):
    """
    Detect anomalous sensor readings in the provided sequence.
    Returns one result per reading.
    """
    _load_models()

    if _preprocessor is None or _anomaly_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run training pipeline first.",
        )

    df = _readings_to_df(request.readings)

    try:
        df_proc = _preprocessor.transform(df)
        X = _preprocessor.build_last_sequences(df_proc)
        flags = _anomaly_detector.predict(X)
        scores = _anomaly_detector.anomaly_score(X)
    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=str(exc))

    results = []
    for i, reading in enumerate(request.readings[-len(flags):]):
        flag = bool(flags[i])
        score = round(float(scores[i]), 4)
        results.append(AnomalyResponse(
            unit_id=reading.unit_id,
            cycle=reading.cycle,
            is_anomaly=flag,
            anomaly_score=score,
            message="⚠️ Anomalous sensor pattern detected!" if flag else "✅ Normal operation",
        ))

    return results
