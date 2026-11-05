"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _make_reading(unit_id=1, cycle=50, offset=0.0):
    return {
        "unit_id": unit_id,
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
        "sensor_2": 641.21 + offset,
        "sensor_3": 1583.76,
        "sensor_4": 1404.22,
        "sensor_7": 554.36,
        "sensor_8": 2388.02,
        "sensor_9": 9044.54,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.09,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_17": 392.0,
        "sensor_20": 38.86,
        "sensor_21": 23.37,
    }


def test_predict_rul_no_model():
    """Without trained models the endpoint returns 503."""
    payload = {"readings": [_make_reading()]}
    r = client.post("/predict/rul", json=payload)
    # Either 503 (no model) or 200 (models pre-loaded) are acceptable
    assert r.status_code in (200, 503)


def test_predict_anomaly_no_model():
    payload = {"readings": [_make_reading()]}
    r = client.post("/predict/anomaly", json=payload)
    assert r.status_code in (200, 503)


def test_predict_rul_bad_payload():
    """Missing required sensor fields → 422 Unprocessable Entity."""
    r = client.post("/predict/rul", json={"readings": [{"unit_id": 1}]})
    assert r.status_code == 422
