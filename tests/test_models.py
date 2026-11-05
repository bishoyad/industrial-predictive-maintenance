"""Tests for RUL and anomaly detection models."""

import numpy as np
import pytest
from src.data.loader import generate_synthetic_cmapss
from src.data.preprocessor import CMAPSSPreprocessor
from src.models.rul import XGBoostRUL
from src.models.anomaly import AnomalyDetector
from src.utils.metrics import evaluate_rul, evaluate_anomaly


@pytest.fixture(scope="module")
def prepared_data():
    train_df, test_df, test_rul = generate_synthetic_cmapss(n_engines=30, seed=42)
    prep = CMAPSSPreprocessor(window=10)
    train_proc = prep.fit_transform(train_df)
    test_proc = prep.transform(test_df)
    X_train, y_train = prep.build_sequences(train_proc, target_col="rul")
    X_last = prep.build_last_sequences(test_proc)
    return X_train, y_train, X_last, test_rul, prep


def test_xgboost_rul_fit_predict(prepared_data):
    X_train, y_train, X_last, test_rul, _ = prepared_data
    model = XGBoostRUL(n_estimators=50)
    model.fit(X_train, y_train)
    preds = model.predict(X_last)
    assert len(preds) == len(test_rul)
    # All predictions must be non-negative
    assert (preds >= 0).all()


def test_xgboost_rul_metrics_reasonable(prepared_data):
    X_train, y_train, X_last, test_rul, _ = prepared_data
    model = XGBoostRUL(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_last)
    metrics = evaluate_rul(test_rul.values, preds)
    # RMSE should be finite and positive
    assert 0 < metrics["rmse"] < 1e6
    assert metrics["r2"] is not None


def test_anomaly_detector_fit_predict(prepared_data):
    X_train, y_train, _, _, _ = prepared_data
    detector = AnomalyDetector(contamination=0.1, n_estimators=50)
    detector.fit(X_train)
    flags = detector.predict(X_train)
    assert flags.shape == (len(X_train),)
    assert set(flags).issubset({0, 1})


def test_anomaly_score_range(prepared_data):
    X_train, y_train, _, _, _ = prepared_data
    detector = AnomalyDetector(contamination=0.1, n_estimators=50)
    detector.fit(X_train)
    scores = detector.anomaly_score(X_train)
    assert len(scores) == len(X_train)
    # Scores should be finite
    assert np.all(np.isfinite(scores))


def test_evaluate_rul_metrics_keys():
    y_true = np.array([50, 30, 100, 10])
    y_pred = np.array([55, 25, 90, 15])
    metrics = evaluate_rul(y_true, y_pred)
    for key in ["rmse", "mae", "nasa_score", "r2"]:
        assert key in metrics


def test_evaluate_anomaly_metrics_keys():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    metrics = evaluate_anomaly(y_true, scores)
    assert "auroc" in metrics
    assert "avg_precision" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
