"""
Evaluation metrics for RUL and anomaly detection.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
)


def evaluate_rul(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard RUL regression metrics.

    Returns
    -------
    dict with keys: rmse, mae, score (NASA scoring function), r2.

    NASA Scoring Function:
        Penalises late predictions more heavily than early ones.
        sum(exp(-d/13) - 1  if d < 0 else  exp(d/10) - 1)
        where d = y_pred - y_true
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # NASA asymmetric scoring function
    d = y_pred - y_true
    nasa_score = float(np.sum(
        np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    ))

    # Coefficient of determination
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "nasa_score": round(nasa_score, 2), "r2": round(r2, 4)}


def evaluate_anomaly(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Evaluate anomaly detector using AUROC and Average Precision.

    Parameters
    ----------
    y_true  : binary labels (1 = anomaly / near-failure).
    scores  : raw anomaly scores (higher → more anomalous).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "avg_precision": None, "note": "Only one class present"}

    auroc = float(roc_auc_score(y_true, scores))
    avg_precision = float(average_precision_score(y_true, scores))

    return {
        "auroc": round(auroc, 4),
        "avg_precision": round(avg_precision, 4),
    }
