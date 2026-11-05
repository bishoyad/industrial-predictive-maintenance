"""
Anomaly detection module.

Uses Isolation Forest (unsupervised) on the engineered feature matrix
to flag unusual sensor behaviour that may precede failure.

Also includes a lightweight rolling-z-score alarm for per-sensor spike
detection — useful for real-time streaming data.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for engine sensor readings.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalous samples in the training data.
        Tune this based on domain knowledge (5–10% is a reasonable start).
    n_estimators : int
        Number of isolation trees.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._scaler = StandardScaler()

    # ---- Fit & predict ----------------------------------------------------

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """
        Fit on the training feature matrix.
        X can be 2-D [samples, features] or 3-D [samples, timesteps, features]
        (flattened automatically).
        """
        X_flat = self._flatten(X)
        X_scaled = self._scaler.fit_transform(X_flat)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        logger.info("AnomalyDetector fitted on %d samples.", len(X_flat))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns binary array: 1 = anomaly, 0 = normal.
        (Isolation Forest returns -1 for anomalies; we flip to 0/1.)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(self._flatten(X))
        raw = self._model.predict(X_scaled)       # -1 or +1
        return np.where(raw == -1, 1, 0).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw anomaly scores (higher → more anomalous).
        Scores are negated decision_function values so higher = worse.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(self._flatten(X))
        return -self._model.decision_function(X_scaled)

    # ---- Persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "scaler": self._scaler}, path)
        logger.info("AnomalyDetector saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        obj = cls.__new__(cls)
        payload = joblib.load(path)
        obj._model = payload["model"]
        obj._scaler = payload["scaler"]
        logger.info("AnomalyDetector loaded ← %s", path)
        return obj

    # ---- Private helpers --------------------------------------------------

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1) if X.ndim == 3 else X

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Detector not fitted yet. Call fit() first.")


# ---------------------------------------------------------------------------
# Rolling Z-score alarm (streaming / per-sensor)
# ---------------------------------------------------------------------------

def rolling_zscore_alarm(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 30,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Flag sensor readings that deviate more than `threshold` standard
    deviations from their rolling mean.

    Returns the input DataFrame with an additional boolean 'alarm' column
    and per-sensor z-score columns.
    """
    df = df.copy()
    alarm = pd.Series(False, index=df.index)

    for col in sensor_cols:
        if col not in df.columns:
            continue
        rolling = df.groupby("unit_id")[col].transform(
            lambda s: s.rolling(window, min_periods=1)
        )
        mean = df.groupby("unit_id")[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        std = df.groupby("unit_id")[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(1e-9)
        )
        zscore = (df[col] - mean) / std
        df[f"{col}_zscore"] = zscore
        alarm |= zscore.abs() > threshold

    df["alarm"] = alarm
    return df
