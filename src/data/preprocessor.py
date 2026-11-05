"""
Feature engineering and preprocessing pipeline for C-MAPSS sensor data.

Steps:
  1. Rolling-window statistics (mean, std) over a configurable window.
  2. Min-Max normalisation fitted on training data.
  3. Sequence building for LSTM input (shape: [samples, timesteps, features]).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Sensor columns retained after dropping low-variance ones (see loader.py)
SENSOR_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
FEATURE_COLS = OP_COLS + SENSOR_COLS


class CMAPSSPreprocessor:
    """
    Fits on training data and transforms both train and test DataFrames
    into ML-ready feature matrices.

    Parameters
    ----------
    window : int
        Number of past cycles used to compute rolling statistics.
        Also the LSTM timestep length.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self.scaler = MinMaxScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling features, fit the scaler, and return the
        normalised training DataFrame (still contains 'unit_id', 'cycle', 'rul').
        """
        df = self._add_rolling_features(df)
        feature_cols = self._get_all_feature_cols(df)
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        self._fitted = True
        logger.info("Preprocessor fitted on %d training samples.", len(df))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using the already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        df = self._add_rolling_features(df)
        feature_cols = self._get_all_feature_cols(df)
        df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def build_sequences(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = "rul",
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert a preprocessed DataFrame into 3-D LSTM sequences.

        Returns
        -------
        X : ndarray of shape (n_samples, window, n_features)
        y : ndarray of shape (n_samples,)  — None if target_col missing.
        """
        feature_cols = self._get_all_feature_cols(df)
        X_list, y_list = [], []

        for uid, group in df.groupby("unit_id"):
            group = group.sort_values("cycle").reset_index(drop=True)
            values = group[feature_cols].values
            targets = group[target_col].values if target_col in group.columns else None

            for i in range(len(group) - self.window + 1):
                X_list.append(values[i : i + self.window])
                if targets is not None:
                    y_list.append(targets[i + self.window - 1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32) if y_list else None
        logger.debug("Built sequences: X=%s, y=%s", X.shape, y.shape if y is not None else None)
        return X, y

    def build_last_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        For the test set, return only the LAST window of each engine.
        Returns ndarray of shape (n_engines, window, n_features).
        """
        feature_cols = self._get_all_feature_cols(df)
        X_list = []

        for uid, group in df.groupby("unit_id"):
            group = group.sort_values("cycle").reset_index(drop=True)
            values = group[feature_cols].values
            if len(values) >= self.window:
                X_list.append(values[-self.window :])
            else:
                # Pad with zeros on the left if engine has fewer cycles
                pad = np.zeros((self.window - len(values), len(feature_cols)), dtype=np.float32)
                X_list.append(np.vstack([pad, values]))

        return np.array(X_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-engine rolling mean and std for every sensor column."""
        df = df.copy()
        new_cols = {}
        for col in SENSOR_COLS:
            if col not in df.columns:
                continue
            rolled = df.groupby("unit_id")[col].transform(
                lambda s: s.rolling(self.window, min_periods=1)
            )
            new_cols[f"{col}_mean"] = rolled.mean() if hasattr(rolled, "mean") else rolled
            # Rolling returns a GroupBy-transformed series — apply correctly
            new_cols[f"{col}_mean"] = df.groupby("unit_id")[col].transform(
                lambda s: s.rolling(self.window, min_periods=1).mean()
            )
            new_cols[f"{col}_std"] = df.groupby("unit_id")[col].transform(
                lambda s: s.rolling(self.window, min_periods=1).std().fillna(0)
            )
        for name, series in new_cols.items():
            df[name] = series
        return df

    def _get_all_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return all feature columns present in df (base + rolling)."""
        rolling_mean = [f"{c}_mean" for c in SENSOR_COLS if f"{c}_mean" in df.columns]
        rolling_std = [f"{c}_std" for c in SENSOR_COLS if f"{c}_std" in df.columns]
        base = [c for c in FEATURE_COLS if c in df.columns]
        return base + rolling_mean + rolling_std
