"""Tests for the data preprocessing pipeline."""

import numpy as np
import pytest
from src.data.loader import generate_synthetic_cmapss
from src.data.preprocessor import CMAPSSPreprocessor


@pytest.fixture
def sample_data():
    train_df, test_df, test_rul = generate_synthetic_cmapss(n_engines=20, seed=0)
    return train_df, test_df, test_rul


def test_fit_transform_shape(sample_data):
    train_df, _, _ = sample_data
    prep = CMAPSSPreprocessor(window=10)
    result = prep.fit_transform(train_df)
    # Must retain unit_id, cycle, rul
    assert "unit_id" in result.columns
    assert "cycle" in result.columns
    assert "rul" in result.columns


def test_rolling_features_added(sample_data):
    train_df, _, _ = sample_data
    prep = CMAPSSPreprocessor(window=10)
    result = prep.fit_transform(train_df)
    # At least one rolling-mean column should exist
    rolling_cols = [c for c in result.columns if c.endswith("_mean")]
    assert len(rolling_cols) > 0


def test_scaler_normalises_to_01(sample_data):
    train_df, _, _ = sample_data
    prep = CMAPSSPreprocessor(window=10)
    result = prep.fit_transform(train_df)
    feature_cols = [c for c in result.columns if c.startswith("sensor_") or c.startswith("op_")]
    vals = result[feature_cols].values
    assert vals.min() >= -0.01  # allow tiny float tolerance
    assert vals.max() <= 1.01


def test_build_sequences_shape(sample_data):
    train_df, _, _ = sample_data
    prep = CMAPSSPreprocessor(window=10)
    result = prep.fit_transform(train_df)
    X, y = prep.build_sequences(result, target_col="rul")
    assert X.ndim == 3
    assert X.shape[1] == 10  # timesteps = window
    assert y.ndim == 1
    assert len(X) == len(y)


def test_build_last_sequences_shape(sample_data):
    train_df, test_df, _ = sample_data
    prep = CMAPSSPreprocessor(window=10)
    prep.fit_transform(train_df)
    test_proc = prep.transform(test_df)
    X_last = prep.build_last_sequences(test_proc)
    n_engines = test_df["unit_id"].nunique()
    assert X_last.shape == (n_engines, 10, X_last.shape[2])


def test_transform_before_fit_raises():
    prep = CMAPSSPreprocessor(window=10)
    _, test_df, _ = generate_synthetic_cmapss(n_engines=5, seed=1)
    with pytest.raises(RuntimeError):
        prep.transform(test_df)
