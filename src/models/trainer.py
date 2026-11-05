"""
End-to-end training pipeline.

Usage (from project root):
    python -m src.models.trainer --model xgboost --subset FD001 --synthetic

The --synthetic flag generates a small synthetic dataset so you can
verify the full pipeline runs correctly without downloading the real data.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from src.data.loader import load_cmapss, generate_synthetic_cmapss
from src.data.preprocessor import CMAPSSPreprocessor
from src.models.anomaly import AnomalyDetector
from src.models.rul import XGBoostRUL, LSTMRul
from src.utils.metrics import evaluate_rul, evaluate_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def train(
    model_type: str = "xgboost",
    subset: str = "FD001",
    window: int = 30,
    synthetic: bool = False,
    rul_clip: int = 125,
):
    """
    Full training run:
      1. Load data (real or synthetic).
      2. Feature engineering + normalisation.
      3. Train RUL model.
      4. Train anomaly detector.
      5. Evaluate both models on the test set.
      6. Persist all artefacts.
    """
    logger.info("=" * 60)
    logger.info("Industrial Predictive Maintenance — Training Pipeline")
    logger.info("Model: %s | Subset: %s | Synthetic: %s", model_type, subset, synthetic)
    logger.info("=" * 60)

    # --- 1. Load data ---------------------------------------------------
    if synthetic:
        train_df, test_df, test_rul = generate_synthetic_cmapss(n_engines=80, seed=42, rul_clip=rul_clip)
    else:
        train_df, test_df, test_rul = load_cmapss(subset=subset, rul_clip=rul_clip)

    # --- 2. Preprocessing -----------------------------------------------
    preprocessor = CMAPSSPreprocessor(window=window)
    train_proc = preprocessor.fit_transform(train_df)
    test_proc = preprocessor.transform(test_df)

    X_train, y_train = preprocessor.build_sequences(train_proc, target_col="rul")
    X_test_last = preprocessor.build_last_sequences(test_proc)

    logger.info("X_train: %s | y_train: %s", X_train.shape, y_train.shape)
    logger.info("X_test_last: %s", X_test_last.shape)

    # --- 3. Train RUL model --------------------------------------------
    if model_type == "xgboost":
        rul_model = XGBoostRUL()
    elif model_type == "lstm":
        rul_model = LSTMRul(epochs=30, patience=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'xgboost' or 'lstm'.")

    rul_model.fit(X_train, y_train)
    rul_preds = rul_model.predict(X_test_last)
    rul_metrics = evaluate_rul(test_rul.values, rul_preds)
    logger.info("RUL Metrics: %s", rul_metrics)

    # --- 4. Train anomaly detector ------------------------------------
    # Use train sequences where RUL > 50 as "healthy" examples
    healthy_mask = y_train > 50
    X_healthy = X_train[healthy_mask]
    anomaly_detector = AnomalyDetector(contamination=0.05)
    anomaly_detector.fit(X_healthy)

    # Evaluate anomaly detection: near-failure cycles (RUL ≤ 30) should be flagged
    all_preds_flat = X_train.reshape(len(X_train), -1)
    all_scores = anomaly_detector.anomaly_score(X_train)
    anomaly_labels = (y_train <= 30).astype(int)
    anomaly_metrics = evaluate_anomaly(anomaly_labels, all_scores)
    logger.info("Anomaly Metrics: %s", anomaly_metrics)

    # --- 5. Save artefacts -------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)

    model_suffix = f"{model_type}_{subset}"
    rul_model.save(MODELS_DIR / f"rul_{model_suffix}.joblib" if model_type == "xgboost"
                   else MODELS_DIR / f"rul_{model_suffix}.pt")
    anomaly_detector.save(MODELS_DIR / f"anomaly_{subset}.joblib")

    # Save preprocessor
    with open(MODELS_DIR / f"preprocessor_{subset}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Save metrics for the dashboard / README badges
    metrics_path = MODELS_DIR / "metrics.json"
    all_metrics = {
        "rul": rul_metrics,
        "anomaly": anomaly_metrics,
        "config": {
            "model_type": model_type,
            "subset": subset,
            "window": window,
            "rul_clip": rul_clip,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("All artefacts saved to '%s/'", MODELS_DIR)
    logger.info("Done! ✅")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lstm"])
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no download required)")
    parser.add_argument("--rul-clip", type=int, default=125)
    args = parser.parse_args()

    train(
        model_type=args.model,
        subset=args.subset,
        window=args.window,
        synthetic=args.synthetic,
        rul_clip=args.rul_clip,
    )
