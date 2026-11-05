"""
Data loader for the NASA C-MAPSS Turbofan Engine Degradation dataset.

Dataset source:
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  Also available on Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

The dataset contains 4 sub-datasets (FD001–FD004).
Each row = one operational cycle of one engine unit.

Columns (original, no header):
  unit_id, cycle, op_setting_1..3, sensor_1..21
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Official column names for C-MAPSS data files
_COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that carry near-zero variance across all operating conditions
# (i.e. constant readings — useless for ML)
_LOW_VARIANCE_SENSORS = [1, 5, 6, 10, 16, 18, 19]
_DROP_COLS = [f"sensor_{s}" for s in _LOW_VARIANCE_SENSORS]


def _read_cmapss_file(filepath: Path) -> pd.DataFrame:
    """Read a single C-MAPSS text file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=_COLUMN_NAMES,
        engine="python",
    )
    df.drop(columns=_DROP_COLS, inplace=True)
    return df


def load_cmapss(
    data_dir: str | Path = "data/raw",
    subset: str = "FD001",
    rul_clip: int = 125,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load a C-MAPSS subset and return (train_df, test_df, test_rul).

    Parameters
    ----------
    data_dir : path to the folder containing the raw .txt files.
    subset   : one of 'FD001', 'FD002', 'FD003', 'FD004'.
    rul_clip : maximum RUL cap used in the literature (piece-wise linear RUL).

    Returns
    -------
    train_df  : DataFrame with sensor readings + 'rul' column.
    test_df   : DataFrame with sensor readings (no RUL — that's the prediction target).
    test_rul  : Series of true RUL values for the last cycle of each test engine.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {train_path}\n"
            "Download the NASA C-MAPSS dataset from:\n"
            "  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps\n"
            "and place the .txt files inside the 'data/raw/' folder."
        )

    train_df = _read_cmapss_file(train_path)
    test_df = _read_cmapss_file(test_path)
    test_rul = pd.read_csv(rul_path, header=None, names=["rul"])["rul"]

    # --- Compute piece-wise linear RUL for training set ---
    max_cycles = train_df.groupby("unit_id")["cycle"].max()
    train_df = train_df.merge(
        max_cycles.rename("max_cycle"), on="unit_id"
    )
    train_df["rul"] = (train_df["max_cycle"] - train_df["cycle"]).clip(upper=rul_clip)
    train_df.drop(columns=["max_cycle"], inplace=True)

    logger.info(
        "Loaded %s — train: %d rows, %d engines | test: %d rows, %d engines",
        subset,
        len(train_df),
        train_df["unit_id"].nunique(),
        len(test_df),
        test_df["unit_id"].nunique(),
    )
    return train_df, test_df, test_rul


def generate_synthetic_cmapss(
    n_engines: int = 100,
    max_cycles: int = 300,
    seed: int = 42,
    rul_clip: int = 125,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate a small synthetic C-MAPSS-like dataset for testing
    when the real data files are not available.

    Uses realistic sensor degradation curves (linear + noise).
    """
    rng = np.random.default_rng(seed)
    n_sensors = 14  # matching post-drop sensor count
    sensor_cols = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ]
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

    rows = []
    test_rows = []
    test_rul_vals = []

    for uid in range(1, n_engines + 1):
        life = rng.integers(150, max_cycles + 1)
        for cycle in range(1, life + 1):
            degradation = cycle / life  # 0→1 as engine wears out
            sensor_vals = {
                sc: rng.normal(loc=1.0 + degradation * rng.uniform(0.1, 0.5), scale=0.05)
                for sc in sensor_cols
            }
            op_vals = {oc: rng.normal(0, 0.1) for oc in op_cols}
            true_rul = min(life - cycle, rul_clip)
            rows.append({"unit_id": uid, "cycle": cycle, **op_vals, **sensor_vals, "rul": true_rul})

        # Test: use last 30% of cycles for one test sequence per engine
        test_life = rng.integers(50, 150)
        for cycle in range(1, test_life + 1):
            degradation = cycle / test_life
            sensor_vals = {
                sc: rng.normal(loc=1.0 + degradation * rng.uniform(0.1, 0.5), scale=0.05)
                for sc in sensor_cols
            }
            op_vals = {oc: rng.normal(0, 0.1) for oc in op_cols}
            test_rows.append({"unit_id": uid, "cycle": cycle, **op_vals, **sensor_vals})
        test_rul_vals.append(rng.integers(10, rul_clip))

    train_df = pd.DataFrame(rows)
    test_df = pd.DataFrame(test_rows)
    test_rul = pd.Series(test_rul_vals, name="rul")

    logger.info(
        "Generated synthetic dataset — train: %d rows, test: %d rows",
        len(train_df), len(test_df),
    )
    return train_df, test_df, test_rul
