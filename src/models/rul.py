"""
Remaining Useful Life (RUL) prediction model.

Implements two interchangeable estimators:
  - XGBoostRUL  : gradient-boosted trees (fast, interpretable, great baseline).
  - LSTMRul     : PyTorch LSTM for sequence modelling (more accurate on long windows).

Both expose the same fit() / predict() interface so they can be swapped
without changing any pipeline code.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRULModel(ABC):
    """Common interface for all RUL estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRULModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRULModel":
        ...


# ---------------------------------------------------------------------------
# XGBoost model (3-D → 2-D flattened)
# ---------------------------------------------------------------------------

class XGBoostRUL(BaseRULModel):
    """
    Gradient-boosted trees regressor for RUL prediction.
    Uses XGBoost if available; gracefully falls back to Scikit-Learn's
    HistGradientBoostingRegressor if libomp (OpenMP) is missing on macOS.
    Flattens the [samples, timesteps, features] array before fitting.
    """

    def __init__(self, **kwargs):
        try:
            from xgboost import XGBRegressor
            defaults = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            defaults.update(kwargs)
            self.model = XGBRegressor(**defaults)
        except Exception as e:
            logger.warning(
                "XGBoost native library not available (%s). "
                "Using Scikit-Learn HistGradientBoostingRegressor.",
                e,
            )
            from sklearn.ensemble import HistGradientBoostingRegressor
            defaults = dict(
                max_iter=300,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
            )
            clean_kwargs = dict(kwargs)
            if "n_estimators" in clean_kwargs:
                clean_kwargs["max_iter"] = clean_kwargs.pop("n_estimators")
            clean_kwargs.pop("subsample", None)
            clean_kwargs.pop("colsample_bytree", None)
            clean_kwargs.pop("n_jobs", None)
            clean_kwargs.pop("verbosity", None)
            defaults.update(clean_kwargs)
            self.model = HistGradientBoostingRegressor(**defaults)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten [N, T, F] → [N, T*F] if 3-D, else pass through."""
        return X.reshape(len(X), -1) if X.ndim == 3 else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRUL":
        logger.info("Training %s on %d samples …", type(self.model).__name__, len(X))
        self.model.fit(self._flatten(X), y)
        logger.info("RUL model training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(self._flatten(X))
        return np.clip(preds, 0, None)  # RUL cannot be negative

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRUL":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return obj


# ---------------------------------------------------------------------------
# PyTorch LSTM model
# ---------------------------------------------------------------------------

class LSTMRul(BaseRULModel):
    """
    Bidirectional LSTM for sequence-based RUL regression.

    Expects X of shape [N, timesteps, features].
    Uses early stopping based on a validation split to avoid overfitting.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_split: float = 0.1,
        patience: int = 8,
        device: Optional[str] = None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        self.patience = patience
        self._net = None
        self._device = device

    # ---- Internal helpers ------------------------------------------------

    def _get_device(self):
        import torch
        if self._device:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_net(self, input_size: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=True,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        return _Net(input_size, self.hidden_size, self.num_layers, self.dropout)

    # ---- Public API -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMRul":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("Install PyTorch: pip install torch")

        device = self._get_device()
        logger.info("Training LSTMRul on %d samples (device=%s) …", len(X), device)

        input_size = X.shape[2]
        self._net = self._build_net(input_size).to(device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Train / validation split
        n_val = max(1, int(len(X_t) * self.val_split))
        perm = torch.randperm(len(X_t))
        idx_val, idx_train = perm[:n_val], perm[n_val:]

        train_dl = DataLoader(
            TensorDataset(X_t[idx_train], y_t[idx_train]),
            batch_size=self.batch_size, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(X_t[idx_val], y_t[idx_val]),
            batch_size=self.batch_size,
        )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

            self._net.eval()
            val_loss = sum(
                criterion(self._net(xb.to(device)), yb.to(device)).item()
                for xb, yb in val_dl
            ) / len(val_dl)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                self._best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                no_improve += 1

            if epoch % 5 == 0:
                logger.info("Epoch %3d/%d  val_loss=%.4f", epoch, self.epochs, val_loss)

            if no_improve >= self.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

        self._net.load_state_dict(self._best_state)
        logger.info("LSTMRul training complete. Best val_loss=%.4f", best_val_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = self._get_device()
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self._net(X_t).cpu().numpy()
        return np.clip(preds, 0, None)

    def save(self, path: str | Path) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "input_size": next(iter(self._net.parameters())).shape[-1] if self._net else None,
                },
            },
            path,
        )
        logger.info("LSTM model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMRul":
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        cfg = checkpoint["config"]
        obj = cls(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        obj._net = obj._build_net(cfg["input_size"])
        obj._net.load_state_dict(checkpoint["state_dict"])
        obj._net.eval()
        logger.info("LSTM model loaded ← %s", path)
        return obj
