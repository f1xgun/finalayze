"""LSTM per-segment model (Layer 3)."""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5
_TRAIN_EPOCHS = 50
_LEARNING_RATE = 0.001
_CALIBRATION_HOLDOUT_FRACTION = 0.2
_MIN_CALIBRATION_SAMPLES = 10

# 6C.5: Early stopping + gradient clipping
_PATIENCE = 5
_MAX_GRAD_NORM = 1.0

# 6C.6: Dropout + weight decay
_DROPOUT = 0.2
_WEIGHT_DECAY = 1e-4


class _LSTMNet(nn.Module):
    """Internal PyTorch LSTM network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._dropout = nn.Dropout(dropout)
        self._linear = nn.Linear(hidden_size, 1)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, seq_len, input_size) -> (batch, 1)."""
        lstm_out, _ = self._lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self._dropout(last_hidden)
        result: torch.Tensor = self._sigmoid(self._linear(dropped))
        return result


class LSTMModel(BaseMLModel):
    """2-layer LSTM classifier for directional prediction per segment."""

    segment_id: str

    def __init__(
        self,
        segment_id: str,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        self.segment_id = segment_id
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._n_features: int | None = None
        self._feature_names: list[str] | None = None
        self._model: _LSTMNet | None = None
        self._trained: bool = False
        self._feature_buffers: dict[str, deque[list[float]]] = {}
        self._lock = threading.Lock()
        # Scaler fitted on training data; applied during inference (#152)
        self._scaler: StandardScaler | None = None
        # Platt scaling: logistic regression mapping raw sigmoid -> calibrated probability
        self._platt_scaler: LogisticRegression | None = None

    def predict_proba(self, features: dict[str, float], *, symbol: str = "__default__") -> float:
        """Return BUY probability in [0.0, 1.0]. Returns 0.5 when untrained."""
        if not self._trained or self._model is None:
            return _UNTRAINED_PROB

        if self._feature_names is not None and sorted(features.keys()) != self._feature_names:
            msg = (
                f"Feature mismatch: expected {self._feature_names}, "
                f"got {sorted(features.keys())}"
            )
            raise InsufficientDataError(msg)

        sorted_vals = [features[k] for k in sorted(features)]

        # Apply the same scaler used during training (#152)
        if self._scaler is not None:
            sorted_vals = self._scaler.transform([sorted_vals])[0].tolist()

        # --- Per-symbol buffer: avoids cross-contamination (issue 5.6) ---
        with self._lock:
            buf = self._feature_buffers.setdefault(
                symbol, deque(maxlen=self._sequence_length)
            )
            buf.append(sorted_vals)
            buffer_copy = list(buf)  # snapshot under lock

        # Pad the *copy* -- the shared buffer is never touched again outside the lock
        if len(buffer_copy) < self._sequence_length:
            while len(buffer_copy) < self._sequence_length:
                buffer_copy.insert(0, sorted_vals)

        tensor = torch.tensor(buffer_copy, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            output: torch.Tensor = self._model(tensor)
        raw_prob = float(output.squeeze())

        # Apply Platt scaling calibration if available
        if self._platt_scaler is not None:
            calibrated: float = float(
                self._platt_scaler.predict_proba(np.array([[raw_prob]]))[0][1]
            )
            return calibrated
        return raw_prob

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the LSTM on feature dicts and binary labels.

        Includes early stopping with patience, gradient clipping, dropout,
        and weight decay for regularization (6C.5 + 6C.6).

        Args:
            X: List of feature dicts (all dicts must have identical keys).
            y: Binary labels (1=BUY, 0=SELL/HOLD), same length as X.

        Raises:
            InsufficientDataError: When len(X) < sequence_length.
        """
        if len(X) < self._sequence_length:
            msg = (
                f"Need at least {self._sequence_length} samples for LSTM training, "
                f"got {len(X)}"
            )
            raise InsufficientDataError(msg)

        feature_keys = sorted(X[0])
        n_features = len(feature_keys)
        self._n_features = n_features
        self._feature_names = feature_keys

        # Raw feature matrix: shape (n_samples, n_features)
        raw_matrix = np.array(
            [[row[k] for k in feature_keys] for row in X], dtype=np.float32
        )

        # Fit scaler on all training samples and normalise (#152)
        self._scaler = StandardScaler()
        scaled_matrix = self._scaler.fit_transform(raw_matrix)

        # Build tensor: shape (n_sequences, sequence_length, n_features)
        sequences: list[list[list[float]]] = []
        labels: list[float] = []
        for i in range(len(X) - self._sequence_length + 1):
            seq = scaled_matrix[i : i + self._sequence_length].tolist()
            sequences.append(seq)
            labels.append(float(y[i + self._sequence_length - 1]))

        x_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        # 6C.6: Pass dropout to _LSTMNet
        self._model = _LSTMNet(
            n_features, self._hidden_size, self._num_layers, dropout=_DROPOUT
        )
        # 6C.6: Add weight_decay to optimizer
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY
        )
        criterion = nn.BCELoss()

        # Split sequences into train and calibration holdout
        n_sequences = len(sequences)
        n_cal = max(int(n_sequences * _CALIBRATION_HOLDOUT_FRACTION), 1)
        n_train = n_sequences - n_cal

        x_train = x_tensor[:n_train]
        y_train = y_tensor[:n_train]
        x_cal = x_tensor[n_train:]
        y_cal_labels = np.array(labels[n_train:], dtype=int)

        # 6C.5: Further split train into train_inner + val_inner for early stopping
        n_inner_val = max(int(n_train * 0.1), 1)
        n_inner_train = n_train - n_inner_val

        x_inner_train = x_train[:n_inner_train]
        y_inner_train = y_train[:n_inner_train]
        x_inner_val = x_train[n_inner_train:]
        y_inner_val = y_train[n_inner_train:]

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_counter = 0

        self._model.train()
        for _ in range(_TRAIN_EPOCHS):
            optimizer.zero_grad()
            output = self._model(x_inner_train)
            loss = criterion(output, y_inner_train)
            loss.backward()
            # 6C.5: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), _MAX_GRAD_NORM)
            optimizer.step()

            # Validation loss for early stopping
            self._model.eval()
            with torch.no_grad():
                val_output = self._model(x_inner_val)
                val_loss = float(criterion(val_output, y_inner_val))
            self._model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= _PATIENCE:
                    break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Fit Platt scaler (logistic regression) on calibration holdout
        # Requires both classes present and sufficient samples
        self._model.eval()
        if (
            len(np.unique(y_cal_labels)) > 1
            and len(y_cal_labels) >= _MIN_CALIBRATION_SAMPLES
        ):
            with torch.no_grad():
                cal_raw: torch.Tensor = self._model(x_cal)
            cal_raw_np = cal_raw.squeeze().numpy()
            if cal_raw_np.ndim == 0:
                cal_raw_np = cal_raw_np.reshape(1)
            self._platt_scaler = LogisticRegression(solver="lbfgs", max_iter=1000)
            self._platt_scaler.fit(cal_raw_np.reshape(-1, 1), y_cal_labels)
        else:
            self._platt_scaler = None

        self._trained = True
        self._feature_buffers.clear()

    def save(self, path: Path) -> None:
        """Save model state dict and config to *path* atomically.

        Uses temp file + rename pattern for all three files (weights, scaler,
        platt scaler) to avoid corruption on interrupted writes (6C.9).
        """
        if self._model is None:
            msg = "Cannot save an untrained LSTMModel"
            raise ValueError(msg)
        payload: dict[str, Any] = {
            "state_dict": self._model.state_dict(),
            "config": {
                "segment_id": self.segment_id,
                "sequence_length": self._sequence_length,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "n_features": self._n_features,
                "feature_names": self._feature_names,
            },
        }

        # Atomic save: weights
        _atomic_write_torch(payload, path)

        # Atomic save: scaler
        scaler_path = path.parent / (path.name + ".scaler.pkl")
        _atomic_write_pickle(self._scaler, scaler_path)

        # Atomic save: platt scaler
        platt_path = path.parent / (path.name + ".platt.pkl")
        _atomic_write_pickle(self._platt_scaler, platt_path)

    def load(self, path: Path) -> None:
        """Load model state dict and config from path.

        Uses ``weights_only=True`` to prevent arbitrary code execution when
        loading the PyTorch checkpoint (#169).  The companion scaler file is
        loaded separately via pickle.
        """
        payload: dict[str, Any] = torch.load(path, weights_only=True)
        cfg: dict[str, Any] = payload["config"]
        self._sequence_length = int(cfg["sequence_length"])
        self._hidden_size = int(cfg["hidden_size"])
        self._num_layers = int(cfg["num_layers"])
        self._n_features = int(cfg["n_features"])
        self._model = _LSTMNet(
            self._n_features, self._hidden_size, self._num_layers, dropout=_DROPOUT
        )
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._trained = True
        self._feature_buffers = {}
        feature_names = cfg.get("feature_names")
        self._feature_names = list(feature_names) if feature_names is not None else None
        scaler_path = path.parent / (path.name + ".scaler.pkl")
        if scaler_path.exists():
            with scaler_path.open("rb") as fh:
                self._scaler = pickle.load(fh)  # noqa: S301
        else:
            self._scaler = None
        platt_path = path.parent / (path.name + ".platt.pkl")
        if platt_path.exists():
            with platt_path.open("rb") as fh:
                self._platt_scaler = pickle.load(fh)  # noqa: S301
        else:
            self._platt_scaler = None


def _atomic_write_torch(payload: dict[str, Any], target: Path) -> None:
    """Write a torch checkpoint atomically via temp + rename."""
    from pathlib import Path as _Path  # noqa: PLC0415

    fd, tmp_str = tempfile.mkstemp(
        dir=target.parent, suffix=".tmp", prefix=target.stem
    )
    tmp_path = _Path(tmp_str)
    try:
        os.close(fd)
        torch.save(payload, tmp_path)
        tmp_path.rename(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_pickle(obj: object, target: Path) -> None:
    """Write a pickle file atomically via temp + rename."""
    from pathlib import Path as _Path  # noqa: PLC0415

    fd, tmp_str = tempfile.mkstemp(
        dir=target.parent, suffix=".tmp", prefix=target.stem
    )
    tmp_path = _Path(tmp_str)
    try:
        os.close(fd)
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh)
        tmp_path.rename(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
