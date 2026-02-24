"""LSTM per-segment model (Layer 3)."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5
_TRAIN_EPOCHS = 50
_LEARNING_RATE = 0.001


class _LSTMNet(nn.Module):
    """Internal PyTorch LSTM network."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self._lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self._linear = nn.Linear(hidden_size, 1)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, seq_len, input_size) → (batch, 1)."""
        lstm_out, _ = self._lstm(x)
        last_hidden = lstm_out[:, -1, :]
        result: torch.Tensor = self._sigmoid(self._linear(last_hidden))
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
        self._feature_buffer: deque[list[float]] = deque(maxlen=sequence_length)
        self._lock = threading.Lock()

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability in [0.0, 1.0]. Returns 0.5 when untrained."""
        if not self._trained or self._model is None:
            return _UNTRAINED_PROB

        if self._feature_names is not None and sorted(features.keys()) != self._feature_names:
            msg = f"Feature mismatch: expected {self._feature_names}, got {sorted(features.keys())}"
            raise InsufficientDataError(msg)

        sorted_vals = [features[k] for k in sorted(features)]

        with self._lock:
            self._feature_buffer.append(sorted_vals)

            if len(self._feature_buffer) < self._sequence_length:
                # Pad with the current observation repeated until buffer is full
                while len(self._feature_buffer) < self._sequence_length:
                    self._feature_buffer.appendleft(sorted_vals)

            seq = list(self._feature_buffer)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            output: torch.Tensor = self._model(tensor)
        return float(output.squeeze())

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the LSTM on feature dicts and binary labels.

        Args:
            X: List of feature dicts (all dicts must have identical keys).
            y: Binary labels (1=BUY, 0=SELL/HOLD), same length as X.

        Raises:
            InsufficientDataError: When len(X) < sequence_length.
        """
        if len(X) < self._sequence_length:
            msg = f"Need at least {self._sequence_length} samples for LSTM training, got {len(X)}"
            raise InsufficientDataError(msg)

        feature_keys = sorted(X[0])
        n_features = len(feature_keys)
        self._n_features = n_features
        self._feature_names = feature_keys

        # Build tensor: shape (n_sequences, sequence_length, n_features)
        sequences: list[list[list[float]]] = []
        labels: list[float] = []
        for i in range(len(X) - self._sequence_length + 1):
            seq = [[row[k] for k in feature_keys] for row in X[i : i + self._sequence_length]]
            sequences.append(seq)
            labels.append(float(y[i + self._sequence_length - 1]))

        x_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        self._model = _LSTMNet(n_features, self._hidden_size, self._num_layers)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=_LEARNING_RATE)
        criterion = nn.BCELoss()

        self._model.train()
        for _ in range(_TRAIN_EPOCHS):
            optimizer.zero_grad()
            output = self._model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        self._trained = True
        self._feature_buffer.clear()

    def save(self, path: Path) -> None:
        """Save model state dict and config to path."""
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
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Load model state dict and config from path."""
        payload: dict[str, Any] = torch.load(path, weights_only=False)
        cfg: dict[str, Any] = payload["config"]
        self._sequence_length = int(cfg["sequence_length"])
        self._hidden_size = int(cfg["hidden_size"])
        self._num_layers = int(cfg["num_layers"])
        self._n_features = int(cfg["n_features"])
        self._model = _LSTMNet(self._n_features, self._hidden_size, self._num_layers)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._trained = True
        self._feature_buffer = deque(maxlen=self._sequence_length)
        feature_names = cfg.get("feature_names")
        self._feature_names = list(feature_names) if feature_names is not None else None
