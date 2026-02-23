"""LightGBM per-segment model (Layer 3)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class LightGBMModel(BaseMLModel):
    """LightGBM classifier for directional prediction per segment."""

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability (0.0-1.0). Returns 0.5 when untrained."""
        if self._model is None:
            return _UNTRAINED_PROB
        if self._feature_names is not None:
            incoming = sorted(features)
            if incoming != self._feature_names:
                msg = (
                    f"Feature mismatch for segment {self.segment_id!r}: "
                    f"expected {self._feature_names}, got {incoming}"
                )
                raise InsufficientDataError(msg)
        features_arr = np.array([[features[k] for k in sorted(features)]], dtype=float)
        proba: float = float(self._model.predict_proba(features_arr)[0][1])
        return proba

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the model on feature dicts and binary labels."""
        if X:
            self._feature_names = sorted(X[0])
        x_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)
        self._model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=-1
        )
        self._model.fit(x_arr, y_arr)
