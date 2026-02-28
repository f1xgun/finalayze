"""Per-segment ML model registry (Layer 3)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.ensemble import EnsembleModel


class MLModelRegistry:
    """Maps segment IDs to trained EnsembleModel instances.

    Thread-safe: ``get()`` and ``register()`` are protected by a lock so that
    models can be hot-swapped during automated retraining without races.
    """

    def __init__(self) -> None:
        self._models: dict[str, EnsembleModel] = {}
        self._lock = threading.Lock()

    def register(self, segment_id: str, model: EnsembleModel) -> None:
        """Register or replace a model for a segment."""
        with self._lock:
            self._models[segment_id] = model

    def get(self, segment_id: str) -> EnsembleModel | None:
        """Return the model for the segment, or None if not registered."""
        with self._lock:
            return self._models.get(segment_id)

    def create_ensemble(self, segment_id: str) -> EnsembleModel:
        """Create a new EnsembleModel with XGBoost + LightGBM + LSTM for a segment.

        The models are untrained; call ``ensemble.fit(X, y)`` or load saved
        weights via each model's ``.load()`` before prediction.
        """
        from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415
        from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415
        from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

        xgb = XGBoostModel(segment_id=segment_id)
        lgbm = LightGBMModel(segment_id=segment_id)
        lstm = LSTMModel(segment_id=segment_id)
        return EnsembleModel(models=[xgb, lgbm], lstm_model=lstm)
