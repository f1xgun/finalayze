"""Ensemble model combining XGBoost + LightGBM + optional LSTM (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.lstm_model import LSTMModel

_DEFAULT_PROB = 0.5


class EnsembleModel:
    """Averages probability predictions from multiple trained BaseMLModel instances.

    Only models that are trained contribute to the average.  Untrained models
    are skipped, so the denominator always reflects active models.  When no
    models are trained, returns 0.5 (neutral probability).
    """

    def __init__(
        self,
        models: list[BaseMLModel],
        lstm_model: LSTMModel | None = None,
    ) -> None:
        self._models = models
        self._lstm_model = lstm_model

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return mean BUY probability across all *trained* models.

        Falls back to 0.5 when no models are trained.
        """
        # XGBoostModel and LightGBMModel are trained when _model is not None
        probs = [
            m.predict_proba(features)
            for m in self._models
            if getattr(m, "_model", None) is not None
        ]

        if self._lstm_model is not None and getattr(self._lstm_model, "_trained", False):
            probs.append(self._lstm_model.predict_proba(features))

        if not probs:
            return _DEFAULT_PROB
        return sum(probs) / len(probs)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train all constituent models (including LSTM if present)."""
        for model in self._models:
            model.fit(X, y)
        if self._lstm_model is not None:
            self._lstm_model.fit(X, y)
