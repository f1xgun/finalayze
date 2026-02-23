"""Ensemble model combining XGBoost + LightGBM (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel

_DEFAULT_PROB = 0.5


class EnsembleModel:
    """Averages probability predictions from multiple BaseMLModel instances."""

    def __init__(self, models: list[BaseMLModel]) -> None:
        self._models = models

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return mean BUY probability across all models. Returns 0.5 when empty."""
        if not self._models:
            return _DEFAULT_PROB
        probs = [m.predict_proba(features) for m in self._models]
        return sum(probs) / len(probs)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train all constituent models."""
        for model in self._models:
            model.fit(X, y)
