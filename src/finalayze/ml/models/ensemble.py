"""Ensemble model combining XGBoost + LightGBM + optional LSTM (Layer 3)."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from finalayze.core.exceptions import InsufficientDataError, PredictionError

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.lstm_model import LSTMModel

_DEFAULT_PROB = 0.5
_log = logging.getLogger(__name__)


class EnsembleModel:
    """Averages probability predictions from multiple trained BaseMLModel instances.

    Only models that are trained contribute to the average.  Untrained models
    are skipped, so the denominator always reflects active models.  When no
    models are trained, returns 0.5 (neutral probability).  When trained models
    all raise exceptions, raises ``PredictionError``.
    """

    def __init__(
        self,
        models: list[BaseMLModel],
        lstm_model: LSTMModel | None = None,
    ) -> None:
        self._models = models
        self._lstm_model = lstm_model

    def predict_proba(self, features: dict[str, float], *, symbol: str = "__default__") -> float:
        """Return mean BUY probability across all *trained* models.

        Falls back to 0.5 when no models are trained.
        Raises PredictionError when all trained models fail.
        """
        probs: list[float] = []
        any_trained = False

        for m in self._models:
            if getattr(m, "_model", None) is None:
                continue
            any_trained = True
            try:
                probs.append(m.predict_proba(features))
            except Exception:
                _log.warning(
                    "Ensemble: %s failed, skipping", type(m).__name__, exc_info=True
                )

        if self._lstm_model is not None and getattr(self._lstm_model, "_trained", False):
            any_trained = True
            try:
                probs.append(self._lstm_model.predict_proba(features, symbol=symbol))
            except Exception:
                _log.warning("Ensemble: LSTM failed, skipping", exc_info=True)

        if not probs:
            if any_trained:
                raise PredictionError(
                    "All ensemble sub-models failed to produce a prediction"
                )
            return _DEFAULT_PROB
        return sum(probs) / len(probs)

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train all constituent models (including LSTM if present).

        Each model is trained independently. If a model raises InsufficientDataError
        (e.g. LSTM when len(X) < sequence_length), it is left untrained and will
        return 0.5 in predict_proba -- graceful degradation.
        """
        for model in self._models:
            with contextlib.suppress(InsufficientDataError):
                model.fit(X, y)
        if self._lstm_model is not None:
            with contextlib.suppress(InsufficientDataError):
                self._lstm_model.fit(X, y)
