"""Meta-labeling: ML predicts P(profitable) for rule-based signals (E1).

Instead of predicting market direction directly, the meta-labeler
predicts whether a signal from rule-based strategies will be profitable.
This is more tractable and eliminates calibration bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.core.schemas import Signal
    from finalayze.ml.models.ensemble import EnsembleModel

_log = structlog.get_logger()

_DEFAULT_THRESHOLD = 0.40
_UNTRAINED_EPSILON = 1e-9


class MetaLabeler:
    """Predict P(profitable) for a rule-based signal.

    Usage:
        meta = MetaLabeler(ensemble)
        p_profit = meta.predict(signal, features)
        if p_profit is not None and p_profit > 0.40:
            sizing_factor = (p_profit - 0.40) / 0.60
    """

    def __init__(
        self,
        ensemble: EnsembleModel,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._ensemble = ensemble
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(
        self,
        signal: Signal,
        features: dict[str, float],
    ) -> float | None:
        """Predict P(profitable) for the given signal.

        Returns None if ensemble has no trained models or prediction fails.
        """
        # Add signal metadata to features for meta-labeling
        meta_features = dict(features)
        meta_features["signal_confidence"] = signal.confidence
        meta_features["signal_direction_buy"] = 1.0 if signal.direction.value == "BUY" else 0.0

        try:
            prob = self._ensemble.predict_proba(meta_features, symbol=signal.symbol)
        except Exception:
            _log.warning("meta_labeler_predict_failed", symbol=signal.symbol, exc_info=True)
            return None

        # Untrained model returns exactly 0.5 — skip
        if abs(prob - 0.5) < _UNTRAINED_EPSILON:
            return None

        return prob

    def should_trade(self, p_profit: float) -> bool:
        """Return True if P(profitable) exceeds threshold."""
        return p_profit > self._threshold

    def sizing_factor(self, p_profit: float) -> float:
        """Return position sizing multiplier [0, 1] based on P(profitable).

        Maps [threshold, 1.0] -> [0.0, 1.0] linearly.
        Returns 0.0 if below threshold.
        """
        if p_profit <= self._threshold:
            return 0.0
        scaling_range = 1.0 - self._threshold
        return min(1.0, (p_profit - self._threshold) / scaling_range)
