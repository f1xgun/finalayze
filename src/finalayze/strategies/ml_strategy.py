"""ML ensemble trading strategy (Layer 4).

Wraps ``MLModelRegistry`` + ``EnsembleModel.predict_proba()`` as a
``BaseStrategy`` so the ``StrategyCombiner`` can include ML predictions
alongside rule-based strategies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.ml.features.technical import compute_features
from finalayze.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from finalayze.ml.registry import MLModelRegistry

_PRESETS_DIR = Path(__file__).parent / "presets"
_log = structlog.get_logger()

_DEFAULT_THRESHOLD = 0.15
_DEFAULT_MIN_CONFIDENCE = 0.5
_UNTRAINED_PROB = 0.5
_UNTRAINED_EPSILON = 1e-9


class MLStrategy(BaseStrategy):
    """Generate signals from ML ensemble probability predictions.

    The strategy delegates to ``MLModelRegistry`` to obtain a trained
    ``EnsembleModel`` per segment, then maps the BUY probability to a
    directional signal with a configurable deadzone threshold.

    When no model is registered for a segment (or the model is untrained),
    the strategy returns ``None`` — graceful degradation.
    """

    def __init__(self, registry: MLModelRegistry) -> None:
        self._registry = registry
        self._params_cache: dict[str, dict[str, object]] = {}

    @property
    def name(self) -> str:
        return "ml_ensemble"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where ml_ensemble is enabled in YAML presets."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            strategies = data.get("strategies", {})
            ml_cfg = strategies.get("ml_ensemble", {})
            if ml_cfg.get("enabled", False):
                segments.append(data["segment_id"])
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load ml_ensemble parameters from the YAML preset for the given segment."""
        if segment_id in self._params_cache:
            return self._params_cache[segment_id]
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            params = dict(data["strategies"]["ml_ensemble"]["params"])
        except (FileNotFoundError, KeyError, TypeError):
            params = {}
        self._params_cache[segment_id] = params
        return params

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
    ) -> Signal | None:
        """Generate a signal from ML ensemble prediction.

        Sentiment is intentionally passed as 0.0 to ``compute_features`` to
        maintain train/inference feature consistency (training data does not
        include historical sentiment).
        """
        ensemble = self._registry.get(segment_id)
        if ensemble is None:
            return None

        try:
            features = compute_features(candles, sentiment_score=0.0)
        except InsufficientDataError:
            return None

        try:
            prob = ensemble.predict_proba(features, symbol=symbol)
        except Exception:
            _log.exception("MLStrategy: predict_proba failed for %s/%s", segment_id, symbol)
            return None

        # Untrained model returns exactly 0.5 — skip to avoid noise
        if abs(prob - _UNTRAINED_PROB) < _UNTRAINED_EPSILON:
            return None

        result = self._map_probability(prob, segment_id)
        if result is None:
            return None
        direction, confidence = result

        params = self.get_parameters(segment_id)
        threshold = float(params.get("threshold", _DEFAULT_THRESHOLD))  # type: ignore[arg-type]
        market_id = candles[0].market_id
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={"ml_prob": round(prob, 4)},
            reasoning=f"ML ensemble prob={prob:.3f} (threshold={threshold})",
        )

    def _map_probability(
        self, prob: float, segment_id: str
    ) -> tuple[SignalDirection, float] | None:
        """Map a BUY probability to direction + confidence, or None if in deadzone."""
        params = self.get_parameters(segment_id)
        threshold = float(params.get("threshold", _DEFAULT_THRESHOLD))  # type: ignore[arg-type]
        min_confidence = float(params.get("min_confidence", _DEFAULT_MIN_CONFIDENCE))  # type: ignore[arg-type]

        if prob > _UNTRAINED_PROB + threshold:
            direction = SignalDirection.BUY
            confidence = (prob - _UNTRAINED_PROB) * 2
        elif prob < _UNTRAINED_PROB - threshold:
            direction = SignalDirection.SELL
            confidence = (_UNTRAINED_PROB - prob) * 2
        else:
            return None

        if confidence < min_confidence:
            return None
        return direction, confidence
