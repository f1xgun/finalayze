"""Journaling strategy combiner — records per-strategy signals before combining.

This is evaluation-only code that copies the StrategyCombiner loop to avoid
double-invoking stateful strategies via super().

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.combiner import (
    _BUY_SCORE,
    _MAX_CONFIDENCE,
    _MIN_COMBINED_CONFIDENCE,
    _MIN_EXIT_CONFIDENCE,
    _SELL_SCORE,
    _ZERO,
    StrategyCombiner,
)

if TYPE_CHECKING:
    from finalayze.strategies.base import BaseStrategy


class JournalingStrategyCombiner(StrategyCombiner):
    """Records per-strategy signals before combining them.

    After generate_signal() is called, the last_signals and last_weights
    dicts are populated for the backtest engine to read.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
    ) -> None:
        super().__init__(strategies, normalize_mode)
        self._last_signals: dict[str, Signal | None] = {}
        self._last_weights: dict[str, Decimal] = {}
        self._last_net_score: float | None = None
        self._last_features: dict[str, float] = {}
        self._last_model_probas: dict[str, float] | None = None

    @property
    def last_signals(self) -> dict[str, Signal | None]:
        """Per-strategy signals from the most recent generate_signal() call."""
        return dict(self._last_signals)

    @property
    def last_weights(self) -> dict[str, Decimal]:
        """Per-strategy weights from the most recent generate_signal() call."""
        return dict(self._last_weights)

    @property
    def last_net_score(self) -> float | None:
        """Net weighted score from the most recent generate_signal() call."""
        return self._last_net_score

    @property
    def last_features(self) -> dict[str, float]:
        """Aggregated features from all strategy signals, prefixed by strategy name."""
        return dict(self._last_features)

    @property
    def last_model_probas(self) -> dict[str, float] | None:
        """Per-model probabilities from MLStrategy's EnsembleModel, if present."""
        return dict(self._last_model_probas) if self._last_model_probas is not None else None

    def generate_signal(  # noqa: PLR0912, PLR0915
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        """Generate a combined signal, capturing per-strategy signals."""
        # Reset tracking state
        self._last_signals = {}
        self._last_weights = {}
        self._last_net_score = None
        self._last_features = {}
        self._last_model_probas = None

        config = self._load_config(segment_id)
        strategies_cfg_raw = config.get("strategies", {})
        strategies_cfg: dict[str, object] = (
            strategies_cfg_raw if isinstance(strategies_cfg_raw, dict) else {}
        )

        # Per-segment overrides for normalize_mode and min_combined_confidence
        effective_normalize = str(config.get("normalize_mode", self._normalize_mode))
        try:
            effective_min_confidence = Decimal(
                str(config.get("min_combined_confidence", _MIN_COMBINED_CONFIDENCE))
            )
        except InvalidOperation:
            effective_min_confidence = _MIN_COMBINED_CONFIDENCE

        weighted_score = _ZERO
        total_weight = _ZERO
        total_enabled_weight = _ZERO
        feature_contributions: dict[str, float] = {}

        for strategy_name, strategy_cfg in strategies_cfg.items():
            if not isinstance(strategy_cfg, dict):
                continue
            if not strategy_cfg.get("enabled", True):
                continue

            try:
                weight = Decimal(str(strategy_cfg.get("weight", "1.0")))
            except InvalidOperation:
                weight = Decimal("1.0")
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue
            total_enabled_weight += weight

            signal = strategy.generate_signal(
                symbol, candles, segment_id, sentiment_score=sentiment_score
            )

            # Record per-strategy signal and weight
            self._last_signals[strategy_name] = signal
            self._last_weights[strategy_name] = weight

            if signal is None:
                continue

            # Aggregate per-strategy features prefixed by strategy name
            for feat_key, feat_val in signal.features.items():
                self._last_features[f"{strategy_name}.{feat_key}"] = feat_val

            # Capture per-model probas from MLStrategy's EnsembleModel
            if hasattr(strategy, "_registry"):
                ensemble = getattr(strategy._registry, "get", lambda _s: None)(segment_id)
                if ensemble is not None and hasattr(ensemble, "last_model_probas"):
                    probas = ensemble.last_model_probas
                    if probas:
                        self._last_model_probas = dict(probas)

            score = _BUY_SCORE if signal.direction == SignalDirection.BUY else _SELL_SCORE
            contribution = score * Decimal(str(signal.confidence)) * weight
            weighted_score += contribution
            total_weight += weight
            feature_contributions[f"{strategy_name}_confidence"] = signal.confidence
            feature_contributions[f"{strategy_name}_direction"] = (
                1.0 if signal.direction == SignalDirection.BUY else -1.0
            )

        if total_weight == _ZERO:
            self._last_net_score = 0.0
            return None

        denominator = total_enabled_weight if effective_normalize == "total" else total_weight
        if denominator == _ZERO:
            self._last_net_score = 0.0
            return None
        net = weighted_score / denominator
        self._last_net_score = float(net)
        abs_net = abs(net)

        # Lower threshold for SELL signals when holding an open position
        effective_threshold = effective_min_confidence
        if has_open_position and net < _ZERO:
            exit_conf = Decimal(str(config.get("min_exit_confidence", _MIN_EXIT_CONFIDENCE)))
            effective_threshold = min(effective_min_confidence, exit_conf)

        if abs_net < effective_threshold:
            return None

        direction = SignalDirection.BUY if net > _ZERO else SignalDirection.SELL
        confidence = float(min(abs_net, _MAX_CONFIDENCE))
        strategy_count = len(feature_contributions) // 2

        market_id = candles[0].market_id

        return Signal(
            strategy_name="combined",
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features=feature_contributions,
            reasoning=(
                f"Combined signal: net_score={float(net):.3f} from {strategy_count} strategies"
            ),
        )
