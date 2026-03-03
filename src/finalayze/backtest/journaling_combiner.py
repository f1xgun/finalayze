"""Journaling strategy combiner — records per-strategy signals before combining.

This is evaluation-only code that copies the StrategyCombiner loop to avoid
double-invoking stateful strategies via super().

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.combiner import (
    _BUY_SCORE,
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
        allocation_mode: str = "static",
    ) -> None:
        super().__init__(strategies, normalize_mode, allocation_mode)
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
        weight_overrides: dict[str, Decimal] | None = None,
    ) -> Signal | None:
        """Generate a combined signal, capturing per-strategy signals."""
        # Reset tracking state
        self._last_signals = {}
        self._last_weights = {}
        self._last_net_score = None
        self._last_features = {}
        self._last_model_probas = None

        config = self._load_config(segment_id)
        strategies_cfg, effective_normalize, effective_min_confidence = self._parse_config(config)
        effective_overrides, hrp_overrides = self._resolve_effective_overrides(weight_overrides)

        weighted_score = _ZERO
        total_weight = _ZERO
        total_enabled_weight = _ZERO
        feature_contributions: dict[str, float] = {}

        # Hurst exponent routing: compute dynamic weight multipliers
        h, hurst_multipliers = self._compute_hurst_multipliers(candles)
        feature_contributions["hurst_exponent"] = h

        for strategy_name, strategy_cfg in strategies_cfg.items():
            if not isinstance(strategy_cfg, dict):
                continue
            if not strategy_cfg.get("enabled", True):
                continue

            weight = self._resolve_weight(strategy_name, strategy_cfg, effective_overrides)
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
            hurst_mult = Decimal(str(hurst_multipliers.get(strategy_name, 1.0)))
            contribution = score * Decimal(str(signal.confidence)) * weight * hurst_mult
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

        # Turn-of-month effect: boost BUY confidence during the window
        if self._is_turn_of_month(candles[-1].timestamp) and net > _ZERO:
            from finalayze.strategies.combiner import _TOM_BUY_BOOST  # noqa: PLC0415

            net += _TOM_BUY_BOOST
            feature_contributions["turn_of_month"] = 1.0
        else:
            feature_contributions["turn_of_month"] = 0.0

        # Add HRP weight features when using HRP allocation
        if hrp_overrides is not None:
            for sname, sweight in hrp_overrides.items():
                feature_contributions[f"hrp_weight_{sname}"] = float(sweight)

        self._last_net_score = float(net)

        if abs(net) < self._effective_threshold(
            config, effective_min_confidence, has_open_position, net
        ):
            return None

        return self._build_result(
            net, feature_contributions, symbol, candles[0].market_id, segment_id
        )
