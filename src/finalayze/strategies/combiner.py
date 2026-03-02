"""Per-segment weighted strategy combiner (Layer 4)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection

if TYPE_CHECKING:
    from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_COMBINED_CONFIDENCE = Decimal("0.50")
_MIN_EXIT_CONFIDENCE = Decimal("0.10")
_BUY_SCORE = Decimal(1)
_SELL_SCORE = Decimal(-1)
_MAX_CONFIDENCE = Decimal("1.0")
_ZERO = Decimal(0)
_DEFAULT_WEIGHT = Decimal("1.0")


class StrategyCombiner:
    """Combines multiple strategy signals using per-segment YAML weights."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
    ) -> None:
        self._strategies: dict[str, BaseStrategy] = {s.name: s for s in strategies}
        self._presets_dir = _PRESETS_DIR
        self._normalize_mode = normalize_mode

    @staticmethod
    def _resolve_weight(
        strategy_name: str,
        strategy_cfg: dict[str, object],
        weight_overrides: dict[str, Decimal] | None,
    ) -> Decimal:
        """Return the effective weight for a strategy."""
        if weight_overrides and strategy_name in weight_overrides:
            return weight_overrides[strategy_name]
        try:
            return Decimal(str(strategy_cfg.get("weight", "1.0")))
        except InvalidOperation:
            return _DEFAULT_WEIGHT

    def _build_result(
        self,
        net: Decimal,
        feature_contributions: dict[str, float],
        symbol: str,
        market_id: str,
        segment_id: str,
    ) -> Signal:
        """Create the combined Signal from net score and features."""
        direction = SignalDirection.BUY if net > _ZERO else SignalDirection.SELL
        confidence = float(min(abs(net), _MAX_CONFIDENCE))
        strategy_count = len(feature_contributions) // 2
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

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
        weight_overrides: dict[str, Decimal] | None = None,
    ) -> Signal | None:
        """Generate a combined signal by weighting enabled strategy signals.

        Args:
            weight_overrides: When provided, these weights are used instead of
                the YAML-configured weights for each named strategy.
        """
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

            weight = self._resolve_weight(strategy_name, strategy_cfg, weight_overrides)
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue
            total_enabled_weight += weight

            signal = strategy.generate_signal(
                symbol, candles, segment_id, sentiment_score=sentiment_score
            )
            if signal is None:
                continue

            score = _BUY_SCORE if signal.direction == SignalDirection.BUY else _SELL_SCORE
            contribution = score * Decimal(str(signal.confidence)) * weight
            weighted_score += contribution
            total_weight += weight
            feature_contributions[f"{strategy_name}_confidence"] = signal.confidence
            feature_contributions[f"{strategy_name}_direction"] = (
                1.0 if signal.direction == SignalDirection.BUY else -1.0
            )

        if total_weight == _ZERO:
            return None

        denominator = total_enabled_weight if effective_normalize == "total" else total_weight
        if denominator == _ZERO:
            return None
        net = weighted_score / denominator
        abs_net = abs(net)

        # Lower threshold for SELL signals when holding an open position
        effective_threshold = effective_min_confidence
        if has_open_position and net < _ZERO:
            exit_conf = Decimal(str(config.get("min_exit_confidence", _MIN_EXIT_CONFIDENCE)))
            effective_threshold = min(effective_min_confidence, exit_conf)

        if abs_net < effective_threshold:
            return None

        return self._build_result(
            net, feature_contributions, symbol, candles[0].market_id, segment_id
        )

    def _load_config(self, segment_id: str) -> dict[str, object]:
        """Load segment YAML preset, returning an empty dict if not found or malformed."""
        try:
            path = self._presets_dir / f"{segment_id}.yaml"
            with path.open() as f:
                result = yaml.safe_load(f)
            return dict(result) if isinstance(result, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}
