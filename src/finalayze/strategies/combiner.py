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
_BUY_SCORE = Decimal(1)
_SELL_SCORE = Decimal(-1)
_MAX_CONFIDENCE = Decimal("1.0")
_ZERO = Decimal(0)


class StrategyCombiner:
    """Combines multiple strategy signals using per-segment YAML weights."""

    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self._strategies: dict[str, BaseStrategy] = {s.name: s for s in strategies}
        self._presets_dir = _PRESETS_DIR

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
    ) -> Signal | None:
        """Generate a combined signal by weighting enabled strategy signals."""
        config = self._load_config(segment_id)
        strategies_cfg_raw = config.get("strategies", {})
        strategies_cfg: dict[str, object] = (
            strategies_cfg_raw if isinstance(strategies_cfg_raw, dict) else {}
        )

        weighted_score = _ZERO
        total_weight = _ZERO
        feature_contributions: dict[str, float] = {}

        for strategy_name, strategy_cfg in strategies_cfg.items():
            if not isinstance(strategy_cfg, dict):
                continue
            if not strategy_cfg.get("enabled", True):
                continue
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue

            try:
                weight = Decimal(str(strategy_cfg.get("weight", "1.0")))
            except InvalidOperation:
                weight = Decimal("1.0")
            signal = strategy.generate_signal(symbol, candles, segment_id)
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

        net = weighted_score / total_weight
        abs_net = abs(net)

        if abs_net < _MIN_COMBINED_CONFIDENCE:
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

    def _load_config(self, segment_id: str) -> dict[str, object]:
        """Load segment YAML preset, returning an empty dict if not found or malformed."""
        try:
            path = self._presets_dir / f"{segment_id}.yaml"
            with path.open() as f:
                result = yaml.safe_load(f)
            return dict(result) if isinstance(result, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}
