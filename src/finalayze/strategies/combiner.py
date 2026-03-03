"""Per-segment weighted strategy combiner (Layer 4)."""

from __future__ import annotations

import calendar
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.hrp import compute_hrp_weights
from finalayze.strategies.hurst import compute_hurst_exponent

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_COMBINED_CONFIDENCE = Decimal("0.50")
_MIN_EXIT_CONFIDENCE = Decimal("0.10")
_BUY_SCORE = Decimal(1)
_SELL_SCORE = Decimal(-1)
_MAX_CONFIDENCE = Decimal("1.0")
_ZERO = Decimal(0)
_DEFAULT_WEIGHT = Decimal("1.0")

# Hurst exponent routing constants
_MOMENTUM_STRATEGIES = frozenset({"momentum", "dual_momentum"})
_MR_STRATEGIES = frozenset({"mean_reversion", "pairs", "ou_mean_reversion", "rsi2_connors"})
_HURST_TRENDING_THRESHOLD = 0.55
_HURST_MR_THRESHOLD = 0.45
_HURST_BOOST = 1.5
_HURST_SUPPRESS = 1.0 / _HURST_BOOST

# Turn-of-month effect: boost BUY confidence during last 1 + first 3 calendar days
_TOM_BUY_BOOST = Decimal("0.05")

# HRP allocation constants
_HRP_MIN_HISTORY = 20


class StrategyCombiner:
    """Combines multiple strategy signals using per-segment YAML weights."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
        allocation_mode: str = "static",
    ) -> None:
        self._strategies: dict[str, BaseStrategy] = {s.name: s for s in strategies}
        self._presets_dir = _PRESETS_DIR
        self._normalize_mode = normalize_mode
        self._allocation_mode = allocation_mode
        self._strategy_returns: dict[str, list[float]] = defaultdict(list)
        self._hrp_weights: dict[str, Decimal] | None = None

    def record_strategy_return(self, strategy_name: str, ret: float) -> None:
        """Record a strategy return observation for HRP weight computation.

        Only accumulates data when allocation_mode is 'hrp'.
        """
        if self._allocation_mode != "hrp":
            return
        self._strategy_returns[strategy_name].append(ret)
        # Invalidate cached weights so they are recomputed next time
        self._hrp_weights = None

    def _has_hrp_weights(self) -> bool:
        """Return True if HRP weights can be computed (enough history)."""
        if self._allocation_mode != "hrp":
            return False
        if len(self._strategy_returns) < 2:  # noqa: PLR2004
            return False
        min_len = min(len(v) for v in self._strategy_returns.values())
        return min_len >= _HRP_MIN_HISTORY

    def _compute_hrp_overrides(self) -> dict[str, Decimal]:
        """Compute HRP weight overrides from recorded strategy returns."""
        if self._hrp_weights is not None:
            return self._hrp_weights
        names = sorted(self._strategy_returns.keys())
        min_len = min(len(self._strategy_returns[n]) for n in names)
        returns_matrix = [self._strategy_returns[n][:min_len] for n in names]
        raw_weights = compute_hrp_weights(returns_matrix, names)
        self._hrp_weights = {k: Decimal(str(v)) for k, v in raw_weights.items()}
        return self._hrp_weights

    @staticmethod
    def _is_turn_of_month(timestamp: datetime) -> bool:
        """Return True if the date falls in the last 1 or first 3 calendar days of the month."""
        day = timestamp.day
        if day <= 3:  # noqa: PLR2004
            return True
        _, last_day = calendar.monthrange(timestamp.year, timestamp.month)
        return day >= last_day

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

    @staticmethod
    def _compute_hurst_multipliers(candles: list[Candle]) -> tuple[float, dict[str, float]]:
        """Compute Hurst exponent and per-strategy weight multipliers."""
        h = compute_hurst_exponent([float(c.close) for c in candles])
        multipliers: dict[str, float] = {}
        if h > _HURST_TRENDING_THRESHOLD:
            for s in _MOMENTUM_STRATEGIES:
                multipliers[s] = _HURST_BOOST
            for s in _MR_STRATEGIES:
                multipliers[s] = _HURST_SUPPRESS
        elif h < _HURST_MR_THRESHOLD:
            for s in _MOMENTUM_STRATEGIES:
                multipliers[s] = _HURST_SUPPRESS
            for s in _MR_STRATEGIES:
                multipliers[s] = _HURST_BOOST
        return h, multipliers

    def _parse_config(self, config: dict[str, object]) -> tuple[dict[str, object], str, Decimal]:
        """Extract strategies config, normalize mode, and min confidence from config."""
        strategies_cfg_raw = config.get("strategies", {})
        strategies_cfg: dict[str, object] = (
            strategies_cfg_raw if isinstance(strategies_cfg_raw, dict) else {}
        )
        effective_normalize = str(config.get("normalize_mode", self._normalize_mode))
        try:
            effective_min_confidence = Decimal(
                str(config.get("min_combined_confidence", _MIN_COMBINED_CONFIDENCE))
            )
        except InvalidOperation:
            effective_min_confidence = _MIN_COMBINED_CONFIDENCE
        return strategies_cfg, effective_normalize, effective_min_confidence

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

    def _resolve_effective_overrides(
        self,
        weight_overrides: dict[str, Decimal] | None,
    ) -> tuple[dict[str, Decimal] | None, dict[str, Decimal] | None]:
        """Resolve weight overrides, preferring explicit overrides over HRP.

        Returns (effective_overrides, hrp_overrides) tuple.
        """
        hrp_overrides: dict[str, Decimal] | None = None
        if self._has_hrp_weights():
            hrp_overrides = self._compute_hrp_overrides()
        effective = weight_overrides if weight_overrides is not None else hrp_overrides
        return effective, hrp_overrides

    @staticmethod
    def _effective_threshold(
        config: dict[str, object],
        min_confidence: Decimal,
        has_open_position: bool,
        net: Decimal,
    ) -> Decimal:
        """Compute the effective confidence threshold, lowering for exit signals."""
        threshold = min_confidence
        if has_open_position and net < _ZERO:
            exit_conf = Decimal(str(config.get("min_exit_confidence", _MIN_EXIT_CONFIDENCE)))
            threshold = min(min_confidence, exit_conf)
        return threshold

    def generate_signal(  # noqa: PLR0912
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
            if signal is None:
                continue

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
            return None

        denominator = total_enabled_weight if effective_normalize == "total" else total_weight
        if denominator == _ZERO:
            return None
        net = weighted_score / denominator

        # Turn-of-month effect: boost BUY confidence during the window
        if self._is_turn_of_month(candles[-1].timestamp) and net > _ZERO:
            net += _TOM_BUY_BOOST
            feature_contributions["turn_of_month"] = 1.0
        else:
            feature_contributions["turn_of_month"] = 0.0

        # Add HRP weight features when using HRP allocation
        if hrp_overrides is not None:
            for sname, sweight in hrp_overrides.items():
                feature_contributions[f"hrp_weight_{sname}"] = float(sweight)

        threshold = self._effective_threshold(
            config, effective_min_confidence, has_open_position, net
        )
        if abs(net) < threshold:
            # Log when strategies fired but combined score was below threshold
            firing = {
                name: {
                    "direction": (
                        "BUY" if feature_contributions.get(f"{name}_direction", 0) > 0 else "SELL"
                    ),
                    "confidence": feature_contributions.get(f"{name}_confidence", 0),
                }
                for name in self._strategies
                if f"{name}_confidence" in feature_contributions
            }
            if firing:
                logger.debug(
                    "signals_below_threshold",
                    symbol=symbol,
                    firing_strategies=firing,
                    net_score=float(net),
                    threshold=float(threshold),
                )
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
