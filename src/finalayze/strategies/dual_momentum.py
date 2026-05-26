"""Dual momentum strategy combining relative and absolute momentum (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection

# S4.2: previously imported compute_realized_vol from risk.position_sizer,
# which mixes position-sizing concerns with the pure vol calc. risk.regime
# carries the same annualised stdev-of-log-returns helper; it returns
# Decimal(0) for insufficient data which is falsy in the ``or Decimal("0.15")``
# fallback below, matching the previous None semantics.
from finalayze.risk.regime import compute_realized_vol as _compute_rv
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.vol_targeting import compute_vol_scale

_PRESETS_DIR = Path(__file__).parent / "presets"

_MIN_CANDLES = 126
_WEIGHT_1M = 0.4
_WEIGHT_3M = 0.3
_WEIGHT_6M = 0.3
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 1.0
_MAX_CONFIDENCE = 0.95
_VOL_BASELINE = 0.15  # baseline annual vol for confidence normalization
_LOOKBACK_1M = 21
_LOOKBACK_3M = 63
_LOOKBACK_6M = 126
_SELL_THRESHOLD = -0.05
_DEFAULT_NEUTRAL_RESET_BARS = 8

_ALL_SEGMENTS = [
    "us_tech",
    "us_broad",
    "us_healthcare",
    "us_finance",
    "ru_blue_chips",
    "ru_energy",
    "ru_tech",
    "ru_finance",
]


class _SignalState:
    """Tracks signal state per symbol to prevent duplicate signal emission."""

    def __init__(self, neutral_reset_bars: int = _DEFAULT_NEUTRAL_RESET_BARS) -> None:
        self._last_direction: dict[str, SignalDirection] = {}
        self._bars_since_signal: dict[str, int] = {}
        self._neutral_reset_bars = neutral_reset_bars

    def tick(self, symbol: str) -> None:
        """Call once per bar to track time since last signal."""
        if symbol in self._bars_since_signal:
            self._bars_since_signal[symbol] += 1
            if self._bars_since_signal[symbol] >= self._neutral_reset_bars:
                self._last_direction.pop(symbol, None)
                self._bars_since_signal.pop(symbol, None)

    def should_emit(self, symbol: str, direction: SignalDirection) -> bool:
        """Return True if this signal should be emitted (not a duplicate)."""
        last = self._last_direction.get(symbol)
        if last == direction:
            return False
        self._last_direction[symbol] = direction
        self._bars_since_signal[symbol] = 0
        return True


class DualMomentumStrategy(BaseStrategy):
    """Dual momentum: weighted relative + absolute momentum gate.

    Combines 1-month, 3-month, and 6-month returns with 40/30/30 weighting.
    Only goes long when the composite momentum score is positive (absolute gate).

    Parameters are read from YAML presets when available, falling back to
    module-level defaults.
    """

    _MAX_POSITIONS = 5

    def __init__(
        self,
        vol_target_enabled: bool = False,
        vol_target: float = 0.15,
    ) -> None:
        self._open_positions: int = 0
        self._vol_target_enabled = vol_target_enabled
        self._vol_target = vol_target
        # Cache YAML params per segment to avoid reloading on every bar
        self._params_cache: dict[str, dict[str, object]] = {}
        self._signal_states: dict[str, _SignalState] = {}

    def _get_signal_state(self, segment_id: str, neutral_reset_bars: int) -> _SignalState:
        """Get or create a per-segment signal state."""
        if segment_id not in self._signal_states:
            self._signal_states[segment_id] = _SignalState(neutral_reset_bars)
        return self._signal_states[segment_id]

    @property
    def name(self) -> str:
        return "dual_momentum"

    def supported_segments(self) -> list[str]:
        return list(_ALL_SEGMENTS)

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load dual_momentum parameters from the YAML preset for the given segment.

        Results are cached per segment_id to avoid reloading the YAML file on every bar.
        """
        if segment_id in self._params_cache:
            return self._params_cache[segment_id]
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            params = dict(data["strategies"]["dual_momentum"]["params"])
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
        has_open_position: bool = False,
    ) -> Signal | None:
        """Generate dual momentum signal.

        Args:
            symbol: Ticker symbol.
            candles: OHLCV candles (need >= max lookback).
            segment_id: Market segment ID.
            sentiment_score: Unused, kept for ABC compatibility.
            has_open_position: Whether caller already holds a position.

        Returns:
            BUY Signal if momentum score > 0, SELL if score < -0.05, else None.
        """
        params = self.get_parameters(segment_id)

        # Read lookback periods from YAML, falling back to module defaults
        lookback_1m = int(params.get("lookback_1m", _LOOKBACK_1M))  # type: ignore[call-overload]
        lookback_3m = int(params.get("lookback_3m", _LOOKBACK_3M))  # type: ignore[call-overload]
        lookback_6m = int(params.get("lookback_6m", _LOOKBACK_6M))  # type: ignore[call-overload]
        min_confidence = float(params.get("min_confidence", _CONFIDENCE_BASE))  # type: ignore[arg-type]
        weight_1m = float(params.get("weight_1m", _WEIGHT_1M))  # type: ignore[arg-type]
        weight_3m = float(params.get("weight_3m", _WEIGHT_3M))  # type: ignore[arg-type]
        weight_6m = float(params.get("weight_6m", _WEIGHT_6M))  # type: ignore[arg-type]
        neutral_reset_bars = int(
            params.get("neutral_reset_bars", _DEFAULT_NEUTRAL_RESET_BARS)  # type: ignore[call-overload]
        )

        signal_state = self._get_signal_state(segment_id, neutral_reset_bars)
        signal_state.tick(symbol)

        # Adapt minimum candle count to actual lookback
        min_candles = max(lookback_1m, lookback_3m, lookback_6m)
        if len(candles) < min_candles:
            return None

        # Position cap: skip if at max and no open position to manage
        if not has_open_position and self._open_positions >= self._MAX_POSITIONS:
            return None

        # Weighted momentum score
        close_now = float(candles[-1].close)
        close_1m = float(candles[-lookback_1m].close)
        close_3m = float(candles[-lookback_3m].close)
        close_6m = float(candles[-lookback_6m].close)

        ret_1m = (close_now - close_1m) / close_1m
        ret_3m = (close_now - close_3m) / close_3m
        ret_6m = (close_now - close_6m) / close_6m

        score = ret_1m * weight_1m + ret_3m * weight_3m + ret_6m * weight_6m

        # Direction gate: BUY on positive, SELL on meaningfully negative, else None
        sell_threshold = float(
            params.get("sell_threshold", _SELL_THRESHOLD)  # type: ignore[arg-type]
        )
        if score <= sell_threshold:
            direction = SignalDirection.SELL
        elif score <= 0:
            return None
        else:
            direction = SignalDirection.BUY

        # Vol-normalize: same return at higher vol produces lower confidence
        asset_vol = float(_compute_rv(candles) or Decimal("0.15"))
        normalized_score = abs(score) / max(asset_vol, 0.01) * _VOL_BASELINE
        confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + normalized_score * _CONFIDENCE_SCALE)

        # Min confidence filter
        if confidence < min_confidence:
            return None

        # Volatility targeting: scale confidence by target_vol / realized_vol
        vol_target_enabled = bool(params.get("vol_target_enabled", self._vol_target_enabled))
        if vol_target_enabled:
            vol_target = float(params.get("vol_target", self._vol_target))  # type: ignore[arg-type]
            closes = [float(c.close) for c in candles]
            vol_scale = compute_vol_scale(closes, target_vol=vol_target)
            confidence = min(1.0, max(0.0, confidence * vol_scale))

        # Deduplicate: don't emit same direction on consecutive bars
        if not signal_state.should_emit(symbol, direction):
            return None

        market_id = candles[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            strategy_payload={
                "score_1m": ret_1m,
                "score_3m": ret_3m,
                "score_6m": ret_6m,
            },
            reasoning=f"Dual momentum score={score:.4f}",
        )
