"""Mean reversion strategy using Bollinger Bands (Layer 4)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_DEFAULT_BB_PERIOD = 20
_DEFAULT_BB_STD = Decimal("2.0")
_DEFAULT_MIN_CONFIDENCE = Decimal("0.55")
_DEFAULT_SQUEEZE_THRESHOLD = 0.02
_DEFAULT_MIN_BAND_DISTANCE_PCT = 0.005
_DEFAULT_RSI_OVERSOLD_MR = 40
_DEFAULT_RSI_OVERBOUGHT_MR = 60
_DEFAULT_RSI_PERIOD = 14
_CONFIDENCE_BASE = 0.5
_CONFIDENCE_DISTANCE_MULTIPLIER = 2.0


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy: BUY below lower BB, SELL above upper BB.

    Tracks signal state per symbol to avoid emitting repeated signals while price
    stays outside the band.  The active signal is cleared when price returns inside
    the Bollinger Bands (neutral zone).
    """

    def __init__(self) -> None:
        # Maps symbol -> last emitted direction (None = no active signal)
        self._active_signal: dict[str, SignalDirection] = {}

    @property
    def name(self) -> str:
        return "mean_reversion"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where mean_reversion strategy is enabled."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            try:
                with preset_path.open() as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                segment_id = data.get("segment_id")
                if segment_id is None:
                    continue
                strats = data.get("strategies", {})
                mr_cfg = strats.get("mean_reversion", {}) if isinstance(strats, dict) else {}
                if mr_cfg.get("enabled", False):
                    segments.append(str(segment_id))
            except (OSError, yaml.YAMLError):
                continue
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load mean_reversion parameters from the YAML preset for the given segment."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return {}
            strategies = data.get("strategies", {})
            if not isinstance(strategies, dict):
                return {}
            mr_cfg = strategies.get("mean_reversion", {})
            if not isinstance(mr_cfg, dict):
                return {}
            params = mr_cfg.get("params", {})
            return dict(params) if isinstance(params, dict) else {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}

    def generate_signal(  # noqa: PLR0911, PLR0912
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
    ) -> Signal | None:
        """Generate a mean reversion signal using Bollinger Bands."""
        params = self.get_parameters(segment_id)
        bb_period_raw = params.get("bb_period", _DEFAULT_BB_PERIOD)
        if isinstance(bb_period_raw, (int, float, str)):
            bb_period = int(bb_period_raw)
        else:
            bb_period = _DEFAULT_BB_PERIOD
        bb_std = Decimal(str(params.get("bb_std_dev", _DEFAULT_BB_STD)))
        min_confidence = Decimal(str(params.get("min_confidence", _DEFAULT_MIN_CONFIDENCE)))

        squeeze_threshold = float(params.get("squeeze_threshold", _DEFAULT_SQUEEZE_THRESHOLD))  # type: ignore[arg-type]
        min_band_distance_pct = float(
            params.get("min_band_distance_pct", _DEFAULT_MIN_BAND_DISTANCE_PCT)  # type: ignore[arg-type]
        )
        rsi_oversold_mr = float(params.get("rsi_oversold_mr", _DEFAULT_RSI_OVERSOLD_MR))  # type: ignore[arg-type]
        rsi_overbought_mr = float(params.get("rsi_overbought_mr", _DEFAULT_RSI_OVERBOUGHT_MR))  # type: ignore[arg-type]
        rsi_period = int(params.get("rsi_period", _DEFAULT_RSI_PERIOD))  # type: ignore[call-overload]

        bb_values = _compute_bb_values(candles, bb_period, float(bb_std))
        if bb_values is None:
            return None
        lower, upper, mid, last_close = bb_values

        # Squeeze filter: skip signals when bands are too narrow (low volatility)
        band_width_pct = (upper - lower) / mid if mid > 0 else 0.0
        if band_width_pct < squeeze_threshold:
            return None

        exit_at_mean = bool(params.get("exit_at_mean", False))

        band_width = upper - lower
        direction: SignalDirection | None = None
        confidence: float = 0.0

        if last_close < lower:
            # Minimum band distance filter
            distance_pct = (lower - last_close) / lower if lower > 0 else 0.0
            if distance_pct < min_band_distance_pct:
                self._active_signal.pop(symbol, None)
                return None
            direction = SignalDirection.BUY
            distance = (lower - last_close) / band_width
            confidence = min(1.0, _CONFIDENCE_BASE + distance * _CONFIDENCE_DISTANCE_MULTIPLIER)
        elif last_close > upper:
            # Minimum band distance filter
            distance_pct = (last_close - upper) / upper if upper > 0 else 0.0
            if distance_pct < min_band_distance_pct:
                self._active_signal.pop(symbol, None)
                return None
            direction = SignalDirection.SELL
            distance = (last_close - upper) / band_width
            confidence = min(1.0, _CONFIDENCE_BASE + distance * _CONFIDENCE_DISTANCE_MULTIPLIER)
        else:
            # Price has returned inside the bands
            active = self._active_signal.pop(symbol, None)
            if exit_at_mean and active is not None:
                # Emit exit signal: reverse direction to close position
                exit_direction = (
                    SignalDirection.SELL
                    if active == SignalDirection.BUY
                    else SignalDirection.BUY
                )
                mid_distance = abs(last_close - mid) / band_width if band_width > 0 else 0.0
                exit_confidence = min(1.0, 0.6 + (1.0 - mid_distance) * 0.3)
                if Decimal(str(exit_confidence)) >= min_confidence:
                    return Signal(
                        strategy_name=self.name,
                        symbol=symbol,
                        market_id=candles[0].market_id,
                        segment_id=segment_id,
                        direction=exit_direction,
                        confidence=exit_confidence,
                        features={
                            "bb_lower": lower,
                            "bb_upper": upper,
                            "bb_mid": mid,
                            "close": last_close,
                            "exit_at_mean": 1.0,
                        },
                        reasoning=(
                            f"Price {last_close:.2f} returned to mean region "
                            f"BB [{lower:.2f}, {upper:.2f}] (exit at mean)"
                        ),
                    )

        if direction is None or Decimal(str(confidence)) < min_confidence:
            return None

        # RSI confirmation filter
        rsi_value = _compute_rsi(candles, rsi_period)
        if rsi_value is not None:
            if direction == SignalDirection.BUY and rsi_value > rsi_oversold_mr:
                return None
            if direction == SignalDirection.SELL and rsi_value < rsi_overbought_mr:
                return None

        # Suppress repeated signals in the same direction while price stays outside band
        if self._active_signal.get(symbol) == direction:
            return None

        self._active_signal[symbol] = direction

        market_id = candles[0].market_id
        band_label = "lower" if direction == SignalDirection.BUY else "upper"

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "bb_lower": lower,
                "bb_upper": upper,
                "bb_mid": mid,
                "close": last_close,
            },
            reasoning=(
                f"Price {last_close:.2f} "
                f"{'below lower' if direction == SignalDirection.BUY else 'above upper'}"
                f" BB [{lower:.2f}, {upper:.2f}] ({band_label} band breach)"
            ),
        )


def _compute_bb_values(
    candles: list[Candle], bb_period: int, bb_std: float
) -> tuple[float, float, float, float] | None:
    """Compute Bollinger Band values from candles.

    Returns (lower, upper, mid, last_close) or None if computation fails.
    """
    if len(candles) < bb_period + 1:
        return None
    closes = pd.Series([float(c.close) for c in candles])
    bb = ta.bbands(closes, length=bb_period, lower_std=bb_std, upper_std=bb_std)
    if bb is None or bb.empty:
        return None
    lower_col = _find_bb_column(bb, "BBL_")
    upper_col = _find_bb_column(bb, "BBU_")
    mid_col = _find_bb_column(bb, "BBM_")
    if lower_col is None or upper_col is None or mid_col is None:
        return None
    lower = float(bb.iloc[-1][lower_col])
    upper = float(bb.iloc[-1][upper_col])
    mid = float(bb.iloc[-1][mid_col])
    if upper - lower <= 0:
        return None
    return lower, upper, mid, float(candles[-1].close)


def _find_bb_column(bb: pd.DataFrame, prefix: str) -> str | None:
    """Find a Bollinger Band column by prefix (handles pandas_ta version differences)."""
    for col in bb.columns:
        if str(col).startswith(prefix):
            return str(col)
    return None


def _compute_rsi(candles: list[Candle], period: int) -> float | None:
    """Compute current RSI value from candles. Returns None if not enough data."""
    if len(candles) < period + 1:
        return None
    closes = pd.Series([float(c.close) for c in candles])
    rsi_series = ta.rsi(closes, length=period)
    if rsi_series is None or rsi_series.isna().all():
        return None
    val = rsi_series.iloc[-1]
    if np.isnan(val):
        return None
    return float(val)
