"""Momentum trading strategy using RSI + MACD with regime lookback (Layer 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 30
_DEFAULT_LOOKBACK_BARS = 5
_DEFAULT_NEUTRAL_RESET_BARS = 20


@dataclass(frozen=True, slots=True)
class _Indicators:
    current_rsi: float
    rsi_window: list[float]
    current_hist: float
    prev_hist: float
    current_close: float
    min_confidence: float
    current_sma: float | None
    current_adx: float | None
    volume_ratio: float | None


class _SignalState:
    """Tracks signal state per symbol to prevent duplicate signals."""

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


class MomentumStrategy(BaseStrategy):
    """RSI + MACD momentum strategy with regime lookback.

    Generates BUY signals when RSI was recently oversold (within lookback window)
    and MACD histogram is rising, and SELL signals when RSI was recently overbought
    and MACD histogram is falling.
    """

    def __init__(self) -> None:
        self._signal_state = _SignalState()

    @property
    def name(self) -> str:
        return "momentum"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where momentum strategy is enabled."""
        segments: list[str] = []
        for preset_path in sorted(_PRESETS_DIR.glob("*.yaml")):
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            strategies = data.get("strategies", {})
            momentum_cfg = strategies.get("momentum", {})
            if momentum_cfg.get("enabled", False):
                segments.append(data["segment_id"])
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load momentum parameters from the YAML preset for the given segment."""
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            return dict(data["strategies"]["momentum"]["params"])
        except FileNotFoundError:
            return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
    ) -> Signal | None:
        """Generate a momentum signal from RSI and MACD indicators."""
        params = self.get_parameters(segment_id)
        neutral_reset_bars = int(
            params.get("neutral_reset_bars", _DEFAULT_NEUTRAL_RESET_BARS)  # type: ignore[call-overload]
        )
        self._signal_state._neutral_reset_bars = neutral_reset_bars

        self._signal_state.tick(symbol)

        if len(candles) < _MIN_CANDLES:
            return None

        indicators = self._compute_indicators(candles, segment_id)
        if indicators is None:
            return None

        result = self._evaluate_signal(indicators, segment_id)
        if result is None:
            return None

        direction, confidence = result

        if not self._signal_state.should_emit(symbol, direction):
            return None

        is_buy = direction == SignalDirection.BUY
        market_id = candles[0].market_id
        rsi_label = "oversold" if is_buy else "overbought"
        hist_label = "rising" if is_buy else "falling"

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "rsi": round(indicators.current_rsi, 2),
                "macd_hist": round(indicators.current_hist, 4),
            },
            reasoning=(
                f"RSI={indicators.current_rsi:.1f} (recently {rsi_label}), "
                f"MACD histogram {hist_label}"
            ),
        )

    def _compute_indicators(self, candles: list[Candle], segment_id: str) -> _Indicators | None:
        """Compute RSI and MACD indicators, returning None if data is invalid."""
        params = self.get_parameters(segment_id)
        rsi_period = int(params["rsi_period"])  # type: ignore[call-overload]
        macd_fast = int(params["macd_fast"])  # type: ignore[call-overload]
        macd_slow = int(params["macd_slow"])  # type: ignore[call-overload]
        min_confidence = float(params["min_confidence"])  # type: ignore[arg-type]
        lookback_bars = int(params.get("lookback_bars", _DEFAULT_LOOKBACK_BARS))  # type: ignore[call-overload]

        closes = pd.Series([float(c.close) for c in candles])

        rsi_series = ta.rsi(closes, length=rsi_period)
        if rsi_series is None or rsi_series.isna().all():
            return None

        macd_df = ta.macd(closes, fast=macd_fast, slow=macd_slow)
        if macd_df is None:
            return None

        hist_col = _find_histogram_column(macd_df)
        if hist_col is None:
            return None

        hist = macd_df[hist_col]
        current_rsi = rsi_series.iloc[-1]
        current_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2]

        if pd.isna(current_rsi) or pd.isna(current_hist) or pd.isna(prev_hist):
            return None

        rsi_window = [float(v) for v in rsi_series.iloc[-lookback_bars:] if not pd.isna(v)]
        current_close = float(candles[-1].close)

        # SMA for trend filter
        current_sma: float | None = None
        trend_sma_period = int(params.get("trend_sma_period", 50))  # type: ignore[call-overload]
        if len(closes) >= trend_sma_period:
            sma_series = closes.rolling(trend_sma_period).mean()
            sma_val = sma_series.iloc[-1]
            if not pd.isna(sma_val):
                current_sma = float(sma_val)

        # ADX for regime filter
        current_adx: float | None = None
        adx_filter = bool(params.get("adx_filter", False))
        if adx_filter:
            adx_period = int(params.get("adx_period", 14))  # type: ignore[call-overload]
            high_series = pd.Series([float(c.high) for c in candles])
            low_series = pd.Series([float(c.low) for c in candles])
            adx_df = ta.adx(high_series, low_series, closes, length=adx_period)
            if adx_df is not None:
                adx_col = f"ADX_{adx_period}"
                if adx_col in adx_df.columns:
                    adx_val = adx_df[adx_col].iloc[-1]
                    if not pd.isna(adx_val):
                        current_adx = float(adx_val)

        # Volume ratio for volume confirmation
        volume_ratio: float | None = None
        volume_filter = bool(params.get("volume_filter", False))
        if volume_filter:
            volume_sma_period = int(params.get("volume_sma_period", 20))  # type: ignore[call-overload]
            volumes = pd.Series([c.volume for c in candles])
            vol_sma = volumes.rolling(volume_sma_period).mean().iloc[-1]
            if not pd.isna(vol_sma) and float(vol_sma) > 0:
                volume_ratio = float(candles[-1].volume) / float(vol_sma)

        return _Indicators(
            current_rsi=float(current_rsi),
            rsi_window=rsi_window,
            current_hist=float(current_hist),
            prev_hist=float(prev_hist),
            current_close=current_close,
            min_confidence=min_confidence,
            current_sma=current_sma,
            current_adx=current_adx,
            volume_ratio=volume_ratio,
        )

    def _evaluate_signal(  # noqa: PLR0911
        self,
        indicators: _Indicators,
        segment_id: str,
    ) -> tuple[SignalDirection, float] | None:
        """Evaluate RSI regime + MACD histogram trend and return direction + confidence."""
        params = self.get_parameters(segment_id)
        rsi_oversold = float(params["rsi_oversold"])  # type: ignore[arg-type]
        rsi_overbought = float(params["rsi_overbought"])  # type: ignore[arg-type]

        recently_oversold = any(v < rsi_oversold for v in indicators.rsi_window)
        recently_overbought = any(v > rsi_overbought for v in indicators.rsi_window)
        hist_rising = indicators.current_hist > indicators.prev_hist
        hist_falling = indicators.current_hist < indicators.prev_hist

        if recently_oversold and hist_rising and indicators.current_rsi < rsi_overbought:
            direction = SignalDirection.BUY
            rsi_distance = (rsi_oversold - min(indicators.rsi_window)) / rsi_oversold
        elif recently_overbought and hist_falling and indicators.current_rsi > rsi_oversold:
            direction = SignalDirection.SELL
            rsi_distance = (max(indicators.rsi_window) - rsi_overbought) / (100.0 - rsi_overbought)
        else:
            return None

        # ADX regime filter: suppress in range-bound markets
        adx_filter = bool(params.get("adx_filter", False))
        if adx_filter and indicators.current_adx is not None:
            adx_threshold = float(params.get("adx_threshold", 25))  # type: ignore[arg-type]
            if indicators.current_adx < adx_threshold:
                return None

        # Volume confirmation filter
        volume_filter = bool(params.get("volume_filter", False))
        if volume_filter and indicators.volume_ratio is not None:
            volume_min_ratio = float(params.get("volume_min_ratio", 1.0))  # type: ignore[arg-type]
            if indicators.volume_ratio < volume_min_ratio:
                return None

        # Trend filter (SMA gate): suppress counter-trend signals
        trend_filter = bool(params.get("trend_filter", False))
        if trend_filter and indicators.current_sma is not None:
            trend_sma_buffer_pct = float(params.get("trend_sma_buffer_pct", 2.0))  # type: ignore[arg-type]
            buffer = indicators.current_sma * trend_sma_buffer_pct / 100.0
            if (
                indicators.current_close > indicators.current_sma + buffer
                and direction == SignalDirection.SELL
            ):
                return None
            if (
                indicators.current_close < indicators.current_sma - buffer
                and direction == SignalDirection.BUY
            ):
                return None

        confidence = min(1.0, 0.5 + rsi_distance * 0.5)
        if confidence < indicators.min_confidence:
            return None

        return direction, confidence


def _find_histogram_column(macd_df: pd.DataFrame) -> str | None:
    """Find the MACD histogram column by name pattern."""
    for col in macd_df.columns:
        col_lower = str(col).lower()
        if "hist" in col_lower or col_lower.startswith("macdh"):
            return str(col)
    return None
