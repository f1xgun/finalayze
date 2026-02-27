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


@dataclass(frozen=True, slots=True)
class _Indicators:
    current_rsi: float
    rsi_window: list[float]
    current_hist: float
    prev_hist: float
    current_close: float
    min_confidence: float


class MomentumStrategy(BaseStrategy):
    """RSI + MACD momentum strategy with regime lookback.

    Generates BUY signals when RSI was recently oversold (within lookback window)
    and MACD histogram is rising, and SELL signals when RSI was recently overbought
    and MACD histogram is falling.
    """

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
        if len(candles) < _MIN_CANDLES:
            return None

        indicators = self._compute_indicators(candles, segment_id)
        if indicators is None:
            return None

        result = self._evaluate_signal(indicators, segment_id)
        if result is None:
            return None

        direction, confidence = result
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

        return _Indicators(
            current_rsi=float(current_rsi),
            rsi_window=rsi_window,
            current_hist=float(current_hist),
            prev_hist=float(prev_hist),
            current_close=current_close,
            min_confidence=min_confidence,
        )

    def _evaluate_signal(
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

        confidence = min(
            1.0,
            0.5
            + rsi_distance * 0.3
            + (abs(indicators.current_hist) / indicators.current_close * 100) * 0.1,
        )
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
