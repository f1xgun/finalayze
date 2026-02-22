"""Momentum trading strategy using RSI + MACD (Layer 4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_ta as ta
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 30


class MomentumStrategy(BaseStrategy):
    """RSI + MACD momentum strategy.

    Generates BUY signals when RSI is oversold and MACD histogram crosses
    above zero, and SELL signals when RSI is overbought and MACD histogram
    crosses below zero.
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

    def generate_signal(self, symbol: str, candles: list[Candle], segment_id: str) -> Signal | None:
        """Generate a momentum signal from RSI and MACD indicators."""
        if len(candles) < _MIN_CANDLES:
            return None

        indicators = self._compute_indicators(candles, segment_id)
        if indicators is None:
            return None

        current_rsi, current_hist, prev_hist, min_confidence = indicators
        result = self._evaluate_signal(
            current_rsi,
            current_hist,
            prev_hist,
            min_confidence,
            segment_id,
        )
        if result is None:
            return None

        direction, confidence = result
        is_buy = direction == SignalDirection.BUY
        market_id = candles[0].market_id
        rsi_label = "oversold" if is_buy else "overbought"
        cross_label = "above" if is_buy else "below"

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "rsi": round(current_rsi, 2),
                "macd_hist": round(current_hist, 4),
            },
            reasoning=(
                f"RSI={current_rsi:.1f} ({rsi_label}), MACD histogram crossed {cross_label} zero"
            ),
        )

    def _compute_indicators(
        self, candles: list[Candle], segment_id: str
    ) -> tuple[float, float, float, float] | None:
        """Compute RSI and MACD indicators, returning None if data is invalid."""
        params = self.get_parameters(segment_id)
        rsi_period = int(params["rsi_period"])  # type: ignore[call-overload]
        macd_fast = int(params["macd_fast"])  # type: ignore[call-overload]
        macd_slow = int(params["macd_slow"])  # type: ignore[call-overload]
        min_confidence = float(params["min_confidence"])  # type: ignore[arg-type]

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

        return (
            float(current_rsi),
            float(current_hist),
            float(prev_hist),
            min_confidence,
        )

    def _evaluate_signal(
        self,
        current_rsi: float,
        current_hist: float,
        prev_hist: float,
        min_confidence: float,
        segment_id: str,
    ) -> tuple[SignalDirection, float] | None:
        """Evaluate RSI + MACD crossover and return direction + confidence."""
        params = self.get_parameters(segment_id)
        rsi_oversold = float(params["rsi_oversold"])  # type: ignore[arg-type]
        rsi_overbought = float(params["rsi_overbought"])  # type: ignore[arg-type]

        if current_rsi < rsi_oversold and prev_hist < 0 and current_hist > 0:
            direction = SignalDirection.BUY
            rsi_distance = (rsi_oversold - current_rsi) / rsi_oversold
        elif current_rsi > rsi_overbought and prev_hist > 0 and current_hist < 0:
            direction = SignalDirection.SELL
            rsi_distance = (current_rsi - rsi_overbought) / (100.0 - rsi_overbought)
        else:
            return None

        confidence = min(
            1.0,
            0.5 + rsi_distance * 0.3 + abs(current_hist) * 0.1,
        )
        if confidence < min_confidence:
            return None

        return direction, confidence


def _find_histogram_column(macd_df: pd.DataFrame) -> str | None:
    """Find the MACD histogram column by name pattern."""
    for col in macd_df.columns:
        col_lower = str(col).lower()
        if "hist" in col_lower or col_lower.startswith("macdh"):
            return str(col)
    return None
