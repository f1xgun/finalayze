"""RSI(2) Connors mean-reversion strategy (Layer 4).

Short-term mean-reversion strategy using a 2-period RSI:
- BUY when RSI(2) < 10 AND price > SMA(200)
- SELL when RSI(2) > 90 AND price < SMA(200)
- Exit when price crosses 5-day SMA

Confidence scaling:
- BUY:  (10 - rsi2) / 10 * 0.8 + 0.2  -> range [0.2, 1.0]
- SELL: (rsi2 - 90) / 10 * 0.8 + 0.2  -> range [0.2, 1.0]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_ta as ta
import structlog
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)

_PRESETS_DIR = Path(__file__).parent / "presets"

# Default parameter values
_DEFAULT_RSI_PERIOD = 2
_DEFAULT_RSI_BUY_THRESHOLD = 10.0
_DEFAULT_RSI_SELL_THRESHOLD = 90.0
_DEFAULT_SMA_TREND_PERIOD = 200
_DEFAULT_SMA_EXIT_PERIOD = 5
_DEFAULT_MIN_CONFIDENCE = 0.35


class RSI2ConnorsStrategy(BaseStrategy):
    """RSI(2) Connors short-term mean-reversion strategy.

    Uses a very short-term RSI (period=2) to identify oversold/overbought
    conditions, filtered by a long-term SMA(200) trend gate.
    """

    def __init__(self) -> None:
        self._params_cache: dict[str, dict[str, object]] = {}

    @property
    def name(self) -> str:
        return "rsi2_connors"

    def supported_segments(self) -> list[str]:
        """Return segment IDs where rsi2_connors strategy is enabled."""
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
                cfg = strats.get("rsi2_connors", {}) if isinstance(strats, dict) else {}
                if cfg.get("enabled", False):
                    segments.append(str(segment_id))
            except (OSError, yaml.YAMLError):
                continue
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load rsi2_connors parameters from the YAML preset (cached per segment)."""
        if segment_id in self._params_cache:
            return self._params_cache[segment_id]
        try:
            preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
            with preset_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return {}
            strategies = data.get("strategies", {})
            if not isinstance(strategies, dict):
                return {}
            cfg = strategies.get("rsi2_connors", {})
            if not isinstance(cfg, dict):
                return {}
            params = cfg.get("params", {})
            result = dict(params) if isinstance(params, dict) else {}
            self._params_cache[segment_id] = result
            return result
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return {}

    def generate_signal(  # noqa: PLR0911
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
    ) -> Signal | None:
        """Generate RSI(2) Connors signal."""
        params = self.get_parameters(segment_id)
        rsi_period = int(params.get("rsi_period", _DEFAULT_RSI_PERIOD))  # type: ignore[call-overload]
        rsi_buy_threshold = float(params.get("rsi_buy_threshold", _DEFAULT_RSI_BUY_THRESHOLD))  # type: ignore[arg-type]
        rsi_sell_threshold = float(params.get("rsi_sell_threshold", _DEFAULT_RSI_SELL_THRESHOLD))  # type: ignore[arg-type]
        sma_trend_period = int(params.get("sma_trend_period", _DEFAULT_SMA_TREND_PERIOD))  # type: ignore[call-overload]
        min_confidence = float(params.get("min_confidence", _DEFAULT_MIN_CONFIDENCE))  # type: ignore[arg-type]

        # Need enough candles for SMA(trend_period)
        if len(candles) < sma_trend_period + 1:
            logger.debug(
                "rsi2_connors: insufficient data",
                symbol=symbol,
                candles=len(candles),
                required=sma_trend_period + 1,
            )
            return None

        closes = pd.Series([float(c.close) for c in candles])
        current_close = float(candles[-1].close)

        # Compute SMA(200) trend filter
        sma_trend = closes.rolling(sma_trend_period).mean()
        current_sma_trend = sma_trend.iloc[-1]
        if pd.isna(current_sma_trend):
            return None
        current_sma_trend = float(current_sma_trend)

        # Compute RSI(2)
        rsi_series = ta.rsi(closes, length=rsi_period)
        if rsi_series is None or rsi_series.isna().all():
            return None
        current_rsi = rsi_series.iloc[-1]
        if pd.isna(current_rsi):
            return None
        current_rsi = float(current_rsi)

        # Determine direction
        direction: SignalDirection | None = None
        confidence: float = 0.0

        if current_rsi < rsi_buy_threshold and current_close > current_sma_trend:
            direction = SignalDirection.BUY
            confidence = self._compute_buy_confidence(current_rsi)
        elif current_rsi > rsi_sell_threshold and current_close < current_sma_trend:
            direction = SignalDirection.SELL
            confidence = self._compute_sell_confidence(current_rsi)

        if direction is None:
            return None

        # Apply min_confidence filter
        if confidence < min_confidence:
            logger.debug(
                "rsi2_connors: below min_confidence",
                symbol=symbol,
                confidence=confidence,
                min_confidence=min_confidence,
            )
            return None

        market_id = candles[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "rsi2": round(current_rsi, 2),
                "sma_trend": round(current_sma_trend, 2),
                "close": round(current_close, 2),
            },
            reasoning=(
                f"RSI(2)={current_rsi:.1f} "
                f"({'oversold' if direction == SignalDirection.BUY else 'overbought'}), "
                f"price={current_close:.2f} "
                f"{'>' if direction == SignalDirection.BUY else '<'} "
                f"SMA({sma_trend_period})={current_sma_trend:.2f}"
            ),
        )

    def _compute_buy_confidence(self, rsi2: float) -> float:
        """Compute BUY confidence from RSI(2) value.

        Formula: (10 - rsi2) / 10 * 0.8 + 0.2
        Range: [0.2, 1.0] for rsi2 in [0, 10]
        """
        raw = (10.0 - rsi2) / 10.0
        return min(1.0, max(0.0, raw * 0.8 + 0.2))

    def _compute_sell_confidence(self, rsi2: float) -> float:
        """Compute SELL confidence from RSI(2) value.

        Formula: (rsi2 - 90) / 10 * 0.8 + 0.2
        Range: [0.2, 1.0] for rsi2 in [90, 100]
        """
        raw = (rsi2 - 90.0) / 10.0
        return min(1.0, max(0.0, raw * 0.8 + 0.2))
