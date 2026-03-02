"""Dual momentum strategy combining relative and absolute momentum (Layer 4)."""

from __future__ import annotations

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_MIN_CANDLES = 126
_WEIGHT_1M = 0.4
_WEIGHT_3M = 0.3
_WEIGHT_6M = 0.3
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 1.0
_MAX_CONFIDENCE = 0.95
_LOOKBACK_1M = 21
_LOOKBACK_3M = 63
_LOOKBACK_6M = 126

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


class DualMomentumStrategy(BaseStrategy):
    """Dual momentum: weighted relative + absolute momentum gate.

    Combines 1-month, 3-month, and 6-month returns with 40/30/30 weighting.
    Only goes long when the composite momentum score is positive (absolute gate).
    """

    _MAX_POSITIONS = 5

    def __init__(self) -> None:
        self._open_positions: int = 0

    @property
    def name(self) -> str:
        return "dual_momentum"

    def supported_segments(self) -> list[str]:
        return list(_ALL_SEGMENTS)

    def get_parameters(self, segment_id: str) -> dict[str, object]:  # noqa: ARG002
        return {
            "weight_1m": _WEIGHT_1M,
            "weight_3m": _WEIGHT_3M,
            "weight_6m": _WEIGHT_6M,
            "min_candles": _MIN_CANDLES,
            "max_positions": self._MAX_POSITIONS,
        }

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,
        **kwargs: object,  # noqa: ARG002
    ) -> Signal | None:
        """Generate dual momentum signal.

        Args:
            symbol: Ticker symbol.
            candles: OHLCV candles (need >= 126).
            segment_id: Market segment ID.
            sentiment_score: Unused, kept for ABC compatibility.
            has_open_position: Whether caller already holds a position.

        Returns:
            BUY Signal if momentum score > 0, None otherwise.
        """
        if len(candles) < _MIN_CANDLES:
            return None

        # Position cap: skip if at max and no open position to manage
        if not has_open_position and self._open_positions >= self._MAX_POSITIONS:
            return None

        # Weighted momentum score
        close_now = float(candles[-1].close)
        close_1m = float(candles[-_LOOKBACK_1M].close)
        close_3m = float(candles[-_LOOKBACK_3M].close)
        close_6m = float(candles[-_LOOKBACK_6M].close)

        ret_1m = (close_now - close_1m) / close_1m
        ret_3m = (close_now - close_3m) / close_3m
        ret_6m = (close_now - close_6m) / close_6m

        score = ret_1m * _WEIGHT_1M + ret_3m * _WEIGHT_3M + ret_6m * _WEIGHT_6M

        # Absolute momentum gate
        if score <= 0:
            return None

        confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + abs(score) * _CONFIDENCE_SCALE)

        market_id = candles[0].market_id

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=confidence,
            features={
                "score_1m": ret_1m,
                "score_3m": ret_3m,
                "score_6m": ret_6m,
            },
            reasoning=f"Dual momentum score={score:.4f}",
        )
