"""Post-Earnings Announcement Drift (PEAD) strategy (Layer 4).

Stocks drift in the direction of earnings surprises for 60-90 days
post-announcement. This effect is stronger in emerging markets and
mid-caps due to lower institutional coverage.

Strategy logic:
- Register earnings surprises via add_earnings_surprise().
- After announcement: BUY if sue_score > positive_threshold, SELL if < negative_threshold.
- Signals remain active for drift_window_bars after announcement.
- Confidence scales with |sue_score| magnitude.
- Both US and MOEX markets supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from datetime import datetime

import structlog

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)

# Confidence scaling constants
_CONFIDENCE_BASE = 0.35
_CONFIDENCE_SCALE = 0.10  # per unit of |sue_score| above threshold
_MAX_CONFIDENCE = 0.90


@dataclass(frozen=True, slots=True)
class EarningsSurprise:
    """A single earnings surprise event for a symbol."""

    symbol: str
    announcement_date: datetime
    sue_score: float  # Standardized Unexpected Earnings
    actual_eps: float
    expected_eps: float


class PEADStrategy(BaseStrategy):
    """Post-Earnings Announcement Drift strategy.

    Generates BUY signals for positive earnings surprises and SELL signals
    for negative surprises. Signals are active for a configurable drift
    window after the earnings announcement date.
    """

    _SUPPORTED_SEGMENTS: ClassVar[list[str]] = [
        "us_tech",
        "us_broad",
        "us_healthcare",
        "us_finance",
        "ru_blue_chips",
        "ru_energy",
        "ru_finance",
        "ru_tech",
    ]

    def __init__(
        self,
        positive_threshold: float = 1.0,
        negative_threshold: float = -1.0,
        drift_window_bars: int = 60,
        min_confidence: float = 0.35,
    ) -> None:
        self._positive_threshold = positive_threshold
        self._negative_threshold = negative_threshold
        self._drift_window_bars = drift_window_bars
        self._min_confidence = min_confidence

        # symbol -> list of earnings surprises
        self._surprises: dict[str, list[EarningsSurprise]] = {}

    @property
    def name(self) -> str:
        return "pead"

    def supported_segments(self) -> list[str]:
        return list(self._SUPPORTED_SEGMENTS)

    def get_parameters(self, segment_id: str) -> dict[str, object]:  # noqa: ARG002
        return {
            "positive_threshold": self._positive_threshold,
            "negative_threshold": self._negative_threshold,
            "drift_window_bars": self._drift_window_bars,
            "min_confidence": self._min_confidence,
        }

    def add_earnings_surprise(self, surprise: EarningsSurprise) -> None:
        """Register an earnings surprise event."""
        self._surprises.setdefault(surprise.symbol, []).append(surprise)

    def reset(self) -> None:
        """Clear all state between backtest runs."""
        self._surprises.clear()

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> Signal | None:
        """Generate PEAD signal based on earnings surprise data.

        Checks all registered surprises for the symbol and generates a signal
        if the current candle falls within the drift window of any surprise
        that exceeds the threshold.
        """
        if not candles:
            return None

        surprises = self._surprises.get(symbol)
        if not surprises:
            return None

        current_candle = candles[-1]
        current_date = current_candle.timestamp
        market_id = current_candle.market_id

        # Find the most recent applicable surprise
        best_surprise: EarningsSurprise | None = None
        best_bars_since: int | None = None

        for surprise in surprises:
            # Current candle must be on or after announcement date
            if current_date.date() < surprise.announcement_date.date():
                continue

            # Count bars since announcement
            bars_since = sum(
                1 for c in candles if c.timestamp.date() > surprise.announcement_date.date()
            )

            # Check drift window (signal active for drift_window_bars after announcement)
            if bars_since > self._drift_window_bars:
                continue

            # Pick the most recent surprise
            if best_surprise is None or (
                surprise.announcement_date > best_surprise.announcement_date
            ):
                best_surprise = surprise
                best_bars_since = bars_since

        if best_surprise is None or best_bars_since is None:
            return None

        sue = best_surprise.sue_score

        # Determine direction based on thresholds
        direction: SignalDirection | None = None
        if sue > self._positive_threshold:
            direction = SignalDirection.BUY
        elif sue < self._negative_threshold:
            direction = SignalDirection.SELL
        else:
            return None

        # Compute confidence scaled by |sue_score| magnitude
        excess = abs(sue) - abs(
            self._positive_threshold
            if direction == SignalDirection.BUY
            else self._negative_threshold
        )
        confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + excess * _CONFIDENCE_SCALE)

        if confidence < self._min_confidence:
            logger.debug(
                "pead: below min_confidence",
                symbol=symbol,
                confidence=confidence,
                min_confidence=self._min_confidence,
            )
            return None

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={
                "sue_score": best_surprise.sue_score,
                "actual_eps": best_surprise.actual_eps,
                "expected_eps": best_surprise.expected_eps,
                "bars_since_announcement": float(best_bars_since),
            },
            reasoning=(
                f"PEAD: SUE={sue:.2f} "
                f"({'positive' if direction == SignalDirection.BUY else 'negative'} surprise), "
                f"{best_bars_since} bars post-announcement "
                f"(window={self._drift_window_bars})"
            ),
        )
