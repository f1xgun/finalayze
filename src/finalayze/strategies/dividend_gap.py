"""Dividend-gap closure strategy for MOEX stocks (Layer 4).

MOEX stocks often have 8-12% dividend yields, creating large ex-dividend gaps.
These gaps close at predictable rates (1-8 weeks), especially for blue chips
like TATN, FosAgro, SBERP, LKOH.

Strategy logic:
- On ex-dividend date, generate BUY if gap_pct > min_gap_pct and regime is ok.
- Track gap closure: gap_pct = dividend_amount / pre_exdiv_close * 100.
- Exit (SELL) when price recovers to pre-exdiv close OR max_hold_bars reached.
- Expected Sharpe: 1.2-1.8 when regime-filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from datetime import datetime

import structlog

from finalayze.core.schemas import (
    Candle,
    EventType,
    Signal,
    SignalDirection,
    SignalMetadata,
)
from finalayze.strategies.base import BaseStrategy

# S5.2: every dividend_gap signal carries this metadata so the combiner's
# _dedup_event_signals (which gates on EventType.DIVIDEND) collapses
# duplicate same-cycle signals on the same ticker / event.
_DIVIDEND_META = SignalMetadata(event_type=EventType.DIVIDEND)

logger = structlog.get_logger(__name__)

# Confidence scaling constants
_CONFIDENCE_BASE = 0.45
_CONFIDENCE_SCALE = 0.06  # per 1% gap above threshold
_MAX_CONFIDENCE = 0.90

# Yield-based hold bar thresholds and periods
_HIGH_YIELD_PCT = 8.0
_MID_YIELD_PCT = 5.0
_HIGH_YIELD_HOLD_BARS = 60
_MID_YIELD_HOLD_BARS = 40
_LOW_YIELD_HOLD_BARS = 25


@dataclass(frozen=True, slots=True)
class DividendEntry:
    """A single dividend event for a symbol."""

    ex_date: datetime
    amount: float  # dividend per share in local currency
    status: str = "paid"  # paid | cancelled | reduced


@dataclass(slots=True)
class _GapTracker:
    """Internal tracker for an active dividend gap position."""

    ex_date: datetime
    pre_exdiv_close: float
    gap_pct: float
    bars_since_entry: int = 0
    max_hold_bars: int = 40  # default, overridden by yield tier


class DividendGapStrategy(BaseStrategy):
    """Dividend gap closure strategy for high-yield MOEX stocks.

    Buys on ex-dividend date when the gap exceeds a threshold, and exits
    when the gap closes (price recovers) or max hold period is reached.
    """

    _SUPPORTED_SEGMENTS: ClassVar[list[str]] = [
        "ru_blue_chips",
        "ru_energy",
        "ru_finance",
        "ru_tech",
    ]

    def __init__(
        self,
        min_gap_pct: float = 3.0,
        max_hold_bars: int = 40,
        min_confidence: float = 0.40,
    ) -> None:
        self._min_gap_pct = min_gap_pct
        self._max_hold_bars = max_hold_bars
        self._min_confidence = min_confidence

        # symbol -> list of dividend entries
        self._calendar: dict[str, list[DividendEntry]] = {}
        # symbol -> active gap tracker (set on BUY, cleared on SELL/exit)
        self._active_gaps: dict[str, _GapTracker] = {}

    @property
    def name(self) -> str:
        return "dividend_gap"

    def supported_segments(self) -> list[str]:
        return list(self._SUPPORTED_SEGMENTS)

    def get_parameters(self, segment_id: str) -> dict[str, object]:  # noqa: ARG002
        return {
            "min_gap_pct": self._min_gap_pct,
            "max_hold_bars": self._max_hold_bars,
            "min_confidence": self._min_confidence,
        }

    @staticmethod
    def _yield_hold_bars(gap_pct: float) -> int:
        """Map dividend yield (gap %) to max hold bars.

        Higher yields get longer recovery windows:
        - >= 8%: 60 bars (large MOEX dividends need ~3 months)
        - >= 5%: 40 bars (~2 months)
        - < 5%: 25 bars (~5 weeks)
        """
        if gap_pct >= _HIGH_YIELD_PCT:
            return _HIGH_YIELD_HOLD_BARS
        if gap_pct >= _MID_YIELD_PCT:
            return _MID_YIELD_HOLD_BARS
        return _LOW_YIELD_HOLD_BARS

    def add_dividend(self, symbol: str, entry: DividendEntry) -> None:
        """Register a dividend event in the calendar."""
        self._calendar.setdefault(symbol, []).append(entry)

    def reset(self) -> None:
        """Clear all state between backtest runs."""
        self._active_gaps.clear()

    def generate_signal(  # noqa: PLR0911, PLR0912
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,
    ) -> Signal | None:
        """Generate dividend gap signal.

        On ex-div date: BUY if gap > threshold.
        After entry: SELL on gap closure or max_hold_bars reached.
        """
        if len(candles) < 2:  # noqa: PLR2004
            return None

        current_candle = candles[-1]
        current_date = current_candle.timestamp
        current_close = float(current_candle.close)
        market_id = current_candle.market_id

        # --- Check for exit signals first (if we have an active gap) ---
        if has_open_position and symbol in self._active_gaps:
            tracker = self._active_gaps[symbol]

            # Count bars since ex-div date using candle timestamps
            bars_since = sum(1 for c in candles if c.timestamp > tracker.ex_date)
            tracker.bars_since_entry = bars_since

            # Exit 1: gap closure (price recovered to pre-exdiv level)
            if current_close >= tracker.pre_exdiv_close:
                del self._active_gaps[symbol]
                return Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    market_id=market_id,
                    segment_id=segment_id,
                    direction=SignalDirection.SELL,
                    confidence=_MAX_CONFIDENCE,
                    metadata=_DIVIDEND_META,
                    strategy_payload={
                        "gap_pct": round(tracker.gap_pct, 2),
                        "bars_held": float(tracker.bars_since_entry),
                        "recovery_pct": 100.0,
                    },
                    reasoning=(
                        f"Gap closed: price {current_close:.2f} >= "
                        f"pre-exdiv {tracker.pre_exdiv_close:.2f} "
                        f"after {tracker.bars_since_entry} bars"
                    ),
                )

            # Exit 2: max hold period reached (per-entry yield-based limit)
            if tracker.bars_since_entry >= tracker.max_hold_bars:
                if tracker.gap_pct <= 0:
                    recovery_pct = 0.0
                else:
                    recovery_pct = (
                        (current_close - (tracker.pre_exdiv_close * (1 - tracker.gap_pct / 100)))
                        / (tracker.pre_exdiv_close * tracker.gap_pct / 100)
                        * 100
                    )
                del self._active_gaps[symbol]
                return Signal(
                    strategy_name=self.name,
                    symbol=symbol,
                    market_id=market_id,
                    segment_id=segment_id,
                    direction=SignalDirection.SELL,
                    confidence=_CONFIDENCE_BASE,
                    metadata=_DIVIDEND_META,
                    strategy_payload={
                        "gap_pct": round(tracker.gap_pct, 2),
                        "bars_held": float(tracker.bars_since_entry),
                        "recovery_pct": round(recovery_pct, 2),
                    },
                    reasoning=(
                        f"Max hold ({tracker.max_hold_bars} bars) reached, "
                        f"gap recovery {recovery_pct:.1f}%"
                    ),
                )

            # Still holding, no exit signal
            return None

        # --- Check for entry signals (ex-div date match) ---
        dividends = self._calendar.get(symbol, [])
        if not dividends:
            return None

        # Find dividend matching the current candle date (skip cancelled/reduced)
        matching_div: DividendEntry | None = None
        for div in dividends:
            if div.ex_date.date() == current_date.date() and div.status == "paid":
                matching_div = div
                break

        if matching_div is None:
            return None

        # Compute gap percentage
        # Pre-exdiv close is the close of the bar before the current one
        pre_exdiv_close = float(candles[-2].close)
        if pre_exdiv_close <= 0:
            return None

        gap_pct = matching_div.amount / pre_exdiv_close * 100.0

        # Threshold check
        if gap_pct < self._min_gap_pct:
            logger.debug(
                "dividend_gap: gap below threshold",
                symbol=symbol,
                gap_pct=gap_pct,
                min_gap_pct=self._min_gap_pct,
            )
            return None

        # Compute confidence: higher gap -> higher confidence
        confidence = min(
            _MAX_CONFIDENCE,
            _CONFIDENCE_BASE + (gap_pct - self._min_gap_pct) * _CONFIDENCE_SCALE,
        )
        if confidence < self._min_confidence:
            return None

        # Track the gap with yield-based hold bars
        self._active_gaps[symbol] = _GapTracker(
            ex_date=current_date,
            pre_exdiv_close=pre_exdiv_close,
            gap_pct=gap_pct,
            max_hold_bars=self._yield_hold_bars(gap_pct),
        )

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=confidence,
            metadata=_DIVIDEND_META,
            strategy_payload={
                "gap_pct": round(gap_pct, 2),
                "dividend_amount": matching_div.amount,
                "pre_exdiv_close": round(pre_exdiv_close, 2),
            },
            reasoning=(
                f"Ex-div gap {gap_pct:.1f}% "
                f"(div={matching_div.amount}, pre_close={pre_exdiv_close:.2f})"
            ),
        )
