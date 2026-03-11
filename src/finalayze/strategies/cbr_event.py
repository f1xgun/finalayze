"""CBR event strategy for OFZ bonds (Tactical layer).

Trades around CBR rate meetings. Entry 3-5 days before,
exit T+1 or T+2 after. Uses ONLY pre-meeting signals.

Statistical note: ~8 meetings/year x 3 year backtest = ~24 trades.
Insufficient for Sharpe validation (need >=30). Validated via
paper trading and domain judgment.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from finalayze.core.schemas import Signal, SignalDirection
from finalayze.data.fetchers.cbr import days_to_next_cbr, get_next_cbr_meeting

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

# Entry window: 2-7 calendar days before meeting
# Wider than original 3-5 to capture more meeting events
_ENTRY_WINDOW_MIN_DAYS = 2
_ENTRY_WINDOW_MAX_DAYS = 7

# Exit window: 1-2 trading days after meeting
_EXIT_WINDOW_MAX_DAYS = 2

# RUONIA-key rate gap threshold for entry (percentage points).
# gap < -0.15pp means market is pricing in easing (dovish).
# Relaxed from 0.30 to capture more CBR event trades.
_GAP_THRESHOLD = Decimal("0.15")

_STRATEGY_NAME = "cbr_event"
_MARKET_ID = "moex"
_SEGMENT_ID = "ru_ofz_pd"

_BUY_CONFIDENCE = 0.7
_SELL_CONFIDENCE = 0.9

# Weekend days (Saturday=5, Sunday=6)
_SATURDAY = 5


def _skip_weekends(d: date) -> date:
    """Advance *d* to the next Monday if it falls on a weekend."""
    while d.weekday() >= _SATURDAY:
        d += timedelta(days=1)
    return d


class CBREventStrategy:
    """Tactical layer: trade OFZ around CBR rate meetings.

    Entries based on RUONIA-key rate gap (pre-meeting).
    Exits are mechanical (T+1/T+2 after meeting).
    """

    def __init__(
        self,
        preferred_symbols: list[str] | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            preferred_symbols: Bonds to trade around events.
                Default: medium-duration OFZ-PD.
        """
        self._preferred_symbols = preferred_symbols or [
            "SU26244RMFS2",  # 11.25% coupon, ~5Y duration
            "SU26241RMFS8",  # 9.50% coupon, ~4.2Y duration
        ]
        self._in_event_trade: dict[str, date] = {}  # symbol -> entry date
        self._meeting_trade_exits: dict[str, date] = {}  # symbol -> target exit date

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        open_positions: dict[str, Any],
        bar_idx: int,  # noqa: ARG002
        *,
        key_rate: Decimal | None = None,
        ruonia_7d_avg: Decimal | None = None,
        **kwargs: Any,  # noqa: ARG002  # accept extra macro kwargs (cpi_yoy, etc.)
    ) -> Signal | None:
        """Generate event-driven signal around CBR meetings.

        Args:
            symbol: Bond ticker.
            candles: Candles up to current bar.
            open_positions: Currently open positions.
            bar_idx: Current bar index.
            key_rate: Current CBR key rate in percentage points (e.g. 21.00 for 21%).
            ruonia_7d_avg: 7-day average RUONIA rate in percentage points.

        Returns:
            Signal or None.
        """
        if not candles:
            return None

        current_date = candles[-1].timestamp.date()

        # Check if we are in an event trade that should exit
        exit_signal = self._check_exit(symbol, current_date, open_positions)
        if exit_signal is not None:
            return exit_signal

        # Try to generate an entry signal
        return self._check_entry(symbol, current_date, key_rate, ruonia_7d_avg)

    def _check_exit(
        self,
        symbol: str,
        current_date: date,
        open_positions: dict[str, Any],
    ) -> Signal | None:
        """Return a SELL signal if *symbol* has a pending mechanical exit."""
        if symbol not in self._meeting_trade_exits:
            return None
        target_exit = self._meeting_trade_exits[symbol]
        if current_date < target_exit or symbol not in open_positions:
            return None

        del self._meeting_trade_exits[symbol]
        self._in_event_trade.pop(symbol, None)
        return Signal(
            strategy_name=_STRATEGY_NAME,
            symbol=symbol,
            market_id=_MARKET_ID,
            segment_id=_SEGMENT_ID,
            direction=SignalDirection.SELL,
            confidence=_SELL_CONFIDENCE,
            features={"exit_type": 1.0, "days_after_meeting": 1.0},
            reasoning=f"CBR event exit: T+1/T+2 mechanical close for {symbol}",
            instrument_type="bond",
        )

    def _check_entry(
        self,
        symbol: str,
        current_date: date,
        key_rate: Decimal | None,
        ruonia_7d_avg: Decimal | None,
    ) -> Signal | None:
        """Return a BUY signal if entry conditions are met."""
        if symbol not in self._preferred_symbols or symbol in self._in_event_trade:
            return None

        days_remaining = days_to_next_cbr(current_date)
        if (
            days_remaining is None
            or not (_ENTRY_WINDOW_MIN_DAYS <= days_remaining <= _ENTRY_WINDOW_MAX_DAYS)
            or key_rate is None
            or ruonia_7d_avg is None
        ):
            return None

        gap = ruonia_7d_avg - key_rate
        if gap >= -_GAP_THRESHOLD:
            return None

        next_meeting = get_next_cbr_meeting(current_date)
        if next_meeting is None:
            return None

        exit_date = _skip_weekends(next_meeting.date + timedelta(days=_EXIT_WINDOW_MAX_DAYS))
        self._in_event_trade[symbol] = current_date
        self._meeting_trade_exits[symbol] = exit_date

        return Signal(
            strategy_name=_STRATEGY_NAME,
            symbol=symbol,
            market_id=_MARKET_ID,
            segment_id=_SEGMENT_ID,
            direction=SignalDirection.BUY,
            confidence=_BUY_CONFIDENCE,
            features={
                "days_to_meeting": float(days_remaining),
                "ruonia_gap": float(gap),
            },
            reasoning=f"CBR event entry: {days_remaining}d before meeting, gap={float(gap):.3f}",
            instrument_type="bond",
        )
