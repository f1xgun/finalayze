"""Unit tests for CBRStrategyWrapper (Layer 4 strategy)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.cbr_calendar import CBRCalendar, CBRRateEvent
from finalayze.strategies.cbr_strategy_wrapper import CBRStrategyWrapper

# ── Named constants (no magic numbers) ───────────────────────────────────────

EXPECTED_SEGMENTS_COUNT = 4
EXPECTED_SEGMENTS = ["ru_blue_chips", "ru_energy", "ru_finance", "ru_tech"]
EXPECTED_STRATEGY_NAME = "cbr_calendar"

SURPRISE_HIKE_BPS = 100
SURPRISE_CUT_BPS = -100
NO_SURPRISE_BPS = 0

RATE_DECISION = 17.0
EXPECTED_RATE = 16.0

EVENT_DATE = date(2026, 2, 1)

BARS_SINCE_ZERO = 0
BARS_SINCE_CONTRARIAN = 4  # within [3, 5] window

MAX_BARS_SINCE_EVENT = 10

SBER_SYMBOL = "SBER"
VTBR_SYMBOL = "VTBR"
UNAFFECTED_SYMBOL = "GAZP"

CANDLE_TIMEFRAME = "1d"
CANDLE_MARKET_ID = "moex"
CANDLE_OPEN = Decimal("300.00")
CANDLE_HIGH = Decimal("305.00")
CANDLE_LOW = Decimal("298.00")
CANDLE_CLOSE = Decimal("302.00")
CANDLE_VOLUME = 1_000_000


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candle(symbol: str, ts: datetime) -> Candle:
    """Create a minimal valid Candle for testing."""
    return Candle(
        symbol=symbol,
        market_id=CANDLE_MARKET_ID,
        timeframe=CANDLE_TIMEFRAME,
        timestamp=ts,
        open=CANDLE_OPEN,
        high=CANDLE_HIGH,
        low=CANDLE_LOW,
        close=CANDLE_CLOSE,
        volume=CANDLE_VOLUME,
    )


def _event_day_ts() -> datetime:
    """Return a datetime on EVENT_DATE with UTC timezone."""
    return datetime(
        EVENT_DATE.year,
        EVENT_DATE.month,
        EVENT_DATE.day,
        10,
        tzinfo=UTC,
    )


def _candles_on_event_day(symbol: str, count: int = 1) -> list[Candle]:
    """Return ``count`` candles whose date equals EVENT_DATE (bars_since=0)."""
    ts = _event_day_ts()
    return [_make_candle(symbol, ts) for _ in range(count)]


def _candles_past_event(symbol: str, bars_past: int) -> list[Candle]:
    """Return candles where ``bars_past`` of them are strictly after EVENT_DATE."""
    candles: list[Candle] = [_make_candle(symbol, _event_day_ts())]
    for offset in range(1, bars_past + 1):
        day = EVENT_DATE + timedelta(days=offset)
        ts = datetime.combine(day, datetime.min.time()).replace(
            tzinfo=UTC,
        )
        candles.append(_make_candle(symbol, ts))
    return candles


def _make_calendar(surprise_bps: int = SURPRISE_HIKE_BPS) -> CBRCalendar:
    """Create a CBRCalendar with a single event on EVENT_DATE."""
    cal = CBRCalendar()
    cal.add_event(
        CBRRateEvent(
            date=EVENT_DATE,
            rate_decision=RATE_DECISION,
            expected_rate=EXPECTED_RATE,
            surprise_bps=surprise_bps,
        )
    )
    return cal


def _make_wrapper(
    surprise_bps: int = SURPRISE_HIKE_BPS,
    affected_symbols: list[str] | None = None,
) -> CBRStrategyWrapper:
    return CBRStrategyWrapper(
        calendar=_make_calendar(surprise_bps),
        affected_symbols=affected_symbols,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCBRStrategyWrapperProperties:
    def test_name(self) -> None:
        wrapper = CBRStrategyWrapper(calendar=CBRCalendar())
        assert wrapper.name == EXPECTED_STRATEGY_NAME

    def test_supported_segments(self) -> None:
        wrapper = CBRStrategyWrapper(calendar=CBRCalendar())
        segments = wrapper.supported_segments()
        assert len(segments) == EXPECTED_SEGMENTS_COUNT
        assert segments == EXPECTED_SEGMENTS

    def test_get_parameters(self) -> None:
        wrapper = CBRStrategyWrapper(calendar=CBRCalendar())
        params = wrapper.get_parameters(segment_id="ru_finance")
        assert "affected_symbols" in params
        assert "max_bars_since_event" in params
        assert params["max_bars_since_event"] == MAX_BARS_SINCE_EVENT
        assert isinstance(params["affected_symbols"], list)


class TestCBRStrategyWrapperGenerateSignal:
    def test_generate_signal_no_candles(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.generate_signal(SBER_SYMBOL, [], segment_id="ru_finance")
        assert result is None

    def test_generate_signal_no_events(self) -> None:
        """Empty calendar → no events → no signal even with candles."""
        wrapper = CBRStrategyWrapper(calendar=CBRCalendar())
        candles = _candles_on_event_day(SBER_SYMBOL)
        result = wrapper.generate_signal(SBER_SYMBOL, candles, segment_id="ru_finance")
        assert result is None

    def test_generate_signal_surprise_hike_day_zero(self) -> None:
        """Surprise hike on event day (bars_since=0) → SELL signal for SBER."""
        wrapper = _make_wrapper(surprise_bps=SURPRISE_HIKE_BPS)
        candles = _candles_on_event_day(SBER_SYMBOL)
        result = wrapper.generate_signal(SBER_SYMBOL, candles, segment_id="ru_finance")
        assert result is not None
        assert result.direction == SignalDirection.SELL
        assert result.symbol == SBER_SYMBOL

    def test_generate_signal_surprise_hike_contrarian(self) -> None:
        """Surprise hike with bars_since in [3, 5] → contrarian BUY signal."""
        wrapper = _make_wrapper(surprise_bps=SURPRISE_HIKE_BPS)
        # bars_past=BARS_SINCE_CONTRARIAN means BARS_SINCE_CONTRARIAN candles after EVENT_DATE
        candles = _candles_past_event(SBER_SYMBOL, bars_past=BARS_SINCE_CONTRARIAN)
        result = wrapper.generate_signal(SBER_SYMBOL, candles, segment_id="ru_finance")
        assert result is not None
        assert result.direction == SignalDirection.BUY
        assert result.symbol == SBER_SYMBOL

    def test_generate_signal_no_surprise(self) -> None:
        """Event with 0 surprise_bps → no signal."""
        wrapper = _make_wrapper(surprise_bps=NO_SURPRISE_BPS)
        candles = _candles_on_event_day(SBER_SYMBOL)
        result = wrapper.generate_signal(SBER_SYMBOL, candles, segment_id="ru_finance")
        assert result is None

    def test_generate_signal_symbol_not_affected(self) -> None:
        """Symbol not in affected_symbols list → no signal returned."""
        wrapper = _make_wrapper(surprise_bps=SURPRISE_HIKE_BPS)
        candles = _candles_on_event_day(UNAFFECTED_SYMBOL)
        result = wrapper.generate_signal(UNAFFECTED_SYMBOL, candles, segment_id="ru_finance")
        assert result is None
