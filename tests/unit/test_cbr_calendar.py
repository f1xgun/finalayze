"""Unit tests for CBR rate calendar and contrarian signal generation."""

from __future__ import annotations

from datetime import date

import pytest

from finalayze.core.schemas import SignalDirection
from finalayze.strategies.cbr_calendar import (
    CBRCalendar,
    CBRRateEvent,
    generate_cbr_signal,
)

# ── Constants (no magic numbers) ─────────────────────────────────────────────

RATE_16 = 16.0
RATE_15 = 15.0
RATE_17 = 17.0
RATE_14 = 14.0
EXPECTED_RATE_15 = 15.0
SURPRISE_HIKE_BPS = 100  # 16% actual vs 15% expected = +100bp
SURPRISE_CUT_BPS = -100  # 14% actual vs 15% expected = -100bp
NO_SURPRISE_BPS = 0
MIN_SURPRISE_BPS_DEFAULT = 50
MIN_SURPRISE_BPS_LARGE = 150
CONFIDENCE_UPPER_BOUND = 1.0
CONFIDENCE_LOWER_BOUND = 0.0

EVENT_DATE = date(2026, 2, 14)
EVENT_DATE_2 = date(2026, 3, 21)
MISSING_DATE = date(2026, 1, 1)

DEFAULT_AFFECTED_SYMBOLS = ["SBER", "VTBR", "SBERP"]
CUSTOM_AFFECTED_SYMBOLS = ["SBER", "VTBR"]

BARS_IMMEDIATE = 0
BARS_DAY_2 = 2
BARS_DAY_3 = 3
BARS_DAY_5 = 5
BARS_DAY_6 = 6
CONTRARIAN_DELAY_MIN = 3
CONTRARIAN_DELAY_MAX = 5


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def surprise_hike_event() -> CBRRateEvent:
    return CBRRateEvent(
        date=EVENT_DATE,
        rate_decision=RATE_16,
        expected_rate=EXPECTED_RATE_15,
        surprise_bps=SURPRISE_HIKE_BPS,
    )


@pytest.fixture
def surprise_cut_event() -> CBRRateEvent:
    return CBRRateEvent(
        date=EVENT_DATE,
        rate_decision=RATE_14,
        expected_rate=EXPECTED_RATE_15,
        surprise_bps=SURPRISE_CUT_BPS,
    )


@pytest.fixture
def no_surprise_event() -> CBRRateEvent:
    return CBRRateEvent(
        date=EVENT_DATE,
        rate_decision=EXPECTED_RATE_15,
        expected_rate=EXPECTED_RATE_15,
        surprise_bps=NO_SURPRISE_BPS,
    )


@pytest.fixture
def calendar(surprise_hike_event: CBRRateEvent) -> CBRCalendar:
    cal = CBRCalendar()
    cal.add_event(surprise_hike_event)
    return cal


# ── CBRRateEvent dataclass tests ─────────────────────────────────────────────


class TestCBRRateEvent:
    def test_fields(self, surprise_hike_event: CBRRateEvent) -> None:
        assert surprise_hike_event.date == EVENT_DATE
        assert surprise_hike_event.rate_decision == RATE_16
        assert surprise_hike_event.expected_rate == EXPECTED_RATE_15
        assert surprise_hike_event.surprise_bps == SURPRISE_HIKE_BPS


# ── CBRCalendar tests ────────────────────────────────────────────────────────


class TestCBRCalendar:
    def test_get_event_for_date_found(
        self, calendar: CBRCalendar, surprise_hike_event: CBRRateEvent
    ) -> None:
        result = calendar.get_event_for_date(EVENT_DATE)
        assert result == surprise_hike_event

    def test_get_event_for_date_not_found(self, calendar: CBRCalendar) -> None:
        result = calendar.get_event_for_date(MISSING_DATE)
        assert result is None

    def test_add_multiple_events(self) -> None:
        cal = CBRCalendar()
        event1 = CBRRateEvent(
            date=EVENT_DATE,
            rate_decision=RATE_16,
            expected_rate=EXPECTED_RATE_15,
            surprise_bps=SURPRISE_HIKE_BPS,
        )
        event2 = CBRRateEvent(
            date=EVENT_DATE_2,
            rate_decision=RATE_14,
            expected_rate=EXPECTED_RATE_15,
            surprise_bps=SURPRISE_CUT_BPS,
        )
        cal.add_event(event1)
        cal.add_event(event2)
        assert cal.get_event_for_date(EVENT_DATE) == event1
        assert cal.get_event_for_date(EVENT_DATE_2) == event2

    def test_is_surprise_hike(
        self, calendar: CBRCalendar, surprise_hike_event: CBRRateEvent
    ) -> None:
        assert calendar.is_surprise_hike(surprise_hike_event) is True

    def test_is_surprise_hike_below_threshold(
        self, calendar: CBRCalendar, surprise_hike_event: CBRRateEvent
    ) -> None:
        assert (
            calendar.is_surprise_hike(surprise_hike_event, min_surprise_bps=MIN_SURPRISE_BPS_LARGE)
            is False
        )

    def test_is_surprise_cut(self, calendar: CBRCalendar, surprise_cut_event: CBRRateEvent) -> None:
        assert calendar.is_surprise_cut(surprise_cut_event) is True

    def test_is_surprise_cut_below_threshold(
        self, calendar: CBRCalendar, surprise_cut_event: CBRRateEvent
    ) -> None:
        assert (
            calendar.is_surprise_cut(surprise_cut_event, min_surprise_bps=MIN_SURPRISE_BPS_LARGE)
            is False
        )

    def test_no_surprise_is_not_hike_or_cut(
        self, calendar: CBRCalendar, no_surprise_event: CBRRateEvent
    ) -> None:
        assert calendar.is_surprise_hike(no_surprise_event) is False
        assert calendar.is_surprise_cut(no_surprise_event) is False


# ── Signal generation tests ──────────────────────────────────────────────────


class TestGenerateCBRSignal:
    def test_surprise_hike_immediate_sell(self, surprise_hike_event: CBRRateEvent) -> None:
        """Surprise hike => immediate SELL for all affected bank stocks."""
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            sig = signals[sym]
            assert sig is not None
            assert sig.direction == SignalDirection.SELL
            assert CONFIDENCE_LOWER_BOUND < sig.confidence <= CONFIDENCE_UPPER_BOUND

    def test_surprise_hike_contrarian_buy_after_delay(
        self, surprise_hike_event: CBRRateEvent
    ) -> None:
        """After 3-5 bars, surprise hike flips to contrarian BUY."""
        for bars in (BARS_DAY_3, BARS_DAY_5):
            signals = generate_cbr_signal(
                event=surprise_hike_event,
                bars_since_event=bars,
                affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
            )
            for sym in DEFAULT_AFFECTED_SYMBOLS:
                sig = signals[sym]
                assert sig is not None
                assert sig.direction == SignalDirection.BUY

    def test_surprise_hike_no_signal_before_contrarian(
        self, surprise_hike_event: CBRRateEvent
    ) -> None:
        """Between immediate SELL and contrarian window (bars 1-2), no signal."""
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_DAY_2,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            assert signals[sym] is None

    def test_surprise_hike_no_signal_after_contrarian_window(
        self, surprise_hike_event: CBRRateEvent
    ) -> None:
        """After the contrarian window closes (>5 bars), no signal."""
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_DAY_6,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            assert signals[sym] is None

    def test_surprise_cut_immediate_buy(self, surprise_cut_event: CBRRateEvent) -> None:
        """Surprise cut => immediate BUY for all affected bank stocks."""
        signals = generate_cbr_signal(
            event=surprise_cut_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            sig = signals[sym]
            assert sig is not None
            assert sig.direction == SignalDirection.BUY
            assert CONFIDENCE_LOWER_BOUND < sig.confidence <= CONFIDENCE_UPPER_BOUND

    def test_surprise_cut_no_signal_later(self, surprise_cut_event: CBRRateEvent) -> None:
        """Surprise cut signal is only on day 0."""
        signals = generate_cbr_signal(
            event=surprise_cut_event,
            bars_since_event=BARS_DAY_2,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            assert signals[sym] is None

    def test_no_surprise_no_signal(self, no_surprise_event: CBRRateEvent) -> None:
        """No surprise => no signal for any bar count."""
        for bars in (BARS_IMMEDIATE, BARS_DAY_3, BARS_DAY_5):
            signals = generate_cbr_signal(
                event=no_surprise_event,
                bars_since_event=bars,
                affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
            )
            for sym in DEFAULT_AFFECTED_SYMBOLS:
                assert signals[sym] is None

    def test_custom_affected_symbols(self, surprise_cut_event: CBRRateEvent) -> None:
        """Only the specified symbols appear in the result."""
        signals = generate_cbr_signal(
            event=surprise_cut_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=CUSTOM_AFFECTED_SYMBOLS,
        )
        assert set(signals.keys()) == set(CUSTOM_AFFECTED_SYMBOLS)
        assert "SBERP" not in signals

    def test_signal_confidence_in_range(self, surprise_hike_event: CBRRateEvent) -> None:
        """All generated signals must have confidence in [0, 1]."""
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            sig = signals[sym]
            assert sig is not None
            assert CONFIDENCE_LOWER_BOUND <= sig.confidence <= CONFIDENCE_UPPER_BOUND

    def test_signal_has_correct_strategy_name(self, surprise_hike_event: CBRRateEvent) -> None:
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            sig = signals[sym]
            assert sig is not None
            assert sig.strategy_name == "cbr_calendar"

    def test_signal_features_contain_surprise_bps(self, surprise_hike_event: CBRRateEvent) -> None:
        signals = generate_cbr_signal(
            event=surprise_hike_event,
            bars_since_event=BARS_IMMEDIATE,
            affected_symbols=DEFAULT_AFFECTED_SYMBOLS,
        )
        for sym in DEFAULT_AFFECTED_SYMBOLS:
            sig = signals[sym]
            assert sig is not None
            assert "surprise_bps" in sig.strategy_payload
            assert sig.strategy_payload["surprise_bps"] == float(SURPRISE_HIKE_BPS)
