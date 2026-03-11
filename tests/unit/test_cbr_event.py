"""Unit tests for CBREventStrategy (Layer 4 — Tactical OFZ bond strategy).

Tests pass params directly — no YAML I/O.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.cbr_event import (
    _ENTRY_WINDOW_MAX_DAYS,
    _ENTRY_WINDOW_MIN_DAYS,
    _EXIT_WINDOW_MAX_DAYS,
    _GAP_THRESHOLD,
    CBREventStrategy,
)

# ── Named constants ──────────────────────────────────────────────────────────

PREFERRED_SYMBOL = "SU26244RMFS2"
NON_PREFERRED_SYMBOL = "GAZP"
SECOND_PREFERRED_SYMBOL = "SU26241RMFS8"

# Meeting date for testing: 2025-04-25 is a real CBR meeting in the calendar
MEETING_DATE = date(2025, 4, 25)

CANDLE_TIMEFRAME = "1d"
CANDLE_MARKET_ID = "moex"
CANDLE_OPEN = Decimal("97.50")
CANDLE_HIGH = Decimal("98.00")
CANDLE_LOW = Decimal("97.00")
CANDLE_CLOSE = Decimal("97.80")
CANDLE_VOLUME = 50_000

# Rate data
KEY_RATE = Decimal("0.2100")  # 21% as decimal fraction
RUONIA_DOVISH = Decimal("0.1760")  # gap = 0.176 - 0.21 = -0.034 → -3.4% → < -0.0030
RUONIA_HAWKISH = Decimal("0.2150")  # gap = +0.005 → > +0.0030
RUONIA_AMBIGUOUS = Decimal("0.2080")  # gap = -0.002 → within [-0.0030, +0.0030]

# Gap threshold is 0.30 percentage points. CBR rates in the strategy are stored
# as percentage points (e.g. 21.00), so key_rate and ruonia_7d_avg are passed in
# the same units as the CBR calendar uses. Let's recalculate with proper units.
# The task says gap < -0.30 (30bps = 0.30pp). So key_rate and ruonia in pp.
KEY_RATE_PP = Decimal("21.00")
RUONIA_DOVISH_PP = Decimal("20.50")  # gap = 20.50 - 21.00 = -0.50 < -0.15 ✓
RUONIA_HAWKISH_PP = Decimal("21.50")  # gap = +0.50 > +0.15 ✓
RUONIA_AMBIGUOUS_PP = Decimal("20.90")  # gap = -0.10, |gap| < 0.15 → ambiguous

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0
EXPECTED_BUY_CONFIDENCE = 0.7
EXPECTED_SELL_CONFIDENCE = 0.9

EXIT_DAYS_AFTER_MEETING = 2  # T+2

EXPECTED_STRATEGY_NAME = "cbr_event"
EXPECTED_INSTRUMENT_TYPE = "bond"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candle(symbol: str, d: date) -> Candle:
    """Create a minimal valid Candle on a given date."""
    ts = datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC)
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


def _candles_on_date(symbol: str, d: date, count: int = 1) -> list[Candle]:
    """Create candles ending on the given date (one per day going backwards)."""
    result: list[Candle] = []
    for i in range(count - 1, -1, -1):
        day = d - timedelta(days=i)
        result.append(_make_candle(symbol, day))
    return result


def _strategy(**kwargs: object) -> CBREventStrategy:
    """Create a CBREventStrategy with optional overrides."""
    return CBREventStrategy(**kwargs)  # type: ignore[arg-type]


# ── Test: No signal outside entry window ──────────────────────────────────────


class TestNoSignalOutsideEntryWindow:
    """Test that no signal is generated when >5 or <3 days before meeting."""

    def test_too_far_from_meeting(self) -> None:
        """10 days before meeting -> no signal."""
        strat = _strategy()
        # Meeting is 2025-04-25; 10 days before = 2025-04-15
        candle_date = MEETING_DATE - timedelta(days=10)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None

    def test_too_close_to_meeting(self) -> None:
        """1 day before meeting -> no signal (below entry window)."""
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=1)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None

    def test_on_meeting_day_no_entry(self) -> None:
        """On meeting day itself (0 days before next meeting) -> no entry signal."""
        strat = _strategy()
        # The meeting date itself. days_to_next_cbr returns days to the NEXT meeting
        # after as_of. On 2025-04-25, the next meeting is 2025-06-06 (42 days away).
        # So this is outside the 3-5 day window.
        candles = _candles_on_date(PREFERRED_SYMBOL, MEETING_DATE)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None


# ── Test: BUY in entry window with dovish gap ────────────────────────────────


class TestBuyInEntryWindow:
    """BUY signal when 3-5 days before meeting and RUONIA gap < -0.30."""

    def test_buy_3_days_before(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=_ENTRY_WINDOW_MIN_DAYS)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.direction == SignalDirection.BUY
        assert result.confidence == EXPECTED_BUY_CONFIDENCE

    def test_buy_5_days_before(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=_ENTRY_WINDOW_MAX_DAYS)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.direction == SignalDirection.BUY

    def test_buy_4_days_before(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.direction == SignalDirection.BUY


# ── Test: No signal with hawkish gap ──────────────────────────────────────────


class TestNoSignalHawkishGap:
    """No entry when RUONIA gap > +0.30 (market pricing in hikes)."""

    def test_hawkish_gap_no_signal(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_HAWKISH_PP,
        )
        assert result is None


# ── Test: No signal with ambiguous gap ────────────────────────────────────────


class TestNoSignalAmbiguousGap:
    """No entry when |gap| < 0.30 (ambiguous market pricing)."""

    def test_ambiguous_gap_no_signal(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_AMBIGUOUS_PP,
        )
        assert result is None


# ── Test: Mechanical exit T+2 after meeting ──────────────────────────────────


class TestMechanicalExit:
    """SELL signal on exit date (T+2 after meeting), regardless of outcome."""

    def test_exit_after_entry(self) -> None:
        """Enter pre-meeting, exit T+2 after meeting."""
        strat = _strategy()

        # Step 1: Entry (4 days before meeting)
        entry_date = MEETING_DATE - timedelta(days=4)
        entry_candles = _candles_on_date(PREFERRED_SYMBOL, entry_date)
        entry_signal = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=entry_candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert entry_signal is not None
        assert entry_signal.direction == SignalDirection.BUY

        # Step 2: Simulate being in position. Exit should fire T+2 after meeting.
        # Meeting is 2025-04-25 (Friday). T+2 = 2025-04-27 (Sunday) -> rolled to Monday 2025-04-28.
        exit_date = MEETING_DATE + timedelta(days=_EXIT_WINDOW_MAX_DAYS)
        # Skip weekends
        while exit_date.weekday() >= 5:  # noqa: PLR2004
            exit_date += timedelta(days=1)

        exit_candles = _candles_on_date(PREFERRED_SYMBOL, exit_date)
        exit_signal = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=exit_candles,
            open_positions={PREFERRED_SYMBOL: Decimal(100)},
            bar_idx=1,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert exit_signal is not None
        assert exit_signal.direction == SignalDirection.SELL
        assert exit_signal.confidence == EXPECTED_SELL_CONFIDENCE

    def test_no_exit_without_position(self) -> None:
        """No SELL signal if no open position on exit date."""
        strat = _strategy()

        # Step 1: Entry
        entry_date = MEETING_DATE - timedelta(days=4)
        entry_candles = _candles_on_date(PREFERRED_SYMBOL, entry_date)
        strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=entry_candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )

        # Step 2: Exit date but no open position
        exit_date = MEETING_DATE + timedelta(days=_EXIT_WINDOW_MAX_DAYS)
        while exit_date.weekday() >= 5:  # noqa: PLR2004
            exit_date += timedelta(days=1)

        exit_candles = _candles_on_date(PREFERRED_SYMBOL, exit_date)
        exit_signal = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=exit_candles,
            open_positions={},
            bar_idx=1,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        # Should still fire the SELL to clean up state, but with no position
        # the exit signal fires regardless (strategy tracks internally)
        # The implementation checks `symbol in open_positions` for exit
        assert exit_signal is None


# ── Test: No double entry ─────────────────────────────────────────────────────


class TestNoDoubleEntry:
    """Already in an event trade -> no second entry signal."""

    def test_no_second_entry(self) -> None:
        strat = _strategy()

        # First entry
        entry_date = MEETING_DATE - timedelta(days=4)
        entry_candles = _candles_on_date(PREFERRED_SYMBOL, entry_date)
        first = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=entry_candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert first is not None
        assert first.direction == SignalDirection.BUY

        # Same window, next day — should be blocked
        next_day = entry_date + timedelta(days=1)
        next_candles = _candles_on_date(PREFERRED_SYMBOL, next_day)
        second = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=next_candles,
            open_positions={PREFERRED_SYMBOL: Decimal(100)},
            bar_idx=1,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert second is None


# ── Test: Only preferred symbols ──────────────────────────────────────────────


class TestOnlyPreferredSymbols:
    """Non-preferred symbol -> no signal."""

    def test_non_preferred_no_signal(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(NON_PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=NON_PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None

    def test_custom_preferred_symbols(self) -> None:
        """Custom preferred symbols list is respected."""
        custom_sym = "SU26238RMFS4"
        strat = _strategy(preferred_symbols=[custom_sym])
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(custom_sym, candle_date)
        result = strat.generate_signal(
            symbol=custom_sym,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.direction == SignalDirection.BUY


# ── Test: No signal without rate data ─────────────────────────────────────────


class TestNoSignalWithoutRateData:
    """key_rate or ruonia_7d_avg is None -> no signal."""

    def test_no_key_rate(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=None,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None

    def test_no_ruonia(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=None,
        )
        assert result is None

    def test_both_none(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=None,
            ruonia_7d_avg=None,
        )
        assert result is None


# ── Test: Signal fields ───────────────────────────────────────────────────────


class TestSignalFields:
    """Verify instrument_type, strategy_name, market_id, segment_id, features."""

    def test_buy_signal_fields(self) -> None:
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.strategy_name == EXPECTED_STRATEGY_NAME
        assert result.instrument_type == EXPECTED_INSTRUMENT_TYPE
        assert result.market_id == "moex"
        assert result.segment_id == "ru_ofz_pd"
        assert result.symbol == PREFERRED_SYMBOL
        assert CONFIDENCE_MIN <= result.confidence <= CONFIDENCE_MAX
        assert "days_to_meeting" in result.features
        assert "ruonia_gap" in result.features

    def test_sell_signal_fields(self) -> None:
        """Exit signal also has correct fields."""
        strat = _strategy()

        # Entry first
        entry_date = MEETING_DATE - timedelta(days=4)
        entry_candles = _candles_on_date(PREFERRED_SYMBOL, entry_date)
        strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=entry_candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )

        # Exit
        exit_date = MEETING_DATE + timedelta(days=_EXIT_WINDOW_MAX_DAYS)
        while exit_date.weekday() >= 5:  # noqa: PLR2004
            exit_date += timedelta(days=1)

        exit_candles = _candles_on_date(PREFERRED_SYMBOL, exit_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=exit_candles,
            open_positions={PREFERRED_SYMBOL: Decimal(100)},
            bar_idx=1,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.strategy_name == EXPECTED_STRATEGY_NAME
        assert result.instrument_type == EXPECTED_INSTRUMENT_TYPE
        assert result.direction == SignalDirection.SELL
        assert "exit_type" in result.features


# ── Test: Look-ahead check ────────────────────────────────────────────────────


class TestLookAheadCheck:
    """Verify entry signal does not reference post-meeting data."""

    def test_entry_uses_only_pre_meeting_data(self) -> None:
        """Entry reasoning and features must NOT contain decision/outcome info."""
        strat = _strategy()
        candle_date = MEETING_DATE - timedelta(days=4)
        candles = _candles_on_date(PREFERRED_SYMBOL, candle_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        # Reasoning should NOT contain "decision", "cut", "hold", "hike"
        reasoning_lower = result.reasoning.lower()
        for forbidden_word in ("decision", "cut", "hold", "hike"):
            assert forbidden_word not in reasoning_lower, (
                f"Entry reasoning contains look-ahead word '{forbidden_word}': {result.reasoning}"
            )
        # Features should NOT contain decision-related keys
        for key in result.features:
            assert "decision" not in key.lower()
            assert "rate_after" not in key.lower()

    def test_exit_is_mechanical_not_decision_based(self) -> None:
        """Exit fires purely on date, not conditioned on the actual decision."""
        strat = _strategy()

        # Entry
        entry_date = MEETING_DATE - timedelta(days=4)
        entry_candles = _candles_on_date(PREFERRED_SYMBOL, entry_date)
        strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=entry_candles,
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )

        # Exit T+2
        exit_date = MEETING_DATE + timedelta(days=_EXIT_WINDOW_MAX_DAYS)
        while exit_date.weekday() >= 5:  # noqa: PLR2004
            exit_date += timedelta(days=1)

        exit_candles = _candles_on_date(PREFERRED_SYMBOL, exit_date)
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=exit_candles,
            open_positions={PREFERRED_SYMBOL: Decimal(100)},
            bar_idx=1,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is not None
        assert result.direction == SignalDirection.SELL
        # Exit reasoning should mention "mechanical" or "T+1/T+2", not decision
        reasoning_lower = result.reasoning.lower()
        assert "mechanical" in reasoning_lower or "t+" in reasoning_lower


# ── Test: Empty candles ───────────────────────────────────────────────────────


class TestEmptyCandles:
    """Empty candle list -> None."""

    def test_empty_candles(self) -> None:
        strat = _strategy()
        result = strat.generate_signal(
            symbol=PREFERRED_SYMBOL,
            candles=[],
            open_positions={},
            bar_idx=0,
            key_rate=KEY_RATE_PP,
            ruonia_7d_avg=RUONIA_DOVISH_PP,
        )
        assert result is None
