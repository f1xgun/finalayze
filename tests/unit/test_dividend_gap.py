"""Tests for DividendGapStrategy (Layer 4).

All parameters are passed directly -- no YAML I/O in unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.risk.regime import MarketRegime, RegimeState
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy

# ---------------------------------------------------------------------------
# Constants (ruff PLR2004)
# ---------------------------------------------------------------------------
_SYMBOL = "SBER"
_MARKET_ID = "MOEX"
_SEGMENT_ID = "ru_blue_chips"
_TIMEFRAME = "1d"

_BASE_PRICE = Decimal("300.00")
_DIVIDEND_AMOUNT = 18.0  # 6% yield on 300
_LARGE_DIVIDEND = 30.0  # 10% yield on 300
_SMALL_DIVIDEND = 3.0  # 1% yield on 300 -> gap < 3% default threshold

_MIN_GAP_PCT = 3.0
_MAX_HOLD_BARS = 40
_MIN_CONFIDENCE = 0.40

_PRE_EXDIV_BARS = 5  # bars before ex-div date
_EXPECTED_ZERO = 0


def _make_candle(
    close: Decimal,
    ts: datetime,
    *,
    symbol: str = _SYMBOL,
    market_id: str = _MARKET_ID,
) -> Candle:
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe=_TIMEFRAME,
        timestamp=ts,
        open=close,
        high=close + Decimal(1),
        low=close - Decimal(1),
        close=close,
        volume=1000,
    )


def _make_candles(
    prices: list[Decimal],
    start: datetime,
) -> list[Candle]:
    """Build a candle series from a price list, one bar per day."""
    return [_make_candle(p, start + timedelta(days=i)) for i, p in enumerate(prices)]


def _exdiv_date() -> datetime:
    """A fixed ex-dividend date for tests."""
    return datetime(2026, 7, 10, tzinfo=UTC)


def _make_strategy(
    *,
    min_gap_pct: float = _MIN_GAP_PCT,
    max_hold_bars: int = _MAX_HOLD_BARS,
    min_confidence: float = _MIN_CONFIDENCE,
) -> DividendGapStrategy:
    return DividendGapStrategy(
        min_gap_pct=min_gap_pct,
        max_hold_bars=max_hold_bars,
        min_confidence=min_confidence,
    )


# ===================================================================
# Test: signal generation on ex-div date
# ===================================================================
class TestSignalOnExDivDate:
    """BUY signal should fire on the ex-dividend date when gap > threshold."""

    def test_buy_signal_generated(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()

        # Register dividend
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        # Build candle series: 5 bars at base price, then ex-div bar with gap
        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close]
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
        )

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "dividend_gap"
        assert signal.confidence >= _MIN_CONFIDENCE
        assert signal.confidence <= 1.0

    def test_features_contain_gap_pct(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close]
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
        )

        assert signal is not None
        assert "gap_pct" in signal.strategy_payload
        expected_gap = _DIVIDEND_AMOUNT / float(_BASE_PRICE) * 100.0
        assert abs(signal.strategy_payload["gap_pct"] - expected_gap) < 0.1


# ===================================================================
# Test: no signal when gap too small
# ===================================================================
class TestNoSignalSmallGap:
    """No signal when dividend gap is below min_gap_pct threshold."""

    def test_no_signal_below_threshold(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_SMALL_DIVIDEND),
        )

        gap_close = _BASE_PRICE - Decimal(str(_SMALL_DIVIDEND))
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close]
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
        )

        assert signal is None

    def test_no_signal_when_no_dividend_registered(self) -> None:
        strategy = _make_strategy()
        exdiv = _exdiv_date()

        prices = [_BASE_PRICE] * (_PRE_EXDIV_BARS + 1)
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
        )

        assert signal is None


# ===================================================================
# Test: exit signal on gap closure
# ===================================================================
class TestExitOnGapClosure:
    """SELL signal when price recovers to pre-ex-div close (gap closed)."""

    def test_sell_signal_on_gap_closure(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        # Pre ex-div bars -> ex-div gap -> gradual recovery -> full closure
        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        recovery_close = _BASE_PRICE  # full recovery

        prices = (
            [_BASE_PRICE] * _PRE_EXDIV_BARS
            + [gap_close]  # ex-div day (bar index 5)
            + [recovery_close]  # recovery day (bar index 6)
        )
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        # First call on ex-div day should generate BUY and track the gap
        candles_exdiv = candles[: _PRE_EXDIV_BARS + 1]
        buy_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles_exdiv,
            segment_id=_SEGMENT_ID,
        )
        assert buy_signal is not None
        assert buy_signal.direction == SignalDirection.BUY

        # Second call with recovery candle + has_open_position -> SELL
        sell_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            has_open_position=True,
        )
        assert sell_signal is not None
        assert sell_signal.direction == SignalDirection.SELL


# ===================================================================
# Test: max hold exit
# ===================================================================
class TestMaxHoldExit:
    """SELL signal when max_hold_bars reached without gap closure."""

    def test_sell_signal_at_max_hold(self) -> None:
        # _DIVIDEND_AMOUNT=18 on _BASE_PRICE=300 is 6% yield -> 40 bars tier
        yield_hold = 40
        exdiv = _exdiv_date()
        strategy = _make_strategy(max_hold_bars=60)  # safety ceiling
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))

        # Pre-exdiv + ex-div day + yield_hold bars at gap level (no recovery)
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close] + [gap_close] * yield_hold
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        # Trigger BUY on ex-div day
        candles_exdiv = candles[: _PRE_EXDIV_BARS + 1]
        buy_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles_exdiv,
            segment_id=_SEGMENT_ID,
        )
        assert buy_signal is not None
        assert buy_signal.direction == SignalDirection.BUY

        # After yield_hold bars with open position -> SELL
        sell_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            has_open_position=True,
        )
        assert sell_signal is not None
        assert sell_signal.direction == SignalDirection.SELL

    def test_no_sell_before_max_hold(self) -> None:
        # _DIVIDEND_AMOUNT=18 on _BASE_PRICE=300 is 6% -> yield tier 40 bars
        exdiv = _exdiv_date()
        strategy = _make_strategy(max_hold_bars=60)
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        bars_after = 3  # well before yield tier of 40

        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close] + [gap_close] * bars_after
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        # Trigger BUY
        candles_exdiv = candles[: _PRE_EXDIV_BARS + 1]
        strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles_exdiv,
            segment_id=_SEGMENT_ID,
        )

        # Not enough bars for yield-tier exit, price still gapped -> HOLD (None)
        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            has_open_position=True,
        )
        assert signal is None


# ===================================================================
# Test: regime filter blocking
# ===================================================================
class TestRegimeFilter:
    """Regime filter blocks signals in CRISIS regime."""

    def test_crisis_regime_blocks_buy(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close]
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        crisis = RegimeState(
            regime=MarketRegime.CRISIS,
            allow_new_longs=False,
            position_scale=Decimal("0.10"),
        )

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            regime_state=crisis,
        )

        assert signal is None

    def test_normal_regime_allows_buy(self) -> None:
        exdiv = _exdiv_date()
        strategy = _make_strategy()
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_DIVIDEND_AMOUNT),
        )

        gap_close = _BASE_PRICE - Decimal(str(_DIVIDEND_AMOUNT))
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close]
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        normal = RegimeState(
            regime=MarketRegime.NORMAL,
            allow_new_longs=True,
            position_scale=Decimal("1.00"),
        )

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            regime_state=normal,
        )

        assert signal is not None
        assert signal.direction == SignalDirection.BUY


# ===================================================================
# Test: BaseStrategy interface
# ===================================================================
class TestBaseInterface:
    """Verify DividendGapStrategy satisfies BaseStrategy ABC."""

    def test_name(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "dividend_gap"

    def test_supported_segments(self) -> None:
        strategy = _make_strategy()
        segments = strategy.supported_segments()
        assert isinstance(segments, list)
        assert len(segments) > _EXPECTED_ZERO

    def test_get_parameters(self) -> None:
        strategy = _make_strategy()
        params = strategy.get_parameters(_SEGMENT_ID)
        assert "min_gap_pct" in params
        assert "max_hold_bars" in params
        assert "min_confidence" in params


# ===================================================================
# Test: Yield-based hold bars
# ===================================================================
_HIGH_YIELD_DIVIDEND = 25.0  # 8.3% yield on 300 -> 60 bars
_MID_YIELD_DIVIDEND = 18.0  # 6.0% yield on 300 -> 40 bars
_LOW_YIELD_DIVIDEND = 12.0  # 4.0% yield on 300 -> 25 bars

_YIELD_HIGH_THRESHOLD = 8.0
_YIELD_MID_THRESHOLD = 5.0

_HOLD_BARS_HIGH = 60
_HOLD_BARS_MID = 40
_HOLD_BARS_LOW = 25


class TestYieldHoldBars:
    """Yield-based hold bars: higher gap % -> longer hold period."""

    def test_yield_hold_bars_high(self) -> None:
        """gap_pct >= 8.0 returns 60 hold bars."""
        assert DividendGapStrategy._yield_hold_bars(_YIELD_HIGH_THRESHOLD) == _HOLD_BARS_HIGH
        assert DividendGapStrategy._yield_hold_bars(10.0) == _HOLD_BARS_HIGH

    def test_yield_hold_bars_mid(self) -> None:
        """gap_pct 5.0-7.99 returns 40 hold bars."""
        assert DividendGapStrategy._yield_hold_bars(_YIELD_MID_THRESHOLD) == _HOLD_BARS_MID
        assert DividendGapStrategy._yield_hold_bars(7.5) == _HOLD_BARS_MID

    def test_yield_hold_bars_low(self) -> None:
        """gap_pct < 5.0 returns 25 hold bars."""
        assert DividendGapStrategy._yield_hold_bars(3.5) == _HOLD_BARS_LOW
        assert DividendGapStrategy._yield_hold_bars(4.9) == _HOLD_BARS_LOW

    def test_low_yield_sell_at_25_bars(self) -> None:
        """Position with gap_pct=4.0 should SELL at 25 bars."""
        exdiv = _exdiv_date()
        strategy = _make_strategy(max_hold_bars=60)  # safety ceiling
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_LOW_YIELD_DIVIDEND),
        )

        gap_close = _BASE_PRICE - Decimal(str(_LOW_YIELD_DIVIDEND))

        # Pre-exdiv + ex-div day + 25 bars at gap level (no recovery)
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close] + [gap_close] * _HOLD_BARS_LOW
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        # Trigger BUY on ex-div day
        candles_exdiv = candles[: _PRE_EXDIV_BARS + 1]
        buy_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles_exdiv,
            segment_id=_SEGMENT_ID,
        )
        assert buy_signal is not None
        assert buy_signal.direction == SignalDirection.BUY

        # After 25 bars -> SELL (low yield tier reached)
        sell_signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            has_open_position=True,
        )
        assert sell_signal is not None
        assert sell_signal.direction == SignalDirection.SELL

    def test_high_yield_no_sell_at_25_bars(self) -> None:
        """Position with gap_pct=8.3% should NOT sell at 25 bars (needs 60)."""
        exdiv = _exdiv_date()
        strategy = _make_strategy(max_hold_bars=60)
        strategy.add_dividend(
            _SYMBOL,
            DividendEntry(ex_date=exdiv, amount=_HIGH_YIELD_DIVIDEND),
        )

        gap_close = _BASE_PRICE - Decimal(str(_HIGH_YIELD_DIVIDEND))

        # Pre-exdiv + ex-div day + 25 bars at gap level (no recovery)
        prices = [_BASE_PRICE] * _PRE_EXDIV_BARS + [gap_close] + [gap_close] * _HOLD_BARS_LOW
        start = exdiv - timedelta(days=_PRE_EXDIV_BARS)
        candles = _make_candles(prices, start)

        # Trigger BUY
        candles_exdiv = candles[: _PRE_EXDIV_BARS + 1]
        strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles_exdiv,
            segment_id=_SEGMENT_ID,
        )

        # After 25 bars -> should still be holding (None), not SELL
        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_ID,
            has_open_position=True,
        )
        assert signal is None
