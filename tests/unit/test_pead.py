"""Tests for PEADStrategy (Layer 4).

All parameters are passed directly -- no YAML I/O in unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.pead import EarningsSurprise, PEADStrategy

# ---------------------------------------------------------------------------
# Constants (ruff PLR2004)
# ---------------------------------------------------------------------------
_SYMBOL = "AAPL"
_MARKET_ID_US = "US"
_MARKET_ID_MOEX = "MOEX"
_SEGMENT_US = "us_tech"
_SEGMENT_RU = "ru_blue_chips"
_TIMEFRAME = "1d"

_BASE_PRICE = Decimal("150.00")

_POSITIVE_THRESHOLD = 1.0
_NEGATIVE_THRESHOLD = -1.0
_DRIFT_WINDOW_BARS = 60
_MIN_CONFIDENCE = 0.35

_HIGH_SUE = 2.5
_LOW_SUE = -2.0
_BORDERLINE_SUE_POS = 0.5  # below positive threshold
_BORDERLINE_SUE_NEG = -0.5  # above negative threshold
_VERY_HIGH_SUE = 5.0

_ACTUAL_EPS = 3.50
_EXPECTED_EPS = 3.00

_EXPECTED_ZERO = 0
_TWO_CANDLES = 2


def _make_candle(
    close: Decimal,
    ts: datetime,
    *,
    symbol: str = _SYMBOL,
    market_id: str = _MARKET_ID_US,
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
    count: int,
    start: datetime,
    *,
    close: Decimal = _BASE_PRICE,
    symbol: str = _SYMBOL,
    market_id: str = _MARKET_ID_US,
) -> list[Candle]:
    """Build a candle series of `count` bars, one bar per day."""
    return [
        _make_candle(close, start + timedelta(days=i), symbol=symbol, market_id=market_id)
        for i in range(count)
    ]


def _announcement_date() -> datetime:
    """A fixed earnings announcement date for tests."""
    return datetime(2026, 4, 25, tzinfo=UTC)


def _make_strategy(
    *,
    positive_threshold: float = _POSITIVE_THRESHOLD,
    negative_threshold: float = _NEGATIVE_THRESHOLD,
    drift_window_bars: int = _DRIFT_WINDOW_BARS,
    min_confidence: float = _MIN_CONFIDENCE,
) -> PEADStrategy:
    return PEADStrategy(
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        drift_window_bars=drift_window_bars,
        min_confidence=min_confidence,
    )


def _make_surprise(
    *,
    sue_score: float = _HIGH_SUE,
    announcement_date: datetime | None = None,
    symbol: str = _SYMBOL,
    actual_eps: float = _ACTUAL_EPS,
    expected_eps: float = _EXPECTED_EPS,
) -> EarningsSurprise:
    return EarningsSurprise(
        symbol=symbol,
        announcement_date=announcement_date or _announcement_date(),
        sue_score=sue_score,
        actual_eps=actual_eps,
        expected_eps=expected_eps,
    )


# ===================================================================
# Test: BUY signal on positive surprise
# ===================================================================
class TestBuyOnPositiveSurprise:
    """BUY signal when sue_score > positive_threshold after announcement."""

    def test_buy_signal_generated(self) -> None:
        strategy = _make_strategy()
        ann_date = _announcement_date()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        # Candles: a few before announcement, then the announcement day + a few after
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)  # day 5 = ann_date, day 6 = 1 day after

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "pead"
        assert signal.confidence >= _MIN_CONFIDENCE
        assert signal.confidence <= 1.0

    def test_buy_features_contain_sue_score(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert "sue_score" in signal.features
        assert signal.features["sue_score"] == _HIGH_SUE

    def test_buy_signal_us_market(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start, market_id=_MARKET_ID_US)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.market_id == _MARKET_ID_US

    def test_buy_signal_moex_market(self) -> None:
        strategy = _make_strategy()
        moex_symbol = "SBER"
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE, symbol=moex_symbol))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start, symbol=moex_symbol, market_id=_MARKET_ID_MOEX)

        signal = strategy.generate_signal(
            symbol=moex_symbol,
            candles=candles,
            segment_id=_SEGMENT_RU,
        )

        assert signal is not None
        assert signal.market_id == _MARKET_ID_MOEX
        assert signal.direction == SignalDirection.BUY


# ===================================================================
# Test: SELL signal on negative surprise
# ===================================================================
class TestSellOnNegativeSurprise:
    """SELL signal when sue_score < negative_threshold after announcement."""

    def test_sell_signal_generated(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_LOW_SUE))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence >= _MIN_CONFIDENCE
        assert signal.confidence <= 1.0

    def test_sell_features_contain_sue_score(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_LOW_SUE))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.features["sue_score"] == _LOW_SUE


# ===================================================================
# Test: No signal when |sue_score| below thresholds
# ===================================================================
class TestNoSignalBelowThreshold:
    """No signal when sue_score is between negative and positive thresholds."""

    def test_no_signal_positive_below_threshold(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_BORDERLINE_SUE_POS))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is None

    def test_no_signal_negative_above_threshold(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_BORDERLINE_SUE_NEG))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is None

    def test_no_signal_before_announcement(self) -> None:
        """No signal when current candle is before the announcement date."""
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        ann_date = _announcement_date()
        # All candles are before the announcement date
        start = ann_date - timedelta(days=10)
        candles = _make_candles(5, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is None

    def test_no_signal_no_surprise_registered(self) -> None:
        """No signal when no earnings surprise has been registered for the symbol."""
        strategy = _make_strategy()

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is None


# ===================================================================
# Test: Signal expires after drift_window_bars
# ===================================================================
class TestDriftWindowExpiry:
    """Signal should not be generated after drift_window_bars have passed."""

    def test_signal_active_within_window(self) -> None:
        drift_window = 10
        strategy = _make_strategy(drift_window_bars=drift_window)
        ann_date = _announcement_date()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        # Current candle is 5 bars after announcement (within window)
        bars_after = 5
        start = ann_date - timedelta(days=3)
        candles = _make_candles(3 + bars_after + 1, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_signal_expired_outside_window(self) -> None:
        drift_window = 10
        strategy = _make_strategy(drift_window_bars=drift_window)
        ann_date = _announcement_date()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        # Current candle is drift_window + 5 bars after announcement (outside window)
        bars_after = drift_window + 5
        start = ann_date - timedelta(days=3)
        candles = _make_candles(3 + bars_after + 1, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is None

    def test_signal_at_exact_boundary(self) -> None:
        """Signal should still be active at exactly drift_window_bars."""
        drift_window = 10
        strategy = _make_strategy(drift_window_bars=drift_window)
        ann_date = _announcement_date()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))

        # Current candle is exactly at the drift window boundary
        start = ann_date - timedelta(days=3)
        candles = _make_candles(3 + drift_window + 1, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None


# ===================================================================
# Test: Confidence scaling
# ===================================================================
class TestConfidenceScaling:
    """Confidence should scale with |sue_score| magnitude."""

    def test_higher_sue_gives_higher_confidence(self) -> None:
        ann_date = _announcement_date()

        # Moderate positive surprise
        strategy_mod = _make_strategy()
        strategy_mod.add_earnings_surprise(_make_surprise(sue_score=_POSITIVE_THRESHOLD + 0.5))

        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal_mod = strategy_mod.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        # High positive surprise
        strategy_high = _make_strategy()
        strategy_high.add_earnings_surprise(_make_surprise(sue_score=_VERY_HIGH_SUE))

        signal_high = strategy_high.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal_mod is not None
        assert signal_high is not None
        assert signal_high.confidence > signal_mod.confidence

    def test_confidence_capped_at_one(self) -> None:
        strategy = _make_strategy()
        extreme_sue = 100.0
        strategy.add_earnings_surprise(_make_surprise(sue_score=extreme_sue))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.confidence <= 1.0

    def test_confidence_above_min(self) -> None:
        """Confidence for a signal just above threshold should be >= min_confidence."""
        strategy = _make_strategy()
        just_above = _POSITIVE_THRESHOLD + 0.1
        strategy.add_earnings_surprise(_make_surprise(sue_score=just_above))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal is not None
        assert signal.confidence >= _MIN_CONFIDENCE

    def test_negative_confidence_scales_with_magnitude(self) -> None:
        """Negative SUE confidence also scales with magnitude."""
        strategy_low = _make_strategy()
        strategy_low.add_earnings_surprise(_make_surprise(sue_score=_NEGATIVE_THRESHOLD - 0.5))

        strategy_very_low = _make_strategy()
        strategy_very_low.add_earnings_surprise(_make_surprise(sue_score=_NEGATIVE_THRESHOLD - 3.0))

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal_low = strategy_low.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )
        signal_very_low = strategy_very_low.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )

        assert signal_low is not None
        assert signal_very_low is not None
        assert signal_very_low.confidence > signal_low.confidence


# ===================================================================
# Test: BaseStrategy interface
# ===================================================================
class TestBaseInterface:
    """Verify PEADStrategy satisfies BaseStrategy ABC."""

    def test_name(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "pead"

    def test_supported_segments(self) -> None:
        strategy = _make_strategy()
        segments = strategy.supported_segments()
        assert isinstance(segments, list)
        assert len(segments) > _EXPECTED_ZERO
        # Both US and RU segments should be present
        assert any(s.startswith("us_") for s in segments)
        assert any(s.startswith("ru_") for s in segments)

    def test_get_parameters(self) -> None:
        strategy = _make_strategy()
        params = strategy.get_parameters(_SEGMENT_US)
        assert "positive_threshold" in params
        assert "negative_threshold" in params
        assert "drift_window_bars" in params
        assert "min_confidence" in params

    def test_reset_clears_surprises(self) -> None:
        strategy = _make_strategy()
        strategy.add_earnings_surprise(_make_surprise(sue_score=_HIGH_SUE))
        strategy.reset()

        ann_date = _announcement_date()
        start = ann_date - timedelta(days=5)
        candles = _make_candles(7, start)

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_US,
        )
        assert signal is None
