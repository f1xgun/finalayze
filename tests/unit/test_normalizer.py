"""Unit tests for DataNormalizer (TDD — RED phase)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import Candle
from finalayze.data.normalizer import DataNormalizer

# ---------------------------------------------------------------------------
# Constants (no magic numbers per ruff PLR2004)
# ---------------------------------------------------------------------------
MARKET_ID = "us_equity"
SOURCE = "alpaca"
SYMBOL = "AAPL"
TIMEFRAME = "1m"
TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
BATCH_SIZE_3 = 3
BATCH_SIZE_2 = 2

OPEN = Decimal("150.00")
HIGH = Decimal("155.00")
LOW = Decimal("149.00")
CLOSE = Decimal("153.00")
VOLUME = 1000

NEGATIVE_PRICE = Decimal("-1.00")
PRICE_ABOVE_HIGH = Decimal("160.00")
PRICE_BELOW_LOW = Decimal("140.00")


def _make_candle(
    open_: Decimal = OPEN,
    high: Decimal = HIGH,
    low: Decimal = LOW,
    close: Decimal = CLOSE,
    volume: int = VOLUME,
    market_id: str = "",
) -> Candle:
    """Factory for test candles."""
    return Candle(
        symbol=SYMBOL,
        market_id=market_id,
        timeframe=TIMEFRAME,
        timestamp=TIMESTAMP,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestDataNormalizerSingle:
    """Tests for the normalize() method on individual candles."""

    def test_normalize_candle_sets_market_id(self) -> None:
        """normalize() must set the market_id on the returned candle."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle()
        result = normalizer.normalize(candle)
        assert result.market_id == MARKET_ID

    def test_normalize_candle_sets_source(self) -> None:
        """normalize() must set the source tag on the returned candle."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle()
        result = normalizer.normalize(candle)
        assert result.source == SOURCE

    def test_normalize_rejects_negative_price(self) -> None:
        """Candle with open < 0 must raise DataFetchError."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle(open_=NEGATIVE_PRICE)
        with pytest.raises(DataFetchError):
            normalizer.normalize(candle)

    def test_normalize_rejects_low_above_high(self) -> None:
        """Candle where low > high must raise DataFetchError."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        # low=155, high=150 — invalid
        candle = _make_candle(low=HIGH, high=LOW)
        with pytest.raises(DataFetchError):
            normalizer.normalize(candle)

    def test_normalize_rejects_close_outside_range(self) -> None:
        """Candle where close > high must raise DataFetchError."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle(close=PRICE_ABOVE_HIGH)
        with pytest.raises(DataFetchError):
            normalizer.normalize(candle)

    def test_normalize_rejects_close_below_low(self) -> None:
        """Candle where close < low must raise DataFetchError."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle(close=PRICE_BELOW_LOW)
        with pytest.raises(DataFetchError):
            normalizer.normalize(candle)

    def test_normalize_passes_valid_candle(self) -> None:
        """Valid candle passes through with market_id and source updated."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candle = _make_candle()
        result = normalizer.normalize(candle)
        assert result.symbol == candle.symbol
        assert result.open == candle.open
        assert result.high == candle.high
        assert result.low == candle.low
        assert result.close == candle.close
        assert result.volume == candle.volume
        assert result.timestamp == candle.timestamp
        assert result.timeframe == candle.timeframe


class TestDataNormalizerBatch:
    """Tests for the normalize_batch() method."""

    def test_normalize_batch_filters_invalid(self) -> None:
        """normalize_batch() skips invalid candles and returns only valid ones."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        valid_candle = _make_candle()
        invalid_candle = _make_candle(open_=NEGATIVE_PRICE)
        results = normalizer.normalize_batch([valid_candle, invalid_candle])
        # Only the valid candle should be in the result
        assert len(results) == 1
        assert results[0].symbol == valid_candle.symbol

    def test_normalize_batch_all_valid(self) -> None:
        """normalize_batch() returns all candles when all are valid."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candles = [_make_candle() for _ in range(BATCH_SIZE_3)]
        results = normalizer.normalize_batch(candles)
        assert len(results) == BATCH_SIZE_3

    def test_normalize_batch_all_invalid(self) -> None:
        """normalize_batch() returns empty list when all candles are invalid."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candles = [_make_candle(open_=NEGATIVE_PRICE) for _ in range(BATCH_SIZE_2)]
        results = normalizer.normalize_batch(candles)
        assert results == []

    def test_normalize_batch_sets_source_on_all(self) -> None:
        """normalize_batch() sets source on every returned candle."""
        normalizer = DataNormalizer(market_id=MARKET_ID, source=SOURCE)
        candles = [_make_candle() for _ in range(BATCH_SIZE_3)]
        results = normalizer.normalize_batch(candles)
        assert all(c.source == SOURCE for c in results)
