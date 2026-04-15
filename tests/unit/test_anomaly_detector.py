from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from finalayze.analysis.anomaly_detector import AnomalyDetector, AnomalyResult
from finalayze.core.schemas import Candle

_SYMBOL = "SBER"
_MARKET_ID = "ru_blue_chips"


def _make_candle(close: float, volume: int, offset_minutes: int = 0) -> Candle:
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        timeframe="1h",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes),
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=volume,
    )


def _make_normal_candles(n: int, base_close: float = 100.0, base_vol: int = 1000) -> list[Candle]:
    """Create n candles with small random-ish variation around base values."""
    candles = []
    for i in range(n):
        # Small oscillation: +/- 0.5% price, +/- 10% volume
        factor = 1.0 + (i % 5 - 2) * 0.002  # -0.4% to +0.4%
        vol_factor = 1.0 + (i % 3 - 1) * 0.1  # -10% to +10%
        candles.append(
            _make_candle(
                close=base_close * factor,
                volume=int(base_vol * vol_factor),
                offset_minutes=i * 60,
            )
        )
    return candles


class TestInsufficientData:
    def test_returns_none_with_fewer_than_21_candles(self) -> None:
        detector = AnomalyDetector()
        candles = _make_normal_candles(20)
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert result is None

    def test_returns_none_with_empty_candles(self) -> None:
        detector = AnomalyDetector()
        result = detector.check([], _SYMBOL, _MARKET_ID)
        assert result is None


class TestPriceAnomaly:
    def test_detects_large_price_move(self) -> None:
        detector = AnomalyDetector()
        candles = _make_normal_candles(25, base_close=100.0, base_vol=1000)
        # Replace last candle with a massive price spike (+15%)
        candles[-1] = _make_candle(close=115.0, volume=1000, offset_minutes=24 * 60)
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert result is not None
        assert result.anomaly_type in ("price", "both")
        assert result.price_move_pct > 10.0
        assert result.sigma >= 3.0
        assert result.symbol == _SYMBOL
        assert result.market_id == _MARKET_ID


class TestVolumeAnomaly:
    def test_detects_volume_spike(self) -> None:
        detector = AnomalyDetector()
        candles = _make_normal_candles(25, base_close=100.0, base_vol=1000)
        # Replace last candle with normal price but huge volume (5x)
        candles[-1] = _make_candle(close=100.0, volume=5000, offset_minutes=24 * 60)
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert result is not None
        assert result.anomaly_type in ("volume", "both")
        assert result.volume_ratio >= 2.0


class TestBothAnomaly:
    def test_detects_both_price_and_volume(self) -> None:
        detector = AnomalyDetector()
        candles = _make_normal_candles(25, base_close=100.0, base_vol=1000)
        # Replace last candle with both large price move AND volume spike
        candles[-1] = _make_candle(close=120.0, volume=5000, offset_minutes=24 * 60)
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert result is not None
        assert result.anomaly_type == "both"


class TestNormalRange:
    def test_returns_none_for_normal_candles(self) -> None:
        detector = AnomalyDetector()
        candles = _make_normal_candles(25)
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert result is None


class TestEdgeCases:
    def test_zero_std_does_not_crash(self) -> None:
        """When all closes are identical, std=0 -- must not ZeroDivisionError."""
        detector = AnomalyDetector()
        candles = [_make_candle(close=100.0, volume=1000, offset_minutes=i * 60) for i in range(25)]
        result = detector.check(candles, _SYMBOL, _MARKET_ID)
        # No crash; result is None because sigma=0 < 3.0
        assert result is None

    def test_anomaly_result_is_frozen(self) -> None:
        result = AnomalyResult(
            symbol=_SYMBOL,
            market_id=_MARKET_ID,
            price_move_pct=10.0,
            sigma=4.5,
            volume_ratio=3.0,
            anomaly_type="price",
        )
        with pytest.raises(ValidationError):
            result.sigma = 1.0  # type: ignore[misc]
