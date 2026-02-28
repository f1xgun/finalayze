"""Tests for TradingLoop._retrain_cycle and _retrain_segment."""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.core.schemas import Candle
from finalayze.ml.registry import MLModelRegistry

_BASE_DT = datetime.datetime(2025, 6, 1, tzinfo=datetime.UTC)


def _make_candles(n: int, symbol: str = "AAPL", base_price: float = 100.0) -> list[Candle]:
    """Create n synthetic candles."""
    candles: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.5))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
                + datetime.timedelta(days=i),
                open=price - Decimal("0.1"),
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1000,
            )
        )
    return candles


def _make_trading_loop(
    ml_registry: MLModelRegistry | None = None,
    ml_enabled: bool = True,
    ml_min_train_samples: int = 10,
) -> MagicMock:
    """Create a minimal TradingLoop-like object for testing _retrain_segment."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.ml_enabled = ml_enabled
    settings.ml_min_train_samples = ml_min_train_samples
    settings.ml_model_dir = "models/"
    settings.ml_retrain_interval_hours = 168
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.mode = "test"
    settings.max_position_pct = 0.20
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5

    # Create mock instrument
    instrument = MagicMock()
    instrument.symbol = "AAPL"
    instrument.segment_id = "us_tech"

    instrument_registry = MagicMock()
    instrument_registry.list_by_market.return_value = [instrument]

    fetcher = MagicMock()

    loop = MagicMock(spec=TradingLoop)
    loop._settings = settings
    loop._ml_registry = ml_registry
    loop._registry = instrument_registry
    loop._fetchers = {"us": fetcher}
    loop._alerter = MagicMock()
    loop._collect_active_segments = MagicMock(return_value=["us_tech"])

    # Bind real methods
    loop._retrain_cycle = TradingLoop._retrain_cycle.__get__(loop)
    loop._retrain_segment = TradingLoop._retrain_segment.__get__(loop)

    return loop


# Patch targets: these are imported locally inside _retrain_cycle
_SAVE_PATCH = "finalayze.ml.loader.save_ensemble"
_BW_PATCH = "finalayze.ml.training.build_windows"


class TestRetrainCycle:
    def test_retrain_registers_new_ensemble(self) -> None:
        """Successful retrain should register a new ensemble."""
        registry = MLModelRegistry()
        loop = _make_trading_loop(ml_registry=registry, ml_min_train_samples=5)

        candles = _make_candles(300)
        loop._fetchers["us"].fetch_candles.return_value = candles

        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = 0.8
        registry.create_ensemble = MagicMock(return_value=mock_ensemble)

        with patch(_SAVE_PATCH):
            loop._retrain_cycle()

        assert registry.get("us_tech") is mock_ensemble

    def test_retrain_skips_segment_with_insufficient_data(self) -> None:
        """When not enough training samples, skip the segment."""
        registry = MLModelRegistry()
        loop = _make_trading_loop(ml_registry=registry, ml_min_train_samples=1000)

        candles = _make_candles(70)
        loop._fetchers["us"].fetch_candles.return_value = candles

        with patch(_SAVE_PATCH):
            loop._retrain_cycle()

        assert registry.get("us_tech") is None

    def test_retrain_does_not_crash_on_fetch_error(self) -> None:
        """Fetch errors should be caught, not crash the cycle."""
        registry = MLModelRegistry()
        loop = _make_trading_loop(ml_registry=registry)

        loop._fetchers["us"].fetch_candles.side_effect = RuntimeError("network error")

        with patch(_SAVE_PATCH):
            loop._retrain_cycle()

        assert registry.get("us_tech") is None

    def test_retrain_validation_gate_rejects_bad_model(self) -> None:
        """Model with low validation accuracy should be rejected."""
        registry = MLModelRegistry()
        loop = _make_trading_loop(ml_registry=registry, ml_min_train_samples=5)

        candles = _make_candles(300)
        loop._fetchers["us"].fetch_candles.return_value = candles

        # Model always predicts 0.5 → rounds to 0, all labels are 1 → 0% acc
        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = 0.5
        registry.create_ensemble = MagicMock(return_value=mock_ensemble)

        with patch(_SAVE_PATCH), patch(_BW_PATCH) as mock_bw:
            features = [{"a": float(i)} for i in range(100)]
            labels = [1] * 100
            timestamps = [_BASE_DT + datetime.timedelta(days=i) for i in range(100)]
            mock_bw.return_value = (features, labels, timestamps)
            loop._retrain_cycle()

        assert registry.get("us_tech") is None

    def test_retrain_uses_temporal_gap_in_split(self) -> None:
        """Verify that a temporal gap is applied between train and val."""
        registry = MLModelRegistry()
        loop = _make_trading_loop(ml_registry=registry, ml_min_train_samples=5)

        candles = _make_candles(300)
        loop._fetchers["us"].fetch_candles.return_value = candles

        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = 0.9
        registry.create_ensemble = MagicMock(return_value=mock_ensemble)

        with patch(_SAVE_PATCH), patch(_BW_PATCH) as mock_bw:
            # 200 samples: train=140, gap=60, gap_end=200 → no val data
            features = [{"a": float(i)} for i in range(200)]
            labels = [1] * 200
            timestamps = [_BASE_DT + datetime.timedelta(days=i) for i in range(200)]
            mock_bw.return_value = (features, labels, timestamps)
            loop._retrain_cycle()

        # No validation data after gap → model not registered
        assert registry.get("us_tech") is None
