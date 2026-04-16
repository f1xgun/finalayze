"""Tests for data validation wiring in trading loop and IMOEX volume fix (DATA-01/02/03)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.moex_iss import MoexISSFetcher

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_candle(
    symbol: str = "TEST",
    ts: datetime | None = None,
    close: Decimal = Decimal("100.0"),
) -> Candle:
    if ts is None:
        ts = datetime.now(UTC) - timedelta(hours=1)
    return Candle(
        symbol=symbol,
        market_id="moex",
        timeframe="1d",
        timestamp=ts,
        open=Decimal("99.0"),
        high=Decimal("101.0"),
        low=Decimal("98.0"),
        close=close,
        volume=1000,
        source="test",
    )


# ── DATA-03: IMOEX volume uses row[5] (share volume), not row[4] (turnover) ──


class TestImoexVolumeColumn:
    """Verify _parse_candle_row uses row[5] for volume field."""

    def test_parse_candle_row_uses_share_volume_not_turnover(self) -> None:
        """row[4] is turnover, row[5] is share volume; Candle.volume must be row[5]."""
        fetcher = MoexISSFetcher()
        # ISS column order: open, close, high, low, value, volume, begin, end
        row = [
            100.0,  # [0] open
            105.0,  # [1] close
            110.0,  # [2] high
            95.0,  # [3] low
            9999999,  # [4] value (turnover in RUB) -- should NOT be used
            42000,  # [5] volume (share volume) -- SHOULD be used
            "2024-01-15 10:00:00",  # [6] begin
            "2024-01-15 23:59:59",  # [7] end
        ]
        candle = fetcher._parse_candle_row(row, "IMOEX", "1d")
        assert candle is not None
        assert candle.volume == 42000, (
            f"Expected share volume 42000 from row[5], got {candle.volume} "
            "(likely using turnover from row[4])"
        )

    def test_parse_candle_row_zero_share_volume(self) -> None:
        """When row[5] is 0, volume should be 0."""
        fetcher = MoexISSFetcher()
        row = [100.0, 105.0, 110.0, 95.0, 9999999, 0, "2024-01-15 10:00:00", "2024-01-15 23:59:59"]
        candle = fetcher._parse_candle_row(row, "IMOEX", "1d")
        assert candle is not None
        assert candle.volume == 0

    def test_parse_candle_row_none_share_volume(self) -> None:
        """When row[5] is None (missing), volume should fallback to 0."""
        fetcher = MoexISSFetcher()
        row = [
            100.0,
            105.0,
            110.0,
            95.0,
            9999999,
            None,
            "2024-01-15 10:00:00",
            "2024-01-15 23:59:59",
        ]
        candle = fetcher._parse_candle_row(row, "IMOEX", "1d")
        assert candle is not None
        assert candle.volume == 0


# ── DATA-01: DataNormalizer.normalize_batch called in _process_instrument ──


class TestDataNormalizerWiring:
    """Verify DataNormalizer is wired into _process_instrument."""

    def _make_loop_and_deps(self) -> tuple:
        """Create a minimal TradingLoop with mocked deps for _process_instrument."""
        import threading

        from finalayze.orchestration.trading_loop import TradingLoop

        loop = TradingLoop.__new__(TradingLoop)
        # Minimal state needed by _process_instrument
        loop._last_prices = {}
        loop._cycle_errors_caught = 0
        loop._cycle_signals_generated = 0
        loop._cycle_orders_submitted = 0
        loop._cycle_orders_filled = 0
        loop._cycle_exited_symbols = set()
        loop._health_monitor = None
        loop._metrics = None
        loop._broker_router = MagicMock()
        loop._strategy = MagicMock()
        loop._strategy.generate_signal.return_value = None
        loop._validation_logger = MagicMock(spec_set=["log_cycle"])
        loop._settings = MagicMock()
        loop._settings.mode = "sandbox"
        loop._stop_loss_lock = threading.Lock()
        loop._stop_states = {}
        loop._sentiment_cache = {}
        loop._sentiment_lock = threading.Lock()
        loop._cache = None
        loop._anomaly_detector = MagicMock()
        loop._anomaly_detector.check.return_value = None
        loop._async_loop = None
        loop._cycle_dropped_no_bars = 0
        loop._cycle_dropped_below_threshold = 0
        loop._cycle_dropped_pre_trade = 0
        loop._cycle_instruments_processed = 0

        instrument = MagicMock()
        instrument.symbol = "SBER"
        instrument.figi = "BBG004730N88"
        instrument.segment_id = "ru_blue_chips"

        fetcher = MagicMock()
        candles = [_make_candle("SBER")]
        fetcher.fetch_candles.return_value = candles

        return loop, instrument, fetcher, candles

    @patch("finalayze.orchestration.trading_loop.DataNormalizer")
    def test_normalize_batch_called_before_generate_signal(
        self, mock_normalizer_cls: MagicMock
    ) -> None:
        """normalize_batch must be called on fetched candles before generate_signal."""
        loop, instrument, fetcher, candles = self._make_loop_and_deps()

        mock_normalizer = MagicMock()
        mock_normalizer.normalize_batch.return_value = candles
        mock_normalizer_cls.return_value = mock_normalizer

        now = datetime.now(UTC)
        loop._process_instrument(instrument, "moex", MagicMock(), fetcher, now)

        mock_normalizer.normalize_batch.assert_called_once_with(candles)
        loop._strategy.generate_signal.assert_called_once()

    @patch("finalayze.orchestration.trading_loop.DataNormalizer")
    def test_empty_after_normalization_skips_generate_signal(
        self, mock_normalizer_cls: MagicMock
    ) -> None:
        """If normalize_batch filters out all candles, generate_signal must not be called."""
        loop, instrument, fetcher, _candles = self._make_loop_and_deps()

        mock_normalizer = MagicMock()
        mock_normalizer.normalize_batch.return_value = []  # all invalid
        mock_normalizer_cls.return_value = mock_normalizer

        now = datetime.now(UTC)
        loop._process_instrument(instrument, "moex", MagicMock(), fetcher, now)

        loop._strategy.generate_signal.assert_not_called()


# ── DATA-02: _is_candle_stale called in _process_instrument ──


class TestStalenessCheck:
    """Verify stale candle detection in _process_instrument."""

    @patch("finalayze.orchestration.trading_loop.DataNormalizer")
    def test_stale_candles_skip_generate_signal(self, mock_normalizer_cls: MagicMock) -> None:
        """When latest candle is older than threshold, generate_signal must not be called."""
        import threading

        from finalayze.orchestration.trading_loop import TradingLoop

        loop = TradingLoop.__new__(TradingLoop)
        loop._last_prices = {}
        loop._cycle_errors_caught = 0
        loop._cycle_signals_generated = 0
        loop._cycle_orders_submitted = 0
        loop._cycle_orders_filled = 0
        loop._cycle_exited_symbols = set()
        loop._health_monitor = None
        loop._metrics = None
        loop._broker_router = MagicMock()
        loop._strategy = MagicMock()
        loop._strategy.generate_signal.return_value = None
        loop._validation_logger = MagicMock(spec_set=["log_cycle"])
        loop._settings = MagicMock()
        loop._settings.mode = "sandbox"
        loop._stop_loss_lock = threading.Lock()
        loop._stop_states = {}
        loop._sentiment_cache = {}
        loop._sentiment_lock = threading.Lock()
        loop._cache = None
        loop._anomaly_detector = MagicMock()
        loop._anomaly_detector.check.return_value = None
        loop._async_loop = None
        loop._cycle_dropped_no_bars = 0
        loop._cycle_dropped_below_threshold = 0
        loop._cycle_dropped_pre_trade = 0
        loop._cycle_instruments_processed = 0

        instrument = MagicMock()
        instrument.symbol = "SBER"
        instrument.figi = "BBG004730N88"
        instrument.segment_id = "ru_blue_chips"

        # Candle from 3 days ago (stale -- exceeds 48h threshold)
        stale_ts = datetime.now(UTC) - timedelta(hours=72)
        stale_candle = _make_candle("SBER", ts=stale_ts)

        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = [stale_candle]

        mock_normalizer = MagicMock()
        mock_normalizer.normalize_batch.return_value = [stale_candle]
        mock_normalizer_cls.return_value = mock_normalizer

        now = datetime.now(UTC)
        loop._process_instrument(instrument, "moex", MagicMock(), fetcher, now)

        loop._strategy.generate_signal.assert_not_called()

    @patch("finalayze.orchestration.trading_loop.DataNormalizer")
    def test_fresh_candles_proceed_to_generate_signal(self, mock_normalizer_cls: MagicMock) -> None:
        """When candles are fresh (within threshold), generate_signal must be called."""
        import threading

        from finalayze.orchestration.trading_loop import TradingLoop

        loop = TradingLoop.__new__(TradingLoop)
        loop._last_prices = {}
        loop._cycle_errors_caught = 0
        loop._cycle_signals_generated = 0
        loop._cycle_orders_submitted = 0
        loop._cycle_orders_filled = 0
        loop._cycle_exited_symbols = set()
        loop._health_monitor = None
        loop._metrics = None
        loop._broker_router = MagicMock()
        loop._strategy = MagicMock()
        loop._strategy.generate_signal.return_value = None
        loop._validation_logger = MagicMock(spec_set=["log_cycle"])
        loop._settings = MagicMock()
        loop._settings.mode = "sandbox"
        loop._stop_loss_lock = threading.Lock()
        loop._stop_states = {}
        loop._sentiment_cache = {}
        loop._sentiment_lock = threading.Lock()
        loop._cache = None
        loop._anomaly_detector = MagicMock()
        loop._anomaly_detector.check.return_value = None
        loop._async_loop = None
        loop._cycle_dropped_no_bars = 0
        loop._cycle_dropped_below_threshold = 0
        loop._cycle_dropped_pre_trade = 0
        loop._cycle_instruments_processed = 0

        instrument = MagicMock()
        instrument.symbol = "SBER"
        instrument.figi = "BBG004730N88"
        instrument.segment_id = "ru_blue_chips"

        # Fresh candle from 1 hour ago
        fresh_ts = datetime.now(UTC) - timedelta(hours=1)
        fresh_candle = _make_candle("SBER", ts=fresh_ts)

        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = [fresh_candle]

        mock_normalizer = MagicMock()
        mock_normalizer.normalize_batch.return_value = [fresh_candle]
        mock_normalizer_cls.return_value = mock_normalizer

        now = datetime.now(UTC)
        loop._process_instrument(instrument, "moex", MagicMock(), fetcher, now)

        loop._strategy.generate_signal.assert_called_once()
