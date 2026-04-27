"""Tests for MetricsCollector wiring in TradingLoop (6D.9).

MetricsCollector is now injected via constructor (self._metrics).
Tests inject a MagicMock as the metrics_collector to verify calls.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest


class TestSubmitOrderMetrics:
    """Verify MetricsCollector is called on trade fill and rejection."""

    def test_record_trade_on_fill(self) -> None:
        """MetricsCollector.record_trade called when order is filled."""
        from finalayze.orchestration.signal_executor import SignalExecutor

        mc = MagicMock()
        loop = _make_loop_stub(metrics=mc)
        order = MagicMock()
        order.side = "BUY"
        order.symbol = "AAPL"

        result = MagicMock()
        result.filled = True
        result.fill_price = Decimal("150.00")
        loop._broker_router.submit.return_value = result

        SignalExecutor._submit_order(loop, order, "us", candles=[_fake_candle()])

        mc.record_trade.assert_called_once()
        call_kwargs = mc.record_trade.call_args
        assert call_kwargs[1]["market"] == "us" or call_kwargs[0][0] == "us"

    def test_record_rejection_on_unfilled(self) -> None:
        """MetricsCollector.record_rejection called when order is not filled."""
        from finalayze.orchestration.signal_executor import SignalExecutor

        mc = MagicMock()
        loop = _make_loop_stub(metrics=mc)
        order = MagicMock()
        order.side = "BUY"
        order.symbol = "AAPL"

        result = MagicMock()
        result.filled = False
        result.reason = "insufficient funds"
        loop._broker_router.submit.return_value = result

        SignalExecutor._submit_order(loop, order, "us")

        mc.record_rejection.assert_called_once()


class TestProcessInstrumentMetrics:
    """Verify MetricsCollector.record_signal is called after signal generation."""

    def test_record_signal_called(self) -> None:
        from finalayze.core.schemas import SignalDirection
        from finalayze.orchestration.signal_executor import SignalExecutor

        mc = MagicMock()
        loop = _make_loop_stub(metrics=mc)

        signal = MagicMock()
        signal.direction = SignalDirection.BUY
        signal.confidence = 0.8
        signal.strategy_name = "momentum"
        loop._strategy.generate_signal.return_value = signal

        candles = [_fake_candle()]
        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = candles

        instrument = MagicMock()
        instrument.symbol = "AAPL"
        instrument.segment_id = "us_tech"

        from finalayze.risk.circuit_breaker import CircuitLevel

        # Mock pre-trade checker to pass
        pre_result = MagicMock()
        pre_result.passed = True
        loop._pre_trade_checker.check.return_value = pre_result

        # Mock portfolio
        portfolio = MagicMock()
        portfolio.cash = Decimal(10000)
        portfolio.equity = Decimal(10000)
        portfolio.positions = {}

        broker = MagicMock()
        broker.has_position.return_value = False
        loop._broker_router.route.return_value = broker

        # Mock submit result
        submit_result = MagicMock()
        submit_result.filled = True
        submit_result.fill_price = Decimal("150.00")
        loop._broker_router.submit.return_value = submit_result

        loop._sentiment_mgr = MagicMock()
        loop._sentiment_mgr.get_sentiment.return_value = 0.0
        loop._segment_min_confidence = {}
        loop._last_prices = {}
        loop._ml_registry = None
        loop._loss_limit_tracker = MagicMock()
        loop._macro_cache = None
        loop._compute_total_equity_base = MagicMock(return_value=Decimal(10000))
        loop._get_market_equity = MagicMock(return_value=Decimal(10000))

        SignalExecutor.process_instrument(
            loop,
            instrument,
            "us",
            CircuitLevel.NORMAL,
            fetcher,
            MagicMock(),
            equity=Decimal(10000),
            cash=Decimal(10000),
            portfolio=portfolio,
        )

        mc.record_signal.assert_called_once()


class TestMarketCycleMetrics:
    """Verify portfolio equity and circuit breaker level metrics are set."""

    def test_equity_and_cb_level_set(self) -> None:
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.risk.circuit_breaker import CircuitLevel

        mc = MagicMock()
        loop = _make_loop_stub(metrics=mc)
        loop._registry.list_by_market.return_value = []  # no instruments
        loop._signal_executor = MagicMock()
        loop._is_market_open = MagicMock(return_value=True)
        loop._get_cached_portfolio = MagicMock(return_value=None)
        # Mock broker portfolio
        portfolio_mock = MagicMock()
        portfolio_mock.cash = Decimal(50000)
        portfolio_mock.equity = Decimal(50000)
        loop._broker_router.route.return_value.get_portfolio.return_value = portfolio_mock
        loop._cycle_instruments_processed = 0
        loop._cycle_signals_generated = 0
        loop._cycle_orders_submitted = 0
        loop._cycle_orders_filled = 0
        loop._cycle_errors_caught = 0
        loop._cycle_dropped_no_bars = 0
        loop._cycle_dropped_below_threshold = 0
        loop._cycle_dropped_pre_trade = 0

        market_equities = {"us": Decimal(50000)}

        TradingLoop._process_market_cycle(
            loop, "us", CircuitLevel.NORMAL, market_equities, MagicMock()
        )

        mc.set_portfolio_equity.assert_called_once_with("us", 50000.0)
        mc.set_circuit_breaker_level.assert_called_once()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_candle() -> object:
    from datetime import UTC, datetime, timedelta

    from finalayze.core.schemas import Candle

    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
        open=Decimal("148.00"),
        high=Decimal("155.00"),
        low=Decimal("145.00"),
        close=Decimal("150.00"),
        volume=1000,
    )


def _make_loop_stub(*, metrics: MagicMock | None = None) -> MagicMock:
    """Create a MagicMock that acts as a TradingLoop instance with enough attributes."""
    import threading

    from finalayze.core.trading_loop import TradingLoop
    from finalayze.execution.broker_base import OrderRequest
    from finalayze.risk.circuit_breaker import CircuitLevel

    loop = MagicMock()
    loop._OrderRequest = OrderRequest
    loop._CircuitLevel = CircuitLevel
    loop._stop_loss_lock = threading.Lock()
    loop._stop_states = {}
    loop._sentiment_lock = threading.Lock()
    loop._sentiment_cache = {}
    loop._cache = None
    loop._event_bus = None
    loop._fx_service = None
    loop._metrics = metrics
    loop._alerter = MagicMock()
    loop._broker_router = MagicMock()
    loop._broker_router.route.return_value.has_position.return_value = False
    loop._strategy = MagicMock()
    loop._registry = MagicMock()
    loop._settings = MagicMock()
    loop._settings.max_cross_market_exposure_pct = 0.80
    loop._settings.kelly_fraction = 0.5
    loop._pre_trade_checker = MagicMock()
    loop._kelly_sizer = MagicMock()
    loop._kelly_sizer.optimal_fraction.return_value = Decimal("0.1")
    loop._circuit_breakers = {}
    loop._fx = MagicMock()
    loop._cycle_portfolio_cache = {}
    loop._is_candle_stale = TradingLoop._is_candle_stale
    loop._position_tracker = MagicMock()
    loop._position_tracker._stop_states = {}
    loop._position_tracker._entry_prices = {}
    loop._position_tracker._entry_strategy = {}
    loop._position_tracker._cycle_exited_symbols = set()
    loop._persistence = MagicMock()
    loop._health_monitor = None
    loop._sandbox_monitor = None
    loop._cycle_instruments_processed = 0
    loop._cycle_signals_generated = 0
    loop._cycle_orders_submitted = 0
    loop._cycle_orders_filled = 0
    loop._cycle_errors_caught = 0
    loop._cycle_exited_symbols = set()
    loop._anomaly_detector = MagicMock()
    loop._anomaly_detector.check.return_value = None
    loop._daily_reporter = MagicMock()
    return loop


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 56 EQTY-01 / D-02 Route B: per-cycle equity snapshot wiring
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyCyclePersistsEquitySnapshot:
    """Verify _strategy_cycle_impl wires DailyReportingService.persist_cycle_snapshot."""

    @staticmethod
    def _make_strategy_cycle_stub() -> MagicMock:
        """Stub a TradingLoop suitable for invoking _strategy_cycle_impl()."""
        from datetime import UTC, datetime

        from finalayze.risk.circuit_breaker import CircuitLevel

        loop = _make_loop_stub()
        loop._CircuitLevel = CircuitLevel
        loop._now = MagicMock(return_value=datetime(2026, 4, 19, 12, 0, tzinfo=UTC))
        loop._circuit_breakers = {"us": MagicMock()}
        loop._circuit_breakers["us"].check.return_value = CircuitLevel.NORMAL
        loop._baseline_equities = {"us": Decimal(50000)}
        loop._get_market_equity = MagicMock(return_value=Decimal(50000))
        loop._cross_market_breaker = MagicMock()
        loop._cross_market_breaker.check.return_value = False
        loop._loss_limit_tracker = MagicMock()
        loop._loss_limit_tracker.is_halted.return_value = False
        loop._liquidate_market = MagicMock()
        loop._process_market_cycle = MagicMock()
        loop._build_symbol_to_market_map = MagicMock(return_value={})
        loop._last_prices = {}
        return loop

    def test_strategy_cycle_persists_equity_snapshot_after_market_loop(self) -> None:
        """Cycle hook fires DailyReportingService.persist_cycle_snapshot AFTER per-market loop.

        EQTY-01 D-02 Route B: snapshot writer is called from
        TradingLoop._strategy_cycle_impl() right after the per-market loop and
        the existing position_tracker.snapshot_all_stops_to_db() call.
        """
        from finalayze.core.trading_loop import TradingLoop

        loop = self._make_strategy_cycle_stub()

        # Use a parent Mock to capture call ordering across both attached children
        parent = MagicMock()
        parent.attach_mock(loop._position_tracker.snapshot_all_stops_to_db, "snap_stops")
        parent.attach_mock(loop._daily_reporter.persist_cycle_snapshot, "persist_snap")

        TradingLoop._strategy_cycle_impl(loop)

        # Assert: persist_cycle_snapshot is called once with cycle's "now"
        loop._daily_reporter.persist_cycle_snapshot.assert_called_once_with(loop._now.return_value)

        # Assert ordering: stop snapshot comes BEFORE persist_cycle_snapshot
        call_names = [c[0] for c in parent.mock_calls]
        assert "snap_stops" in call_names
        assert "persist_snap" in call_names
        assert call_names.index("snap_stops") < call_names.index("persist_snap")

    def test_strategy_cycle_skips_persist_on_loss_limit_halt(self) -> None:
        """When loss_limit_tracker is halted, persist_cycle_snapshot must NOT be called.

        Per Pitfall 6 in 56-RESEARCH: halt-path early returns must skip the
        snapshot writer (a halted cycle did not complete normally).
        """
        from finalayze.core.trading_loop import TradingLoop

        loop = self._make_strategy_cycle_stub()
        loop._loss_limit_tracker.is_halted.return_value = True

        TradingLoop._strategy_cycle_impl(loop)

        loop._daily_reporter.persist_cycle_snapshot.assert_not_called()

    def test_strategy_cycle_skips_persist_on_cross_market_halt(self) -> None:
        """When cross-market breaker trips, persist_cycle_snapshot must NOT be called."""
        from finalayze.core.trading_loop import TradingLoop

        loop = self._make_strategy_cycle_stub()
        loop._cross_market_breaker.check.return_value = True

        TradingLoop._strategy_cycle_impl(loop)

        loop._daily_reporter.persist_cycle_snapshot.assert_not_called()

    def test_strategy_cycle_swallows_persist_failure(self) -> None:
        """When persist_cycle_snapshot raises, the cycle must NOT propagate (PERSIST-05)."""
        from finalayze.core.trading_loop import TradingLoop

        loop = self._make_strategy_cycle_stub()
        loop._daily_reporter.persist_cycle_snapshot.side_effect = RuntimeError("DB down")

        # Should NOT raise -- failure is logged and swallowed inside _strategy_cycle_impl
        TradingLoop._strategy_cycle_impl(loop)

        loop._daily_reporter.persist_cycle_snapshot.assert_called_once()
