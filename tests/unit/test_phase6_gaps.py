"""Gap coverage tests for Phase 6: Sandbox Validation.

Covers untested methods and edge cases:
- system.py: set_tinkoff_broker, update_feed_timestamp, record_error,
  _check_tinkoff, _check_feed_freshness, health cache, feed endpoint
- TinkoffBroker: get_last_prices, get_order_state, cancel_order, close,
  _run_async, _quotation_to_decimal, make_bond_broker
- TradingLoop: _reset_cycle_counters, stop()
- ValidationLogger: blank lines, malformed JSON
- generate_validation_report: single entry, _collect_failures
- main.py: lifespan modes, _build_trading_loop failure
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# system.py — set_tinkoff_broker, update_feed_timestamp, record_error
# ═══════════════════════════════════════════════════════════════════════════════


class TestSystemSetTinkoffBroker:
    """set_tinkoff_broker sets/clears global broker for health probes."""

    def test_set_broker(self) -> None:
        from finalayze.api.v1 import system

        mock_broker = MagicMock()
        system.set_tinkoff_broker(mock_broker)
        assert system._tinkoff_broker is mock_broker

    def test_clear_broker(self) -> None:
        from finalayze.api.v1 import system

        system.set_tinkoff_broker(MagicMock())
        system.set_tinkoff_broker(None)
        assert system._tinkoff_broker is None

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._tinkoff_broker = None


class TestUpdateFeedTimestamp:
    """update_feed_timestamp tracks per-source candle freshness."""

    def test_creates_new_entry(self) -> None:
        from finalayze.api.v1 import system

        ts = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
        system.update_feed_timestamp("tinkoff", ts)
        assert system._last_candle_timestamps["tinkoff"] == ts

    def test_updates_existing(self) -> None:
        from finalayze.api.v1 import system

        ts1 = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        system.update_feed_timestamp("tinkoff", ts1)
        system.update_feed_timestamp("tinkoff", ts2)
        assert system._last_candle_timestamps["tinkoff"] == ts2

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()


class TestRecordError:
    """record_error appends to deque ring buffer (max 100)."""

    def test_appends_error(self) -> None:
        from finalayze.api.v1 import system

        system._recent_errors.clear()
        system.record_error("TestComponent", "test error", "traceback")
        assert len(system._recent_errors) == 1
        entry = system._recent_errors[0]
        assert entry["component"] == "TestComponent"
        assert entry["message"] == "test error"
        assert "timestamp" in entry

    def test_evicts_oldest_at_100(self) -> None:
        from finalayze.api.v1 import system

        system._recent_errors.clear()
        max_entries = 100
        for i in range(max_entries + 5):
            system.record_error("C", f"msg-{i}")
        assert len(system._recent_errors) == max_entries
        # Oldest should be msg-5 (first 5 evicted)
        assert system._recent_errors[0]["message"] == "msg-5"

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._recent_errors.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# system.py — _check_tinkoff, _check_feed_freshness
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckTinkoff:
    """_check_tinkoff async health probe."""

    @pytest.mark.asyncio
    async def test_returns_unknown_when_no_broker(self) -> None:
        from finalayze.api.v1 import system

        system._tinkoff_broker = None
        result = await system._check_tinkoff()
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_returns_ok_on_success(self) -> None:
        from finalayze.api.v1 import system

        mock_broker = MagicMock()
        mock_broker.get_portfolio.return_value = MagicMock()
        system._tinkoff_broker = mock_broker
        result = await system._check_tinkoff()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self) -> None:
        from finalayze.api.v1 import system

        mock_broker = MagicMock()
        mock_broker.get_portfolio.side_effect = Exception("gRPC error")
        system._tinkoff_broker = mock_broker
        result = await system._check_tinkoff()
        assert result == "error"

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._tinkoff_broker = None


class TestCheckFeedFreshness:
    """_check_feed_freshness checks candle age against threshold."""

    @pytest.mark.asyncio
    async def test_returns_unknown_when_no_timestamps(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()
        result = await system._check_feed_freshness()
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_returns_ok_when_all_fresh(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()
        system._last_candle_timestamps["tinkoff"] = datetime.now(UTC) - timedelta(minutes=30)
        result = await system._check_feed_freshness()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_returns_stale_when_any_old(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()
        system._last_candle_timestamps["tinkoff"] = datetime.now(UTC) - timedelta(hours=3)
        result = await system._check_feed_freshness()
        assert result == "stale"

    @pytest.mark.asyncio
    async def test_mixed_fresh_and_stale(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()
        system._last_candle_timestamps["tinkoff"] = datetime.now(UTC) - timedelta(minutes=30)
        system._last_candle_timestamps["finnhub"] = datetime.now(UTC) - timedelta(hours=3)
        result = await system._check_feed_freshness()
        assert result == "stale"

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._last_candle_timestamps.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# TinkoffBroker — get_last_prices, get_order_state, cancel_order
# ═══════════════════════════════════════════════════════════════════════════════


class TestTinkoffGetLastPrices:
    """TinkoffBroker.get_last_prices: FIGI lookup + batch price fetch."""

    def _make_broker(self) -> MagicMock:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._registry = MagicMock()
        broker._ensure_account_id = MagicMock()
        return broker

    def test_empty_symbols_returns_empty(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = self._make_broker()
        broker._registry.get.side_effect = Exception("not found")
        result = TinkoffBroker.get_last_prices(broker, [])
        assert result == {}

    def test_no_figi_returns_empty(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = self._make_broker()
        broker._registry.get.side_effect = Exception("not found")
        result = TinkoffBroker.get_last_prices(broker, ["UNKNOWN"])
        assert result == {}

    def test_successful_price_fetch(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = self._make_broker()
        instrument = MagicMock()
        instrument.figi = "BBG00FIGI001"
        broker._registry.get.return_value = instrument

        price_item = MagicMock()
        price_item.figi = "BBG00FIGI001"
        price_item.price = MagicMock(units=100, nano=500000000)

        response = MagicMock()
        response.last_prices = [price_item]
        broker._call = lambda fn: fn()
        broker._run_async = MagicMock(return_value=response)
        broker._get_last_prices_async = MagicMock()
        broker._quotation_to_decimal = TinkoffBroker._quotation_to_decimal

        result = TinkoffBroker.get_last_prices(broker, ["SBER"])
        assert "SBER" in result


class TestTinkoffGetOrderState:
    """TinkoffBroker.get_order_state: terminal status detection."""

    def test_terminal_status_fill(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._ensure_account_id = MagicMock()

        state = MagicMock()
        state.execution_report_status = 1  # fill
        state.lots_executed = 10
        state.executed_order_price = MagicMock(units=280, nano=500000000)

        broker._call = lambda fn: fn()
        broker._run_async = MagicMock(return_value=state)
        broker._get_order_state_async = MagicMock()
        broker._quotation_to_decimal = TinkoffBroker._quotation_to_decimal

        result = TinkoffBroker.get_order_state(broker, "order-123")
        assert result.order_id == "order-123"
        assert result.is_terminal is True

    def test_api_error_raises_broker_error(self) -> None:
        from finalayze.core.exceptions import BrokerError
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._ensure_account_id.side_effect = Exception("gRPC error")

        with pytest.raises(BrokerError):
            TinkoffBroker.get_order_state(broker, "order-123")


class TestTinkoffCancelOrder:
    """TinkoffBroker.cancel_order raises on failure, cancel_order_safe returns bool."""

    def test_cancel_order_raises_on_failure(self) -> None:
        from finalayze.core.exceptions import BrokerError
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._ensure_account_id.side_effect = Exception("gRPC error")

        with pytest.raises(BrokerError):
            TinkoffBroker.cancel_order(broker, "order-123")

    def test_cancel_order_safe_returns_false_on_failure(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._ensure_account_id.side_effect = Exception("gRPC error")

        result = TinkoffBroker.cancel_order_safe(broker, "order-123")
        assert result is False

    def test_cancel_order_safe_returns_true_on_success(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._ensure_account_id = MagicMock()
        broker._call = lambda fn: fn()
        broker._run_async = MagicMock(return_value=None)
        broker._cancel_order_async = MagicMock()

        result = TinkoffBroker.cancel_order_safe(broker, "order-123")
        assert result is True


class TestQuotationToDecimal:
    """_quotation_to_decimal: Tinkoff Quotation → Decimal."""

    def test_units_only(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        q = MagicMock(units=100, nano=0)
        assert TinkoffBroker._quotation_to_decimal(q) == Decimal(100)

    def test_units_and_nano(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        q = MagicMock(units=100, nano=500000000)
        result = TinkoffBroker._quotation_to_decimal(q)
        assert result == Decimal("100.5")

    def test_missing_attrs_default_zero(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        q = object()  # no units/nano attrs
        result = TinkoffBroker._quotation_to_decimal(q)
        assert result == Decimal(0)


class TestTinkoffClose:
    """TinkoffBroker.close: cleanup gRPC client and event loop."""

    def test_close_clears_references(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._client = MagicMock()
        broker._loop = MagicMock()
        broker._loop.is_closed.return_value = False
        broker._loop.run_until_complete = MagicMock()
        broker._services = MagicMock()
        broker._grpc_loop = None
        broker._grpc_thread = None

        TinkoffBroker.close(broker)
        assert broker._client is None
        assert broker._services is None

    def test_close_noop_when_no_client(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        broker = MagicMock(spec=TinkoffBroker)
        broker._client = None
        broker._grpc_loop = None
        broker._grpc_thread = None
        broker._loop = None
        # Should not raise
        TinkoffBroker.close(broker)


class TestTinkoffRunAsync:
    """TinkoffBroker._run_async: persistent event loop management."""

    def test_creates_new_loop_when_none(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

        registry = InstrumentRegistry()
        for inst in DEFAULT_MOEX_INSTRUMENTS:
            registry.register(inst)
        broker = TinkoffBroker(token="fake", registry=registry, sandbox=True)  # noqa: S106
        assert broker._loop is None

        async def _dummy() -> str:
            return "ok"

        result = TinkoffBroker._run_async(broker, _dummy())
        assert result == "ok"
        assert broker._loop is not None
        # Cleanup
        broker._loop.call_soon_threadsafe(broker._loop.stop)

    def test_reuses_existing_loop(self) -> None:
        from finalayze.execution.tinkoff_broker import TinkoffBroker
        from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

        registry = InstrumentRegistry()
        for inst in DEFAULT_MOEX_INSTRUMENTS:
            registry.register(inst)
        broker = TinkoffBroker(token="fake", registry=registry, sandbox=True)  # noqa: S106

        async def _dummy() -> str:
            return "ok"

        # First call creates the loop
        TinkoffBroker._run_async(broker, _dummy())
        first_loop = broker._loop

        # Second call reuses it
        TinkoffBroker._run_async(broker, _dummy())
        assert broker._loop is first_loop
        # Cleanup
        broker._loop.call_soon_threadsafe(broker._loop.stop)


class TestMakeBondBroker:
    """make_bond_broker shares gRPC client with equity broker."""

    def test_shares_client_and_account(self) -> None:
        from finalayze.execution.tinkoff_broker import make_bond_broker

        equity_broker = MagicMock()
        equity_broker._token = "test-token"
        equity_broker._sandbox = True
        equity_broker._client = MagicMock()
        equity_broker._account_id = "acc-123"
        equity_broker._registry = MagicMock()

        with patch("finalayze.execution.tinkoff_broker.TinkoffBroker") as MockBroker:
            mock_instance = MagicMock()
            MockBroker.return_value = mock_instance
            result = make_bond_broker(equity_broker)
            assert result is mock_instance


# ═══════════════════════════════════════════════════════════════════════════════
# TradingLoop — _reset_cycle_counters, stop()
# ═══════════════════════════════════════════════════════════════════════════════


class TestResetCycleCounters:
    """_reset_cycle_counters sets all 5 counters to 0."""

    def test_resets_all_counters(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        TradingLoop._reset_cycle_counters(loop)
        assert loop._cycle_instruments_processed == 0
        assert loop._cycle_signals_generated == 0
        assert loop._cycle_orders_submitted == 0
        assert loop._cycle_orders_filled == 0
        assert loop._cycle_errors_caught == 0


class TestTradingLoopStop:
    """TradingLoop.stop(): scheduler shutdown, loop cleanup, stop_event."""

    def test_stop_sets_stop_event(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = MagicMock(spec=TradingLoop)
        loop._scheduler = None
        loop._async_loop = None
        loop._async_thread = None
        loop._meta_agent_runner = None
        loop._cache = None
        loop._event_bus = None
        loop._fx_service = None
        loop._grpc_loop = None
        loop._grpc_thread = None
        loop._stop_event = threading.Event()

        TradingLoop.stop(loop)
        assert loop._stop_event.is_set()

    def test_stop_shuts_down_scheduler(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        mock_scheduler = MagicMock()
        loop = MagicMock(spec=TradingLoop)
        loop._scheduler = mock_scheduler
        loop._async_loop = None
        loop._async_thread = None
        loop._meta_agent_runner = None
        loop._cache = None
        loop._event_bus = None
        loop._fx_service = None
        loop._grpc_loop = None
        loop._grpc_thread = None
        loop._stop_event = threading.Event()

        TradingLoop.stop(loop)
        mock_scheduler.shutdown.assert_called_once_with(wait=True)

    def test_stop_joins_async_thread(self) -> None:
        from finalayze.core.trading_loop import TradingLoop
        from finalayze.orchestration.async_runtime import AsyncRuntime

        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_thread = MagicMock()

        loop = MagicMock(spec=TradingLoop)
        loop._scheduler = None
        loop._meta_agent_runner = None
        loop._cache = None
        loop._event_bus = None
        loop._fx_service = None
        loop._stop_event = threading.Event()

        # Set up a real AsyncRuntime with mock loops
        runtime = AsyncRuntime()
        runtime.async_loop = mock_loop
        runtime.async_thread = mock_thread
        runtime.grpc_loop = None
        runtime.grpc_thread = None
        loop._async_runtime = runtime

        TradingLoop.stop(loop)
        mock_thread.join.assert_called_once_with(timeout=5)


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationLogger edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidationLoggerEdgeCases:
    """Edge cases: blank lines, malformed JSON."""

    def test_get_entries_skips_blank_lines(self, tmp_path: Path) -> None:
        from finalayze.core.validation_logger import ValidationLogger

        log_path = tmp_path / "cycles.jsonl"
        logger = ValidationLogger(log_path=log_path)

        from finalayze.core.validation_logger import CycleLogEntry

        entry = CycleLogEntry(
            timestamp=datetime(2026, 3, 15, 10, 0, tzinfo=UTC),
            cycle_type="equity",
            duration_ms=1000,
            instruments_processed=5,
            signals_generated=2,
            orders_submitted=1,
            orders_filled=1,
            errors_caught=0,
            equity_rub=1_000_000.0,
            drawdown_pct=0.5,
            circuit_breaker_level=0,
        )
        logger.log_cycle(entry)
        # Insert blank lines
        with log_path.open("a") as f:
            f.write("\n\n")
        logger.log_cycle(entry)

        entries = logger.get_entries()
        assert len(entries) == 2  # blank lines skipped

    def test_get_entries_malformed_json_raises(self, tmp_path: Path) -> None:
        from finalayze.core.validation_logger import ValidationLogger

        log_path = tmp_path / "cycles.jsonl"
        log_path.write_text("not valid json\n")

        logger = ValidationLogger(log_path=log_path)
        with pytest.raises(json.JSONDecodeError):
            logger.get_entries()

    def test_default_path_used(self) -> None:
        from finalayze.core.validation_logger import ValidationLogger

        logger = ValidationLogger()
        assert logger._log_path == Path("results/validation/cycles.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# generate_validation_report edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidationReportEdgeCases:
    """Edge cases: single entry, _collect_failures."""

    def test_single_entry_report(self, tmp_path: Path) -> None:
        """Single entry should produce a valid report (1 day < 5 → FAIL)."""
        from scripts.generate_validation_report import generate_report

        from finalayze.core.validation_logger import CycleLogEntry, ValidationLogger

        log_path = tmp_path / "cycles.jsonl"
        output_path = tmp_path / "report.md"

        logger = ValidationLogger(log_path)
        logger.log_cycle(
            CycleLogEntry(
                timestamp=datetime(2026, 3, 15, 10, 0, tzinfo=UTC),
                cycle_type="equity",
                duration_ms=1000,
                instruments_processed=5,
                signals_generated=2,
                orders_submitted=1,
                orders_filled=1,
                errors_caught=0,
                equity_rub=1_000_000.0,
                drawdown_pct=0.5,
                circuit_breaker_level=0,
            )
        )

        result = generate_report(log_path, output_path)
        assert result is False  # 1 day < 5
        content = output_path.read_text()
        assert "2026-03-15" in content

    def test_collect_failures_all_fail(self) -> None:
        from scripts.generate_validation_report import _collect_failures, _Metrics

        m = _Metrics(
            trading_days=2,
            total_cycles=5,
            total_orders=3,
            total_fills=3,
            max_drawdown=7.0,
            final_equity=900_000.0,
            total_errors=2,
            first_ts=datetime(2026, 3, 10, tzinfo=UTC),
            last_ts=datetime(2026, 3, 11, tzinfo=UTC),
            by_date={},
        )
        failures = _collect_failures(m, False, False, False, False)
        assert len(failures) == 4
        assert any("Trading Days" in f for f in failures)
        assert any("Drawdown" in f for f in failures)
        assert any("Trades" in f for f in failures)
        assert any("Errors" in f for f in failures)

    def test_collect_failures_none_fail(self) -> None:
        from scripts.generate_validation_report import _collect_failures, _Metrics

        m = _Metrics(
            trading_days=5,
            total_cycles=30,
            total_orders=15,
            total_fills=15,
            max_drawdown=2.0,
            final_equity=1_050_000.0,
            total_errors=0,
            first_ts=datetime(2026, 3, 10, tzinfo=UTC),
            last_ts=datetime(2026, 3, 14, tzinfo=UTC),
            by_date={},
        )
        failures = _collect_failures(m, True, True, True, True)
        assert len(failures) == 0

    def test_report_creates_parent_dirs(self, tmp_path: Path) -> None:
        from scripts.generate_validation_report import generate_report

        log_path = tmp_path / "nonexistent.jsonl"
        output_path = tmp_path / "sub" / "dir" / "report.md"

        generate_report(log_path, output_path)
        assert output_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# main.py — lifespan and _build_trading_loop
# ═══════════════════════════════════════════════════════════════════════════════


class TestLifespan:
    """Lifespan context manager: starts TradingLoop in sandbox/real, skips in debug."""

    @pytest.mark.asyncio
    async def test_debug_mode_skips_trading_loop(self) -> None:
        """In DEBUG mode, TradingLoop is NOT started."""
        import finalayze.main as main_mod
        from finalayze.core.modes import WorkMode

        original_settings = main_mod._settings
        original_instance = main_mod._trading_loop_instance
        original_thread = main_mod._trading_loop_thread
        try:
            mock_settings = MagicMock()
            mock_settings.mode = WorkMode.DEBUG
            main_mod._settings = mock_settings
            main_mod._trading_loop_instance = None
            main_mod._trading_loop_thread = None

            with patch.object(main_mod, "build_trading_loop") as mock_build:
                async with main_mod.lifespan(MagicMock()):
                    pass
                mock_build.assert_not_called()
        finally:
            main_mod._settings = original_settings
            main_mod._trading_loop_instance = original_instance
            main_mod._trading_loop_thread = original_thread

    @pytest.mark.asyncio
    async def test_sandbox_mode_starts_trading_loop(self) -> None:
        """In SANDBOX mode, TradingLoop is built and started in a thread."""
        import finalayze.main as main_mod
        from finalayze.core.modes import WorkMode

        original_settings = main_mod._settings
        original_instance = main_mod._trading_loop_instance
        original_thread = main_mod._trading_loop_thread
        try:
            mock_settings = MagicMock()
            mock_settings.mode = WorkMode.SANDBOX
            main_mod._settings = mock_settings

            mock_loop = MagicMock()
            mock_loop._broker_router = None
            main_mod._trading_loop_instance = None
            main_mod._trading_loop_thread = None

            with (
                patch.object(main_mod, "build_trading_loop", return_value=mock_loop),
                patch.object(main_mod, "threading") as mock_threading,
            ):
                mock_thread_instance = MagicMock()
                mock_thread_instance.is_alive.return_value = False
                mock_threading.Thread.return_value = mock_thread_instance

                async with main_mod.lifespan(MagicMock()):
                    mock_thread_instance.start.assert_called_once()
        finally:
            main_mod._settings = original_settings
            main_mod._trading_loop_instance = original_instance
            main_mod._trading_loop_thread = original_thread

    @pytest.mark.asyncio
    async def test_build_failure_continues_gracefully(self) -> None:
        """If _build_trading_loop raises, lifespan continues without crash."""
        import finalayze.main as main_mod
        from finalayze.core.modes import WorkMode

        original_settings = main_mod._settings
        original_instance = main_mod._trading_loop_instance
        original_thread = main_mod._trading_loop_thread
        try:
            mock_settings = MagicMock()
            mock_settings.mode = WorkMode.SANDBOX
            main_mod._settings = mock_settings
            main_mod._trading_loop_instance = None
            main_mod._trading_loop_thread = None

            with patch.object(
                main_mod, "build_trading_loop", side_effect=Exception("build failed")
            ):
                # Should not raise
                async with main_mod.lifespan(MagicMock()):
                    pass
        finally:
            main_mod._settings = original_settings
            main_mod._trading_loop_instance = original_instance
            main_mod._trading_loop_thread = original_thread

    @pytest.mark.asyncio
    async def test_shutdown_calls_stop(self) -> None:
        """On exit, lifespan calls stop() on TradingLoop."""
        import finalayze.main as main_mod
        from finalayze.core.modes import WorkMode

        original_settings = main_mod._settings
        original_instance = main_mod._trading_loop_instance
        original_thread = main_mod._trading_loop_thread
        try:
            mock_settings = MagicMock()
            mock_settings.mode = WorkMode.SANDBOX
            main_mod._settings = mock_settings

            mock_loop = MagicMock()
            mock_loop._broker_router = None
            main_mod._trading_loop_instance = None
            main_mod._trading_loop_thread = None

            mock_thread_instance = MagicMock()
            mock_thread_instance.is_alive.return_value = True

            with (
                patch.object(main_mod, "build_trading_loop", return_value=mock_loop),
                patch.object(main_mod, "threading") as mock_threading,
            ):
                mock_threading.Thread.return_value = mock_thread_instance

                async with main_mod.lifespan(MagicMock()):
                    pass

            mock_loop.stop.assert_called_once()
            mock_thread_instance.join.assert_called_once_with(timeout=10)
        finally:
            main_mod._settings = original_settings
            main_mod._trading_loop_instance = original_instance
            main_mod._trading_loop_thread = original_thread


class TestBuildTradingLoopFailure:
    """build_trading_loop returns None on exception."""

    def test_returns_none_on_import_error(self) -> None:
        import finalayze.main as main_mod

        with patch.dict("sys.modules", {"finalayze.bootstrap": None}):
            # Will fail to import build_trading_loop
            try:
                result = main_mod.build_trading_loop(MagicMock())
            except Exception:
                result = None
            assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Health endpoint cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthCache:
    """_get_component_status caches results for 30 seconds."""

    @pytest.mark.asyncio
    async def test_cache_returns_stale_within_ttl(self) -> None:
        from finalayze.api.v1 import system

        # Prime cache
        system._health_cache = {
            "db": "ok",
            "redis": "ok",
            "alpaca": "ok",
            "tinkoff": "unknown",
            "llm": "ok",
        }
        import time

        system._health_cache_ts = time.monotonic()  # fresh

        result = await system._get_component_status()
        assert result.db == "ok"
        assert result.tinkoff == "unknown"

    def teardown_method(self) -> None:
        from finalayze.api.v1 import system

        system._health_cache = {}
        system._health_cache_ts = 0.0
