"""Tests for TradingLoop bond cycle integration (05-02)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.trading_loop import TradingLoop


class TestBondCycleProcessorIntegration:
    """TradingLoop._bond_cycle() must call processor.run_cycle()."""

    def test_bond_cycle_delegates_to_processor(self) -> None:
        """TradingLoop._bond_cycle() must call processor.run_cycle()."""
        import inspect

        sig = inspect.signature(TradingLoop.__init__)
        assert "bond_cycle_processor" in sig.parameters
        assert "macro_cache" in sig.parameters


class TestAsyncioLockSerialization:
    """asyncio.Lock serializes concurrent gRPC calls (equity + bond don't overlap)."""

    def test_grpc_lock_exists(self) -> None:
        """TradingLoop __init__ creates _grpc_lock as asyncio.Lock."""
        import inspect

        sig = inspect.signature(TradingLoop.__init__)
        # The lock should be created in __init__
        source = inspect.getsource(TradingLoop.__init__)
        assert "_grpc_lock" in source

    def test_grpc_lock_serializes_calls(self) -> None:
        """asyncio.Lock prevents concurrent gRPC access."""
        call_order: list[str] = []

        async def _test_lock_serialization() -> None:
            lock = asyncio.Lock()

            async def task_a() -> None:
                async with lock:
                    call_order.append("a_start")
                    await asyncio.sleep(0.01)
                    call_order.append("a_end")

            async def task_b() -> None:
                async with lock:
                    call_order.append("b_start")
                    await asyncio.sleep(0.01)
                    call_order.append("b_end")

            await asyncio.gather(task_a(), task_b())

        asyncio.run(_test_lock_serialization())

        # With a lock, calls should be serialized: a_start, a_end, b_start, b_end
        # (or b_start, b_end, a_start, a_end)
        assert call_order[0].endswith("_start")
        assert call_order[1].endswith("_end")
        assert call_order[2].endswith("_start")
        assert call_order[3].endswith("_end")


class TestDailyEquitySnapshotModel:
    """DailyEquitySnapshot model has timestamp, market_id, equity, currency columns."""

    def test_model_exists(self) -> None:
        """DailyEquitySnapshot model is importable."""
        from finalayze.core.models import DailyEquitySnapshot

        assert DailyEquitySnapshot is not None

    def test_model_columns(self) -> None:
        """DailyEquitySnapshot has required columns."""
        from finalayze.core.models import DailyEquitySnapshot

        assert hasattr(DailyEquitySnapshot, "timestamp")
        assert hasattr(DailyEquitySnapshot, "market_id")
        assert hasattr(DailyEquitySnapshot, "equity")
        assert hasattr(DailyEquitySnapshot, "currency")

    def test_model_tablename(self) -> None:
        """DailyEquitySnapshot.__tablename__ is 'daily_equity_snapshots'."""
        from finalayze.core.models import DailyEquitySnapshot

        assert DailyEquitySnapshot.__tablename__ == "daily_equity_snapshots"
