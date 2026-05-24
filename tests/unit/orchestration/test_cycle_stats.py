"""Tests for CycleStats — typed return type for SignalExecutor.process_instrument.

Replaces the legacy dict[str, Any] with an immutable dataclass plus an
aggregation operator. Verifies field semantics, immutability, factory
classmethods for each drop reason, and the __add__ accumulator used by
TradingLoop._strategy_cycle.
"""

from __future__ import annotations

import dataclasses

import pytest

from finalayze.orchestration.cycle_stats import CycleStats


class TestDefaults:
    def test_all_counters_default_to_zero(self) -> None:
        stats = CycleStats()
        assert stats.signals_generated == 0
        assert stats.orders_submitted == 0
        assert stats.orders_filled == 0
        assert stats.errors_caught == 0
        assert stats.dropped_no_bars == 0
        assert stats.dropped_below_threshold == 0
        assert stats.dropped_pre_trade == 0

    def test_explicit_construction_preserves_fields(self) -> None:
        stats = CycleStats(signals_generated=1, orders_submitted=1, orders_filled=1)
        assert stats.signals_generated == 1
        assert stats.orders_submitted == 1
        assert stats.orders_filled == 1
        assert stats.errors_caught == 0


class TestImmutability:
    def test_is_frozen(self) -> None:
        stats = CycleStats()
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.signals_generated = 5  # type: ignore[misc]


class TestFactories:
    """Factory classmethods make each drop reason a one-liner at the call site."""

    def test_signal_dropped_threshold(self) -> None:
        assert CycleStats.signal_dropped_threshold() == CycleStats(dropped_below_threshold=1)

    def test_pre_trade_rejected(self) -> None:
        # Preserves legacy semantics: signal_generated counted before reject.
        assert CycleStats.pre_trade_rejected() == CycleStats(
            signals_generated=1, dropped_pre_trade=1
        )

    def test_no_bars(self) -> None:
        assert CycleStats.no_bars() == CycleStats(dropped_no_bars=1)

    def test_error_caught(self) -> None:
        assert CycleStats.error_caught() == CycleStats(errors_caught=1)

    def test_signal_generated(self) -> None:
        assert CycleStats.signal_generated() == CycleStats(signals_generated=1)

    def test_order_submitted_not_filled(self) -> None:
        assert CycleStats.order_submitted(filled=False) == CycleStats(
            signals_generated=1, orders_submitted=1
        )

    def test_order_submitted_and_filled(self) -> None:
        assert CycleStats.order_submitted(filled=True) == CycleStats(
            signals_generated=1, orders_submitted=1, orders_filled=1
        )


class TestAggregation:
    def test_add_combines_field_by_field(self) -> None:
        a = CycleStats(signals_generated=2, dropped_no_bars=1)
        b = CycleStats(signals_generated=3, orders_filled=1)
        total = a + b
        assert total.signals_generated == 5
        assert total.dropped_no_bars == 1
        assert total.orders_filled == 1

    def test_add_returns_new_instance(self) -> None:
        a = CycleStats(signals_generated=2)
        b = CycleStats(signals_generated=3)
        total = a + b
        assert total is not a
        assert total is not b
        # originals untouched
        assert a.signals_generated == 2
        assert b.signals_generated == 3

    def test_zero_is_additive_identity(self) -> None:
        a = CycleStats(signals_generated=2, orders_filled=1)
        assert a + CycleStats() == a
        assert CycleStats() + a == a

    def test_sum_over_iterable(self) -> None:
        instruments = [
            CycleStats.signal_dropped_threshold(),
            CycleStats.pre_trade_rejected(),
            CycleStats.order_submitted(filled=True),
        ]
        total = sum(instruments, start=CycleStats())
        assert total.dropped_below_threshold == 1
        assert total.dropped_pre_trade == 1
        assert total.orders_filled == 1
        # signals_generated is counted twice: once by pre_trade_rejected, once by order_submitted.
        assert total.signals_generated == 2
