"""Direct unit tests for PositionTracker.maybe_register_retroactive_stop.

Extracted from signal_executor.process_instrument (Phase 2a). The legacy
behavior is also covered indirectly by tests/unit/test_stop_restore_on_restart.py
through process_instrument; these tests target the method API directly.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core.schemas import Candle
from finalayze.orchestration.position_manager import PositionTracker

# Test constants (no magic numbers per ruff PLR2004).
_BASE_PRICE = 7.3
_QTY = Decimal(170000)
_ZERO = Decimal(0)


def _make_candles(n: int = 30, base_price: float = _BASE_PRICE) -> list[Candle]:
    now = datetime.datetime.now(tz=datetime.UTC)
    out: list[Candle] = []
    for i in range(n):
        p = Decimal(str(base_price + i * 0.01))
        ts = now - datetime.timedelta(days=n - i - 1)
        out.append(
            Candle(
                symbol="CBOM",
                market_id="moex",
                timeframe="1d",
                timestamp=ts,
                open=p - Decimal("0.1"),
                high=p + Decimal("0.2"),
                low=p - Decimal("0.2"),
                close=p,
                volume=10000,
            )
        )
    return out


def _make_tracker() -> PositionTracker:
    return PositionTracker(kelly_sizer=MagicMock(), broker_router=MagicMock())


class TestSkipPaths:
    def test_already_has_stop_returns_false_and_noop(self) -> None:
        tracker = _make_tracker()
        candles = _make_candles()
        # Pre-register a stop so the method sees has_stop=True
        from finalayze.execution.simulated_broker import StopLossState

        existing = StopLossState(
            initial_stop=Decimal(7),
            current_stop=Decimal(7),
            highest_price=Decimal(8),
            trail_activated=False,
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
            entry_price=Decimal(8),
            atr_value=Decimal("0.5"),
        )
        tracker.register_entry("CBOM", Decimal(8), "momentum", existing, market_id="moex")

        result = tracker.maybe_register_retroactive_stop("CBOM", candles, "moex")
        assert result is False
        # Existing stop is unchanged
        state = tracker.get_stop_state("CBOM")
        assert state is not None
        assert state.current_stop == Decimal(7)

    def test_empty_candles_returns_false_and_noop(self) -> None:
        tracker = _make_tracker()
        assert tracker.maybe_register_retroactive_stop("CBOM", [], "moex") is False
        assert not tracker.has_stop("CBOM")


class TestRegistration:
    def test_registers_stop_when_no_existing_state(self) -> None:
        tracker = _make_tracker()
        candles = _make_candles()
        result = tracker.maybe_register_retroactive_stop("CBOM", candles, "moex")
        assert result is True
        assert tracker.has_stop("CBOM")
        state = tracker.get_stop_state("CBOM")
        assert state is not None
        assert state.current_stop > _ZERO

    def test_moex_uses_higher_atr_multiplier_than_us(self) -> None:
        # MOEX uses 2.5x, US uses 2.0x — wider stop on MOEX for the same candles
        candles_moex = _make_candles()
        candles_us = _make_candles()
        t_moex = _make_tracker()
        t_us = _make_tracker()
        t_moex.maybe_register_retroactive_stop("CBOM", candles_moex, "moex")
        t_us.maybe_register_retroactive_stop("CBOM", candles_us, "us")
        s_moex = t_moex.get_stop_state("CBOM")
        s_us = t_us.get_stop_state("CBOM")
        assert s_moex is not None and s_us is not None
        # Wider stop = lower current_stop relative to entry; assert MOEX <= US stop
        assert s_moex.current_stop <= s_us.current_stop

    def test_entry_strategy_default_is_retroactive(self) -> None:
        tracker = _make_tracker()
        tracker.maybe_register_retroactive_stop("CBOM", _make_candles(), "moex")
        assert tracker._entry_strategy["CBOM"] == "retroactive"

    def test_preserves_existing_entry_strategy(self) -> None:
        tracker = _make_tracker()
        tracker._entry_strategy["CBOM"] = "momentum"
        tracker.maybe_register_retroactive_stop("CBOM", _make_candles(), "moex")
        # When _entry_strategy is already populated, retroactive registration reuses it
        assert tracker._entry_strategy["CBOM"] == "momentum"
