"""Unit tests for trailing stop-loss logic in SimulatedBroker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.execution.simulated_broker import SimulatedBroker, StopLossState


def _candle(
    price: Decimal,
    high: Decimal | None = None,
    low: Decimal | None = None,
    symbol: str = "TEST",
    day_offset: int = 0,
) -> Candle:
    """Helper to create a candle with sensible defaults."""
    h = high if high is not None else price + Decimal(1)
    lo = low if low is not None else price - Decimal(1)
    return Candle(
        symbol=symbol,
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=day_offset),
        open=price,
        high=h,
        low=lo,
        close=price,
        volume=1_000_000,
    )


class TestStopLossStateDataclass:
    """StopLossState holds trailing stop metadata."""

    def test_create_stop_loss_state(self) -> None:
        state = StopLossState(
            initial_stop=Decimal(95),
            current_stop=Decimal(95),
            highest_price=Decimal(100),
            trail_activated=False,
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
            entry_price=Decimal(100),
            atr_value=Decimal(5),
        )
        assert state.initial_stop == Decimal(95)
        assert state.current_stop == Decimal(95)
        assert not state.trail_activated

    def test_state_is_mutable(self) -> None:
        state = StopLossState(
            initial_stop=Decimal(95),
            current_stop=Decimal(95),
            highest_price=Decimal(100),
            trail_activated=False,
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
            entry_price=Decimal(100),
            atr_value=Decimal(5),
        )
        state.current_stop = Decimal(97)
        assert state.current_stop == Decimal(97)


class TestSetTrailingStop:
    """SimulatedBroker.set_trailing_stop() creates a StopLossState."""

    def test_set_trailing_stop_creates_state(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        broker.set_trailing_stop(
            symbol="AAPL",
            entry_price=Decimal(100),
            initial_stop=Decimal(90),
            atr_value=Decimal(5),
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        assert "AAPL" in broker._stop_states

    def test_set_stop_loss_backward_compat(self) -> None:
        """set_stop_loss still works and creates a non-trailing state."""
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        broker.set_stop_loss("AAPL", Decimal(90))
        # Should still trigger at the stop price
        # Give the broker a position first
        candle = _candle(Decimal(100), symbol="AAPL")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10)), candle)
        low_candle = _candle(Decimal(89), high=Decimal(91), low=Decimal(88), symbol="AAPL")
        results = broker.check_stop_losses(low_candle)
        assert len(results) == 1
        assert results[0].filled


class TestTrailingStopActivation:
    """Trail activates once price reaches entry + activation_atr * ATR."""

    def test_trail_not_activated_below_threshold(self) -> None:
        """Trail stays inactive when price hasn't exceeded threshold."""
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        candle_buy = _candle(Decimal(100), symbol="NVDA")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(
            OrderRequest(symbol="NVDA", side="BUY", quantity=Decimal(10)), candle_buy
        )
        # Entry 100, ATR 5, activation_atr 1.0 => threshold = 105
        broker.set_trailing_stop(
            symbol="NVDA",
            entry_price=Decimal(100),
            initial_stop=Decimal(90),
            atr_value=Decimal(5),
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        # Price goes up to 104 (below 105 threshold)
        candle = _candle(
            Decimal(103), high=Decimal(104), low=Decimal(101), symbol="NVDA", day_offset=1
        )
        results = broker.check_stop_losses(candle)
        assert len(results) == 0
        assert not broker._stop_states["NVDA"].trail_activated

    def test_trail_activates_at_threshold(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        candle_buy = _candle(Decimal(100), symbol="NVDA")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(
            OrderRequest(symbol="NVDA", side="BUY", quantity=Decimal(10)), candle_buy
        )
        # Entry 100, ATR 5, activation_atr 1.0 => threshold = 105
        broker.set_trailing_stop(
            symbol="NVDA",
            entry_price=Decimal(100),
            initial_stop=Decimal(90),
            atr_value=Decimal(5),
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        # Price reaches 106 (above 105)
        candle = _candle(
            Decimal(105), high=Decimal(106), low=Decimal(103), symbol="NVDA", day_offset=1
        )
        results = broker.check_stop_losses(candle)
        assert len(results) == 0
        assert broker._stop_states["NVDA"].trail_activated


class TestTrailingStopRatchets:
    """Once activated, the trailing stop ratchets up but never down."""

    def test_stop_ratchets_up(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        candle_buy = _candle(Decimal(100), symbol="TEST")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(
            OrderRequest(symbol="TEST", side="BUY", quantity=Decimal(10)), candle_buy
        )
        # Entry 100, ATR 5, trail_atr 1.5 => trail distance = 7.5
        broker.set_trailing_stop(
            symbol="TEST",
            entry_price=Decimal(100),
            initial_stop=Decimal(90),
            atr_value=Decimal(5),
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        # Candle 1: price hits 106 => activates, trail_stop = 106 - 7.5 = 98.5
        c1 = _candle(Decimal(105), high=Decimal(106), low=Decimal(103), symbol="TEST", day_offset=1)
        broker.check_stop_losses(c1)
        assert broker._stop_states["TEST"].current_stop == Decimal("98.5")

        # Candle 2: price hits 110 => trail_stop = 110 - 7.5 = 102.5
        c2 = _candle(Decimal(109), high=Decimal(110), low=Decimal(107), symbol="TEST", day_offset=2)
        broker.check_stop_losses(c2)
        assert broker._stop_states["TEST"].current_stop == Decimal("102.5")

        # Candle 3: price drops to 108 => trail_stop = 108 - 7.5 = 100.5, but stays 102.5
        c3 = _candle(Decimal(107), high=Decimal(108), low=Decimal(105), symbol="TEST", day_offset=3)
        broker.check_stop_losses(c3)
        assert broker._stop_states["TEST"].current_stop == Decimal("102.5")

    def test_trailing_stop_triggers_on_breach(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        candle_buy = _candle(Decimal(100), symbol="TEST")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(
            OrderRequest(symbol="TEST", side="BUY", quantity=Decimal(10)), candle_buy
        )
        broker.set_trailing_stop(
            symbol="TEST",
            entry_price=Decimal(100),
            initial_stop=Decimal(90),
            atr_value=Decimal(5),
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        # Activate: high=106
        c1 = _candle(Decimal(105), high=Decimal(106), low=Decimal(103), symbol="TEST", day_offset=1)
        broker.check_stop_losses(c1)
        # current_stop = 98.5

        # Breach: low=98 <= 98.5
        c2 = _candle(Decimal(99), high=Decimal(100), low=Decimal(98), symbol="TEST", day_offset=2)
        results = broker.check_stop_losses(c2)
        assert len(results) == 1
        assert results[0].filled
        assert results[0].fill_price == Decimal("98.5")


class TestNvdaScenario:
    """NVDA-like scenario: entry at $18.93, trail not yet activated, holds past $16.39."""

    def test_nvda_stays_open_before_activation(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal(100000))
        entry = Decimal("18.93")
        candle_buy = _candle(entry, symbol="NVDA")
        from finalayze.execution.broker_base import OrderRequest

        broker.submit_order(
            OrderRequest(symbol="NVDA", side="BUY", quantity=Decimal(100)), candle_buy
        )
        # ATR=2, activation_atr=1.0 => threshold = 18.93 + 1.0*2 = 20.93
        # initial_stop = 18.93 - 2.5*2 = 13.93
        atr = Decimal(2)
        broker.set_trailing_stop(
            symbol="NVDA",
            entry_price=entry,
            initial_stop=entry - Decimal("2.5") * atr,
            atr_value=atr,
            activation_atr=Decimal("1.0"),
            trail_atr=Decimal("1.5"),
        )
        # Price dips to 16.39 but initial_stop is 13.93, so no trigger
        dip_candle = _candle(
            Decimal(17), high=Decimal(18), low=Decimal("16.39"), symbol="NVDA", day_offset=1
        )
        results = broker.check_stop_losses(dip_candle)
        assert len(results) == 0
        assert broker.has_position("NVDA")
        assert not broker._stop_states["NVDA"].trail_activated
