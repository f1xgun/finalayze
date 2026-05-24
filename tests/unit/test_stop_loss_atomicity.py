"""Tests proving stop-loss check-and-sell is atomic under a single lock hold (CONC-01).

Verifies that the entire read-check-sell-remove sequence in _check_stop_losses
happens under a single _stop_loss_lock acquisition, preventing double-sells from
concurrent threads.
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from finalayze.execution.simulated_broker import StopLossState

_ZERO = Decimal(0)


def _make_stop_state(stop_price: Decimal, entry_price: Decimal | None = None) -> StopLossState:
    """Create a StopLossState with given stop price for backward-compat tests."""
    ep = entry_price or stop_price + Decimal(10)
    atr = (ep - stop_price) / Decimal("2.0")
    return StopLossState(
        initial_stop=stop_price,
        current_stop=stop_price,
        highest_price=ep,
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=ep,
        atr_value=atr,
    )


def _make_trading_loop(
    broker_positions: dict[str, Decimal] | None = None,
    submit_side_effect: Any = None,
) -> Any:
    """Create a minimal TradingLoop with mocked dependencies for stop-loss testing."""
    # Avoid importing the full module at module level -- heavy deps
    with patch.dict("sys.modules", {}):
        pass

    from finalayze.core.trading_loop import TradingLoop

    mock_settings = MagicMock()
    mock_settings.effective_risk_limits.return_value = MagicMock(
        max_position_pct=Decimal("0.1"),
        max_positions_per_market=10,
        max_sector_concentration_pct=Decimal("0.3"),
        min_cash_reserve_pct=Decimal("0.05"),
    )
    mock_settings.segment_ids = ["us_tech"]
    mock_settings.market_ids = ["us"]
    mock_settings.work_mode = "sandbox"
    mock_settings.kelly_fraction = 0.5

    mock_broker = MagicMock()
    if broker_positions is not None:
        mock_broker.get_positions.return_value = broker_positions
    else:
        mock_broker.get_positions.return_value = {"AAPL": Decimal(100)}

    if submit_side_effect is not None:
        mock_broker.submit_order.side_effect = submit_side_effect

    mock_router = MagicMock()
    mock_router.route.return_value = mock_broker

    loop = TradingLoop(
        settings=mock_settings,
        fetchers={"us": MagicMock()},
        news_fetcher=MagicMock(),
        news_analyzer=MagicMock(),
        event_classifier=MagicMock(),
        impact_estimator=MagicMock(),
        strategy=MagicMock(),
        broker_router=mock_router,
        circuit_breakers={"us": MagicMock()},
        cross_market_breaker=MagicMock(),
        alerter=MagicMock(),
        instrument_registry=MagicMock(),
    )

    return loop, mock_broker


class TestStopLossAtomicity:
    """CONC-01: Stop-loss check-and-sell must be atomic under single lock hold."""

    def test_single_lock_hold_for_check_sell_remove(self) -> None:
        """The entire check+sell+remove happens under one _stop_loss_lock hold.

        We verify this by checking that between reading the stop price and
        submitting the order, the lock is continuously held (no intermediate release).
        """
        loop, mock_broker = _make_trading_loop()

        # Set a stop-loss state
        loop._position_tracker._stop_states["AAPL"] = _make_stop_state(
            Decimal("140.00"), Decimal("150.00")
        )
        # Set an entry price for Kelly update
        loop._position_tracker._entry_prices["AAPL"] = Decimal("150.00")

        lock_release_detected = threading.Event()
        original_route = loop._broker_router.route

        def route_checking_lock(market_id: str) -> Any:
            """Check that _stop_loss_lock is still held when route() is called."""
            # Try to acquire the lock with timeout=0 (non-blocking)
            # If we CAN acquire it, the lock was released between read and sell (BUG)
            acquired = loop._position_tracker._stop_loss_lock.acquire(blocking=False)
            if acquired:
                loop._position_tracker._stop_loss_lock.release()
                lock_release_detected.set()
            return original_route(market_id)

        loop._broker_router.route = route_checking_lock

        # Trigger stop-loss (price below stop)
        loop._position_tracker.check_stop_losses("us", "AAPL", Decimal("139.00"))

        # The lock should NOT have been released between read and sell
        assert not lock_release_detected.is_set(), (
            "Lock was released between reading stop price and submitting order (TOCTOU bug)"
        )

        # Order should have been submitted
        assert mock_broker.submit_order.call_count == 1

        # Stop state should be cleared
        assert "AAPL" not in loop._position_tracker._stop_states

    def test_concurrent_threads_only_one_sell(self) -> None:
        """If two concurrent threads trigger stop for same symbol, only one submits a sell.

        Uses threading.Barrier to synchronize two threads calling _check_stop_losses
        at the same instant.
        """
        loop, mock_broker = _make_trading_loop()

        loop._position_tracker._stop_states["AAPL"] = _make_stop_state(
            Decimal("140.00"), Decimal("150.00")
        )
        loop._position_tracker._entry_prices["AAPL"] = Decimal("150.00")

        barrier = threading.Barrier(2, timeout=5)
        results: list[Exception | None] = [None, None]

        def thread_fn(idx: int) -> None:
            try:
                barrier.wait()
                loop._position_tracker.check_stop_losses("us", "AAPL", Decimal("139.00"))
            except Exception as e:
                results[idx] = e

        t1 = threading.Thread(target=thread_fn, args=(0,))
        t2 = threading.Thread(target=thread_fn, args=(1,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # No exceptions
        assert results[0] is None, f"Thread 0 raised: {results[0]}"
        assert results[1] is None, f"Thread 1 raised: {results[1]}"

        # Only ONE sell order should have been submitted (not two)
        assert mock_broker.submit_order.call_count == 1, (
            f"Expected 1 submit_order call, got {mock_broker.submit_order.call_count} "
            f"(double-sell bug)"
        )

        # Stop state should be cleared
        assert "AAPL" not in loop._position_tracker._stop_states

    def test_stop_price_preserved_on_submit_failure(self) -> None:
        """If submit_order raises, stop price is NOT cleared (retry next cycle)."""
        loop, mock_broker = _make_trading_loop(
            submit_side_effect=RuntimeError("broker down"),
        )

        loop._position_tracker._stop_states["AAPL"] = _make_stop_state(
            Decimal("140.00"), Decimal("150.00")
        )
        loop._position_tracker._entry_prices["AAPL"] = Decimal("150.00")

        # Should not raise -- exception is caught internally
        loop._position_tracker.check_stop_losses("us", "AAPL", Decimal("139.00"))

        # submit_order was called (and raised)
        assert mock_broker.submit_order.call_count == 1

        # Stop state should still be set (preserved for retry)
        assert "AAPL" in loop._position_tracker._stop_states
        assert loop._position_tracker._stop_states["AAPL"].current_stop == Decimal("140.00")

    def test_no_sell_when_price_above_stop(self) -> None:
        """No order submitted when current price is above stop price."""
        loop, mock_broker = _make_trading_loop()
        loop._position_tracker._stop_states["AAPL"] = _make_stop_state(Decimal("140.00"))

        loop._position_tracker.check_stop_losses("us", "AAPL", Decimal("145.00"))

        mock_broker.submit_order.assert_not_called()
        # Stop state should still be set
        assert "AAPL" in loop._position_tracker._stop_states

    def test_no_sell_when_no_stop_price(self) -> None:
        """No order submitted when symbol has no stop price set."""
        loop, mock_broker = _make_trading_loop()

        loop._position_tracker.check_stop_losses("us", "AAPL", Decimal("139.00"))

        mock_broker.submit_order.assert_not_called()
        mock_broker.get_positions.assert_not_called()
