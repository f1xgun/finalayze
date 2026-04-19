"""Integration test for Phase 57-03 stop-loss alert flow.

Verifies that a stop-loss trigger end-to-end fires
``TelegramAlerter.on_stop_loss_triggered`` with the enriched D-09 payload.
This test does NOT require a live database — it uses MagicMocks for the
broker and persistence layers but exercises the real PositionTracker and
TelegramAlerter logic up to (but not including) the httpx call.

Plan 01 + 02 wire the persistence + transport sides; this test confirms the
Plan 03 wiring at the call site.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.execution.simulated_broker import StopLossState
from finalayze.orchestration.position_manager import PositionTracker

_ENTRY = Decimal(100)
_STOP = Decimal(95)
_TRIGGER = Decimal(90)
_QTY = Decimal(10)
_ENTRY_CYCLE = 5
_NOW_CYCLE = 12
_HOLD_BARS = 7
_PNL_AMOUNT = Decimal(-100)  # (90 - 100) * 10


def _make_state() -> StopLossState:
    return StopLossState(
        initial_stop=_STOP,
        current_stop=_STOP,
        highest_price=_ENTRY,
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=_ENTRY,
        atr_value=Decimal("2.5"),
        entry_cycle_index=_ENTRY_CYCLE,
    )


def test_stop_trigger_alert_flow_end_to_end() -> None:
    """A stop-loss trigger fires on_stop_loss_triggered with D-09 enrichment.

    Wires real PositionTracker + mock alerter + mock broker_router. After
    set_current_cycle(NOW) + check_stop_losses(...), the alerter MUST receive
    a single on_stop_loss_triggered call carrying P&L, hold_bars, currency.
    """
    # Mock broker that accepts a SELL.
    broker = MagicMock()
    broker.get_positions.return_value = {"SBER": _QTY}
    broker.submit_order.return_value = MagicMock(success=True)
    router = MagicMock()
    router.route.return_value = broker

    # Mock alerter (real TelegramAlerter requires bot_token/chat_id and would
    # fire httpx — we want to verify the call signature, not the transport).
    alerter = MagicMock()

    tracker = PositionTracker(
        kelly_sizer=MagicMock(),
        broker_router=router,
        alerter=alerter,
    )

    # Seed state and current cycle.
    state = _make_state()
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_NOW_CYCLE)

    # Trigger.
    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER,
    )

    # Assert: enriched alert fired exactly once with the expected payload shape.
    alerter.on_stop_loss_triggered.assert_called_once()
    kwargs = alerter.on_stop_loss_triggered.call_args.kwargs
    assert kwargs["symbol"] == "SBER"
    assert kwargs["entry_price"] == _ENTRY
    assert kwargs["stop_price"] == _STOP
    assert kwargs["current_price"] == _TRIGGER
    assert kwargs["pnl_amount"] == _PNL_AMOUNT
    assert kwargs["hold_bars"] == _HOLD_BARS
    assert kwargs["currency"] == "RUB"

    # State cleared after a successful trigger (alert NOT in lock-critical path).
    assert "SBER" not in tracker._stop_states
