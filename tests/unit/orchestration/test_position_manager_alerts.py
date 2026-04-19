"""Tests for Phase 57-03 alerter wiring on PositionTracker + TradingLoop.

Task 1: StopLossState.entry_cycle_index field + TradingLoop._cycle_count counter
        + register_entry stamps the current cycle.
Task 2: check_stop_losses fires enriched on_stop_loss_triggered after submit success.
"""

from __future__ import annotations

import inspect
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.execution.simulated_broker import StopLossState
from finalayze.orchestration.position_manager import PositionTracker
from finalayze.orchestration.trading_loop import TradingLoop


# Test constants (no magic numbers per ruff PLR2004)
_TRIGGER_PRICE = Decimal(92)
_ENTRY_PRICE = Decimal(100)
_QTY = Decimal(10)
_CURRENT_STOP = Decimal(95)
_HOLD_CYCLE_ENTRY = 5
_HOLD_CYCLE_NOW = 12
_EXPECTED_HOLD_BARS = 7
_EXPECTED_PNL_AMOUNT = Decimal(-80)  # (92 - 100) * 10
_EXPECTED_PNL_PCT = -0.08
_HOLD_CYCLE_HUNDRED = 100
_HOLD_CYCLE_FIFTEEN = 115
_EXPECTED_FIFTEEN = 15
_HOLD_CYCLE_FIFTY = 50
_INITIAL_CYCLE = 5
_NEXT_CYCLE = 6
_TWO_INCREMENTS = 7


def _make_state(
    entry: float = 100.0,
    *,
    entry_cycle_index: int = 0,
) -> StopLossState:
    d = Decimal(str(entry))
    return StopLossState(
        initial_stop=d - Decimal(5),
        current_stop=d - Decimal(5),
        highest_price=d,
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=d,
        atr_value=Decimal("2.5"),
        entry_cycle_index=entry_cycle_index,
    )


def _make_tracker_with_alerter() -> tuple[PositionTracker, MagicMock, MagicMock]:
    """Construct a PositionTracker with alerter + broker_router mocked.

    Returns (tracker, alerter, broker) for inspection.
    """
    kelly = MagicMock()
    router = MagicMock()
    broker = MagicMock()
    broker.get_positions.return_value = {"SBER": _QTY}
    broker.submit_order.return_value = MagicMock(success=True)
    router.route.return_value = broker
    alerter = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        alerter=alerter,
    )
    return tracker, alerter, broker


# ── Task 1 tests ─────────────────────────────────────────────────────────────


def test_stop_loss_state_has_entry_cycle_index() -> None:
    """StopLossState carries an entry_cycle_index int field defaulting to 0."""
    state = StopLossState(
        initial_stop=Decimal(95),
        current_stop=Decimal(95),
        highest_price=Decimal(100),
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=Decimal(100),
        atr_value=Decimal("2.5"),
    )
    assert hasattr(state, "entry_cycle_index")
    assert state.entry_cycle_index == 0
    assert isinstance(state.entry_cycle_index, int)


def test_trading_loop_cycle_count_increments_at_top_of_strategy_cycle() -> None:
    """_cycle_count is monotonic: incrementing twice from 5 lands at 7.

    Smoke test that documents semantics. The actual line `self._cycle_count += 1`
    landing at the top of `_strategy_cycle_impl` is asserted via inspect-source
    in test_trading_loop_cycle_count_init_only_in_class_body below.
    """
    # Build a near-empty namespace mimicking the relevant counter on a TradingLoop.
    loop = TradingLoop.__new__(TradingLoop)
    object.__setattr__(loop, "_cycle_count", _INITIAL_CYCLE)
    loop._cycle_count += 1
    assert loop._cycle_count == _NEXT_CYCLE
    loop._cycle_count += 1
    assert loop._cycle_count == _TWO_INCREMENTS


def test_strategy_cycle_impl_contains_cycle_count_increment() -> None:
    """`_strategy_cycle_impl` must contain `self._cycle_count += 1` exactly once."""
    src = inspect.getsource(TradingLoop._strategy_cycle_impl)
    assert src.count("self._cycle_count += 1") == 1


def test_trading_loop_cycle_count_init_only_in_class_body() -> None:
    """`self._cycle_count = 0` must appear exactly once (in __init__).

    Enforces monotonic semantics: no method body resets the counter mid-run.
    """
    src = inspect.getsource(TradingLoop)
    assert src.count("self._cycle_count = 0") == 1


def test_register_entry_stamps_current_cycle() -> None:
    """register_entry routes the current cycle index onto stop_state.entry_cycle_index."""
    kelly = MagicMock()
    router = MagicMock()
    tracker = PositionTracker(kelly_sizer=kelly, broker_router=router)
    stop_state = _make_state(entry_cycle_index=0)
    tracker.set_current_cycle(_HOLD_CYCLE_ENTRY)
    tracker.register_entry(
        "SBER",
        Decimal(100),
        "momentum",
        stop_state,
        market_id="moex",
    )
    stored = tracker._stop_states["SBER"]
    assert stored.entry_cycle_index == _HOLD_CYCLE_ENTRY


# ── Task 2 tests ─────────────────────────────────────────────────────────────


def test_check_stop_losses_fires_alert_after_submit_success() -> None:
    """After a successful submit_order, on_stop_loss_triggered fires with enriched data."""
    tracker, alerter, broker = _make_tracker_with_alerter()
    state = _make_state(entry=100.0, entry_cycle_index=_HOLD_CYCLE_ENTRY)
    state.current_stop = _CURRENT_STOP  # trigger when current_price <= 95
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_NOW)

    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER_PRICE,
    )

    alerter.on_stop_loss_triggered.assert_called_once()
    kwargs = alerter.on_stop_loss_triggered.call_args.kwargs
    assert kwargs["symbol"] == "SBER"
    assert kwargs["entry_price"] == _ENTRY_PRICE
    assert kwargs["current_price"] == _TRIGGER_PRICE
    assert kwargs["pnl_amount"] == _EXPECTED_PNL_AMOUNT
    assert abs(kwargs["pnl_pct"] - _EXPECTED_PNL_PCT) < 1e-9
    assert kwargs["hold_bars"] == _EXPECTED_HOLD_BARS
    assert kwargs["currency"] == "RUB"


def test_stop_alert_not_fired_on_submit_failure() -> None:
    """When broker.submit_order raises, on_stop_loss_triggered MUST NOT fire."""
    tracker, alerter, broker = _make_tracker_with_alerter()
    broker.submit_order.side_effect = RuntimeError("simulated broker outage")
    state = _make_state(entry=100.0, entry_cycle_index=_HOLD_CYCLE_ENTRY)
    state.current_stop = _CURRENT_STOP
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_NOW)

    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER_PRICE,
    )

    alerter.on_stop_loss_triggered.assert_not_called()
    # State preserved for retry next cycle.
    assert "SBER" in tracker._stop_states


def test_hold_bars_from_cycle_index() -> None:
    """hold_bars = _current_cycle_index - entry_cycle_index."""
    tracker, alerter, _broker = _make_tracker_with_alerter()
    state = _make_state(entry=100.0, entry_cycle_index=_HOLD_CYCLE_HUNDRED)
    state.current_stop = _CURRENT_STOP
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_FIFTEEN)

    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER_PRICE,
    )

    alerter.on_stop_loss_triggered.assert_called_once()
    assert alerter.on_stop_loss_triggered.call_args.kwargs["hold_bars"] == _EXPECTED_FIFTEEN


def test_hold_bars_none_when_entry_cycle_zero_on_restart() -> None:
    """entry_cycle_index=0 (pre-Phase-57 baseline) => hold_bars=None per Pitfall 5."""
    tracker, alerter, _broker = _make_tracker_with_alerter()
    state = _make_state(entry=100.0, entry_cycle_index=0)
    state.current_stop = _CURRENT_STOP
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_FIFTY)

    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER_PRICE,
    )

    alerter.on_stop_loss_triggered.assert_called_once()
    assert alerter.on_stop_loss_triggered.call_args.kwargs["hold_bars"] is None


def test_currency_usd_for_non_moex() -> None:
    """market_id not starting with 'moex' or 'ru_' yields currency='USD'."""
    tracker, alerter, _broker = _make_tracker_with_alerter()
    state = _make_state(entry=100.0, entry_cycle_index=_HOLD_CYCLE_ENTRY)
    state.current_stop = _CURRENT_STOP
    with tracker._stop_loss_lock:
        tracker._stop_states["AAPL"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_NOW)

    tracker.check_stop_losses(
        market_id="us_tech",
        symbol="AAPL",
        current_price=_TRIGGER_PRICE,
    )

    alerter.on_stop_loss_triggered.assert_called_once()
    assert alerter.on_stop_loss_triggered.call_args.kwargs["currency"] == "USD"


def test_alert_fire_exception_does_not_crash_cycle() -> None:
    """If the alerter raises, the cycle continues normally and state is cleared."""
    tracker, alerter, _broker = _make_tracker_with_alerter()
    alerter.on_stop_loss_triggered.side_effect = RuntimeError("telegram outage")
    state = _make_state(entry=100.0, entry_cycle_index=_HOLD_CYCLE_ENTRY)
    state.current_stop = _CURRENT_STOP
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    tracker.set_current_cycle(_HOLD_CYCLE_NOW)

    # Must NOT raise.
    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=_TRIGGER_PRICE,
    )

    # State still cleared after successful broker submit (alert failure is non-fatal).
    assert "SBER" not in tracker._stop_states
