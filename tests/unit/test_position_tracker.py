"""Tests for PositionTracker accessors added in Phase 54 (STOP-01, STOP-03).

Task 2.1: get_stop_state and snapshot_all_stops accessors — lock-safe copies.
Task 2.3: persistence wiring (register_entry market_id, check_stop_losses
trigger event, snapshot_all_stops_to_db hook).
"""

from __future__ import annotations

import threading
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.execution.simulated_broker import StopLossState
from finalayze.orchestration.position_manager import PositionTracker


def _make_state(entry: float = 100.0) -> StopLossState:
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
    )


def _make_tracker() -> PositionTracker:
    kelly = MagicMock()
    router = MagicMock()
    return PositionTracker(kelly_sizer=kelly, broker_router=router)


def test_get_stop_state_returns_none_when_absent() -> None:
    tracker = _make_tracker()
    assert tracker.get_stop_state("SBER") is None


def test_get_stop_state_returns_copy() -> None:
    tracker = _make_tracker()
    original = _make_state()
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = original
    returned = tracker.get_stop_state("SBER")
    assert returned is not None
    assert returned is not original, "must return a copy, not the internal reference"
    assert returned.entry_price == original.entry_price


def test_get_stop_state_mutation_does_not_bleed_back() -> None:
    tracker = _make_tracker()
    original = _make_state()
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = original
    returned = tracker.get_stop_state("SBER")
    assert returned is not None
    returned.highest_price = Decimal(999)
    # Second read gets untouched internal state
    again = tracker.get_stop_state("SBER")
    assert again is not None
    assert again.highest_price == Decimal("100.0")


def test_snapshot_all_stops_returns_copies() -> None:
    tracker = _make_tracker()
    st1 = _make_state(100.0)
    st2 = _make_state(200.0)
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = st1
        tracker._stop_states["GAZP"] = st2
    snap = tracker.snapshot_all_stops()
    assert set(snap.keys()) == {"SBER", "GAZP"}
    assert snap["SBER"] is not st1
    assert snap["GAZP"] is not st2
    assert snap["SBER"].entry_price == Decimal("100.0")


def test_snapshot_all_stops_empty() -> None:
    tracker = _make_tracker()
    assert tracker.snapshot_all_stops() == {}


def test_get_stop_state_concurrent_read() -> None:
    """Concurrent reader must never observe a partially-mutated state.

    Simulates check_stop_losses mutating highest_price + current_stop in
    place while a reader calls get_stop_state. The reader must see a
    self-consistent snapshot where (if the ratchet just happened)
    highest_price >= current_stop.
    """
    tracker = _make_tracker()
    state = _make_state(100.0)
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state
    stop_event = threading.Event()
    errors: list[str] = []

    def mutator() -> None:
        while not stop_event.is_set():
            with tracker._stop_loss_lock:
                s = tracker._stop_states.get("SBER")
                if s is not None:
                    s.highest_price += Decimal("0.01")
                    s.current_stop = s.highest_price - Decimal(2)

    def reader() -> None:
        for _ in range(2000):
            snap = tracker.get_stop_state("SBER")
            if snap is not None and snap.highest_price < snap.current_stop:
                errors.append(f"bad snapshot hp={snap.highest_price} cs={snap.current_stop}")
                return

    mt = threading.Thread(target=mutator)
    rt = threading.Thread(target=reader)
    mt.start()
    rt.start()
    rt.join(timeout=5)
    stop_event.set()
    mt.join(timeout=5)
    assert not errors, errors


# ---------------------------------------------------------------------------
# Task 2.3: persistence wiring — register_entry(market_id), check_stop_losses
# trigger event, snapshot_all_stops_to_db hook.
# ---------------------------------------------------------------------------


def test_register_entry_signature_requires_market_id() -> None:
    """register_entry must accept market_id (I-03 resolution -- option A)."""
    import inspect

    sig = inspect.signature(PositionTracker.register_entry)
    assert "market_id" in sig.parameters, (
        "register_entry must accept market_id directly -- no broker scan"
    )


def test_snapshot_all_stops_to_db_no_persistence() -> None:
    """When persistence=None, snapshot_all_stops_to_db is a silent no-op."""
    from datetime import UTC, datetime

    tracker = _make_tracker()  # persistence=None
    # Must not raise AttributeError.
    tracker.snapshot_all_stops_to_db(market_ids={}, prices={}, now=datetime.now(UTC))


def test_snapshot_all_stops_to_db_empty() -> None:
    """With persistence wired but no stops active, we never call the writer."""
    from datetime import UTC, datetime

    kelly = MagicMock()
    router = MagicMock()
    persistence = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        persistence=persistence,
    )
    tracker.snapshot_all_stops_to_db(market_ids={}, prices={}, now=datetime.now(UTC))
    persistence.persist_stop_snapshots.assert_not_called()


def test_snapshot_all_stops_to_db_writes_snapshot() -> None:
    """With an active stop, snapshot_all_stops_to_db calls persistence with
    event_type='snapshot' and forwards the caller-supplied market_ids + prices."""
    from datetime import UTC, datetime

    kelly = MagicMock()
    router = MagicMock()
    persistence = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        persistence=persistence,
    )
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = _make_state()
    now = datetime.now(UTC)
    tracker.snapshot_all_stops_to_db(
        market_ids={"SBER": "moex"},
        prices={"SBER": Decimal(105)},
        now=now,
    )
    persistence.persist_stop_snapshots.assert_called_once()
    kwargs = persistence.persist_stop_snapshots.call_args.kwargs
    assert kwargs["event_type"] == "snapshot"
    assert "SBER" in kwargs["states"]
    assert kwargs["market_ids"] == {"SBER": "moex"}
    assert kwargs["prices"] == {"SBER": Decimal(105)}
    assert kwargs["now"] == now


def test_register_entry_fires_entry_event_with_caller_market_id() -> None:
    """register_entry must use the caller-provided market_id -- no broker scan."""
    kelly = MagicMock()
    router = MagicMock()
    persistence = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        persistence=persistence,
    )
    tracker.register_entry(
        "SBER",
        Decimal(100),
        "momentum",
        _make_state(),
        market_id="moex",
    )
    persistence.persist_stop_snapshots.assert_called_once()
    kwargs = persistence.persist_stop_snapshots.call_args.kwargs
    assert kwargs["event_type"] == "entry"
    assert kwargs["market_ids"] == {"SBER": "moex"}
    assert "SBER" in kwargs["states"]
    assert kwargs["prices"] == {"SBER": Decimal(100)}
    # Crucially: broker_router.route was NOT called by register_entry (no O(N*M) scan).
    router.route.assert_not_called()


def test_register_entry_no_persistence_does_not_fail() -> None:
    """When persistence=None, register_entry must still register the position."""
    tracker = _make_tracker()  # persistence=None
    stop_state = _make_state()
    tracker.register_entry("SBER", Decimal(100), "momentum", stop_state, market_id="moex")
    assert tracker.has_stop("SBER")
    assert tracker._entry_prices["SBER"] == Decimal(100)
    assert tracker._entry_strategy["SBER"] == "momentum"


def test_check_stop_losses_trigger_fires_trigger_event() -> None:
    """When check_stop_losses triggers a SELL, a 'trigger' event is fired
    AFTER the _stop_loss_lock critical section closes."""
    kelly = MagicMock()
    router = MagicMock()
    broker = MagicMock()
    broker.get_positions.return_value = {"SBER": Decimal(10)}
    broker.submit_order.return_value = MagicMock()
    router.route.return_value = broker
    persistence = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        persistence=persistence,
    )
    state = _make_state()
    state.current_stop = Decimal(95)  # trigger when current_price <= 95
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state

    tracker.check_stop_losses(
        market_id="moex",
        symbol="SBER",
        current_price=Decimal(90),
    )

    # Expect EXACTLY one call with event_type='trigger'.
    calls = [
        c
        for c in persistence.persist_stop_snapshots.call_args_list
        if c.kwargs.get("event_type") == "trigger"
    ]
    assert len(calls) == 1
    assert "SBER" in calls[0].kwargs["states"]
    assert calls[0].kwargs["market_ids"] == {"SBER": "moex"}
    assert calls[0].kwargs["prices"] == {"SBER": Decimal(90)}
    # State deleted after trigger.
    assert "SBER" not in tracker._stop_states


def test_check_stop_losses_no_trigger_does_not_fire() -> None:
    """When check_stop_losses does NOT trigger (price above stop), no event."""
    kelly = MagicMock()
    router = MagicMock()
    persistence = MagicMock()
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        persistence=persistence,
    )
    state = _make_state()
    state.current_stop = Decimal(95)
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state

    # current_price=100 > current_stop=95 -- no trigger.
    tracker.check_stop_losses(market_id="moex", symbol="SBER", current_price=Decimal(100))

    # No 'trigger' event was fired.
    trigger_calls = [
        c
        for c in persistence.persist_stop_snapshots.call_args_list
        if c.kwargs.get("event_type") == "trigger"
    ]
    assert len(trigger_calls) == 0
    # Stop state is still present (no trigger happened).
    assert "SBER" in tracker._stop_states


def test_check_stop_losses_no_persistence_does_not_fail() -> None:
    """When persistence=None, check_stop_losses trigger branch must not crash."""
    kelly = MagicMock()
    router = MagicMock()
    broker = MagicMock()
    broker.get_positions.return_value = {"SBER": Decimal(10)}
    broker.submit_order.return_value = MagicMock()
    router.route.return_value = broker
    tracker = PositionTracker(
        kelly_sizer=kelly,
        broker_router=router,
        # persistence omitted -- defaults to None.
    )
    state = _make_state()
    state.current_stop = Decimal(95)
    with tracker._stop_loss_lock:
        tracker._stop_states["SBER"] = state

    # Must not raise AttributeError on self._persistence.persist_stop_snapshots.
    tracker.check_stop_losses(market_id="moex", symbol="SBER", current_price=Decimal(90))
    # Trigger still happened — state cleared.
    assert "SBER" not in tracker._stop_states
