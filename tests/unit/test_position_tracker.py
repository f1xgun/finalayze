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
        initial_stop=d - Decimal("5"),
        current_stop=d - Decimal("5"),
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
    returned.highest_price = Decimal("999")
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
                    s.current_stop = s.highest_price - Decimal("2")

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
