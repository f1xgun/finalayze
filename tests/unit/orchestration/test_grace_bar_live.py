"""S3.1 — Grace bar in the LIVE PositionTracker.check_stop_losses.

The backtest engine skips the stop-loss check on the bar immediately after
entry (``entry_bars[sym] + 1 == i``), with a 15 % catastrophic-drop override.
Live previously had no equivalent guard: a freshly-opened position could be
stopped out on the very next cycle just because the candle closed slightly
below the freshly-placed stop. This caused backtest/live divergence in
trade counts and drawdowns.

Contract for live (mirroring backtest):
  GBAR-LIVE-01: When ``_current_cycle_index == entry_cycle_index + 1``,
                ``check_stop_losses`` must NOT submit a SELL — unless...
  GBAR-LIVE-02: ...the current price is < entry_price * (1 - 0.15), in
                which case the catastrophic override fires the stop normally.
  GBAR-LIVE-03: On any cycle other than the first post-entry one
                (entry_cycle, entry_cycle + 2, entry_cycle + N>2), the
                regular stop check runs unchanged.
  GBAR-LIVE-04: If ``_current_cycle_index == 0`` (set_current_cycle never
                called — e.g. legacy tests, post-restart with stale state),
                grace must NOT kick in spuriously. The default path runs.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.execution.simulated_broker import StopLossState
from finalayze.orchestration.position_manager import PositionTracker

_ENTRY_PRICE = Decimal(100)
_ENTRY_CYCLE = 5
_QTY = Decimal(10)
_CURRENT_STOP = Decimal(95)
_BELOW_STOP_PRICE = Decimal(94)
_CATASTROPHIC_PRICE = Decimal(84)
_NON_CATASTROPHIC_PRICE = Decimal(86)
_FAR_BELOW_STOP = Decimal(80)


def _make_state(entry_cycle_index: int = _ENTRY_CYCLE) -> StopLossState:
    return StopLossState(
        initial_stop=_CURRENT_STOP,
        current_stop=_CURRENT_STOP,
        highest_price=_ENTRY_PRICE,
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=_ENTRY_PRICE,
        atr_value=Decimal("2.5"),
        entry_cycle_index=entry_cycle_index,
    )


def _make_tracker() -> tuple[PositionTracker, MagicMock]:
    kelly = MagicMock()
    router = MagicMock()
    broker = MagicMock()
    broker.get_positions.return_value = {"SBER": _QTY}
    broker.submit_order.return_value = MagicMock(success=True)
    router.route.return_value = broker
    return (
        PositionTracker(kelly_sizer=kelly, broker_router=router, alerter=MagicMock()),
        broker,
    )


# ─── GBAR-LIVE-01 ────────────────────────────────────────────────────────────
def test_grace_bar_skips_stop_on_first_post_entry_cycle() -> None:
    """Price below stop on entry_cycle+1 must NOT trigger a SELL."""
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=_ENTRY_CYCLE)
    tracker.set_current_cycle(_ENTRY_CYCLE + 1)

    tracker.check_stop_losses("moex", "SBER", _BELOW_STOP_PRICE)

    broker.submit_order.assert_not_called()
    assert "SBER" in tracker._stop_states  # state retained


# ─── GBAR-LIVE-02 ────────────────────────────────────────────────────────────
def test_grace_bar_catastrophic_drop_overrides_grace() -> None:
    """A drop of >= 15% on entry_cycle+1 still fires the stop."""
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=_ENTRY_CYCLE)
    tracker.set_current_cycle(_ENTRY_CYCLE + 1)

    tracker.check_stop_losses("moex", "SBER", _CATASTROPHIC_PRICE)

    broker.submit_order.assert_called_once()
    assert "SBER" not in tracker._stop_states  # cleared after trigger


def test_grace_bar_near_miss_drop_does_not_override() -> None:
    """A drop of ~14 % (just under 15 %) is still grace-protected."""
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=_ENTRY_CYCLE)
    tracker.set_current_cycle(_ENTRY_CYCLE + 1)

    tracker.check_stop_losses("moex", "SBER", _NON_CATASTROPHIC_PRICE)

    broker.submit_order.assert_not_called()
    assert "SBER" in tracker._stop_states


# ─── GBAR-LIVE-03 ────────────────────────────────────────────────────────────
def test_grace_bar_does_not_apply_two_cycles_post_entry() -> None:
    """Cycle entry+2 is past grace: a sub-stop price triggers the SELL."""
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=_ENTRY_CYCLE)
    tracker.set_current_cycle(_ENTRY_CYCLE + 2)

    tracker.check_stop_losses("moex", "SBER", _FAR_BELOW_STOP)

    broker.submit_order.assert_called_once()
    assert "SBER" not in tracker._stop_states


def test_grace_bar_does_not_apply_at_entry_cycle_itself() -> None:
    """Cycle == entry_cycle (e.g. same-cycle re-check) is NOT grace.

    Note: in normal flow this branch is unreachable because the entry is
    registered AFTER the cycle's stop check. The guard must still be safe
    if it ever fires: same-cycle check should run normally.
    """
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=_ENTRY_CYCLE)
    tracker.set_current_cycle(_ENTRY_CYCLE)  # same cycle

    tracker.check_stop_losses("moex", "SBER", _FAR_BELOW_STOP)

    broker.submit_order.assert_called_once()


# ─── GBAR-LIVE-04 ────────────────────────────────────────────────────────────
def test_grace_bar_inactive_when_cycle_index_unset() -> None:
    """Both indices defaulting to 0 must NOT spuriously trigger grace.

    A stale state from before set_current_cycle ran has entry_cycle_index=0
    AND tracker._current_cycle_index=0. The arithmetic `0 == 0+1` is False —
    grace is inactive. The regular stop check runs.
    """
    tracker, broker = _make_tracker()
    tracker._stop_states["SBER"] = _make_state(entry_cycle_index=0)
    # tracker._current_cycle_index defaults to 0 — set_current_cycle NOT called

    tracker.check_stop_losses("moex", "SBER", _FAR_BELOW_STOP)

    broker.submit_order.assert_called_once()
