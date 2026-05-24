"""Position tracking and stop-loss management (Phase 1.6).

Extracted from TradingLoop to isolate:
  - Stop-loss state tracking (trailing stops)
  - Entry price tracking for Kelly sizing
  - Entry/exit registration
  - Per-cycle exit guard (PARITY-04)

Thread safety: all methods use _stop_loss_lock for atomic state updates.
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.core.schemas import Candle
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.execution.simulated_broker import StopLossState
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.risk.kelly import RollingKelly

_log = structlog.get_logger(__name__)
_ZERO = Decimal(0)

# Retroactive stop ATR multipliers (matches signal_executor defaults).
_RETRO_ATR_MULT_US = Decimal("2.0")
_RETRO_ATR_MULT_MOEX = Decimal("2.5")
_RETRO_GRACE_FRACTION = Decimal("0.5")  # 0.5 ATR below current when already underwater
_RETRO_ACTIVATION_ATR = Decimal("1.0")
_RETRO_TRAIL_ATR = Decimal("1.5")


class PositionTracker:
    """Thread-safe position tracking with stop-loss and Kelly integration.

    Manages:
      - Trailing stop-loss states per symbol
      - Entry prices for Kelly P&L computation
      - Entry strategy ownership (for PresetApplicator)
      - Per-cycle exit set (PARITY-04 re-entry guard)
    """

    def __init__(
        self,
        kelly_sizer: RollingKelly,
        broker_router: BrokerRouter,
        alerter: TelegramAlerter | None = None,
        persistence: TradingPersistence | None = None,
    ) -> None:
        """Initialize position tracker.

        Args:
            kelly_sizer: RollingKelly instance for P&L updates
            broker_router: BrokerRouter for submitting SELL orders on stop-loss
            alerter: Optional TelegramAlerter for alerts
            persistence: Optional TradingPersistence for fire-and-forget
                stop_loss_events writes (STOP-03, D-06). None in TEST/DEBUG modes.
        """
        # Import here to avoid circular imports
        from finalayze.execution.simulated_broker import StopLossState  # noqa: PLC0415

        self._StopLossState = StopLossState

        self._kelly_sizer = kelly_sizer
        self._broker_router = broker_router
        self._alerter = alerter
        self._persistence = persistence

        # Stop-loss state: symbol -> StopLossState (trailing, thread-safe via lock)
        self._stop_states: dict[str, StopLossState] = {}
        self._stop_loss_lock = threading.Lock()

        # Entry price tracking for Kelly P&L computation
        self._entry_prices: dict[str, Decimal] = {}

        # Position ownership tracking: symbol -> strategy_name
        self._entry_strategy: dict[str, str] = {}

        # Per-cycle re-entry guard: symbols stopped out this cycle skip signal gen
        self._cycle_exited_symbols: set[str] = set()

        # ALRT-01 D-07: monotonic cycle index mirrored from TradingLoop._cycle_count
        # via set_current_cycle(). Read by check_stop_losses to compute hold_bars
        # without extending the 3-param public signature (revision B3).
        self._current_cycle_index: int = 0

    def check_stop_losses(
        self,
        market_id: str,
        symbol: str,
        current_price: Decimal,
    ) -> None:
        """Check trailing stop-loss state and trigger SELL if breached.

        Implements the same 5-step trailing logic as SimulatedBroker:
        1. Update high-water mark
        2. Check activation threshold
        3. Ratchet trail stop upward (never down)
        4. Check trigger condition
        5. Submit SELL and record in _cycle_exited_symbols (PARITY-04)

        The entire check-sell-remove is atomic under _stop_loss_lock to prevent
        double-sell from concurrent threads (CONC-01).

        On a trigger, a ``'trigger'`` event is fired to ``persist_stop_snapshots``
        AFTER the ``with _stop_loss_lock:`` block closes, so the fire-and-forget
        DB write never blocks the critical section (STOP-03, D-06).

        Args:
            market_id: Market identifier (e.g., "us", "moex")
            symbol: Instrument symbol
            current_price: Current market price
        """
        trigger_snapshot: StopLossState | None = None
        with self._stop_loss_lock:
            state = self._stop_states.get(symbol)
            if state is None:
                return

            # Step 1: Update high-water mark
            state.highest_price = max(state.highest_price, current_price)

            # Step 2: Check activation
            if not state.trail_activated:
                activation_threshold = state.entry_price + state.activation_atr * state.atr_value
                if state.highest_price >= activation_threshold:
                    state.trail_activated = True

            # Step 3: Ratchet trail stop (only moves up)
            if state.trail_activated:
                trail_stop = state.highest_price - state.trail_atr * state.atr_value
                state.current_stop = max(state.current_stop, trail_stop)

            # Step 4: Trigger check
            if current_price > state.current_stop:
                return

            # Step 5: Stop triggered
            _log.warning(
                "stop_triggered",
                symbol=symbol,
                price=float(current_price),
                stop=float(state.current_stop),
                trailing=state.trail_activated,
            )
            # Capture a copy of the triggering state so we can persist it
            # outside the critical section.
            from dataclasses import replace  # noqa: PLC0415

            trigger_snapshot = replace(state)
            broker = self._broker_router.route(market_id)

            # Import OrderRequest at call site to avoid circular imports
            from finalayze.execution.broker_base import OrderRequest  # noqa: PLC0415

            positions = broker.get_positions()
            qty = positions.get(symbol, _ZERO)
            if qty > _ZERO:
                order = OrderRequest(symbol=symbol, side="SELL", quantity=qty)
                try:
                    broker.submit_order(order)
                except Exception:
                    _log.exception("check_stop_losses: failed to submit stop-loss for %s", symbol)
                    return  # Don't clear stop state -- retry next cycle
                # Update Kelly with stop-loss exit
                self._update_kelly(symbol, current_price)
                # ALRT-01 (D-08, D-10): compute P&L + hold_bars and fire enriched
                # on_stop_loss_triggered AFTER successful broker submit, BEFORE
                # clearing the stop state. Wrapped so a Telegram outage NEVER
                # crashes the cycle.
                if self._alerter is not None:
                    pnl_amount = (current_price - state.entry_price) * qty
                    pnl_pct = (
                        float((current_price - state.entry_price) / state.entry_price)
                        if state.entry_price > _ZERO
                        else None
                    )
                    # hold_bars: cycle-now minus cycle-at-entry. None when the
                    # state pre-dates Phase 57 (entry_cycle_index defaulted to 0
                    # on a stale row, e.g. post-restart) per Pitfall 5.
                    hold_bars = (
                        self._current_cycle_index - state.entry_cycle_index
                        if state.entry_cycle_index > 0
                        else None
                    )
                    currency = "RUB" if market_id.startswith(("moex", "ru_")) else "USD"
                    try:
                        self._alerter.on_stop_loss_triggered(
                            symbol=symbol,
                            entry_price=state.entry_price,
                            stop_price=state.current_stop,
                            current_price=current_price,
                            pnl_amount=pnl_amount,
                            pnl_pct=pnl_pct,
                            hold_bars=hold_bars,
                            currency=currency,
                        )
                    except Exception:
                        _log.exception("stop_alert_fire_failed", symbol=symbol)
            # Clear stop state after successful trigger (or zero position)
            del self._stop_states[symbol]
            self._entry_strategy.pop(symbol, None)
            self._cycle_exited_symbols.add(symbol)  # PARITY-04

        # Fire 'trigger' event to stop_loss_events (D-06) — OUTSIDE the lock.
        # Writer has its own synchronization; keeping it off the critical
        # section lets other trades proceed immediately.
        if trigger_snapshot is not None and self._persistence is not None:
            from datetime import UTC  # noqa: PLC0415
            from datetime import datetime as _dt  # noqa: PLC0415

            self._persistence.persist_stop_snapshots(
                states={symbol: trigger_snapshot},
                market_ids={symbol: market_id},
                prices={symbol: current_price},
                now=_dt.now(UTC),
                event_type="trigger",
            )

    def _update_kelly(self, symbol: str, fill_price: Decimal) -> None:
        """Compute P&L from entry price and feed a TradeRecord to RollingKelly.

        Args:
            symbol: Instrument symbol
            fill_price: Exit fill price
        """
        from finalayze.risk.kelly import TradeRecord  # noqa: PLC0415

        entry = self._entry_prices.pop(symbol, None)
        if entry is None or entry <= _ZERO:
            return
        pnl = fill_price - entry
        pnl_pct = pnl / entry
        self._kelly_sizer.update(TradeRecord(pnl=pnl, pnl_pct=pnl_pct))

    def register_entry(
        self,
        symbol: str,
        price: Decimal,
        strategy: str,
        stop_state: StopLossState,
        market_id: str,
    ) -> None:
        """Register a new position entry and fire an 'entry' event to persistence.

        Called after a BUY fill to track entry price, strategy ownership, and
        stop-loss state. When ``self._persistence`` is wired, also fires an
        ``event_type='entry'`` row into ``stop_loss_events`` (D-06).

        Args:
            symbol: Instrument symbol
            price: Entry fill price
            strategy: Strategy name that opened the position
            stop_state: StopLossState object for this position
            market_id: Market id ("us" | "moex") -- caller-supplied so we avoid
                an O(markets x positions) broker scan on the fill critical path
                (I-03 resolution, option A).
        """
        self._entry_prices[symbol] = price
        self._entry_strategy[symbol] = strategy
        # ALRT-01 D-07: stamp the current monotonic cycle index onto the state
        # so check_stop_losses can compute hold_bars on trigger.
        stop_state.entry_cycle_index = self._current_cycle_index
        with self._stop_loss_lock:
            self._stop_states[symbol] = stop_state
        # Fire 'entry' event to stop_loss_events (D-06) -- no broker scan.
        if self._persistence is not None:
            from datetime import UTC  # noqa: PLC0415
            from datetime import datetime as _dt  # noqa: PLC0415

            self._persistence.persist_stop_snapshots(
                states={symbol: stop_state},
                market_ids={symbol: market_id},
                prices={symbol: price},
                now=_dt.now(UTC),
                event_type="entry",
            )

    def maybe_register_retroactive_stop(
        self,
        symbol: str,
        candles: list[Candle],
        market_id: str,
    ) -> bool:
        """Register an ATR-based stop for an orphaned open position.

        Used after container restart when the broker reports an open position
        but the in-memory ``_stop_states`` is empty (no DB snapshot replayed).
        Caller is responsible for confirming the broker holds the position
        before invoking; this method only checks PositionTracker-internal state.

        Args:
            symbol: Instrument symbol.
            candles: Recent candle history; ATR is computed from this.
            market_id: ``"us"`` or ``"moex"`` — selects the ATR multiplier.

        Returns:
            True when a new stop was registered, False if skipped (already has
            a stop, no candles, or ATR could not be computed).
        """
        from finalayze.execution.simulated_broker import StopLossState  # noqa: PLC0415
        from finalayze.risk.stop_loss import compute_atr_stop_loss  # noqa: PLC0415

        if not candles or self.has_stop(symbol):
            return False

        mult = _RETRO_ATR_MULT_MOEX if market_id == "moex" else _RETRO_ATR_MULT_US
        cur = Decimal(str(candles[-1].close))
        entry = self._entry_prices.get(symbol, cur)
        natural_stop = compute_atr_stop_loss(entry, candles, atr_multiplier=mult)
        if natural_stop is None or mult <= _ZERO:
            return False

        atr_val = (entry - natural_stop) / mult
        if cur >= natural_stop:
            stop_price = natural_stop
            trail_activated = False
            highest = max(entry, cur)
        else:
            # Already underwater — grace stop sits _RETRO_GRACE_FRACTION ATR below current
            stop_price = max(cur - _RETRO_GRACE_FRACTION * atr_val, _ZERO)
            trail_activated = True
            highest = cur

        strategy = self._entry_strategy.get(symbol, "retroactive")
        stop_state = StopLossState(
            initial_stop=stop_price,
            current_stop=stop_price,
            highest_price=highest,
            trail_activated=trail_activated,
            activation_atr=_RETRO_ACTIVATION_ATR,
            trail_atr=_RETRO_TRAIL_ATR,
            entry_price=entry,
            atr_value=atr_val,
        )
        self.register_entry(symbol, entry, strategy, stop_state, market_id=market_id)
        _log.warning(
            "stop_retroactive_set",
            symbol=symbol,
            stop_price=float(stop_price),
            entry_price=float(entry),
            trail_activated=trail_activated,
            market=market_id,
        )
        return True

    def restore_stops(self, states: dict[str, tuple[str, StopLossState]]) -> None:
        """Re-hydrate stop-loss state from a DB snapshot after container restart.

        Only writes to symbols that are NOT already tracked (i.e., a fresh BUY
        fill that happened between restart and this call takes precedence).

        Args:
            states: symbol -> (market_id, StopLossState) from load_stop_snapshots.
        """
        if not states:
            return
        restored: list[str] = []
        with self._stop_loss_lock:
            for symbol, (_, state) in states.items():
                if symbol not in self._stop_states:
                    self._stop_states[symbol] = state
                    self._entry_prices[symbol] = state.entry_price
                    restored.append(symbol)
        if restored:
            _log.info("stop_states_restored_from_db", count=len(restored), symbols=restored)

    def register_exit(self, symbol: str) -> None:
        """Register a position exit (SELL fill).

        Called after a SELL fill to clear stop-loss and entry tracking.

        Args:
            symbol: Instrument symbol
        """
        with self._stop_loss_lock:
            self._stop_states.pop(symbol, None)
        self._entry_strategy.pop(symbol, None)

    def has_stop(self, symbol: str) -> bool:
        """Check if a symbol has an active stop-loss.

        Args:
            symbol: Instrument symbol

        Returns:
            True if stop-loss is active for the symbol
        """
        with self._stop_loss_lock:
            return symbol in self._stop_states

    def reset_cycle_exits(self) -> None:
        """Reset per-cycle exit set at start of each strategy cycle.

        Called at the beginning of _strategy_cycle() to clear the previous
        cycle's exited symbols (PARITY-04).
        """
        self._cycle_exited_symbols = set()

    def set_current_cycle(self, cycle_index: int) -> None:
        """Mirror TradingLoop._cycle_count onto the tracker (ALRT-01 D-07).

        Called from the top of ``TradingLoop._strategy_cycle_impl`` so
        ``check_stop_losses`` can compute ``hold_bars`` on trigger without an
        extension to its 3-param public signature (revision B3). Stamped onto
        ``StopLossState.entry_cycle_index`` at ``register_entry`` time.
        """
        self._current_cycle_index = cycle_index

    @property
    def exited_symbols(self) -> set[str]:
        """Return symbols that exited this cycle (via stop-loss or manual SELL).

        Used for PARITY-04 skip logic: symbols in this set skip signal
        generation for the current cycle.

        Returns:
            Set of symbols that exited this cycle
        """
        return self._cycle_exited_symbols

    def get_entry_strategies(self) -> dict[str, str]:
        """Return a snapshot of {symbol: strategy_name} for currently open positions.

        Used by PresetApplicator to check position ownership before disabling a
        strategy via auto-apply. Returns a copy so callers cannot mutate internal state.

        Returns:
            Dictionary mapping symbol to strategy name
        """
        return dict(self._entry_strategy)

    def get_stop_loss_price(self, symbol: str) -> Decimal | None:
        """Get current stop-loss price for a symbol.

        Returns the current_stop from the StopLossState if it exists,
        otherwise None. Thread-safe.

        Args:
            symbol: Instrument symbol

        Returns:
            Current stop-loss price or None if no stop-loss is active
        """
        with self._stop_loss_lock:
            state = self._stop_states.get(symbol)
            return state.current_stop if state is not None else None

    def get_stop_state(self, symbol: str) -> StopLossState | None:
        """Return a read-consistent snapshot of the full stop-loss state.

        Returns a COPY (via ``dataclasses.replace``) so callers cannot mutate
        internal state, and so concurrent ratchet in check_stop_losses does
        not split the caller's read. Thread-safe via ``_stop_loss_lock``.

        Args:
            symbol: Instrument symbol

        Returns:
            Copy of StopLossState or None if no stop-loss is active
        """
        from dataclasses import replace  # noqa: PLC0415

        with self._stop_loss_lock:
            state = self._stop_states.get(symbol)
            return replace(state) if state is not None else None

    def snapshot_all_stops(self) -> dict[str, StopLossState]:
        """Return copies of every active stop-loss state.

        Used by TradingLoop._strategy_cycle_impl to emit a per-cycle snapshot.
        Returns copies under lock so the caller can hold them off-lock without
        risking mid-ratchet inconsistency (Pitfall 1 of 54-RESEARCH).
        """
        from dataclasses import replace  # noqa: PLC0415

        with self._stop_loss_lock:
            return {sym: replace(st) for sym, st in self._stop_states.items()}

    def snapshot_all_stops_to_db(
        self,
        market_ids: dict[str, str],
        prices: dict[str, Decimal],
        now: datetime,
    ) -> None:
        """Fire-and-forget snapshot of every active stop to ``stop_loss_events``.

        Called once per strategy cycle by
        ``TradingLoop._strategy_cycle_impl()``. No-op when ``self._persistence``
        is None (TEST/DEBUG mode) or when no stops are active.

        Args:
            market_ids: ``{symbol: market_id}`` map for each active position.
            prices: ``{symbol: current_price}`` map for chart overlay; optional
                per symbol (writer forwards ``None`` when absent).
            now: Snapshot timestamp (UTC-aware ``datetime``).
        """
        if self._persistence is None:
            return
        states = self.snapshot_all_stops()
        if not states:
            return
        self._persistence.persist_stop_snapshots(
            states=states,
            market_ids=market_ids,
            prices=prices,
            now=now,
            event_type="snapshot",
        )
