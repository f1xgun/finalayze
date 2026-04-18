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
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.execution.simulated_broker import StopLossState
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.risk.kelly import RollingKelly

_log = structlog.get_logger(__name__)
_ZERO = Decimal(0)


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

        Args:
            market_id: Market identifier (e.g., "us", "moex")
            symbol: Instrument symbol
            current_price: Current market price
        """
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
            # Clear stop state after successful trigger (or zero position)
            del self._stop_states[symbol]
            self._entry_strategy.pop(symbol, None)
            self._cycle_exited_symbols.add(symbol)  # PARITY-04

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
    ) -> None:
        """Register a new position entry.

        Called after a BUY fill to track entry price, strategy ownership, and
        stop-loss state.

        Args:
            symbol: Instrument symbol
            price: Entry fill price
            strategy: Strategy name that opened the position
            stop_state: StopLossState object for this position
        """
        self._entry_prices[symbol] = price
        self._entry_strategy[symbol] = strategy
        with self._stop_loss_lock:
            self._stop_states[symbol] = stop_state

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
