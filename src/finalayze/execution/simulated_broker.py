"""Simulated broker for backtesting (Layer 5).

Fills orders at candle open prices with no slippage or commission.
Supports stop-loss orders (fixed and trailing) that trigger when candle low
breaches the stop price.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import Candle, PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult


@dataclass
class StopLossState:
    """Holds trailing stop-loss state for a position.

    When ``trail_activated`` is False, the stop fires only at ``initial_stop``.
    Once the price reaches ``entry_price + activation_atr * atr_value``, trailing
    begins and ``current_stop`` ratchets upward as new highs are made.
    """

    initial_stop: Decimal  # entry - N * ATR
    current_stop: Decimal  # may trail upward
    highest_price: Decimal  # high-water mark since entry
    trail_activated: bool  # True once price reaches activation threshold
    activation_atr: Decimal  # ATR multiplier to activate trailing (default 1.0)
    trail_atr: Decimal  # ATR multiplier for trailing distance (default 1.5)
    entry_price: Decimal  # entry price (to compute activation threshold)
    atr_value: Decimal  # ATR at time of entry


class SimulatedBroker(BrokerBase):
    """In-memory simulated broker for backtesting."""

    def __init__(self, initial_cash: Decimal) -> None:
        self._cash: Decimal = initial_cash
        self._positions: dict[str, Decimal] = {}
        self._stop_states: dict[str, StopLossState] = {}
        self._last_prices: dict[str, Decimal] = {}
        self._current_timestamp: datetime | None = None

    def submit_order(self, order: OrderRequest, fill_candle: Candle | None = None) -> OrderResult:
        """Fill an order at the candle's open price.

        BUY: deducts cash, adds to position.
        SELL: adds proceeds to cash, reduces/removes position.
        """
        if fill_candle is None:
            msg = "SimulatedBroker requires fill_candle to determine the fill price"
            raise ValueError(msg)
        fill_price = fill_candle.open

        if order.side == "BUY":
            return self._execute_buy(order, fill_price, fill_candle)
        if order.side == "SELL":
            return self._execute_sell(order, fill_price, fill_candle)

        return OrderResult(
            filled=False,
            symbol=order.symbol,
            side=order.side,
            reason=f"Unknown side: {order.side}",
        )

    def set_timestamp(self, ts: datetime) -> None:
        """Set the current simulation timestamp (used in portfolio snapshots)."""
        self._current_timestamp = ts

    def set_stop_loss(self, symbol: str, price: Decimal) -> None:
        """Set a fixed (non-trailing) stop-loss price for a symbol.

        Backward-compatible: creates a StopLossState that never activates trailing.
        """
        self._stop_states[symbol] = StopLossState(
            initial_stop=price,
            current_stop=price,
            highest_price=price,
            trail_activated=False,
            activation_atr=Decimal(999999),  # effectively never activates
            trail_atr=Decimal(0),
            entry_price=price,
            atr_value=Decimal(1),  # non-zero so threshold = price + 999999 ≈ ∞
        )

    def set_trailing_stop(
        self,
        symbol: str,
        entry_price: Decimal,
        initial_stop: Decimal,
        atr_value: Decimal,
        activation_atr: Decimal = Decimal("1.0"),
        trail_atr: Decimal = Decimal("1.5"),
    ) -> None:
        """Set a trailing stop-loss for a symbol.

        Args:
            symbol: Ticker symbol.
            entry_price: Price at which the position was entered.
            initial_stop: Initial stop price (entry - N * ATR).
            atr_value: ATR value at time of entry.
            activation_atr: ATR multiplier for trail activation threshold.
            trail_atr: ATR multiplier for trailing distance.
        """
        self._stop_states[symbol] = StopLossState(
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_price=entry_price,
            trail_activated=False,
            activation_atr=activation_atr,
            trail_atr=trail_atr,
            entry_price=entry_price,
            atr_value=atr_value,
        )

    def check_stop_losses(self, candle: Candle) -> list[OrderResult]:
        """Check if any stop losses triggered on this candle.

        For trailing stops:
          1. Update highest_price = max(highest_price, candle.high)
          2. Activate trailing when highest_price >= entry_price + activation_atr * atr_value
          3. Once activated: trail_stop = highest_price - trail_atr * atr_value
          4. current_stop = max(current_stop, trail_stop) -- stop only moves up
          5. Trigger if candle.low <= current_stop

        For fixed stops (set via set_stop_loss), triggers at current_stop.
        """
        results: list[OrderResult] = []
        symbol = candle.symbol

        if symbol not in self._stop_states:
            return results

        state = self._stop_states[symbol]

        # Step 1: Update high-water mark
        state.highest_price = max(state.highest_price, candle.high)

        # Step 2: Check activation
        if not state.trail_activated:
            activation_threshold = state.entry_price + state.activation_atr * state.atr_value
            if state.highest_price >= activation_threshold:
                state.trail_activated = True

        # Step 3 & 4: Compute and ratchet trail stop
        if state.trail_activated:
            trail_stop = state.highest_price - state.trail_atr * state.atr_value
            state.current_stop = max(state.current_stop, trail_stop)

        # Step 5: Check trigger
        if candle.low <= state.current_stop and symbol in self._positions:
            qty = self._positions[symbol]
            # Gap fill for long positions: if the candle opens below the stop price
            # (a gap down), fill at candle.open — the market never traded at the stop.
            # Use min() because for a long exit (SELL), a lower open is a worse fill.
            fill_price = min(state.current_stop, candle.open)
            proceeds = fill_price * qty
            self._cash += proceeds
            del self._positions[symbol]
            del self._stop_states[symbol]
            self._last_prices[symbol] = candle.close

            results.append(
                OrderResult(
                    filled=True,
                    fill_price=fill_price,
                    symbol=symbol,
                    side="SELL",
                    quantity=qty,
                )
            )

        return results

    def update_prices(self, candle: Candle) -> None:
        """Update last known price for a symbol from a candle's close."""
        self._last_prices[candle.symbol] = candle.close

    def has_position(self, symbol: str) -> bool:
        """Return True if the broker holds a non-zero position in symbol."""
        return symbol in self._positions and self._positions[symbol] > 0

    def get_positions(self) -> dict[str, Decimal]:
        """Return a copy of the current open positions keyed by symbol."""
        return dict(self._positions)

    def cancel_order(self, order_id: str) -> None:
        """No-op for simulated broker -- stop-loss keyed by symbol, not order ID."""
        self._stop_states.pop(order_id, None)

    def get_portfolio(self) -> PortfolioState:
        """Return current portfolio state with computed equity."""
        position_value = sum(
            qty * self._last_prices.get(symbol, Decimal(0))
            for symbol, qty in self._positions.items()
        )
        equity = self._cash + position_value
        timestamp = (
            self._current_timestamp if self._current_timestamp is not None else datetime.now(tz=UTC)
        )

        return PortfolioState(
            cash=self._cash,
            positions=dict(self._positions),
            equity=equity,
            timestamp=timestamp,
        )

    def _execute_buy(
        self, order: OrderRequest, fill_price: Decimal, fill_candle: Candle
    ) -> OrderResult:
        """Execute a buy order."""
        cost = fill_price * order.quantity

        if self._cash < cost:
            return OrderResult(
                filled=False,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                reason=f"Insufficient cash: need {cost}, have {self._cash}",
            )

        self._cash -= cost
        current = self._positions.get(order.symbol, Decimal(0))
        self._positions[order.symbol] = current + order.quantity
        self._last_prices[order.symbol] = fill_candle.close

        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
        )

    def _execute_sell(
        self, order: OrderRequest, fill_price: Decimal, fill_candle: Candle
    ) -> OrderResult:
        """Execute a sell order. Sells min(requested, held)."""
        held = self._positions.get(order.symbol, Decimal(0))
        actual_qty = min(order.quantity, held)

        if actual_qty <= 0:
            return OrderResult(
                filled=False,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal(0),
                reason=f"No position in {order.symbol}",
            )

        proceeds = fill_price * actual_qty
        self._cash += proceeds

        remaining = held - actual_qty
        if remaining > 0:
            self._positions[order.symbol] = remaining
        else:
            self._positions.pop(order.symbol, None)
            self._stop_states.pop(order.symbol, None)

        self._last_prices[order.symbol] = fill_candle.close

        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=actual_qty,
        )
