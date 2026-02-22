"""Simulated broker for backtesting (Layer 5).

Fills orders at candle open prices with no slippage or commission.
Supports stop-loss orders that trigger when candle low breaches the stop price.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import Candle, PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult


class SimulatedBroker(BrokerBase):
    """In-memory simulated broker for backtesting."""

    def __init__(self, initial_cash: Decimal) -> None:
        self._cash: Decimal = initial_cash
        self._positions: dict[str, Decimal] = {}
        self._stop_losses: dict[str, Decimal] = {}
        self._last_prices: dict[str, Decimal] = {}
        self._current_timestamp: datetime | None = None

    def submit_order(self, order: OrderRequest, fill_candle: Candle) -> OrderResult:
        """Fill an order at the candle's open price.

        BUY: deducts cash, adds to position.
        SELL: adds proceeds to cash, reduces/removes position.
        """
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
        """Set a stop-loss price for a symbol."""
        self._stop_losses[symbol] = price

    def check_stop_losses(self, candle: Candle) -> list[OrderResult]:
        """Check if any stop losses triggered on this candle.

        A stop loss triggers when candle.low <= stop_price.
        Fills at the stop price and closes the full position.
        """
        results: list[OrderResult] = []
        symbol = candle.symbol

        if symbol not in self._stop_losses:
            return results

        stop_price = self._stop_losses[symbol]

        if candle.low <= stop_price and symbol in self._positions:
            qty = self._positions[symbol]
            proceeds = stop_price * qty
            self._cash += proceeds
            del self._positions[symbol]
            del self._stop_losses[symbol]
            self._last_prices[symbol] = candle.close

            results.append(
                OrderResult(
                    filled=True,
                    fill_price=stop_price,
                    symbol=symbol,
                    side="SELL",
                    quantity=qty,
                )
            )

        return results

    def update_prices(self, candle: Candle) -> None:
        """Update last known price for a symbol from a candle's close."""
        self._last_prices[candle.symbol] = candle.close

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

        self._last_prices[order.symbol] = fill_candle.close

        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=actual_qty,
        )
