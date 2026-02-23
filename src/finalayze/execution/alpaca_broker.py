"""Alpaca broker for US paper/live trading (Layer 5).

Uses alpaca-py SDK for order submission and portfolio management.
Paper trading uses Alpaca's paper endpoint; live uses production.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from finalayze.core.exceptions import BrokerError, InsufficientFundsError
from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_INSUFFICIENT_FUNDS_KEYWORDS = ("insufficient", "buying power", "not enough")


class AlpacaBroker(BrokerBase):
    """Alpaca paper/live broker for US market trading.

    Submits market orders via alpaca-py TradingClient.
    Raises BrokerError on API failures and InsufficientFundsError
    when buying power is too low.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
    ) -> None:
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

    def submit_order(
        self,
        order: OrderRequest,
        fill_candle: Candle | None = None,  # noqa: ARG002 -- ignored for live broker
    ) -> OrderResult:
        """Submit a market order to Alpaca. fill_candle is not used."""
        side = OrderSide.BUY if order.side == "BUY" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=float(order.quantity),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            result = self._client.submit_order(order_data=request)
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(kw in exc_str for kw in _INSUFFICIENT_FUNDS_KEYWORDS):
                msg = f"Insufficient funds for {order.side} {order.quantity} {order.symbol}"
                raise InsufficientFundsError(msg) from exc
            msg = f"Alpaca order failed: {exc}"
            raise BrokerError(msg) from exc

        fill_price = (
            Decimal(str(result.filled_avg_price))  # type: ignore[union-attr]
            if result.filled_avg_price  # type: ignore[union-attr]
            else None
        )
        return OrderResult(
            filled=fill_price is not None,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(str(result.filled_qty or order.quantity)),  # type: ignore[union-attr]
        )

    def get_portfolio(self) -> PortfolioState:
        """Return current portfolio state from Alpaca."""
        try:
            account = self._client.get_account()
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca portfolio fetch failed: {exc}"
            raise BrokerError(msg) from exc

        cash = Decimal(str(account.cash))  # type: ignore[union-attr]
        pos_map: dict[str, Decimal] = {}
        position_value = Decimal(0)
        for pos in positions:
            qty = Decimal(str(pos.qty))  # type: ignore[union-attr]
            pos_map[pos.symbol] = qty  # type: ignore[union-attr]
            position_value += Decimal(str(pos.market_value))  # type: ignore[union-attr]

        return PortfolioState(
            cash=cash,
            positions=pos_map,
            equity=cash + position_value,
            timestamp=datetime.now(tz=UTC),
        )

    def has_position(self, symbol: str) -> bool:
        """Return True if Alpaca account holds a non-zero position in symbol."""
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca positions fetch failed: {exc}"
            raise BrokerError(msg) from exc
        return any(p.symbol == symbol and Decimal(str(p.qty)) > 0 for p in positions)  # type: ignore[union-attr]

    def get_positions(self) -> dict[str, Decimal]:
        """Return a copy of current Alpaca positions keyed by symbol."""
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca positions fetch failed: {exc}"
            raise BrokerError(msg) from exc
        return {p.symbol: Decimal(str(p.qty)) for p in positions}  # type: ignore[union-attr]

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending Alpaca order by ID."""
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as exc:
            msg = f"Alpaca cancel_order failed for {order_id}: {exc}"
            raise BrokerError(msg) from exc
