"""Broker router — dispatches orders to the correct broker by market ID (Layer 5).

Routes orders based on the order's market_id:
  - "us"   -> AlpacaBroker (or any BrokerBase registered for "us")
  - "moex" -> TinkoffBroker (or any BrokerBase registered for "moex")

Raises BrokerError if no broker is registered for the requested market_id.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.core.exceptions import BrokerError

if TYPE_CHECKING:
    from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult


class BrokerRouter:
    """Routes orders to the appropriate broker based on market_id.

    Example:
        router = BrokerRouter({
            "us": alpaca_broker,
            "moex": tinkoff_broker,
        })
        broker = router.route("us")
        result = router.submit(order, market_id="moex")
    """

    def __init__(self, brokers: dict[str, BrokerBase]) -> None:
        self._brokers = dict(brokers)

    def route(self, market_id: str) -> BrokerBase:
        """Return the broker registered for market_id.

        Raises:
            BrokerError: If no broker is registered for the given market_id.
        """
        broker = self._brokers.get(market_id)
        if broker is None:
            registered = ", ".join(sorted(self._brokers)) or "(none)"
            msg = f"No broker registered for market '{market_id}'. Registered markets: {registered}"
            raise BrokerError(msg)
        return broker

    def submit(
        self,
        order: OrderRequest,
        market_id: str,
        fill_candle: object = None,
    ) -> OrderResult:
        """Route and submit an order in one step.

        Args:
            order: The order to submit.
            market_id: The market this order belongs to.
            fill_candle: Optional candle for simulated brokers (None for live).

        Returns:
            OrderResult from the routed broker.
        """
        broker = self.route(market_id)
        return broker.submit_order(order, fill_candle=fill_candle)  # type: ignore[arg-type]

    @property
    def registered_markets(self) -> list[str]:
        """Return a sorted list of registered market IDs."""
        return sorted(self._brokers)
