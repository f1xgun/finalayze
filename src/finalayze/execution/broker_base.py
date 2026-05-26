"""Abstract broker interface (Layer 5).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from finalayze.core.schemas import Candle, PortfolioState  # noqa: TC001  # needed for abstract sig


def _generate_client_order_id() -> str:
    """Generate a Finalayze-prefixed unique client_order_id.

    S1.1: every OrderRequest must carry a stable id so that
    RetryPolicy.execute() can re-submit on transient gRPC / HTTP errors
    without producing duplicate orders at the broker. The Tinkoff
    post_order(order_id=...) API is idempotent on this value; Alpaca's
    MarketOrderRequest accepts it as `client_order_id`.

    Prefix `fnz-` makes audit logs / broker UIs unambiguous; uuid4 ensures
    uniqueness across processes and restarts.
    """
    return f"fnz-{uuid.uuid4().hex[:24]}"


@dataclass(frozen=True)
class OrderRequest:
    """A request to buy or sell a given quantity of a symbol."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: Decimal
    # S1.1: idempotency key forwarded to the broker. Auto-generated unless
    # the caller supplies an explicit value (e.g. when persisting/replaying).
    client_order_id: str = field(default_factory=_generate_client_order_id)


@dataclass(frozen=True)
class OrderResult:
    """Result of an order submission."""

    filled: bool
    fill_price: Decimal | None = None
    symbol: str = ""
    side: Literal["BUY", "SELL"] | str = ""
    quantity: Decimal = Decimal(0)
    reason: str = ""
    order_id: str = ""


class BrokerBase(ABC):
    """Abstract base class for all broker implementations."""

    @abstractmethod
    def submit_order(self, order: OrderRequest, fill_candle: Candle | None = None) -> OrderResult:
        """Submit an order for execution.

        Args:
            order: The order to execute.
            fill_candle: For simulated brokers -- fill price is taken from candle open.
                         Live brokers ignore this parameter (pass None).
        """
        ...

    @abstractmethod
    def get_portfolio(self) -> PortfolioState:
        """Return the current portfolio state."""
        ...

    @abstractmethod
    def has_position(self, symbol: str) -> bool:
        """Return True if the broker holds a non-zero position in symbol."""
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, Decimal]:
        """Return a copy of the current open positions keyed by symbol."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order by its ID.

        Args:
            order_id: The broker-assigned order identifier to cancel.
        """
        ...
