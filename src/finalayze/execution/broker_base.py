"""Abstract broker interface (Layer 5).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from finalayze.core.schemas import Candle, PortfolioState  # noqa: TC001  # needed for abstract sig


@dataclass(frozen=True)
class OrderRequest:
    """A request to buy or sell a given quantity of a symbol."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: Decimal


@dataclass(frozen=True)
class OrderResult:
    """Result of an order submission."""

    filled: bool
    fill_price: Decimal | None = None
    symbol: str = ""
    side: Literal["BUY", "SELL"] | str = ""
    quantity: Decimal = Decimal(0)
    reason: str = ""


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
