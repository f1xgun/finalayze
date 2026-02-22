"""Abstract broker interface (Layer 5).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from finalayze.core.schemas import Candle, PortfolioState  # noqa: TC001


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
    def submit_order(self, order: OrderRequest, fill_candle: Candle) -> OrderResult:
        """Submit an order for execution against the given candle."""
        ...

    @abstractmethod
    def get_portfolio(self) -> PortfolioState:
        """Return the current portfolio state."""
        ...
