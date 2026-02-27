"""Transaction cost model for backtesting.

Accounts for commission, spread, and slippage when calculating trade costs.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_BPS_DIVISOR = Decimal(10000)


@dataclass(frozen=True)
class TransactionCosts:
    """Immutable transaction cost parameters.

    Attributes:
        commission_per_share: Per-share commission (default $0.005).
        min_commission: Minimum commission per trade (default $1.00).
        spread_bps: Half-spread in basis points (default 5 bps).
        slippage_bps: Slippage in basis points (default 3 bps).
    """

    commission_per_share: Decimal = Decimal("0.005")
    min_commission: Decimal = Decimal("1.00")
    spread_bps: Decimal = Decimal(5)
    slippage_bps: Decimal = Decimal(3)

    def total_cost(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Compute total transaction cost for a single trade.

        Args:
            price: Fill price per share.
            quantity: Number of shares traded.

        Returns:
            Total cost = commission + (spread + slippage) * quantity.
        """
        commission = max(self.min_commission, self.commission_per_share * quantity)
        spread = price * self.spread_bps / _BPS_DIVISOR
        slippage = price * self.slippage_bps / _BPS_DIVISOR
        return commission + (spread + slippage) * quantity
