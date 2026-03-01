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
        commission_per_share: Per-share commission (default $0.005 for US markets).
        min_commission: Minimum commission per trade (default $1.00).
        spread_bps: Half-spread in basis points (default 5 bps).
        slippage_bps: Slippage in basis points (default 3 bps).
    """

    commission_per_share: Decimal = Decimal("0.005")
    min_commission: Decimal = Decimal("1.00")
    spread_bps: Decimal = Decimal(5)
    slippage_bps: Decimal = Decimal(3)
    commission_rate: Decimal = Decimal(0)

    def total_cost(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Compute total transaction cost for a single trade.

        Args:
            price: Fill price per share.
            quantity: Number of shares traded.

        Returns:
            Total cost = commission + (spread + slippage) * quantity.
        """
        if self.commission_rate > 0:
            commission = max(self.min_commission, price * quantity * self.commission_rate)
        else:
            commission = max(self.min_commission, self.commission_per_share * quantity)
        spread = price * self.spread_bps / _BPS_DIVISOR
        slippage = price * self.slippage_bps / _BPS_DIVISOR
        return commission + (spread + slippage) * quantity


# ── Market-specific cost presets ─────────────────────────────────────────────

# US equities: $0.005/share (Alpaca-like), 5 bps half-spread, 3 bps slippage
US_COSTS = TransactionCosts(
    commission_per_share=Decimal("0.005"),
    min_commission=Decimal("1.00"),
    spread_bps=Decimal(5),
    slippage_bps=Decimal(3),
)

# MOEX (Tinkoff Invest): ~0.03% (~3 bps) commission as fraction of trade value,
# modelled as commission_per_share=0 with a higher spread to capture the percentage cost.
# MOEX typical costs: 0.03% commission + 10 bps spread + 7 bps slippage.
# commission_per_share=0.003 * price is approximated by setting spread_bps appropriately.
# We use a per-share commission that is a small fixed amount and rely on spread/slippage
# to capture the percentage-based MOEX fee structure.
MOEX_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),  # Not used for MOEX
    commission_rate=Decimal("0.0003"),  # 0.03% of trade value (Tinkoff Invest standard)
    min_commission=Decimal("0.10"),  # Very low min (ruble markets have small ticks)
    spread_bps=Decimal(10),  # Wider spreads on MOEX
    slippage_bps=Decimal(7),  # Higher slippage on less liquid MOEX
)
