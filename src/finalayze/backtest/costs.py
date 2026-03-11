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

# MOEX OFZ bonds (Tinkoff Invest Trader tariff):
# - Commission: 0.05% of trade value
# - Spread: 5 bps (on-the-run OFZ benchmarks, normal conditions)
# - Slippage: 3 bps (OFZ are relatively liquid on MOEX)
MOEX_BOND_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal("0.0005"),  # 0.05% of trade value
    min_commission=Decimal("0.01"),
    spread_bps=Decimal(5),
    slippage_bps=Decimal(3),
)

# MOEX OFZ bonds around CBR rate meetings:
# Spreads widen 3-5x around major monetary policy events.
# Used for CBREventStrategy trades (T-3 to T+2 around meetings).
MOEX_BOND_EVENT_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal("0.0005"),  # Same commission
    min_commission=Decimal("0.01"),
    spread_bps=Decimal(15),  # 3x normal spread around events
    slippage_bps=Decimal(10),  # 3x normal slippage
)

# Per-instrument spread uplift for off-the-run bonds (bps).
# These bonds have lower liquidity and wider bid-ask spreads.
OFF_THE_RUN_SPREAD_UPLIFT_BPS: dict[str, Decimal] = {
    "SU26238RMFS4": Decimal(10),  # Deep discount, low liquidity
    "SU26239RMFS2": Decimal(10),  # Older issue
}


def bond_total_cost(
    costs: TransactionCosts,
    clean_price_pct: Decimal,
    face_value: Decimal,
    quantity: Decimal,
    ticker: str | None = None,
) -> Decimal:
    """Compute total cost for a bond trade.

    Bond prices are quoted as % of face value (e.g., 85.50 means 85.50% of 1000 RUB).
    The cost is computed on the actual RUB value, not the percentage.

    Args:
        costs: TransactionCosts preset to use.
        clean_price_pct: Clean price as % of face value (e.g., 85.50).
        face_value: Face value per bond (typically 1000 RUB for OFZ).
        quantity: Number of bonds traded.
        ticker: Optional ticker for off-the-run spread uplift.

    Returns:
        Total cost in RUB.
    """
    price_rub = clean_price_pct / Decimal(100) * face_value
    base_cost = costs.total_cost(price_rub, quantity)

    # Add off-the-run spread uplift if applicable
    if ticker and ticker in OFF_THE_RUN_SPREAD_UPLIFT_BPS:
        uplift_bps = OFF_THE_RUN_SPREAD_UPLIFT_BPS[ticker]
        uplift_per_bond = price_rub * uplift_bps / Decimal(10000)
        base_cost += uplift_per_bond * quantity

    return base_cost
