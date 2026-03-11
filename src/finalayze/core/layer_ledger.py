"""Layer ledger -- tracks per-layer cash, positions, and drawdown (Layer 0).

Each portfolio layer operates as a virtual sub-account with:
- Own cash allocation
- Own position tracking
- Own peak-to-trough drawdown monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class LayerLedger:
    """Mutable ledger for a single portfolio layer.

    Tracks cash, positions, equity, and peak-to-trough drawdown.
    """

    layer_id: str  # PortfolioLayer value
    cash: Decimal
    positions: dict[str, Decimal] = field(default_factory=dict)  # symbol -> quantity
    peak_equity: Decimal = Decimal(0)
    current_equity: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.peak_equity == 0:
            self.peak_equity = self.cash
        if self.current_equity == 0:
            self.current_equity = self.cash

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity and peak tracking."""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    @property
    def drawdown_pct(self) -> Decimal:
        """Current peak-to-trough drawdown as decimal fraction."""
        if self.peak_equity <= 0:
            return Decimal(0)
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def is_empty(self) -> bool:
        """True if no positions held."""
        return not self.positions or all(q == 0 for q in self.positions.values())

    def add_position(self, symbol: str, quantity: Decimal) -> None:
        """Add to a position (or create new)."""
        current = self.positions.get(symbol, Decimal(0))
        self.positions[symbol] = current + quantity

    def remove_position(self, symbol: str, quantity: Decimal) -> None:
        """Reduce a position. Removes entry if quantity reaches 0."""
        current = self.positions.get(symbol, Decimal(0))
        new_qty = current - quantity
        if new_qty <= 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty

    def debit_cash(self, amount: Decimal) -> bool:
        """Debit cash. Returns False if insufficient."""
        if amount > self.cash:
            return False
        self.cash -= amount
        return True

    def credit_cash(self, amount: Decimal) -> None:
        """Credit cash."""
        self.cash += amount
