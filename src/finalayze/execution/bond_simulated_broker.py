"""Bond-aware simulated broker for backtesting (Layer 5).

Subclass of SimulatedBroker that:
1. Adds NKD to buy cost (dirty price = clean + NKD)
2. Adds NKD to sell proceeds
3. Processes coupon payments during hold periods
4. Tracks coupon income separately (for tax calculation)
5. Values portfolio at dirty prices (clean + accrued NKD)

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.constants import NDFL_RATE
from finalayze.execution.simulated_broker import SimulatedBroker

if TYPE_CHECKING:
    from datetime import date

    from finalayze.core.schemas import CouponPayment


_OFZ_FACE_VALUE = Decimal(1000)


class BondSimulatedBroker(SimulatedBroker):
    """Simulated broker with bond-specific mechanics.

    Bond prices are clean prices as % of face value (e.g. 85.50 means 855 RUB per bond).
    When buying, the actual cost includes NKD (accrued interest).
    When selling, the proceeds include NKD.

    This class uses its own ``buy_bond``/``sell_bond`` methods instead of the parent's
    ``submit_order``, because bond orders require NKD and clean-price-percentage inputs.
    """

    def __init__(
        self,
        initial_cash: Decimal,
        coupon_schedule: dict[str, list[CouponPayment]],
        face_value: Decimal = _OFZ_FACE_VALUE,
        tax_rate: Decimal = NDFL_RATE,  # L0 single source (D-12)
    ) -> None:
        super().__init__(initial_cash=initial_cash)
        self._coupon_schedule = coupon_schedule
        self._face_value = face_value
        self._tax_rate = tax_rate
        self._total_coupon_income_gross = Decimal(0)
        self._total_coupon_income_net = Decimal(0)
        self._total_tax_paid = Decimal(0)
        # Track NKD paid on purchase (to avoid double-counting)
        self._nkd_paid: dict[str, Decimal] = {}

    def buy_bond(
        self,
        symbol: str,
        quantity: int,
        clean_price_pct: Decimal,
        nkd_per_bond: Decimal,
        transaction_cost: Decimal = Decimal(0),
    ) -> bool:
        """Execute a bond buy order.

        Total cost = (clean_price_pct/100 * face_value + nkd) * quantity + fees

        Args:
            symbol: Bond ticker/FIGI.
            quantity: Number of bonds to buy.
            clean_price_pct: Clean price as % of face (e.g. 85.50).
            nkd_per_bond: Accrued interest per bond in RUB.
            transaction_cost: Total transaction cost in RUB.

        Returns:
            True if order filled (sufficient cash), False otherwise.
        """
        price_per_bond = clean_price_pct / Decimal(100) * self._face_value
        nkd_total = nkd_per_bond * quantity
        total_cost = price_per_bond * quantity + nkd_total + transaction_cost

        if total_cost > self._cash:
            return False

        self._cash -= total_cost
        current_qty = self._positions.get(symbol, Decimal(0))
        self._positions[symbol] = current_qty + Decimal(quantity)

        # Track NKD paid (accumulate if adding to existing position)
        existing_nkd = self._nkd_paid.get(symbol, Decimal(0))
        self._nkd_paid[symbol] = existing_nkd + nkd_total

        return True

    def sell_bond(
        self,
        symbol: str,
        quantity: int,
        clean_price_pct: Decimal,
        nkd_per_bond: Decimal,
        transaction_cost: Decimal = Decimal(0),
    ) -> Decimal:
        """Execute a bond sell order.

        Proceeds = (clean_price_pct/100 * face_value + nkd) * quantity - fees

        Args:
            symbol: Bond ticker/FIGI.
            quantity: Number of bonds to sell.
            clean_price_pct: Clean price as % of face (e.g. 85.50).
            nkd_per_bond: Accrued interest per bond in RUB.
            transaction_cost: Total transaction cost in RUB.

        Returns:
            Proceeds from the trade. Returns Decimal(0) if insufficient position.
        """
        current_qty = self._positions.get(symbol, Decimal(0))
        if current_qty < quantity:
            return Decimal(0)

        price_per_bond = clean_price_pct / Decimal(100) * self._face_value
        nkd_total = nkd_per_bond * quantity
        proceeds = price_per_bond * quantity + nkd_total - transaction_cost

        self._cash += proceeds
        self._positions[symbol] = current_qty - Decimal(quantity)
        if self._positions[symbol] == 0:
            del self._positions[symbol]
            # Clean up NKD tracking
            self._nkd_paid.pop(symbol, None)

        return proceeds

    def process_coupons(self, current_date: date) -> Decimal:
        """Process coupon payments for all held positions on this date.

        For each position, check if any coupon payment falls on ``current_date``.
        Credit net coupon (after NDFL tax) to cash.

        Args:
            current_date: Current bar date.

        Returns:
            Total net coupon income credited on this date.
        """
        total_net = Decimal(0)
        for symbol, qty in list(self._positions.items()):
            if qty <= 0:
                continue
            coupons = self._coupon_schedule.get(symbol, [])
            for coupon in coupons:
                if coupon.coupon_date == current_date:
                    gross = coupon.amount_per_bond * qty
                    tax = gross * self._tax_rate
                    net = gross - tax
                    self._cash += net
                    self._total_coupon_income_gross += gross
                    self._total_coupon_income_net += net
                    self._total_tax_paid += tax
                    total_net += net
        return total_net

    def portfolio_value_at(
        self,
        prices: dict[str, Decimal],
        nkd_values: dict[str, Decimal],
    ) -> Decimal:
        """Compute portfolio value at dirty prices.

        Value = cash + sum(qty * (clean_price/100 * face + nkd))

        Args:
            prices: Mapping of symbol to clean price as % of face.
            nkd_values: Mapping of symbol to NKD per bond in RUB.

        Returns:
            Total portfolio value in RUB.
        """
        value = self._cash
        for symbol, qty in self._positions.items():
            clean_pct = prices.get(symbol, Decimal(0))
            nkd = nkd_values.get(symbol, Decimal(0))
            dirty = clean_pct / Decimal(100) * self._face_value + nkd
            value += qty * dirty
        return value

    @property
    def coupon_income_gross(self) -> Decimal:
        """Total gross coupon income received."""
        return self._total_coupon_income_gross

    @property
    def coupon_income_net(self) -> Decimal:
        """Total net coupon income (after tax)."""
        return self._total_coupon_income_net

    @property
    def tax_paid(self) -> Decimal:
        """Total NDFL tax paid on coupon income."""
        return self._total_tax_paid
