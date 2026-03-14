"""Bond position sizing: DV01-budget and equal-weight approaches (Layer 4).

DV01BudgetStep: for fixed-rate bonds (OFZ-PD).  Sizes so aggregate portfolio
DV01 stays within a risk budget.

EqualWeightBondSizer: for floating-rate bonds (OFZ-PK).  Divides capital equally
across N symbols by nominal value.  DV01 doesn't apply to floaters because their
duration is near-zero (coupon resets track RUONIA).
"""

from __future__ import annotations

from decimal import Decimal


class DV01BudgetStep:
    """Position sizing step for bonds using DV01 budgeting.

    Computes the maximum number of bonds to buy given the DV01 budget
    remaining in the portfolio and a single-position cap.

    Args:
        max_dd_pct: Maximum acceptable drawdown as a fraction (default 0.05 = 5%).
        expected_max_rate_move_bps: Worst-case parallel rate shock in basis points.
        max_single_position_pct: Maximum fraction of equity in a single bond issue.
        max_dv01_per_bond_pct: Maximum fraction of total DV01 budget a single bond
            issue may consume (default 0.40 = 40%).
    """

    def __init__(
        self,
        max_dd_pct: Decimal = Decimal("0.05"),
        expected_max_rate_move_bps: int = 500,
        max_single_position_pct: Decimal = Decimal("0.30"),
        max_dv01_per_bond_pct: Decimal = Decimal("0.40"),
    ) -> None:
        self._max_dd_pct = max_dd_pct
        self._expected_max_rate_move_bps = expected_max_rate_move_bps
        self._max_single_position_pct = max_single_position_pct
        self._max_dv01_per_bond_pct = max_dv01_per_bond_pct

    def compute_position_size(
        self,
        layer_equity: Decimal,
        bond_dv01_per_unit: Decimal,
        current_portfolio_dv01: Decimal,
        unit_cost: Decimal = Decimal(1000),
        transaction_costs_per_unit: Decimal = Decimal(0),
    ) -> int:
        """Compute number of bonds to buy.

        Args:
            layer_equity: Total equity allocated to this layer.
            bond_dv01_per_unit: DV01 of one bond (from bond_math.dv01()).
            current_portfolio_dv01: Sum of DV01 across all existing positions.
            unit_cost: Cost per bond unit -- dirty price for bonds (face + NKD),
                not face value.  Default 1000 for backward compatibility.
            transaction_costs_per_unit: Broker commission + exchange fee per bond
                unit.  Added to unit_cost for the cash sufficiency check.

        Returns:
            Number of bonds (integer, may be 0 if budget exhausted).
        """
        max_dv01 = layer_equity * self._max_dd_pct / Decimal(self._expected_max_rate_move_bps)
        remaining_dv01 = max_dv01 - current_portfolio_dv01

        if remaining_dv01 <= 0 or bond_dv01_per_unit <= 0:
            return 0

        # Max bonds by total DV01 budget
        max_by_dv01 = int(remaining_dv01 / bond_dv01_per_unit)

        # Max bonds by per-bond DV01 cap (fraction of total budget)
        per_bond_dv01_cap = max_dv01 * self._max_dv01_per_bond_pct
        max_by_per_bond = int(per_bond_dv01_cap / bond_dv01_per_unit)

        # Max bonds by single position limit (uses dirty price + transaction costs)
        effective_cost = unit_cost + transaction_costs_per_unit
        max_by_position = int(layer_equity * self._max_single_position_pct / effective_cost)

        return min(max_by_dv01, max_by_per_bond, max_by_position)


class EqualWeightBondSizer:
    """Equal-weight position sizing for floating-rate bonds.

    Allocates capital equally across *n_symbols*.  Ignores DV01 arguments
    (floaters have near-zero duration, so DV01 budget is meaningless).
    """

    def __init__(
        self,
        n_symbols: int,
        max_single_position_pct: Decimal = Decimal("0.25"),
    ) -> None:
        self._n_symbols = max(n_symbols, 1)
        self._max_single_position_pct = max_single_position_pct

    def compute_position_size(
        self,
        layer_equity: Decimal,
        bond_dv01_per_unit: Decimal,  # noqa: ARG002
        current_portfolio_dv01: Decimal,  # noqa: ARG002
        unit_cost: Decimal = Decimal(1000),
        transaction_costs_per_unit: Decimal = Decimal(0),
    ) -> int:
        """Compute number of bonds for an equal-weight allocation.

        Args:
            layer_equity: Total equity allocated to this layer.
            bond_dv01_per_unit: Ignored (floaters have near-zero duration).
            current_portfolio_dv01: Ignored.
            unit_cost: Cost per bond unit -- dirty price for bonds.
                Default 1000 for backward compatibility.
            transaction_costs_per_unit: Broker commission + exchange fee per unit.
        """
        target = layer_equity / Decimal(self._n_symbols)
        cap = layer_equity * self._max_single_position_pct
        effective_cost = unit_cost + transaction_costs_per_unit
        return int(min(target, cap) / effective_cost)
