"""DV01-based bond position sizing (Layer 4).

Instead of Kelly (equity-oriented), bonds are sized so aggregate portfolio
DV01 stays within a budget derived from:

    max_dv01 = layer_equity * max_dd_pct / expected_max_rate_move_bps

Example: 1.5M equity, 5% DD limit, expect max 500bps move:
    max_dv01 = 1_500_000 * 0.05 / 500 = 150 RUB per basis point

Each bond position is sized proportional to available DV01 budget,
capped by a single-position limit as fraction of equity.
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
    """

    def __init__(
        self,
        max_dd_pct: Decimal = Decimal("0.05"),
        expected_max_rate_move_bps: int = 500,
        max_single_position_pct: Decimal = Decimal("0.30"),
    ) -> None:
        self._max_dd_pct = max_dd_pct
        self._expected_max_rate_move_bps = expected_max_rate_move_bps
        self._max_single_position_pct = max_single_position_pct

    def compute_position_size(
        self,
        layer_equity: Decimal,
        bond_dv01_per_unit: Decimal,
        current_portfolio_dv01: Decimal,
        face_value: Decimal = Decimal(1000),
    ) -> int:
        """Compute number of bonds to buy.

        Args:
            layer_equity: Total equity allocated to this layer.
            bond_dv01_per_unit: DV01 of one bond (from bond_math.dv01()).
            current_portfolio_dv01: Sum of DV01 across all existing positions.
            face_value: Face value per bond.

        Returns:
            Number of bonds (integer, may be 0 if budget exhausted).
        """
        max_dv01 = layer_equity * self._max_dd_pct / Decimal(self._expected_max_rate_move_bps)
        remaining_dv01 = max_dv01 - current_portfolio_dv01

        if remaining_dv01 <= 0 or bond_dv01_per_unit <= 0:
            return 0

        # Max bonds by DV01 budget
        max_by_dv01 = int(remaining_dv01 / bond_dv01_per_unit)

        # Max bonds by single position limit
        max_by_position = int(layer_equity * self._max_single_position_pct / face_value)

        return min(max_by_dv01, max_by_position)
