"""Pre-trade risk checks (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class PreTradeResult:
    """Result of pre-trade risk validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


class PreTradeChecker:
    """Validates orders against risk limits before execution.

    Checks:
        1. Position size does not exceed ``max_position_pct`` of equity.
        2. Sufficient cash is available.
        3. Open position count is below ``max_positions_per_market``.
    """

    def __init__(
        self,
        max_position_pct: Decimal = Decimal("0.20"),
        max_positions_per_market: int = 10,
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions_per_market

    def check(
        self,
        order_value: Decimal,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        open_position_count: int,
    ) -> PreTradeResult:
        """Run all pre-trade risk checks.

        Args:
            order_value: Notional value of the proposed order.
            portfolio_equity: Current total portfolio equity.
            available_cash: Cash available for trading.
            open_position_count: Number of currently open positions.

        Returns:
            A :class:`PreTradeResult` indicating pass/fail and any violations.
        """
        violations: list[str] = []

        # 1. Position size check
        if portfolio_equity == 0:
            violations.append("Portfolio equity is zero; no trades permitted")
        else:
            pct = order_value / portfolio_equity
            if pct > self._max_position_pct:
                max_pct = float(self._max_position_pct)
                violations.append(f"Position size {float(pct):.1%} exceeds max {max_pct:.1%}")

        # 2. Cash check
        if order_value > available_cash:
            violations.append(f"Insufficient cash: need {order_value}, have {available_cash}")

        # 3. Position count check
        if open_position_count >= self._max_positions:
            violations.append(
                f"Open positions ({open_position_count}) >= max ({self._max_positions})"
            )

        return PreTradeResult(passed=len(violations) == 0, violations=violations)
