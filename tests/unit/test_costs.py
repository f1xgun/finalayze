"""Unit tests for TransactionCosts model (6B.4)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.backtest.costs import MOEX_COSTS, US_COSTS, TransactionCosts

# Constants
US_PRICE = Decimal(150)
US_QTY = Decimal(100)
MOEX_PRICE = Decimal(100)
MOEX_QTY = Decimal(100)
TINY_PRICE = Decimal(1)
TINY_QTY = Decimal(1)


class TestTransactionCosts:
    def test_us_costs_per_share_unchanged(self) -> None:
        """US_COSTS uses per-share commission model."""
        cost = US_COSTS.total_cost(US_PRICE, US_QTY)
        # commission = max(1.00, 0.005 * 100) = 1.00 (min applies)
        # spread = 150 * 5/10000 = 0.075
        # slippage = 150 * 3/10000 = 0.045
        # total = 1.00 + (0.075 + 0.045) * 100 = 1.00 + 12.00 = 13.00
        expected_commission = max(Decimal("1.00"), Decimal("0.005") * US_QTY)
        spread = US_PRICE * Decimal(5) / Decimal(10000)
        slippage = US_PRICE * Decimal(3) / Decimal(10000)
        expected = expected_commission + (spread + slippage) * US_QTY
        assert cost == expected

    def test_moex_costs_rate_based(self) -> None:
        """MOEX_COSTS uses commission_rate (percentage of trade value)."""
        cost = MOEX_COSTS.total_cost(MOEX_PRICE, MOEX_QTY)
        # commission = max(0.10, 100 * 100 * 0.0003) = max(0.10, 3.0) = 3.0
        expected_commission = max(
            Decimal("0.10"), MOEX_PRICE * MOEX_QTY * Decimal("0.0003")
        )
        spread = MOEX_PRICE * Decimal(10) / Decimal(10000)
        slippage = MOEX_PRICE * Decimal(7) / Decimal(10000)
        expected = expected_commission + (spread + slippage) * MOEX_QTY
        assert cost == expected
        # Verify commission is 3.0 (not the old 0.30)
        assert expected_commission == Decimal("3.0")

    def test_commission_rate_respects_min_commission(self) -> None:
        """For tiny trades, min_commission kicks in."""
        # commission = max(0.10, 1 * 1 * 0.0003) = max(0.10, 0.0003) = 0.10
        costs = TransactionCosts(
            commission_per_share=Decimal(0),
            commission_rate=Decimal("0.0003"),
            min_commission=Decimal("0.10"),
            spread_bps=Decimal(0),
            slippage_bps=Decimal(0),
        )
        cost = costs.total_cost(TINY_PRICE, TINY_QTY)
        assert cost == Decimal("0.10")
