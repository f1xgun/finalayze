"""Unit tests for transaction cost model."""

from __future__ import annotations

from decimal import Decimal

from finalayze.backtest.costs import TransactionCosts


class TestTransactionCostsDefaults:
    """Default cost parameters are sensible."""

    def test_default_values(self) -> None:
        costs = TransactionCosts()
        assert costs.commission_per_share == Decimal("0.005")
        assert costs.min_commission == Decimal("1.00")
        assert costs.spread_bps == Decimal(5)
        assert costs.slippage_bps == Decimal(3)


class TestTotalCost:
    """total_cost computes commission + spread + slippage."""

    def test_min_commission_applies(self) -> None:
        """When per-share commission is less than min, use min."""
        costs = TransactionCosts()
        # 10 shares * 0.005 = 0.05, below min_commission of 1.00
        price = Decimal(100)
        qty = Decimal(10)
        total = costs.total_cost(price, qty)
        # commission = max(1.00, 0.05) = 1.00
        # spread = 100 * 5 / 10000 = 0.05
        # slippage = 100 * 3 / 10000 = 0.03
        # total = 1.00 + (0.05 + 0.03) * 10 = 1.00 + 0.80 = 1.80
        assert total == Decimal("1.80")

    def test_per_share_commission_applies(self) -> None:
        """When per-share commission exceeds min, use per-share."""
        costs = TransactionCosts()
        price = Decimal(50)
        qty = Decimal(1000)
        total = costs.total_cost(price, qty)
        # commission = max(1.00, 0.005 * 1000) = max(1.00, 5.00) = 5.00
        # spread = 50 * 5 / 10000 = 0.025
        # slippage = 50 * 3 / 10000 = 0.015
        # total = 5.00 + (0.025 + 0.015) * 1000 = 5.00 + 40.00 = 45.00
        assert total == Decimal("45.00")

    def test_zero_cost_model(self) -> None:
        """All-zeros cost model produces zero cost."""
        costs = TransactionCosts(
            commission_per_share=Decimal(0),
            min_commission=Decimal(0),
            spread_bps=Decimal(0),
            slippage_bps=Decimal(0),
        )
        assert costs.total_cost(Decimal(100), Decimal(50)) == Decimal(0)

    def test_frozen_dataclass(self) -> None:
        """TransactionCosts is immutable."""
        costs = TransactionCosts()
        import dataclasses

        assert dataclasses.is_dataclass(costs)
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            costs.min_commission = Decimal("2.00")  # type: ignore[misc]


class TestCostImpactOnReturns:
    """Transaction costs should reduce effective PnL."""

    def test_costs_reduce_returns(self) -> None:
        costs = TransactionCosts()
        price = Decimal(150)
        qty = Decimal(100)
        cost = costs.total_cost(price, qty)
        assert cost > 0
        # Entry + exit cost both reduce PnL
        total_round_trip = cost + costs.total_cost(Decimal(155), qty)
        assert total_round_trip > cost
