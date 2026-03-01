"""Unit tests for SimulatedBroker.deduct_fees (6B.8)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.execution.simulated_broker import SimulatedBroker

INITIAL_CASH = Decimal(10000)
FEE_AMOUNT = Decimal(10)


class TestDeductFees:
    def test_deduct_fees_reduces_cash(self) -> None:
        """deduct_fees() reduces available cash by the specified amount."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        broker.deduct_fees(FEE_AMOUNT)
        portfolio = broker.get_portfolio()
        assert portfolio.cash == INITIAL_CASH - FEE_AMOUNT

    def test_deduct_fees_negative_raises(self) -> None:
        """deduct_fees() raises ValueError for negative amounts."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        with pytest.raises(ValueError, match="non-negative"):
            broker.deduct_fees(Decimal(-1))

    def test_deduct_fees_zero_is_noop(self) -> None:
        """deduct_fees(0) does not change cash."""
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        broker.deduct_fees(Decimal(0))
        assert broker.get_portfolio().cash == INITIAL_CASH
