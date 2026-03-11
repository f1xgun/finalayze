"""Unit tests for BondSimulatedBroker (Task 2.3)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.schemas import CouponPayment
from finalayze.execution.bond_simulated_broker import BondSimulatedBroker

# ── Constants ──────────────────────────────────────────────────────────────────

_INITIAL_CASH = Decimal(1_000_000)
_FACE_VALUE = Decimal(1000)
_TAX_RATE = Decimal("0.13")
_CLEAN_PRICE_PCT = Decimal("85.50")  # 85.50% of face = 855 RUB per bond
_NKD_PER_BOND = Decimal("20.00")  # 20 RUB accrued interest per bond
_QUANTITY = 10
_COUPON_AMOUNT = Decimal("35.40")  # coupon per bond in RUB
_COUPON_DATE = date(2026, 6, 15)
_NON_COUPON_DATE = date(2026, 6, 14)
_FIGI_A = "BBG00NRTM735"
_FIGI_B = "BBG00QHJG742"

# Derived values for assertions
_PRICE_PER_BOND = _CLEAN_PRICE_PCT / Decimal(100) * _FACE_VALUE  # 855
_NKD_TOTAL = _NKD_PER_BOND * _QUANTITY  # 200
_BUY_COST = _PRICE_PER_BOND * _QUANTITY + _NKD_TOTAL  # 8750


def _make_coupon_schedule(
    figi: str = _FIGI_A,
    coupon_date: date = _COUPON_DATE,
    amount: Decimal = _COUPON_AMOUNT,
) -> dict[str, list[CouponPayment]]:
    """Create a minimal coupon schedule for testing."""
    return {
        figi: [
            CouponPayment(
                bond_figi=figi,
                coupon_date=coupon_date,
                record_date=date(2026, 6, 12),  # T-2 business days
                amount_per_bond=amount,
                coupon_number=1,
            ),
        ],
    }


def _make_broker(
    initial_cash: Decimal = _INITIAL_CASH,
    coupon_schedule: dict[str, list[CouponPayment]] | None = None,
) -> BondSimulatedBroker:
    """Create a BondSimulatedBroker with defaults."""
    if coupon_schedule is None:
        coupon_schedule = _make_coupon_schedule()
    return BondSimulatedBroker(
        initial_cash=initial_cash,
        coupon_schedule=coupon_schedule,
        face_value=_FACE_VALUE,
        tax_rate=_TAX_RATE,
    )


class TestBuyBond:
    """Test buy_bond method."""

    def test_buy_deducts_correct_cost(self) -> None:
        """Buy 10 bonds at 85.50% + 20 RUB NKD: cost = 855*10 + 20*10 = 8750."""
        broker = _make_broker()
        result = broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        assert result is True
        expected_cash = _INITIAL_CASH - _BUY_COST
        assert broker.get_portfolio().cash == expected_cash

    def test_buy_creates_position(self) -> None:
        """After buying, position quantity matches."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        positions = broker.get_positions()
        assert positions[_FIGI_A] == Decimal(_QUANTITY)

    def test_buy_with_transaction_cost(self) -> None:
        """Transaction costs are added to the total buy cost."""
        broker = _make_broker()
        fee = Decimal(50)
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
            transaction_cost=fee,
        )

        expected_cash = _INITIAL_CASH - _BUY_COST - fee
        assert broker.get_portfolio().cash == expected_cash

    def test_buy_insufficient_cash_returns_false(self) -> None:
        """Buy fails when total cost exceeds available cash."""
        broker = _make_broker(initial_cash=Decimal(5000))
        result = broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        assert result is False
        # Cash unchanged
        assert broker.get_portfolio().cash == Decimal(5000)
        # No position created
        assert not broker.has_position(_FIGI_A)

    def test_buy_adds_to_existing_position(self) -> None:
        """Buying more bonds adds to existing quantity."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=5,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=3,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        positions = broker.get_positions()
        assert positions[_FIGI_A] == Decimal(8)


class TestSellBond:
    """Test sell_bond method."""

    def test_sell_adds_proceeds_to_cash(self) -> None:
        """Sell proceeds = (clean_price/100 * face + nkd) * qty."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        sell_nkd = Decimal("25.00")
        sell_price_pct = Decimal("86.00")
        proceeds = broker.sell_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=sell_price_pct,
            nkd_per_bond=sell_nkd,
        )

        expected_proceeds = (
            sell_price_pct / Decimal(100) * _FACE_VALUE * _QUANTITY + sell_nkd * _QUANTITY
        )
        assert proceeds == expected_proceeds

        # Cash = initial - buy_cost + sell_proceeds
        expected_cash = _INITIAL_CASH - _BUY_COST + expected_proceeds
        assert broker.get_portfolio().cash == expected_cash

    def test_sell_removes_position(self) -> None:
        """Full sell removes the position from tracking."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        broker.sell_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        assert not broker.has_position(_FIGI_A)

    def test_sell_partial_position(self) -> None:
        """Partial sell reduces quantity but keeps position."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        broker.sell_bond(
            symbol=_FIGI_A,
            quantity=3,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        positions = broker.get_positions()
        assert positions[_FIGI_A] == Decimal(7)

    def test_sell_no_position_returns_zero(self) -> None:
        """Selling without a position returns Decimal(0)."""
        broker = _make_broker()
        result = broker.sell_bond(
            symbol=_FIGI_A,
            quantity=5,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        assert result == Decimal(0)

    def test_sell_more_than_held_returns_zero(self) -> None:
        """Trying to sell more than held returns Decimal(0) (no partial fill)."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=5,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        result = broker.sell_bond(
            symbol=_FIGI_A,
            quantity=10,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        assert result == Decimal(0)
        # Position unchanged
        assert broker.get_positions()[_FIGI_A] == Decimal(5)

    def test_sell_with_transaction_cost(self) -> None:
        """Transaction cost is subtracted from sell proceeds."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        fee = Decimal(30)
        proceeds = broker.sell_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
            transaction_cost=fee,
        )

        expected_proceeds = _BUY_COST - fee  # same price, so proceeds = cost - fee
        assert proceeds == expected_proceeds


class TestProcessCoupons:
    """Test coupon processing during hold periods."""

    def test_coupon_credited_on_payment_date(self) -> None:
        """Net coupon (after 13% tax) is credited to cash on coupon date."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        cash_before = broker.get_portfolio().cash
        net_income = broker.process_coupons(_COUPON_DATE)

        gross = _COUPON_AMOUNT * _QUANTITY
        tax = gross * _TAX_RATE
        expected_net = gross - tax

        assert net_income == expected_net
        assert broker.get_portfolio().cash == cash_before + expected_net

    def test_no_coupon_on_non_payment_date(self) -> None:
        """process_coupons returns 0 on a date with no coupon payment."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        cash_before = broker.get_portfolio().cash
        net_income = broker.process_coupons(_NON_COUPON_DATE)

        assert net_income == Decimal(0)
        assert broker.get_portfolio().cash == cash_before

    def test_no_coupon_without_position(self) -> None:
        """No coupon if no position is held (even on coupon date)."""
        broker = _make_broker()
        net_income = broker.process_coupons(_COUPON_DATE)
        assert net_income == Decimal(0)

    def test_coupon_income_tracking(self) -> None:
        """Gross, net, and tax properties track cumulative coupon income."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )
        broker.process_coupons(_COUPON_DATE)

        gross = _COUPON_AMOUNT * _QUANTITY
        tax = gross * _TAX_RATE
        net = gross - tax

        assert broker.coupon_income_gross == gross
        assert broker.coupon_income_net == net
        assert broker.tax_paid == tax


class TestPortfolioValue:
    """Test portfolio_value_at method for dirty price valuation."""

    def test_portfolio_value_at_dirty_prices(self) -> None:
        """Portfolio value = cash + sum(qty * dirty_price)."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        current_clean = Decimal("86.00")
        current_nkd = Decimal("22.00")
        value = broker.portfolio_value_at(
            prices={_FIGI_A: current_clean},
            nkd_values={_FIGI_A: current_nkd},
        )

        dirty_per_bond = current_clean / Decimal(100) * _FACE_VALUE + current_nkd
        position_value = Decimal(_QUANTITY) * dirty_per_bond
        expected = (_INITIAL_CASH - _BUY_COST) + position_value

        assert value == expected

    def test_portfolio_value_cash_only(self) -> None:
        """Portfolio value equals cash when no positions held."""
        broker = _make_broker()
        value = broker.portfolio_value_at(prices={}, nkd_values={})
        assert value == _INITIAL_CASH

    def test_portfolio_value_missing_price_uses_zero(self) -> None:
        """If a held symbol has no entry in prices dict, value is 0 for it."""
        broker = _make_broker()
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=_QUANTITY,
            clean_price_pct=_CLEAN_PRICE_PCT,
            nkd_per_bond=_NKD_PER_BOND,
        )

        # No price provided for FIGI_A
        value = broker.portfolio_value_at(prices={}, nkd_values={})
        expected = _INITIAL_CASH - _BUY_COST  # position valued at 0
        assert value == expected


class TestMultiplePositions:
    """Test isolation between multiple bond positions."""

    def test_two_bonds_independent(self) -> None:
        """Buying/selling two different bonds maintains independent tracking."""
        schedule = {
            **_make_coupon_schedule(_FIGI_A),
            **_make_coupon_schedule(
                _FIGI_B,
                coupon_date=date(2026, 7, 15),
                amount=Decimal("40.00"),
            ),
        }
        broker = _make_broker(coupon_schedule=schedule)

        # Buy both
        broker.buy_bond(
            symbol=_FIGI_A,
            quantity=10,
            clean_price_pct=Decimal("85.50"),
            nkd_per_bond=Decimal("20.00"),
        )
        broker.buy_bond(
            symbol=_FIGI_B,
            quantity=5,
            clean_price_pct=Decimal("90.00"),
            nkd_per_bond=Decimal("15.00"),
        )

        positions = broker.get_positions()
        assert positions[_FIGI_A] == Decimal(10)
        assert positions[_FIGI_B] == Decimal(5)

        # Sell only FIGI_A
        broker.sell_bond(
            symbol=_FIGI_A,
            quantity=10,
            clean_price_pct=Decimal("86.00"),
            nkd_per_bond=Decimal("22.00"),
        )

        assert not broker.has_position(_FIGI_A)
        assert broker.has_position(_FIGI_B)
        assert broker.get_positions()[_FIGI_B] == Decimal(5)

    def test_coupons_only_for_held_bonds(self) -> None:
        """Coupons are only paid for bonds actually held on the coupon date."""
        schedule = {
            **_make_coupon_schedule(_FIGI_A, coupon_date=_COUPON_DATE),
            **_make_coupon_schedule(_FIGI_B, coupon_date=_COUPON_DATE, amount=Decimal("40.00")),
        }
        broker = _make_broker(coupon_schedule=schedule)

        # Only buy FIGI_B
        broker.buy_bond(
            symbol=_FIGI_B,
            quantity=5,
            clean_price_pct=Decimal("90.00"),
            nkd_per_bond=Decimal("15.00"),
        )

        net_income = broker.process_coupons(_COUPON_DATE)

        # Only FIGI_B coupon should be paid
        gross_b = Decimal("40.00") * 5
        tax_b = gross_b * _TAX_RATE
        expected_net = gross_b - tax_b

        assert net_income == expected_net
