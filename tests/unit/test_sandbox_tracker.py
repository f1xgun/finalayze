"""Unit tests for SandboxPortfolioTracker and ShadowLedger."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from finalayze.core.schemas import CouponPayment, PortfolioState
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.sandbox_tracker import (
    DividendEntry,
    SandboxAdjustment,
    SandboxPortfolioTracker,
    ShadowLedger,
)
from finalayze.markets.instruments import Instrument, InstrumentRegistry

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_EQUITY = Decimal(1_000_000)
_DEFAULT_CASH = Decimal(500_000)
_SBER_FIGI = "BBG004730N88"
_BOND_FIGI = "TCS00A1074G2"
_COUPON_AMOUNT = Decimal("56.25")  # typical OFZ coupon per bond
_DIVIDEND_AMOUNT = Decimal("33.30")  # typical SBER dividend per share
_TAX_RATE_13 = Decimal("0.13")
_TAX_RATE_15 = Decimal("0.15")
_BOND_QTY = Decimal(100)
_STOCK_QTY = Decimal(10)


# ── Fixtures / helpers ───────────────────────────────────────────────────────


def _make_portfolio_state(
    *,
    cash: Decimal = _DEFAULT_CASH,
    positions: dict[str, Decimal] | None = None,
    equity: Decimal = _DEFAULT_EQUITY,
) -> PortfolioState:
    return PortfolioState(
        cash=cash,
        positions=positions or {},
        equity=equity,
        timestamp=datetime.now(UTC),
    )


def _mock_broker(
    *,
    positions: dict[str, Decimal] | None = None,
    equity: Decimal = _DEFAULT_EQUITY,
    cash: Decimal = _DEFAULT_CASH,
) -> MagicMock:
    """Create a mock TinkoffBroker with configurable portfolio state."""
    broker = MagicMock()
    pos = positions or {}
    broker.get_positions.return_value = pos
    broker.get_portfolio.return_value = _make_portfolio_state(
        cash=cash,
        positions=pos,
        equity=equity,
    )
    broker.has_position.return_value = False
    broker.submit_order.return_value = OrderResult(
        filled=True,
        fill_price=Decimal(270),
        symbol="SBER",
        side="BUY",
        quantity=Decimal(10),
    )
    broker.cancel_order.return_value = None
    return broker


def _make_registry() -> InstrumentRegistry:
    """Create a registry with SBER and a bond for FIGI resolution tests."""
    registry = InstrumentRegistry()
    registry.register(
        Instrument(
            symbol="SBER",
            market_id="moex",
            name="Sberbank",
            figi=_SBER_FIGI,
            lot_size=10,
            currency="RUB",
        )
    )
    registry.register(
        Instrument(
            symbol="SU26244RMFS2",
            market_id="moex",
            name="OFZ 26244",
            instrument_type="bond",
            figi=_BOND_FIGI,
            lot_size=1,
            currency="RUB",
        )
    )
    return registry


def _make_coupon(
    *,
    figi: str = _BOND_FIGI,
    coupon_date: date = date(2026, 3, 15),
    record_date: date = date(2026, 3, 11),
    amount: Decimal = _COUPON_AMOUNT,
    number: int = 1,
) -> CouponPayment:
    return CouponPayment(
        bond_figi=figi,
        coupon_date=coupon_date,
        record_date=record_date,
        amount_per_bond=amount,
        coupon_number=number,
    )


def _make_dividend(
    *,
    symbol: str = "SBER",
    ex_date: date = date(2026, 7, 15),
    amount: Decimal = _DIVIDEND_AMOUNT,
) -> DividendEntry:
    return DividendEntry(symbol=symbol, ex_date=ex_date, amount_per_share=amount)


# ═══════════════════════════════════════════════════════════════════════════════
# TestShadowLedger
# ═══════════════════════════════════════════════════════════════════════════════


class TestShadowLedger:
    """Tests for ShadowLedger accounting dataclass."""

    def test_initial_state(self) -> None:
        """Empty ledger has zero totals and no adjustments."""
        ledger = ShadowLedger()

        assert ledger.total_coupon_gross == Decimal(0)
        assert ledger.total_coupon_net == Decimal(0)
        assert ledger.total_dividend_gross == Decimal(0)
        assert ledger.total_dividend_net == Decimal(0)
        assert ledger.total_tax == Decimal(0)
        assert ledger.total_adjustment == Decimal(0)
        assert ledger.adjustments == []

    def test_add_coupon(self) -> None:
        """Records a coupon with correct tax (13% NDFL) calculation."""
        ledger = ShadowLedger()
        gross = Decimal(5625)  # 100 bonds * 56.25 per bond

        adj = ledger.add_coupon("SU26244RMFS2", date(2026, 3, 15), gross)

        assert adj is not None
        expected_tax = gross * _TAX_RATE_13
        expected_net = gross - expected_tax
        assert adj.gross_amount == gross
        assert adj.net_amount == expected_net
        assert adj.tax == expected_tax
        assert adj.type == "coupon"
        assert adj.symbol == "SU26244RMFS2"
        assert ledger.total_coupon_gross == gross
        assert ledger.total_coupon_net == expected_net
        assert ledger.total_tax == expected_tax

    def test_add_coupon_idempotent(self) -> None:
        """Same (symbol, date) pair returns None second time."""
        ledger = ShadowLedger()
        gross = Decimal(5625)
        payment_date = date(2026, 3, 15)

        first = ledger.add_coupon("SU26244RMFS2", payment_date, gross)
        second = ledger.add_coupon("SU26244RMFS2", payment_date, gross)

        assert first is not None
        assert second is None
        assert len(ledger.adjustments) == 1
        assert ledger.total_coupon_gross == gross  # not doubled

    def test_add_dividend(self) -> None:
        """Records a dividend with correct tax calculation."""
        ledger = ShadowLedger()
        gross = Decimal(333)  # 10 shares * 33.30

        adj = ledger.add_dividend("SBER", date(2026, 7, 15), gross)

        assert adj is not None
        expected_tax = gross * _TAX_RATE_13
        expected_net = gross - expected_tax
        assert adj.gross_amount == gross
        assert adj.net_amount == expected_net
        assert adj.tax == expected_tax
        assert adj.type == "dividend"
        assert ledger.total_dividend_gross == gross
        assert ledger.total_dividend_net == expected_net

    def test_add_dividend_idempotent(self) -> None:
        """Same (symbol, date) pair returns None second time."""
        ledger = ShadowLedger()
        gross = Decimal(333)
        ex_date = date(2026, 7, 15)

        first = ledger.add_dividend("SBER", ex_date, gross)
        second = ledger.add_dividend("SBER", ex_date, gross)

        assert first is not None
        assert second is None
        assert len(ledger.adjustments) == 1

    def test_total_adjustment(self) -> None:
        """total_adjustment = coupon_net + dividend_net."""
        ledger = ShadowLedger()
        coupon_gross = Decimal(5625)
        dividend_gross = Decimal(333)

        ledger.add_coupon("BOND1", date(2026, 3, 15), coupon_gross)
        ledger.add_dividend("SBER", date(2026, 7, 15), dividend_gross)

        coupon_net = coupon_gross * (1 - _TAX_RATE_13)
        dividend_net = dividend_gross * (1 - _TAX_RATE_13)
        expected = coupon_net + dividend_net
        assert ledger.total_adjustment == expected

    def test_multiple_adjustments(self) -> None:
        """Several coupons and dividends accumulate correctly."""
        ledger = ShadowLedger()

        # Two bond coupons on different dates
        ledger.add_coupon("BOND1", date(2026, 3, 15), Decimal(1000))
        ledger.add_coupon("BOND1", date(2026, 9, 15), Decimal(1000))
        # One coupon for a different bond
        ledger.add_coupon("BOND2", date(2026, 3, 15), Decimal(500))
        # One dividend
        ledger.add_dividend("SBER", date(2026, 7, 15), Decimal(300))

        assert len(ledger.adjustments) == 4
        assert ledger.total_coupon_gross == Decimal(2500)
        assert ledger.total_dividend_gross == Decimal(300)
        total_gross = Decimal(2800)
        expected_tax = total_gross * _TAX_RATE_13
        assert ledger.total_tax == expected_tax

    def test_custom_tax_rate(self) -> None:
        """Non-13% tax rate works for both coupons and dividends."""
        ledger = ShadowLedger()
        gross = Decimal(1000)

        adj = ledger.add_coupon("BOND1", date(2026, 3, 15), gross, tax_rate=_TAX_RATE_15)

        assert adj is not None
        expected_tax = gross * _TAX_RATE_15
        assert adj.tax == expected_tax
        assert adj.net_amount == gross - expected_tax


# ═══════════════════════════════════════════════════════════════════════════════
# TestSandboxPortfolioTracker
# ═══════════════════════════════════════════════════════════════════════════════


class TestSandboxPortfolioTracker:
    """Tests for SandboxPortfolioTracker broker wrapper."""

    def test_submit_order_forwarded(self) -> None:
        """submit_order delegates to inner broker."""
        broker = _mock_broker()
        tracker = SandboxPortfolioTracker(broker=broker)
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))

        result = tracker.submit_order(order)

        broker.submit_order.assert_called_once_with(order, None)
        assert result.filled is True

    def test_has_position_forwarded(self) -> None:
        """has_position delegates to inner broker."""
        broker = _mock_broker()
        tracker = SandboxPortfolioTracker(broker=broker)

        tracker.has_position("SBER")

        broker.has_position.assert_called_once_with("SBER")

    def test_cancel_order_forwarded(self) -> None:
        """cancel_order delegates to inner broker."""
        broker = _mock_broker()
        tracker = SandboxPortfolioTracker(broker=broker)

        tracker.cancel_order("ord-123")

        broker.cancel_order.assert_called_once_with("ord-123")

    def test_shadow_portfolio_no_adjustments(self) -> None:
        """Without processing, shadow portfolio equals sandbox portfolio."""
        broker = _mock_broker(equity=_DEFAULT_EQUITY, cash=_DEFAULT_CASH)
        tracker = SandboxPortfolioTracker(broker=broker)

        shadow = tracker.shadow_portfolio()

        assert shadow.equity == _DEFAULT_EQUITY
        assert shadow.cash == _DEFAULT_CASH

    def test_shadow_portfolio_with_coupon(self) -> None:
        """After coupon processing, equity increases by net coupon amount."""
        positions = {"SU26244RMFS2": _BOND_QTY}
        broker = _mock_broker(positions=positions, equity=_DEFAULT_EQUITY)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_coupon_schedule("SU26244RMFS2", [_make_coupon(coupon_date=date(2026, 3, 15))])

        tracker.process_daily(date(2026, 3, 15))
        shadow = tracker.shadow_portfolio()

        gross = _COUPON_AMOUNT * _BOND_QTY
        net = gross - gross * _TAX_RATE_13
        assert shadow.equity == _DEFAULT_EQUITY + net

    def test_shadow_portfolio_with_dividend(self) -> None:
        """After dividend processing, equity increases by net dividend amount."""
        positions = {"SBER": _STOCK_QTY}
        broker = _mock_broker(positions=positions, equity=_DEFAULT_EQUITY)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_dividend_calendar("SBER", [_make_dividend(ex_date=date(2026, 7, 15))])

        tracker.process_daily(date(2026, 7, 15))
        shadow = tracker.shadow_portfolio()

        gross = _DIVIDEND_AMOUNT * _STOCK_QTY
        net = gross - gross * _TAX_RATE_13
        assert shadow.equity == _DEFAULT_EQUITY + net

    def test_process_daily_idempotent(self) -> None:
        """Calling process_daily twice for the same date returns empty list second time."""
        positions = {"SU26244RMFS2": _BOND_QTY}
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_coupon_schedule("SU26244RMFS2", [_make_coupon(coupon_date=date(2026, 3, 15))])

        first = tracker.process_daily(date(2026, 3, 15))
        second = tracker.process_daily(date(2026, 3, 15))

        assert len(first) == 1
        assert len(second) == 0

    def test_process_daily_no_position(self) -> None:
        """No adjustment if we do not hold the bond."""
        broker = _mock_broker(positions={})  # empty portfolio
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_coupon_schedule("SU26244RMFS2", [_make_coupon(coupon_date=date(2026, 3, 15))])

        adjustments = tracker.process_daily(date(2026, 3, 15))

        assert len(adjustments) == 0

    def test_equity_discrepancy(self) -> None:
        """equity_discrepancy returns ledger.total_adjustment value."""
        positions = {"SBER": _STOCK_QTY}
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_dividend_calendar("SBER", [_make_dividend(ex_date=date(2026, 7, 15))])

        tracker.process_daily(date(2026, 7, 15))

        gross = _DIVIDEND_AMOUNT * _STOCK_QTY
        net = gross - gross * _TAX_RATE_13
        assert tracker.equity_discrepancy == net

    def test_figi_resolution(self) -> None:
        """Positions keyed by FIGI are matched to symbol-based coupon schedules."""
        # Sandbox positions use FIGI keys, but coupon schedules are loaded by symbol
        figi_positions = {_BOND_FIGI: _BOND_QTY}
        broker = _mock_broker(positions=figi_positions)
        registry = _make_registry()
        tracker = SandboxPortfolioTracker(broker=broker, registry=registry)

        # Load coupon schedule by symbol
        tracker.load_coupon_schedule("SU26244RMFS2", [_make_coupon(coupon_date=date(2026, 3, 15))])

        adjustments = tracker.process_daily(date(2026, 3, 15))

        # Should find the position via FIGI resolution
        assert len(adjustments) == 1
        gross = _COUPON_AMOUNT * _BOND_QTY
        assert adjustments[0].gross_amount == gross


# ═══════════════════════════════════════════════════════════════════════════════
# TestDailyProcessingScenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestDailyProcessingScenarios:
    """Scenario-based tests for daily processing logic."""

    def test_bond_coupon_on_payment_date(self) -> None:
        """Coupon is paid on the exact coupon_date."""
        positions = {"SU26244RMFS2": Decimal(50)}
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_coupon_schedule(
            "SU26244RMFS2",
            [_make_coupon(coupon_date=date(2026, 3, 15), amount=Decimal("56.25"))],
        )

        adjustments = tracker.process_daily(date(2026, 3, 15))

        assert len(adjustments) == 1
        expected_gross = Decimal("56.25") * Decimal(50)
        assert adjustments[0].gross_amount == expected_gross
        assert adjustments[0].type == "coupon"

    def test_bond_coupon_not_on_payment_date(self) -> None:
        """No coupon is paid on a non-payment date."""
        positions = {"SU26244RMFS2": Decimal(50)}
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)
        tracker.load_coupon_schedule(
            "SU26244RMFS2",
            [_make_coupon(coupon_date=date(2026, 3, 15))],
        )

        # Process on a different date
        adjustments = tracker.process_daily(date(2026, 3, 14))

        assert len(adjustments) == 0

    def test_multiple_bonds_same_day(self) -> None:
        """Two bonds with coupons on the same date both get processed."""
        positions = {
            "BOND_A": Decimal(100),
            "BOND_B": Decimal(50),
        }
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)

        payment_date = date(2026, 3, 15)
        tracker.load_coupon_schedule(
            "BOND_A",
            [
                _make_coupon(
                    figi="FIGI_A",
                    coupon_date=payment_date,
                    amount=Decimal("40.00"),
                )
            ],
        )
        tracker.load_coupon_schedule(
            "BOND_B",
            [
                _make_coupon(
                    figi="FIGI_B",
                    coupon_date=payment_date,
                    amount=Decimal("30.00"),
                )
            ],
        )

        adjustments = tracker.process_daily(payment_date)

        assert len(adjustments) == 2
        symbols = {adj.symbol for adj in adjustments}
        assert symbols == {"BOND_A", "BOND_B"}
        # Check gross amounts
        gross_a = Decimal("40.00") * Decimal(100)
        gross_b = Decimal("30.00") * Decimal(50)
        total_gross = sum(adj.gross_amount for adj in adjustments)
        assert total_gross == gross_a + gross_b

    def test_mixed_bonds_and_equities(self) -> None:
        """Coupons and dividends are processed together on the same date."""
        target_date = date(2026, 7, 15)
        positions = {
            "SU26244RMFS2": Decimal(100),
            "SBER": Decimal(10),
        }
        broker = _mock_broker(positions=positions)
        tracker = SandboxPortfolioTracker(broker=broker)

        tracker.load_coupon_schedule(
            "SU26244RMFS2",
            [_make_coupon(coupon_date=target_date, amount=Decimal("56.25"))],
        )
        tracker.load_dividend_calendar(
            "SBER",
            [_make_dividend(ex_date=target_date, amount=Decimal("33.30"))],
        )

        adjustments = tracker.process_daily(target_date)

        assert len(adjustments) == 2
        types = {adj.type for adj in adjustments}
        assert types == {"coupon", "dividend"}

        coupon_adj = next(a for a in adjustments if a.type == "coupon")
        dividend_adj = next(a for a in adjustments if a.type == "dividend")
        assert coupon_adj.gross_amount == Decimal("56.25") * Decimal(100)
        assert dividend_adj.gross_amount == Decimal("33.30") * Decimal(10)
