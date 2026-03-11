"""Sandbox portfolio tracker -- wraps TinkoffBroker with self-computed income.

Tinkoff Sandbox limitations:
- No coupon payments on bonds
- No dividend payments on stocks
- No tax calculation
- Simplified execution (market orders at last price)

This tracker wraps TinkoffBroker and adds a shadow accounting layer that:
1. Forwards all orders to sandbox TinkoffBroker
2. Self-computes coupon income from known coupon schedules
3. Self-computes dividend income from known dividend calendar
4. Tracks NDFL tax (13%) on coupon income
5. Maintains shadow_equity = sandbox_equity + cumulative_adjustments
6. Logs discrepancies for monitoring

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

if TYPE_CHECKING:
    from datetime import date

    from finalayze.core.schemas import Candle, CouponPayment
    from finalayze.execution.tinkoff_broker import TinkoffBroker
    from finalayze.markets.instruments import InstrumentRegistry

logger = structlog.get_logger(__name__)

_NDFL_RATE = Decimal("0.13")


@dataclass
class DividendEntry:
    """A known dividend payment."""

    symbol: str
    ex_date: date
    amount_per_share: Decimal


@dataclass(frozen=True)
class SandboxAdjustment:
    """A single income/tax adjustment not provided by sandbox."""

    date: date
    type: str  # "coupon", "dividend", "ndfl_tax"
    symbol: str
    gross_amount: Decimal
    net_amount: Decimal  # after tax for coupons
    tax: Decimal
    description: str


@dataclass
class ShadowLedger:
    """Shadow accounting ledger tracking sandbox adjustments."""

    adjustments: list[SandboxAdjustment] = field(default_factory=list)

    # Cumulative totals
    total_coupon_gross: Decimal = field(default=Decimal(0))
    total_coupon_net: Decimal = field(default=Decimal(0))
    total_dividend_gross: Decimal = field(default=Decimal(0))
    total_dividend_net: Decimal = field(default=Decimal(0))
    total_tax: Decimal = field(default=Decimal(0))

    # Processed dates (to avoid double-processing)
    _processed_coupon_dates: set[tuple[str, date]] = field(default_factory=set)
    _processed_dividend_dates: set[tuple[str, date]] = field(default_factory=set)

    @property
    def total_adjustment(self) -> Decimal:
        """Total cash adjustment to add to sandbox equity."""
        return self.total_coupon_net + self.total_dividend_net

    def add_coupon(
        self,
        symbol: str,
        payment_date: date,
        gross: Decimal,
        tax_rate: Decimal = _NDFL_RATE,
    ) -> SandboxAdjustment | None:
        """Record a coupon payment. Returns adjustment or None if already processed."""
        key = (symbol, payment_date)
        if key in self._processed_coupon_dates:
            return None
        self._processed_coupon_dates.add(key)

        tax = gross * tax_rate
        net = gross - tax
        self.total_coupon_gross += gross
        self.total_coupon_net += net
        self.total_tax += tax

        adj = SandboxAdjustment(
            date=payment_date,
            type="coupon",
            symbol=symbol,
            gross_amount=gross,
            net_amount=net,
            tax=tax,
            description=f"Coupon {symbol}: {gross} gross, {net} net (NDFL {tax})",
        )
        self.adjustments.append(adj)
        return adj

    def add_dividend(
        self,
        symbol: str,
        ex_date: date,
        gross: Decimal,
        tax_rate: Decimal = _NDFL_RATE,
    ) -> SandboxAdjustment | None:
        """Record a dividend payment. Returns adjustment or None if already processed."""
        key = (symbol, ex_date)
        if key in self._processed_dividend_dates:
            return None
        self._processed_dividend_dates.add(key)

        tax = gross * tax_rate
        net = gross - tax
        self.total_dividend_gross += gross
        self.total_dividend_net += net
        self.total_tax += tax

        adj = SandboxAdjustment(
            date=ex_date,
            type="dividend",
            symbol=symbol,
            gross_amount=gross,
            net_amount=net,
            tax=tax,
            description=f"Dividend {symbol}: {gross} gross, {net} net (NDFL {tax})",
        )
        self.adjustments.append(adj)
        return adj


class SandboxPortfolioTracker(BrokerBase):
    """Wraps TinkoffBroker (sandbox) with self-computed coupon/dividend income.

    All BrokerBase methods are forwarded to the inner TinkoffBroker.
    Additionally provides:
    - process_daily(current_date): process coupons and dividends for today
    - shadow_portfolio(): PortfolioState with corrected equity
    - ledger: full adjustment history

    Usage::

        fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=True)
        broker = TinkoffBroker(token=token, registry=registry, sandbox=True)
        tracker = SandboxPortfolioTracker(broker=broker, registry=registry)

        # Load coupon schedules for bonds we'll trade
        tracker.load_coupon_schedule("SU26244RMFS2", coupon_payments)

        # Load dividend calendar for equities
        tracker.load_dividend_calendar("SBER", [DividendEntry(...)])

        # Daily loop
        for day in trading_days:
            adjustments = tracker.process_daily(day)
            shadow = tracker.shadow_portfolio()
            # shadow.equity = sandbox equity + accumulated coupon/dividend income
    """

    def __init__(
        self,
        broker: TinkoffBroker,
        registry: InstrumentRegistry | None = None,
        tax_rate: Decimal = _NDFL_RATE,
    ) -> None:
        self._broker = broker
        self._registry = registry
        self._tax_rate = tax_rate
        self._ledger = ShadowLedger()

        # Coupon schedules: symbol -> list of CouponPayment
        self._coupon_schedules: dict[str, list[CouponPayment]] = {}

        # Dividend calendars: symbol -> list of DividendEntry
        self._dividend_calendars: dict[str, list[DividendEntry]] = {}

        # Track last processed date to prevent double-processing
        self._last_processed_date: date | None = None

    # -- Data loading ----------------------------------------------------------

    def load_coupon_schedule(self, symbol: str, coupons: list[CouponPayment]) -> None:
        """Load coupon payment schedule for a bond symbol."""
        self._coupon_schedules[symbol] = sorted(coupons, key=lambda c: c.coupon_date)
        logger.info("sandbox_tracker.coupons_loaded", symbol=symbol, count=len(coupons))

    def load_dividend_calendar(self, symbol: str, dividends: list[DividendEntry]) -> None:
        """Load dividend payment calendar for an equity symbol."""
        self._dividend_calendars[symbol] = sorted(dividends, key=lambda d: d.ex_date)
        logger.info("sandbox_tracker.dividends_loaded", symbol=symbol, count=len(dividends))

    # -- Daily processing ------------------------------------------------------

    def process_daily(self, current_date: date) -> list[SandboxAdjustment]:
        """Process all coupon and dividend payments for the given date.

        Call this once per trading day. Returns list of adjustments made.
        Idempotent -- calling twice for the same date is safe (returns empty).
        """
        adjustments: list[SandboxAdjustment] = []

        # Get current positions from sandbox
        positions = self._broker.get_positions()

        # Process coupons for held bond positions
        for symbol, coupons in self._coupon_schedules.items():
            qty = self._resolve_position_qty(symbol, positions)
            if qty <= 0:
                continue

            for coupon in coupons:
                if coupon.coupon_date == current_date:
                    gross = coupon.amount_per_bond * qty
                    adj = self._ledger.add_coupon(symbol, current_date, gross, self._tax_rate)
                    if adj is not None:
                        adjustments.append(adj)
                        logger.info(
                            "sandbox_tracker.coupon_processed",
                            symbol=symbol,
                            date=str(current_date),
                            gross=str(gross),
                            net=str(adj.net_amount),
                        )

        # Process dividends for held equity positions
        for symbol, dividends in self._dividend_calendars.items():
            qty = self._resolve_position_qty(symbol, positions)
            if qty <= 0:
                continue

            for div in dividends:
                if div.ex_date == current_date:
                    gross = div.amount_per_share * qty
                    adj = self._ledger.add_dividend(symbol, current_date, gross, self._tax_rate)
                    if adj is not None:
                        adjustments.append(adj)
                        logger.info(
                            "sandbox_tracker.dividend_processed",
                            symbol=symbol,
                            date=str(current_date),
                            gross=str(gross),
                            net=str(adj.net_amount),
                        )

        self._last_processed_date = current_date
        return adjustments

    def _resolve_position_qty(self, symbol: str, positions: dict[str, Decimal]) -> Decimal:
        """Resolve position quantity -- sandbox uses FIGI keys, we may have symbols."""
        # Direct match (symbol or FIGI already in positions)
        if symbol in positions:
            return positions[symbol]

        # Try to resolve symbol -> FIGI via registry
        if self._registry is not None:
            try:
                instrument = self._registry.get(symbol, "moex")
                if instrument.figi and instrument.figi in positions:
                    return positions[instrument.figi]
            except Exception:
                logger.debug("sandbox_tracker.figi_resolution_failed", symbol=symbol)

        return Decimal(0)

    # -- Shadow portfolio ------------------------------------------------------

    def shadow_portfolio(self) -> PortfolioState:
        """Return portfolio state with shadow adjustments applied.

        shadow_equity = sandbox_equity + total_coupon_net + total_dividend_net
        """
        sandbox_state = self._broker.get_portfolio()
        adjustment = self._ledger.total_adjustment

        return PortfolioState(
            cash=sandbox_state.cash + adjustment,
            positions=sandbox_state.positions,
            equity=sandbox_state.equity + adjustment,
            timestamp=sandbox_state.timestamp,
        )

    @property
    def ledger(self) -> ShadowLedger:
        """Return the shadow accounting ledger."""
        return self._ledger

    @property
    def sandbox_equity(self) -> Decimal:
        """Return raw sandbox equity (without adjustments)."""
        return self._broker.get_portfolio().equity

    @property
    def shadow_equity(self) -> Decimal:
        """Return adjusted equity (sandbox + coupon/dividend income)."""
        return self.shadow_portfolio().equity

    @property
    def equity_discrepancy(self) -> Decimal:
        """Return the discrepancy: shadow - sandbox."""
        return self._ledger.total_adjustment

    # -- BrokerBase interface (forwarded to inner broker) ----------------------

    def submit_order(
        self,
        order: OrderRequest,
        fill_candle: Candle | None = None,
    ) -> OrderResult:
        """Forward order to sandbox TinkoffBroker."""
        return self._broker.submit_order(order, fill_candle)

    def get_portfolio(self) -> PortfolioState:
        """Return shadow portfolio (with adjustments)."""
        return self.shadow_portfolio()

    def has_position(self, symbol: str) -> bool:
        """Forward to sandbox TinkoffBroker."""
        return self._broker.has_position(symbol)

    def get_positions(self) -> dict[str, Decimal]:
        """Forward to sandbox TinkoffBroker."""
        return self._broker.get_positions()

    def cancel_order(self, order_id: str) -> None:
        """Forward to sandbox TinkoffBroker."""
        self._broker.cancel_order(order_id)
