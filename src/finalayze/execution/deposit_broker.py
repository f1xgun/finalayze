"""Deposit-sleeve simulated broker (Layer 5).

Subclass of :class:`SimulatedBroker` that models a first-class, risk-free,
allocatable deposit asset for backtesting -- mirroring the shape of
:class:`BondSimulatedBroker` (own schedule in ``__init__``, a per-bar accrual
hook that credits net income to cash exactly like ``process_coupons``, and
income-total properties).

The deposit sleeve (operator decisions D-01/D-06/D-07) is a LADDER of
3/6/12-month tranches. Each live tranche accrues daily-compounded interest,
taxed via the L0 running-max-floor NDFL helper (R-2), with the NET amount
credited to cash every bar (ACCT-02). The mark is ``principal + accrued net``
and NEVER reprices on any market-price input (DEP-01 / D-05: zero market risk).
Breaking a tranche before maturity forfeits accrued interest down to the demand
rate (DEP-02 / D-03). A matured tranche auto-rolls into a fresh same-term
tranche at the as-of deposit rate with no penalty (R-4 -- the honest no-allocator
W1 default; the W2 allocator decision that would TRIGGER a break/roll is out of
scope here, D-06).

``split_across_banks`` / ``uninsured_excess`` (DEP-03) are pure ASV sizing
functions: they take a principal and return per-bank insured slices, flagging
accrued-over-cap exposure as not-risk-free (D-09). They read no risk profiles
or target weights (R-5 boundary discipline -- no W2/allocator coupling).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

from finalayze.core.constants import (
    DEPOSIT_DEMAND_RATE,
    DEPOSIT_FLOOR_BASE,
    NDFL_RATE,
)
from finalayze.core.ndfl import ndfl_on_deposit_interest
from finalayze.core.schemas import DepositTranche
from finalayze.data.fetchers.cbr import deposit_rate_as_of
from finalayze.execution.simulated_broker import SimulatedBroker

if TYPE_CHECKING:
    from datetime import date


_TRADING_DAYS = Decimal(252)
_ZERO_SPREAD_PP = Decimal(0)  # raw key-rate fraction for the running-max floor


class DepositSimulatedBroker(SimulatedBroker):
    """Simulated broker for a laddered, risk-free deposit sleeve.

    Holds a ladder of :class:`DepositTranche` rungs (3/6/12-month, D-01). Each
    bar, :meth:`accrue` compounds one trading day of net-of-NDFL interest for
    every live tranche and credits it to cash (the single ``self._cash +=``
    point, mirroring ``BondSimulatedBroker.process_coupons``). The deposit mark
    (:meth:`deposit_value`) is ``principal + accrued net`` and is independent of
    any market price (D-05). :meth:`break_tranche` forfeits accrued interest to
    the demand rate (D-03); :meth:`roll_at_maturity` rolls a matured tranche into
    a fresh same-term tranche at the as-of rate with full accrued kept (R-4).
    """

    def __init__(
        self,
        initial_cash: Decimal,
        tranches: list[DepositTranche],
        tax_rate: Decimal = NDFL_RATE,  # L0 single source (D-12)
    ) -> None:
        super().__init__(initial_cash=initial_cash)
        self._tranches = tranches
        self._tax_rate = tax_rate
        self._total_interest_gross = Decimal(0)
        self._total_interest_net = Decimal(0)
        self._total_tax_paid = Decimal(0)
        # R-2 running-max non-taxable-floor accumulators (look-ahead-safe).
        self._ytd_deposit_gross = Decimal(0)
        self._running_max_key_rate = Decimal(0)
        self._current_year: int | None = None

    def accrue(self, current_date: date) -> Decimal:
        """Accrue one bar of net-of-NDFL interest for every live tranche.

        Daily-compounding convention (codebase-wide,
        ``portfolio_orchestrator.py``): ``(1 + annual) ** (1/252) - 1``. The
        gross is taxed via :func:`ndfl_on_deposit_interest` against the YTD
        running-max floor (R-2), and the NET is credited to cash -- the single
        credit point, mirroring ``process_coupons``. Returns the total net
        interest credited on this bar.
        """
        # R-2 tax-year boundary: reset the YTD/floor accumulators on Jan 1.
        if current_date.year != self._current_year:
            self._ytd_deposit_gross = Decimal(0)
            self._running_max_key_rate = Decimal(0)
            self._current_year = current_date.year

        # Update the running-max key-rate floor from on/before this bar only
        # (look-ahead-safe; monotone-rising). spread_pp=0 -> raw key-rate fraction.
        key_rate_fraction = deposit_rate_as_of(current_date, spread_pp=_ZERO_SPREAD_PP)
        self._running_max_key_rate = max(self._running_max_key_rate, key_rate_fraction)
        running_floor = DEPOSIT_FLOOR_BASE * self._running_max_key_rate

        total_net = Decimal(0)
        for tranche in self._tranches:
            if tranche.broken or tranche.maturity_date < current_date:
                continue
            daily = (Decimal(1) + tranche.annual_rate) ** (Decimal(1) / _TRADING_DAYS) - Decimal(1)
            gross = tranche.principal * daily
            tax = ndfl_on_deposit_interest(gross, self._ytd_deposit_gross, running_floor)
            net = gross - tax
            self._cash += net  # <- THE single credit point (mirrors process_coupons)
            tranche.accrued_gross += gross
            tranche.accrued_net += net
            self._ytd_deposit_gross += gross
            self._total_interest_gross += gross
            self._total_interest_net += net
            self._total_tax_paid += tax
            total_net += net
        return total_net

    def deposit_value(self) -> Decimal:
        """Mark = principal + accrued net interest; independent of market price.

        DEP-01 / D-05: the deposit sleeve carries zero market risk. This is a
        standalone accessor -- it never consults ``self._last_prices``, so any
        ``update_prices`` call leaves the mark untouched.
        """
        return sum((tr.principal + tr.accrued_net for tr in self._tranches), Decimal(0))

    def break_tranche(self, tranche: DepositTranche, current_date: date) -> None:
        """Pre-maturity break: forfeit accrued interest to the demand rate (D-03).

        Marks the tranche broken and resets its accrued net to
        ``principal * DEPOSIT_DEMAND_RATE`` (~0.01%) -- no liquid-cash-at-full-rate
        fiction (Pitfall 2 / T-71-10). The W2 allocator decision that TRIGGERS a
        break is out of scope here (D-06); this method only models the penalty.
        """
        del current_date  # the penalty does not depend on the bar date in W1
        tranche.broken = True
        tranche.accrued_net = tranche.principal * DEPOSIT_DEMAND_RATE

    def roll_at_maturity(self, tranche: DepositTranche, current_date: date) -> DepositTranche:
        """Roll a matured tranche into a fresh same-term tranche at the as-of rate.

        Principal + full accrued net carries forward (no penalty -- maturity
        reached, R-4). The new annual rate is the as-of deposit rate on this bar
        (look-ahead-safe). The roll stays INSIDE the sleeve -- no cross-asset or
        W2 funding-order logic (D-06).
        """
        rolled_principal = tranche.principal + tranche.accrued_net
        rate = deposit_rate_as_of(current_date)
        return DepositTranche(
            principal=rolled_principal,
            term_months=tranche.term_months,
            annual_rate=rate,
            open_date=current_date,
            maturity_date=current_date + relativedelta(months=tranche.term_months),
            accrued_net=Decimal(0),
            accrued_gross=Decimal(0),
        )

    @property
    def interest_income_net(self) -> Decimal:
        """Total net deposit interest credited (after NDFL)."""
        return self._total_interest_net

    @property
    def interest_income_gross(self) -> Decimal:
        """Total gross deposit interest accrued (before NDFL)."""
        return self._total_interest_gross

    @property
    def tax_paid(self) -> Decimal:
        """Total NDFL tax paid on deposit interest."""
        return self._total_tax_paid
