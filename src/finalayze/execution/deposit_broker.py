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
tranche at the as-of deposit rate with no penalty (R-4 -- the honest no-W2
W1 default; the W2 wave decision that would TRIGGER a break/roll is out of
scope here, D-06).

``split_across_banks`` / ``uninsured_excess`` (DEP-03) are pure ASV sizing
functions: they take a principal and return per-bank insured slices, flagging
accrued-over-cap exposure as not-risk-free (D-09). They read no risk profiles
or portfolio weights (R-5 boundary discipline -- no W2 sizing-layer coupling).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from dateutil.relativedelta import relativedelta

from finalayze.core.constants import (
    ASV_CAP_PER_BANK,
    DEPOSIT_DEMAND_RATE,
    DEPOSIT_FLOOR_BASE,
    NDFL_RATE,
)
from finalayze.core.ndfl import ndfl_on_deposit_interest
from finalayze.core.schemas import BankAllocation, DepositTranche
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
        # WR-04: dates already accrued -- a repeated calendar date (sub-daily
        # timeframe, >1 bar per day) must NOT compound the deposit twice in one day.
        self._processed_accrual_dates: set[date] = set()

    def accrue(self, current_date: date) -> Decimal:
        """Accrue one bar of net-of-NDFL interest for every live tranche.

        Daily-compounding convention (codebase-wide,
        ``portfolio_orchestrator.py``): ``(1 + annual) ** (1/252) - 1``. The
        gross is taxed via :func:`ndfl_on_deposit_interest` against the YTD
        running-max floor (R-2), and the NET is folded into the tranche
        ``accrued_net`` -- i.e. the deposit's interest stays INSIDE the deposit
        (mark-only, CR-01) and itself earns interest (true compounding: the next
        bar's gross is computed on ``principal + accrued_net``, not the flat
        principal). It is NOT swept to ``self._cash`` -- that would
        double-represent it because :meth:`deposit_value` already carries the
        accrued mark, overstating ``cash + deposit_value()`` by the accrued net
        and leaving a pre-maturity :meth:`break_tranche` unable to claw it back.
        Returns the total net interest accrued on this bar (for callers who want
        the bar's credit; they must not ALSO read it from the mark). Accrual is
        idempotent per calendar date: a repeated date (a multi-bar day on a
        sub-daily timeframe) is a no-op, so the deposit never compounds twice in
        one day (WR-04).
        """
        # WR-04: idempotency guard -- a date already accrued does not compound again.
        if current_date in self._processed_accrual_dates:
            return Decimal(0)
        self._processed_accrual_dates.add(current_date)

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
            # True compounding: the accrued net stays in the deposit and earns
            # interest, so the base is (principal + accrued_net), not flat principal.
            gross = (tranche.principal + tranche.accrued_net) * daily
            tax = ndfl_on_deposit_interest(gross, self._ytd_deposit_gross, running_floor)
            net = gross - tax
            # Mark-only (CR-01): fold net into the tranche mark; do NOT sweep to
            # self._cash. deposit_value() already carries accrued_net, so sweeping
            # would double-count it (cash + mark) and a pre-maturity break could
            # not claw the swept interest back.
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
        fiction (Pitfall 2 / T-71-10). The W2 wave decision that TRIGGERS a
        break is out of scope here (D-06); this method only models the penalty.

        ``current_date`` is RETAINED but W1-inert: the penalty depends only on the
        principal, so the bar date does not change the mark in W1 (WR-05). It is
        kept in the signature because W2 (where the date-aware rebalance decision
        triggers the break) will need it; deleting it now and re-adding it later
        would churn every caller.
        """
        del current_date  # W1-inert: the penalty does not depend on the bar date (WR-05)
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


def split_across_banks(
    total_principal: Decimal, cap: Decimal = ASV_CAP_PER_BANK
) -> list[BankAllocation]:
    """Greedily distribute principal across banks so each stays <= the insured cap.

    Pure sizing (DEP-03 / R-5): takes a principal, returns per-bank allocations.
    Reads no risk profiles, portfolio weights or other asset classes -- no W2
    sizing-layer coupling. At the 2.5M RUB default capital (D-08) this yields 2 banks
    (1.4M + 1.1M): the split IS exercised. Each bank's principal is capped at
    ``cap`` (default ``ASV_CAP_PER_BANK``).
    """
    n_banks = max(1, math.ceil(total_principal / cap))
    allocations: list[BankAllocation] = []
    remaining = total_principal
    for i in range(n_banks):
        principal = min(remaining, cap)
        allocations.append(BankAllocation(bank_id=f"bank_{i}", principal=principal))
        remaining -= principal
    return allocations


def uninsured_excess(bank: BankAllocation, cap: Decimal = ASV_CAP_PER_BANK) -> Decimal:
    """Insured exposure above the cap, flagged not-risk-free (D-09).

    Accrued interest counts toward the cap (``insured_exposure = principal +
    accrued_net``), so a bank at the cap exceeds it as interest accrues. W1 only
    FLAGS the excess; the W2 sizing wave is what would act on it (R-5 boundary).
    """
    return max(Decimal(0), bank.insured_exposure - cap)
