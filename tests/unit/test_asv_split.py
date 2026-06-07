"""RED scaffold: ASV multi-bank split + uninsured-excess flag (DEP-03 / R-5).

Pins the deposit-sleeve ASV sizing contract before it exists:
- ``split_across_banks(2.5M RUB)`` -> 2 banks (1.4M + 1.1M), each <= the per-bank
  insured cap (D-07 / D-08: the default capital makes the 1.4M cap bite);
- accrued interest counts toward the cap (D-09), so a bank at the cap with any
  accrued interest exposes ``uninsured_excess`` > 0 (flagged not-risk-free);
- under-cap principal needs a single bank with zero uninsured excess.

RED now: ``split_across_banks`` / ``uninsured_excess`` (Plan 04) and
``BankAllocation`` / ``ASV_CAP_PER_BANK`` (Plan 02) do not exist yet.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.constants import ASV_CAP_PER_BANK
from finalayze.core.schemas import BankAllocation
from finalayze.execution.deposit_broker import split_across_banks, uninsured_excess

# ── Constants (named -- no magic numbers) ───────────────────────────────────

_DEFAULT_CAPITAL = Decimal(2_500_000)  # D-08 default target capital -> exercises the split
_SECOND_BANK_PRINCIPAL = Decimal(1_100_000)  # 2.5M - 1.4M cap
_UNDER_CAP_PRINCIPAL = Decimal(1_000_000)  # below the cap -> single bank
_ACCRUED_OVER_CAP = Decimal(5_000)  # interest that pushes a full bank over the cap
_TWO_BANKS = 2
_ONE_BANK = 1
_ZERO = Decimal(0)
_BANK_ID = "bank_0"


def test_split_2_500_000_two_banks() -> None:
    """2.5M RUB splits into two insured banks: 1.4M cap + 1.1M remainder (DEP-03)."""
    allocs = split_across_banks(_DEFAULT_CAPITAL)
    assert len(allocs) == _TWO_BANKS
    assert allocs[0].principal == ASV_CAP_PER_BANK
    assert allocs[1].principal == _SECOND_BANK_PRINCIPAL
    for alloc in allocs:
        assert alloc.principal <= ASV_CAP_PER_BANK


def test_accrued_counts_toward_cap_flags_excess() -> None:
    """Accrued interest at a full bank pushes insured exposure over the cap (D-09)."""
    bank = BankAllocation(
        bank_id=_BANK_ID,
        principal=ASV_CAP_PER_BANK,
        accrued_net=_ACCRUED_OVER_CAP,
    )
    assert uninsured_excess(bank) == _ACCRUED_OVER_CAP


def test_under_cap_single_bank() -> None:
    """Principal below the cap needs one bank with zero uninsured excess."""
    allocs = split_across_banks(_UNDER_CAP_PRINCIPAL)
    assert len(allocs) == _ONE_BANK
    assert uninsured_excess(allocs[0]) == _ZERO
