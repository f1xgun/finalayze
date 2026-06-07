"""RED scaffold: SAA-02/03/04 + D-13 AllocationOrchestrator (Phase 72 Wave-0).

Pins the L5 ``AllocationOrchestrator`` + ``AllocationResult`` contract before it
exists:
- SAA-02: the rebalance fires on a pure quarterly calendar -- at most 4 times/yr,
  each at the first bar of a quarter (D-08);
- SAA-03 / D-09: a traded eq/ofz leg is charged ``sum|Δvalue| * round_trip`` (the
  MOEX_RETAIL_COSTS Investor round-trip), the deposit leg is cost-free, and the
  charge is an EXPLICIT ``rebalance_cost`` line item on the result (not buried in
  the curve);
- SAA-04: the equity sleeve is the passive MCFTR series -- NO combiner / strategy
  / momentum / ML import appears in the allocation path (closed-alpha invariant);
- D-13 / R-5 / A4: the 3-way merge at deposit=0, under the LEGACY monthly+drift
  cadence and ZERO cost, reproduces the legacy 60/40 curve (STRUCTURAL
  EQUIVALENCE -- NOT a byte-match of the live quarterly+cost SAA path);
- D-12: the orchestrator consumes pre-computed curves and never re-runs an engine.

All curves are tiny in-memory ``(date, Decimal)`` fixtures -- no live engine/API.

RED now: ``finalayze.orchestration.allocation`` (AllocationOrchestrator /
AllocationResult, Plan 05) + ``finalayze.core.schemas`` (RiskProfile / AssetClass,
Plan 02) do not exist yet.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.orchestration.allocation import AllocationOrchestrator, AllocationResult

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

_YEAR = 2023
_DAYS_IN_YEAR = 365
_EXPECTED_QUARTERLY_REBALANCES = 4
_MONTHS_PER_QUARTER = 3

# MOEX_RETAIL_COSTS Investor round-trip = 2 * (commission_rate + (spread+slip)/1e4)
#   per side = 0.003 + (15 + 10)/10000 = 0.003 + 0.0025 = 0.0055
#   round-trip = 0.011 (~1.10%, the verified Investor cost).
_PER_SIDE_COST = Decimal("0.0055")
_ROUND_TRIP_COST = _PER_SIDE_COST * Decimal(2)

_ZERO = Decimal(0)

# Tiny in-memory total-return curves (flat -> trivially mergeable; the spine,
# trigger and cost line are what these tests pin, not curve dynamics).
_FLAT_LEVEL = Decimal(100)
_EQUITY_LEVEL = Decimal(120)
_OFZ_LEVEL = Decimal(110)

# A known traded delta on the equity + ofz legs at a forced rebalance.
_EQ_DELTA = Decimal(10_000)
_OFZ_DELTA = Decimal(5_000)
_EXPECTED_COST = (_EQ_DELTA + _OFZ_DELTA) * _ROUND_TRIP_COST

# Forbidden active-selection substrings (SAA-04 closed-alpha import-guard).
_FORBIDDEN_IMPORTS = ("combiner", "StrategyCombiner", "momentum", "strategies.ml")

_ALLOCATION_SRC = Path("src/finalayze/orchestration/allocation.py")


def _daily_index(year: int, days: int) -> list[date]:
    start = date(year, 1, 1)
    return [start + timedelta(days=i) for i in range(days)]


def _flat_series(dates: list[date], level: Decimal) -> list[tuple[date, Decimal]]:
    return [(d, level) for d in dates]


def test_quarterly_trigger() -> None:
    """The rebalance fires at most 4x/yr, each at a quarter boundary (SAA-02 / D-08)."""
    dates = _daily_index(_YEAR, _DAYS_IN_YEAR)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result: AllocationResult = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
    )
    rebalance_dates = result.rebalance_dates
    assert len(rebalance_dates) <= _EXPECTED_QUARTERLY_REBALANCES
    # Each rebalance opens a new quarter: its quarter index differs from the
    # prior bar's -- (month - 1) // 3 boundary change.
    for when in rebalance_dates:
        assert (when.month - 1) % _MONTHS_PER_QUARTER == 0


def test_rebalance_cost_line_item() -> None:
    """A traded eq/ofz leg is charged round-trip cost as an EXPLICIT line item (SAA-03 / D-09)."""
    dates = _daily_index(_YEAR, _DAYS_IN_YEAR)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
        forced_leg_deltas={AssetClass.EQUITY: _EQ_DELTA, AssetClass.OFZ_PK: _OFZ_DELTA},
    )
    # The cost is the round-trip charge on the traded eq + ofz value; deposit free.
    assert result.rebalance_cost == _EXPECTED_COST
    assert result.rebalance_cost > _ZERO  # an explicit, non-zero field on the result

    # A deposit-only delta contributes nothing to the cost (deposit leg cost-free).
    deposit_only = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
        forced_leg_deltas={AssetClass.DEPOSIT: _EQ_DELTA},
    )
    assert deposit_only.rebalance_cost == _ZERO


def test_equity_sleeve_passive() -> None:
    """No active-selection module is imported through the allocation path (SAA-04).

    The equity leg is the MCFTR series only; the closed-alpha invariant forbids
    any combiner / strategy / momentum / ML import in the allocation source.
    """
    source = _ALLOCATION_SRC.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_IMPORTS:
        assert forbidden not in source


def test_60_40_deposit_zero_reproduction() -> None:
    """3-way at deposit=0 reproduces the legacy 60/40 curve (D-13 / R-5 / A4).

    STRUCTURAL EQUIVALENCE per A4: run the merge spine with deposit weight = 0,
    the SAME legacy MONTHLY+drift>0.05 cadence and ZERO rebalance cost, and assert
    the merged curve equals the legacy 2-way 60/40 merged curve to the kopeck.
    This is NOT a byte-match of the live quarterly+cost SAA path (impossible by
    construction -- A4); do NOT run the quarterly+cost path here.
    """
    dates = _daily_index(_YEAR, _DAYS_IN_YEAR)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result = orch.run(
        deposit_curve=_flat_series(dates, _ZERO),  # deposit weight collapsed to 0
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
        legacy_monthly_drift_cadence=True,  # A4: the LEGACY cadence
        zero_cost=True,  # A4: ZERO cost for the structural-equivalence comparison
    )
    legacy = orch.reproduce_legacy_60_40(
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
    )
    assert result.merged_equity_curve == legacy


def test_never_reruns_engines() -> None:
    """The orchestrator consumes pre-computed curves and runs no engine (D-12)."""
    dates = _daily_index(_YEAR, _DAYS_IN_YEAR)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    # run() takes plain (date, Decimal) series -- never a BacktestEngine instance.
    result = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
    )
    assert len(result.merged_equity_curve) == len(dates)
    assert result.dates == dates
