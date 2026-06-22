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

from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.orchestration.allocation import (
    AllocationOrchestrator,
    AllocationResult,
    CostBasisLedger,
)

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
_ONE = Decimal(1)

# Tiny in-memory total-return curves (flat -> trivially mergeable; the spine,
# trigger and cost line are what these tests pin, not curve dynamics).
_FLAT_LEVEL = Decimal(100)
_EQUITY_LEVEL = Decimal(120)
_OFZ_LEVEL = Decimal(110)

# BALANCED *realized* weights in this fixture's year (Phase 76 regime tilt): _YEAR=2023
# has NO CBR cuts, so rate_regime_as_of is high_rate at every boundary and BALANCED tilts
# to its high_rate vector {deposit 0.60 / ofz_pk 0.10 / equity 0.30}
# (config/allocation_profiles.yaml regime_weights.high_rate.balanced). The base static
# vector (0.45/0.25/0.30) is only the untilted fallback, not what a 2023 run charges.
_BALANCED_DEPOSIT_W = Decimal("0.60")
_BALANCED_OFZ_W = Decimal("0.10")
_BALANCED_EQUITY_W = Decimal("0.30")

# The flat fixture's first-quarter REAL rebalance (no forced hook), high_rate tilt:
#   total = 100 + 110 + 120 = 330
#   target ofz = 330 * 0.10 = 33.0  (current 110 -> SELL 77.0)
#   target eq  = 330 * 0.30 = 99.0  (current 120 -> SELL 21.0)
#   target dep = 330 * 0.60 = 198.0 (current 100 -> BUY 98.0, cost-free, D-09)
# After the first rebalance the flat legs sit at target, so later quarters trade
# ~0 -> the run's total rebalance_cost equals this first-quarter charge.
_FLAT_TOTAL = _FLAT_LEVEL + _OFZ_LEVEL + _EQUITY_LEVEL
_FLAT_OFZ_SELL = _OFZ_LEVEL - _FLAT_TOTAL * _BALANCED_OFZ_W  # 110 - 33 = 77
_FLAT_EQ_SELL = _EQUITY_LEVEL - _FLAT_TOTAL * _BALANCED_EQUITY_W  # 120 - 99 = 21
_EXPECTED_COST = (_FLAT_OFZ_SELL + _FLAT_EQ_SELL) * _ROUND_TRIP_COST
# The cost-free deposit notional (198 - 100 = 98) is EXCLUDED from the charge:
# the round-trip cost on the traded eq+ofz value alone is < the cost would be if
# the deposit notional were charged, proving the deposit leg is cost-free.
_DEPOSIT_BUY = _FLAT_TOTAL * _BALANCED_DEPOSIT_W - _FLAT_LEVEL  # 98

# NDFL flat 13% band (below the 2.4M YTD threshold).
_NDFL_13 = Decimal("0.13")

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
    """A REAL quarterly rebalance charges round-trip cost as an EXPLICIT line item (SAA-03 / D-09).

    No ``forced_leg_deltas`` hook: the cost is computed from the genuine per-leg
    rescale delta (CR-01). 2023 is a high_rate regime (no CBR cuts), so BALANCED tilts
    to {0.60/0.10/0.30}; on the flat fixture the first quarter sells ofz 77 + eq 21 (and
    BUYS deposit 98 cost-free), so the run's total ``rebalance_cost`` equals
    ``(77 + 21) * round_trip`` -- the cost-free 98 deposit notional is excluded by
    construction (deposit leg is cost-free, D-09).
    """
    dates = _daily_index(_YEAR, _DAYS_IN_YEAR)
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)
    result = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=_flat_series(dates, _EQUITY_LEVEL),
    )
    # The cost is the round-trip charge on the traded eq + ofz value only.
    assert result.rebalance_cost == _EXPECTED_COST
    assert result.rebalance_cost > _ZERO  # an explicit, non-zero field on the result

    # Deposit is cost-free, asserted via magnitude: had the cost-free 48.5 deposit
    # BUY notional been charged, the cost would be strictly larger. The actual
    # charge excludes it, so it is below the deposit-inclusive upper bound.
    cost_if_deposit_charged = (_FLAT_OFZ_SELL + _FLAT_EQ_SELL + _DEPOSIT_BUY) * _ROUND_TRIP_COST
    assert result.rebalance_cost < cost_if_deposit_charged

    # A run whose legs already sit exactly at target trades ~0 -> charges ~0 cost.
    # Build a flat fixture pre-balanced to the high_rate tilt weights (dep 60 / ofz 10 /
    # eq 30 on a 100 book): the first quarter has zero per-leg drift -> zero cost.
    at_target = orch.run(
        deposit_curve=_flat_series(dates, _BALANCED_DEPOSIT_W * _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _BALANCED_OFZ_W * _FLAT_LEVEL),
        equity_curve=_flat_series(dates, _BALANCED_EQUITY_W * _FLAT_LEVEL),
    )
    assert at_target.rebalance_cost == _ZERO


def test_realized_ndfl_on_real_rebalance_sell() -> None:
    """A RISING equity leg sold at a quarter boundary realizes FIFO-gain NDFL (D-07 / WR-01).

    Drives the orchestrator's OWN ``realized_ndfl`` through a real ``run()`` (no
    hook): a single-quarter window (Jan 1 -> Apr 2) with a strongly-rallying equity
    leg makes equity overweight at the Apr-1 boundary, so it is SOLD above its
    seeded basis (``eq[0]``). The ofz leg is flat (sold at its basis -> 0 gain), so
    the run's realized NDFL is purely the equity FIFO gain * 13% (below the band).
    """
    start = date(_YEAR, 1, 1)
    span_end = date(_YEAR, 4, 2)  # spans exactly ONE quarter boundary (Apr 1)
    dates = [start + timedelta(days=i) for i in range((span_end - start).days + 1)]
    orch = AllocationOrchestrator(risk_profile=RiskProfile.BALANCED)

    eq_daily = Decimal("1.01")  # strong rally -> equity overweight at the boundary
    eq_curve = [(d, _EQUITY_LEVEL * eq_daily**i) for i, d in enumerate(dates)]

    result = orch.run(
        deposit_curve=_flat_series(dates, _FLAT_LEVEL),
        ofz_pk_curve=_flat_series(dates, _OFZ_LEVEL),
        equity_curve=eq_curve,
    )
    assert len(result.rebalance_dates) == 1  # exactly one boundary in the window
    assert result.realized_ndfl > _ZERO

    # Independently replay the single equity sell to pin realized_ndfl == gain*0.13.
    apr1_idx = (date(_YEAR, 4, 1) - start).days
    eq_price = eq_curve[apr1_idx][1]
    weights = orch._profile_weights()  # noqa: SLF001 -- test pins the exact charge
    total = _FLAT_LEVEL + _OFZ_LEVEL + eq_price
    target_eq = total * weights.equity
    eq_units_new = target_eq / eq_price  # scale started at 1 unit @ eq[0]
    eq_sold_units = _ONE - eq_units_new  # positive: the leg is overweight -> sells
    ledger = CostBasisLedger()
    ledger.buy(AssetClass.EQUITY, _ONE, eq_curve[0][1])
    gain = ledger.sell(AssetClass.EQUITY, eq_sold_units, eq_price)
    expected_ndfl = YtdTaxAccumulator().tax(gain, _YEAR)
    assert expected_ndfl == gain * _NDFL_13  # below the 2.4M band -> flat 13%
    assert result.realized_ndfl == expected_ndfl


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
