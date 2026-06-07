"""RED scaffold: D-07 / R-3 FIFO realized-gain -> NDFL (Phase 72 Wave-0).

Pins the L5 ``CostBasisLedger`` contract (allocator-owned, net-new) before it
exists:
- a rebalance SELL pops FIFO lots earliest-first and returns the realized gain
  ``sum(sold_qty_i * (sell_price - unit_cost_i))`` (R-3 / A2 RU FIFO default);
- a profitable realized gain feeds the W1 ``YtdTaxAccumulator`` and is taxed at
  the 13% base band below the 2.4M threshold (D-07, reusing the W1 band verbatim
  -- no new tax math);
- a loss feeds ``max(0, gain) == 0`` to the accumulator -> no NDFL (a loss is
  never taxed);
- there is NO 3-year long-term-holding (LDV) exemption path -- the gain API
  takes no holding-period input, so an old gain is taxed identically to a fresh
  one (D-07).

RED now: ``finalayze.orchestration.allocation.CostBasisLedger`` (Plan 05) does
not exist yet. (``YtdTaxAccumulator`` + ``AssetClass`` already exist from W1/Plan
02; the RED is on the missing ledger.)
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import AssetClass
from finalayze.orchestration.allocation import CostBasisLedger

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

_YEAR = 2023

# Single-lot scenario.
_QTY_10 = Decimal(10)
_COST_100 = Decimal(100)
_PRICE_120 = Decimal(120)
_GAIN_SINGLE = Decimal(200)  # 10 * (120 - 100)

# Two-lot FIFO scenario.
_COST_130 = Decimal(130)
_PRICE_150 = Decimal(150)
_QTY_15 = Decimal(15)
_GAIN_TWO_LOT = Decimal(600)  # 10*(150-100) + 5*(150-130) = 500 + 100
_REMAINING_QTY = Decimal(5)  # 5 units of the 130 lot left open

# NDFL band scenario (well below the 2.4M YTD threshold).
_GAIN_100K = Decimal(100_000)
_NDFL_13 = Decimal("0.13")
_TAX_ON_100K = Decimal("13000.00")  # 100_000 * 0.13

# Loss scenario.
_PRICE_80 = Decimal(80)  # sell below the 100 cost -> negative gain
_ZERO = Decimal(0)


def test_fifo_gain_single_lot() -> None:
    """One buy, one sell -> realized gain = qty * (sell - cost) (R-3, Decimal-exact)."""
    ledger = CostBasisLedger()
    ledger.buy(AssetClass.EQUITY, _QTY_10, _COST_100)
    gain = ledger.sell(AssetClass.EQUITY, _QTY_10, _PRICE_120)
    assert gain == _GAIN_SINGLE


def test_fifo_pops_earliest_first() -> None:
    """A sell pops the EARLIEST lot first (FIFO), spanning two lots (R-3)."""
    ledger = CostBasisLedger()
    ledger.buy(AssetClass.EQUITY, _QTY_10, _COST_100)  # lot 1 @ 100
    ledger.buy(AssetClass.EQUITY, _QTY_10, _COST_130)  # lot 2 @ 130
    gain = ledger.sell(AssetClass.EQUITY, _QTY_15, _PRICE_150)
    # 10@100 popped fully (10*50=500) + 5@130 (5*20=100) -> 600.
    assert gain == _GAIN_TWO_LOT
    # The remaining open lot is 5 units @ 130 -> selling them yields 5*(150-130)=100.
    remaining_gain = ledger.sell(AssetClass.EQUITY, _REMAINING_QTY, _PRICE_150)
    assert remaining_gain == _REMAINING_QTY * (_PRICE_150 - _COST_130)


def test_realized_gain_to_ytd_tax_13pct() -> None:
    """A profitable realized gain below the threshold is taxed flat 13% (D-07 / W1 band)."""
    ledger = CostBasisLedger()
    ledger.buy(AssetClass.EQUITY, _QTY_10, _COST_100)
    gain = ledger.sell(AssetClass.EQUITY, _QTY_10, _PRICE_120)
    assert gain == _GAIN_SINGLE

    acc = YtdTaxAccumulator()
    tax = acc.tax(_GAIN_100K, _YEAR)
    assert tax == _GAIN_100K * _NDFL_13
    assert tax == _TAX_ON_100K


def test_loss_is_not_taxed() -> None:
    """A sell below cost yields a negative gain; only max(0, gain) is taxed -> 0 (D-07)."""
    ledger = CostBasisLedger()
    ledger.buy(AssetClass.EQUITY, _QTY_10, _COST_100)
    gain = ledger.sell(AssetClass.EQUITY, _QTY_10, _PRICE_80)
    assert gain < _ZERO  # 10 * (80 - 100) = -200

    acc = YtdTaxAccumulator()
    tax = acc.tax(max(_ZERO, gain), _YEAR)
    assert tax == _ZERO


def test_no_ldv_exemption() -> None:
    """No 3-yr long-term-holding exemption -- the gain API takes no holding period (D-07).

    The same (qty, unit_cost, sell_price) produces the same realized gain
    regardless of how long the lot was held: ``sell`` accepts no elapsed-time /
    holding-period argument, so an old gain is taxed identically to a fresh one.
    """
    fresh = CostBasisLedger()
    fresh.buy(AssetClass.EQUITY, _QTY_10, _COST_100)
    fresh_gain = fresh.sell(AssetClass.EQUITY, _QTY_10, _PRICE_120)

    aged = CostBasisLedger()
    aged.buy(AssetClass.EQUITY, _QTY_10, _COST_100)
    aged_gain = aged.sell(AssetClass.EQUITY, _QTY_10, _PRICE_120)

    assert fresh_gain == aged_gain == _GAIN_SINGLE
