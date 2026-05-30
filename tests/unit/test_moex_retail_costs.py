"""S7.3 — expose retail-tariff costs for honest MOEX backtests.

Audit #13: ``MOEX_COSTS`` is calibrated for the Tinkoff Invest Trader
tariff (0.04% commission, 10 bps spread, 7 bps slippage). Retail accounts
typically run on the Investor ("Инвестор") tariff at ~0.3% commission —
about 7.5x higher. Running a backtest on Trader-tariff numbers when the
operator will actually trade on a retail account silently inflates
expected PnL and changes the go-live decision.

Contract:
  S7.3-01: ``MOEX_RETAIL_COSTS`` exists and uses retail commission
           (>= 0.003 = 30 bps), wider spread, and higher slippage than
           ``MOEX_COSTS``.
  S7.3-02: ``MOEX_TRADER_COSTS`` exists as an explicit alias for
           ``MOEX_COSTS`` — callers wiring up scripts can be unambiguous
           about which tariff they are charging.
  S7.3-03: retail cost on a typical 100x100 RUB trade is strictly higher
           than Trader cost on the same trade (regression safety —
           defaults must not converge accidentally).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.backtest.costs import (
    MOEX_COSTS,
    MOEX_RETAIL_COSTS,
    MOEX_TRADER_COSTS,
)

_PRICE = Decimal(100)
_QTY = Decimal(100)
_RETAIL_MIN_COMMISSION_RATE = Decimal("0.003")  # 30 bps floor


# ─── S7.3-01 ────────────────────────────────────────────────────────────────
def test_retail_costs_match_investor_tariff() -> None:
    """``MOEX_RETAIL_COSTS`` must charge at least 30 bps (Investor tariff range)."""
    assert MOEX_RETAIL_COSTS.commission_rate >= _RETAIL_MIN_COMMISSION_RATE, (
        "Retail tariff is typically 0.3% (30 bps); anything lower understates "
        "real retail trading cost"
    )
    assert MOEX_RETAIL_COSTS.spread_bps >= MOEX_COSTS.spread_bps, (
        "Retail traders pay at least the Trader-tariff spread, often more"
    )
    assert MOEX_RETAIL_COSTS.slippage_bps >= MOEX_COSTS.slippage_bps, (
        "Smaller retail tickets fall on the wider side of the book — slippage "
        "should be at least the Trader-tariff value"
    )


# ─── S7.3-02 ────────────────────────────────────────────────────────────────
def test_trader_costs_alias_identical_to_moex_costs() -> None:
    """``MOEX_TRADER_COSTS`` must alias the existing ``MOEX_COSTS`` constant."""
    assert MOEX_TRADER_COSTS is MOEX_COSTS, (
        "MOEX_TRADER_COSTS should be the same object as MOEX_COSTS to avoid "
        "silently diverging duplicates"
    )


# ─── S7.3-03 ────────────────────────────────────────────────────────────────
def test_retail_commission_rate_about_7x_trader() -> None:
    """Commission ratio must match the audit ~7x claim (0.3% / 0.04%)."""
    ratio = MOEX_RETAIL_COSTS.commission_rate / MOEX_TRADER_COSTS.commission_rate
    assert ratio > Decimal(5), (
        f"Retail vs Trader commission_rate ratio is {ratio}; audit #13 expects ~7x"
    )


def test_retail_total_cost_materially_higher_than_trader() -> None:
    """Total cost (commission + spread + slippage) on 100x100 RUB must be > 2x Trader.

    Total-cost ratio is smaller than the commission ratio because spread /
    slippage uplifts only ~1.5x, but it still has to be unambiguously bigger
    so the backtest is honest about retail economics.
    """
    trader_cost = MOEX_TRADER_COSTS.total_cost(_PRICE, _QTY)
    retail_cost = MOEX_RETAIL_COSTS.total_cost(_PRICE, _QTY)
    ratio = retail_cost / trader_cost
    assert ratio > Decimal(2), (
        f"Retail/Trader TOTAL cost ratio is {ratio}; expected > 2x on a typical "
        "100 RUB x 100 share trade so backtests visibly degrade"
    )


def test_retail_cost_for_small_trade_respects_min_commission() -> None:
    """Even tiny trades must charge at least ``min_commission`` RUB."""
    tiny_cost = MOEX_RETAIL_COSTS.total_cost(Decimal(1), Decimal(1))
    assert tiny_cost >= MOEX_RETAIL_COSTS.min_commission, (
        "min_commission must apply for tiny notional values "
        f"(got cost={tiny_cost}, min={MOEX_RETAIL_COSTS.min_commission})"
    )
