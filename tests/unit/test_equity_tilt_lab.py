"""Tests for the diagnostic equity-tilt basket simulator (active-equity-sleeve R&D).

This is a DIAGNOSTIC harness (not production trading code): a transparent,
deterministic weighted-basket simulator used to answer one honest question —
does a low-turnover active weighting (equal-weight / ADV-cap-proxy / dividend)
beat a cap-weight proxy on RISK-ADJUSTED total return NET of the real retail
1.10% round-trip cost and net-of-NDFL dividends. See
docs/research/active_equity_sleeve_experiment.md.

The pure logic (weight policies + the basket NAV simulator) is tested here on
synthetic fixtures so correctness is provable without a Tinkoff token.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.costs import TransactionCosts
from finalayze.backtest.equity_tilt_lab import (
    PricePoint,
    adv_cap_proxy_weights,
    equal_weights,
    inverse_vol_weights,
    make_dividend_yield_policy,
    max_drawdown_pct,
    quarter_end_dates,
    simulate_basket,
)

# ── Named constants (no PLR2004 magic numbers) ───────────────────────────────
_INITIAL_NAV = Decimal(1000000)
_FLAT_PRICE = Decimal(100)
_WEIGHT_TOL = Decimal("0.0000001")
_TWO = Decimal(2)
_HALF = Decimal("0.5")
_ZERO_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal(0),
    min_commission=Decimal(0),
    spread_bps=Decimal(0),
    slippage_bps=Decimal(0),
)
_RETAIL_PCT_SIDE = Decimal("0.0055")  # 0.30% comm + 15bps + 10bps = 0.55%/side
_RETAIL_COSTS = TransactionCosts(
    commission_per_share=Decimal(0),
    commission_rate=Decimal("0.003"),
    min_commission=Decimal(0),  # drop the 1 RUB floor so the test math is exact
    spread_bps=Decimal(15),
    slippage_bps=Decimal(10),
)


def _daily(start: date, n: int) -> list[date]:
    """n consecutive calendar days (weekend-agnostic; fine for a synthetic axis)."""
    return [start + timedelta(days=i) for i in range(n)]


def _flat_panel(
    symbols: list[str], dates: list[date], price: Decimal
) -> dict[str, list[PricePoint]]:
    return {s: [(d, price, Decimal(1000)) for d in dates] for s in symbols}


def test_quarter_end_dates_picks_last_trading_day_of_each_quarter() -> None:
    # span Q1..Q3 2024 with one date per month-end-ish
    dates = [
        date(2024, 1, 15),
        date(2024, 3, 28),
        date(2024, 3, 29),  # last in Q1
        date(2024, 4, 2),
        date(2024, 6, 27),  # last in Q2
        date(2024, 7, 1),
        date(2024, 9, 30),  # last in Q3
    ]
    qe = quarter_end_dates(dates)
    assert date(2024, 3, 29) in qe
    assert date(2024, 6, 27) in qe
    assert date(2024, 9, 30) in qe
    # not the mid-quarter dates
    assert date(2024, 1, 15) not in qe
    assert date(2024, 4, 2) not in qe


def test_equal_weights_splits_only_available_names() -> None:
    dates = _daily(date(2024, 1, 1), 5)
    panel: dict[str, list[PricePoint]] = {
        "A": [(d, _FLAT_PRICE, Decimal(1000)) for d in dates],
        "B": [(d, _FLAT_PRICE, Decimal(1000)) for d in dates],
        # C only starts on the last day -> excluded at an early as-of
        "C": [(dates[-1], _FLAT_PRICE, Decimal(1000))],
    }
    w = equal_weights(dates[0], panel)
    assert set(w) == {"A", "B"}
    assert w["A"] == w["B"]
    assert abs(sum(w.values()) - Decimal(1)) < _WEIGHT_TOL
    # once C has data it joins
    w2 = equal_weights(dates[-1], panel)
    assert set(w2) == {"A", "B", "C"}


def test_adv_cap_proxy_weights_scale_with_traded_value() -> None:
    dates = _daily(date(2024, 1, 1), 10)
    panel: dict[str, list[PricePoint]] = {
        # same price, B trades 2x the volume -> ~2x ADV weight
        "A": [(d, _FLAT_PRICE, Decimal(1000)) for d in dates],
        "B": [(d, _FLAT_PRICE, Decimal(2000)) for d in dates],
    }
    w = adv_cap_proxy_weights(dates[-1], panel, lookback=5)
    assert abs(sum(w.values()) - Decimal(1)) < _WEIGHT_TOL
    # B ~ 2/3, A ~ 1/3
    assert abs(w["B"] - (_TWO / Decimal(3))) < _WEIGHT_TOL
    assert abs(w["A"] - (Decimal(1) / Decimal(3))) < _WEIGHT_TOL


def test_simulate_basket_tracks_price_with_zero_costs_no_divs() -> None:
    # single name, price doubles -> NAV doubles (100% allocation, no costs, no divs)
    dates = _daily(date(2024, 1, 1), 4)
    panel: dict[str, list[PricePoint]] = {
        "A": [
            (dates[0], Decimal(100), Decimal(1000)),
            (dates[1], Decimal(100), Decimal(1000)),
            (dates[2], Decimal(150), Decimal(1000)),
            (dates[3], Decimal(200), Decimal(1000)),
        ]
    }
    res = simulate_basket(
        panel=panel,
        dividend_schedule={},
        weight_policy=equal_weights,
        rebalance_dates=[dates[0]],
        costs=_ZERO_COSTS,
        initial_nav=_INITIAL_NAV,
    )
    assert res.nav_curve[0] == _INITIAL_NAV
    assert res.nav_curve[-1] == _INITIAL_NAV * _TWO  # price doubled
    assert res.total_cost == Decimal(0)
    assert res.total_tax == Decimal(0)


def test_simulate_basket_credits_net_of_ndfl_dividends() -> None:
    # flat price, one dividend; NAV rises by net-of-13% dividend (below 2.4M band)
    dates = _daily(date(2024, 1, 1), 3)
    panel = _flat_panel(["A"], dates, Decimal(100))
    # 1,000,000 NAV / 100 price = 10,000 shares; 5 RUB/share gross = 50,000 gross
    div = {("A", dates[1]): Decimal(5)}
    res = simulate_basket(
        panel=panel,
        dividend_schedule=div,
        weight_policy=equal_weights,
        rebalance_dates=[dates[0]],
        costs=_ZERO_COSTS,
        initial_nav=_INITIAL_NAV,
    )
    gross = Decimal(10000) * Decimal(5)  # 50,000
    net = gross * (Decimal(1) - Decimal("0.13"))  # 13% band below threshold
    assert res.dividend_gross == gross
    assert res.total_tax == gross * Decimal("0.13")
    assert res.nav_curve[-1] == _INITIAL_NAV + net


def test_simulate_basket_charges_retail_round_trip_on_rebalance() -> None:
    # buy on day0 then fully rotate to the other name on day1 -> ~0.55%/side each leg
    dates = _daily(date(2024, 1, 1), 2)
    panel = _flat_panel(["A", "B"], dates, Decimal(100))

    def all_in_a(_asof: date, _hist: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        return {"A": Decimal(1)}

    def all_in_b(_asof: date, _hist: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        return {"B": Decimal(1)}

    policies = {dates[0]: all_in_a, dates[1]: all_in_b}

    def switch(asof: date, hist: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        return policies[asof](asof, hist)

    res = simulate_basket(
        panel=panel,
        dividend_schedule={},
        weight_policy=switch,
        rebalance_dates=[dates[0], dates[1]],
        costs=_RETAIL_COSTS,
        initial_nav=_INITIAL_NAV,
    )
    # day0: buy ~1,000,000 of A -> 0.55% cost. day1: sell A + buy B -> ~2x0.55%.
    # total cost is strictly positive and on the order of 1.6% of NAV.
    assert res.total_cost > _INITIAL_NAV * Decimal("0.015")
    assert res.total_cost < _INITIAL_NAV * Decimal("0.02")
    assert res.nav_curve[-1] < _INITIAL_NAV  # costs ate into NAV


def test_inverse_vol_weights_favor_the_calmer_name() -> None:
    dates = _daily(date(2024, 1, 1), 60)
    # A: steady drift (low vol). B: alternating jumps (high vol).
    a_pts: list[PricePoint] = []
    b_pts: list[PricePoint] = []
    pa, pb = Decimal(100), Decimal(100)
    for i, d in enumerate(dates):
        pa = pa + Decimal("0.1")  # smooth
        pb = Decimal(110) if i % 2 == 0 else Decimal(90)  # choppy
        a_pts.append((d, pa, Decimal(1000)))
        b_pts.append((d, pb, Decimal(1000)))
    w = inverse_vol_weights(dates[-1], {"A": a_pts, "B": b_pts}, lookback=50)
    assert abs(sum(w.values()) - Decimal(1)) < _WEIGHT_TOL
    assert w["A"] > w["B"]  # calmer name gets more weight


def test_dividend_yield_policy_overweights_higher_yield_within_clip() -> None:
    dates = _daily(date(2024, 1, 1), 40)
    panel = _flat_panel(["A", "B"], dates, Decimal(100))
    # A pays a fat trailing dividend, B pays nothing -> A overweighted (capped at 2/N)
    sched = {("A", dates[10]): Decimal(10)}
    policy = make_dividend_yield_policy(sched)
    w = policy(dates[-1], panel)
    assert abs(sum(w.values()) - Decimal(1)) < _WEIGHT_TOL
    assert w["A"] > w["B"]
    # clip keeps it bounded: A cannot exceed 2x its 1/N base share
    assert w["A"] <= _TWO * _HALF + _WEIGHT_TOL  # 2 * (1/2) = the clip ceiling here


def test_max_drawdown_pct_basic() -> None:
    curve = [100.0, 120.0, 90.0, 110.0]  # peak 120 -> trough 90 = 25% DD
    assert abs(max_drawdown_pct(curve) - 25.0) < 0.01
