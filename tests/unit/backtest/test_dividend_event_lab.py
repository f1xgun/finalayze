"""Unit tests for the pure dividend-event study lab.

Deterministic, no-network tests of the run-up / capture primitives and the sleeve
NAV builder, over a SYNTHETIC MOEX trading calendar (weekends/holidays excluded by
construction). All magic values are named constants (ruff PLR2004).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.backtest.dividend_event_lab import (
    DEPLOY_FRACTION,
    MODE_EQUITY_OVERLAY,
    MODE_RUNUP_ONLY,
    NDFL_RATE,
    RETAIL_PER_SIDE_COST,
    SleeveEvent,
    build_sleeve_nav,
    capture_return,
    detect_ex_date,
    ex_gap_pct,
    last_day_with_dividend,
    mcftrr_daily_factors,
    runup_return,
)

# ── synthetic trading calendar (Mon-Fri of two weeks; weekend gap is real) ────
# Week 1: Mon 2024-01-08 .. Fri 2024-01-12 ; Week 2: Mon 2024-01-15 .. Fri 2024-01-19.
_MON_W1 = date(2024, 1, 8)
_TUE_W1 = date(2024, 1, 9)
_WED_W1 = date(2024, 1, 10)
_THU_W1 = date(2024, 1, 11)
_FRI_W1 = date(2024, 1, 12)
_SAT_W1 = date(2024, 1, 13)  # NOT a trading day
_SUN_W1 = date(2024, 1, 14)  # NOT a trading day
_MON_W2 = date(2024, 1, 15)
_TUE_W2 = date(2024, 1, 16)
_WED_W2 = date(2024, 1, 17)
_THU_W2 = date(2024, 1, 18)
_FRI_W2 = date(2024, 1, 19)

_CALENDAR = [
    _MON_W1,
    _TUE_W1,
    _WED_W1,
    _THU_W1,
    _FRI_W1,
    _MON_W2,
    _TUE_W2,
    _WED_W2,
    _THU_W2,
    _FRI_W2,
]

# ── 2023-era calendar (pre-T1_TRANSITION) for the T+2 fallback scenarios ───────
# Mon-Fri of one 2023 week; record date < 2024-01-01 exercises the T+2 fallback branch.
_W2023_MON = date(2023, 6, 5)
_W2023_TUE = date(2023, 6, 6)
_W2023_WED = date(2023, 6, 7)
_W2023_THU = date(2023, 6, 8)
_W2023_FRI = date(2023, 6, 9)
_CAL_2023 = [_W2023_MON, _W2023_TUE, _W2023_WED, _W2023_THU, _W2023_FRI]

# ── SBER-2021 real-shape constants (the confirmed off-by-one case) ─────────────
# record 2021-05-12, div 18.7; real closes: 05-07=317.94, 05-10=320.19, 05-11=307.16
# (the true -4.07% ex-gap), 05-12=302.02. The pre-2024 T+2 board -> true ex = 05-11.
_SB_D07 = date(2021, 5, 7)
_SB_D10 = date(2021, 5, 10)
_SB_D11 = date(2021, 5, 11)
_SB_D12 = date(2021, 5, 12)
_SB_P07 = Decimal("317.94")
_SB_P10 = Decimal("320.19")
_SB_P11 = Decimal("307.16")
_SB_P12 = Decimal("302.02")
_SB_DIV = Decimal("18.70")
_SB_CAL = [_SB_D07, _SB_D10, _SB_D11, _SB_D12]

# ── price / return constants ──────────────────────────────────────────────────
_P_100 = Decimal(100)
_P_105 = Decimal(105)  # +5% vs 100
_P_103 = Decimal(103)
_P_95 = Decimal(95)  # ex-gap target (~ -9.5 vs 105)
_DIV_10 = Decimal(10)  # gross dividend per share
_ZERO = Decimal(0)
_ONE = Decimal(1)
_K_2 = 2
_K_1 = 1
_M_1 = 1
_M_0 = 0

# tolerance for Decimal comparisons of compounded / rooted factors
_TOL = Decimal("0.00000001")
_PLACES = Decimal("0.000001")


def _q(x: Decimal) -> Decimal:
    return x.quantize(_PLACES)


def _round_trip(cost: Decimal) -> Decimal:
    return (_ONE - cost) * (_ONE - cost)


# ── last_day_with_dividend / ex-date arithmetic ───────────────────────────────


def test_ldd_and_ex_when_record_date_is_a_trading_day() -> None:
    # Record date = Wed W2. LDD = last trading day strictly before = Tue W2.
    # ex-date = first trading day after LDD = Wed W2 (the record date itself, T+1 board).
    result = last_day_with_dividend(_WED_W2, _CALENDAR)
    assert result == (_TUE_W2, _WED_W2)


def test_ldd_and_ex_skip_the_weekend() -> None:
    # Record date = Mon W2. LDD = Fri W1 (skips Sat/Sun — TRADING days, not calendar).
    # ex-date = first trading day after Fri W1 = Mon W2.
    result = last_day_with_dividend(_MON_W2, _CALENDAR)
    assert result == (_FRI_W1, _MON_W2)


def test_ldd_when_record_date_is_a_holiday_gap() -> None:
    # Record date lands on the (non-trading) Saturday: LDD = Fri W1, ex = Mon W2.
    result = last_day_with_dividend(_SAT_W1, _CALENDAR)
    assert result == (_FRI_W1, _MON_W2)


def test_ldd_none_when_no_trading_day_before_record() -> None:
    # Record date on/before the very first calendar day -> cannot be holder-of-record.
    assert last_day_with_dividend(_MON_W1, _CALENDAR) is None


def test_ldd_none_when_ex_gap_unobservable() -> None:
    # LDD would be the final calendar day -> no trading day after it to observe the ex-gap.
    # Record date after the last trading day, whose only prior trading day is the last one.
    after_last = date(2024, 1, 22)  # Mon, not on the calendar (calendar ends Fri W2)
    assert last_day_with_dividend(after_last, _CALENDAR) is None


# ── detect_ex_date (FIX 1: robust, gap-driven ex-date detection) ──────────────
# Named price/dividend constants for the detection scenarios (no magic literals).
_DET_PRIOR = Decimal("320.00")  # close on the LDD (pre-gap)
_DET_EX_DROP = Decimal("300.00")  # post-gap close (~ -6.25% vs 320)
_DET_FLAT_UP = Decimal("321.00")  # a mild up-day (no gap)
_DET_FLAT_DN = Decimal("319.00")  # a mild down-day (< the detection threshold)
_DET_DIV = Decimal("20.00")  # gross dividend (theoretical drop ~ -6.25% vs 320)
_DET_SMALL_DIV = Decimal("0.50")  # tiny masked dividend -> no qualifying gap -> fallback


def test_detect_ex_date_gap_on_rt_t1_era() -> None:
    # T+1 era (2024): the -6.25% gap prints ON Rt (the record day itself, Wed W2).
    # closes: Tue W2 = 320 (LDD/pre-gap), Wed W2 = 300 (ex/gap). Record = Wed W2.
    closes = {_TUE_W2: _DET_PRIOR, _WED_W2: _DET_EX_DROP}
    result = detect_ex_date(closes, _WED_W2, _DET_DIV, _CALENDAR)
    assert result == (_WED_W2, _TUE_W2)  # (ex, ldd)


def test_detect_ex_date_gap_on_rt_minus_1_t2_era() -> None:
    # T+2 era (pre-2024): the -6.25% gap prints on Rt-1 (Tue W2), NOT on the record day.
    # closes: Mon W2 = 320 (LDD/pre-gap), Tue W2 = 300 (ex/gap), Wed W2 = record day.
    # Record date is a 2023 date so the fallback (unused here) would be the T+2 branch.
    closes = {_MON_W2: _DET_PRIOR, _TUE_W2: _DET_EX_DROP, _WED_W2: _DET_FLAT_UP}
    result = detect_ex_date(closes, _WED_W2, _DET_DIV, _CALENDAR)
    assert result == (_TUE_W2, _MON_W2)  # ex on Rt-1, ldd the session before


def test_detect_ex_date_falls_back_to_convention_when_no_gap_t2() -> None:
    # A tiny masked dividend: no candidate shows a qualifying (>= 25% of theoretical) drop,
    # so detection falls back to the settlement convention. Record 2023 -> T+2 -> ex = Rt-1.
    # Use a 2023-era calendar so record_date < T1_TRANSITION.
    closes = dict.fromkeys(_CAL_2023, _DET_PRIOR)
    closes[_W2023_THU] = _DET_FLAT_DN  # mild noise, well below the detection threshold
    result = detect_ex_date(closes, _W2023_WED, _DET_SMALL_DIV, _CAL_2023)
    # Rt = Wed (record is a trading day); T+2 fallback -> ex = Rt-1 = Tue; ldd = Mon.
    assert result == (_W2023_TUE, _W2023_MON)


def test_detect_ex_date_falls_back_to_convention_when_no_gap_t1() -> None:
    # Same masked-dividend fallback but in the T+1 era (2024) -> ex = Rt (record day).
    closes = dict.fromkeys(_CALENDAR, _DET_PRIOR)
    closes[_THU_W2] = _DET_FLAT_DN  # mild noise below threshold
    result = detect_ex_date(closes, _WED_W2, _DET_SMALL_DIV, _CALENDAR)
    # Rt = Wed W2 (record day, a trading day); T+1 fallback -> ex = Rt = Wed; ldd = Tue.
    assert result == (_WED_W2, _TUE_W2)


def test_detect_ex_date_weekend_record_resolves_to_last_trading_day() -> None:
    # Record date falls on the (non-trading) Saturday. Rt = Fri W1. The gap prints on Fri
    # W1 (T+1-style), so ex = Fri W1, ldd = Thu W1.
    closes = {_THU_W1: _DET_PRIOR, _FRI_W1: _DET_EX_DROP}
    result = detect_ex_date(closes, _SAT_W1, _DET_DIV, _CALENDAR)
    assert result == (_FRI_W1, _THU_W1)


def test_detect_ex_date_sber_2021_style_gap_excluded_from_runup() -> None:
    # SBER-2021-style: a ~-4% drop on the "05-11" bar is the TRUE ex-gap. detect_ex_date
    # must return ex on that gap bar and ldd on the PRIOR (pre-gap) bar, so a k=1 run-up
    # window ends at the ldd close and EXCLUDES the drop.
    closes = {
        _SB_D07: _SB_P07,  # buy day (k=1 window start) 05-07 close 317.94
        _SB_D10: _SB_P10,  # ldd / pre-gap 05-10 close 320.19
        _SB_D11: _SB_P11,  # ex / gap 05-11 close 307.16 (~ -4.07%)
        _SB_D12: _SB_P12,  # record day 05-12 close 302.02
    }
    result = detect_ex_date(closes, _SB_D12, _SB_DIV, _SB_CAL)
    assert result == (_SB_D11, _SB_D10)  # ex = gap bar, ldd = pre-gap bar
    ex_date, ldd = result
    # A k=1 run-up window [ldd - 1, ldd] = [05-07, 05-10] ends the day BEFORE the gap.
    runup = runup_return(closes, ldd, _K_1, _SB_CAL, per_side_cost=_ZERO)
    assert runup is not None
    # gross = 320.19 / 317.94 - 1 > 0 (a mild PRE-ex run-up, the -4% gap excluded).
    assert _q(runup) == _q(_SB_P10 / _SB_P07 - _ONE)
    assert runup > _ZERO
    # The excluded ex-gap itself is the sharp negative move (sanity on the split).
    gap = ex_gap_pct(closes, ldd, ex_date)
    assert gap is not None
    assert gap < _ZERO


# ── runup_return (Variant A) ──────────────────────────────────────────────────


def test_runup_return_known_five_percent_minus_two_costs() -> None:
    # BUY at LDD-2 = Mon W2 (100), SELL at LDD = Wed W2 (105): +5% gross, minus round trip.
    ldd = _WED_W2
    prices = {_MON_W2: _P_100, _WED_W2: _P_105}
    result = runup_return(prices, ldd, _K_2, _CALENDAR, per_side_cost=RETAIL_PER_SIDE_COST)
    assert result is not None
    expected = (_P_105 / _P_100) * _round_trip(RETAIL_PER_SIDE_COST) - _ONE
    assert _q(result) == _q(expected)
    # sanity: the 5% gross gain survives the 1.10% round trip and stays positive.
    assert result > _ZERO


def test_runup_return_zero_cost_is_pure_price_change() -> None:
    ldd = _WED_W2
    prices = {_MON_W2: _P_100, _WED_W2: _P_105}
    result = runup_return(prices, ldd, _K_2, _CALENDAR, per_side_cost=_ZERO)
    assert result is not None
    assert _q(result) == _q(_P_105 / _P_100 - _ONE)


def test_runup_return_none_on_missing_close() -> None:
    ldd = _WED_W2
    prices = {_WED_W2: _P_105}  # buy-day close absent
    assert runup_return(prices, ldd, _K_2, _CALENDAR) is None


def test_runup_return_none_when_buy_day_off_calendar() -> None:
    # LDD = Tue W1; LDD - 5 trading days walks off the front of the calendar.
    ldd = _TUE_W1
    prices = {_TUE_W1: _P_105}
    assert runup_return(prices, ldd, 5, _CALENDAR) is None


# ── capture_return (Variant B) — eats the gap, collects the net dividend ───────


def test_capture_return_eats_gap_adds_net_dividend() -> None:
    # BUY LDD-1 = Tue W2 (100). Hold through ex. SELL ex+1.
    # LDD = Wed W2, ex = Thu W2, sell day = ex+1 = Fri W2 (post-gap price 95).
    # Collect net dividend = 10 * (1 - 0.13).
    ldd, ex_date = _WED_W2, _THU_W2
    prices = {_TUE_W2: _P_100, _FRI_W2: _P_95}
    result = capture_return(
        prices,
        ldd,
        ex_date,
        _K_1,
        _M_1,
        _DIV_10,
        _CALENDAR,
        ndfl=NDFL_RATE,
        per_side_cost=RETAIL_PER_SIDE_COST,
    )
    assert result is not None
    div_net = _DIV_10 * (_ONE - NDFL_RATE)
    expected = ((_P_95 + div_net) / _P_100) * _round_trip(RETAIL_PER_SIDE_COST) - _ONE
    assert _q(result) == _q(expected)


def test_capture_return_gap_swamps_net_dividend_is_negative() -> None:
    # Buy pre-gap at 105 (LDD-1 = Tue W2), sell on the ex-date itself (m=0 = Thu W2) at
    # the post-gap 95. The ~-9.5% gap is only partly refunded by the 8.7-net dividend, so
    # net-of-tax the collector loses the wedge: buy 105 vs sell(95)+net_div(8.7)=103.7.
    ldd, ex_date = _WED_W2, _THU_W2
    prices = {_TUE_W2: _P_105, _THU_W2: _P_95}
    result = capture_return(
        prices,
        ldd,
        ex_date,
        _K_1,
        _M_0,
        _DIV_10,
        _CALENDAR,
    )
    assert result is not None
    # sell (95) + net div (8.7) = 103.7 vs buy 105 -> negative before cost, more so after.
    assert result < _ZERO


def test_capture_return_none_on_missing_price() -> None:
    ldd, ex_date = _WED_W2, _THU_W2
    prices = {_TUE_W2: _P_100}  # sell-day close absent
    assert capture_return(prices, ldd, ex_date, _K_1, _M_1, _DIV_10, _CALENDAR) is None


# ── ex_gap_pct diagnostic ─────────────────────────────────────────────────────


def test_ex_gap_pct_is_raw_price_ratio() -> None:
    prices = {_WED_W2: _P_105, _THU_W2: _P_95}
    gap = ex_gap_pct(prices, _WED_W2, _THU_W2)
    assert gap is not None
    assert _q(gap) == _q(_P_95 / _P_105 - _ONE)
    assert gap < _ZERO  # ex-date gaps DOWN


def test_ex_gap_pct_none_on_missing_close() -> None:
    assert ex_gap_pct({_WED_W2: _P_105}, _WED_W2, _THU_W2) is None


# ── build_sleeve_nav ──────────────────────────────────────────────────────────


def test_idle_runup_sleeve_stays_flat_at_initial_nav() -> None:
    # No events -> every bar is idle -> NAV is flat == initial (1).
    nav = build_sleeve_nav([], _CALENDAR, MODE_RUNUP_ONLY)
    assert [v for _, v in nav] == [_ONE] * len(_CALENDAR)
    assert nav[0] == (_CALENDAR[0], _ONE)


def test_single_event_runup_sleeve_reproduces_the_net_window_return() -> None:
    # One window: buy Mon W2 (100), exit Wed W2 (105). Active bars = (Mon W2, Wed W2] =
    # {Tue W2, Wed W2}. End-to-end the NAV must equal the window's net factor exactly.
    ev = SleeveEvent(
        ticker="SBER",
        buy_day=_MON_W2,
        ldd=_WED_W2,
        entry_price=_P_100,
        exit_price=_P_105,
    )
    nav = build_sleeve_nav([ev], _CALENDAR, MODE_RUNUP_ONLY)
    final = nav[-1][1]
    expected_factor = (_P_105 / _P_100) * _round_trip(RETAIL_PER_SIDE_COST)
    assert abs(final - expected_factor) < _TOL
    # Bars before the window are flat at 1.0 (idle); the growth is confined to the window.
    idx_before = next(i for i, (d, _) in enumerate(nav) if d == _MON_W2)
    assert nav[idx_before][1] == _ONE


def test_runup_sleeve_matches_runup_return_end_to_end() -> None:
    # The sleeve's final NAV - 1 must equal runup_return for the same single window.
    ev = SleeveEvent(
        ticker="LKOH",
        buy_day=_MON_W2,
        ldd=_WED_W2,
        entry_price=_P_100,
        exit_price=_P_105,
    )
    nav = build_sleeve_nav([ev], _CALENDAR, MODE_RUNUP_ONLY)
    sleeve_ret = nav[-1][1] - _ONE
    prices = {_MON_W2: _P_100, _WED_W2: _P_105}
    direct = runup_return(prices, _WED_W2, _K_2, _CALENDAR)
    assert direct is not None
    assert abs(sleeve_ret - direct) < _TOL


def test_equity_overlay_convex_blend_on_window_bars() -> None:
    # The overlay is a CONVEX fractional tilt: on an idle bar day_factor = mcftrr factor;
    # on a window bar day_factor = (1 - DEPLOY_FRACTION) * mcftrr + DEPLOY_FRACTION * window.
    # With a flat (entry==exit, zero-cost) window the window factor is exactly 1, so the
    # overlay must equal the pure MCFTRR compounding EXCEPT on the two window bars, where
    # each mcftrr factor is convex-blended toward 1 by DEPLOY_FRACTION.
    mcftrr = [(d, _P_100 + Decimal(i)) for i, d in enumerate(_CALENDAR)]
    factors = mcftrr_daily_factors(mcftrr)
    ev = SleeveEvent(
        ticker="GAZP",
        buy_day=_MON_W2,
        ldd=_WED_W2,
        entry_price=_P_100,
        exit_price=_P_100,
    )
    nav = build_sleeve_nav(
        [ev],
        _CALENDAR,
        MODE_EQUITY_OVERLAY,
        mcftrr_factors=factors,
        per_side_cost=_ZERO,
    )
    # Active window bars are (buy_day, ldd] = {Tue W2, Wed W2}; flat window factor = 1.
    window_bars = {_TUE_W2, _WED_W2}
    expected = _ONE
    for d in _CALENDAR[1:]:
        base = factors[d]
        if d in window_bars:
            expected *= (_ONE - DEPLOY_FRACTION) * base + DEPLOY_FRACTION * _ONE
        else:
            expected *= base
    assert abs(nav[-1][1] - expected) < _TOL


def test_equity_overlay_idle_bars_ride_pure_mcftrr() -> None:
    # With NO events every bar is idle -> the overlay must equal pure MCFTRR compounding
    # (the tilt only fires on window bars; idle bars ride the passive core untouched).
    mcftrr = [(d, _P_100 + Decimal(i)) for i, d in enumerate(_CALENDAR)]
    factors = mcftrr_daily_factors(mcftrr)
    nav = build_sleeve_nav(
        [],
        _CALENDAR,
        MODE_EQUITY_OVERLAY,
        mcftrr_factors=factors,
        per_side_cost=_ZERO,
    )
    expected_final = mcftrr[-1][1] / mcftrr[0][1]
    assert abs(nav[-1][1] - expected_final) < _TOL


def test_equity_overlay_requires_mcftrr_factors() -> None:
    ev = SleeveEvent(
        ticker="GMKN",
        buy_day=_MON_W2,
        ldd=_WED_W2,
        entry_price=_P_100,
        exit_price=_P_105,
    )
    try:
        build_sleeve_nav([ev], _CALENDAR, MODE_EQUITY_OVERLAY)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlay mode without mcftrr_factors")


def test_unknown_mode_raises() -> None:
    try:
        build_sleeve_nav([], _CALENDAR, "bogus_mode")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_mcftrr_daily_factors_shape() -> None:
    mcftrr = [(_MON_W1, _P_100), (_TUE_W1, _P_105), (_WED_W1, _P_103)]
    factors = mcftrr_daily_factors(mcftrr)
    # first bar has no prior -> omitted; two factors returned.
    assert set(factors) == {_TUE_W1, _WED_W1}
    assert _q(factors[_TUE_W1]) == _q(_P_105 / _P_100)
    assert _q(factors[_WED_W1]) == _q(_P_103 / _P_105)
