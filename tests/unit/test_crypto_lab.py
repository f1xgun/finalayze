"""Unit tests for crypto_lab — pure arb-edge + TSMOM-sleeve primitives (no I/O, no network)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.backtest.crypto_lab import (
    best_cross_venue_spread,
    crypto_trend_nav,
    nearest_rank_percentile,
    net_arb_edge_frac,
    time_series_momentum_signal,
)

_COST = Decimal("0.01")
_NDFL = Decimal("0.13")
_LB = 2
_TOL = Decimal("0.000001")
_GAIN_FINAL = Decimal("1.2636264")
_LOSS_FINAL = Decimal("0.793881")
_ZERO_COST = Decimal(0)
_NOLOOK_FINAL = Decimal("0.5")
_VOL_ONETAX_FINAL = Decimal("3.61")
_MULTITRIP_FINAL = Decimal("3.4969")


def _dates(n: int) -> list[date]:
    return [date(2022, 1, 1 + i) for i in range(n)]


# ── TSMOM signal ─────────────────────────────────────────────────────────────
def test_tsmom_signal_long_on_uptrend() -> None:
    sig = time_series_momentum_signal([Decimal(x) for x in (1, 2, 3, 4, 5)], _LB)
    assert sig == [0, 0, 1, 1, 1]


def test_tsmom_signal_flat_on_downtrend() -> None:
    sig = time_series_momentum_signal([Decimal(x) for x in (5, 4, 3, 2, 1)], _LB)
    assert sig == [0, 0, 0, 0, 0]


def test_tsmom_signal_no_lookahead_first_bars_flat() -> None:
    # the first `lookback` bars can never be long (insufficient history)
    sig = time_series_momentum_signal([Decimal(10)] * 6, 3)
    assert sig[:3] == [0, 0, 0]


# ── cross-venue arb ──────────────────────────────────────────────────────────
def test_best_cross_venue_spread_picks_min_ask_max_bid() -> None:
    quotes = {
        "x": (Decimal("100.0"), Decimal("100.2")),
        "y": (Decimal("100.5"), Decimal("100.7")),
    }
    frac, buy, sell = best_cross_venue_spread(quotes)
    assert buy == "x"  # cheapest ask
    assert sell == "y"  # richest bid
    assert abs(frac - (Decimal("100.5") - Decimal("100.2")) / Decimal("100.2")) < _TOL


def test_best_cross_venue_spread_needs_two_venues() -> None:
    with pytest.raises(ValueError, match="two venues"):
        best_cross_venue_spread({"only": (Decimal(1), Decimal(2))})


def test_net_arb_edge_subtracts_two_taker_legs_and_withdrawal() -> None:
    # gross 30bps - 2*5bps taker - 2bps withdrawal = 18bps
    edge = net_arb_edge_frac(Decimal("0.003"), Decimal("0.0005"), Decimal("0.0002"))
    assert edge == Decimal("0.0018")


def test_nearest_rank_percentile() -> None:
    xs = [Decimal(v) for v in (1, 2, 3, 4, 5)]
    assert nearest_rank_percentile(xs, 0.5) == Decimal(3)
    assert nearest_rank_percentile(xs, 1.0) == Decimal(5)
    assert nearest_rank_percentile(xs, 0.0) == Decimal(1)


# ── trend-sleeve NAV ─────────────────────────────────────────────────────────
def test_trend_nav_all_flat_follows_deposit() -> None:
    n = 4
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=[Decimal(1)] * n,
        deposit_factors=[Decimal(1), Decimal("1.01"), Decimal("1.01"), Decimal("1.01")],
        signal=[0, 0, 0, 0],
        per_side_cost=_COST,
        ndfl=_NDFL,
    )
    vals = [v for _, v in nav]
    assert vals[0] == Decimal(1)
    assert abs(vals[-1] - Decimal("1.01") ** 3) < _TOL


def test_trend_nav_long_gain_charges_costs_and_ndfl() -> None:
    # +10%/bar, held long, one entry + forced terminal exit: gross 1.331, minus 2x1% cost, minus
    # 13% NDFL on the realised gain -> 1.2636264
    n = 4
    levels = [Decimal(100), Decimal(110), Decimal(121), Decimal("133.1")]
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=levels,
        deposit_factors=[Decimal(1)] * n,
        signal=[1, 1, 1, 1],
        per_side_cost=_COST,
        ndfl=_NDFL,
    )
    final = nav[-1][1]
    assert abs(final - _GAIN_FINAL) < _TOL
    # net of costs+tax must be BELOW the gross buy-and-hold multiple
    assert final < Decimal("1.331")


def test_trend_nav_losing_trade_pays_no_ndfl() -> None:
    # -10%/bar: entry+exit cost only, NO tax on a loss -> 0.99*0.9*0.9*0.99
    n = 3
    levels = [Decimal(100), Decimal(90), Decimal(81)]
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=levels,
        deposit_factors=[Decimal(1)] * n,
        signal=[1, 1, 1],
        per_side_cost=_COST,
        ndfl=_NDFL,
    )
    assert abs(nav[-1][1] - _LOSS_FINAL) < _TOL


def test_trend_nav_length_matches_dates() -> None:
    n = 5
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=[Decimal(100 + i) for i in range(n)],
        deposit_factors=[Decimal(1)] * n,
        signal=[0, 1, 1, 0, 1],
        per_side_cost=_COST,
        ndfl=_NDFL,
    )
    assert len(nav) == n
    assert [d for d, _ in nav] == _dates(n)


def test_trend_nav_no_lookahead_rides_next_move_not_the_signal_move() -> None:
    # The signal fires AT the up-move bar (k=4, +100%) and is applied to the NEXT interval (4->5),
    # so the sleeve is long during the FOLLOWING -50% move — not the move that triggered it. A
    # look-ahead bug would ride the +100% and end at 2.0; the causal sleeve ends at 0.5.
    levels = [Decimal(100), Decimal(100), Decimal(100), Decimal(100), Decimal(200), Decimal(100)]
    n = len(levels)
    signal = time_series_momentum_signal(levels, 1)
    assert signal == [0, 0, 0, 0, 1, 0]
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=levels,
        deposit_factors=[Decimal(1)] * n,
        signal=signal,
        per_side_cost=_ZERO_COST,
        ndfl=_NDFL,
    )
    assert abs(nav[-1][1] - _NOLOOK_FINAL) < _TOL


def test_trend_nav_volatile_buyhold_taxed_once_not_per_bar() -> None:
    # +100/-75/+700% held long: gross 4x; NDFL taxes the single realised gain ONCE at sale
    # (4.0 - 0.13*3.0 = 3.61). Guards against reintroducing a per-bar (net_index_returns-style) tax
    # that would bleed a volatile path far below this — the exact bug that was fixed.
    levels = [Decimal(100), Decimal(200), Decimal(50), Decimal(400)]
    n = len(levels)
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=levels,
        deposit_factors=[Decimal(1)] * n,
        signal=[1, 1, 1, 1],
        per_side_cost=_ZERO_COST,
        ndfl=_NDFL,
    )
    assert abs(nav[-1][1] - _VOL_ONETAX_FINAL) < _TOL


def test_trend_nav_entry_basis_resets_across_trips() -> None:
    # Two-trip [1,0,1] path: trip-2's NDFL must use trip-2's OWN entry basis (1.87), not a stale
    # trip-1 basis. Trip1 1->2 taxed 0.13 -> 1.87; flat; trip2 1.87->3.74 taxed 0.13*1.87 -> 3.4969.
    # A stale-entry bug would tax 0.13*(3.74-1.0) and yield ~3.384.
    levels = [Decimal(100), Decimal(200), Decimal(200), Decimal(400)]
    n = len(levels)
    nav = crypto_trend_nav(
        dates=_dates(n),
        crypto_levels=levels,
        deposit_factors=[Decimal(1)] * n,
        signal=[1, 0, 1, 0],
        per_side_cost=_ZERO_COST,
        ndfl=_NDFL,
    )
    assert abs(nav[-1][1] - _MULTITRIP_FINAL) < _TOL
