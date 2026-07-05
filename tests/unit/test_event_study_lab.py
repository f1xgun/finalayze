"""Unit tests for the news-event-study lab (jump-vs-drift decomposition primitives).

TDD (RED first): these pin the pure arithmetic that answers the operator's question
-- when unanticipated news hits a MOEX name, how much of the move is GONE before a
retail investor can act (the un-capturable JUMP) versus how much tradeable DRIFT is
left afterwards, market-adjusted and net of retail cost + 13% NDFL.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.backtest.event_study_lab import (
    NDFL_RATE,
    RETAIL_PER_SIDE_COST,
    abnormal_return,
    decompose_event,
    jump_share_reliable,
    net_abnormal_long_return,
    net_after_costs,
    simple_return,
)

# Named test fixtures (no bare magic numbers in assertions -- ruff PLR2004).
_PRE = Decimal(100)
_GAP_ENTRY = Decimal(70)  # a -30% down-gap already printed before entry
_FLAT_EXIT = Decimal(70)  # then flat: no drift left to capture
_BENCH_PRE = Decimal(1000)
_BENCH_ENTRY = Decimal(950)  # the whole market fell 5% too
_BENCH_EXIT = Decimal(950)
_BAD_NEWS = -1
_GOOD_NEWS = 1
_ZERO = Decimal(0)
_TEN_PCT = Decimal("0.10")
_ONE_PCT = Decimal("0.01")


def test_simple_return_basic() -> None:
    assert simple_return(Decimal(100), Decimal(110)) == Decimal("0.10")
    assert simple_return(Decimal(100), Decimal(90)) == Decimal("-0.10")


def test_simple_return_guards_nonpositive_base() -> None:
    assert simple_return(_ZERO, Decimal(50)) == _ZERO


def test_abnormal_return_subtracts_benchmark() -> None:
    # asset +2%, market +5% -> abnormal -3%.
    abn = abnormal_return(Decimal(100), Decimal(102), Decimal(1000), Decimal(1050))
    assert abn == Decimal("-0.03")


def test_net_after_costs_no_frictions_is_identity() -> None:
    assert net_after_costs(_TEN_PCT, _ZERO, _ZERO) == _TEN_PCT


def test_net_after_costs_taxes_only_gains() -> None:
    # +10% gross, no cost, 13% NDFL -> 8.7% net.
    assert net_after_costs(_TEN_PCT, _ZERO, Decimal("0.13")) == Decimal("0.087")
    # a loss is NOT taxed (no negative tax windfall).
    assert net_after_costs(Decimal("-0.10"), _ZERO, Decimal("0.13")) == Decimal("-0.10")


def test_net_after_costs_zero_alpha_round_trip_loses_two_sides() -> None:
    # zero gross return, only the round-trip cost -> a loss of ~2*cost.
    net = net_after_costs(_ZERO, _ONE_PCT, _ZERO)
    expected = (Decimal(1) - _ONE_PCT) * (Decimal(1) - _ONE_PCT) - Decimal(1)
    assert net == expected
    assert net < _ZERO


def test_decompose_bad_news_gap_is_all_jump_no_drift() -> None:
    # -30% gap before entry, then flat: a short would have wanted the jump but could
    # not enter before it; nothing left to trade afterwards.
    dec = decompose_event(
        pre_close=_PRE,
        entry_price=_GAP_ENTRY,
        exit_price=_FLAT_EXIT,
        bench_pre=None,
        bench_entry=None,
        bench_exit=None,
        direction=_BAD_NEWS,
        per_side_cost=_ZERO,
        ndfl=_ZERO,
    )
    assert dec.jump_raw == Decimal("-0.30")
    assert dec.drift_raw == _ZERO
    # favorable move missed (a short's +30%) -- un-capturable.
    assert dec.missed_favorable_jump == Decimal("0.30")
    # nothing to capture after entry.
    assert dec.traded_drift_net == _ZERO
    # 100% of the abnormal move happened before entry.
    assert dec.jump_share == Decimal(1)


def test_decompose_charges_costs_on_zero_drift() -> None:
    dec = decompose_event(
        pre_close=_PRE,
        entry_price=_GAP_ENTRY,
        exit_price=_FLAT_EXIT,
        bench_pre=None,
        bench_entry=None,
        bench_exit=None,
        direction=_BAD_NEWS,
        per_side_cost=RETAIL_PER_SIDE_COST,
        ndfl=NDFL_RATE,
    )
    # even with zero drift, trying to trade it loses the round-trip cost.
    assert dec.traded_drift_net < _ZERO


def test_decompose_abnormal_strips_market_move() -> None:
    dec = decompose_event(
        pre_close=_PRE,
        entry_price=_GAP_ENTRY,
        exit_price=_FLAT_EXIT,
        bench_pre=_BENCH_PRE,
        bench_entry=_BENCH_ENTRY,
        bench_exit=_BENCH_EXIT,
        direction=_BAD_NEWS,
        per_side_cost=_ZERO,
        ndfl=_ZERO,
    )
    # ticker fell 30%, market 5% -> abnormal jump -25%.
    assert dec.jump_abn == Decimal("-0.25")
    assert dec.total_abn == Decimal("-0.25")
    assert dec.drift_abn == _ZERO
    assert dec.jump_share == Decimal(1)


def test_decompose_good_news_capturable_drift_is_taxed_and_costed() -> None:
    # good news: +5% before entry (missed), then +10% abnormal drift you DO ride.
    dec = decompose_event(
        pre_close=Decimal(100),
        entry_price=Decimal(105),
        exit_price=Decimal("115.5"),  # +10% from entry
        bench_pre=None,
        bench_entry=None,
        bench_exit=None,
        direction=_GOOD_NEWS,
        per_side_cost=_ZERO,
        ndfl=Decimal("0.13"),
    )
    assert dec.drift_raw == Decimal("0.10")
    # +10% drift, taxed at 13% -> 8.7% net (no cost in this case).
    assert dec.traded_drift_net == Decimal("0.087")
    assert dec.missed_favorable_jump == Decimal("0.05")


def test_jump_share_reliable_only_for_near_monotone_moves() -> None:
    # near-monotone down move: jump -25%, total -27% -> reliable (share in (0,1]).
    assert jump_share_reliable(Decimal("-0.25"), Decimal("-0.27")) is True
    # overshoot-then-reversal: jump -1.76%, total -0.10% -> ratio 17.6, UNreliable.
    assert jump_share_reliable(Decimal("-0.0176"), Decimal("-0.0010")) is False
    # sign flip (drift more than undid the jump): jump -1%, total +0.5% -> UNreliable.
    assert jump_share_reliable(Decimal("-0.01"), Decimal("0.005")) is False
    # exactly zero total -> UNreliable.
    assert jump_share_reliable(Decimal("-0.01"), _ZERO) is False


def test_net_abnormal_long_return_strips_market_and_nets_frictions() -> None:
    # asset +12%, market +2% -> abnormal +10%; net of 13% NDFL (no cost) -> +8.7%.
    net = net_abnormal_long_return(
        Decimal(100), Decimal(112), Decimal(100), Decimal(102), _ZERO, Decimal("0.13")
    )
    assert net == Decimal("0.087")
    # asset flat, market up 3% -> abnormal -3% (a loss), untaxed, only that loss.
    loss = net_abnormal_long_return(
        Decimal(100), Decimal(100), Decimal(100), Decimal(103), _ZERO, _ZERO
    )
    assert loss < _ZERO


def test_decompose_zero_total_move_has_no_jump_share() -> None:
    dec = decompose_event(
        pre_close=Decimal(100),
        entry_price=Decimal(100),
        exit_price=Decimal(100),
        bench_pre=None,
        bench_entry=None,
        bench_exit=None,
        direction=_GOOD_NEWS,
        per_side_cost=_ZERO,
        ndfl=_ZERO,
    )
    assert dec.jump_share is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
