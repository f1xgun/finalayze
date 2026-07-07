"""Tests for the jump-response lab — the reactive-news alpha-decay primitives.

Every expectation is hand-computed. The load-bearing invariant is *no look-ahead in jump
detection*: the trailing volatility at bar ``i`` must exclude ``return[i]`` itself, otherwise a
genuine shock would inflate its own denominator and hide. ``test_detect_jump_excludes_own_return``
pins that — with the shock inside the vol window it would fail the z-gate and go undetected.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.backtest.jump_response_lab import (
    align_sign,
    bar_returns,
    capture,
    detect_jumps,
    forward_path,
    half_life_bars,
    mean_path,
    net_after_cost,
    split_runs_on_gaps,
    trailing_stdev,
)

_TOL = Decimal("0.000001")


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= _TOL


# ── split_runs_on_gaps (intraday session continuity) ─────────────────────────
def test_split_runs_on_gaps_breaks_on_large_gap() -> None:
    # 60s-spaced bars, then a 3600s (session-break) jump, then two more 60s bars
    stamps = [0, 60, 120, 3720, 3780]
    levels = [Decimal(x) for x in (100, 101, 102, 200, 201)]
    runs = split_runs_on_gaps(stamps, levels, max_gap_s=300)
    assert runs == [[Decimal(100), Decimal(101), Decimal(102)], [Decimal(200), Decimal(201)]]


def test_split_runs_on_gaps_tolerates_small_gaps() -> None:
    # a single 120s gap (one missing minute) is within max_gap -> stays one run
    stamps = [0, 60, 180, 240]
    levels = [Decimal(x) for x in (100, 101, 102, 103)]
    runs = split_runs_on_gaps(stamps, levels, max_gap_s=300)
    assert len(runs) == 1


# ── bar_returns ──────────────────────────────────────────────────────────────
def test_bar_returns_first_is_zero_then_simple() -> None:
    r = bar_returns([Decimal(100), Decimal(110), Decimal(99)])
    assert r[0] == Decimal(0)
    assert _close(r[1], Decimal("0.1"))
    assert _close(r[2], Decimal("-0.1"))


# ── trailing_stdev ───────────────────────────────────────────────────────────
def test_trailing_stdev_prior_window_only() -> None:
    # idx5 sees returns[1..4] = [0.01,-0.01,0.01,-0.01]; sample stdev of +-0.01 = sqrt(1/3)*0.01
    returns = [Decimal(0)] + [Decimal("0.01"), Decimal("-0.01")] * 2 + [Decimal("0.20")]
    sd = trailing_stdev(returns, 5, 4)
    assert sd is not None
    assert _close(sd, Decimal("0.0115470053"))


def test_trailing_stdev_none_until_window_filled() -> None:
    returns = [Decimal(0), Decimal("0.01"), Decimal("0.02")]
    assert trailing_stdev(returns, 2, 4) is None  # only 2 prior values, need 4


# ── detect_jumps (+ the no-look-ahead invariant) ─────────────────────────────
def _spike_levels() -> list[Decimal]:
    # multiplicative: x1.01,x0.99,x1.01,x0.99,x1.20 -> returns 0.01,-0.01,0.01,-0.01,0.20 (exact)
    lv = [Decimal(100)]
    for m in ("1.01", "0.99", "1.01", "0.99", "1.20"):
        lv.append(lv[-1] * Decimal(m))
    return lv


def test_detect_jumps_finds_the_single_spike() -> None:
    jumps = detect_jumps(_spike_levels(), vol_window=4, z_threshold=Decimal(5))
    assert jumps == [(5, 1)]  # only the +20% bar clears 5-sigma; sign +1


def test_detect_jump_excludes_own_return() -> None:
    # z at idx5 = 0.20 / stdev([+-0.01]) ~= 17.3 (>=5) ONLY because 0.20 is NOT in the window.
    # Were it included, stdev jumps to ~0.09 and z~=2.2 (<5), undetected. Detection proves it.
    jumps = detect_jumps(_spike_levels(), vol_window=4, z_threshold=Decimal(5))
    assert len(jumps) == 1


def test_detect_jumps_signs_down_moves() -> None:
    # calm +-1% bars establish the vol window, THEN a -25% crash at idx6 (its trailing window is
    # calm, so the crash is not masked by an earlier shock -- see the note in _spike_levels).
    lv = [Decimal(100)]
    for m in ("1.01", "0.99", "1.01", "0.99", "1.01", "0.75"):
        lv.append(lv[-1] * Decimal(m))
    jumps = detect_jumps(lv, vol_window=4, z_threshold=Decimal(5))
    assert jumps == [(6, -1)]  # only the -25% bar clears 5-sigma; sign -1


# ── forward_path / align_sign / mean_path ────────────────────────────────────
def test_forward_path_cumulative_from_event() -> None:
    p = forward_path([Decimal(100), Decimal(110), Decimal(121), Decimal(110)], 0, 3)
    assert p[0] == Decimal(0)
    assert _close(p[1], Decimal("0.1"))
    assert _close(p[2], Decimal("0.21"))
    assert _close(p[3], Decimal("0.1"))


def test_align_sign_flips_down_jumps() -> None:
    aligned = align_sign([Decimal(0), Decimal("0.1"), Decimal("-0.05")], -1)
    assert aligned == [Decimal(0), Decimal("-0.1"), Decimal("0.05")]


def test_mean_path_elementwise() -> None:
    m = mean_path(
        [[Decimal(0), Decimal("0.1"), Decimal("0.2")], [Decimal(0), Decimal("0.3"), Decimal(0)]]
    )
    assert _close(m[1], Decimal("0.2"))
    assert _close(m[2], Decimal("0.1"))


# ── capture / net_after_cost / half_life ─────────────────────────────────────
def test_capture_is_exit_minus_entry() -> None:
    mf = [Decimal(0), Decimal("0.05"), Decimal("0.08"), Decimal("0.06")]
    assert _close(capture(mf, 1, 3), Decimal("0.01"))


def test_net_after_cost_taxes_only_positive_net() -> None:
    # gross 0.02 - 0.005 cost = 0.015 > 0 -> x(1-0.13) = 0.01305
    assert _close(
        net_after_cost(Decimal("0.02"), Decimal("0.005"), Decimal("0.13")), Decimal("0.01305")
    )
    # gross 0.003 - 0.005 cost = -0.002 < 0 → no tax, stays -0.002
    assert _close(
        net_after_cost(Decimal("0.003"), Decimal("0.005"), Decimal("0.13")), Decimal("-0.002")
    )


def test_half_life_bars_first_to_reach_half_terminal() -> None:
    mf = [Decimal(0), Decimal("0.02"), Decimal("0.06"), Decimal("0.08"), Decimal("0.10")]
    assert half_life_bars(mf) == 2  # terminal 0.10, half 0.05, first >= at h=2


def test_half_life_none_when_no_continuation() -> None:
    assert half_life_bars([Decimal(0), Decimal("-0.01"), Decimal("-0.02")]) is None  # terminal <= 0
