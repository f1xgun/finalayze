"""Crypto research primitives — cross-exchange arb edge + a time-series-momentum sleeve.

Pure, tested, no I/O. Two families:

- **Cross-exchange arbitrage** (``best_cross_venue_spread``, ``net_arb_edge_frac``): the best
  realisable top-of-book spread across venues, and its net edge after both taker legs and the
  amortised withdrawal/rebalancing cost. The cert measures the observed spread distribution and
  asks whether it clears round-trip fees — before even charging the capital-lockup carry.

- **Trend sleeve** (``time_series_momentum_signal``, ``crypto_trend_nav``): a long/flat TSMOM
  overlay on a RUB-denominated crypto price path, net of trading cost and 13% NDFL on realised
  gains, with idle bars parked in the risk-free deposit (the strategy's fairest cash state). Its
  net NAV feeds the canonical Instrument Integration Gate against the MCFTRR equity leg — the same
  gate that rejected gold and real estate.

Everything is measurement only. It authorises no order; real-money execution is a hard stop.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)


def time_series_momentum_signal(levels: list[Decimal], lookback: int) -> list[int]:
    """TSMOM signal aligned to ``levels``: ``1`` (long) if ``level[t] > level[t-lookback]``, else 0.

    The first ``lookback`` entries are 0 (insufficient history). ``signal[t]`` is decided from data
    through bar ``t`` only; :func:`crypto_trend_nav` applies it to the *next* bar's return, so there
    is no look-ahead.
    """
    out: list[int] = []
    for i in range(len(levels)):
        if i >= lookback and levels[i - lookback] > _ZERO and levels[i] > levels[i - lookback]:
            out.append(1)
        else:
            out.append(0)
    return out


def best_cross_venue_spread(
    quotes: dict[str, tuple[Decimal, Decimal]],
) -> tuple[Decimal, str, str]:
    """Best realisable cross-venue spread: buy at the cheapest ask, sell at the richest bid.

    ``quotes`` maps venue -> ``(bid, ask)``. Returns ``(fractional_spread, buy_venue, sell_venue)``
    where ``fractional_spread = (best_bid - best_ask) / best_ask`` (can be negative when no venue is
    dislocated). Requires at least two venues.
    """
    if len(quotes) < _TWO:
        raise ValueError("need at least two venues to measure a cross-venue spread")
    buy_venue = min(quotes, key=lambda v: quotes[v][1])  # cheapest ask
    sell_venue = max(quotes, key=lambda v: quotes[v][0])  # richest bid
    best_ask = quotes[buy_venue][1]
    best_bid = quotes[sell_venue][0]
    frac = (best_bid - best_ask) / best_ask if best_ask > _ZERO else _ZERO
    return frac, buy_venue, sell_venue


def net_arb_edge_frac(
    gross_frac: Decimal, taker_fee_frac: Decimal, withdrawal_frac: Decimal
) -> Decimal:
    """Net fractional edge per executed round trip: ``gross - 2*taker - withdrawal``.

    Two taker legs (buy on one venue, sell on the other) plus the amortised withdrawal/network cost
    of rebalancing inventory back across venues.
    """
    return gross_frac - _TWO * taker_fee_frac - withdrawal_frac


def nearest_rank_percentile(xs: list[Decimal], q: float) -> Decimal:
    """Nearest-rank percentile (``q`` in [0,1]) — Decimal-native for deterministic certs."""
    if not xs:
        raise ValueError("empty sequence")
    ordered = sorted(xs)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    rank = q * len(ordered)
    idx = min(len(ordered) - 1, int(-(-rank // 1)) - 1)  # ceil(rank) - 1, clamped
    return ordered[max(0, idx)]


def crypto_trend_nav(
    *,
    dates: list[date],
    crypto_levels: list[Decimal],
    deposit_factors: list[Decimal],
    signal: list[int],
    per_side_cost: Decimal,
    ndfl: Decimal,
    initial_nav: Decimal = _ONE,
) -> list[tuple[date, Decimal]]:
    """Net NAV of a long/flat crypto trend sleeve (RUB numeraire).

    The position held over interval ``(k -> k+1)`` is ``signal[k]`` (decided at bar ``k``'s
    close, applied to the next interval — no look-ahead). A long interval rides the crypto daily
    factor ``levels[k+1]/levels[k]``; a flat interval earns ``deposit_factors[k+1]`` (the
    risk-free deposit, the strategy's fairest idle state). On entry pay ``per_side_cost``; on exit
    pay ``per_side_cost`` and ``ndfl`` on that trade's positive realised gain (no loss offset —
    conservative). A position open on the last bar is force-closed so its gain is realised (taxed).

    ``crypto_levels``, ``deposit_factors`` and ``signal`` are all aligned to ``dates``. Returns
    ``[(date, nav)]`` of the same length, starting at ``initial_nav``.
    """
    n = len(dates)
    if not (len(crypto_levels) == len(deposit_factors) == len(signal) == n):
        raise ValueError("dates, crypto_levels, deposit_factors, signal must share one length")
    if n == 0:
        return []
    nav = initial_nav
    navs = [nav]
    entry_nav: Decimal | None = None
    prev_pos = 0
    for k in range(n - 1):
        pos = signal[k]
        if pos == 1 and prev_pos == 0:  # buy at close k
            nav *= _ONE - per_side_cost
            entry_nav = nav
        if pos == 1 and crypto_levels[k] > _ZERO:
            nav *= crypto_levels[k + 1] / crypto_levels[k]
        elif pos == 0:
            nav *= deposit_factors[k + 1]
        next_pos = signal[k + 1] if k + 1 < n - 1 else 0  # force-close on the last interval
        if pos == 1 and next_pos == 0:  # sell at close k+1
            nav *= _ONE - per_side_cost
            if entry_nav is not None and nav > entry_nav:
                nav -= ndfl * (nav - entry_nav)
            entry_nav = None
        navs.append(nav)
        prev_pos = pos
    return list(zip(dates, navs, strict=True))
