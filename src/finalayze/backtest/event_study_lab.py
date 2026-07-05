"""News-event study lab -- the pure JUMP-vs-DRIFT decomposition primitives.

The reusable L5 measurement kernel for the "CAN RETAIL TRADE A NEWS SHOCK?" question
the operator raised: when an UNANTICIPATED headline hits a MOEX name (a gasoline
export ban, an app pulled from the store, a dividend cancelled/announced, a
geopolitical shock), how much of the price move is ALREADY GONE before a retail
investor can realistically act (the un-capturable JUMP), and how much tradeable DRIFT
is left afterwards -- market-adjusted and net of retail cost + 13% NDFL?

This is DISTINCT from :mod:`finalayze.backtest.dividend_event_lab`, which studies the
run-up before a KNOWN, scheduled dividend record date (an ANTICIPATED event). Here the
event is a surprise: the whole point is the reaction speed, so the kernel splits the
abnormal move at the first session a retail investor could enter.

NO NETWORK, NO I/O. Every function is a pure Decimal calculation over prices the runner
(:mod:`scripts.research.run_event_study`) already fetched from the token-free MOEX
ISS-REST API. The split keeps the arithmetic independently testable and reproducible.

Definitions (all market-adjusted = "abnormal" when a benchmark is supplied; a beta=1
market model, the transparent conservative choice -- it neither invents nor hides
single-name alpha):

- ``pre_close``  -- the CLOSE of the last session BEFORE the news is in the price.
- ``entry_price`` -- the first price a retail investor could realistically transact at
  (same-session CLOSE for an intraday headline, or the NEXT session's OPEN for an
  overnight/weekend one -- the runner picks the honest one per event).
- ``exit_price`` -- the CLOSE ``H`` trading days later (the drift horizon).
- **JUMP** = ``pre_close -> entry_price`` -- the move that happened BEFORE you could act.
- **DRIFT** = ``entry_price -> exit_price`` -- the move you could actually trade.
- ``jump_share`` = ``jump_abn / total_abn`` -- the fraction of the abnormal move already
  priced before entry (near 1.0 == the market beat you to it; the efficiency wall).
- ``traded_drift_net`` -- the P&L of betting in the news direction over the drift window,
  net of a round-trip retail cost and NDFL on any gain (a positive number is money a
  retail investor could actually have kept).

DIAGNOSTIC ONLY. Never production trading code, never an alpha claim on its own, and it
authorizes NOTHING -- no order, no config weight. Real money is a hard stop. See
docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from finalayze.core.constants import NDFL_RATE

# ── Pre-registered constants (named; the anti-magic-number pins) ──────────────
_ONE = Decimal(1)
_ZERO = Decimal(0)
# Retail "Investor" per-side cost (commission + half-spread + slippage). Same value as
# backtest.costs.MOEX_RETAIL_COSTS, gold_sleeve_lab and dividend_event_lab.
RETAIL_PER_SIDE_COST = Decimal("0.0055")

# NDFL_RATE (0.13) is re-exported from core.constants so the runner/tests have one import.
__all__ = [
    "NDFL_RATE",
    "RETAIL_PER_SIDE_COST",
    "EventDecomposition",
    "abnormal_return",
    "decompose_event",
    "jump_share_reliable",
    "net_abnormal_long_return",
    "net_after_costs",
    "simple_return",
]


def simple_return(p0: Decimal, p1: Decimal) -> Decimal:
    """Simple price return ``p1 / p0 - 1``; ``0`` for a non-positive base (defensive)."""
    if p0 <= _ZERO:
        return _ZERO
    return p1 / p0 - _ONE


def abnormal_return(a0: Decimal, a1: Decimal, b0: Decimal, b1: Decimal) -> Decimal:
    """Market-adjusted (abnormal) return: asset return minus benchmark return (beta=1).

    The beta=1 market model is the transparent conservative choice -- most large MOEX
    names have beta near 1, and assuming it neither manufactures nor conceals
    single-name reaction the way a fitted beta could.
    """
    return simple_return(a0, a1) - simple_return(b0, b1)


def net_after_costs(gross_ret: Decimal, per_side_cost: Decimal, ndfl: Decimal) -> Decimal:
    """Net a gross trade return for a round-trip retail cost and NDFL on any gain.

    The gross return factor ``(1 + gross_ret)`` is charged the per-side cost on BOTH legs
    (``* (1 - c) ** 2``); the resulting net gain -- and ONLY a gain -- is then taxed at
    ``ndfl`` (a loss is never given a negative-tax windfall). A zero-alpha round trip
    therefore returns ``~ -2 * cost``: trying to trade a move that is already gone costs
    money.
    """
    net = (_ONE + gross_ret) * (_ONE - per_side_cost) * (_ONE - per_side_cost) - _ONE
    if net > _ZERO:
        net = net * (_ONE - ndfl)
    return net


@dataclass(frozen=True)
class EventDecomposition:
    """One event's JUMP-vs-DRIFT split (raw and abnormal), plus the net tradeable drift.

    ``missed_favorable_jump`` is the abnormal jump signed by the news direction: a
    POSITIVE number is the favorable move a perfectly-informed trader would have wanted
    but a retail investor could NOT enter before. ``jump_share`` is ``None`` when the
    total abnormal move is exactly zero (no move to attribute).
    """

    jump_raw: Decimal
    drift_raw: Decimal
    total_raw: Decimal
    jump_abn: Decimal
    drift_abn: Decimal
    total_abn: Decimal
    missed_favorable_jump: Decimal
    traded_drift_net: Decimal
    jump_share: Decimal | None


def decompose_event(
    *,
    pre_close: Decimal,
    entry_price: Decimal,
    exit_price: Decimal,
    bench_pre: Decimal | None,
    bench_entry: Decimal | None,
    bench_exit: Decimal | None,
    direction: int,
    per_side_cost: Decimal = RETAIL_PER_SIDE_COST,
    ndfl: Decimal = NDFL_RATE,
) -> EventDecomposition:
    """Split an event into the pre-entry JUMP and the post-entry tradeable DRIFT.

    When ``bench_*`` are all supplied, jump/drift/total are market-adjusted (abnormal);
    otherwise the raw values are used (e.g. the benchmark IS the subject, an index-level
    shock). ``direction`` is ``+1`` for good news, ``-1`` for bad -- it signs both the
    ``missed_favorable_jump`` and the ``traded_drift_net`` so a correct directional bet
    reads positive. ``traded_drift_net`` is that directional drift netted of the
    round-trip cost and NDFL-on-gain via :func:`net_after_costs`.
    """
    jump_raw = simple_return(pre_close, entry_price)
    drift_raw = simple_return(entry_price, exit_price)
    total_raw = simple_return(pre_close, exit_price)

    if bench_pre is not None and bench_entry is not None and bench_exit is not None:
        jump_abn = jump_raw - simple_return(bench_pre, bench_entry)
        drift_abn = drift_raw - simple_return(bench_entry, bench_exit)
        total_abn = total_raw - simple_return(bench_pre, bench_exit)
    else:
        jump_abn, drift_abn, total_abn = jump_raw, drift_raw, total_raw

    sign = Decimal(direction)
    missed_favorable_jump = sign * jump_abn
    traded_drift_net = net_after_costs(sign * drift_abn, per_side_cost, ndfl)
    jump_share = None if total_abn == _ZERO else jump_abn / total_abn

    return EventDecomposition(
        jump_raw=jump_raw,
        drift_raw=drift_raw,
        total_raw=total_raw,
        jump_abn=jump_abn,
        drift_abn=drift_abn,
        total_abn=total_abn,
        missed_favorable_jump=missed_favorable_jump,
        traded_drift_net=traded_drift_net,
        jump_share=jump_share,
    )


def jump_share_reliable(jump_abn: Decimal, total_abn: Decimal) -> bool:
    """Is ``jump_abn / total_abn`` a meaningful fraction-in-``(0, 1]``?

    Only when the jump and the total net move share sign AND the total is at least as
    large in magnitude as the jump (a near-monotone move). When the move OVERSHOOTS and
    reverses, ``total_abn`` collapses toward zero and the raw ratio explodes past 1 or
    flips negative -- an artifact, not a "97% priced before entry". Callers must render
    ``jump_share`` as n/a in that case and lead with a bounded metric instead.
    """
    if total_abn == _ZERO:
        return False
    return jump_abn * total_abn > _ZERO and abs(total_abn) >= abs(jump_abn)


def net_abnormal_long_return(
    entry: Decimal,
    exit_: Decimal,
    bench_entry: Decimal,
    bench_exit: Decimal,
    per_side_cost: Decimal = RETAIL_PER_SIDE_COST,
    ndfl: Decimal = NDFL_RATE,
) -> Decimal:
    """Net market-adjusted return of a LONG held ``entry -> exit_`` (cost + NDFL on gain).

    The base-rate primitive: given a same-field entry/exit for the asset and the
    benchmark, compute the abnormal (asset-minus-market) return and net it for the
    round-trip retail cost and NDFL-on-gain. Used to build a direction-BLIND base rate
    -- the median net long drift after EVERY large positive abnormal move across the
    panel -- so a single event's positive drift can be judged against the systematic
    base rate (event alpha) rather than a name's ambient trend.
    """
    return net_after_costs(
        abnormal_return(entry, exit_, bench_entry, bench_exit), per_side_cost, ndfl
    )
