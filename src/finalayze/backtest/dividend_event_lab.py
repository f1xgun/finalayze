"""Dividend-event study lab — the PURE, deterministic run-up / capture primitives.

The reusable L5 measurement kernel for the "EQUITY DIVIDEND RUN-UP" hypothesis
(docs/research/instrument_integration_program.md, next iteration). It answers ONE
honest question opened by the strategic-direction pivot: is there a monetizable,
net-of-everything drift in a MOEX share's price in the trading days BEFORE a KNOWN,
announced dividend record date — a "run-up" that is NOT already priced into the
ex-date gap?

NO NETWORK, NO I/O. Every function here is a pure Decimal calculation over data the
runner (:mod:`scripts.research.run_dividend_event_study`) has already fetched from
the token-free MOEX ISS-REST API. This split keeps the arithmetic independently
testable and reproducible offline.

Domain facts baked in (from the program ledger; treated as ground truth):

- MOEX equities settle T+1 on board TQBR. To be a holder-of-record on the dividend
  ``registryclosedate`` (record date ``R``) you must BUY no later than ``R`` minus
  1 TRADING day. That last buyable day is the LDD (last-day-with-dividend). The
  EX-date (price gaps down by ~the dividend) is ``LDD + 1`` trading day.
- Trading-day arithmetic (skip weekends/MOEX holidays) is mandatory — never calendar
  days. The caller supplies the realized MOEX trading calendar (the sorted set of
  dates on which the share actually traded), so holidays are handled by construction.
- Retail cost = 0.55%/side = ``Decimal("0.0055")`` (commission + half-spread +
  slippage). A round trip charges it TWICE. Dividend NDFL tax = 13%.

Two arms, both net of everything:

- **Variant A — run-up-and-exit** (:func:`runup_return`): BUY at the CLOSE ``k``
  trading days before LDD, SELL at the CLOSE on LDD. You capture only the pre-payout
  price drift and EXIT before the ex-gap; you never collect the dividend, so there is
  NO NDFL, but you also give up the ex-gap by construction.
- **Variant B — collect-and-hold** (:func:`capture_return`): BUY the same ``k`` days
  before LDD, HOLD THROUGH the ex-date, SELL at the CLOSE ``m`` trading days after the
  ex-date, and ADD the net-of-13%-NDFL dividend actually received. This arm EATS the
  ex-gap and tests the classic mispricing (is the gap < the net dividend?).

Both variants subtract a round-trip retail cost (``2 * per_side_cost``). The
:func:`ex_gap_pct` diagnostic reports the realized ex-date gap for the ex-gap
decomposition. :func:`build_sleeve_nav` aggregates per-event returns into a daily NET
NAV curve suitable to feed the instrument-integration gate as a ``Candidate``.

This is a DIAGNOSTIC lab, never production trading code and never an alpha claim on
its own — a positive result must still clear the pre-registered integration gate. It
authorizes NOTHING: no order, no config weight. Real money is a hard stop. See
docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# ── Pre-registered constants (named; the anti-magic-number pins) ──────────────
_ONE = Decimal(1)
_ZERO = Decimal(0)
# Retail "Investor" per-side cost (commission + half-spread + slippage). Same value
# as backtest.costs.MOEX_RETAIL_COSTS and gold_sleeve_lab._RETAIL_PER_SIDE_COST.
RETAIL_PER_SIDE_COST = Decimal("0.0055")
# Russian dividend/coupon personal income tax withheld at source.
NDFL_RATE = Decimal("0.13")

# ── Ex-date detection constants (FIX 1: robust per-event ex-date) ─────────────
# A candidate session qualifies as the realized ex-gap only if its down-move is at
# least this fraction of the theoretical drop (dividend / prior close). This filters
# masked / small dividends whose real drop is drowned by ordinary daily noise, in
# which case we fall back to the settlement convention below.
GAP_DETECT_MIN_FRACTION = Decimal("0.25")
# MOEX equities settled T+2 through 2023 (ex = record - 1 trading day) and moved to
# T+1 on 2024-01-01 (ex = record). This is the convention used only as a FALLBACK
# when no candidate session shows a qualifying ex-gap.
T1_TRANSITION = date(2024, 1, 1)
# The number of prior trading days scanned as ex-date candidates (Rt, Rt-1, Rt-2).
_EX_CANDIDATE_LOOKBACK = 2


@dataclass(frozen=True)
class DividendEvent:
    """One announced dividend and its computed trading-calendar anchor dates.

    ``ldd`` (last-day-with-dividend) = ``record_date`` minus 1 trading day (the last
    session you can BUY on the T+1 board and still be a holder-of-record). ``ex_date``
    = ``ldd`` plus 1 trading day (the session the clean price gaps down by ~the
    dividend). ``value`` is the GROSS declared dividend per share (RUB, Decimal); the
    net-of-NDFL amount actually received is ``value * (1 - NDFL_RATE)``.
    """

    ticker: str
    record_date: date
    ldd: date
    ex_date: date
    value: Decimal


def last_day_with_dividend(
    record_date: date,
    trading_days: list[date],
) -> tuple[date, date] | None:
    """Compute (LDD, ex_date) for a dividend record date on a real trading calendar.

    ``trading_days`` is the sorted, ascending list of dates on which the share
    actually traded (the realized MOEX calendar — weekends and holidays are absent by
    construction, so this is genuine TRADING-day arithmetic, not calendar days).

    The record date ``R`` itself need not be a trading day (it often is not). The LDD
    is the LAST trading day STRICTLY BEFORE ``R``; the ex-date is the FIRST trading day
    STRICTLY AFTER the LDD (i.e. the first trading day on/after ``R`` — usually ``R``
    on the T+1 board, or the next session when ``R`` is a holiday). Returns ``None``
    when the calendar has no trading day before ``R`` (cannot buy in) or no trading day
    after the LDD (cannot observe the ex-gap) — the caller must skip such events rather
    than fabricate an anchor.
    """
    # index of the first trading day >= record_date
    idx = bisect.bisect_left(trading_days, record_date)
    if idx == 0:
        return None  # no trading day strictly before R -> cannot be a holder-of-record
    ldd = trading_days[idx - 1]
    # ex-date = first trading day strictly after the LDD
    ex_idx = bisect.bisect_right(trading_days, ldd)
    if ex_idx >= len(trading_days):
        return None  # no trading day after the LDD -> ex-gap unobservable
    ex_date = trading_days[ex_idx]
    return ldd, ex_date


def detect_ex_date(
    closes_by_date: dict[date, Decimal],
    record_date: date,
    dividend: Decimal,
    trading_days: list[date],
) -> tuple[date, date] | None:
    """Robustly detect the realized (ex_date, ldd) from the price series itself.

    This removes the systematic off-by-one of a fixed settlement assumption and
    auto-handles both the 2024 T+1 transition and weekend/holiday record dates. MOEX
    equities settled T+2 through 2023 (ex = record - 1 trading day) but moved to T+1 in
    2024 (ex = record); assuming a single convention for all years mis-dates every
    pre-2024 event by one session and measures the ex-gap where the run-up should be.

    Algorithm (pure, Decimal, deterministic, no network):

    - ``Rt`` = the last trading day on/before ``record_date`` (handles a weekend or
      holiday record date, which is common).
    - Candidate ex sessions = the trading days at indices ``[idx(Rt) - 2, idx(Rt) - 1,
      idx(Rt)]`` — ``Rt`` and the two prior trading days.
    - For each candidate ``c`` with a prior trading day ``p``: ``gap_c =
      close(c) / close(p) - 1`` and ``theoretical = -dividend / close(p)``. Pick the
      candidate whose gap is NEGATIVE and ``|gap_c - theoretical|`` is smallest, but only
      among candidates whose drop is at least ``GAP_DETECT_MIN_FRACTION`` of the expected
      drop (``gap_c <= -0.25 * (dividend / close(p))``) — i.e. a real, detectable ex-gap.
    - If NO candidate qualifies (small / masked dividend, or missing closes), FALL BACK
      to the settlement convention: ``ex = (Rt shifted back 1 trading day)`` when
      ``record_date < 2024-01-01`` (T+2 era) else ``ex = Rt`` (T+1 era).
    - ``ldd`` = the trading day immediately BEFORE the chosen ex.

    Returns ``None`` when the calendar has no trading day on/before ``record_date`` or the
    chosen ex has no prior trading day (cannot define an LDD) — the caller skips the event.
    """
    # Rt = last trading day on/before the record date (index of first day > record - 1).
    rt_idx = bisect.bisect_right(trading_days, record_date) - 1
    if rt_idx < 0:
        return None  # no trading day on/before the record date -> cannot anchor
    rt = trading_days[rt_idx]

    # Candidate ex sessions: Rt and the two prior trading days (skip any off the front).
    best_ex: date | None = None
    best_dist: Decimal | None = None
    for lag in range(_EX_CANDIDATE_LOOKBACK, -1, -1):
        cand_idx = rt_idx - lag
        if cand_idx <= 0:
            continue  # need a prior trading day to measure the gap
        cand = trading_days[cand_idx]
        prev = trading_days[cand_idx - 1]
        cand_close = closes_by_date.get(cand)
        prev_close = closes_by_date.get(prev)
        if cand_close is None or prev_close is None or prev_close <= _ZERO:
            continue
        gap = cand_close / prev_close - _ONE
        theoretical = -dividend / prev_close
        # Require a genuine down-gap of at least GAP_DETECT_MIN_FRACTION of the expected drop.
        threshold = GAP_DETECT_MIN_FRACTION * theoretical  # theoretical is negative
        if gap >= _ZERO or gap > threshold:
            continue
        dist = abs(gap - theoretical)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_ex = cand

    if best_ex is None:
        # Fallback: settlement convention by record-date era.
        if record_date < T1_TRANSITION:
            ex_fallback = _shift_trading_days(rt, -1, trading_days)
            if ex_fallback is None:
                return None
            best_ex = ex_fallback
        else:
            best_ex = rt

    ldd = _shift_trading_days(best_ex, -1, trading_days)
    if ldd is None:
        return None  # no trading day before the ex -> cannot define an LDD
    return best_ex, ldd


def _shift_trading_days(
    anchor: date,
    offset: int,
    trading_days: list[date],
) -> date | None:
    """The trading day ``offset`` sessions from ``anchor`` (negative = earlier).

    ``anchor`` must be a trading day (it is always an LDD or ex-date produced by
    :func:`last_day_with_dividend`, so it is on the calendar). Returns ``None`` when the
    shift falls off either end of the calendar — the caller skips the event.
    """
    pos = bisect.bisect_left(trading_days, anchor)
    if pos >= len(trading_days) or trading_days[pos] != anchor:
        return None
    target = pos + offset
    if target < 0 or target >= len(trading_days):
        return None
    return trading_days[target]


def _round_trip_cost(per_side_cost: Decimal) -> Decimal:
    """The multiplicative gross-to-net haircut for a buy+sell round trip.

    A per-side proportional cost ``c`` applied on BOTH legs multiplies the gross return
    factor by ``(1 - c) ** 2``. Kept as a small named helper so both variants charge
    the identical, single-source round trip.
    """
    return (_ONE - per_side_cost) * (_ONE - per_side_cost)


def runup_return(
    prices_by_date: dict[date, Decimal],
    ldd: date,
    k: int,
    trading_days: list[date],
    per_side_cost: Decimal = RETAIL_PER_SIDE_COST,
) -> Decimal | None:
    """Variant A net return: BUY CLOSE at (LDD - k trading days), SELL CLOSE at LDD.

    Captures ONLY the pre-payout price drift and exits before the ex-gap — no dividend,
    no gap, no NDFL. The net return is
    ``(sell / buy) * (1 - c) ** 2 - 1`` (round-trip retail cost on both legs). Returns
    ``None`` when either close is missing (a no-trade session) or the ``LDD - k`` day
    falls off the calendar — the caller skips the event rather than fabricate a price.
    """
    if k <= 0:
        return None
    buy_day = _shift_trading_days(ldd, -k, trading_days)
    if buy_day is None:
        return None
    buy = prices_by_date.get(buy_day)
    sell = prices_by_date.get(ldd)
    if buy is None or sell is None or buy <= _ZERO:
        return None
    gross_factor = sell / buy
    net_factor = gross_factor * _round_trip_cost(per_side_cost)
    return net_factor - _ONE


def capture_return(
    prices_by_date: dict[date, Decimal],
    ldd: date,
    ex_date: date,
    k: int,
    m: int,
    dividend: Decimal,
    trading_days: list[date],
    ndfl: Decimal = NDFL_RATE,
    per_side_cost: Decimal = RETAIL_PER_SIDE_COST,
) -> Decimal | None:
    """Variant B net return: BUY (LDD - k), HOLD through ex, SELL (ex_date + m), collect div.

    The collect-and-hold arm EATS the ex-date gap and ADDS the net-of-NDFL dividend
    actually received. Net return =
    ``(sell + div_net) / buy * (1 - c) ** 2 - 1`` where ``div_net = dividend * (1 -
    ndfl)`` and ``c`` is the per-side cost charged on both legs. Modeling note: the
    round-trip cost is applied to the whole gross exit value INCLUDING the collected
    dividend (a conservative, single-convention haircut — it never flatters the arm).
    Returns ``None`` when a required close is missing or an anchor falls off the
    calendar.
    """
    if k <= 0 or m < 0:
        return None
    buy_day = _shift_trading_days(ldd, -k, trading_days)
    sell_day = _shift_trading_days(ex_date, m, trading_days)
    if buy_day is None or sell_day is None:
        return None
    buy = prices_by_date.get(buy_day)
    sell = prices_by_date.get(sell_day)
    if buy is None or sell is None or buy <= _ZERO:
        return None
    div_net = dividend * (_ONE - ndfl)
    gross_exit = sell + div_net
    net_factor = (gross_exit / buy) * _round_trip_cost(per_side_cost)
    return net_factor - _ONE


def ex_gap_pct(
    prices_by_date: dict[date, Decimal],
    ldd: date,
    ex_date: date,
) -> Decimal | None:
    """The realized ex-date gap ``close(ex) / close(ldd) - 1`` (a raw diagnostic).

    Pure price mechanics — no cost, no tax, no dividend. Used for the ex-gap
    decomposition (is the run-up merely the anticipated ex-date drop?). Returns
    ``None`` when either close is missing.
    """
    ldd_close = prices_by_date.get(ldd)
    ex_close = prices_by_date.get(ex_date)
    if ldd_close is None or ex_close is None or ldd_close <= _ZERO:
        return None
    return ex_close / ldd_close - _ONE


# ── Sleeve NAV construction ───────────────────────────────────────────────────

# Sleeve modes (named; no bare string literals leaking into callers/tests).
MODE_RUNUP_ONLY = "runup_only"
MODE_EQUITY_OVERLAY = "equity_overlay"

# Overlay tilt: the fraction of the long-equity book redeployed into the concentrated
# run-up window on an active bar (a CONVEX fractional tilt, NOT a whole-book multiply).
# The remaining (1 - DEPLOY_FRACTION) stays in the passive MCFTRR core. This keeps the
# overlay a realistic, unlevered position instead of stacking every event's single-name
# window factor on top of the full beta (the old, implausibly-levered artifact).
DEPLOY_FRACTION = Decimal("0.20")


@dataclass(frozen=True)
class SleeveEvent:
    """A resolved run-up window for the sleeve builder (one name, one dividend).

    ``buy_day`` = the ``LDD - k`` entry session; ``ldd`` = the exit session for the
    run-up arm. ``entry_price``/``exit_price`` are the CLOSE marks on those sessions
    (already looked up by the runner, so the sleeve builder stays pure and network-free).
    The window's active trading days are ``(buy_day, ldd]`` — the sessions over which
    the position is held and earns its slice of the net run-up return.
    """

    ticker: str
    buy_day: date
    ldd: date
    entry_price: Decimal
    exit_price: Decimal


def _net_window_factor(
    entry_price: Decimal,
    exit_price: Decimal,
    per_side_cost: Decimal,
) -> Decimal:
    """Total net growth factor of one held run-up window (round-trip cost applied once)."""
    if entry_price <= _ZERO:
        return _ONE
    return (exit_price / entry_price) * _round_trip_cost(per_side_cost)


def _per_bar_factors(
    event: SleeveEvent,
    axis: list[date],
    per_side_cost: Decimal,
) -> dict[date, Decimal]:
    """Spread one window's TOTAL net factor evenly across its active axis bars.

    The window's net factor (entry->exit, cost included) is distributed as an equal
    geometric per-bar factor over the bars in ``(buy_day, ldd]`` that are on ``axis``.
    Equal-geometric-per-bar keeps the sleeve's compounding exact end-to-end (the product
    of the per-bar factors over the window equals the window's total net factor) while
    giving a smooth daily stream, so concurrently-active windows can be averaged per bar.
    A window with no active bars on the axis contributes nothing.
    """
    active = [d for d in axis if event.buy_day < d <= event.ldd]
    if not active:
        return {}
    total = _net_window_factor(event.entry_price, event.exit_price, per_side_cost)
    if total <= _ZERO:
        return {}
    per_bar = total ** (Decimal(1) / Decimal(len(active)))
    return dict.fromkeys(active, per_bar)


def build_sleeve_nav(
    events: list[SleeveEvent],
    axis: list[date],
    mode: str,
    *,
    mcftrr_factors: dict[date, Decimal] | None = None,
    per_side_cost: Decimal = RETAIL_PER_SIDE_COST,
    initial_nav: Decimal = _ONE,
) -> list[tuple[date, Decimal]]:
    """Aggregate resolved run-up windows into a daily NET NAV curve opening at ``initial_nav``.

    ``axis`` is the sorted daily date axis (the runner's union trading calendar). Two
    modes:

    - ``MODE_RUNUP_ONLY``: a standalone TIMING stream. On a bar with NO active run-up
      window the NAV is FLAT (factor 1 — cash, zero return); on a bar with one or more
      active windows the NAV grows by the EQUAL-WEIGHT average of those windows'
      per-bar net factors (equal-weight across concurrently-active positions). Net of
      entry/exit cost by construction (the cost is inside each window factor).
    - ``MODE_EQUITY_OVERLAY``: a long-equity buy-and-hold-plus-timing stream with a
      realistic, unlevered CONVEX fractional tilt. On an IDLE bar the NAV earns the pure
      MCFTRR daily factor (``mcftrr_factors[d]``, defaulting to 1 when a bar is missing).
      On a bar inside one or more run-up windows only a ``DEPLOY_FRACTION`` slice of the
      book is redeployed into the (equal-weight averaged) window: ``day_factor =
      (1 - DEPLOY_FRACTION) * mcftrr_factor + DEPLOY_FRACTION * avg_window``. This is a
      fractional blend, NOT a whole-book multiply — it never levers the position or
      double-counts beta the way the old ``mcftrr_factor * avg_window`` artifact did.
      Requires ``mcftrr_factors``.

    ``mcftrr_factors`` maps each axis bar to that day's MCFTRR gross daily factor
    (``level[i] / level[i-1]``); pass ``None`` only for ``MODE_RUNUP_ONLY``. The NAV is
    deterministic and Decimal throughout.

    Raises ``ValueError`` for an unknown ``mode`` or a missing ``mcftrr_factors`` in
    overlay mode — a mis-wired caller must fail loudly, not silently produce a flat curve.
    """
    if mode not in (MODE_RUNUP_ONLY, MODE_EQUITY_OVERLAY):
        msg = f"unknown sleeve mode {mode!r}"
        raise ValueError(msg)
    if mode == MODE_EQUITY_OVERLAY and mcftrr_factors is None:
        msg = "equity_overlay mode requires mcftrr_factors"
        raise ValueError(msg)
    if not axis:
        return []

    # Per-bar window factors, one map per event; then averaged per bar across active windows.
    event_maps = [_per_bar_factors(ev, axis, per_side_cost) for ev in events]

    nav = initial_nav
    out: list[tuple[date, Decimal]] = [(axis[0], nav)]
    for d in axis[1:]:
        active_factors = [m[d] for m in event_maps if d in m]
        if active_factors:
            avg_window = sum(active_factors, _ZERO) / Decimal(len(active_factors))
        else:
            avg_window = _ONE
        if mode == MODE_RUNUP_ONLY:
            day_factor = avg_window
        else:
            base = (mcftrr_factors or {}).get(d, _ONE)
            if active_factors:
                # Convex fractional tilt: deploy DEPLOY_FRACTION into the window, keep
                # the rest in the passive MCFTRR core. Idle bars ride the core untouched.
                day_factor = (_ONE - DEPLOY_FRACTION) * base + DEPLOY_FRACTION * avg_window
            else:
                day_factor = base
        nav = nav * day_factor
        out.append((d, nav))
    return out


def mcftrr_daily_factors(
    mcftrr_curve: list[tuple[date, Decimal]],
) -> dict[date, Decimal]:
    """Map each MCFTRR bar (after the first) to its gross daily factor ``level[i]/level[i-1]``.

    A small pure adapter so the runner can feed the overlay mode. The first bar has no
    prior and is omitted; a non-positive prior level yields a flat factor of 1 for that
    bar (defensive — never a divide-by-zero).
    """
    out: dict[date, Decimal] = {}
    for i in range(1, len(mcftrr_curve)):
        prev = mcftrr_curve[i - 1][1]
        cur = mcftrr_curve[i][1]
        out[mcftrr_curve[i][0]] = (cur / prev) if prev > _ZERO else _ONE
    return out
