"""Dividend-event study — does a MOEX dividend RUN-UP carry a net-of-everything edge?

Token-free (public MOEX ISS-REST only; NO Tinkoff token, NO keys). The runner for the
"EQUITY DIVIDEND RUN-UP" hypothesis (docs/research/instrument_integration_program.md,
next iteration). It fetches announced dividends + TQBR daily history for the usable
universe and the MCFTRR net total-return index, builds an announced-in-advance (NO
look-ahead) event list, and measures two arms with the pure lab
(:mod:`finalayze.backtest.dividend_event_lab`):

- **Variant A — run-up-and-exit**: BUY the CLOSE ``k`` trading days before the LDD,
  SELL the CLOSE on the LDD. Captures only the pre-payout drift; never collects the
  dividend (no NDFL); gives up the ex-gap. Grid ``k in {1, 3, 5, 10, 20}``.
- **Variant B — collect-and-hold**: BUY the same ``k`` days before the LDD, HOLD
  through the ex-date, SELL ``m`` days after, ADD the net-of-13%-NDFL dividend. Eats
  the ex-gap; tests the classic mispricing (gap < net dividend?).

Both arms are NET of a 2 x 0.55% round-trip retail cost. Event dates are DETECTED
per-event from the realized close series (:func:`detect_ex_date`): the ex-date is the
session whose down-gap best matches the theoretical -dividend/close drop among the
record day and the two prior trading days, with a settlement-convention fallback
(T+2 pre-2024 => ex = record - 1 trading day; T+1 from 2024 => ex = record) when no
qualifying gap prints. This removes the old fixed-T+1 off-by-one that dated the pre-2024
ex-gap AS the LDD and so measured the gap drop, not the pre-ex run-up. The LDD is the
trading day immediately before the detected ex. The 2022 halt is handled by construction:
no trades in the halt window means no closes, so run-up/capture windows crossing it
resolve to ``None`` and are skipped.

Aggregate stats are reported OVERALL, per SEASON (calendar year), per REGIME (high-rate
vs easing split at the verified 2025-06-06 first cut), plus the ex-gap distribution and
the run-up sleeve's alpha vs MCFTRR buy-and-hold. The run-up-only and equity-overlay
sleeves are built as daily NET NAV curves; the overlay sleeve is fed to the
instrument-integration gate as a ``Candidate`` measured against the MCFTRR-net core.

Honest prior: REJECT (the run-up is almost certainly the market pricing-in a KNOWN,
announced distribution; the ex-gap ~= gross dividend, and 13% NDFL + 1.10% round-trip
swamp a few-tenths-of-a-percent drift). Let the real ISS data decide. This is a
DIAGNOSTIC study; it authorizes NOTHING — no order, no config weight, real money a hard
stop.

    uv run python scripts/research/run_dividend_event_study.py
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from finalayze.backtest.allocation_gate import REGIME_SPLIT_BOUNDARY
from finalayze.backtest.dividend_event_lab import (
    MODE_EQUITY_OVERLAY,
    MODE_RUNUP_ONLY,
    NDFL_RATE,
    SleeveEvent,
    build_sleeve_nav,
    capture_return,
    detect_ex_date,
    ex_gap_pct,
    mcftrr_daily_factors,
    runup_return,
)
from finalayze.backtest.gold_sleeve_lab import forward_align_legs, master_axis
from finalayze.backtest.instrument_integration_gate import (
    Candidate,
    run_integration_gate,
)
from finalayze.data.fetchers.moex_iss import MoexISSFetcher
from finalayze.data.loader import load_mcftr_series

if TYPE_CHECKING:
    from collections.abc import Callable

    from finalayze.backtest.instrument_integration_gate import IntegrationVerdict

_LOG = structlog.get_logger(__name__)

# ── configuration (named; no magic literals) ─────────────────────────────────
_OUT_DIR = Path("/Users/f1xgun/finalayze/.claude/worktrees/dividend-rnd/results/research")
_UNIVERSE = [
    "SBER",
    "LKOH",
    "GAZP",
    "GMKN",
    "ROSN",
    "TATN",
    "MGNT",
    "NLMK",
    "CHMF",
    "MAGN",
    "PLZL",
    "MTSS",
    "PHOR",
    "SNGS",
    "SNGSP",
    "SIBN",
    "MOEX",
    "ALRS",
    "IRAO",
    "RTKM",
    "HYDR",
]
_TQBR = "TQBR"
_MCFTRR_SECID = "MCFTRR"  # MOEX NET (after-tax) total-return index — already net
# MCFTRR comes through the index-candle path (TRADEDATE parsed MSK-midnight -> UTC),
# deterministically shifting a trade on T to the stored date T-1; +1 recovers the true
# trade date so the equity leg blends on the same true calendar as the share history.
_INDEX_SHIFT = 1

# Window: 2021 lets the run-up windows straddle the 2022 crash; end clamps to the
# allocation-gate look-ahead binding end. Dividend record dates run to 2025-08-13.
_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)  # exclusive -> last usable bar 2026-06-10
_BINDING_END = date(2026, 6, 10)
_WINDOW_START = date(2021, 1, 1)
_WINDOW_END = date(2026, 6, 10)

# The pre-payout entry grid (trading days before the LDD) and the collect-arm hold-out.
_K_GRID = (1, 3, 5, 10, 20)
_CAPTURE_M = 1  # sell 1 trading day after the ex-date (let the gap fully print)

# Regime split (verified first 2025 CBR cut). high_rate: start..2025-06-05; easing: 06-06..
_REGIME_BOUNDARY = REGIME_SPLIT_BOUNDARY

# Sleeve config: the run-up entry lead used to BUILD the daily sleeve (the mid of the grid).
_SLEEVE_K = 5
_CANDIDATE_NAME = "div_runup"
_CANDIDATE_TIER = "high"
_CANDIDATE_ROLE = "growth"

# I/O resilience: retry a transient ISS hiccup a couple times.
_FETCH_RETRIES = 3
_RETRY_BACKOFF_S = 2.0

# 2022 MOEX equity halt (Feb 28 - Mar 24 2022): record dates inside are void / non-tradeable.
_HALT_START = date(2022, 2, 28)
_HALT_END = date(2022, 3, 24)

_ZERO = Decimal(0)
_ONE = Decimal(1)
_MIN_STATS_N = 2


def _retry[T](label: str, fn: Callable[[], T]) -> T:
    """Call ``fn`` with a couple of retries so a transient ISS hiccup does not abort the run."""
    last: Exception | None = None
    backoff = _RETRY_BACKOFF_S
    for attempt in range(_FETCH_RETRIES):
        try:
            return fn()
        except Exception as exc:
            last = exc
            _LOG.warning("iss_fetch_retry", label=label, attempt=attempt + 1, error=str(exc))
            if attempt < _FETCH_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
    msg = f"fetch failed after {_FETCH_RETRIES} attempts: {label}"
    raise RuntimeError(msg) from last


# ── data structures ──────────────────────────────────────────────────────────


def _load_mcftrr() -> list[tuple[date, Decimal]]:
    """MCFTRR net TR curve on its TRUE trade-date calendar (mirrors the gold/duration runners)."""
    raw = _retry("mcftrr", partial(load_mcftr_series, _MCFTRR_SECID, _START, _END))
    shifted = [(d + timedelta(days=_INDEX_SHIFT), c) for d, c in raw]
    return [(d, c) for d, c in shifted if _WINDOW_START <= d <= _BINDING_END]


def _fetch_universe(
    fetcher: MoexISSFetcher,
) -> tuple[dict[str, list[tuple[date, Decimal, str]]], dict[str, list[tuple[date, Decimal]]]]:
    """Fetch dividends + TQBR daily history for every usable name (retry-guarded)."""
    divs: dict[str, list[tuple[date, Decimal, str]]] = {}
    hist: dict[str, list[tuple[date, Decimal]]] = {}
    for secid in _UNIVERSE:
        divs[secid] = _retry(f"dividends:{secid}", partial(fetcher.fetch_dividends, secid))
        hist[secid] = _retry(
            f"history:{secid}",
            partial(fetcher.fetch_close_history, secid, _START, _END, board=_TQBR),
        )
        _LOG.info("fetched_name", secid=secid, dividends=len(divs[secid]), bars=len(hist[secid]))
    return divs, hist


def _build_events(
    divs: dict[str, list[tuple[date, Decimal, str]]],
    hist: dict[str, list[tuple[date, Decimal]]],
) -> list[dict[str, object]]:
    """Build the announced-in-advance event list with LDD/ex anchors on each name's calendar.

    NO look-ahead: the record date + value are the PUBLIC ex-ante declaration, so anchoring a
    trade N days before a KNOWN record date uses only trade-time information. Events with a
    record date in the 2022 halt window are marked ``halt_void`` (non-tradeable), and events
    whose anchors fall off the name's realized calendar are skipped by the lab (``None``).
    """
    events: list[dict[str, object]] = []
    for secid in _UNIVERSE:
        prices = dict(hist[secid])
        cal = sorted(prices)
        for rec, value, _cur in sorted(divs[secid]):
            if not (_WINDOW_START <= rec <= _WINDOW_END):
                continue
            halt_void = _HALT_START <= rec <= _HALT_END
            anchors = detect_ex_date(prices, rec, value, cal)
            if anchors is None:
                continue
            ex_date, ldd = anchors
            gap = ex_gap_pct(prices, ldd, ex_date)
            runups = {k: runup_return(prices, ldd, k, cal) for k in _K_GRID}
            captures = {
                k: capture_return(prices, ldd, ex_date, k, _CAPTURE_M, value, cal, ndfl=NDFL_RATE)
                for k in _K_GRID
            }
            events.append(
                {
                    "ticker": secid,
                    "record_date": rec.isoformat(),
                    "ldd": ldd.isoformat(),
                    "ex_date": ex_date.isoformat(),
                    "value_gross": str(value),
                    "halt_void": halt_void,
                    "ex_gap_pct": None if gap is None else str(gap),
                    "runup_net": {
                        str(k): (None if v is None else str(v)) for k, v in runups.items()
                    },
                    "capture_net": {
                        str(k): (None if v is None else str(v)) for k, v in captures.items()
                    },
                    "year": rec.year,
                    "regime": "easing" if rec >= _REGIME_BOUNDARY else "high_rate",
                }
            )
    return events


# ── aggregation ──────────────────────────────────────────────────────────────


def _stats(values: list[Decimal]) -> dict[str, object]:
    """Mean / hit-rate / count of a list of net returns (Decimal); empty -> nulls."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean_pct": None, "hit_rate": None, "min_pct": None, "max_pct": None}
    mean = sum(values, _ZERO) / Decimal(n)
    hits = sum(1 for v in values if v > _ZERO)
    return {
        "n": n,
        "mean_pct": round(float(mean) * 100, 4),
        "hit_rate": round(hits / n, 4),
        "min_pct": round(float(min(values)) * 100, 4),
        "max_pct": round(float(max(values)) * 100, 4),
    }


def _collect(events: list[dict[str, object]], arm: str, k: int) -> list[Decimal]:
    """All non-void, non-None net returns for one arm+k across the events (survivorship-safe)."""
    out: list[Decimal] = []
    for ev in events:
        if ev["halt_void"]:
            continue
        raw = ev[arm][str(k)]  # type: ignore[index]
        if raw is not None:
            out.append(Decimal(str(raw)))
    return out


def _aggregate(events: list[dict[str, object]]) -> dict[str, object]:
    """Overall + per-season + per-regime + ex-gap stats for both arms across the k-grid."""
    arms = ("runup_net", "capture_net")
    overall = {arm: {str(k): _stats(_collect(events, arm, k)) for k in _K_GRID} for arm in arms}

    def _subset_stats(subset: list[dict[str, object]]) -> dict[str, object]:
        return {arm: {str(k): _stats(_collect(subset, arm, k)) for k in _K_GRID} for arm in arms}

    years = sorted({int(str(e["year"])) for e in events})
    per_season = {
        str(y): _subset_stats([e for e in events if int(str(e["year"])) == y]) for y in years
    }
    per_regime = {
        r: _subset_stats([e for e in events if e["regime"] == r]) for r in ("high_rate", "easing")
    }

    ex_gaps = [
        Decimal(str(e["ex_gap_pct"]))
        for e in events
        if not e["halt_void"] and e["ex_gap_pct"] is not None
    ]
    return {
        "overall": overall,
        "per_season": per_season,
        "per_regime": per_regime,
        "ex_gap_distribution": _stats(ex_gaps),
    }


def _build_sleeve_events(
    events: list[dict[str, object]],
    hist: dict[str, list[tuple[date, Decimal]]],
    k: int,
) -> list[SleeveEvent]:
    """Resolve the (buy_day, ldd, entry, exit) run-up windows for the sleeve builder."""
    resolved: list[SleeveEvent] = []
    for ev in events:
        if ev["halt_void"]:
            continue
        secid = str(ev["ticker"])
        prices = dict(hist[secid])
        cal = sorted(prices)
        ldd = date.fromisoformat(str(ev["ldd"]))
        pos = cal.index(ldd) if ldd in prices else -1
        if pos < k:
            continue
        buy_day = cal[pos - k]
        entry, exit_px = prices.get(buy_day), prices.get(ldd)
        if entry is None or exit_px is None or entry <= _ZERO:
            continue
        resolved.append(
            SleeveEvent(
                ticker=secid, buy_day=buy_day, ldd=ldd, entry_price=entry, exit_price=exit_px
            )
        )
    return resolved


def _total_return_pct(nav: list[tuple[date, Decimal]]) -> float:
    if len(nav) < _MIN_STATS_N or nav[0][1] <= _ZERO:
        return 0.0
    return float(nav[-1][1] / nav[0][1] - _ONE) * 100.0


def _dump_nav(nav: list[tuple[date, Decimal]]) -> list[list[str]]:
    return [[d.isoformat(), str(v)] for d, v in nav]


def _write(name: str, payload: object) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def main() -> None:
    mcftrr = _load_mcftrr()
    with MoexISSFetcher() as fetcher:
        divs, hist = _fetch_universe(fetcher)

    events = _build_events(divs, hist)
    agg = _aggregate(events)

    # ── daily sleeves on the shared union trading axis ──────────────────────────
    axis = [
        d
        for d in master_axis({"m": mcftrr, **{s: h for s, h in hist.items() if h}})
        if _WINDOW_START <= d <= _BINDING_END
    ]
    mcftrr_aligned = list(zip(axis, forward_align_legs({"m": mcftrr}, axis)["m"], strict=True))
    mcftrr_factors = mcftrr_daily_factors(mcftrr_aligned)

    sleeve_events = _build_sleeve_events(events, hist, _SLEEVE_K)
    runup_nav = build_sleeve_nav(sleeve_events, axis, MODE_RUNUP_ONLY)
    overlay_nav = build_sleeve_nav(
        sleeve_events, axis, MODE_EQUITY_OVERLAY, mcftrr_factors=mcftrr_factors
    )

    # Alpha vs buy-and-hold MCFTRR (the overlay's timing tilt, net of cost).
    overlay_tr = _total_return_pct(overlay_nav)
    mcftrr_tr = _total_return_pct(mcftrr_aligned)
    overlay_alpha_pp = round(overlay_tr - mcftrr_tr, 4)

    # ── the pre-registered integration gate on the overlay sleeve ──────────────
    candidate = Candidate(
        name=_CANDIDATE_NAME,
        net_curve=overlay_nav,
        risk_tier=_CANDIDATE_TIER,
        intended_role=_CANDIDATE_ROLE,
    )
    verdict: IntegrationVerdict = run_integration_gate(candidate, equity_curve=mcftrr_aligned)
    sc = verdict.scorecard

    # ── artifacts ───────────────────────────────────────────────────────────────
    _write("dividend_events.json", {"n_events": len(events), "events": events, "aggregate": agg})
    _write(
        "sleeve_runup.json",
        {
            "mode": MODE_RUNUP_ONLY,
            "k": _SLEEVE_K,
            "n_windows": len(sleeve_events),
            "total_return_pct": round(_total_return_pct(runup_nav), 4),
            "nav": _dump_nav(runup_nav),
        },
    )
    _write(
        "sleeve_overlay.json",
        {
            "mode": MODE_EQUITY_OVERLAY,
            "k": _SLEEVE_K,
            "n_windows": len(sleeve_events),
            "total_return_pct": round(overlay_tr, 4),
            "mcftrr_buy_hold_pct": round(mcftrr_tr, 4),
            "alpha_vs_buy_hold_pp": overlay_alpha_pp,
            "nav": _dump_nav(overlay_nav),
        },
    )
    _write(
        "mcftrr_equity.json",
        {
            "secid": _MCFTRR_SECID,
            "index_shift_days": _INDEX_SHIFT,
            "total_return_pct": round(mcftrr_tr, 4),
            "nav": _dump_nav(mcftrr_aligned),
        },
    )
    gate_payload = {
        "candidate": {
            "name": candidate.name,
            "risk_tier": candidate.risk_tier,
            "intended_role": candidate.intended_role,
        },
        "verdict": verdict.tier,
        "proposed_weight": str(verdict.proposed_weight),
        "carved_from": verdict.carved_from,
        "n1_caveat": verdict.n1_caveat,
        "reasons": verdict.reasons,
        "scorecard": {
            "window_bars": sc.window_bars,
            "regimes_covered": sc.regimes_covered,
            "tail_backtestable": sc.tail_backtestable,
            "marginal_sharpe_delta": sc.marginal_sharpe_delta,
            "marginal_sortino_delta": sc.marginal_sortino_delta,
            "marginal_maxdd_delta_pp": sc.marginal_maxdd_delta_pp,
            "crash_year_maxdd_delta_pp": sc.crash_year_maxdd_delta_pp,
            "max_corr_to_existing_legs": sc.max_corr_to_existing_legs,
            "anti_hollow_ok": sc.anti_hollow_ok,
        },
    }
    _write("dividend_gate_verdict.json", gate_payload)

    # ── compact console summary ─────────────────────────────────────────────────
    print(
        f"events={len(events)} sleeve_windows={len(sleeve_events)} "
        f"axis={len(axis)} bars [{axis[0]}..{axis[-1]}]"
    )
    for k in _K_GRID:
        ra = _stats(_collect(events, "runup_net", k))
        ca = _stats(_collect(events, "capture_net", k))
        print(
            f"  k={k:>2}: runup n={ra['n']:>3} mean={ra['mean_pct']}% hit={ra['hit_rate']} | "
            f"capture n={ca['n']:>3} mean={ca['mean_pct']}% hit={ca['hit_rate']}"
        )
    exg = _stats(
        [
            Decimal(str(e["ex_gap_pct"]))
            for e in events
            if not e["halt_void"] and e["ex_gap_pct"] is not None
        ]
    )
    print(
        f"  ex-gap: n={exg['n']} mean={exg['mean_pct']}% "
        f"[min {exg['min_pct']} max {exg['max_pct']}]"
    )
    print(
        f"  overlay TR={overlay_tr:+.2f}% vs MCFTRR buy&hold {mcftrr_tr:+.2f}% "
        f"-> alpha {overlay_alpha_pp:+.2f}pp"
    )
    print(
        f"  GATE verdict={verdict.tier} weight={verdict.proposed_weight} "
        f"sharpe_delta={sc.marginal_sharpe_delta:+.4f} "
        f"maxdd_cut_pp={sc.marginal_maxdd_delta_pp:+.2f} "
        f"max_corr={sc.max_corr_to_existing_legs:.2f} n1={verdict.n1_caveat}"
    )
    print(f"  reasons: {verdict.reasons}")


if __name__ == "__main__":
    main()
