"""Phase 73 allocation-gate CLI harness + backtest-iteration cert (GATE-01/02/03, CLAUDE.md #4).

The thin CLI wrapper for the FROZEN W2 allocator's §7 measurement gate. It loads (or
deterministically synthesizes) the three benchmark total-return curves -- deposit,
OFZ-PK, equity(MCFTR) -- runs the feature-complete ``finalayze.backtest.allocation_gate``
module (Plans 02-04) end-to-end, and writes the D-11 artifacts: a Markdown report +
a JSON sidecar under ``results/iterations/<run>/`` plus one ``history.jsonl`` verdict
line. This satisfies CLAUDE.md #4 (any ``backtest/`` + scripts change triggers the
mandatory ``backtest-iteration`` cert).

D-12 -- the gate LOGIC lives in the module; this script stays THIN: it only
(1) builds the curves, (2) calls the module, (3) writes the artifacts. No new
allocation/metric logic is introduced here.

CRITICAL -- the Phase 72 anti-hollow lesson: a GREEN A/B can be HOLLOW if the verdict
fires only via a test-only hook (Phase 72's cost/NDFL charged 0 on real runs because a
``forced_leg_deltas`` hook masked it). This cert computes its per-profile PASS/FAIL
through the REAL gate path -- it calls the actual ``build_naive_legs`` ->
``gate_with_autotighten`` on the actual curves, NOT a fixture that pre-bakes the
verdict. There is NO hardcoded pre-baked verdict literal anywhere: every per-profile
verdict is whatever ``gate_with_autotighten`` returns (PASS / PASS_AFTER_TIGHTEN /
HARD_FAIL). A HARD_FAIL is a legitimate, honest, non-softened outcome.

Window: a multi-year daily window that SPANS the documented regime boundary
(``REGIME_SPLIT_BOUNDARY`` = 2025-06-06) so the report renders BOTH the high-rate and
the early-cut regimes. The deposit leg is flat-ish (the strong 16-21% high-rate anchor),
OFZ-PK slowly rises, equity rises faster -- the honest deposit-anchored geometry the §7
thesis is judged against.

Exit-code convention (documented per the plan): ``main()`` returns ``0`` when the run
COMPLETED and wrote BOTH artifacts (summary.json + report.md) and the history line. The
per-profile PASS/FAIL is REPORTED, NOT forced into the exit code -- a HARD_FAIL is an
honest outcome, so the CLI does NOT soften the verdict to coerce a 0 exit. A non-zero
exit signals a HARNESS failure (an artifact was not written / a module call raised),
never a HARD_FAIL verdict.

Logs the run under ``results/iterations/allocation-gate-73-*`` with a ``history.jsonl``
verdict line carrying the SET of three per-profile verdicts (D-06 -- the overall record
is the honest set, never a single softened pass).

Usage::

    uv run python scripts/run_allocation_gate.py            # deterministic offline cert (default)
    uv run python scripts/run_allocation_gate.py --live     # REAL MCFTR/CBR cert (operator)

``--live`` (operator-directed, an explicit override of D-10): the binding v11.0 cert on
the REAL series -- the genuine MCFTR gross total-return index (public ISS REST, no token)
as the equity leg + benchmark bar, and the deposit/OFZ-PK legs daily-accrued from the REAL
look-ahead-safe ``CBR_MEETINGS`` key-rate calendar (deposit = key-1pp, OFZ-PK floater =
full key rate). All three legs share the MCFTR trading-day axis (R-3). The per-profile
verdict is whatever ``gate_with_autotighten`` returns on the real curves -- a HARD_FAIL on
real data is a legitimate, non-softened, honest outcome (the Phase-72 anti-hollow lesson).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finalayze.backtest import allocation_gate as _gate
from finalayze.backtest.allocation_gate import (
    _EASING_UNIT_KEY,
    _HARD_FAIL,
    _HIGH_RATE_UNIT_KEY,
    build_naive_legs,
    derive_escalation,
    excess_sortino_from_equity,
    gate_with_autotighten,
    net_fixed_income_legs_interleaved,
    regime_split,
    regime_verdicts,
    render_json,
    render_report,
)
from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.ndfl import YtdTaxAccumulator
from finalayze.core.schemas import RiskProfile
from finalayze.data.loader import load_mcftr_series

if TYPE_CHECKING:
    from finalayze.orchestration.allocation import AllocationResult

# ── Named-constant header block (no PLR2004 magic numbers) ───────────────────

_PHASE = "73"
_RUN_PREFIX = "allocation-gate-73"
_ITER_DIR = Path(__file__).resolve().parent.parent / "results" / "iterations"

# Deterministic offline window: a multi-year daily span that BRACKETS the documented
# regime boundary (REGIME_SPLIT_BOUNDARY = 2025-06-06) so the report renders BOTH the
# high-rate and the early-cut regimes (D-09 / R-6). Start well before, end well after.
_FIRST_BAR = date(2024, 1, 1)
_N_BARS = 700  # ~2.8y of daily bars -> spans 2025-06-06 + many quarter boundaries

# Curve geometry (deterministic, no network, no token -- a CI-safe reproducible cert).
# Deposit is the flat-ish high-rate anchor; OFZ-PK is a slow carry leg; equity rises
# faster (the honest deposit-anchored geometry the §7 thesis is judged against). Each leg
# carries a SEEDED, reproducible daily noise overlay (_*_VOL) so the naive legs have
# REALISTIC volatility -- a NOISE-FREE monotone geometric curve produces a degenerate
# (near-infinite) Sharpe/Sortino bar that no candidate can ever clear, which would make
# the verdict a fixture artifact, not an honest risk-adjusted measurement. The noise is
# drawn from a fixed-seed RNG (_RNG_SEED), so two runs are byte-identical (reproducible
# cert) while the Sharpe/Sortino comparison stays meaningful and informative.
_DEPOSIT_BASE = Decimal(100_000)
_OFZ_BASE = Decimal(100_000)
_EQUITY_BASE = Decimal(100_000)
_DEPOSIT_DAILY = Decimal("1.00055")  # ~15%/yr flat deposit accrual (the 16-21% anchor)
_OFZ_DAILY = Decimal("1.0004")  # slowly-rising OFZ-PK carry leg
_EQUITY_DAILY = Decimal("1.0008")  # faster-rising equity (MCFTR) sleeve

# Per-leg daily lognormal-ish noise sigma (deterministic via _RNG_SEED). Deposit is the
# near-risk-free anchor (tiny vol), OFZ-PK a low-vol carry leg, equity the volatile sleeve.
_DEPOSIT_VOL = 0.0002
_OFZ_VOL = 0.0010
_EQUITY_VOL = 0.0090  # ~14%/yr annualized -> realistic MOEX equity volatility
_RNG_SEED = 73  # fixed seed -> reproducible cert (CLAUDE.md: no magic, named constant)

_TRADING_DAYS_PER_YEAR = 252
_PERCENT = Decimal(100)

# ── Live window (operator-directed --live run, D-10 override) ────────────────
# The operator REJECTED the seeded-synthetic fixture as the binding v11.0 cert and
# required the cert computed on the REAL MOEX/CBR series. Phase 74 (REGIME-01/04, D-02/
# D-04/D-05) puts the binding cert on a fair after-tax basis over the FULL easing cycle:
# the window now ends 2026-06-10 (the verified terminal-rate bar, R-C/R-D) so the report
# renders BOTH the high-rate plateau and the verified 2025-06-06+ easing on REAL data.
#   - equity = MCFTRR (MOEX's published net-of-tax total-return index, D-02) — already net.
#   - OFZ-PK = RUFLBITR (real MOEX floating-coupon-bond TR index, D-04) — fetched GROSS,
#     then netted of NDFL (D-01 derived implication).
#   - deposit = accrued from the REAL look-ahead-safe CBR_MEETINGS key-rate path, netted.
# The two fixed-income legs net through ONE shared YtdTaxAccumulator per run (cross-leg
# YTD, the W1 design); MCFTRR is already net and is NEVER routed through the accumulator.
_EQUITY_SECID = "MCFTRR"  # MOEX Russia Net-TR-Res — the net-of-tax equity benchmark (D-02)
_OFZ_SECID = "RUFLBITR"  # MOEX Floating-Coupon Bond TR index — the real OFZ-PK leg (D-04)
_LIVE_START = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017
_LIVE_END = datetime(2026, 6, 10, tzinfo=timezone.utc)  # noqa: UP017  # full easing cycle (D-05)
# Real legs share ONE common daily axis (the MCFTRR trading-day calendar): the deposit +
# OFZ-PK legs align to the EXACT MCFTRR dates so build_naive_legs sees one basis (R-3).
_LIVE_DEPOSIT_BASE = Decimal(100_000)
_LIVE_OFZ_BASE = Decimal(100_000)
# Deposit accrues at key-1pp (mirrors W1's deposit spread); OFZ-PK floater tracks the full
# key rate (no spread). Both read the REAL CBR calendar via accrue_real_risk_free_leg.
_LIVE_DEPOSIT_SPREAD_PP = Decimal("1.0")
_LIVE_OFZ_SPREAD_PP = Decimal(0)
# A real ~2.5y MCFTRR window has ~610+ trading-day bars; refuse a hollow run if a fetch
# returns far fewer (HONESTY GATE: never fabricate a synthetic 'live' leg, T-73-12).
_N_LIVE_MIN_BARS = 300

# ── Committed real-data snapshot (REGIME-01 / D-05) ──────────────────────────
# Default --live (and CI) reads the committed snapshot via the FROZEN Plan-02 fail-closed
# loader (deterministic, no network); --live --refresh-snapshot fetches + nets + WRITES it.
# Re-exported here at module scope so the path can be monkeypatched in tests. The binding
# committed fixture itself is produced by the OPERATOR in Plan 04 (the real network fetch),
# never here. The three R-F leg keys mirror the Plan-02 loader's _SNAPSHOT_LEG_KEYS.
_GATE_SNAPSHOT = _gate._GATE_SNAPSHOT  # the committed snapshot path (Plan 02)
_load_gate_snapshot = _gate._load_gate_snapshot  # the fail-closed reader (Plan 02)
_SNAP_LEG_EQUITY = "equity_mcftrr_net"
_SNAP_LEG_OFZ = "ofz_ruflbitr_net"
_SNAP_LEG_DEPOSIT = "deposit_net"

# The three SAA profiles, in the conservative -> balanced -> growth order the report
# renders them. Each carries its own MaxDD cap (8 / 15 / 25%) read from the loaded
# AllocationProfile (load_allocation_profiles), never hardcoded here.
_PROFILE_ORDER = (RiskProfile.CONSERVATIVE, RiskProfile.BALANCED, RiskProfile.GROWTH)

# ── Phase 75 (REGIME-02/05) 3-unit phase-verdict wiring constants ─────────────
# The binding phase verdict = full_window AND high_rate AND easing -> HARD_FAIL if ANY
# PRESENT unit HARD_FAILs (the honest AND across the three units). These are ALIASES of the
# module's canonical constants (IN-03): re-importing instead of re-declaring string copies keeps
# a single source of truth, so a future relabel of a unit key (e.g. "early_cut") propagates here
# automatically instead of silently diverging and KeyError-ing on per_regime[_EASING_UNIT].
_PHASE_HARD_FAIL = _HARD_FAIL  # the terminal HARD_FAIL verdict (gate_with_autotighten output)
_HIGH_RATE_UNIT = _HIGH_RATE_UNIT_KEY  # the high-rate regime unit key (regime_split output)
_EASING_UNIT = _EASING_UNIT_KEY  # the easing binding unit key (regime_split's post-cut segment)
# A non-HARD_FAIL sentinel for the absent-easing single-regime edge: pass it to
# derive_escalation so the escalation stays None when only the high_rate unit exists.
_EASING_ABSENT_VERDICT = "PASS"


def _unit_verdict(per_profile: dict[str, object]) -> str:
    """Collapse a unit's per-profile verdicts to one unit verdict (honest AND-within-unit).

    A unit HARD_FAILs if ANY profile in it HARD_FAILs; otherwise it inherits the
    conjunctive PASS-set joined by ``+`` (the same honest-set shape as the full-window
    ``overall``). No magic literal — uses the named :data:`_PHASE_HARD_FAIL`.
    """
    verdicts = [str(cast("dict[str, object]", v)["verdict"]) for v in per_profile.values()]
    if _PHASE_HARD_FAIL in verdicts:
        return _PHASE_HARD_FAIL
    return "+".join(verdicts)


def _dates() -> list[date]:
    return [_FIRST_BAR + timedelta(days=i) for i in range(_N_BARS)]


def _curve(
    base: Decimal,
    daily: Decimal,
    vol: float,
    dates: list[date],
    rng: random.Random,
) -> list[tuple[date, Decimal]]:
    """Deterministic geometric TR curve with a SEEDED daily noise overlay (reproducible).

    Each bar grows by the deterministic drift ``daily`` times a seeded multiplicative
    noise factor ``(1 + rng.gauss(0, vol))`` (clamped > 0). The RNG is supplied seeded by
    the caller so two runs are byte-identical (a reproducible cert) -- the noise exists so
    the naive legs have REALISTIC volatility and the Sharpe/Sortino bar is finite and
    informative, never the degenerate near-infinite bar a noise-free monotone curve gives.
    """
    out: list[tuple[date, Decimal]] = []
    value = base
    for i, d in enumerate(dates):
        if i == 0:
            out.append((d, value))
            continue
        # Deterministic SEEDED test-fixture noise (the RNG is seeded by the caller).
        shock = max(0.0, 1.0 + rng.gauss(0.0, vol))
        value = value * daily * Decimal(str(shock))
        out.append((d, value))
    return out


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_iteration(name: str, payload: dict[str, object]) -> Path:
    """Write ``summary.json`` under ``results/iterations/<name>/`` (mirrors phase72)."""
    d = _ITER_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return d


def _append_history(name: str, *, git_sha: str, verdict: str, metrics: dict[str, object]) -> None:
    """Append one history.jsonl verdict line (mirrors phase72_allocation_ab.py)."""
    hist = _ITER_DIR / "history.jsonl"
    entry = {
        "name": name,
        "phase": _PHASE,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "git_sha": git_sha,
        "verdict": verdict,
        **metrics,
    }
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _fetch_real_series(secid: str) -> list[tuple[date, Decimal]]:
    """Fetch ONE real ISS-REST index series, fail-closed (the network-seam honesty gate).

    Wraps ``load_mcftr_series(secid=...)`` over the public MOEX ISS-REST index endpoint
    (no token/cert) with the WR-03 / T-73-12 honesty guards applied IDENTICALLY to every
    secid (MCFTRR and RUFLBITR): a raising fetch or a short fetch surfaces a clean,
    actionable operator ``SystemExit`` -- NEVER a fabricated synthetic 'live' leg. A
    non-zero exit signals a HARNESS failure (MOEX unreachable), not a HARD_FAIL verdict.
    """
    try:
        series = load_mcftr_series(secid=secid, start=_LIVE_START, end=_LIVE_END)
    except Exception as exc:  # operator-facing legibility at the network seam (WR-03)
        msg = (
            f"--live {secid} fetch failed ({type(exc).__name__}: {exc}); "
            "MOEX ISS-REST unreachable -- refusing to fabricate a synthetic 'live' leg."
        )
        raise SystemExit(msg) from exc
    if len(series) < _N_LIVE_MIN_BARS:
        msg = (
            f"--live {secid} fetch returned only {len(series)} bars over "
            f"{_LIVE_START.date()}..{_LIVE_END.date()} (need >= {_N_LIVE_MIN_BARS}); "
            "real data unavailable -- refusing to fabricate a synthetic 'live' leg."
        )
        raise SystemExit(msg)
    return series


def _forward_align(
    series: list[tuple[date, Decimal]], axis: list[date]
) -> list[tuple[date, Decimal]]:
    """Forward-fill ``series`` onto the master ``axis`` (the as-of, look-ahead-safe convention).

    RUFLBITR is fetched on its own trading-day calendar; the gate needs all three legs on
    the ONE master (MCFTRR) axis (R-3). For each master date the most-recent series level
    on/before that date is used (the same forward-fill the orchestrator applies), so no
    future bar leaks. Master dates before the first series bar carry the first level.
    """
    if not series:
        return [(d, Decimal(0)) for d in axis]
    out: list[tuple[date, Decimal]] = []
    j = 0
    last = series[0][1]
    for d in axis:
        while j < len(series) and series[j][0] <= d:
            last = series[j][1]
            j += 1
        out.append((d, last))
    return out


def _load_live_curves() -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Load the three REAL, after-tax total-return curves for the ``--live`` cert (Phase 74).

    Operator decision (D-10 override): the seeded-synthetic fixture is REJECTED as the
    binding cert; it MUST be computed on the REAL MOEX/CBR series. Phase 74 (REGIME-01/04,
    D-01/D-02/D-04) puts all three legs on a fair AFTER-TAX (net-of-NDFL) basis:

    - **equity (MCFTRR)** -- REAL + ALREADY NET: ``load_mcftr_series(secid="MCFTRR")`` is
      MOEX's published net-of-tax total-return index (D-02; ``_EQUITY_SECID``). It is BOTH
      the equity sleeve and the 100%-equity benchmark bar (apples-to-apples) and is returned
      UNCHANGED — it is NEVER routed through the NDFL accumulator (Pitfall 1: double-tax).
    - **OFZ-PK (RUFLBITR)** -- REAL, fetched GROSS then NETTED:
      ``load_mcftr_series(secid="RUFLBITR")`` is the real MOEX floating-coupon-bond TR index
      (D-04; ``_OFZ_SECID``), forward-aligned onto the MCFTRR axis and netted of NDFL via
      :func:`net_index_returns` (the gross→net D-01 derived implication).
    - **deposit** -- REAL, NETTED: ``accrue_real_risk_free_leg`` daily-compounds at the
      as-of ``deposit_rate_as_of`` (key - 1pp) from the REAL ``CBR_MEETINGS`` calendar,
      netted of NDFL via the SAME shared accumulator.

    The MCFTRR trading-day calendar is the MASTER axis (R-3). The deposit + RUFLBITR-OFZ
    legs net through ONE shared :class:`YtdTaxAccumulator` per run (cross-leg YTD, the W1
    cross-sleeve design); MCFTRR (already net) is left as-is.
    """
    equity_curve = _fetch_real_series(_EQUITY_SECID)  # MCFTRR — already net (D-02)
    ruflbitr_gross = _fetch_real_series(_OFZ_SECID)  # RUFLBITR — gross, netted below (D-04)

    # The MCFTRR trading-day calendar IS the common axis (R-3).
    axis = [d for d, _ in equity_curve]
    # ONE shared accumulator per run so the deposit + OFZ legs share one cross-leg YTD (W1).
    tax_acc = YtdTaxAccumulator()
    # CR-01: net BOTH fixed-income legs INTERLEAVED BY DATE in ONE shared, date-ordered pass
    # — NOT leg-by-leg. A leg-by-leg netting (full OFZ pass, then full deposit pass through the
    # same accumulator) silently breaks the W1 cross-leg-YTD contract on a multi-tax-year
    # window: the deposit leg's first (earliest) bar would trigger a Jan-1 reset that wipes the
    # OFZ leg's accumulated YTD. Interleaving by date makes both legs' daily increments hit the
    # SAME running YTD before any year-boundary reset (the band crossover the shared YTD exists
    # to detect). On this cert's window both legs stay in the 13% band, so the netted curves are
    # byte-identical to the old leg-by-leg result — only the multi-year correctness changes.
    ruflbitr_on_axis = _forward_align(ruflbitr_gross, axis)
    deposit_curve, ofz_pk_curve = net_fixed_income_legs_interleaved(
        ruflbitr_on_axis,
        axis,
        _LIVE_DEPOSIT_BASE,
        deposit_spread_pp=_LIVE_DEPOSIT_SPREAD_PP,
        tax_acc=tax_acc,
    )
    return deposit_curve, ofz_pk_curve, equity_curve


def _clamp_leg(leg: list[tuple[date, Decimal]]) -> list[list[str]]:
    """Serialize ONE leg to ``[[iso_date, decimal_str], ...]``, clamped to ``_LIVE_END`` (Pitfall3).

    Drops any bar dated after ``_LIVE_END`` (the look-ahead guard, T-74-07): a refresh run on
    a later calendar day cannot leak a future bar into the committed fixture. Decimal is
    serialized as a STRING and date as an ISO string — the exact Phase-65 ``_row_to_instrument``
    re-hydration convention the Plan-02 loader reads back Decimal-exact.
    """
    binding_end = _LIVE_END.date()
    return [[d.isoformat(), str(v)] for d, v in leg if d <= binding_end]


def _write_gate_snapshot(
    deposit: list[tuple[date, Decimal]],
    ofz: list[tuple[date, Decimal]],
    equity: list[tuple[date, Decimal]],
    *,
    start: date,
    end: date,
    git_sha: str,
    path: Path = _GATE_SNAPSHOT,
) -> Path:
    """Write the committed real-data snapshot fixture (R-F shape) — the ONLY network-write path.

    Serializes the three NETTED legs (``equity_mcftrr_net`` / ``ofz_ruflbitr_net`` /
    ``deposit_net``) to the Phase-65 committed-snapshot shape (Decimal-as-string, ISO dates),
    clamped to ``_LIVE_END`` (look-ahead guard, Pitfall 3 / T-74-07), and creates the
    ``src/finalayze/backtest/data/`` directory if absent. Only ``--live --refresh-snapshot``
    calls this; the default ``--live`` read path never writes. The committed binding fixture
    is produced by the OPERATOR in Plan 04 (the real network fetch), not in CI.
    """
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "git_sha": git_sha,
        "legs": {
            _SNAP_LEG_EQUITY: _clamp_leg(equity),
            _SNAP_LEG_OFZ: _clamp_leg(ofz),
            _SNAP_LEG_DEPOSIT: _clamp_leg(deposit),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)  # create src/finalayze/backtest/data/
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_binding_curves(
    *, refresh_snapshot: bool
) -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Resolve the three binding ``--live`` curves: refresh-fetch+write, or read the snapshot.

    - ``refresh_snapshot=True`` (operator network path): fetch + net the REAL series via
      :func:`_load_live_curves`, WRITE the committed fixture via :func:`_write_gate_snapshot`,
      and use those curves.
    - ``refresh_snapshot=False`` (default ``--live`` + CI): READ the committed snapshot via
      the FROZEN Plan-02 fail-closed :func:`_load_gate_snapshot` (deterministic, NO network).

    Either way the verdict still flows through the REAL ``build_naive_legs ->
    gate_with_autotighten`` path on whatever curves were loaded (anti-hollow: no test-only
    hook, no pre-baked verdict literal — the W2/Phase-72 lesson). Returns
    ``(deposit, ofz_pk, equity)`` — the same leg order :func:`_load_live_curves` returns.
    """
    if refresh_snapshot:
        deposit, ofz_pk, equity = _load_live_curves()
        _write_gate_snapshot(
            deposit,
            ofz_pk,
            equity,
            start=_LIVE_START.date(),
            end=_LIVE_END.date(),
            git_sha=_git_sha(),
            path=_GATE_SNAPSHOT,
        )
        return deposit, ofz_pk, equity
    # Default --live / CI: the FROZEN fail-closed snapshot reader returns (equity, ofz, deposit).
    equity, ofz_pk, deposit = _load_gate_snapshot(_GATE_SNAPSHOT)
    return deposit, ofz_pk, equity


def _load_curves(
    *, live: bool, refresh_snapshot: bool, dates: list[date]
) -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Load the three benchmark TR curves.

    Three paths:

    - ``live=True, refresh_snapshot=False`` (the binding default + CI): READ the committed
      net-of-tax snapshot via the FROZEN Plan-02 fail-closed loader (deterministic, no
      network).
    - ``live=True, refresh_snapshot=True`` (operator network path): FETCH + net the REAL
      MCFTRR/RUFLBITR series and (re)write the committed fixture.
    - ``live=False`` (the non-binding CI smoke): the deterministic in-memory geometric
      curves -- CI-safe, reproducible, no token, no network. The offline ``dates`` argument
      is used only here.
    """
    if live:
        return _load_binding_curves(refresh_snapshot=refresh_snapshot)
    # One fixed-seed RNG per leg (seed offset by leg index) so the three noise streams are
    # independent yet fully reproducible -- two runs are byte-identical (CI-safe cert).
    # S311: seeded deterministic test-fixture RNGs, not a cryptographic use.
    dep_rng = random.Random(_RNG_SEED)  # noqa: S311
    ofz_rng = random.Random(_RNG_SEED + 1)  # noqa: S311
    eq_rng = random.Random(_RNG_SEED + 2)  # noqa: S311
    deposit_curve = _curve(_DEPOSIT_BASE, _DEPOSIT_DAILY, _DEPOSIT_VOL, dates, dep_rng)
    ofz_pk_curve = _curve(_OFZ_BASE, _OFZ_DAILY, _OFZ_VOL, dates, ofz_rng)
    equity_curve = _curve(_EQUITY_BASE, _EQUITY_DAILY, _EQUITY_VOL, dates, eq_rng)
    return deposit_curve, ofz_pk_curve, equity_curve


def _naive_metrics(naives: dict[str, AllocationResult]) -> dict[str, object]:
    """Flatten the three naive legs into a reportable Sharpe/Sortino/MaxDD block (D-04)."""
    out: dict[str, object] = {}
    for name, leg in naives.items():
        sortino = excess_sortino_from_equity([float(v) for v in leg.merged_equity_curve])
        out[f"{name}_sharpe"] = leg.sharpe
        out[f"{name}_sortino"] = sortino
        out[f"{name}_maxdd_pct"] = leg.max_drawdown_pct
    return out


def run_gate(
    *, live: bool, git_sha: str, refresh_snapshot: bool = False
) -> tuple[dict[str, object], str, str]:
    """Drive the REAL gate end-to-end and assemble the JSON payload + Markdown report.

    The Phase 72 anti-hollow contract: every per-profile verdict comes from the REAL
    ``gate_with_autotighten`` on the REAL ``build_naive_legs`` output -- NOT a pre-baked
    constant -- whether the curves came from the committed snapshot, a refresh fetch, or the
    offline smoke. Returns ``(payload, report_md, overall_verdict)`` where ``overall_verdict``
    is the honest SET of the three per-profile verdicts (D-06).
    """
    dates = _dates()
    deposit_curve, ofz_pk_curve, equity_curve = _load_curves(
        live=live, refresh_snapshot=refresh_snapshot, dates=dates
    )

    # 1. The three naive benchmark legs on ONE basis (R-3) -- the bar the candidate clears.
    naives = build_naive_legs(deposit_curve, ofz_pk_curve, equity_curve)
    naive_sharpes = [n.sharpe for n in naives.values()]
    naive_sortinos = [
        excess_sortino_from_equity([float(v) for v in n.merged_equity_curve])
        for n in naives.values()
    ]

    # 2. The REAL per-profile verdict for EACH risk profile through the actual gate path.
    profiles = load_allocation_profiles()
    per_profile: dict[str, object] = {}
    for profile_key in _PROFILE_ORDER:
        profile = profiles[profile_key]
        result = gate_with_autotighten(
            profile_key=profile_key,
            base_weights=profile.weights,
            cap_fraction=profile.max_drawdown_pct,
            deposit_curve=deposit_curve,
            ofz_pk_curve=ofz_pk_curve,
            equity_curve=equity_curve,
            naive_sharpes=naive_sharpes,
            naive_sortinos=naive_sortinos,
            regime_weights=profile.regime_weights,
        )
        # Strip the non-serializable carriers. WR-04: mean_wf_sharpe is the SINGLE
        # source of truth -- _run_and_score already attached the module-computed value
        # (allocation_gate.py), so do NOT recompute it here (a second WF pass would be
        # wasted work AND a second owner that could silently diverge on the risk-free
        # default). Just drop the AllocationResult carrier; the attached value stays.
        result.pop("result", None)  # non-serializable AllocationResult carrier
        result.pop("frozen_weights", None)  # weight dict is not JSON-key-safe; verdict suffices
        per_profile[profile_key.value] = result

    # 3. The headline regime split at the documented boundary (D-09 / R-6). The synthetic
    #    framing cut-path was RETIRED in Plan 02 (D-07): the real binding window now CONTAINS
    #    the real easing, so the cut scenario is the real easing sub-window (the
    #    post-REGIME_SPLIT_BOUNDARY segment), surfaced by render_report.
    regime = regime_split([d for d, _ in deposit_curve])

    # 3a. Phase 75 (REGIME-02 / D-01): the per-regime BINDING verdicts, computed on the SAME
    #     already-loaded (already-netted) curves via the REAL frozen path (anti-hollow — no
    #     re-fetch, no re-net). regime_verdicts emits "high_rate" / "early_cut"; "early_cut" IS
    #     the easing binding unit (the post-cut segment).
    per_regime = regime_verdicts(
        deposit_curve, ofz_pk_curve, equity_curve, profiles, _PROFILE_ORDER
    )

    # 3b. The 3-unit phase verdict = full_window AND high_rate AND easing (HARD_FAIL if ANY
    #     PRESENT unit HARD_FAILs). The full_window unit verdict collapses the existing
    #     per_profile dict; the high_rate/easing units collapse the per_regime sub-dicts.
    #     Single-regime edge (Pitfall 3): if "early_cut" is absent (window ends before the
    #     boundary) the easing unit is absent -> phase_verdict = full_window AND high_rate only.
    full_window_unit = _unit_verdict(per_profile)
    high_rate_unit = _unit_verdict(per_regime[_HIGH_RATE_UNIT])
    easing_unit = _unit_verdict(per_regime[_EASING_UNIT]) if _EASING_UNIT in per_regime else None
    present_units = [full_window_unit, high_rate_unit]
    if easing_unit is not None:
        present_units.append(easing_unit)
    phase_verdict = (
        _PHASE_HARD_FAIL if _PHASE_HARD_FAIL in present_units else "+".join(present_units)
    )

    # 3c. Escalation DERIVED from the REAL high_rate/easing unit verdicts (D-03a anti-hollow).
    #     When the easing unit is absent, pass the non-HARD_FAIL sentinel so escalation stays None.
    esc = derive_escalation(
        high_rate_unit, easing_unit if easing_unit is not None else _EASING_ABSENT_VERDICT
    )

    # 4. Render the machine sidecar + the human report through the module renderers (D-11).
    #    Phase 75 passes the per_regime block + derived escalation + n1_caveat ADDITIVELY.
    naive_metrics = _naive_metrics(naives)
    payload = render_json(
        per_profile,
        naive_metrics,
        regime,
        git_sha=git_sha,
        per_regime=per_regime,
        escalation=cast("str | None", esc["escalation"]),
        n1_caveat=cast("bool", esc["n1_caveat"]),
    )
    # Stash the 3-unit phase verdict on the payload (D-06: the full-window honest-set "overall"
    # string stays UNCHANGED; the 3-unit verdict travels in the payload + history).
    payload["phase_verdict"] = phase_verdict
    report_md = render_report(payload)

    # 5. The overall verdict is the honest SET of the three per-profile verdicts (D-06).
    overall = "+".join(
        str(cast("dict[str, object]", per_profile[p.value])["verdict"]) for p in _PROFILE_ORDER
    )
    return payload, report_md, overall


def _print_phase75_block(payload: dict[str, object]) -> None:
    """Print the Phase 75 per-regime units + 3-unit phase verdict + derived escalation.

    Additive console output only — it never touches the exit-code contract; the binding
    machine-readable surface is summary.json / history.jsonl (this is operator legibility).
    """
    per_regime_print = cast("dict[str, dict[str, object]]", payload["per_regime"])
    print("Per-regime binding verdicts (REAL gate path on the sliced sub-windows):")
    for unit, profs in per_regime_print.items():
        unit_verdicts = ", ".join(
            f"{p}={cast('dict[str, object]', pv)['verdict']!s}" for p, pv in profs.items()
        )
        print(f"  {unit:<11} {unit_verdicts}")
    print(f"PHASE VERDICT (full_window AND high_rate AND easing, D-01): {payload['phase_verdict']}")
    print(
        f"escalation (derived, D-03a): {payload['escalation']}  n1_caveat: {payload['n1_caveat']}"
    )


def main() -> int:
    """Run the gate, write the D-11 artifacts + a history.jsonl line; report the verdict.

    Returns 0 when the run completed and BOTH artifacts (summary.json + report.md) plus
    the history line were written -- the per-profile PASS/FAIL is REPORTED, never forced
    into the exit code (a HARD_FAIL is an honest outcome). A non-zero exit signals a
    HARNESS failure (artifact not written / a module call raised), not a HARD_FAIL.
    """
    parser = argparse.ArgumentParser(description="Phase 73 allocation-gate cert (GATE-01/02/03).")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Binding cert on the REAL net-of-tax series (Phase 74). Default --live reads the "
            "committed snapshot deterministically (offline); without --live the offline "
            "synthetic smoke path runs."
        ),
    )
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help=(
            "Fetch the REAL MCFTRR/RUFLBITR series, net them, and (re)write the committed "
            "snapshot fixture (operator-only network path; default --live reads the snapshot)."
        ),
    )
    args = parser.parse_args()

    git_sha = _git_sha()
    payload, report_md, overall = run_gate(
        live=args.live, git_sha=git_sha, refresh_snapshot=args.refresh_snapshot
    )

    run_name = f"{_RUN_PREFIX}-{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}"  # noqa: UP017
    run_dir = _write_iteration(run_name, payload)
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    per_profile = cast("dict[str, object]", payload["per_profile"])
    per_profile_verdicts = {
        name: cast("dict[str, object]", v)["verdict"] for name, v in per_profile.items()
    }
    naive = payload["naive"]
    # Phase 75 (REGIME-02/05): the per-regime verdict map for the history line — the bare
    # per-profile verdict strings per unit (kept compact; the full per-regime metric block
    # lives in summary.json). Additive — every existing key below is preserved.
    per_regime_block = cast("dict[str, dict[str, object]]", payload["per_regime"])
    per_regime_verdicts = {
        unit: {p: cast("dict[str, object]", pv)["verdict"] for p, pv in profs.items()}
        for unit, profs in per_regime_block.items()
    }
    _append_history(
        run_name,
        git_sha=git_sha,
        verdict=overall,
        metrics={
            "kind": "allocation_gate",
            "per_profile_verdicts": per_profile_verdicts,
            "naive_bar": naive,
            # Phase 75 additive decision keys (no existing key removed/renamed; Pitfall 4).
            "per_regime_verdicts": per_regime_verdicts,
            "phase_verdict": payload["phase_verdict"],
            "escalation": payload["escalation"],
            "n1_caveat": payload["n1_caveat"],
        },
    )

    # Verify both artifacts landed (the exit-code contract).
    summary_ok = (run_dir / "summary.json").is_file()
    report_ok = (run_dir / "report.md").is_file()

    print("=" * 78)
    print("PHASE 73 ALLOCATION GATE (GATE-01/02/03) -- backtest-iteration cert")
    print("=" * 78)
    if args.live and args.refresh_snapshot:
        print(
            f"window: REAL data {_LIVE_START.date()}..{_LIVE_END.date()} "
            "(MCFTRR net equity + RUFLBITR net OFZ + real-CBR-rate net deposit; "
            "REFRESHED + committed snapshot, Phase 74 D-05)"
        )
    elif args.live:
        print(
            f"window: REAL data {_LIVE_START.date()}..{_LIVE_END.date()} "
            "(committed net-of-tax snapshot: MCFTRR/RUFLBITR/deposit; offline, Phase 74 D-05)"
        )
    else:
        print(
            f"window: {_FIRST_BAR} + {_N_BARS} daily bars (deterministic offline smoke, no token)"
        )
    print(f"git_sha: {git_sha}")
    print("-" * 78)
    print("Per-profile verdicts (REAL gate_with_autotighten path -- not a pre-baked literal):")
    for profile_key in _PROFILE_ORDER:
        v = cast("dict[str, object]", per_profile[profile_key.value])
        print(
            f"  {profile_key.value:<13} verdict={v['verdict']!s:<18} "
            f"sharpe={cast('float', v['sharpe']):.4f} "
            f"vs best-naive {cast('float', v['best_naive_sharpe']):.4f}  "
            f"maxdd_frac={cast('float', v['realized_maxdd_frac']):.4f} "
            f"cap={cast('float', v['cap_frac']):.4f}"
        )
    print("-" * 78)
    print(f"regime split: {payload['regime_split']}")
    # The synthetic cut-path print was retired (D-07); the real easing sub-window is the
    # post-REGIME_SPLIT_BOUNDARY segment surfaced in render_report's Markdown report.
    regime_block = cast("dict[str, object]", payload["regime_split"])
    if "early_cut" in regime_block:
        print(f"real easing sub-window (evidence-based, D-07): {regime_block['early_cut']}")
    print("-" * 78)
    print(f"artifacts: {run_dir / 'summary.json'}")
    print(f"           {run_dir / 'report.md'}")
    print(f"history:   {_ITER_DIR / 'history.jsonl'}  (appended)")
    print("-" * 78)
    # Phase 75 (REGIME-02/05): the per-regime binding units + the 3-unit phase verdict +
    # the derived escalation flag (additive; the exit-code contract below is UNCHANGED).
    _print_phase75_block(payload)
    print("-" * 78)
    print(f"OVERALL VERDICT (honest set, D-06): {overall}")
    print("=" * 78)

    return 0 if (summary_ok and report_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
