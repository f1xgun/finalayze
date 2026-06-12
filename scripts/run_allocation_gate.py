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
(``REGIME_SPLIT_BOUNDARY`` = 2025-07-25) so the report renders BOTH the high-rate and
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

from finalayze.backtest.allocation_gate import (
    accrue_real_risk_free_leg,
    build_naive_legs,
    excess_sortino_from_equity,
    gate_with_autotighten,
    regime_split,
    render_json,
    render_report,
    run_cut_path,
)
from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.schemas import RiskProfile
from finalayze.data.loader import load_mcftr_series

if TYPE_CHECKING:
    from finalayze.orchestration.allocation import AllocationResult

# ── Named-constant header block (no PLR2004 magic numbers) ───────────────────

_PHASE = "73"
_RUN_PREFIX = "allocation-gate-73"
_ITER_DIR = Path(__file__).resolve().parent.parent / "results" / "iterations"

# Deterministic offline window: a multi-year daily span that BRACKETS the documented
# regime boundary (REGIME_SPLIT_BOUNDARY = 2025-07-25) so the report renders BOTH the
# high-rate and the early-cut regimes (D-09 / R-6). Start well before, end well after.
_FIRST_BAR = date(2024, 1, 1)
_N_BARS = 700  # ~2.8y of daily bars -> spans 2025-07-25 + many quarter boundaries

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
# required the cert computed on the REAL MCFTR/CBR series. This window BRACKETS the
# documented REGIME_SPLIT_BOUNDARY (2025-07-25) so the report renders BOTH regimes on
# REAL data. MCFTR (the equity leg) is the public ISS-REST gross total-return index; the
# deposit + OFZ-PK legs accrue from the REAL look-ahead-safe CBR_MEETINGS key-rate path.
_LIVE_START = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017
_LIVE_END = datetime(2025, 11, 30, tzinfo=timezone.utc)  # noqa: UP017
# Real legs share ONE common daily axis (the MCFTR trading-day calendar): the deposit +
# OFZ-PK legs are accrued on the EXACT MCFTR dates so build_naive_legs sees one basis (R-3).
_LIVE_DEPOSIT_BASE = Decimal(100_000)
_LIVE_OFZ_BASE = Decimal(100_000)
# Deposit accrues at key-1pp (mirrors W1's deposit spread); OFZ-PK floater tracks the full
# key rate (no spread). Both read the REAL CBR calendar via accrue_real_risk_free_leg.
_LIVE_DEPOSIT_SPREAD_PP = Decimal("1.0")
_LIVE_OFZ_SPREAD_PP = Decimal(0)
# A real ~2y MCFTR window has ~480+ trading-day bars; refuse a hollow run if the fetch
# returns far fewer (HONESTY GATE: never fabricate a synthetic 'live' leg, T-73-12).
_N_LIVE_MIN_BARS = 300

# The three SAA profiles, in the conservative -> balanced -> growth order the report
# renders them. Each carries its own MaxDD cap (8 / 15 / 25%) read from the loaded
# AllocationProfile (load_allocation_profiles), never hardcoded here.
_PROFILE_ORDER = (RiskProfile.CONSERVATIVE, RiskProfile.BALANCED, RiskProfile.GROWTH)


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


def _load_live_curves() -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Load the three REAL total-return curves for the operator-directed ``--live`` cert.

    Operator decision (D-10 override): the seeded-synthetic fixture is REJECTED as the
    binding v11.0 cert; the cert MUST be computed on the REAL MCFTR/CBR series. Every leg
    here is genuine real-market data (the Phase-72 anti-hollow lesson, escalated):

    - **equity (MCFTR)** -- REAL: ``load_mcftr_series`` over the public MOEX ISS REST index
      endpoint (gross total-return, NO token/cert; R-1). This is BOTH the equity sleeve and
      the 100%-equity benchmark bar (apples-to-apples).
    - **deposit** -- REAL: ``accrue_real_risk_free_leg`` daily-compounds at the as-of
      ``deposit_rate_as_of`` (key - 1pp) read look-ahead-safe from the REAL ``CBR_MEETINGS``
      key-rate calendar.
    - **OFZ-PK** -- REAL: the same real-rate accrual at the full key rate (no spread) -- the
      floater leg. (The allocator's OFZ-PK leg is an as-of-key-rate accrued floater, not a
      per-bond candle series; the W2/bond layer keys it off the same real CBR key rate,
      73-RESEARCH R-5.)

    The MCFTR trading-day calendar is the MASTER axis: the deposit + OFZ-PK legs accrue on
    the EXACT MCFTR dates so all three legs share ONE common daily basis (R-3) -- the same
    alignment the offline path's shared ``_dates()`` gives.
    """
    try:
        equity_curve = load_mcftr_series(start=_LIVE_START, end=_LIVE_END)
    except Exception as exc:  # operator-facing legibility at the network seam (WR-03)
        # WR-03 / T-73-16: a failed fetch (ISS-REST network error, timeout, HTTP error)
        # must surface as a clean, actionable operator message -- NOT a raw traceback.
        # Mirror the short-fetch guard below: a non-zero exit signals a HARNESS failure,
        # and the message distinguishes "MOEX unreachable" from a real bug.
        msg = (
            f"--live MCFTR fetch failed ({type(exc).__name__}: {exc}); "
            "MOEX ISS-REST unreachable -- refusing to fabricate a synthetic 'live' leg."
        )
        raise SystemExit(msg) from exc
    if len(equity_curve) < _N_LIVE_MIN_BARS:
        msg = (
            f"--live MCFTR fetch returned only {len(equity_curve)} bars over "
            f"{_LIVE_START.date()}..{_LIVE_END.date()} (need >= {_N_LIVE_MIN_BARS}); "
            "real equity data unavailable -- refusing to fabricate a synthetic 'live' leg."
        )
        raise SystemExit(msg)
    # The MCFTR trading-day calendar IS the common axis (R-3): accrue the two risk-free
    # legs on the EXACT MCFTR dates from the REAL CBR key-rate path.
    axis = [d for d, _ in equity_curve]
    deposit_curve = accrue_real_risk_free_leg(
        axis, _LIVE_DEPOSIT_BASE, spread_pp=_LIVE_DEPOSIT_SPREAD_PP
    )
    ofz_pk_curve = accrue_real_risk_free_leg(axis, _LIVE_OFZ_BASE, spread_pp=_LIVE_OFZ_SPREAD_PP)
    return deposit_curve, ofz_pk_curve, equity_curve


def _load_curves(
    *, live: bool, dates: list[date]
) -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Load the three benchmark TR curves.

    Default (``live=False``): the deterministic in-memory geometric curves -- CI-safe,
    reproducible, no token, no network (the cert path). With ``--live`` (the operator's
    D-10 override) the REAL series are loaded via :func:`_load_live_curves`: the genuine
    MCFTR equity index + the real-CBR-rate-accrued deposit/OFZ-PK legs. The live legs drive
    their OWN dates (the MCFTR trading-day calendar), so the offline ``dates`` argument is
    used only for the default path.
    """
    if live:
        return _load_live_curves()
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


def run_gate(*, live: bool, git_sha: str) -> tuple[dict[str, object], str, str]:
    """Drive the REAL gate end-to-end and assemble the JSON payload + Markdown report.

    The Phase 72 anti-hollow contract: every per-profile verdict comes from the REAL
    ``gate_with_autotighten`` on the REAL ``build_naive_legs`` output -- NOT a pre-baked
    constant. Returns ``(payload, report_md, overall_verdict)`` where ``overall_verdict``
    is the honest SET of the three per-profile verdicts (D-06).
    """
    dates = _dates()
    deposit_curve, ofz_pk_curve, equity_curve = _load_curves(live=live, dates=dates)

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
        )
        # Strip the non-serializable carriers. WR-04: mean_wf_sharpe is the SINGLE
        # source of truth -- _run_and_score already attached the module-computed value
        # (allocation_gate.py), so do NOT recompute it here (a second WF pass would be
        # wasted work AND a second owner that could silently diverge on the risk-free
        # default). Just drop the AllocationResult carrier; the attached value stays.
        result.pop("result", None)  # non-serializable AllocationResult carrier
        result.pop("frozen_weights", None)  # weight dict is not JSON-key-safe; verdict suffices
        per_profile[profile_key.value] = result

    # 3. The framing-only cut-path metrics (D-08) -- reported, NEVER fed to the verdict.
    cut = run_cut_path(deposit_curve, ofz_pk_curve, equity_curve)
    cut_path_metrics: dict[str, object] = {
        "sharpe": cut.sharpe,
        "sortino": excess_sortino_from_equity([float(v) for v in cut.merged_equity_curve]),
        "maxdd_pct": cut.max_drawdown_pct,
        "rebalance_cost": str(cut.rebalance_cost),
        "realized_ndfl": str(cut.realized_ndfl),
        "final_equity": str(cut.merged_equity_curve[-1]),
        "note": "FRAMING-ONLY (D-08): risk-free legs lowered under CUT_GLIDE; equity held fixed.",
    }

    # 4. The headline regime split at the documented boundary (D-09 / R-6).
    regime = regime_split([d for d, _ in deposit_curve])

    # 5. Render the machine sidecar + the human report through the module renderers (D-11).
    naive_metrics = _naive_metrics(naives)
    payload = render_json(per_profile, naive_metrics, cut_path_metrics, regime, git_sha=git_sha)
    report_md = render_report(payload)

    # 6. The overall verdict is the honest SET of the three per-profile verdicts (D-06).
    overall = "+".join(
        str(cast("dict[str, object]", per_profile[p.value])["verdict"]) for p in _PROFILE_ORDER
    )
    return payload, report_md, overall


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
        help="Fetch the REAL MCFTR series (reserved operator run; default is offline).",
    )
    args = parser.parse_args()

    git_sha = _git_sha()
    payload, report_md, overall = run_gate(live=args.live, git_sha=git_sha)

    run_name = f"{_RUN_PREFIX}-{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}"  # noqa: UP017
    run_dir = _write_iteration(run_name, payload)
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    per_profile = cast("dict[str, object]", payload["per_profile"])
    per_profile_verdicts = {
        name: cast("dict[str, object]", v)["verdict"] for name, v in per_profile.items()
    }
    naive = payload["naive"]
    _append_history(
        run_name,
        git_sha=git_sha,
        verdict=overall,
        metrics={
            "kind": "allocation_gate",
            "per_profile_verdicts": per_profile_verdicts,
            "naive_bar": naive,
        },
    )

    # Verify both artifacts landed (the exit-code contract).
    summary_ok = (run_dir / "summary.json").is_file()
    report_ok = (run_dir / "report.md").is_file()

    print("=" * 78)
    print("PHASE 73 ALLOCATION GATE (GATE-01/02/03) -- backtest-iteration cert")
    print("=" * 78)
    if args.live:
        print(
            f"window: REAL data {_LIVE_START.date()}..{_LIVE_END.date()} "
            "(MCFTR ISS-REST equity + real-CBR-rate deposit/OFZ-PK legs, operator D-10 override)"
        )
    else:
        print(f"window: {_FIRST_BAR} + {_N_BARS} daily bars (deterministic, offline, no token)")
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
    cut_block = cast("dict[str, object]", payload["cut_path"])
    print(f"cut-path (FRAMING-ONLY): sharpe={cast('float', cut_block['sharpe']):.4f}")
    print("-" * 78)
    print(f"artifacts: {run_dir / 'summary.json'}")
    print(f"           {run_dir / 'report.md'}")
    print(f"history:   {_ITER_DIR / 'history.jsonl'}  (appended)")
    print("-" * 78)
    print(f"OVERALL VERDICT (honest set, D-06): {overall}")
    print("=" * 78)

    return 0 if (summary_ok and report_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
