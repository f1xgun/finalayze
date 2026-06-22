"""Offline tests for the ``scripts/run_allocation_gate.py`` ``--live`` wiring (Phase 74, Plan 03).

NO NETWORK: every test that touches the ``--live`` path monkeypatches the script-bound
``load_mcftr_series`` symbol so no real ISS-REST fetch ever fires. The real
``--live --refresh-snapshot`` fetch is an OPERATOR action in Plan 04 — these tests only
exercise the CLI plumbing:

- Task 1: MCFTRR (net equity, unchanged) + RUFLBITR (gross, then netted via one shared
  ``YtdTaxAccumulator``); window to 2026-06-10; both fetches ``SystemExit``-guarded; the
  retired ``run_cut_path`` is gone (no ``cut_path`` key in the payload).
- Task 2: default ``--live`` reads the committed snapshot via ``_load_gate_snapshot`` (no
  network); ``--live --refresh-snapshot`` fetches + nets + writes a round-trippable, clamped
  snapshot fixture; the verdict still flows through the REAL ``build_naive_legs ->
  gate_with_autotighten`` path (anti-hollow — no test-only hook, no pre-baked literal).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Ensure scripts/ is importable (CLAUDE.md: scripts/ live at the project root, not under src/).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import run_allocation_gate as rag  # noqa: E402

from finalayze.backtest.allocation_gate import (  # noqa: E402
    _ESCALATION_DEPOSIT_ANCHOR,
)

# ── Named constants (no PLR2004 magic numbers in tests) ──────────────────────
_N_FETCH_BARS = 360  # comfortably > _N_LIVE_MIN_BARS (300) for the happy-path fetches
_N_SHORT_BARS = 10  # deliberately < _N_LIVE_MIN_BARS to trip the short-fetch honesty gate
_MCFTRR_BASE = Decimal("6000.00")  # a plausible MCFTRR index level
_MCFTRR_DAILY = Decimal("1.0007")  # net-equity drift
_RUFLBITR_BASE = Decimal("150.00")  # a plausible RUFLBITR floater-index level
_RUFLBITR_DAILY = Decimal("1.0004")  # floater carry drift (positive -> nets below gross)
_FIRST_FETCH_BAR = date(2024, 1, 3)  # the first real MCFTRR/RUFLBITR trading bar (R-D)
_VALID_VERDICTS = {"PASS", "PASS_AFTER_TIGHTEN", "HARD_FAIL"}
# Phase 75 (REGIME-02/05) — the 3-unit phase verdict + per-regime block keys.
_PHASE_VERDICT_HARD_FAIL = "HARD_FAIL"  # the honest expected phase verdict on the snapshot
_REGIME_HIGH_RATE = "high_rate"  # the high-rate binding unit key (regime_split)
_REGIME_EARLY_CUT = "early_cut"  # the easing binding unit key (regime_split post-cut segment)


def _series(base: Decimal, daily: Decimal, n: int) -> list[tuple[date, Decimal]]:
    """A deterministic monotone geometric ``(date, Decimal)`` series of ``n`` bars."""
    out: list[tuple[date, Decimal]] = []
    value = base
    for i in range(n):
        if i:
            value = value * daily
        out.append((_FIRST_FETCH_BAR + timedelta(days=i), value))
    return out


def _secid_fetch(n: int = _N_FETCH_BARS) -> Callable[..., list[tuple[date, Decimal]]]:
    """A ``load_mcftr_series`` stand-in keyed by ``secid`` (MCFTRR vs RUFLBITR, distinct series)."""

    def _fetch(secid: str = "MCFTR", **_kwargs: object) -> list[tuple[date, Decimal]]:
        if secid == "RUFLBITR":
            return _series(_RUFLBITR_BASE, _RUFLBITR_DAILY, n)
        return _series(_MCFTRR_BASE, _MCFTRR_DAILY, n)

    return _fetch


# ── Task 1 ───────────────────────────────────────────────────────────────────


def test_live_curves_mcftrr_unchanged_ruflbitr_netted(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCFTRR equity passes through UNCHANGED (net); RUFLBITR is netted (differs from raw)."""
    monkeypatch.setattr(rag, "load_mcftr_series", _secid_fetch())
    deposit_curve, ofz_pk_curve, equity_curve = rag._load_live_curves()

    raw_mcftrr = _series(_MCFTRR_BASE, _MCFTRR_DAILY, _N_FETCH_BARS)
    raw_ruflbitr = _series(_RUFLBITR_BASE, _RUFLBITR_DAILY, _N_FETCH_BARS)

    # Equity (MCFTRR) is left as-is — it is already MOEX's net-of-tax index (D-02).
    assert equity_curve == raw_mcftrr
    # All three legs share the MCFTRR (master) date axis (R-3).
    axis = [d for d, _ in equity_curve]
    assert [d for d, _ in deposit_curve] == axis
    assert [d for d, _ in ofz_pk_curve] == axis
    # The OFZ leg is the NETTED RUFLBITR — it must differ from the raw gross series (NDFL haircut).
    assert ofz_pk_curve != raw_ruflbitr
    # Both legs open at the gross base (principal is never taxed) then net the income increment.
    assert ofz_pk_curve[0][1] == raw_ruflbitr[0][1]
    assert ofz_pk_curve[-1][1] < raw_ruflbitr[-1][1]


def test_live_mcftrr_fetch_failure_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising MCFTRR fetch surfaces a clean SystemExit — never a synthetic fallback."""

    def _raise(**_kwargs: object) -> list[tuple[date, Decimal]]:
        msg = "ISS-REST unreachable"
        raise ConnectionError(msg)

    monkeypatch.setattr(rag, "load_mcftr_series", _raise)
    with pytest.raises(SystemExit):
        rag._load_live_curves()


def test_live_ruflbitr_short_fetch_raises_systemexit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short RUFLBITR fetch trips the short-fetch honesty gate (extended to the 2nd fetch)."""

    def _fetch(secid: str = "MCFTR", **_kwargs: object) -> list[tuple[date, Decimal]]:
        if secid == "RUFLBITR":
            return _series(_RUFLBITR_BASE, _RUFLBITR_DAILY, _N_SHORT_BARS)  # too few bars
        return _series(_MCFTRR_BASE, _MCFTRR_DAILY, _N_FETCH_BARS)

    monkeypatch.setattr(rag, "load_mcftr_series", _fetch)
    with pytest.raises(SystemExit):
        rag._load_live_curves()


def test_run_gate_has_no_cut_path() -> None:
    """The retired cut-path is gone: ``run_gate`` produces no ``cut_path`` key (offline smoke)."""
    payload, _report, _overall = rag.run_gate(live=False, git_sha="test")
    assert "cut_path" not in payload
    # Anti-hollow: the per-profile verdicts come from the REAL gate path (not a pre-baked literal).
    per_profile = cast("dict[str, dict[str, object]]", payload["per_profile"])
    assert per_profile
    for v in per_profile.values():
        assert v["verdict"] in _VALID_VERDICTS


# ── Task 2: snapshot read (default --live) + refresh-write ───────────────────

# A multi-year daily window (2024-01-03 .. 2026-06-10) so regime_split spans the boundary
# and the OOS walk-forward windows (12/6/3 months) yield real fold Sharpes.
_SNAP_FIRST = date(2024, 1, 3)
_SNAP_N_BARS = 620  # ~2.4y of daily bars -> last bar still <= 2026-06-10


def _snapshot_legs() -> tuple[
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
    list[tuple[date, Decimal]],
]:
    """Three deterministic net TR legs (deposit anchor, OFZ carry, faster equity)."""
    axis = [_SNAP_FIRST + timedelta(days=i) for i in range(_SNAP_N_BARS)]
    deposit = [(d, Decimal(100_000) * Decimal("1.00055") ** i) for i, d in enumerate(axis)]
    ofz = [(d, Decimal(100_000) * Decimal("1.0004") ** i) for i, d in enumerate(axis)]
    equity = [(d, Decimal(6000) * Decimal("1.0006") ** i) for i, d in enumerate(axis)]
    return deposit, ofz, equity


def _write_snapshot_file(
    path: Path,
    *,
    end: date = date(2026, 6, 10),
    legs: tuple[
        list[tuple[date, Decimal]],
        list[tuple[date, Decimal]],
        list[tuple[date, Decimal]],
    ]
    | None = None,
) -> None:
    """Write a valid committed-snapshot fixture (the R-F shape) to ``path``."""
    deposit, ofz, equity = legs if legs is not None else _snapshot_legs()

    def _ser(leg: list[tuple[date, Decimal]]) -> list[list[str]]:
        return [[d.isoformat(), str(v)] for d, v in leg]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-12T00:00:00+00:00",
                "window": {"start": _SNAP_FIRST.isoformat(), "end": end.isoformat()},
                "git_sha": "test",
                "legs": {
                    "equity_mcftrr_net": _ser(equity),
                    "ofz_ruflbitr_net": _ser(ofz),
                    "deposit_net": _ser(deposit),
                },
            }
        ),
        encoding="utf-8",
    )


def test_live_reads_committed_snapshot_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default --live reads the committed snapshot (NO network); verdicts via the REAL gate path."""
    snap = tmp_path / "allocation_gate_snapshot.json"
    _write_snapshot_file(snap)
    monkeypatch.setattr(rag, "_GATE_SNAPSHOT", snap)

    # Any fetch attempt is a hard failure — the snapshot-read path must NOT hit the network.
    def _boom(**_kwargs: object) -> list[tuple[date, Decimal]]:
        msg = "network must not be touched on the snapshot-read path"
        raise AssertionError(msg)

    monkeypatch.setattr(rag, "load_mcftr_series", _boom)

    payload, _report, _overall = rag.run_gate(live=True, git_sha="test", refresh_snapshot=False)
    per_profile = cast("dict[str, dict[str, object]]", payload["per_profile"])
    assert per_profile  # the real gate path produced verdicts
    for v in per_profile.values():
        assert v["verdict"] in _VALID_VERDICTS  # real gate output, not a pre-baked constant

    # Phase 75 (REGIME-02/05): the per-regime binding block is present and every nested
    # verdict is a REAL gate output (anti-hollow — same monkeypatched no-network path).
    assert "per_regime" in payload
    per_regime = cast("dict[str, dict[str, dict[str, object]]]", payload["per_regime"])
    assert per_regime  # both regime units are present on the boundary-spanning snapshot
    for unit in per_regime.values():
        for pv in unit.values():
            assert pv["verdict"] in _VALID_VERDICTS  # real gate output, not a pre-baked constant
    # The N=1 caveat is always-on metadata; the escalation is DERIVED (None or the deposit-anchor).
    assert payload["n1_caveat"] is True
    assert payload["escalation"] in {None, _ESCALATION_DEPOSIT_ANCHOR}


def test_phase_verdict_is_three_unit_and(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """phase_verdict = full_window AND high_rate AND easing — HARD_FAIL if ANY unit HARD_FAILs.

    On the boundary-spanning committed snapshot (the deposit dominates the high-rate regime),
    the gate HARD_FAILs in at least one unit, so the 3-unit AND is HARD_FAIL. Both the
    high_rate and the early_cut (easing) units are present. NO network — the fetch is
    monkeypatched to RAISE (anti-hollow: the per-regime verdicts flow through the REAL path).
    """
    snap = tmp_path / "allocation_gate_snapshot.json"
    _write_snapshot_file(snap)
    monkeypatch.setattr(rag, "_GATE_SNAPSHOT", snap)

    def _boom(**_kwargs: object) -> list[tuple[date, Decimal]]:
        msg = "network must not be touched on the snapshot-read path"
        raise AssertionError(msg)

    monkeypatch.setattr(rag, "load_mcftr_series", _boom)

    payload, _report, _overall = rag.run_gate(live=True, git_sha="test", refresh_snapshot=False)
    assert payload["phase_verdict"] == _PHASE_VERDICT_HARD_FAIL
    per_regime = cast("dict[str, object]", payload["per_regime"])
    assert _REGIME_HIGH_RATE in per_regime
    assert _REGIME_EARLY_CUT in per_regime


def test_refresh_snapshot_writes_round_trippable_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--refresh-snapshot fetches + nets + writes a Decimal-exact, clamped, round-trippable file."""
    monkeypatch.setattr(rag, "load_mcftr_series", _secid_fetch())
    deposit_curve, ofz_pk_curve, equity_curve = rag._load_live_curves()

    snap = tmp_path / "data" / "allocation_gate_snapshot.json"
    rag._write_gate_snapshot(
        deposit_curve,
        ofz_pk_curve,
        equity_curve,
        start=_FIRST_FETCH_BAR,
        end=date(2026, 6, 10),
        git_sha="test",
        path=snap,
    )
    assert snap.is_file()  # the data/ dir was created and the fixture written

    # Round-trips Decimal-exact through the FROZEN Plan-02 loader (the binding read path).
    re_equity, re_ofz, re_deposit = rag._load_gate_snapshot(snap)
    assert re_equity == equity_curve
    assert re_ofz == ofz_pk_curve
    assert re_deposit == deposit_curve
    # Look-ahead clamp (Pitfall 3): no written bar post-dates _LIVE_END.
    binding_end = rag._LIVE_END.date()
    assert all(d <= binding_end for d, _ in re_equity)


def test_refresh_snapshot_clamps_future_bars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refresh-write path drops any fetched bar after _LIVE_END (look-ahead guard)."""
    monkeypatch.setattr(rag, "load_mcftr_series", _secid_fetch())
    deposit_curve, ofz_pk_curve, equity_curve = rag._load_live_curves()
    # Append a synthetic future bar past the clamp to every leg.
    future = rag._LIVE_END.date() + timedelta(days=5)
    deposit_curve.append((future, deposit_curve[-1][1]))
    ofz_pk_curve.append((future, ofz_pk_curve[-1][1]))
    equity_curve.append((future, equity_curve[-1][1]))

    snap = tmp_path / "data" / "allocation_gate_snapshot.json"
    rag._write_gate_snapshot(
        deposit_curve,
        ofz_pk_curve,
        equity_curve,
        start=_FIRST_FETCH_BAR,
        end=rag._LIVE_END.date(),
        git_sha="test",
        path=snap,
    )
    re_equity, _re_ofz, _re_deposit = rag._load_gate_snapshot(snap)
    # The future bar must NOT survive the clamp on write.
    assert all(d <= rag._LIVE_END.date() for d, _ in re_equity)
    assert future not in [d for d, _ in re_equity]
