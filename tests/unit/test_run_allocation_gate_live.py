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

# ── Named constants (no PLR2004 magic numbers in tests) ──────────────────────
_N_FETCH_BARS = 360  # comfortably > _N_LIVE_MIN_BARS (300) for the happy-path fetches
_N_SHORT_BARS = 10  # deliberately < _N_LIVE_MIN_BARS to trip the short-fetch honesty gate
_MCFTRR_BASE = Decimal("6000.00")  # a plausible MCFTRR index level
_MCFTRR_DAILY = Decimal("1.0007")  # net-equity drift
_RUFLBITR_BASE = Decimal("150.00")  # a plausible RUFLBITR floater-index level
_RUFLBITR_DAILY = Decimal("1.0004")  # floater carry drift (positive -> nets below gross)
_FIRST_FETCH_BAR = date(2024, 1, 3)  # the first real MCFTRR/RUFLBITR trading bar (R-D)
_VALID_VERDICTS = {"PASS", "PASS_AFTER_TIGHTEN", "HARD_FAIL"}


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
