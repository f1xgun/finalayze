"""Structural tests for the active-equity-sleeve experiment runner.

Correctness of the underlying simulator/policies is covered in
test_equity_tilt_lab.py; here we pin the runner's CONTRACT (arm set, baseline,
regime split, conjunctive binding verdict) on a synthetic panel that spans the
2025-06-06 regime boundary.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.equity_tilt_experiment import run_experiment
from finalayze.backtest.equity_tilt_lab import PricePoint

_EXPECTED_ARMS = {"cap_proxy_baseline", "equal_weight", "low_vol", "dividend_yield"}


def _panel_spanning_boundary() -> dict[str, list[PricePoint]]:
    start = date(2024, 6, 1)
    dates = [start + timedelta(days=i) for i in range(600)]  # into late 2025, past 2025-06-06
    panel: dict[str, list[PricePoint]] = {}
    for j, sym in enumerate(["AAA", "BBB", "CCC"]):
        pts: list[PricePoint] = []
        price = Decimal(100) + Decimal(j * 10)
        for i, d in enumerate(dates):
            # mild deterministic drift + per-name wobble so vols/returns differ
            price = price + Decimal(str(((i + j) % 7 - 3) * 0.5))
            if price <= 1:
                price = Decimal(50)
            vol = Decimal(1000 + j * 500)
            pts.append((d, price, vol))
        panel[sym] = pts
    return panel


def test_run_experiment_contract() -> None:
    panel = _panel_spanning_boundary()
    # one dividend so the dividend arm and accrual path are exercised
    sched = {("AAA", date(2025, 1, 15)): Decimal(3)}
    out = run_experiment(panel, sched)

    # arm set + baseline
    assert set(out["arms"]) == _EXPECTED_ARMS
    assert out["baseline"] == "cap_proxy_baseline"

    # regime split produced both sub-windows (window spans 2025-06-06)
    regions = out["window"]["regime_boundary"]
    assert "high_rate" in regions
    assert "early_cut" in regions

    # binding verdict is one of the two honest strings
    assert out["binding"]["verdict"] in {"PASS", "HARD_FAIL"}
    assert out["binding"]["n1_caveat"] is True

    # baseline arm carries no self-verdict; tilt arms do
    base_full = out["arms"]["cap_proxy_baseline"]["windows"]["full_window"]
    assert base_full["verdict"] is None
    ew_full = out["arms"]["equal_weight"]["windows"]["full_window"]
    assert set(ew_full["verdict"]) == {
        "beats_sharpe",
        "beats_sortino",
        "dd_within_baseline",
        "passed",
    }
    # every arm reports a cost drag (retail costs were charged)
    assert Decimal(out["arms"]["equal_weight"]["cost_drag_pct_of_initial"]) > 0
