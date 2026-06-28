"""Active-equity-sleeve experiment runner (R&D, diagnostic).

Runs every low-turnover tilt arm (equal-weight, dividend-yield, low-vol) AND the
cap-weight PROXY baseline through the SAME basket simulator
(:mod:`finalayze.backtest.equity_tilt_lab`) — identical universe, dividends
(net-of-NDFL), and retail costs — then judges each tilt against the cap-proxy on
the SAME strict conjunctive bar the binding allocator gate uses: Sharpe AND
Sortino AND MaxDD, on the full window AND the high-rate sub-window, with the
easing sub-window REPORTED under an ``n1_caveat`` (single 2025 easing episode).

This is the honest answer to "does routing some equity into active selection beat
just holding the index?" — measured net-of-everything against a like-for-like
cap-proxy, never a gross published index. Expected outcome per the strategic
review (0/113 prior, N=1 regime, ~34 correlated oil+banks names): most/all tilts
FAIL — which is the useful, honest finding, not a defect.

See docs/research/active_equity_sleeve_experiment.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.backtest.allocation_gate import (
    excess_sortino_from_equity,
    regime_split,
)
from finalayze.backtest.bond_walk_forward import _compute_excess_sharpe_from_equity
from finalayze.backtest.costs import MOEX_RETAIL_COSTS
from finalayze.backtest.equity_tilt_lab import (
    PricePoint,
    adv_cap_proxy_weights,
    equal_weights,
    inverse_vol_weights,
    make_dividend_yield_policy,
    max_drawdown_pct,
    quarter_end_dates,
    simulate_basket,
)

if TYPE_CHECKING:
    from datetime import date

    from finalayze.backtest.equity_tilt_lab import WeightPolicy

# RUONIA-excess risk-free basis — mirrors allocation_gate._DEFAULT_RUONIA_ANNUAL_PCT
# (15.0). It is COMMON to every arm, so the tilt-vs-baseline verdict is invariant to
# the exact value; pinned here so Sharpe/Sortino sit on the same footing as the gate.
RISK_FREE_ANNUAL_PCT = 15.0
_PERCENT = Decimal(100)
_BASELINE_KEY = "cap_proxy_baseline"


@dataclass(frozen=True)
class ArmMetrics:
    """RUONIA-excess risk-adjusted metrics of one arm's NAV curve over one window."""

    sharpe: float
    sortino: float
    maxdd_pct: float
    total_return_pct: float
    n_bars: int


def _metrics(curve: list[float]) -> ArmMetrics:
    sharpe = _compute_excess_sharpe_from_equity(curve, RISK_FREE_ANNUAL_PCT)
    sortino = excess_sortino_from_equity(curve, RISK_FREE_ANNUAL_PCT)
    mdd = max_drawdown_pct(curve)
    tr = (curve[-1] / curve[0] - 1.0) * 100.0 if curve and curve[0] > 0 else 0.0
    return ArmMetrics(
        sharpe=sharpe, sortino=sortino, maxdd_pct=mdd, total_return_pct=tr, n_bars=len(curve)
    )


def _verdict(arm: ArmMetrics, base: ArmMetrics) -> dict[str, bool]:
    """Strict conjunctive bar: a tilt PASSES iff it beats the baseline on all three."""
    beats_sharpe = arm.sharpe > base.sharpe
    beats_sortino = arm.sortino > base.sortino
    dd_ok = arm.maxdd_pct <= base.maxdd_pct
    return {
        "beats_sharpe": beats_sharpe,
        "beats_sortino": beats_sortino,
        "dd_within_baseline": dd_ok,
        "passed": beats_sharpe and beats_sortino and dd_ok,
    }


def _slice(dates: list[date], curve: list[Decimal], start: date, end: date) -> list[float]:
    return [float(v) for d, v in zip(dates, curve, strict=True) if start <= d <= end]


def run_experiment(
    panel: dict[str, list[PricePoint]],
    dividend_schedule: dict[tuple[str, date], Decimal],
    *,
    initial_nav: Decimal = Decimal(1000000),
) -> dict[str, object]:
    """Run all arms, slice by regime, and judge each tilt vs the cap-proxy baseline."""
    all_dates = sorted({d for pts in panel.values() for d, _, _ in pts})
    if not all_dates:
        msg = "empty panel — no dates to simulate"
        raise ValueError(msg)
    rebal = sorted({all_dates[0], *quarter_end_dates(all_dates)})

    policies: dict[str, WeightPolicy] = {
        _BASELINE_KEY: adv_cap_proxy_weights,
        "equal_weight": equal_weights,
        "low_vol": inverse_vol_weights,
        "dividend_yield": make_dividend_yield_policy(dividend_schedule),
    }
    arms = {
        name: simulate_basket(
            panel=panel,
            dividend_schedule=dividend_schedule,
            weight_policy=policy,
            rebalance_dates=rebal,
            costs=MOEX_RETAIL_COSTS,
            initial_nav=initial_nav,
        )
        for name, policy in policies.items()
    }

    regions = regime_split(all_dates)  # {"high_rate": (s,e), "early_cut": (s,e)?}
    base = arms[_BASELINE_KEY]

    per_arm: dict[str, object] = {}
    # typed parallel to per_arm so the binding-verdict step never drills into `object`
    verdict_by_arm: dict[str, dict[str, dict[str, bool] | None]] = {}
    base_full = _metrics(base.equity_floats)
    for name, res in arms.items():
        windows: dict[str, object] = {}
        wv: dict[str, dict[str, bool] | None] = {}
        # full window
        full_m = _metrics(res.equity_floats)
        full_v = _verdict(full_m, base_full) if name != _BASELINE_KEY else None
        wv["full_window"] = full_v
        windows["full_window"] = {"metrics": asdict(full_m), "verdict": full_v}
        # regime sub-windows (slice the ALREADY-simulated full curve; no re-sim)
        for region_key, (start, end) in regions.items():
            arm_m = _metrics(_slice(res.dates, res.nav_curve, start, end))
            base_m = _metrics(_slice(base.dates, base.nav_curve, start, end))
            region_v = _verdict(arm_m, base_m) if name != _BASELINE_KEY else None
            wv[region_key] = region_v
            windows[region_key] = {
                "range": [start.isoformat(), end.isoformat()],
                "metrics": asdict(arm_m),
                "verdict": region_v,
                "n1_caveat": region_key != "high_rate",
            }
        verdict_by_arm[name] = wv
        per_arm[name] = {
            "windows": windows,
            "total_cost": str(res.total_cost),
            "total_tax": str(res.total_tax),
            "dividend_gross": str(res.dividend_gross),
            "cost_drag_pct_of_initial": str(res.total_cost / initial_nav * _PERCENT),
        }

    # Binding verdict: a tilt earns a real allocation only if it PASSES full_window
    # AND high_rate (easing is caveated). Read off the TYPED verdict map.
    passers: list[str] = []
    for name in policies:
        if name == _BASELINE_KEY:
            continue
        wv = verdict_by_arm[name]
        full_v = wv.get("full_window")
        hr_v = wv.get("high_rate")
        if full_v and full_v["passed"] and hr_v and hr_v["passed"]:
            passers.append(name)

    return {
        "window": {
            "start": all_dates[0].isoformat(),
            "end": all_dates[-1].isoformat(),
            "n_bars": len(all_dates),
            "n_rebalances": len(rebal),
            "universe_size": len(panel),
            "regime_boundary": regions,
        },
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "baseline": _BASELINE_KEY,
        "arms": per_arm,
        "binding": {
            "passers": passers,
            "verdict": "PASS" if passers else "HARD_FAIL",
            "finding": (
                f"{len(passers)} tilt(s) beat the cap-proxy on full+high_rate: {passers}"
                if passers
                else "no tilt beats the cap-proxy baseline on full_window+high_rate "
                "(deposit-anchor / passive-sleeve conclusion holds for the equity sleeve)"
            ),
            "n1_caveat": True,
        },
    }
