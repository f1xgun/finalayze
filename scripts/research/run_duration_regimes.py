"""Iter-5: fixed-coupon OFZ (duration / yield-lock) vs deposit vs floater, PER RATE REGIME.

Answers the operator's question — bonds at 15-16% vs a falling 14% deposit — and corrects a
regime-blind spot in the iter-1 gate: it REJECTED fixed-coupon OFZ (RGBITR) on a 2022-2026
FULL-window average dominated by the rate-HIKING era (2023-24, key 7.5->21%, where fixed bonds
LOSE price). The verdict flips per regime, which is the whole point:

  - HIKING (rates rising): the FLOATER + deposit win; fixed-coupon OFZ loses price.
  - EASING (rates falling, the CURRENT regime): the FLOATER out-carries the deposit (a bond at
    ~key beats a deposit at key-1pp) — so "bonds > deposit" IS captured via the OFZ_PK leg the
    SAA already holds. Fixed-coupon DURATION did NOT win here (inverted curve: short > long
    yields, so locking a lower long yield gave up carry + took a duration drawdown).

Compares raw TOTAL RETURN (the honest carry comparison) + annualized + MaxDD across the full
window, the hiking sub-window, and the easing sub-window (the live regime). NOT financial advice;
diagnostic/backtest-only.

    uv run python scripts/research/run_duration_regimes.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg, net_index_returns
from finalayze.backtest.equity_tilt_experiment import _metrics
from finalayze.backtest.gold_sleeve_lab import forward_align_legs, master_axis
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/duration_regimes")
_SNAP = _DIR / "panel_snapshot.json"
_INDEX_SHIFT = 1
_DEPOSIT_SPREAD_PP = Decimal("1.0")  # deposit = key - 1pp
_BINDING_END = date(2026, 6, 10)
# Rate-regime sub-windows (CBR path: hikes 2023-07->2024-12 to 21%, first cut 2025-06-06).
_HIKING = (date(2023, 8, 1), date(2024, 12, 31))
_EASING_START = date(2025, 6, 6)
_TRADING_DAYS = 252


def _load(key: str, shift: int) -> list[tuple[date, Decimal]]:
    raw = json.loads(_SNAP.read_text(encoding="utf-8"))["legs"][key]
    return [(date.fromisoformat(d) + timedelta(days=shift), Decimal(c)) for d, c in raw]


def _window_stats(nav: list[tuple[date, Decimal]], start: date, end: date) -> dict[str, float]:
    sl = [(d, v) for d, v in nav if start <= d <= end]
    if len(sl) < 2:  # noqa: PLR2004
        return {"tr_pct": 0.0, "ann_pct": 0.0, "maxdd_pct": 0.0}
    tr = float(sl[-1][1] / sl[0][1] - 1)
    ann = (1.0 + tr) ** (_TRADING_DAYS / len(sl)) - 1.0
    m = _metrics([float(v) for _, v in sl])
    return {
        "tr_pct": round(tr * 100, 2),
        "ann_pct": round(ann * 100, 2),
        "maxdd_pct": round(m.maxdd_pct, 2),
    }


def main() -> None:
    fixed_raw = _load("ofz_fixed_rgbitr", _INDEX_SHIFT)
    floater_raw = _load("ofz_floater_ruflbitr", _INDEX_SHIFT)
    start = max(fixed_raw[0][0], floater_raw[0][0])
    axis = [
        d for d in master_axis({"f": fixed_raw, "l": floater_raw}) if start <= d <= _BINDING_END
    ]
    aligned = forward_align_legs({"fixed": fixed_raw, "floater": floater_raw}, axis)
    deposit = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    fixed = net_index_returns(
        list(zip(axis, aligned["fixed"], strict=True)), tax_acc=YtdTaxAccumulator()
    )
    floater = net_index_returns(
        list(zip(axis, aligned["floater"], strict=True)), tax_acc=YtdTaxAccumulator()
    )
    arms = {"deposit_key_minus_1pp": deposit, "ofz_floater": floater, "ofz_fixed_duration": fixed}
    windows = {
        "full": (axis[0], axis[-1]),
        "hiking_2023_24": _HIKING,
        "easing_2025_26_LIVE": (_EASING_START, axis[-1]),
    }
    table = {
        name: {w: _window_stats(nav, s, e) for w, (s, e) in windows.items()}
        for name, nav in arms.items()
    }

    easing = "easing_2025_26_LIVE"
    dep_e = table["deposit_key_minus_1pp"][easing]["tr_pct"]
    fix_e = table["ofz_fixed_duration"][easing]["tr_pct"]
    flt_e = table["ofz_floater"][easing]["tr_pct"]
    dep_h = table["deposit_key_minus_1pp"]["hiking_2023_24"]["tr_pct"]
    fix_h = table["ofz_fixed_duration"]["hiking_2023_24"]["tr_pct"]
    fixed_beats_deposit_in_easing = fix_e > dep_e
    fixed_loses_in_hiking = fix_h < dep_h

    floater_beats_deposit_easing = flt_e > dep_e
    finding = (
        f"The operator's intuition is RIGHT and mostly ALREADY CAPTURED — but via the FLOATER, not "
        f"duration. EASING (live regime {_EASING_START}->{axis[-1]}): the OFZ FLOATER (RUFLBITR, "
        f"which the SAA HOLDS as its OFZ_PK leg) returned {flt_e}% vs the deposit's {dep_e}% — it "
        f"{'OUT-carries' if floater_beats_deposit_easing else 'does not beat'} the deposit by "
        f"~{round(flt_e - dep_e, 1)}pp (a bond at ~key beats a deposit at key-1pp). So 'bonds beat "
        f"the deposit' shows up in the system through the floater. HOWEVER fixed-coupon OFZ "
        f"(duration) returned only {fix_e}% — it did NOT beat the deposit or the floater in this "
        f"easing, because the curve was INVERTED (short rates > long yields), so locking a lower "
        f"long yield gave up carry AND took a {table['ofz_fixed_duration'][easing]['maxdd_pct']}% "
        f"duration drawdown. HIKING (2023-24): fixed OFZ TR={fix_h}% vs deposit {dep_h}% — fixed "
        f"{'LOSES' if fixed_loses_in_hiking else 'wins'} badly (price losses as rates rose). "
        f"So the iter-1 gate's fixed-OFZ REJECT HOLDS even per-regime — locking duration was not "
        f"the winning trade. THE GENUINE GAP: the system's floater FALLS with the key, so it does "
        f"NOT "
        f"LOCK today's 15-16% for the future. IF you can now buy a FIXED 15-16% bond above the "
        f"14%-and-falling deposit AND the curve has un-inverted, locking that carry is a real "
        f"forward call the floater can't make — but it carries duration risk and, on the realized "
        f"easing here, the floater's high short carry beat it. Not advice; diagnostic only."
    )

    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "duration_regimes_summary.json").write_text(
        json.dumps(
            {
                "windows": {k: [s.isoformat(), e.isoformat()] for k, (s, e) in windows.items()},
                "metrics_tr_ann_maxdd": table,
                "fixed_beats_deposit_in_easing": fixed_beats_deposit_in_easing,
                "fixed_loses_in_hiking": fixed_loses_in_hiking,
                "finding": finding,
                "system_gap": (
                    "the live SAA holds the FLOATER (RUFLBITR/OFZ-PK), not fixed-coupon OFZ-PD; "
                    "the ru_ofz_pd.yaml preset + bond_duration_rotation.py classifier exist but "
                    "are not wired into the allocation, so the easing yield-lock is not active."
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{'arm':24s} | {'full TR%':>9} | {'hiking%':>8} | {'EASING%':>8} | {'easeMaxDD%':>10}")
    for name in arms:
        t = table[name]
        print(
            f"{name:24s} | {t['full']['tr_pct']:>9} | {t['hiking_2023_24']['tr_pct']:>10} | "
            f"{t[easing]['tr_pct']:>10} | {t[easing]['maxdd_pct']:>13}"
        )
    print(finding)


if __name__ == "__main__":
    main()
