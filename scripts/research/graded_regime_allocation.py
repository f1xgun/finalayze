"""Graded-regime SAA: target-weight table by rate depth + honest comparison vs deposit.

Answers the operator's question — "at what rate should the equity/bond share start
rising, and does a GRADED response beat the deposit?" — two ways:

1. TABLE: the graded target weights (``core.allocation.graded_regime_weights``, tested)
   at rates [21 .. 7.5] for all three risk profiles. The shipped Phase-76 tilt is a
   BINARY switch (full jump to `easing` at the first cut); this grades the shift to how
   far the CBR key rate has fallen from its 21% peak toward a 7.5% neutral anchor.

2. COMPARISON: graded vs binary vs 100%-deposit over the REAL window, using the gate's
   own leg convention — equity = MCFTR total-return index (public ISS REST, no token),
   deposit = (key rate - 1pp) daily accrual, OFZ-PK floater = full key rate. Quarterly
   rebalance. Reports terminal total return + MaxDD per policy.

HONESTY: the easing cycle is N=1 and ~13 months old (first cut 2025-06-06, ongoing to
14.25% in 2026-07). This is a DESCRIPTIVE comparison, not a validating gate — it inherits
the Phase-75 N=1 caveat. Grading changes the POLICY SHAPE (responds to easing depth
instead of over-committing at the first cut); it cannot manufacture an edge on N=1 data.
Real money is untouched (no orders).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal

import httpx

from finalayze.core.allocation import graded_regime_weights
from finalayze.core.schemas import AssetClass

_ISS = "https://iss.moex.com/iss"
_PEAK = Decimal(21)  # cycle peak (CBR, 2024-10)
_NEUTRAL = Decimal("7.5")  # long-run neutral anchor (= the 2023 trough / pre-hike level)
_DEPOSIT_SPREAD = Decimal(1)  # deposit rate = key rate - 1pp (gate convention)

# Real CBR key-rate step path (effective date -> annual %), fetched 2026-07-08.
_KEY_RATE_STEPS: list[tuple[date, Decimal]] = [
    (date(2024, 10, 28), Decimal("21.0")),
    (date(2025, 6, 9), Decimal("20.0")),
    (date(2025, 7, 28), Decimal("18.0")),
    (date(2025, 9, 15), Decimal("17.0")),
    (date(2025, 10, 27), Decimal("16.5")),
    (date(2025, 12, 22), Decimal("16.0")),
    (date(2026, 2, 16), Decimal("15.5")),
    (date(2026, 3, 23), Decimal("15.0")),
    (date(2026, 4, 27), Decimal("14.5")),
    (date(2026, 6, 22), Decimal("14.25")),
]
_WINDOW_START = date(2024, 10, 28)  # peak (include a high_rate stretch before the easing)
_WINDOW_END = date(2026, 7, 8)
_REBALANCE_DAYS = 91  # ~quarterly

# balanced (operator default) regime vectors — config/allocation_profiles.yaml
_PROFILES = {
    "conservative": (
        {
            AssetClass.DEPOSIT: Decimal("0.75"),
            AssetClass.OFZ_PK: Decimal("0.10"),
            AssetClass.EQUITY: Decimal("0.15"),
        },
        {
            AssetClass.DEPOSIT: Decimal("0.45"),
            AssetClass.OFZ_PK: Decimal("0.35"),
            AssetClass.EQUITY: Decimal("0.20"),
        },
    ),
    "balanced": (
        {
            AssetClass.DEPOSIT: Decimal("0.60"),
            AssetClass.OFZ_PK: Decimal("0.10"),
            AssetClass.EQUITY: Decimal("0.30"),
        },
        {
            AssetClass.DEPOSIT: Decimal("0.25"),
            AssetClass.OFZ_PK: Decimal("0.40"),
            AssetClass.EQUITY: Decimal("0.35"),
        },
    ),
    "growth": (
        {
            AssetClass.DEPOSIT: Decimal("0.40"),
            AssetClass.OFZ_PK: Decimal("0.10"),
            AssetClass.EQUITY: Decimal("0.50"),
        },
        {
            AssetClass.DEPOSIT: Decimal("0.10"),
            AssetClass.OFZ_PK: Decimal("0.40"),
            AssetClass.EQUITY: Decimal("0.50"),
        },
    ),
}


def _key_rate(d: date) -> Decimal:
    rate = _KEY_RATE_STEPS[0][1]
    for eff, r in _KEY_RATE_STEPS:
        if d >= eff:
            rate = r
        else:
            break
    return rate


def _weight_table() -> None:
    print("=== GRADED target weights by key rate (deposit / OFZ-PK / equity) ===")
    print("(peak 21% = full high_rate; neutral 7.5% = full easing; today = 14.25%)\n")
    rates = [Decimal(r) for r in ("21", "18", "16", "14.25", "13", "10", "8", "7.5")]
    for name, (hr, ea) in _PROFILES.items():
        print(f"-- {name} --")
        print(f"   {'rate':>6} | {'deposit':>7} {'OFZ-PK':>7} {'equity':>7}")
        for r in rates:
            w = graded_regime_weights(hr, ea, r, _PEAK, _NEUTRAL)
            d = float(w[AssetClass.DEPOSIT]) * 100
            o = float(w[AssetClass.OFZ_PK]) * 100
            e = float(w[AssetClass.EQUITY]) * 100
            tag = "  <- today" if r == Decimal("14.25") else ""
            print(f"   {float(r):>6} | {d:>6.1f}% {o:>6.1f}% {e:>6.1f}%{tag}")
        print()


def _fetch_mcftr() -> dict[date, float]:
    out: dict[date, float] = {}
    start = 0
    while True:
        url = (
            f"{_ISS}/history/engines/stock/markets/index/securities/MCFTR.json"
            f"?iss.meta=off&iss.only=history&history.columns=TRADEDATE,CLOSE"
            f"&from={_WINDOW_START.isoformat()}&till={_WINDOW_END.isoformat()}&start={start}"
        )
        r = httpx.get(url, timeout=30.0)
        rows = r.json().get("history", {}).get("data", [])
        if not rows:
            break
        for d, close in rows:
            if close is not None:
                out[date.fromisoformat(d)] = float(close)
        start += len(rows)
        if len(rows) < 100:  # noqa: PLR2004
            break
    return out


def _daily_leg_returns(rate_fn) -> dict[date, float]:
    """Daily accrual factor per calendar day at an annual rate (ACT/365)."""
    out: dict[date, float] = {}
    d = _WINDOW_START
    while d <= _WINDOW_END:
        out[d] = float(rate_fn(d)) / 100 / 365
        d += timedelta(days=1)
    return out


def _simulate(mcftr: dict[date, float], policy: str) -> dict[str, float]:
    """Simulate a policy's daily TR curve; return terminal return + MaxDD.

    policy: 'graded' | 'binary' | 'deposit100'. Weights refreshed each rebalance from
    the key rate at that date; equity daily return from MCFTR, deposit/OFZ from accrual.
    """
    days = sorted(d for d in mcftr if _WINDOW_START <= d <= _WINDOW_END)
    dep_acc = _daily_leg_returns(lambda x: _key_rate(x) - _DEPOSIT_SPREAD)
    ofz_acc = _daily_leg_returns(_key_rate)
    hr, ea = _PROFILES["balanced"]

    def weights_at(d: date) -> dict[AssetClass, Decimal]:
        if policy == "deposit100":
            return {
                AssetClass.DEPOSIT: Decimal(1),
                AssetClass.OFZ_PK: Decimal(0),
                AssetClass.EQUITY: Decimal(0),
            }
        if policy == "binary":
            return ea if _key_rate(d) < _PEAK else hr  # easing once cut from peak (proxy)
        return graded_regime_weights(hr, ea, _key_rate(d), _PEAK, _NEUTRAL)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    w = weights_at(days[0])
    last_rebal = days[0]
    for i in range(1, len(days)):
        d_prev, d = days[i - 1], days[i]
        if (d - last_rebal).days >= _REBALANCE_DAYS:
            w = weights_at(d)
            last_rebal = d
        eq_ret = (mcftr[d] - mcftr[d_prev]) / mcftr[d_prev]
        # deposit/OFZ accrue over the calendar gap between trading days
        gap = (d - d_prev).days
        dep_ret = sum(dep_acc.get(d_prev + timedelta(days=k), 0.0) for k in range(gap))
        ofz_ret = sum(ofz_acc.get(d_prev + timedelta(days=k), 0.0) for k in range(gap))
        port_ret = (
            float(w[AssetClass.EQUITY]) * eq_ret
            + float(w[AssetClass.DEPOSIT]) * dep_ret
            + float(w[AssetClass.OFZ_PK]) * ofz_ret
        )
        equity *= 1 + port_ret
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    return {"terminal_return": equity - 1, "max_dd": max_dd}


def main() -> None:
    _weight_table()
    print("=== HONEST COMPARISON over the REAL window (balanced profile) ===")
    print(f"window {_WINDOW_START} .. {_WINDOW_END} (peak 21% -> 14.25%); quarterly rebalance")
    print("equity=MCFTR (ISS), deposit=(key-1pp), OFZ-PK=key; N=1 easing cycle -> DESCRIPTIVE\n")
    mcftr = _fetch_mcftr()
    print(f"MCFTR trading days: {len(mcftr)}\n")
    res = {p: _simulate(mcftr, p) for p in ("deposit100", "binary", "graded")}
    print(f"   {'policy':>11} | {'terminal TR':>12} | {'MaxDD':>7}")
    for p in ("deposit100", "binary", "graded"):
        tr = res[p]["terminal_return"] * 100
        dd = res[p]["max_dd"] * 100
        print(f"   {p:>11} | {tr:>10.1f}% | {dd:>6.1f}%")
    dep = res["deposit100"]["terminal_return"]
    print(
        "\n   graded vs deposit: "
        f"{(res['graded']['terminal_return'] - dep) * 100:+.1f}pp terminal; "
        f"binary vs deposit: {(res['binary']['terminal_return'] - dep) * 100:+.1f}pp"
    )
    print(
        json.dumps({"window": [_WINDOW_START.isoformat(), _WINDOW_END.isoformat()], "result": res})
    )


if __name__ == "__main__":
    main()
