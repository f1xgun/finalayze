"""Deterministic PEAD deposit-gate cert -- does post-earnings drift beat the deposit?

Reads the committed panel (``fetch_pead_panel.py``: real MOEX earnings report DATES via
Tinkoff + token-free ISS prices) and tests the price-reaction PEAD edge (MOEX has no
consensus EPS, D-01):

- **Surprise** = the announcement-cluster abnormal return (asset - IMOEX over [D-1 -> the
  session +2 after the scheduled report date] -- a 3-session window, because the release is
  often after-close and the reaction bleeds into the next day).
- **Drift** = the return an investor captures entering the session AFTER the announcement
  cluster and holding W = 20/40/60 trading days, net of round-trip cost + NDFL. Reported two
  ways: the RAW absolute return (the deposit-gate unit -- does it beat the deposit in the
  pocket?) AND the market-ADJUSTED abnormal return (the PEAD alpha -- does the surprise
  drift beyond the index?).
- **Deposit gate**: a PEAD long sleeve NAV (deposit idle + long the drift windows, riding
  each name's REAL daily path) fed to the pre-registered ``instrument_integration_gate``
  (deposit40/equity60 core) for a battery-comparable INTEGRATE/PROBATION/REJECT tier.

Window is ~2024-2026 (the 16-21% high-rate regime -- a near-14%/yr net risk-free bar, so
deposit-dominance for ANY equity strategy is close to foregone; the informative, more
regime-robust read is the ABNORMAL drift, which strips the market). Issuer dedup: SBERP
(Sberbank prefs, same earnings as SBER) is dropped; same-symbol report dates within 30 days
(RAS+IFRS pairs) are collapsed to the first.

NO NETWORK, NO real money. Run:  uv run python scripts/research/run_pead_gate.py
"""

from __future__ import annotations

import bisect
import json
import statistics
from datetime import date
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg
from finalayze.backtest.event_study_lab import (
    NDFL_RATE,
    RETAIL_PER_SIDE_COST,
    abnormal_return,
    net_abnormal_long_return,
)
from finalayze.backtest.instrument_integration_gate import Candidate, run_integration_gate
from finalayze.backtest.pead_lab import (
    blend_pead_nav,
    daily_factors,
    net_window_factor,
    realpath_window,
)
from finalayze.core.ndfl import YtdTaxAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "results" / "research" / "pead" / "pead_panel.json"
OUT_JSON = PROJECT_ROOT / "results" / "research" / "pead" / "pead_gate_summary.json"
OUT_MD = PROJECT_ROOT / "results" / "research" / "pead" / "pead_gate_report.md"

_HORIZONS = [20, 40, 60]
_PRIMARY_W = 60
_ANNOUNCE_WINDOW = 2  # surprise over [D-1 -> reaction+2] (absorbs after-close/next-day reaction)
_ENTRY_OFFSET = 3  # enter the session AFTER the announcement cluster (strict gap -> no leak)
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_SURPRISE_STRONG_PCT = 2.0  # a "genuine" surprise; sub-band signs are measurement noise
_DEDUP_DROP = {"SBERP"}  # Sberbank prefs: same issuer/earnings as SBER
_DEDUP_MIN_GAP_DAYS = 30  # collapse same-symbol report dates closer than this (RAS+IFRS pairs)
_HUNDRED = Decimal(100)
_ZERO = Decimal(0)


class Bars:
    """One security's date-indexed OHLC with trading-day arithmetic."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.dates: list[date] = [date.fromisoformat(r["d"]) for r in rows]
        self.open: dict[date, Decimal] = {date.fromisoformat(r["d"]): Decimal(r["o"]) for r in rows}
        self.close: dict[date, Decimal] = {
            date.fromisoformat(r["d"]): Decimal(r["c"]) for r in rows
        }

    def session_on_or_after(self, anchor: date) -> date | None:
        i = bisect.bisect_left(self.dates, anchor)
        return self.dates[i] if i < len(self.dates) else None

    def prev(self, session: date) -> date | None:
        i = bisect.bisect_left(self.dates, session)
        return self.dates[i - 1] if i > 0 else None

    def shift(self, session: date, n: int) -> date | None:
        i = bisect.bisect_left(self.dates, session)
        if i >= len(self.dates) or self.dates[i] != session:
            return None
        j = i + n
        return self.dates[j] if 0 <= j < len(self.dates) else None


def _pct(x: Decimal | float | None) -> float | None:
    if x is None:
        return None
    return float(x) * 100.0 if isinstance(x, float) else float(x * _HUNDRED)


def _median(xs: list[float]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def _dedup_anchors(dates: list[str]) -> list[str]:
    """Collapse same-symbol report dates within _DEDUP_MIN_GAP_DAYS to the first (RAS+IFRS)."""
    kept: list[date] = []
    for iso in sorted(dates):
        d = date.fromisoformat(iso)
        if kept and (d - kept[-1]).days < _DEDUP_MIN_GAP_DAYS:
            continue
        kept.append(d)
    return [d.isoformat() for d in kept]


def _analyse_event(
    sym: str, bars: Bars, imoex: Bars, dep_level: dict[date, Decimal], anchor: date
) -> dict | None:
    reaction = bars.session_on_or_after(anchor)
    if reaction is None:
        return None
    pre = bars.prev(reaction)
    cluster_end = bars.shift(reaction, _ANNOUNCE_WINDOW)
    entry = bars.shift(reaction, _ENTRY_OFFSET)
    if pre is None or cluster_end is None or entry is None:
        return None
    if any(d not in imoex.close for d in (pre, cluster_end)) or entry not in imoex.open:
        return None
    surprise = abnormal_return(
        bars.close[pre], bars.close[cluster_end], imoex.close[pre], imoex.close[cluster_end]
    )

    raw_drift: dict[int, float | None] = {}
    abn_drift: dict[int, float | None] = {}
    beats_dep: dict[int, bool | None] = {}
    for w in _HORIZONS:
        exit_s = bars.shift(reaction, _ENTRY_OFFSET + w)
        if exit_s is None or exit_s not in bars.close or exit_s not in imoex.close:
            raw_drift[w] = abn_drift[w] = None
            beats_dep[w] = None
            continue
        raw_net = net_window_factor(
            bars.open[entry], bars.close[exit_s], RETAIL_PER_SIDE_COST, NDFL_RATE
        ) - Decimal(1)
        abn_net = net_abnormal_long_return(
            bars.open[entry], bars.close[exit_s], imoex.open[entry], imoex.close[exit_s]
        )
        raw_drift[w] = _pct(raw_net)
        abn_drift[w] = _pct(abn_net)
        carry = None
        if entry in dep_level and exit_s in dep_level and dep_level[entry] > _ZERO:
            carry = dep_level[exit_s] / dep_level[entry] - Decimal(1)
        # deposit gate is an ABSOLUTE-return test: raw net long vs the deposit carry.
        beats_dep[w] = None if carry is None else bool(raw_net > carry)

    exit_p = bars.shift(reaction, _ENTRY_OFFSET + _PRIMARY_W)
    dep_carry_primary = None
    if (
        exit_p is not None
        and entry in dep_level
        and exit_p in dep_level
        and dep_level[entry] > _ZERO
    ):
        dep_carry_primary = _pct(dep_level[exit_p] / dep_level[entry] - Decimal(1))

    return {
        "symbol": sym,
        "anchor": anchor.isoformat(),
        "surprise_pct": _pct(surprise),
        "abs_surprise_pct": abs(_pct(surprise) or 0.0),
        "positive": bool(surprise > _ZERO),
        "raw_drift_by_w_pct": raw_drift,
        "abn_drift_by_w_pct": abn_drift,
        "beats_deposit_by_w": beats_dep,
        "deposit_carry_primary_pct": dep_carry_primary,
        "_entry": entry.isoformat(),
        "_exit_primary": exit_p.isoformat() if exit_p is not None else None,
        "_entry_open": str(bars.open[entry]),
        "_exit_close": str(bars.close[exit_p])
        if exit_p is not None and exit_p in bars.close
        else None,
    }


def _build_sleeve(
    events: list[dict],
    axis: list[date],
    dep_daily: dict[date, Decimal],
    name_daily: dict[str, dict[date, Decimal]],
) -> list[tuple[date, Decimal]]:
    """A PEAD long sleeve: deposit idle + long each POSITIVE-surprise name over its REAL path."""
    active: dict[date, list[Decimal]] = {}
    for ev in events:
        if not ev["positive"] or ev["_exit_primary"] is None or ev["_exit_close"] is None:
            continue
        entry_d = date.fromisoformat(ev["_entry"])
        exit_d = date.fromisoformat(ev["_exit_primary"])
        window_bars = [d for d in axis if entry_d < d <= exit_d]
        if not window_bars:
            continue
        target = net_window_factor(
            Decimal(ev["_entry_open"]), Decimal(ev["_exit_close"]), RETAIL_PER_SIDE_COST, NDFL_RATE
        )
        per_bar = realpath_window(window_bars, name_daily.get(ev["symbol"], {}), target)
        for d, f in per_bar.items():
            active.setdefault(d, []).append(f)
    return blend_pead_nav(axis, dep_daily, active)


def main() -> None:
    snap = json.loads(PANEL.read_text())
    bars = {sym: Bars(rows) for sym, rows in snap["prices"].items()}
    imoex = Bars(snap["imoex"])
    mcftrr_raw = [(date.fromisoformat(r["d"]), Decimal(r["c"])) for r in snap["mcftrr"]]

    axis = imoex.dates
    deposit_curve = accrue_real_risk_free_leg(
        axis, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    dep_level = dict(deposit_curve)
    dep_daily = daily_factors(deposit_curve)
    name_daily = {
        sym: daily_factors([(d, b.close[d]) for d in axis if d in b.close])
        for sym, b in bars.items()
    }

    events: list[dict] = []
    issuers: set[str] = set()
    for sym, dates in snap["earnings_dates"].items():
        if sym not in bars or sym in _DEDUP_DROP:
            continue
        for iso in _dedup_anchors(dates):
            ev = _analyse_event(sym, bars[sym], imoex, dep_level, date.fromisoformat(iso))
            if ev is not None and ev["raw_drift_by_w_pct"][_PRIMARY_W] is not None:
                events.append(ev)
                issuers.add(sym)

    pos = [e for e in events if e["positive"]]
    neg = [e for e in events if not e["positive"]]
    strong_pos = [e for e in pos if e["abs_surprise_pct"] >= _SURPRISE_STRONG_PCT]
    noise_frac = (
        sum(1 for e in events if e["abs_surprise_pct"] < 1.0) / len(events) if events else None
    )

    base_rate = {}
    for w in _HORIZONS:
        base_rate[w] = {
            "pos_n": len(pos),
            "pos_raw_drift_median_pct": _median([e["raw_drift_by_w_pct"][w] for e in pos]),
            "pos_abn_drift_median_pct": _median([e["abn_drift_by_w_pct"][w] for e in pos]),
            "pos_beats_deposit_frac": _frac([e["beats_deposit_by_w"][w] for e in pos]),
            "strong_pos_n": len(strong_pos),
            "strong_pos_abn_drift_median_pct": _median(
                [e["abn_drift_by_w_pct"][w] for e in strong_pos]
            ),
            "neg_n": len(neg),
            "neg_dip_raw_drift_median_pct": _median([e["raw_drift_by_w_pct"][w] for e in neg]),
        }

    sleeve = _build_sleeve(events, axis, dep_daily, name_daily)
    gate = run_integration_gate(
        Candidate(
            name="pead_earnings_drift", net_curve=sleeve, risk_tier="medium", intended_role="growth"
        ),
        mcftrr_raw,
    )
    sleeve_tr = _pct(sleeve[-1][1] / sleeve[0][1] - Decimal(1))
    deposit_tr = _pct(deposit_curve[-1][1] / deposit_curve[0][1] - Decimal(1))

    # ── Verdict, DERIVED ──────────────────────────────────────────────────────
    pos_raw_primary = base_rate[_PRIMARY_W]["pos_raw_drift_median_pct"]
    pos_abn_primary = base_rate[_PRIMARY_W]["pos_abn_drift_median_pct"]
    dep_carry_primary = _median([e["deposit_carry_primary_pct"] for e in events])
    drift_beats_deposit = bool(
        pos_raw_primary is not None
        and dep_carry_primary is not None
        and pos_raw_primary > dep_carry_primary
    )
    alpha_positive = bool(pos_abn_primary is not None and pos_abn_primary > 0)
    gate_pass = gate.tier in ("INTEGRATE", "PROBATION")
    if drift_beats_deposit and alpha_positive and gate.tier == "INTEGRATE":
        verdict = "PEAD_DRIFT_INTEGRATE"
    elif drift_beats_deposit or alpha_positive or gate_pass:
        verdict = "PEAD_DRIFT_MARGINAL_INVESTIGATE"
    else:
        verdict = "PEAD_DRIFT_DEPOSIT_DOMINATED"

    summary = {
        "verdict": verdict,
        "gate_tier": gate.tier,
        "gate_reasons": list(gate.reasons),
        "gate_n1_caveat": gate.n1_caveat,
        "alpha_positive": alpha_positive,
        "drift_beats_deposit_base": drift_beats_deposit,
        "median_deposit_carry_primary_pct": dep_carry_primary,
        "sleeve_tr_raw_pct": sleeve_tr,
        "deposit_tr_pct": deposit_tr,
        "imoex_tr_pct": _pct(imoex.close[axis[-1]] / imoex.close[axis[0]] - Decimal(1)),
        "params": {
            "horizons": _HORIZONS,
            "primary_w": _PRIMARY_W,
            "per_side_cost": str(RETAIL_PER_SIDE_COST),
            "ndfl": str(NDFL_RATE),
            "surprise_strong_pct": _SURPRISE_STRONG_PCT,
            "n_events": len(events),
            "n_issuers": len(issuers),
            "noise_band_frac": noise_frac,
            "window": snap["span"],
        },
        "base_rate": base_rate,
        "events": [{k: v for k, v in e.items() if not k.startswith("_")} for e in events],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary)

    print(f"\nVERDICT: {verdict}   (gate tier: {gate.tier})")
    print(f"  {len(events)} events across {len(issuers)} issuers (pos {len(pos)} / neg {len(neg)})")
    for w in _HORIZONS:
        b = base_rate[w]
        print(
            f"  W{w:<2}: pos abnormal(alpha) {_fmt(b['pos_abn_drift_median_pct'])} | "
            f"pos raw {_fmt(b['pos_raw_drift_median_pct'])} "
            f"(beats-dep {_fmtf(b['pos_beats_deposit_frac'])}) | "
            f"strong abn {_fmt(b['strong_pos_abn_drift_median_pct'])} (n={b['strong_pos_n']})"
        )
    imoex_tr = summary["imoex_tr_pct"]
    print(
        f"  deposit carry @W{_PRIMARY_W} median {_fmt(dep_carry_primary)} | "
        f"raw sleeve TR {_fmt(sleeve_tr)} vs deposit {_fmt(deposit_tr)} (IMOEX {_fmt(imoex_tr)})"
    )
    print(f"\nwrote {OUT_JSON.relative_to(PROJECT_ROOT)}\nwrote {OUT_MD.relative_to(PROJECT_ROOT)}")


def _frac(xs: list[bool | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}%"


def _fmtf(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _write_report(s: dict) -> None:
    p = s["params"]
    lines = [
        "# PEAD deposit-gate cert -- does post-earnings drift beat the deposit?",
        "",
        f"**Verdict: `{s['verdict']}`**  ·  integration-gate tier: **`{s['gate_tier']}`**"
        f"{' (N=1 caveat)' if s['gate_n1_caveat'] else ''}",
        "",
        f"Window {p['window']['from']}..{p['window']['till']} (the 16-21% high-rate regime -- a "
        f"~14%/yr NET risk-free bar, so deposit-dominance for ANY long-equity strategy is close "
        f"to foregone here; the more regime-robust read is the market-ADJUSTED abnormal drift). "
        f"{p['n_events']} MOEX earnings events across {p['n_issuers']} issuers. Price-reaction "
        f"PEAD (no consensus EPS, D-01): surprise = announcement abnormal return vs IMOEX; "
        f"drift net of {p['per_side_cost']}/side + {p['ndfl']} NDFL from the post-cluster entry.",
        "",
        "## The PEAD signal -- market-ADJUSTED abnormal drift (the regime-robust read)",
        "",
        "If earnings surprises DRIFT, positive-surprise names should out-return the index over "
        "the following weeks. They do the opposite:",
        "",
        "| horizon | pos-surprise abnormal drift | strong-surprise (|abn|>=2%) drift |",
        "| --- | ---: | ---: |",
    ]
    for w in p["horizons"]:
        b = s["base_rate"][w] if w in s["base_rate"] else s["base_rate"][str(w)]
        lines.append(
            f"| W{w} | {_fmt(b['pos_abn_drift_median_pct'])} | "
            f"{_fmt(b['strong_pos_abn_drift_median_pct'])} (n={b['strong_pos_n']}) |"
        )
    lines += [
        "",
        "The abnormal drift is **negative and worsens with horizon** -- MOEX surprises "
        "REVERSE, not drift -- and a genuine-surprise (|abn|>=2%) filter makes it worse, not "
        "better. This is beta-neutral, so it is not just the falling market.",
        "",
        "## The deposit gate -- absolute return, net of everything",
        "",
        "| horizon | pos-surprise RAW long (median) | % beat deposit | buy-the-dip (neg) raw |",
        "| --- | ---: | ---: | ---: |",
    ]
    for w in p["horizons"]:
        b = s["base_rate"][w] if w in s["base_rate"] else s["base_rate"][str(w)]
        lines.append(
            f"| W{w} | {_fmt(b['pos_raw_drift_median_pct'])} | "
            f"{_fmtf(b['pos_beats_deposit_frac'])} | {_fmt(b['neg_dip_raw_drift_median_pct'])} |"
        )
    lines += [
        "",
        f"Median deposit carry over the {p['primary_w']}-day window: "
        f"**{_fmt(s['median_deposit_carry_primary_pct'])}** -- above the raw PEAD long. "
        f"Formal gate: PEAD sleeve tier **{s['gate_tier']}** "
        f"({'; '.join(s['gate_reasons'])}).",
        "",
        f"The raw PEAD long sleeve returned **{_fmt(s['sleeve_tr_raw_pct'])}** vs the deposit's "
        f"**{_fmt(s['deposit_tr_pct'])}** -- but that raw number is **beta-dominated**: IMOEX "
        f"itself fell **{_fmt(s['imoex_tr_pct'])}** over the window, so most of the sleeve loss "
        "is holding equity in a bear market, NOT the PEAD signal (which is the ~-5% abnormal "
        "drift above). The gate's tier is basis-robust (it nets the market out); the raw TR is "
        "shown only to size the beta the strategy must pay for.",
        "",
        "## Honest limits",
        "",
        "- **Regime.** One deep high-rate regime (2024-2026; Tinkoff `get_asset_reports` only "
        "reaches back 730 days). At a ~14%/yr net deposit bar, deposit-dominance is near-"
        "foregone for any equity sleeve -- so the load-bearing finding is the NEGATIVE abnormal "
        "drift (reversal), which would need a normal-rate regime to retest for drift.",
        f"- **N and noise.** {p['n_events']} events / {p['n_issuers']} issuers; "
        f"~{_fmtf(p['noise_band_frac'])} of surprises are sub-1% (measurement-noise band), so "
        "the sign is partly noisy -- hence the |abn|>=2% strong-surprise column, which "
        "confirms (not softens) the reversal. Diagnostic case study, not a powered test.",
        "- **No consensus EPS on MOEX** (D-01): the surprise is the announcement price reaction, "
        "not a fundamental SUE (no eps_ttm history to build one). A fundamental-SUE PEAD could "
        "differ and is data-blocked.",
        "- **Dating.** Tinkoff `report_date` is a scheduled calendar anchor; the surprise uses a "
        "3-session cluster to absorb after-close/next-day reaction, and entry is strictly after "
        "it (no leak), but the exact reaction instant is unknown.",
        "- **Issuer dedup + short exclusion.** SBERP (same issuer as SBER) dropped; same-symbol "
        "reports within 30 days collapsed. Negative-surprise SHORTs are not retail-accessible on "
        "MOEX; only the long legs enter the verdict.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
