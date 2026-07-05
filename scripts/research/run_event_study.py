"""Deterministic news-event study cert -- JUMP vs tradeable DRIFT, from the snapshot.

Reads the committed panel (:mod:`scripts.research.fetch_event_study_panel`) and answers,
honestly and with its own limits stated, the operator's question: can a MOEX retail
investor trade an UNANTICIPATED news shock?

Two separate findings (kept apart on purpose -- an adversarial review showed conflating
them overclaims):

1. THE JUMP (robust, across the named events): how much of the abnormal move is already
   priced before a realistic retail entry, and whether the NAIVE headline direction even
   held up. Measured per affected ticker with a STABLE lead metric (missed favourable
   jump, an absolute abnormal %) and a jump_share ratio shown ONLY when it is a
   meaningful fraction (near-monotone move) -- never the near-zero-denominator artifact.

2. THE DRIFT (the retail-tradeable question): 4 of 5 named events are bad-news
   SHORT-ONLY (retail on MOEX cannot short single names), so they say nothing about what
   retail could CAPTURE. The only LONG-accessible named event is SBER, N=1. To avoid a
   verdict resting on one boundary case, the cert also builds a DIRECTION-BLIND BASE
   RATE: across ALL names it takes EVERY large positive/negative abnormal daily move (a
   news-shock proxy, not hand-picked) and measures the median net long drift of
   chasing the pop / buying the dip -- judged against the deposit carry over the same
   window. That base rate, not the single event, drives the drift finding.

NO NETWORK, NO real money. Run:  uv run python scripts/research/run_event_study.py
"""

from __future__ import annotations

import bisect
import json
import statistics
from datetime import date
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.event_study_lab import (
    NDFL_RATE,
    RETAIL_PER_SIDE_COST,
    abnormal_return,
    decompose_event,
    jump_share_reliable,
    net_abnormal_long_return,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "results" / "research" / "event_study" / "panel_snapshot.json"
OUT_JSON = PROJECT_ROOT / "results" / "research" / "event_study" / "event_study_summary.json"
OUT_MD = PROJECT_ROOT / "results" / "research" / "event_study" / "event_study_report.md"

_HORIZONS = [1, 3, 5, 10]
_PRIMARY_H = 5
_HUNDRED = Decimal(100)
_ZERO = Decimal(0)

# ── Deposit-carry benchmark (the retail opportunity cost over the holding window) ──
# CBR key rate in effect on each event anchor date; deposit = key - 1pp, net of NDFL.
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_DAYS_PER_YEAR = Decimal(365)
_KEY_RATE_PCT = {
    "2022-06-30": Decimal("9.5"),
    "2023-09-21": Decimal("13.0"),
    "2022-09-26": Decimal("7.5"),
    "2023-03-17": Decimal("7.5"),
    "2023-06-26": Decimal("7.5"),
}
# A day-of move is "mostly market" (confounded) if the abnormal part is under this share
# of the raw move -- the news added little beyond beta.
_CONFOUND_ABN_SHARE_MAX = Decimal("0.40")
# The jump-capturability gate only applies to a MATERIAL shock (a genuine news move);
# a sub-3% day-of wobble is noise, not a jump, and must not pollute the gate.
_MATERIAL_SHOCK_PCT = 3.0
# Direction-blind base-rate scan: an abnormal daily move at/above this magnitude is a
# "news-shock" proxy. Swept at two thresholds; the retail action is LONG either way
# (chase the pop / buy the dip -- both retail-accessible, unlike shorting).
_SHOCK_THRESHOLDS = [Decimal("0.03"), Decimal("0.05")]
_BASE_RATE_H = [5, 10]
# A base-rate median net drift must clear the deposit carry over the window (~0.2-0.3%
# net per 10 trading days at ~7.5%) to count as systematically capturable.
_DEPOSIT_REF_PCT = 0.3
# A material shock's jump is "uncapturable" when this share of the abnormal move was
# already priced before a realistic retail entry.
_JUMP_PRICED_SHARE_MIN = 0.80


class Series:
    """One security's date-indexed OHLC with trading-day arithmetic."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.dates: list[date] = [date.fromisoformat(r["d"]) for r in rows]
        self.open: dict[date, Decimal] = {date.fromisoformat(r["d"]): Decimal(r["o"]) for r in rows}
        self.close: dict[date, Decimal] = {
            date.fromisoformat(r["d"]): Decimal(r["c"]) for r in rows
        }

    def session_on_or_after(self, anchor: date) -> date | None:
        idx = bisect.bisect_left(self.dates, anchor)
        return self.dates[idx] if idx < len(self.dates) else None

    def prev_session(self, session: date) -> date | None:
        idx = bisect.bisect_left(self.dates, session)
        return self.dates[idx - 1] if idx > 0 else None

    def shift(self, session: date, n: int) -> date | None:
        idx = bisect.bisect_left(self.dates, session)
        if idx >= len(self.dates) or self.dates[idx] != session:
            return None
        tgt = idx + n
        return self.dates[tgt] if 0 <= tgt < len(self.dates) else None


def _pct(x: Decimal | None) -> float | None:
    return None if x is None else float(x * _HUNDRED)


def _deposit_net_annual(anchor_iso: str) -> Decimal:
    key = _KEY_RATE_PCT.get(anchor_iso, Decimal("7.5"))
    return (key - _DEPOSIT_SPREAD_PP) / _HUNDRED * (Decimal(1) - NDFL_RATE)


def _analyse_ticker(
    ticker: str,
    tkr: Series,
    bench: Series | None,
    anchor: date,
    direction: int,
    entry_mode: str,
    deposit_net_annual: Decimal,
) -> dict[str, object] | None:
    reaction = tkr.session_on_or_after(anchor)
    if reaction is None:
        return None
    pre = tkr.prev_session(reaction)
    if pre is None:
        return None
    pre_close = tkr.close[pre]
    reaction_close = tkr.close[reaction]

    def bmark(session: date, field: str) -> Decimal | None:
        if bench is None:
            return None
        book = bench.close if field == "close" else bench.open
        return book.get(session)

    reaction_raw = reaction_close / pre_close - Decimal(1)
    b_pre_c, b_rx_c = bmark(pre, "close"), bmark(reaction, "close")
    reaction_abn = (
        abnormal_return(pre_close, reaction_close, b_pre_c, b_rx_c)
        if b_pre_c is not None and b_rx_c is not None
        else reaction_raw
    )

    if entry_mode == "overnight":
        entry_session, entry_field, entry_price = reaction, "open", tkr.open[reaction]
        aggressive_session, aggressive_price, aggressive_field = (
            reaction,
            tkr.close[reaction],
            "close",
        )
    else:
        nxt = tkr.shift(reaction, 1)
        if nxt is None:
            return None
        entry_session, entry_field, entry_price = nxt, "open", tkr.open[nxt]
        aggressive_session, aggressive_price, aggressive_field = reaction, reaction_close, "close"

    def decomp_at(entry_sess: date, entry_px: Decimal, entry_fld: str, horizon: int) -> dict:
        exit_sess = tkr.shift(reaction, horizon)
        if exit_sess is None or exit_sess not in tkr.close:
            return {}
        dec = decompose_event(
            pre_close=pre_close,
            entry_price=entry_px,
            exit_price=tkr.close[exit_sess],
            bench_pre=bmark(pre, "close"),
            bench_entry=bmark(entry_sess, entry_fld),
            bench_exit=bmark(exit_sess, "close"),
            direction=direction,
            per_side_cost=RETAIL_PER_SIDE_COST,
            ndfl=NDFL_RATE,
        )
        reliable = jump_share_reliable(dec.jump_abn, dec.total_abn)
        carry = deposit_net_annual * Decimal((exit_sess - entry_sess).days) / _DAYS_PER_YEAR
        return {
            "jump_share": float(dec.jump_share) if dec.jump_share is not None else None,
            "jump_share_reliable": reliable,
            "drift_net_pct": _pct(dec.traded_drift_net),
            "missed_jump_pct": _pct(dec.missed_favorable_jump),
            "deposit_carry_pct": _pct(carry),
            # only meaningful for a LONG-accessible (good-news) event
            "beats_deposit": bool(direction > 0 and dec.traded_drift_net > carry),
        }

    realistic = {h: decomp_at(entry_session, entry_price, entry_field, h) for h in _HORIZONS}
    aggressive = decomp_at(aggressive_session, aggressive_price, aggressive_field, _PRIMARY_H)
    prim = realistic[_PRIMARY_H]

    abn_share = abs(reaction_abn) / abs(reaction_raw) if reaction_raw != _ZERO else Decimal(1)
    return {
        "ticker": ticker,
        "reaction_raw_pct": _pct(reaction_raw),
        "reaction_abn_pct": _pct(reaction_abn),
        "prediction_correct": bool(reaction_abn * Decimal(direction) > _ZERO),
        "confounded": bool(abn_share < _CONFOUND_ABN_SHARE_MAX),
        "retail_long_accessible": direction > 0,
        "missed_jump_pct": prim.get("missed_jump_pct"),
        "jump_share_realistic": prim.get("jump_share") if prim.get("jump_share_reliable") else None,
        "jump_share_aggressive": (
            aggressive.get("jump_share") if aggressive.get("jump_share_reliable") else None
        ),
        "drift_net_by_h_pct": {h: realistic[h].get("drift_net_pct") for h in _HORIZONS},
        "beats_deposit_by_h": {h: realistic[h].get("beats_deposit") for h in _HORIZONS},
        "deposit_carry_h5_pct": prim.get("deposit_carry_pct"),
    }


def _median(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def _shock_baseline(
    shares: dict[str, Series],
    bench: Series,
    threshold: Decimal,
    horizon: int,
    exclude: set[date],
) -> dict[str, object]:
    """Direction-blind base rate: median NET LONG drift after every large abnormal move.

    Returns pos (chase the pop) and neg (buy the dip) net-long-drift medians + counts.
    """
    pos: list[float] = []
    neg: list[float] = []
    for s in shares.values():
        for i in range(1, len(s.dates)):
            d, p = s.dates[i], s.dates[i - 1]
            if d in exclude:
                continue
            bp, bd = bench.close.get(p), bench.close.get(d)
            if bp is None or bd is None:
                continue
            move = abnormal_return(s.close[p], s.close[d], bp, bd)
            if abs(move) < threshold:
                continue
            entry, exit_ = s.shift(d, 1), s.shift(d, horizon)
            if entry is None or exit_ is None:
                continue
            eo, xc = s.open.get(entry), s.close.get(exit_)
            beo, bxc = bench.open.get(entry), bench.close.get(exit_)
            if eo is None or xc is None or beo is None or bxc is None:
                continue
            net = float(net_abnormal_long_return(eo, xc, beo, bxc) * _HUNDRED)
            (pos if move > _ZERO else neg).append(net)
    return {
        "threshold_pct": float(threshold * _HUNDRED),
        "horizon": horizon,
        "pos_shock_n": len(pos),
        "pos_chase_median_net_pct": _median(pos),
        "neg_shock_n": len(neg),
        "neg_dip_buy_median_net_pct": _median(neg),
    }


def main() -> None:
    snap = json.loads(PANEL.read_text())
    series = {sid: Series(rows) for sid, rows in snap["prices"].items()}

    results = []
    exclude: set[date] = set()
    for ev in snap["events"]:
        anchor = date.fromisoformat(ev["anchor"])
        exclude.add(anchor)
        bench = series.get(ev["benchmark"]) if ev["benchmark"] else None
        dep_annual = _deposit_net_annual(ev["anchor"])
        per_ticker = [
            a
            for tk in ev["tickers"]
            if tk in series
            for a in [
                _analyse_ticker(
                    tk, series[tk], bench, anchor, ev["direction"], ev["entry_mode"], dep_annual
                )
            ]
            if a is not None
        ]
        n_correct = sum(1 for t in per_ticker if t["prediction_correct"])
        results.append(
            {
                "key": ev["key"],
                "label": ev["label"],
                "anchor": ev["anchor"],
                "direction": ev["direction"],
                "entry_mode": ev["entry_mode"],
                "note": ev["note"],
                "retail_long_accessible": ev["direction"] > 0,
                "per_ticker": per_ticker,
                "median_missed_jump_pct": _median([t["missed_jump_pct"] for t in per_ticker]),
                "median_drift_net_h5_pct": _median(
                    [t["drift_net_by_h_pct"][_PRIMARY_H] for t in per_ticker]
                ),
                "prediction_correct_count": f"{n_correct}/{len(per_ticker)}",
            }
        )

    # Direction-blind base rate (the drift finding's real evidence, not the N=1 event).
    shares = {k: v for k, v in series.items() if k != "IMOEX"}
    imoex = series["IMOEX"]
    base_rate = [
        _shock_baseline(shares, imoex, thr, h, exclude)
        for thr in _SHOCK_THRESHOLDS
        for h in _BASE_RATE_H
    ]

    # ── Verdict, DERIVED from the numbers (never pre-baked) ────────────────────
    # JUMP: for MATERIAL shocks (real news moves), is the jump priced before a realistic
    # entry? Sub-3% day-of wobbles are noise and excluded from the gate.
    reliable_jumps = [
        t["jump_share_realistic"]
        for ev in results
        for t in ev["per_ticker"]
        if t["jump_share_realistic"] is not None
        and abs(t["reaction_abn_pct"] or 0.0) >= _MATERIAL_SHOCK_PCT
    ]
    jump_uncapturable = bool(reliable_jumps) and all(
        js >= _JUMP_PRICED_SHARE_MIN for js in reliable_jumps
    )

    # DRIFT: does chasing pops / buying dips systematically beat the deposit carry?
    base_medians = [
        m
        for b in base_rate
        for m in (b["pos_chase_median_net_pct"], b["neg_dip_buy_median_net_pct"])
        if m is not None
    ]
    drift_capturable = any(m > _DEPOSIT_REF_PCT for m in base_medians)

    jump_token = "JUMP_UNCAPTURABLE" if jump_uncapturable else "JUMP_PARTLY_CAPTURABLE"
    drift_token = (
        "POST_SHOCK_DRIFT_PRESENT_INVESTIGATE"
        if drift_capturable
        else "POST_SHOCK_LONG_DRIFT_NOT_SYSTEMATICALLY_CAPTURABLE"
    )
    verdict = f"{jump_token}__{drift_token}"

    summary = {
        "verdict": verdict,
        "jump_finding": jump_token,
        "drift_finding": drift_token,
        "params": {
            "horizons": _HORIZONS,
            "primary_horizon": _PRIMARY_H,
            "per_side_cost": str(RETAIL_PER_SIDE_COST),
            "ndfl": str(NDFL_RATE),
            "deposit_ref_pct": _DEPOSIT_REF_PCT,
        },
        "events": results,
        "base_rate": base_rate,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary)

    print(f"\nVERDICT: {verdict}")
    for ev in results:
        acc = "LONG-OK" if ev["retail_long_accessible"] else "short-only"
        print(
            f"  {ev['label']:<38} [{acc:<9}] pred={ev['prediction_correct_count']} "
            f"missed_jump={_fmt_pct(ev['median_missed_jump_pct'])} "
            f"drift_net_H5={_fmt_pct(ev['median_drift_net_h5_pct'])}"
        )
    print("  base rate (direction-blind, LONG, net):")
    for b in base_rate:
        print(
            f"    >|{b['threshold_pct']:.0f}%| H{b['horizon']:<2} "
            f"chase(n={b['pos_shock_n']})={_fmt_pct(b['pos_chase_median_net_pct'])} "
            f"dip(n={b['neg_shock_n']})={_fmt_pct(b['neg_dip_buy_median_net_pct'])}"
        )
    print(f"\nwrote {OUT_JSON.relative_to(PROJECT_ROOT)}\nwrote {OUT_MD.relative_to(PROJECT_ROOT)}")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}%"


def _write_report(summary: dict) -> None:
    p = summary["params"]
    lines = [
        "# News-event study -- JUMP vs tradeable DRIFT (deterministic cert)",
        "",
        f"**Verdict: `{summary['verdict']}`**",
        f"(jump: `{summary['jump_finding']}` · drift: `{summary['drift_finding']}`)",
        "",
        f"Costs: {p['per_side_cost']}/side round-trip, NDFL {p['ndfl']} on gains. Abnormal = "
        "asset minus **price-return** IMOEX (beta=1; total-return would shave ~0.15%/5d off "
        "long drift -- conservative for the long test). Realistic entry = next-session open "
        "for intraday news, the gap open for overnight/weekend. `missed_favourable_jump` (an "
        "absolute abnormal %) is the STABLE lead metric; `jump_share` is shown ONLY when it is "
        "a meaningful in-`[0,1]` fraction (near-monotone move) and `n/a` when the move "
        "overshot and reversed.",
        "",
        "## 1. The JUMP -- was the move gone before a retail reader could act?",
        "",
    ]
    for ev in summary["events"]:
        acc = (
            "LONG-accessible"
            if ev["retail_long_accessible"]
            else "SHORT-only (retail cannot short)"
        )
        missed = _fmt_pct(ev["median_missed_jump_pct"])
        drift_h5 = _fmt_pct(ev["median_drift_net_h5_pct"])
        lines += [
            f"### {ev['label']}  ({ev['anchor']}, naive dir {ev['direction']:+d}, {acc})",
            "",
            f"_{ev['note']}_",
            "",
            f"- predicted direction correct: **{ev['prediction_correct_count']}** tickers",
            f"- median favourable jump MISSED before entry: **{missed}**",
            f"- median NET tradeable drift @H{_PRIMARY_H}: **{drift_h5}**",
            "",
            "| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | "
            "jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |",
            "| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |",
        ]
        for t in ev["per_ticker"]:
            dh = t["drift_net_by_h_pct"]
            bh = t["beats_deposit_by_h"]
            drifts = "/".join(_fmt_pct(dh[h]) for h in _HORIZONS)
            if ev["retail_long_accessible"]:
                beats = f"{'Y' if bh[5] else 'N'}/{'Y' if bh[10] else 'N'}"
            else:
                beats = "n/a"
            lines.append(
                f"| {t['ticker']} | {_fmt_pct(t['reaction_raw_pct'])} | "
                f"{_fmt_pct(t['reaction_abn_pct'])} | "
                f"{'Y' if t['prediction_correct'] else 'N'} | "
                f"{'Y' if t['confounded'] else 'N'} | "
                f"{_fmt_pct(t['missed_jump_pct'])} | "
                f"{_fmt(t['jump_share_realistic'])}/{_fmt(t['jump_share_aggressive'])} | "
                f"{drifts} | {beats} |"
            )
        lines.append("")

    lines += [
        "## 2. The DRIFT -- direction-blind base rate (the real tradeable question)",
        "",
        "Only 1 of the 5 named events (SBER) is LONG-accessible, so the named set cannot "
        "settle retail capturability. This scans EVERY large abnormal daily move across all "
        "8 names (a news-shock proxy, not hand-picked) and reports the median NET LONG drift "
        "of chasing the pop / buying the dip, vs the deposit carry over the same window "
        "(~0.2-0.3% net per 10 trading days).",
        "",
        "| shock >= | horizon | chase pop (n, median net) | buy dip (n, median net) |",
        "| --- | :-: | ---: | ---: |",
    ]
    for b in summary["base_rate"]:
        lines.append(
            f"| |{b['threshold_pct']:.0f}%| | H{b['horizon']} | "
            f"n={b['pos_shock_n']}, {_fmt_pct(b['pos_chase_median_net_pct'])} | "
            f"n={b['neg_shock_n']}, {_fmt_pct(b['neg_dip_buy_median_net_pct'])} |"
        )
    lines += [
        "",
        "## Honest limits",
        "",
        "- **N and selection.** The 5 named events are hand-picked ex-post-large shocks; "
        "that biases the JUMP finding toward 'uncapturable'. The DRIFT finding leans on the "
        "direction-blind base rate instead (dozens of shocks), which is the fairer test.",
        "- **Daily data.** With no intraday bars the whole reaction-day move is charged to "
        "the un-capturable jump, so measured jump_share is an UPPER bound -- conservative.",
        "- **Short-only exclusion.** 4/5 named events are bad news; retail on MOEX cannot "
        "short single names, so those carry no capturability information (excluded from the "
        "drift verdict; shown only for the jump).",
        "- **One long event.** SBER's post-jump long drift is net-positive at H1/H5/H10 and "
        "beats the deposit at H10 -- but it is N=1 and could be the name's 2023 uptrend; the "
        "base rate is what tells us whether that generalises.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
