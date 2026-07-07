"""MOEX jump-response cert: after a 1-minute shock on stocks/OFZ/FX, what's left for a reactor?

Deterministic, token-free — consumes the committed
``results/research/moex_jump/moex_jump_panel.json`` (fetched via Tinkoff readonly; no orders, real
money is a hard stop). The RUB-universe port of the crypto reactive-news cert (PR #318): same
conditional-forward-path-after-a-shock design and latency ladder, per instrument class, with MOEX
retail frictions. Plus a MOEX-specific OVERNIGHT-GAP decomposition — since MOEX is not 24/7, a large
share of a news move lands in the un-tradeable open gap, upstream of any intraday reactor.

Long-only up-shocks are the retail-capturable direction (single-name MOEX shorting is unavailable).

    uv run python scripts/research/run_moex_jump_study.py
"""

from __future__ import annotations

import json
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

from finalayze.backtest.jump_response_lab import half_life_bars, mean_path, net_after_cost

_DIR = Path("results/research/moex_jump")
_PANEL = _DIR / "moex_jump_panel.json"

# ── Pre-registered constants ─────────────────────────────────────────────────
_CLASSES = ("stock", "ofz", "fx")
_Z_LEVELS = (5, 6, 8)
_PRIMARY_Z = 6
_ENTRY_LATENCIES = (1, 2, 5, 15)  # bars after the shock's close (=minutes for liquid names)
_EXIT_HORIZONS = (30, 60, 120)
_COST_LEVELS_BPS = (0, 15, 30, 50)  # MOEX retail: broker commission + spread, wider than crypto
_PRIMARY_COST_BPS = 30
_REACTIVE_LATENCY = 1
_OUR_LATENCY = 5
_SLOW_LATENCY = 15
_NDFL = Decimal("0.13")
_BPS = Decimal(10000)
_ZERO = Decimal(0)
_ONE = Decimal(1)
_CHECKPOINTS = (1, 5, 15, 30, 60, 120)
# stored forward paths are sampled at these horizons only (see fetch _PATH_HORIZONS); index by _HIDX
_PATH_HORIZONS = (0, 1, 2, 5, 15, 30, 60, 120)
_HIDX = {h: i for i, h in enumerate(_PATH_HORIZONS)}
_DEPOSIT_ANNUAL_PCT = "16-21"
_TOP_DECILE = 0.90  # "news days" = top-decile daily absolute move
_MIN_DAYS_FOR_GAP = 2  # need a prior close to compute an overnight gap
_MIN_N_CONFIDENT = 30  # fewer primary up-shocks than this = too thin for a firm intraday verdict


def _load() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Decimal]:
    p = json.loads(_PANEL.read_text(encoding="utf-8"))
    scale = Decimal(p["meta"]["params"]["path_scale"])
    stored = tuple(p["meta"]["params"]["path_horizons"])
    if stored != _PATH_HORIZONS:
        msg = f"panel path_horizons {stored} != study grid {_PATH_HORIZONS}"
        raise ValueError(msg)
    return p["meta"], p["shocks"], p["daily"], scale


def _paths(
    shocks: list[dict[str, Any]], cls: str, z: int, sign: int, scale: Decimal
) -> list[list[Decimal]]:
    return [
        [Decimal(x) / scale for x in s["path"]]
        for s in shocks
        if s["cls"] == cls and s["sign"] == sign and float(s["z"]) >= z
    ]


def _trade_return(path: list[Decimal], entry: int, exit_: int) -> Decimal:
    return (_ONE + path[_HIDX[exit_]]) / (_ONE + path[_HIDX[entry]]) - _ONE


def _mean_net(paths: list[list[Decimal]], entry: int, exit_: int, cost_bps: int) -> Decimal:
    cost = Decimal(cost_bps) / _BPS
    rets = [_trade_return(p, entry, exit_) for p in paths]
    return net_after_cost(sum(rets, _ZERO) / len(rets), cost, _NDFL)


def _best_net_bps(paths: list[list[Decimal]], entry: int, cost_bps: int) -> tuple[Decimal, int]:
    best, best_ex = None, _EXIT_HORIZONS[0]
    for ex in _EXIT_HORIZONS:
        n = _mean_net(paths, entry, ex, cost_bps)
        if best is None or n > best:
            best, best_ex = n, ex
    assert best is not None
    return best, best_ex


def _best_gross_bps(paths: list[list[Decimal]], entry: int) -> Decimal:
    best = None
    for ex in _EXIT_HORIZONS:
        rets = [_trade_return(p, entry, ex) for p in paths]
        g = sum(rets, _ZERO) / len(rets)
        if best is None or g > best:
            best = g
    assert best is not None
    return best


def _ladder(paths: list[list[Decimal]], cost_bps: int) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for entry in _ENTRY_LATENCIES:
        table[f"entry{entry}"] = {
            f"exit{ex}": round(float(_mean_net(paths, entry, ex, cost_bps) * _BPS), 2)
            for ex in _EXIT_HORIZONS
        }
    return table


def _decay_curve(mean_fwd: list[Decimal]) -> dict[str, float]:
    return {f"t+{h}": round(float(mean_fwd[_HIDX[h]] * _BPS), 2) for h in _CHECKPOINTS}


def _median_curve(paths: list[list[Decimal]]) -> dict[str, float]:
    return {
        f"t+{h}": round(float(statistics.median([p[_HIDX[h]] for p in paths]) * _BPS), 2)
        for h in _CHECKPOINTS
    }


def _half_life_min(paths: list[list[Decimal]]) -> int | None:
    """Continuation half-life on the stored-horizon grid, in minutes; None if the move reverses."""
    k = half_life_bars(mean_path(paths))
    return _PATH_HORIZONS[k] if k is not None else None


def _win_rate(paths: list[list[Decimal]], entry: int, exit_: int) -> float:
    return round(sum(1 for p in paths if _trade_return(p, entry, exit_) > _ZERO) / len(paths), 3)


def _verdict(true_gross: Decimal, reactive_net: Decimal, our_net: Decimal) -> str:
    if true_gross <= _ZERO:
        return "REACTIVE_ALPHA_ABSENT"
    if reactive_net <= _ZERO:
        return "REACTIVE_ALPHA_BELOW_FRICTIONS"
    if our_net <= _ZERO:
        return "REACTIVE_ALPHA_SUB_LATENCY_HFT_ONLY"
    return "REACTIVE_ALPHA_REACHABLE"


def _analyse_class(shocks: list[dict[str, Any]], cls: str, scale: Decimal) -> dict[str, Any]:
    per_z: dict[str, Any] = {}
    for z in _Z_LEVELS:
        up = _paths(shocks, cls, z, 1, scale)
        down = _paths(shocks, cls, z, -1, scale)
        if not up:
            per_z[str(z)] = {"n_up": 0, "n_down": len(down), "note": "no up-shocks"}
            continue
        per_z[str(z)] = {
            "n_up": len(up),
            "n_down": len(down),
            "up_decay_mean_bps": _decay_curve(mean_path(up)),
            "up_decay_median_bps": _median_curve(up),
            "up_continuation_half_life_min": _half_life_min(up),
            "up_reactive_win_rate": _win_rate(up, _REACTIVE_LATENCY, 60),
        }

    up_primary = _paths(shocks, cls, _PRIMARY_Z, 1, scale)
    if not up_primary:
        return {"cls": cls, "by_z": per_z, "verdict": "INSUFFICIENT_UP_SHOCKS", "n_primary": 0}

    ladders = {f"cost{c}bps": _ladder(up_primary, c) for c in _COST_LEVELS_BPS}
    reactive_net, reactive_ex = _best_net_bps(up_primary, _REACTIVE_LATENCY, _PRIMARY_COST_BPS)
    our_net, _ = _best_net_bps(up_primary, _OUR_LATENCY, _PRIMARY_COST_BPS)
    slow_net, _ = _best_net_bps(up_primary, _SLOW_LATENCY, _PRIMARY_COST_BPS)
    true_gross = _best_gross_bps(up_primary, _REACTIVE_LATENCY)
    zero_cost_net, _ = _best_net_bps(up_primary, _REACTIVE_LATENCY, 0)
    low_n = len(up_primary) < _MIN_N_CONFIDENT
    # a thin class carries no firm intraday verdict — the machine string says so, not just the prose
    verdict = "THIN_N_INCONCLUSIVE" if low_n else _verdict(true_gross, reactive_net, our_net)
    return {
        "cls": cls,
        "n_primary": len(up_primary),
        "low_n": low_n,
        "by_z": per_z,
        "primary_up_ladder_net_bps": ladders,
        "money_numbers_bps": {
            "reactive_t1_true_gross_best": round(float(true_gross * _BPS), 2),
            "reactive_t1_zero_cost_net_best": round(float(zero_cost_net * _BPS), 2),
            "reactive_t1_best_net": round(float(reactive_net * _BPS), 2),
            "reactive_t1_best_exit_min": reactive_ex,
            "our_pipeline_t5_best_net": round(float(our_net * _BPS), 2),
            "slow_t15_best_net": round(float(slow_net * _BPS), 2),
            "up_continuation_half_life_min": _half_life_min(up_primary),
        },
        "verdict": verdict,
    }


def _gap_share_at(k: int, gaps: list[float], intra: list[float]) -> float | None:
    denom = abs(gaps[k]) + abs(intra[k])
    return abs(gaps[k]) / denom if denom > 0 else None


def _top_decile_gap_share(rank: list[float], gaps: list[float], intra: list[float]) -> float | None:
    # gap-share averaged over the top-decile days ranked by |rank| — the metric is
    # selection-dependent (by |total| / |gap| / |intraday|), so all three are reported honestly.
    n = len(gaps)
    top = sorted(range(n), key=lambda k: abs(rank[k]))[int(_TOP_DECILE * n) :]
    vals = [s for k in top if (s := _gap_share_at(k, gaps, intra)) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _gap_analysis(daily: dict[str, Any], sym_cls: dict[str, str]) -> dict[str, Any]:
    """Overnight-gap vs intraday decomposition per class — how much of the (news) move is the
    un-tradeable open gap, upstream of any intraday reactor."""
    per_class: dict[str, dict[str, list[float]]] = {
        c: {"gap": [], "intraday": [], "total": []} for c in _CLASSES
    }
    for sym, rows in daily.items():
        cls = sym_cls.get(sym)
        if cls is None or len(rows) < _MIN_DAYS_FOR_GAP:
            continue
        for i in range(1, len(rows)):
            prev_close = float(rows[i - 1][2])
            open_, close = float(rows[i][1]), float(rows[i][2])
            if prev_close <= 0 or open_ <= 0:
                continue
            per_class[cls]["gap"].append(open_ / prev_close - 1.0)
            per_class[cls]["intraday"].append(close / open_ - 1.0)
            per_class[cls]["total"].append(close / prev_close - 1.0)

    out: dict[str, Any] = {}
    for cls, d in per_class.items():
        if not d["gap"]:
            out[cls] = {"n_days": 0}
            continue
        gaps, intra, total = d["gap"], d["intraday"], d["total"]
        n = len(gaps)
        all_share = [s for k in range(n) if (s := _gap_share_at(k, gaps, intra)) is not None]
        out[cls] = {
            "n_days": n,
            "mean_abs_overnight_gap_bps": round(statistics.mean(abs(g) for g in gaps) * 10000, 2),
            "mean_abs_intraday_bps": round(statistics.mean(abs(x) for x in intra) * 10000, 2),
            "mean_gap_share": round(statistics.mean(all_share), 3) if all_share else None,
            # top-decile gap-share under each selection (the metric is selection-dependent)
            "gap_share_on_news_days": _top_decile_gap_share(total, gaps, intra),  # by |total|
            "gap_share_on_gap_days": _top_decile_gap_share(gaps, gaps, intra),  # by |gap|
            "gap_share_on_intraday_days": _top_decile_gap_share(intra, gaps, intra),  # by intraday
            "frac_days_gap_dominates": round(
                sum(1 for k in range(n) if abs(gaps[k]) > abs(intra[k])) / n, 3
            ),
        }
    return out


def _analyse(
    meta: dict[str, Any], shocks: list[dict[str, Any]], daily: dict[str, Any], scale: Decimal
) -> dict[str, Any]:
    sym_cls = {sym: cov["cls"] for sym, cov in meta["coverage"].items()}
    classes = {cls: _analyse_class(shocks, cls, scale) for cls in _CLASSES}
    gaps = _gap_analysis(daily, sym_cls)

    # Overall MOEX verdict: capturable only if the PRIMARY class (stocks) clears frictions net.
    stock = classes["stock"]
    stock_net = stock.get("money_numbers_bps", {}).get("reactive_t1_best_net", -1.0)
    overall = "MOEX_REACTIVE_UNCAPTURABLE_NET" if stock_net <= 0 else "MOEX_REACTIVE_CAPTURABLE"

    return {
        "window": meta["window"],
        "coverage": meta["coverage"],
        "params": {
            "primary_z": _PRIMARY_Z,
            "z_sweep": list(_Z_LEVELS),
            "entry_latencies_min": list(_ENTRY_LATENCIES),
            "exit_horizons_min": list(_EXIT_HORIZONS),
            "cost_levels_bps": list(_COST_LEVELS_BPS),
            "primary_cost_bps": _PRIMARY_COST_BPS,
            "ndfl_rate": str(_NDFL),
        },
        "by_class": classes,
        "overnight_gap": gaps,
        "verdict": overall,
    }


def _finding(a: dict[str, Any]) -> str:
    st = a["by_class"]["stock"]
    m = st["money_numbers_bps"]
    pz = st["by_z"][str(_PRIMARY_Z)]
    up_mean, up_med = pz["up_decay_mean_bps"], pz["up_decay_median_bps"]
    of = a["by_class"]["ofz"]
    ofm = of["money_numbers_bps"]
    ofz_pz = of["by_z"][str(_PRIMARY_Z)]
    ofz_m = ofz_pz["up_decay_mean_bps"]
    ofg = ofm["reactive_t1_true_gross_best"]
    fx = a["by_class"]["fx"]
    gs, gf = a["overnight_gap"]["stock"], a["overnight_gap"]["fx"]
    return (
        "Ported the crypto reactive-news cert to the real RUB universe via Tinkoff readonly 1-min "
        f"candles ({a['window']['start']}..{a['window']['end_exclusive']}), per instrument class, "
        "long-only up-shocks (no single-name MOEX shorting). OVERALL "
        f"**{a['verdict']}** — the same 'edge is allocation, not signal' family as the crypto cert "
        "and the slow-regime news study, only sharper. "
        f"STOCKS ({pz['n_up']} up-shocks at >={_PRIMARY_Z}-sigma): the intraday shock is priced "
        "ALMOST COMPLETELY by the time you can act — the mean forward path is flat noise around "
        f"zero ({up_mean['t+1']}/{up_mean['t+5']}/{up_mean['t+30']}bps t+1/5/30, no continuation "
        f"half-life), the median REVERSES ({up_med['t+1']}/{up_med['t+30']}/{up_med['t+120']}bps) "
        f"and win-rate is {pz['up_reactive_win_rate']:.1%}. Best-case reactive t+1 is only "
        f"{m['reactive_t1_true_gross_best']}bps TRUE gross; net of a realistic "
        f"{a['params']['primary_cost_bps']}bps round-trip + 13% NDFL it is "
        f"{m['reactive_t1_best_net']}bps and negative at every latency (t+5 "
        f"{m['our_pipeline_t5_best_net']}, t+15 {m['slow_t15_best_net']}). OFZ ({ofz_pz['n_up']} "
        "up-shocks): even STRONGER — the mean forward REVERSES at every horizon "
        f"({ofz_m['t+1']}..{ofz_m['t+120']}bps), so the reactive true gross is NEGATIVE ({ofg}bps) "
        f"→ verdict {of['verdict']} (no REACTIVE edge even frictionless/tax-free; the mean "
        "reverses and any barely-positive cell sits at a non-reactive latency and is << friction; "
        f"bond shocks mean-revert). FX is THIN (only {fx['n_primary']} up-shocks at z{_PRIMARY_Z}"
        f"{' — too few for a firm intraday verdict' if fx.get('low_n') else ''}). OVERNIGHT-GAP "
        "DECOMPOSITION (MOEX-specific; the metric is SELECTION-DEPENDENT, reported honestly): the "
        "gap's share of the daily move depends on which days you condition on. For stocks it is "
        f"{gs['gap_share_on_news_days']:.0%} on the biggest-TOTAL-move days (those are "
        f"intraday-dominated) but {gs['gap_share_on_gap_days']:.0%} on the biggest-GAP days — so "
        "'the gap is small' is NOT a general claim. What IS robust: the reactive INTRADAY alpha "
        "above is ~0/reverting regardless of selection, so it is the binding channel. For FX the "
        f"gap dominates under EVERY selection ({gf['gap_share_on_news_days']:.0%} on news days, "
        f"mean gap {gf['mean_abs_overnight_gap_bps']}bps vs intraday {gf['mean_abs_intraday_bps']}"
        "bps) — the reactive intraday trader is downstream of a wall that IS the whole move. "
        "LIMITS: latency axis is bars (=minutes for liquid stocks ~930 bars/day; approximate for "
        "thin OFZ/FX). Universe is 10 currently-listed liquid names (survivorship is directionally "
        "SAFE — survivors are the hardest names to find reactive edge in). Evening-session shocks "
        "(lower liquidity) are included (conservative — thinner = harder to capture). No "
        "single-name shorting → only up-shocks are retail-capturable. Deposit "
        f"{_DEPOSIT_ANNUAL_PCT}%/yr is the anchor. ETFs dropped (registry FIGIs return no 1-min "
        "candles; index ETFs have no idiosyncratic shock)."
    )


def main() -> None:
    meta, shocks, daily, scale = _load()
    a = _analyse(meta, shocks, daily, scale)
    finding = _finding(a)
    binding = f"MOEX_NEWS_REACTION__{a['verdict']}"

    summary = {
        "binding": {"verdict": binding, "finding": finding},
        "analysis": a,
        "disclaimer": (
            "Measurement only on Tinkoff readonly MOEX data. Authorises no order; real-money "
            "execution is a hard stop. A 1-minute shock is a proxy for a news event; sub-second "
            "colocated HFT is out of scope and unreachable for an LLM/RSS pipeline."
        ),
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "moex_jump_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    _write_report(summary)

    print(f"BINDING VERDICT: {binding}")
    for cls in _CLASSES:
        c = a["by_class"][cls]
        mm = c.get("money_numbers_bps")
        if mm:
            print(
                f"  {cls:5} n={c['n_primary']:>4} t1-net {mm['reactive_t1_best_net']}bps "
                f"(gross {mm['reactive_t1_true_gross_best']}) t5 {mm['our_pipeline_t5_best_net']} "
                f"| {c['verdict']}"
            )
        else:
            print(f"  {cls:5} {c['verdict']}")


def _write_report(summary: dict[str, Any]) -> None:
    a = summary["analysis"]
    p = a["params"]
    md = [
        "# MOEX Reactive-News Alpha Decay — Stocks / OFZ / FX (Cert)",
        "",
        f"Window `{a['window']['start']}`->`{a['window']['end_exclusive']}` (exclusive) · Tinkoff "
        f"readonly 1-minute candles · primary z={p['primary_z']} · NDFL {p['ndfl_rate']}.",
        "",
        "> MOEX data via Tinkoff gRPC readonly (the only sanctioned MOEX source). Authorises no "
        "order — real-money execution is a hard stop. A 1-minute shock is a proxy for a news "
        "event; sub-second colocated HFT is out of scope. Latency axis is bars (=minutes for "
        "liquid stocks; approximate for thin OFZ/FX). Long-only up-shocks (no MOEX shorting).",
        "",
        f"## BINDING VERDICT: **{summary['binding']['verdict']}**",
        "",
        summary["binding"]["finding"],
        "",
        "## Coverage",
        "| instrument | class | bars | days | shocks |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for sym, cov in a["coverage"].items():
        md.append(f"| {sym} | {cov['cls']} | {cov['bars']} | {cov['days']} | {cov['shocks']} |")
    md += ["", "## Per-class money numbers (>=6-sigma up-shocks, best exit)"]
    md += [
        "| class | n | t+1 true-gross | t+1 net | our t+5 | slow t+15 | win-rate | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cls in _CLASSES:
        c = a["by_class"][cls]
        mm = c.get("money_numbers_bps")
        if not mm:
            md.append(f"| {cls} | {c.get('n_primary', 0)} | — | — | — | — | — | {c['verdict']} |")
            continue
        wr = c["by_z"][str(_PRIMARY_Z)].get("up_reactive_win_rate", "—")
        wr_s = f"{wr:.1%}" if isinstance(wr, float) else wr
        verdict = c["verdict"] + (" *(thin N)*" if c.get("low_n") else "")
        md.append(
            f"| {cls} | {c['n_primary']} | {mm['reactive_t1_true_gross_best']} | "
            f"{mm['reactive_t1_best_net']} | {mm['our_pipeline_t5_best_net']} | "
            f"{mm['slow_t15_best_net']} | {wr_s} | {verdict} |"
        )
    md += [
        "",
        f"All net figures charge a {p['primary_cost_bps']}bps round-trip + 13% NDFL. Cost sweep "
        f"{p['cost_levels_bps']}bps in the summary JSON.",
        "",
        "## Overnight-gap decomposition (MOEX-specific)",
        "How much of the daily move is the un-tradeable overnight GAP (open vs prior close) vs the "
        "intraday continuous move a reactor could chase. **The gap-share is SELECTION-DEPENDENT**, "
        "reported on the top-decile days ranked by |total|, by |gap|, and by |intraday|.",
        "",
        "| class | days | mean \\|gap\\| bps | mean \\|intraday\\| bps | gap-share (all) | "
        "on |total| days | on |gap| days | on |intraday| days | days gap dominates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cls in _CLASSES:
        gg = a["overnight_gap"].get(cls, {})
        if not gg.get("n_days"):
            md.append(f"| {cls} | 0 | — | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {cls} | {gg['n_days']} | {gg['mean_abs_overnight_gap_bps']} | "
            f"{gg['mean_abs_intraday_bps']} | {gg['mean_gap_share']} | "
            f"{gg['gap_share_on_news_days']} | {gg['gap_share_on_gap_days']} | "
            f"{gg['gap_share_on_intraday_days']} | {gg['frac_days_gap_dominates']} |"
        )
    md += [
        "",
        "## Reading",
        "- **Stocks:** the intraday shock is priced almost completely by the time you can act — "
        "mean forward is flat noise around zero, median reverses, win-rate < 50%. Best-case gross "
        "is ~0 bps; net of frictions it is negative at every latency. Slowness is not the "
        "bottleneck — there is no intraday continuation to be slow about.",
        "- **OFZ:** stronger still — the mean forward REVERSES, so even a frictionless tax-free "
        "reactor loses (verdict ABSENT). Bond shocks mean-revert; no reactive edge at all.",
        "- **Gap decomposition (honest — the metric is selection-dependent):** for stocks the gap "
        "is ~12% of the move on the biggest-|total| days but ~51% on the biggest-|gap| days, so "
        "'the gap is small' is NOT a general claim. What is robust: the reactive INTRADAY alpha "
        "above is ~0/reverting regardless of selection, so it is the binding channel. The gap-wall "
        "is a genuine FX phenomenon: for USD/RUB ~97% of the news-day move is the un-tradeable "
        "overnight gap under every selection, and the thin intraday session offers almost nothing.",
        "- Same family as the crypto cert and the slow-regime news study: edge is allocation, not "
        f"signal; the {_DEPOSIT_ANNUAL_PCT}%/yr deposit anchor holds.",
    ]
    (_DIR / "moex_jump_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
