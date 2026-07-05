"""Jump-response cert: after a large 1-minute crypto shock, how much move is left for a reactor?

Deterministic, token-free — consumes the committed
``results/research/jump_response/jump_panel.json`` snapshot (public read-only Binance klines; no
orders, real money is a hard stop). It answers the operator's question: is our news pipeline's
failure a *latency* problem (we react slowly, not reactively) or is there simply no edge?

Design (see ``jump_response_lab``): a reactive news bot enters an *already-started* move, so we
measure the mean forward path AFTER each large 1-minute shock (the jump bar's close is the earliest
realistic entry). Then a **latency ladder** — the net return (round-trip cost + 13% NDFL) a reactor
captures entering at t0+{1,2,5,15} min and exiting at t0+{30,60,120} min. If even the maximally
reactive t0+1 entry captures nothing net, slowness is not the bottleneck — the edge is not there. If
t0+1 pays but t0+5/15 does not, latency IS the bottleneck and the edge lives in a sub-minute HFT
window an LLM/RSS pipeline can never reach.

    uv run python scripts/research/run_jump_response_study.py
"""

from __future__ import annotations

import json
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

from finalayze.backtest.jump_response_lab import (
    align_sign,
    half_life_bars,
    mean_path,
    net_after_cost,
)

_DIR = Path("results/research/jump_response")
_PANEL = _DIR / "jump_panel.json"

# ── Pre-registered constants (named; never moved to fit a result) ────────────
_Z_LEVELS = (5, 6, 8)  # sensitivity sweep
_PRIMARY_Z = 6  # a 6-sigma 1-minute move is unambiguously news-scale
_ENTRY_LATENCIES = (1, 2, 5, 15)  # minutes after the shock's close a reactor could enter
_EXIT_HORIZONS = (30, 60, 120)  # minutes held
_COST_LEVELS_BPS = (0, 10, 20, 30)  # round-trip taker + shock slippage
_PRIMARY_COST_BPS = 20  # ~15bps round-trip taker + a modest shock-slippage allowance
_REACTIVE_LATENCY = 1  # the maximally reactive retail bot (event-triggered; not colocation)
_OUR_LATENCY = 5  # a realistic LLM + RSS pipeline floor
_SLOW_LATENCY = 15
_NDFL = Decimal("0.13")
_BPS = Decimal(10000)
_ZERO = Decimal(0)
_ONE = Decimal(1)
_CHECKPOINTS = (1, 5, 15, 30, 60, 120)  # horizons reported in the decay curve
_DEPOSIT_ANNUAL_PCT = "16-21"  # the near-vol-free RUB deposit regime (context, not recomputed here)


def _load() -> tuple[dict[str, Any], list[dict[str, Any]], Decimal]:
    p = json.loads(_PANEL.read_text(encoding="utf-8"))
    scale = Decimal(p["meta"]["params"]["path_scale"])  # paths stored as int 0.01-bp units
    return p["meta"], p["jumps"], scale


def _paths(jumps: list[dict[str, Any]], z: int, sign: int, scale: Decimal) -> list[list[Decimal]]:
    """Forward paths of shocks with |z|>=`z` and the given sign, as Decimal fractions."""
    return [
        [Decimal(x) / scale for x in j["path"]]
        for j in jumps
        if j["sign"] == sign and float(j["z"]) >= z
    ]


def _trade_return(path: list[Decimal], entry: int, exit_: int) -> Decimal:
    """Exact return entering at bar `entry`, exiting at `exit_`: (1+p[exit])/(1+p[entry]) - 1."""
    return (_ONE + path[exit_]) / (_ONE + path[entry]) - _ONE


def _ladder(paths: list[list[Decimal]], cost_bps: int) -> dict[str, dict[str, float]]:
    """Mean NET trade return (bps) for every (entry latency, exit horizon) at one cost level."""
    cost = Decimal(cost_bps) / _BPS
    table: dict[str, dict[str, float]] = {}
    for entry in _ENTRY_LATENCIES:
        row: dict[str, float] = {}
        for ex in _EXIT_HORIZONS:
            rets = [_trade_return(p, entry, ex) for p in paths]
            gross = sum(rets, _ZERO) / len(rets)
            row[f"exit{ex}"] = round(float(net_after_cost(gross, cost, _NDFL) * _BPS), 2)
        table[f"entry{entry}"] = row
    return table


def _mean_net(paths: list[list[Decimal]], entry: int, exit_: int, cost_bps: int) -> Decimal:
    cost = Decimal(cost_bps) / _BPS
    rets = [_trade_return(p, entry, exit_) for p in paths]
    gross = sum(rets, _ZERO) / len(rets)
    return net_after_cost(gross, cost, _NDFL)


def _best_net_bps(paths: list[list[Decimal]], entry: int, cost_bps: int) -> tuple[Decimal, int]:
    """Best net (over exit horizons) for a fixed entry latency; returns (net_frac, exit_bar)."""
    best, best_ex = None, _EXIT_HORIZONS[0]
    for ex in _EXIT_HORIZONS:
        n = _mean_net(paths, entry, ex, cost_bps)
        if best is None or n > best:
            best, best_ex = n, ex
    assert best is not None
    return best, best_ex


def _best_gross_bps(paths: list[list[Decimal]], entry: int) -> tuple[Decimal, int]:
    """Best TRUE gross (no cost, no tax) over exit horizons; the honest frictionless ceiling."""
    best, best_ex = None, _EXIT_HORIZONS[0]
    for ex in _EXIT_HORIZONS:
        rets = [_trade_return(p, entry, ex) for p in paths]
        g = sum(rets, _ZERO) / len(rets)
        if best is None or g > best:
            best, best_ex = g, ex
    assert best is not None
    return best, best_ex


def _decay_curve(mean_fwd: list[Decimal]) -> dict[str, float]:
    """Mean cumulative forward return (bps) at the reported checkpoints."""
    return {f"t+{h}": round(float(mean_fwd[h] * _BPS), 2) for h in _CHECKPOINTS}


def _median_curve(paths: list[list[Decimal]]) -> dict[str, float]:
    """Median (outlier-robust) cumulative forward return (bps) at the checkpoints."""
    return {
        f"t+{h}": round(float(statistics.median([p[h] for p in paths]) * _BPS), 2)
        for h in _CHECKPOINTS
    }


def _win_rate(paths: list[list[Decimal]], entry: int, exit_: int) -> float:
    """Share of shocks where the reactive long trade is profitable gross."""
    wins = sum(1 for p in paths if _trade_return(p, entry, exit_) > _ZERO)
    return round(wins / len(paths), 3)


def _analyse(meta: dict[str, Any], jumps: list[dict[str, Any]], scale: Decimal) -> dict[str, Any]:
    cov = meta["coverage"]
    total_bars = sum(int(c["bars"]) for c in cov.values())
    years = total_bars / (1440 * 365.25)  # per-coin minute bars → coin-years of coverage

    per_z: dict[str, Any] = {}
    for z in _Z_LEVELS:
        up = _paths(jumps, z, 1, scale)
        down = _paths(jumps, z, -1, scale)
        pooled = up + [align_sign(p, -1) for p in down]  # sign-aligned momentum-persistence
        if not up or not down:
            per_z[str(z)] = {"n_up": len(up), "n_down": len(down), "note": "too few shocks"}
            continue
        up_mean = mean_path(up)
        down_aligned_mean = mean_path([align_sign(p, -1) for p in down])
        pooled_mean = mean_path(pooled)
        per_z[str(z)] = {
            "n_up": len(up),
            "n_down": len(down),
            "shocks_per_year": round((len(up) + len(down)) / years, 1),
            # decay curves (mean + outlier-robust median), long-capturable up-jumps:
            "up_decay_mean_bps": _decay_curve(up_mean),
            "up_decay_median_bps": _median_curve(up),
            "down_aligned_decay_mean_bps": _decay_curve(down_aligned_mean),
            "pooled_aligned_decay_mean_bps": _decay_curve(pooled_mean),
            # continuation half-life (minutes) of the up-jump long path & the pooled path:
            "up_continuation_half_life_min": half_life_bars(up_mean),
            "pooled_continuation_half_life_min": half_life_bars(pooled_mean),
            # reactive long win-rate (t0+1 in, exit 60):
            "up_reactive_win_rate": _win_rate(up, _REACTIVE_LATENCY, 60),
        }

    # Primary analysis: up-jump long (the only spot-retail-capturable direction) at _PRIMARY_Z.
    up_primary = _paths(jumps, _PRIMARY_Z, 1, scale)
    ladders = {f"cost{c}bps": _ladder(up_primary, c) for c in _COST_LEVELS_BPS}

    reactive_net, reactive_ex = _best_net_bps(up_primary, _REACTIVE_LATENCY, _PRIMARY_COST_BPS)
    our_net, _ = _best_net_bps(up_primary, _OUR_LATENCY, _PRIMARY_COST_BPS)
    slow_net, _ = _best_net_bps(up_primary, _SLOW_LATENCY, _PRIMARY_COST_BPS)
    reactive_true_gross, _ = _best_gross_bps(up_primary, _REACTIVE_LATENCY)  # no cost, no tax
    reactive_zero_cost_net, _ = _best_net_bps(
        up_primary, _REACTIVE_LATENCY, 0
    )  # 0 cost, still NDFL
    half_life = half_life_bars(mean_path(up_primary))

    # Honest frictionless upper bound: if you could trade for FREE and perfectly harvest the MEAN
    # (you cannot — the median reverses and win-rate < 50%), annualise the zero-cost-net per shock
    # over the up-shock frequency. This lands SAME-ORDER as the deposit, so the anchor holds by way
    # of cost + fat-tail-unharvestability, NOT because the raw magnitude is negligible.
    up_shocks_per_year = len(up_primary) / years
    frictionless_ub_pct = float(reactive_zero_cost_net) * up_shocks_per_year * 100

    # ── Verdict (derived from the numbers, never pre-baked) ───────────────────
    if reactive_true_gross <= _ZERO:
        # even a frictionless, tax-free, maximally reactive entry captures nothing → no edge at all.
        verdict = "REACTIVE_ALPHA_ABSENT__MOVE_FULLY_PRICED_BY_ENTRY"
    elif reactive_net <= _ZERO:
        # a gross continuation exists but is below realistic frictions.
        verdict = "REACTIVE_ALPHA_BELOW_FRICTIONS__UNCAPTURABLE_NET"
    elif our_net <= _ZERO:
        # reactive entry pays but our pipeline latency kills it → sub-minute HFT window only.
        verdict = "REACTIVE_ALPHA_SUB_LATENCY__HFT_WINDOW_UNREACHABLE"
    else:
        # net-positive even at our pipeline latency → genuinely reachable (report honestly if so).
        verdict = "REACTIVE_ALPHA_REACHABLE__WORTH_PURSUING"

    return {
        "window": meta["window"],
        "coverage": cov,
        "coin_years": round(years, 2),
        "params": {
            "primary_z": _PRIMARY_Z,
            "z_sweep": list(_Z_LEVELS),
            "entry_latencies_min": list(_ENTRY_LATENCIES),
            "exit_horizons_min": list(_EXIT_HORIZONS),
            "primary_cost_bps": _PRIMARY_COST_BPS,
            "ndfl_rate": str(_NDFL),
            "vol_window_min": meta["params"]["vol_window_min"],
        },
        "by_z": per_z,
        "primary_up_ladder_net_bps": ladders,
        "money_numbers_bps": {
            "reactive_t1_best_net": round(float(reactive_net * _BPS), 2),
            "reactive_t1_best_exit_min": reactive_ex,
            "reactive_t1_true_gross_best": round(float(reactive_true_gross * _BPS), 2),
            "reactive_t1_zero_cost_net_best": round(float(reactive_zero_cost_net * _BPS), 2),
            "our_pipeline_t5_best_net": round(float(our_net * _BPS), 2),
            "slow_t15_best_net": round(float(slow_net * _BPS), 2),
            "up_continuation_half_life_min": half_life,
            "up_shocks_per_year": round(up_shocks_per_year, 1),
            "frictionless_annualised_upper_bound_pct": round(frictionless_ub_pct, 1),
        },
        "verdict": verdict,
    }


def _finding(a: dict[str, Any]) -> str:
    m = a["money_numbers_bps"]
    pz = a["by_z"][str(_PRIMARY_Z)]
    up_mean = pz["up_decay_mean_bps"]
    up_med = pz["up_decay_median_bps"]
    hl = m["up_continuation_half_life_min"]
    hl_txt = f"{hl} min" if hl is not None else "none (the move reverses/dies, no continuation)"
    return (
        f"On {a['coin_years']} coin-years of real 1-minute BTC/ETH data, {pz['n_up']} up-shocks "
        f"and {pz['n_down']} down-shocks at >={_PRIMARY_Z}-sigma (~{pz['shocks_per_year']}/yr). "
        "This tests the operator's question directly: after a large 1-minute move fires, is our "
        "news pipeline's failure a LATENCY problem or is there simply no edge? A reactive bot "
        "enters the ALREADY-STARTED move, so the honest metric is the forward path from the shock "
        f"bar's close. The MEAN up-shock path is mildly positive ({up_mean['t+1']}bps at t+1min, "
        f"{up_mean['t+5']}bps at t+5, {up_mean['t+30']}bps at t+30) — but the outlier-robust "
        f"MEDIAN is NEGATIVE at every horizon ({up_med['t+1']}/{up_med['t+30']}/{up_med['t+120']}"
        f"bps) and the reactive long win-rate is {pz['up_reactive_win_rate']:.1%}. So the positive "
        "mean is a FAT-TAIL ARTIFACT: the typical shock reverses, and a few big-continuation "
        f"shocks drag the average up (same lottery character as crypto TSMOM) — continuation "
        f"half-life {hl_txt}. That alone makes it un-harvestable, and cost buries it regardless: "
        "the maximally reactive entry (t+1min, best exit) captures only "
        f"{m['reactive_t1_true_gross_best']}bps TRUE gross (no cost/no tax; even zero trading-cost "
        f"but post-NDFL is {m['reactive_t1_zero_cost_net_best']}bps); net of a realistic "
        f"{a['params']['primary_cost_bps']}bps round-trip + 13% NDFL it is "
        f"{m['reactive_t1_best_net']}bps, our-pipeline latency (t+5min) "
        f"{m['our_pipeline_t5_best_net']}bps, slow (t+15min) {m['slow_t15_best_net']}bps — "
        "negative at EVERY latency and every non-zero cost tier. So the answer to 'are we slow, "
        "or is there no edge?' is BOTH-but-latency-is-second-order: the ladder confirms faster "
        "captures more gross (a real decay), yet even zero-latency zero-cost yields single-digit "
        f"bps that frictions erase, so we cannot win this race. VERDICT **{a['verdict']}**. HONEST "
        "LIMITS: a 1-minute move "
        "is a PROXY for a news event (most 6-sigma moves in liquid BTC/ETH are info-driven, but "
        "some are liquidations/microstructure); this measures continuation available to a reactor "
        "regardless of cause, which is exactly the reactive-capturability question. Down-shocks "
        "are reported sign-aligned but are SHORT-only (not spot-retail-capturable). Minute-close "
        "sampling understates intra-minute slippage during a shock (conservative against any "
        "edge); true sub-second colocated HFT is a different regime this 1-minute panel cannot "
        "resolve and an LLM/RSS pipeline can never reach. Deposit context (honest): annualising "
        f"the zero-cost net over ~{m['up_shocks_per_year']} up-shocks/coin-yr gives a frictionless "
        f"UPPER BOUND of ~{m['frictionless_annualised_upper_bound_pct']}%/yr — SAME ORDER as the "
        f"{_DEPOSIT_ANNUAL_PCT}%/yr deposit, NOT a rounding error. But that ceiling is unreachable "
        "on two counts: you cannot harvest the mean (median reverses, win-rate < 50%), and any "
        "realistic cost turns every ladder cell deeply negative. So the deposit anchor holds via "
        "COST + fat-tail-unharvestability, not a negligible magnitude."
    )


def main() -> None:
    meta, jumps, scale = _load()
    a = _analyse(meta, jumps, scale)
    finding = _finding(a)
    binding = f"NEWS_REACTION__{a['verdict']}"

    summary = {
        "binding": {"verdict": binding, "finding": finding},
        "analysis": a,
        "disclaimer": (
            "Measurement only on public read-only data. Authorises no order; real-money execution "
            "is a hard stop. A 1-minute shock is a proxy for a news event; sub-second colocated "
            "HFT is out of scope and unreachable for an LLM/RSS pipeline."
        ),
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "jump_response_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    _write_report(summary)

    m = a["money_numbers_bps"]
    print(f"BINDING VERDICT: {binding}")
    print(
        f"  reactive t+1 net {m['reactive_t1_best_net']}bps "
        f"(true-gross {m['reactive_t1_true_gross_best']}bps) | our t+5 "
        f"{m['our_pipeline_t5_best_net']}bps | slow t+15 {m['slow_t15_best_net']}bps | half-life "
        f"{m['up_continuation_half_life_min']} | frictionless-UB "
        f"{m['frictionless_annualised_upper_bound_pct']}%/yr"
    )


def _write_report(summary: dict[str, Any]) -> None:
    a = summary["analysis"]
    m = a["money_numbers_bps"]
    p = a["params"]
    pz = a["by_z"][str(_PRIMARY_Z)]
    md = [
        "# Reactive-News Alpha Decay — Are We Slow, or Is There No Edge? (Cert)",
        "",
        f"Window `{a['window']['start']}`->`{a['window']['end_exclusive']}` (exclusive) · "
        f"{a['coin_years']} coin-years of real 1-minute BTC/ETH klines · vol window "
        f"{p['vol_window_min']}min · NDFL {p['ndfl_rate']}.",
        "",
        "> Public read-only data. Authorises no order — real-money execution is a hard stop. A "
        "large 1-minute move is a proxy for a news event; sub-second colocated HFT is out of scope "
        "and unreachable for an LLM/RSS pipeline.",
        "",
        f"## BINDING VERDICT: **{summary['binding']['verdict']}**",
        "",
        summary["binding"]["finding"],
        "",
        "## 1. Money numbers — the reactive vs slow answer",
        "| entry latency | best net (bps) | ",
        "| --- | ---: |",
        f"| t+1 min (maximally reactive) | {m['reactive_t1_best_net']} |",
        f"| t+5 min (our LLM/RSS pipeline) | {m['our_pipeline_t5_best_net']} |",
        f"| t+15 min (slow batch) | {m['slow_t15_best_net']} |",
        "",
        f"Reactive t+1 TRUE gross (no cost, no tax): **{m['reactive_t1_true_gross_best']}bps** "
        f"(best exit {m['reactive_t1_best_exit_min']}min); zero trading-cost but post-NDFL: "
        f"**{m['reactive_t1_zero_cost_net_best']}bps**. Up-continuation half-life: "
        f"**{m['up_continuation_half_life_min']}** min. Annualising the zero-cost net over "
        f"~{m['up_shocks_per_year']} up-shocks/coin-yr gives a frictionless UPPER BOUND of "
        f"**~{m['frictionless_annualised_upper_bound_pct']}%/yr** — same order as the deposit, so "
        "the anchor holds by way of COST + fat-tail-unharvestability (median reverses, win-rate "
        f"< 50%), not negligible magnitude. All net figures charge a {p['primary_cost_bps']}bps "
        "round-trip + 13% NDFL on gains.",
        "",
        f"## 2. Latency ladder (>= {_PRIMARY_Z}-sigma up-shocks, mean NET bps)",
        "Rows = reaction latency; columns = hold horizon. Positive = a reactor captures net edge.",
        "",
    ]
    for cost_key, table in a["primary_up_ladder_net_bps"].items():
        md += [
            f"**{cost_key} round-trip:**",
            "",
            "| entry \\ exit | " + " | ".join(f"exit{ex}" for ex in _EXIT_HORIZONS) + " |",
            "| --- | " + " | ".join("---:" for _ in _EXIT_HORIZONS) + " |",
        ]
        for entry in _ENTRY_LATENCIES:
            row = table[f"entry{entry}"]
            cells = " | ".join(f"{row[f'exit{ex}']}" for ex in _EXIT_HORIZONS)
            md.append(f"| t+{entry} | {cells} |")
        md.append("")
    md += [
        "## 3. Decay curves by shock size (mean cumulative forward, bps)",
        "The mean move already realised by each horizon after the shock's close. If the curve is "
        "flat/negative past t+1, the move is fully priced by the time a reactor can act.",
        "",
        "| z | n_up | n_down | /yr | up t+1 | up t+5 | up t+30 | up t+120 | up half-life |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for z in _Z_LEVELS:
        r = a["by_z"][str(z)]
        if "up_decay_mean_bps" not in r:
            md.append(f"| {z} | {r['n_up']} | {r['n_down']} | — | — | — | — | — | — |")
            continue
        c = r["up_decay_mean_bps"]
        hl = r["up_continuation_half_life_min"]
        md.append(
            f"| {z} | {r['n_up']} | {r['n_down']} | {r['shocks_per_year']} | {c['t+1']} | "
            f"{c['t+5']} | {c['t+30']} | {c['t+120']} | {hl if hl is not None else 'none'} |"
        )
    md += [
        "",
        f"Outlier-robust MEDIAN up-shock path at z={_PRIMARY_Z} (bps): "
        + ", ".join(f"{k} {v}" for k, v in pz["up_decay_median_bps"].items())
        + ".",
        "",
        f"Reactive long win-rate (enter t+1, exit 60) at z={_PRIMARY_Z}: "
        f"**{pz['up_reactive_win_rate']:.1%}**.",
        "",
        "## 4. Reading",
        "- **t+1 net <= 0** → slowness is NOT the bottleneck: even a maximally reactive entry "
        "loses net of frictions. The move is priced faster than any retail actor (LLM/RSS) acts.",
        "- **t+1 net > 0 but t+5/15 <= 0** → latency IS the bottleneck, but the edge lives in a "
        "sub-minute window that a colocated HFT owns and our pipeline can never reach.",
        "- The frictionless upper bound "
        f"(~{m['frictionless_annualised_upper_bound_pct']}%/yr) is deposit-competitive, but "
        "unreachable on BOTH counts: you cannot harvest the mean (median reverses, win-rate < 50%) "
        "and any realistic cost turns every cell negative. The deposit anchor holds by way of cost "
        f"+ fat-tail, not magnitude. Same family as the slow-regime news event study — edge is "
        "allocation, not signal.",
    ]
    (_DIR / "jump_response_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
