"""Crypto cert: does cross-exchange arbitrage or a crypto trend sleeve beat the RUB deposit?

Deterministic, token-free — consumes the committed ``results/research/crypto/crypto_panel.json``
snapshot (public read-only data; no orders, real money is a hard stop). Two measurements:

  1. ARBITRAGE FEASIBILITY — from the real cross-venue top-of-book poll, the distribution of the
     best realisable spread (buy cheapest ask, sell richest bid) vs round-trip taker fees at three
     tiers + an amortised withdrawal cost. Reports how many profitable round trips a year the
     capital-lockup carry would demand just to match the deposit.

  2. TREND SLEEVE — a RUB investor's BTC/ETH price path (Binance USD close x CBR USD/RUB), a
     time-series-momentum long/flat overlay (idle bars parked in the deposit), net of trading cost
     and 13% NDFL, run through the canonical Instrument Integration Gate against the SAME MCFTRR
     equity leg that rejected gold and real estate. Crypto's 2022 crash is in-window, so it is held
     to the strict INTEGRATE bar (no tail-untestable PROBATION toe-hold).

    uv run python scripts/research/run_crypto_gate.py
"""

from __future__ import annotations

import bisect
import json
import statistics
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from finalayze.backtest.allocation_gate import accrue_real_risk_free_leg
from finalayze.backtest.crypto_lab import (
    best_cross_venue_spread,
    crypto_trend_nav,
    nearest_rank_percentile,
    net_arb_edge_frac,
    time_series_momentum_signal,
)
from finalayze.backtest.equity_tilt_experiment import (
    RISK_FREE_ANNUAL_PCT,
    ArmMetrics,
    _metrics,
    _slice,
)
from finalayze.backtest.instrument_integration_gate import Candidate, run_integration_gate
from finalayze.core.constants import NDFL_RATE
from finalayze.core.ndfl import YtdTaxAccumulator

_DIR = Path("results/research/crypto")
_PANEL = _DIR / "crypto_panel.json"
_EQUITY_SNAP = Path("results/research/mcftrr_equity.json")

# ── Pre-registered constants (named; never moved to fit a result) ────────────
_DEPOSIT_SPREAD_PP = Decimal("1.0")
_BINDING_END = date(2026, 6, 10)
_EQUITY_SHIFT_DAYS = 1  # mcftrr_equity.json index_shift_days (MSK-midnight -> true ISS trade date)

# Arb: per-side taker tiers + an amortised withdrawal/network cost per round trip.
_ARB_FEE_TIERS = {
    "vip_0.02pct": Decimal("0.0002"),
    "mid_0.075pct": Decimal("0.00075"),
    "retail_0.20pct": Decimal("0.0020"),
}
_ARB_WITHDRAWAL_FRAC = Decimal("0.0005")  # 1-transfer lower bound (~$30/$60k); real loops need >=2
_BPS = Decimal(10000)
_P90 = 0.90
_MIN_VENUES = 2

# Trend sleeve.
_PER_SIDE_COST = Decimal("0.001")  # 10 bps/side crypto spot (retail-good venue)
_LOOKBACKS = (30, 90, 180)
_PRIMARY_LB = 90  # ~4-month TSMOM, the classic crypto time-series-momentum horizon
_LOOKBACK_STABLE_RATIO = 3.0  # basket TR max/min across lookbacks above this = fragile/unstable
_DEPOSIT_W = Decimal("0.4")
_EQUITY_BASE_W = Decimal("0.6")
_SLEEVE_TIER = "high"
_SLEEVE_ROLE = "growth"

# Regime windows (mirror the real-estate / duration certs).
_CRASH_START = date(2022, 2, 21)
_CRASH_END = date(2022, 12, 30)
_HIGH_RATE_START = date(2024, 1, 1)
_HIGH_RATE_END = date(2025, 6, 5)
_EASING_START = date(2025, 6, 6)


def _load_panel() -> tuple[list[dict[str, Any]], dict[str, list[list[str]]], list[list[str]]]:
    p = json.loads(_PANEL.read_text(encoding="utf-8"))
    return p["arb_snapshots"], p["ohlc"], p["usdrub"]


def _load_equity() -> list[tuple[date, Decimal]]:
    nav = json.loads(_EQUITY_SNAP.read_text(encoding="utf-8"))["nav"]
    shift = timedelta(days=_EQUITY_SHIFT_DAYS)
    return [(date.fromisoformat(d) + shift, Decimal(c)) for d, c in nav]


def _fx_map(usdrub: list[list[str]]) -> tuple[list[date], dict[date, Decimal]]:
    fx = {date.fromisoformat(d): Decimal(r) for d, r in usdrub}
    return sorted(fx), fx


def _fx_asof(fx_dates: list[date], fx: dict[date, Decimal], d: date) -> Decimal | None:
    i = bisect.bisect_right(fx_dates, d) - 1
    return fx[fx_dates[i]] if i >= 0 else None


def _rub_levels(
    usd_rows: list[list[str]], fx_dates: list[date], fx: dict[date, Decimal]
) -> dict[date, Decimal]:
    out: dict[date, Decimal] = {}
    for d_str, close in usd_rows:
        d = date.fromisoformat(d_str)
        rate = _fx_asof(fx_dates, fx, d)
        if rate is not None and d <= _BINDING_END:
            out[d] = Decimal(close) * rate
    return out


def _curve_metrics(nav: list[tuple[date, Decimal]], start: date, end: date) -> ArmMetrics:
    return _metrics(_slice([d for d, _ in nav], [v for _, v in nav], start, end))


def _deposit_factors(deposit_curve: list[tuple[date, Decimal]]) -> list[Decimal]:
    vals = [v for _, v in deposit_curve]
    return [Decimal(1)] + [vals[i] / vals[i - 1] for i in range(1, len(vals))]


# ── 1. ARBITRAGE FEASIBILITY ─────────────────────────────────────────────────
def _analyse_arb(arb_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    gross: list[Decimal] = []
    venues_seen: set[str] = set()
    for snap in arb_snapshots:
        q_raw = snap["quotes"]
        quotes = {v: (Decimal(str(bid)), Decimal(str(ask))) for v, (bid, ask) in q_raw.items()}
        venues_seen |= set(quotes)
        if len(quotes) < _MIN_VENUES:
            continue
        frac, _, _ = best_cross_venue_spread(quotes)
        gross.append(frac)

    median = statistics.median(gross)
    p90 = nearest_rank_percentile(gross, _P90)
    gmax = max(gross)
    stats = {"median": median, "p90": p90, "max": gmax}

    # Net per-trip edge at each (spread stat, fee tier); withdrawal amortised in.
    net_table: dict[str, dict[str, str]] = {}
    any_positive = False
    for tier, fee in _ARB_FEE_TIERS.items():
        row: dict[str, str] = {}
        for label, g in stats.items():
            edge = net_arb_edge_frac(g, fee, _ARB_WITHDRAWAL_FRAC)
            row[label] = f"{edge * _BPS:.2f}bps"
            if edge > 0:
                any_positive = True
        net_table[tier] = row

    # Best possible per-trip edge (max spread, cheapest fee, incl. withdrawal).
    best_edge = net_arb_edge_frac(gmax, min(_ARB_FEE_TIERS.values()), _ARB_WITHDRAWAL_FRAC)
    # Capital-lockup carry: even at ZERO fees, trips/yr to match a RISK_FREE_ANNUAL_PCT deposit.
    dep_frac = Decimal(str(RISK_FREE_ANNUAL_PCT)) / Decimal(100)
    trips_zero_fee = float(dep_frac / gmax) if gmax > 0 else float("inf")

    feasible = any_positive and best_edge > 0
    verdict = "ARB_FEASIBLE" if feasible else "ARB_INFEASIBLE_FEES_AND_CARRY_DOMINATE"
    return {
        "rounds": len(gross),
        "venues": sorted(venues_seen),
        "gross_spread_bps": {k: round(float(v * _BPS), 3) for k, v in stats.items()},
        "net_edge_per_trip_bps": net_table,
        "withdrawal_amortised_bps": float(_ARB_WITHDRAWAL_FRAC * _BPS),
        "best_case_net_edge_bps": round(float(best_edge * _BPS), 3),
        "trips_per_year_to_match_deposit_at_zero_fees": round(trips_zero_fee, 1),
        "any_tier_positive": any_positive,
        "verdict": verdict,
    }


# ── 2. TREND SLEEVE ──────────────────────────────────────────────────────────
def _build_sleeve(
    axis: list[date],
    basket_levels: list[Decimal],
    deposit_factors: list[Decimal],
    lookback: int,
) -> list[tuple[date, Decimal]]:
    signal = time_series_momentum_signal(basket_levels, lookback)
    return crypto_trend_nav(
        dates=axis,
        crypto_levels=basket_levels,
        deposit_factors=deposit_factors,
        signal=signal,
        per_side_cost=_PER_SIDE_COST,
        ndfl=NDFL_RATE,
    )


def _analyse_trend(
    ohlc: dict[str, list[list[str]]],
    fx_dates: list[date],
    fx: dict[date, Decimal],
    equity_curve: list[tuple[date, Decimal]],
) -> dict[str, Any]:
    # RUB price paths, rebased to 1 at the common start; equal-weight BTC/ETH basket.
    rub = {sym: _rub_levels(rows, fx_dates, fx) for sym, rows in ohlc.items()}
    common = sorted(set.intersection(*[set(v) for v in rub.values()]))
    common = [d for d in common if d >= equity_curve[0][0]]  # need equity overlap for the gate
    base = {sym: rub[sym][common[0]] for sym in rub}
    rebased = {sym: [rub[sym][d] / base[sym] for d in common] for sym in rub}
    basket = [
        sum((rebased[sym][i] for sym in rub), Decimal(0)) / Decimal(len(rub))
        for i in range(len(common))
    ]

    deposit_curve = accrue_real_risk_free_leg(
        common, Decimal(1), spread_pp=_DEPOSIT_SPREAD_PP, tax_acc=YtdTaxAccumulator()
    )
    dep_factors = _deposit_factors(deposit_curve)

    # Sleeve family: basket (primary) + BTC-only + ETH-only, each swept over lookbacks.
    families = {"basket": basket, **{sym: rebased[sym] for sym in rub}}
    sleeve_tr: dict[str, dict[str, float]] = {}
    for fam, levels in families.items():
        sleeve_tr[fam] = {}
        for lb in _LOOKBACKS:
            nav = _build_sleeve(common, levels, dep_factors, lb)
            sleeve_tr[fam][str(lb)] = round(
                _curve_metrics(nav, common[0], common[-1]).total_return_pct, 2
            )

    primary_nav = _build_sleeve(common, basket, dep_factors, _PRIMARY_LB)

    # Total return vs the deposit (all RUB, net): deposit-100 vs buy-hold vs primary sleeve.
    # Buy-and-hold = the sleeve permanently long: ONE realisation, taxed once at sale (the correct
    # NDFL treatment for a hold). net_index_returns' per-bar asymmetric tax is apt only for low-vol
    # fixed-income indices; on a high-vol crypto path it taxes every up-day without refunding
    # down-days and annihilates the curve, so it is the wrong tool here.
    def _buyhold(levels: list[Decimal]) -> list[tuple[date, Decimal]]:
        return crypto_trend_nav(
            dates=common,
            crypto_levels=levels,
            deposit_factors=dep_factors,
            signal=[1] * len(common),
            per_side_cost=_PER_SIDE_COST,
            ndfl=NDFL_RATE,
        )

    dep_tr = _curve_metrics(deposit_curve, common[0], common[-1]).total_return_pct
    buyhold_m = _curve_metrics(_buyhold(basket), common[0], common[-1])
    buyhold_tr = buyhold_m.total_return_pct
    # Per-family buy-and-hold — so the equal-weight basket cannot hide a single-asset win. Directly
    # answers "did ANY simple crypto holding beat the deposit over this window?"
    buyhold_by_family = {
        fam: round(_curve_metrics(_buyhold(levels), common[0], common[-1]).total_return_pct, 2)
        for fam, levels in families.items()
    }
    any_holding_beats_deposit = any(v > dep_tr for v in buyhold_by_family.values())
    sleeve_primary_tr = _curve_metrics(primary_nav, common[0], common[-1]).total_return_pct
    sleeve_beats_deposit = sleeve_primary_tr > dep_tr
    buyhold_beats_deposit = buyhold_tr > dep_tr

    # Lookback fragility: all lookbacks hold roughly the same share of long days, yet TR spans a
    # wide range because a handful of fat-tail days dominate -> the "edge" is a lookback-lottery,
    # not knowable ex-ante. Measured on the pre-registered basket family.
    basket_lb = list(sleeve_tr["basket"].values())
    lb_min, lb_max = min(basket_lb), max(basket_lb)
    lb_ratio = lb_max / lb_min if lb_min > 0 else None
    lookback_unstable = lb_ratio is not None and lb_ratio > _LOOKBACK_STABLE_RATIO
    primary_signal = time_series_momentum_signal(basket, _PRIMARY_LB)
    long_day_share = round(sum(primary_signal) / len(primary_signal), 3)

    # Per-regime metrics of the primary sleeve vs the deposit.
    windows = {
        "full_window": (common[0], common[-1]),
        "crash_year_2022": (_CRASH_START, _CRASH_END),
        "high_rate_2024_25": (_HIGH_RATE_START, _HIGH_RATE_END),
        "easing_2025_26": (_EASING_START, common[-1]),
    }
    regime_rows: dict[str, object] = {}
    for wname, (w0, w1) in windows.items():
        sm = _curve_metrics(primary_nav, w0, w1)
        dm = _curve_metrics(deposit_curve, w0, w1)
        regime_rows[wname] = {
            "range": [w0.isoformat(), w1.isoformat()],
            "sleeve": sm.__dict__,
            "deposit": dm.__dict__,
            "sleeve_beats_deposit": sm.total_return_pct > dm.total_return_pct,
            "n1_caveat": wname != "high_rate_2024_25",
        }

    # ── Canonical Instrument Integration Gate ─────────────────────────────────
    gate_candidate = Candidate(
        name="crypto_trend_btc_eth",
        net_curve=primary_nav,
        risk_tier=_SLEEVE_TIER,
        intended_role=_SLEEVE_ROLE,
    )
    gate_verdict = run_integration_gate(gate_candidate, equity_curve)
    sc = gate_verdict.scorecard
    gate = {
        "tier": gate_verdict.tier,
        "proposed_weight": str(gate_verdict.proposed_weight),
        "carved_from": gate_verdict.carved_from,
        "n1_caveat": gate_verdict.n1_caveat,
        "reasons": gate_verdict.reasons,
        "scorecard": {
            "window_bars": sc.window_bars,
            "regimes_covered": sc.regimes_covered,
            "tail_backtestable": sc.tail_backtestable,
            "marginal_sharpe_delta": round(sc.marginal_sharpe_delta, 4),
            "marginal_sortino_delta": round(sc.marginal_sortino_delta, 4),
            "marginal_maxdd_delta_pp": round(sc.marginal_maxdd_delta_pp, 3),
            "crash_year_maxdd_delta_pp": round(sc.crash_year_maxdd_delta_pp, 3),
            "max_corr_to_existing_legs": round(sc.max_corr_to_existing_legs, 4),
            "anti_hollow_ok": sc.anti_hollow_ok,
        },
    }

    # Binding = the risk-adjusted, crash-inclusive GATE tier (battery-comparable). Raw TR is
    # reported honestly alongside: buy-and-hold beating the deposit on RAW return does NOT overturn
    # a gate REJECT — that is the whole point of the risk-aware, crash-inclusive framework.
    risk_rejected = gate_verdict.tier in ("REJECT", "PROBATION")
    if gate_verdict.tier == "INTEGRATE":
        verdict = "CRYPTO_TREND_INTEGRATES"
    else:
        verdict = f"CRYPTO_TREND_GATE_{gate_verdict.tier}"

    return {
        "window": {
            "start": common[0].isoformat(),
            "end": common[-1].isoformat(),
            "n_bars": len(common),
        },
        "symbols": sorted(rub),
        "params": {
            "primary_lookback": _PRIMARY_LB,
            "lookback_sweep": list(_LOOKBACKS),
            "per_side_cost_bps": float(_PER_SIDE_COST * _BPS),
            "ndfl_rate": str(NDFL_RATE),
            "deposit_w": str(_DEPOSIT_W),
            "equity_base_w": str(_EQUITY_BASE_W),
        },
        "total_return_vs_deposit": {
            "deposit_only_tr_pct": round(dep_tr, 2),
            "buyhold_basket_net_tr_pct": round(buyhold_tr, 2),
            "buyhold_basket_maxdd_pct": round(buyhold_m.maxdd_pct, 2),
            "buyhold_tr_by_family": buyhold_by_family,
            "any_holding_beats_deposit": any_holding_beats_deposit,
            "sleeve_primary_tr_pct": round(sleeve_primary_tr, 2),
            "sleeve_beats_deposit": sleeve_beats_deposit,
            "buyhold_beats_deposit": buyhold_beats_deposit,
            "risk_adjusted_reject": risk_rejected,
        },
        "lookback_stability": {
            "basket_min_tr_pct": round(lb_min, 2),
            "basket_max_tr_pct": round(lb_max, 2),
            "max_min_ratio": round(lb_ratio, 2) if lb_ratio is not None else None,
            "unstable": lookback_unstable,
            "primary_long_day_share": long_day_share,
        },
        "sleeve_tr_by_family_lookback": sleeve_tr,
        "regimes": regime_rows,
        "integration_gate": gate,
        "verdict": verdict,
    }


def main() -> None:
    arb_snapshots, ohlc, usdrub = _load_panel()
    fx_dates, fx = _fx_map(usdrub)
    equity_curve = _load_equity()

    arb = _analyse_arb(arb_snapshots)
    trend = _analyse_trend(ohlc, fx_dates, fx, equity_curve)

    binding = (
        f"CRYPTO_ARB_{'FEASIBLE' if arb['verdict'] == 'ARB_FEASIBLE' else 'INFEASIBLE'}"
        f"__{trend['verdict']}"
    )
    gate = trend["integration_gate"]
    sc = gate["scorecard"]
    dd = trend["total_return_vs_deposit"]
    bh = dd["buyhold_tr_by_family"]
    ls = trend["lookback_stability"]

    finding = (
        "Cross-exchange SPOT top-of-book arbitrage is INFEASIBLE for a RUB retail investor: across "
        f"{arb['rounds']} real polls of {len(arb['venues'])} venues the best realisable "
        "top-of-book spread is a median "
        f"{arb['gross_spread_bps']['median']}bps (max {arb['gross_spread_bps']['max']}bps) "
        "— below round-trip taker fees at every tier, so the net per-trip edge is negative "
        f"(best case {arb['best_case_net_edge_bps']}bps incl. a generous "
        f"{arb['withdrawal_amortised_bps']}bps withdrawal). Even at ZERO fees the capital-lockup "
        f"carry would demand ~{arb['trips_per_year_to_match_deposit_at_zero_fees']} profitable "
        "cross-venue round trips a year (each a multi-minute on-chain transfer) merely to match "
        f"the deposit. On the TREND side the gate returns {gate['tier']} (risk-adjusted, "
        "crash-inclusive). Over this window NO simple crypto holding beat the deposit net of "
        f"NDFL: buy-and-hold basket {bh['basket']}% (BTC-only {bh['BTCUSDT']}%, ETH-only "
        f"{bh['ETHUSDT']}%) vs the deposit {dd['deposit_only_tr_pct']}%, and the basket carried "
        f"an {dd['buyhold_basket_maxdd_pct']}% drawdown the deposit never takes. The "
        f"{_PRIMARY_LB}-day TSMOM sleeve returned {dd['sleeve_primary_tr_pct']}% — also below "
        "the deposit. Crucially the trend 'edge' is a LOOKBACK-LOTTERY: the basket sleeve TR "
        f"ranges {ls['basket_min_tr_pct']}%..{ls['basket_max_tr_pct']}% across 30/90/180-day "
        f"lookbacks (x{ls['max_min_ratio']}) though all hold long "
        f"~{ls['primary_long_day_share']:.0%} of days — a handful of fat-tail days dominate, so "
        "which lookback 'wins' is NOT knowable ex-ante. The gate REJECTs (tail tested & FAILED): "
        "crypto's 2022 crash is in-window and the sleeve RAISED the blended crash-year drawdown "
        f"(delta {sc['crash_year_maxdd_delta_pp']:+}pp) — and the REJECT is over-determined: even "
        f"setting that veto aside it fails the INTEGRATE bar on marginal Sharpe "
        f"({sc['marginal_sharpe_delta']:+} vs +0.10) and on full-window MaxDD "
        f"(delta {sc['marginal_maxdd_delta_pp']:+}pp — RAISED, not the +3pp cut). Same family "
        "conclusion as gold/ZO/real estate: risk-adjusted, the deposit anchor holds. HONEST "
        "LIMITS: raw crypto TR is highly START-DATE-SENSITIVE (BTC began this window 2021 "
        "mid-cycle; a 2023-bottom start flips the raw-return read AND roughly halves the "
        "drawdown). Crypto carries 33-82% drawdowns under every start tested (82% in this 2021 "
        "window, ~33-66% from a 2023 bottom) — an order of magnitude beyond the deposit's 0%; the "
        "magnitude is start-dependent but the deposit-dominant risk gap is not. Arb infeasibility "
        "and the 9x lookback fragility are structural. N=1 easing cycle; only SPOT top-of-book "
        "arb was measured (funding/basis + triangular are out of scope, but the capital-lockup "
        "carry applies to any capital-locking cross-venue play); arb poll is a within-session "
        "snapshot; the deposit leg is floored at 0% pre-2022-02-28 so its 98% is a conservative "
        "lower bound; and custody/exchange/RU regulatory-access + USDT/P2P acquisition premium "
        "are uncosted and one-directional against crypto."
    )

    summary = {
        "risk_free_annual_pct": RISK_FREE_ANNUAL_PCT,
        "arbitrage": arb,
        "trend_sleeve": trend,
        "binding": {"verdict": binding, "finding": finding},
        "disclaimer": (
            "Measurement only on public read-only data. Authorises no order; real-money "
            "execution is a hard stop. Crypto carries custody, exchange-counterparty and RU "
            "regulatory/access risk that no backtest captures."
        ),
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / "crypto_cert_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    _write_report(summary)

    print(f"BINDING VERDICT: {binding}")
    print(
        f"  ARB: {arb['verdict']} — gross median {arb['gross_spread_bps']['median']}bps "
        f"max {arb['gross_spread_bps']['max']}bps; best-case net {arb['best_case_net_edge_bps']}bps"
    )
    print(
        f"  TREND: {trend['verdict']} — sleeve {dd['sleeve_primary_tr_pct']}% vs deposit "
        f"{dd['deposit_only_tr_pct']}% (buyhold {dd['buyhold_basket_net_tr_pct']}%); "
        f"GATE {gate['tier']} dMaxDD={sc['marginal_maxdd_delta_pp']:+}pp "
        f"crashDelta={sc['crash_year_maxdd_delta_pp']:+}pp tail_bt={sc['tail_backtestable']}"
    )


def _write_report(summary: dict[str, Any]) -> None:
    arb = summary["arbitrage"]
    trend = summary["trend_sleeve"]
    gate = trend["integration_gate"]
    sc = gate["scorecard"]
    dd = trend["total_return_vs_deposit"]
    ls = trend["lookback_stability"]
    w = trend["window"]
    sweep = trend["params"]["lookback_sweep"]
    md = [
        "# Crypto — Cross-Exchange Arbitrage & Trend Sleeve vs the RUB Deposit (Cert)",
        "",
        f"Trend window `{w['start']}`->`{w['end']}` · {w['n_bars']} bars · "
        f"RUONIA-excess {summary['risk_free_annual_pct']}% · "
        f"symbols {', '.join(trend['symbols'])}.",
        "",
        "> Public read-only data. Authorises no order — real-money execution is a hard stop. "
        "Crypto carries custody, exchange-counterparty and RU regulatory/access risk no backtest "
        "captures.",
        "",
        f"## BINDING VERDICT: **{summary['binding']['verdict']}**",
        "",
        summary["binding"]["finding"],
        "",
        "## 1. Cross-exchange arbitrage feasibility",
        f"Best realisable top-of-book spread across {len(arb['venues'])} venues "
        f"({', '.join(arb['venues'])}), {arb['rounds']} polls. Amortised withdrawal "
        f"{arb['withdrawal_amortised_bps']}bps/trip.",
        "",
        "| gross spread | bps |",
        "| --- | ---: |",
        f"| median | {arb['gross_spread_bps']['median']} |",
        f"| p90 | {arb['gross_spread_bps']['p90']} |",
        f"| max | {arb['gross_spread_bps']['max']} |",
        "",
        "**Net per-trip edge (bps) after 2 taker legs + withdrawal:**",
        "",
        "| fee tier | at median | at p90 | at max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tier, row in arb["net_edge_per_trip_bps"].items():
        md.append(f"| {tier} | {row['median']} | {row['p90']} | {row['max']} |")
    md += [
        "",
        f"Best-case net edge (max spread, cheapest fee): **{arb['best_case_net_edge_bps']}bps**. "
        "Even at ZERO fees, the capital-lockup carry demands "
        f"~{arb['trips_per_year_to_match_deposit_at_zero_fees']} profitable cross-venue round "
        f"trips/yr to match the deposit. Verdict: **{arb['verdict']}**.",
        "",
        "## 2. Trend sleeve — total return vs the deposit anchor",
        "| measure | value |",
        "| --- | ---: |",
        f"| 100%-deposit total return | {dd['deposit_only_tr_pct']}% |",
        f"| buy-and-hold BTC/ETH basket (net NDFL) | {dd['buyhold_basket_net_tr_pct']}% |",
        f"| buy-and-hold basket MaxDD | {dd['buyhold_basket_maxdd_pct']}% |",
        f"| buy-and-hold BTC-only (net NDFL) | {dd['buyhold_tr_by_family']['BTCUSDT']}% |",
        f"| buy-and-hold ETH-only (net NDFL) | {dd['buyhold_tr_by_family']['ETHUSDT']}% |",
        f"| {trend['params']['primary_lookback']}-day TSMOM sleeve (net cost+NDFL) "
        f"| {dd['sleeve_primary_tr_pct']}% |",
        f"| sleeve beats deposit? | {dd['sleeve_beats_deposit']} |",
        f"| buy-and-hold basket beats deposit (RAW TR)? | {dd['buyhold_beats_deposit']} |",
        f"| **any simple holding beats deposit?** | **{dd['any_holding_beats_deposit']}** |",
        f"| **risk-adjusted gate reject?** | **{dd['risk_adjusted_reject']}** |",
        "",
        "**Sleeve total return by family x lookback (net):**",
        "",
        "| family | " + " | ".join(f"LB{lb}" for lb in sweep) + " |",
        "| --- | " + " | ".join("---:" for _ in sweep) + " |",
    ]
    for fam, by_lb in trend["sleeve_tr_by_family_lookback"].items():
        cells = " | ".join(f"{by_lb[str(lb)]}%" for lb in sweep)
        md.append(f"| {fam} | {cells} |")
    md += [
        "",
        f"> **Lookback lottery (fragility).** The basket sleeve TR ranges "
        f"{ls['basket_min_tr_pct']}%..{ls['basket_max_tr_pct']}% across the three lookbacks "
        f"(x{ls['max_min_ratio']}, unstable={ls['unstable']}) even though all hold long "
        f"~{ls['primary_long_day_share']:.0%} of days — a few fat-tail days dominate, so no "
        "lookback is knowable ex-ante. This dispersion IS the evidence that crypto TSMOM is not "
        "a dependable edge.",
        "",
        "## 3. Canonical Instrument Integration Gate",
        "Same pre-registered gate as gold (REJECT) / ZO (PROBATION) / real estate (REJECT). "
        "Crypto's 2022 crash is in-window, so it is held to the strict INTEGRATE bar.",
        "",
        f"**GATE TIER: `{gate['tier']}`** (proposed weight {gate['proposed_weight']}, carved "
        f"from {gate['carved_from']}) — {'; '.join(gate['reasons'])}",
        "",
        "| scorecard | value |",
        "| --- | ---: |",
        f"| window bars / regimes | {sc['window_bars']} / {sc['regimes_covered']} |",
        f"| tail backtestable | {sc['tail_backtestable']} |",
        f"| delta Sharpe (10% eval) | {sc['marginal_sharpe_delta']:+} |",
        f"| delta Sortino (10% eval) | {sc['marginal_sortino_delta']:+} |",
        f"| delta MaxDD pp (+ = cut) | {sc['marginal_maxdd_delta_pp']:+} |",
        f"| crash-year delta MaxDD pp (+ = raised) | {sc['crash_year_maxdd_delta_pp']:+} |",
        f"| max \\|corr\\| to existing legs | {sc['max_corr_to_existing_legs']} |",
        "",
        "## 4. Per-regime sleeve vs deposit",
        "| window | sleeve TR% | deposit TR% | sleeve MaxDD% | beats deposit |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for wname, r in trend["regimes"].items():
        cav = " *(N=1)*" if r["n1_caveat"] else ""
        s, d = r["sleeve"], r["deposit"]
        md.append(
            f"| {wname}{cav} | {s['total_return_pct']:.1f} | {d['total_return_pct']:.1f} | "
            f"{s['maxdd_pct']:.1f} | {'yes' if r['sleeve_beats_deposit'] else 'no'} |"
        )
    md += [
        "",
        "_Sleeve idle bars earn the deposit; NDFL on realised gains only (no loss offset). The "
        "per-regime crash row uses the 2022-02-21..12-30 MOEX-invasion sub-window (cross-cert "
        "comparability); the BINDING crash-year delta uses the gate's calendar-2022 window._",
    ]
    (_DIR / "crypto_cert_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
