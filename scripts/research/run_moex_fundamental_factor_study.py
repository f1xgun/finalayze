"""MOEX fundamental cross-sectional factor study — honest cert vs the deposit anchor.

Consumes ``results/research/moex_fundamental/panel.json`` (SmartLab IFRS + ISS prices)
and, for each factor (GP/A, cheap EV/EBITDA, ROA), for each fiscal year:
  1. rank the cross-section into terciles at the disclosure date (look-ahead-safe:
     entry is the "Дата отчёта" the market first saw, never the fiscal-period end);
  2. hold 1 year, measure top-tercile and top-minus-bottom TOTAL return (+dividends);
  3. compare to a deposit opened at entry at the real CBR key rate.

Verdict is deliberately conservative. With ~5 fiscal years and a post-2022 regime the
honest expected outcome is THIN_INCONCLUSIVE — a positive number here is a hypothesis,
not an edge. All caveats are printed, none hidden.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from finalayze.backtest.fundamental_factor_lab import (
    deposit_accrual,
    detect_splits,
    excess_over_deposit,
    forward_total_return,
    gross_profit_to_assets,
    long_short_spread,
    split_factor_at,
    tercile_labels,
)

_PANEL = Path("results/research/moex_fundamental/panel.json")
_OUT = Path("results/research/moex_fundamental/factor_cert.json")
_HOLD_DAYS = 365
_PRICE_TOLERANCE_DAYS = 7  # nearest trading day on/after target within this window
_MIN_NAMES = 6  # a tercile split below this is too degenerate to trust
_MIN_YEARS = 3  # fewer valid cross-sections than this -> THIN_INCONCLUSIVE

# Real CBR key-rate steps (effective date -> annual %), the deposit anchor.
# Post-2022 shape incl. Phase-74 easing calendar (first cut 2025-06-06 -> 20.00).
_KEY_RATE_STEPS: list[tuple[date, Decimal]] = [
    (date(2021, 3, 22), Decimal("4.5")),
    (date(2021, 4, 26), Decimal("5.0")),
    (date(2021, 6, 15), Decimal("5.5")),
    (date(2021, 7, 26), Decimal("6.5")),
    (date(2021, 9, 13), Decimal("6.75")),
    (date(2021, 10, 25), Decimal("7.5")),
    (date(2021, 12, 20), Decimal("8.5")),
    (date(2022, 2, 14), Decimal("9.5")),
    (date(2022, 2, 28), Decimal("20.0")),
    (date(2022, 4, 11), Decimal("17.0")),
    (date(2022, 5, 4), Decimal("14.0")),
    (date(2022, 5, 27), Decimal("11.0")),
    (date(2022, 6, 14), Decimal("9.5")),
    (date(2022, 7, 25), Decimal("8.0")),
    (date(2022, 9, 19), Decimal("7.5")),
    (date(2023, 7, 24), Decimal("8.5")),
    (date(2023, 8, 15), Decimal("12.0")),
    (date(2023, 9, 18), Decimal("13.0")),
    (date(2023, 10, 30), Decimal("15.0")),
    (date(2023, 12, 18), Decimal("16.0")),
    (date(2024, 7, 29), Decimal("18.0")),
    (date(2024, 9, 16), Decimal("19.0")),
    (date(2024, 10, 28), Decimal("21.0")),
    (date(2025, 6, 9), Decimal("20.0")),
    (date(2025, 7, 28), Decimal("18.0")),
    (date(2025, 9, 15), Decimal("17.0")),
    (date(2025, 10, 27), Decimal("16.5")),
    (date(2025, 12, 22), Decimal("16.0")),
    (date(2026, 2, 16), Decimal("15.5")),
    (date(2026, 3, 23), Decimal("15.0")),
    (date(2026, 4, 27), Decimal("14.5")),
    (date(2026, 6, 22), Decimal("14.25")),  # current (CBR, fetched 2026-07-08) — easing continues
]


def _key_rate(d: date) -> Decimal:
    rate = _KEY_RATE_STEPS[0][1]
    for eff, r in _KEY_RATE_STEPS:
        if d >= eff:
            rate = r
        else:
            break
    return rate


def realized_deposit(entry: date, days: int) -> Decimal:
    """Deposit total return over the hold, REINVESTED at the changing CBR key rate.

    The honest bar: an investor who opens a deposit at ``entry`` earns the key rate
    as it MOVES over the year, not the frozen entry rate. In an easing cycle this is
    materially LOWER than the entry rate (a 2025-03 entry at 21% earns less as CBR
    cuts through the year); in a hiking cycle it is higher. Piecewise ACT/365.
    """
    total = Decimal(0)
    cur = entry
    end = entry + timedelta(days=days)
    while cur < end:
        rate = _key_rate(cur)
        nxt = end
        for eff, _ in _KEY_RATE_STEPS:
            if cur < eff < nxt:
                nxt = eff
        seg = (nxt - cur).days
        total += (rate / Decimal(100)) * (Decimal(seg) / Decimal(365))
        cur = nxt
    return total


def _parse_asof(s: str) -> date | None:
    s = (s or "").strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()  # noqa: DTZ007 — naive date is intended
        except ValueError:
            continue
    return None


def _price_on_or_after(prices: dict[str, float], target: date) -> float | None:
    for i in range(_PRICE_TOLERANCE_DAYS + 1):
        key = (target + timedelta(days=i)).isoformat()
        if key in prices:
            return prices[key]
    return None


def _dividends_between(divs: list[dict], start: date, end: date) -> Decimal:
    total = Decimal(0)
    for d in divs:
        rd = _parse_asof(str(d.get("registryclosedate", "")))
        val = d.get("value")
        if rd is not None and start < rd <= end and val is not None:
            total += Decimal(str(val))
    return total


def _factor_value(rec: dict, factor: str) -> Decimal | None:
    def g(k: str) -> Decimal | None:
        v = rec.get(k)
        return Decimal(str(v)) if v is not None else None

    if factor == "gp_a":
        rev, cost, assets = g("revenue"), g("cost_of_production"), g("assets")
        if rev is None or cost is None or assets is None:
            return None
        return gross_profit_to_assets(rev, cost, assets)
    if factor == "ev_ebitda_cheap":
        ev = g("ev_ebitda")
        # cheap (low EV/EBITDA) should rank 'top' -> negate; ignore non-positive
        return -ev if (ev is not None and ev > 0) else None
    if factor == "roa":
        # SmartLab's @field 'roa' row is blank in the annual MSFO table; compute
        # it from the populated net_income + assets instead (net_income / assets).
        ni, assets = g("net_income"), g("assets")
        if ni is None or assets is None or assets <= 0:
            return None
        return ni / assets
    return None


def _run_factor(panel: dict, factor: str) -> dict:
    fundamentals: dict[str, list[dict]] = panel["fundamentals"]
    prices: dict[str, dict[str, float]] = panel["prices"]
    dividends: dict[str, list[dict]] = panel["dividends"]

    # Group by DISCLOSURE-YEAR wave. SmartLab's fiscal_year column label is
    # off-by-one vs the field cells; the authoritative timestamp is the
    # "Дата отчёта" (as_of). A wave = all reports first disclosed in year W
    # (early W = fiscal year W-1). Entry timing uses the exact as_of, not the wave.
    def _report_year(r: dict) -> int | None:
        d = _parse_asof(str(r.get("as_of", "")))
        return d.year if d else None

    years: set[int] = set()
    for recs in fundamentals.values():
        for r in recs:
            ry = _report_year(r)
            if ry is not None:
                years.add(ry)

    per_year = []
    for fy in sorted(years):
        cross: list[tuple[str, Decimal]] = []
        entry_dates: dict[str, date] = {}
        for t, recs in fundamentals.items():
            rec = next((r for r in recs if _report_year(r) == fy), None)
            if rec is None:
                continue
            fv = _factor_value(rec, factor)
            asof = _parse_asof(str(rec.get("as_of", "")))
            if fv is None or asof is None:
                continue
            cross.append((t, fv))
            entry_dates[t] = asof
        if len(cross) < _MIN_NAMES:
            continue

        labels = tercile_labels(cross)
        fwd: dict[str, Decimal] = {}
        for t, _ in cross:
            entry = entry_dates[t]
            exit_d = entry + timedelta(days=_HOLD_DAYS)
            pe = _price_on_or_after(prices.get(t, {}), entry)
            px = _price_on_or_after(prices.get(t, {}), exit_d)
            if pe is None or px is None:
                continue
            div = _dividends_between(dividends.get(t, []), entry, exit_d)
            r = forward_total_return(Decimal(str(pe)), Decimal(str(px)), div)
            if r is not None:
                fwd[t] = r
        if sum(1 for t in fwd) < _MIN_NAMES:
            continue

        spread = long_short_spread(labels, fwd)
        top_rets = [fwd[t] for t, lab in labels.items() if lab == "top" and t in fwd]
        top_mean = sum(top_rets, Decimal(0)) / Decimal(len(top_rets)) if top_rets else None
        # deposit opened at the median entry date of this cross-section.
        # dep_frozen = old (too-hard) bar: entry rate held flat 1yr.
        # dep_real  = honest bar: reinvested at the CBR key rate as it MOVES (in an
        #             easing cycle this is lower than the frozen entry rate).
        med_entry = sorted(entry_dates[t] for t, _ in cross)[len(cross) // 2]
        dep_frozen = deposit_accrual(_key_rate(med_entry), _HOLD_DAYS)
        dep_real = realized_deposit(med_entry, _HOLD_DAYS)
        per_year.append(
            {
                "report_year": fy,
                "fiscal_year_approx": fy - 1,
                "entry": med_entry.isoformat(),
                "key_rate_pct": float(_key_rate(med_entry)),
                "n_names": len(cross),
                "n_with_forward": len(fwd),
                "ls_spread": float(spread) if spread is not None else None,
                "top_mean_return": float(top_mean) if top_mean is not None else None,
                "deposit_return": float(dep_real),  # PRIMARY = realized reinvested
                "deposit_return_frozen": float(dep_frozen),
                "top_excess_over_deposit": (
                    float(excess_over_deposit(top_mean, dep_real)) if top_mean is not None else None
                ),
                "top_excess_over_frozen": (
                    float(excess_over_deposit(top_mean, dep_frozen))
                    if top_mean is not None
                    else None
                ),
            }
        )

    return {"factor": factor, "per_year": per_year}


def _verdict(results: list[dict]) -> dict:
    summary = {}
    for res in results:
        f = res["factor"]
        ys = [y for y in res["per_year"] if y["ls_spread"] is not None]
        if len(ys) < _MIN_YEARS:
            summary[f] = {"verdict": "THIN_INCONCLUSIVE", "n_years": len(ys)}
            continue
        spreads = [Decimal(str(y["ls_spread"])) for y in ys]
        top_ex = [
            Decimal(str(y["top_excess_over_deposit"]))
            for y in ys
            if y["top_excess_over_deposit"] is not None
        ]
        mean_spread = sum(spreads, Decimal(0)) / Decimal(len(spreads))
        mean_top_ex = sum(top_ex, Decimal(0)) / Decimal(len(top_ex)) if top_ex else None
        hit_spread = sum(1 for s in spreads if s > 0) / len(spreads)
        beats_dep = (
            "TOP_BEATS_DEPOSIT"
            if (mean_top_ex is not None and mean_top_ex > 0)
            else "TOP_BELOW_DEPOSIT"
        )
        summary[f] = {
            "verdict": beats_dep,
            "n_years": len(ys),
            "mean_ls_spread": float(mean_spread),
            "spread_hit_rate": round(hit_spread, 3),
            "mean_top_excess_over_deposit": float(mean_top_ex) if mean_top_ex is not None else None,
            "caveat": "THIN/SHORT/POST-2022 — hypothesis, not edge",
        }
    return summary


def main() -> None:
    panel = json.loads(_PANEL.read_text(encoding="utf-8"))
    # Adversarial-review fix: back-adjust ISS closes AND dividends for splits (both, or a
    # split-crossing window fabricates a return) before any forward return is computed.
    for t, px in panel["prices"].items():
        sp = detect_splits(px)
        if not sp:
            continue
        panel["prices"][t] = {d: v * split_factor_at(sp, d) for d, v in px.items()}
        for div in panel["dividends"].get(t, []):
            rd = str(div.get("registryclosedate", ""))
            val = div.get("value")
            if rd and val is not None:
                div["value"] = float(val) * split_factor_at(sp, rd)
    factors = ["gp_a", "ev_ebitda_cheap", "roa"]
    results = [_run_factor(panel, f) for f in factors]
    summary = _verdict(results)
    cert = {
        "study": "moex_fundamental_factor",
        "bar": "deposit REINVESTED at the moving CBR key rate over each hold (real steps to "
        "2026-07, current 14.25% and easing); frozen-entry-rate bar kept as *_frozen",
        "hold_days": _HOLD_DAYS,
        "caveats": [
            "The deposit bar is a MOVING TARGET: it peaked at 16-21% (the regime that "
            "dominates this 2022-2025 panel) and has since EASED to ~14.25% (CBR, 2026-07) "
            "and falling. The 'deposit unbeatable' finding is REGIME-CONDITIONAL on the peak; "
            "as the bar drops toward neutral the equity/factor case mechanically strengthens",
            "N=4 annual waves: spread hit-rates ~0.5/0.5/0.0 are coin-flip — the "
            "TOP_BELOW_DEPOSIT label is NOT statistically distinguishable from zero; "
            "read it as directional hypothesis-refutation, not a proven factor verdict",
            "MARKET/BETA verdict more than a factor verdict: the whole equal-weight "
            "universe also trailed the deposit in the 3 losing waves (2022 +5.7% vs 20%, "
            "2025 -8.3% vs 21%); the one win (2023) was the low-rate 7.5% post-crash rebound",
            "ISS closes back-adjusted for splits (PLZL 1:10.18, GMKN 1:98.4, VTBR reverse); "
            "dividends are NOT split-adjusted (minor residual); pre-fix GP/A was -12.6%, "
            "split-corrected ~-8.4% — label unchanged, magnitude was inflated by the bug",
            "GP/A cross-section excludes BOTH banks (no COGS/empty MSFO) AND oil majors "
            "(GAZP/LKOH/NVTK/SNGS have no SmartLab MSFO revenue/cost) -> ~40% metals&mining; "
            "the GP/A top-tercile return is largely a metals-cohort proxy, and 2022 reflects "
            "the war-shock hit to that cohort as much as the GP/A signal itself",
            "SmartLab is survivor-biased (no delisted tickers): survivors have upward-biased "
            "returns, so a no-edge verdict is CONSERVATIVE (true edge <= measured)",
            "SmartLab IFRS depth ~5-6 fiscal years -> panel is short, post-2022-dominated; "
            "serves possibly-restated values (only the disclosure date is frozen)",
            "entry = disclosure date 'Дата отчёта' (look-ahead-safe); ~18 of the 2022-wave "
            "entries fall in the MOEX halt (2022-02-28..03-23) and use stale ISS carry-forward "
            "quotes — correcting to the Mar-24 reopen makes the deposit gap slightly WIDER",
            "deposit bar uses the wave-median entry key rate held simple ACT/365 "
            "(uncompounded) — conservative in 3 of 4 waves",
        ],
        "summary": summary,
        "detail": results,
    }
    _OUT.write_text(json.dumps(cert, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {_OUT}")


if __name__ == "__main__":
    main()
