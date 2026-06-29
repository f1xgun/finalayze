"""Diagnostic equity-tilt basket simulator (active-equity-sleeve R&D).

NOT production trading code. This is a transparent, deterministic weighted-basket
simulator that answers ONE honest question raised by the strategic-direction
pivot (docs/research/strategic_direction_review.md): does a LOW-TURNOVER active
weighting of the liquid MOEX share universe (equal-weight / ADV-cap-proxy /
dividend-yield) beat a cap-weight proxy on RISK-ADJUSTED total return, NET of the
operator's real retail 1.10% round-trip cost and NET-of-NDFL dividends?

Design choices (kept deliberately simple and auditable):

- A single weighted basket of a FIXED universe, rebalanced on supplied dates
  (quarterly), trading only the delta to target. Costs are charged via the
  canonical :data:`finalayze.backtest.costs.MOEX_RETAIL_COSTS` so the cost basis
  is identical to the rest of the engine.
- Dividends accrue on held shares on their ex-date (committed
  ``moex_dividends.yaml`` via ``load_dividend_schedule``), credited NET-of-NDFL
  through the L0 :class:`finalayze.core.ndfl.YtdTaxAccumulator` (progressive
  13/15% band) so the income tax basis matches the real allocator.
- Weight policies are pure, AS-OF functions: they receive the rebalance date and
  the full panel and MUST filter to data ``<=`` the as-of date themselves (no
  look-ahead). Fractional shares are allowed (this is an index-replication-style
  NAV diagnostic, not an order generator).

The crucial honesty point (from the R&D consilium): a tilt is judged against a
LIKE-FOR-LIKE cap-proxy run through THIS SAME simulator (same dividends, same
costs, same tax) — never against a gross published index, which a net-of-tax
basket cannot beat by construction.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.ndfl import YtdTaxAccumulator

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts

# (ex_date close, daily volume in shares) for one trading day of one symbol.
PricePoint = tuple[date, Decimal, Decimal]

# A pure, as-of weighting function: (rebalance_date, full_panel) -> {symbol: weight}.
# The policy MUST only look at panel points dated <= the rebalance date.
WeightPolicy = Callable[[date, dict[str, list[PricePoint]]], dict[str, Decimal]]

_DEFAULT_INITIAL_NAV = Decimal(1000000)
_DEFAULT_ADV_LOOKBACK = 63  # ~one quarter of trading days
_DEFAULT_VOL_LOOKBACK = 126  # ~two quarters for a stable realized vol
_MIN_VOL_RETURNS = 20  # below this a vol estimate is too noisy -> name excluded
_DIV_YIELD_LOOKBACK_DAYS = 365  # trailing-12m realized (paid) dividends
_DIV_TILT_K = Decimal("0.5")  # pre-registered tilt strength (NOT fitted)
_DIV_CLIP_LO = Decimal("0.5")  # weight floor as a multiple of 1/N
_DIV_CLIP_HI = Decimal("2.0")  # weight ceiling as a multiple of 1/N
# Low-vol BLEND (step 3): FINAL = (1-lambda)*cap_weight + lambda*inverse_vol(lowest-vol half).
_DEFAULT_LOWVOL_LAMBDA = Decimal("0.25")
_QUARTER_LEN = 3
_PERCENT = Decimal(100)


@dataclass(frozen=True)
class BasketResult:
    """The daily NAV curve of one basket arm plus its cost/tax/income totals."""

    dates: list[date]
    nav_curve: list[Decimal]
    total_cost: Decimal
    total_tax: Decimal
    dividend_gross: Decimal
    rebalance_dates: list[date]

    @property
    def equity_floats(self) -> list[float]:
        """NAV curve as floats for the (float-based) Sharpe/Sortino primitives."""
        return [float(x) for x in self.nav_curve]


def quarter_end_dates(dates: list[date]) -> list[date]:
    """Return the last trading day present in each calendar quarter of ``dates``.

    These are the natural quarterly rebalance boundaries. The very first date is
    NOT forced in here; callers prepend it as the initial-allocation rebalance.
    """
    last_in_quarter: dict[tuple[int, int], date] = {}
    for d in sorted(dates):
        key = (d.year, (d.month - 1) // _QUARTER_LEN)
        # later dates overwrite earlier ones -> ends on the quarter's last day
        if key not in last_in_quarter or d > last_in_quarter[key]:
            last_in_quarter[key] = d
    return sorted(last_in_quarter.values())


def _available(asof: date, panel: dict[str, list[PricePoint]]) -> list[str]:
    """Symbols that have at least one price point dated on/before ``asof``."""
    return [s for s, pts in panel.items() if pts and pts[0][0] <= asof]


def equal_weights(asof: date, panel: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
    """1/N over every symbol that already has data as of ``asof`` (the dumbest tilt)."""
    names = _available(asof, panel)
    if not names:
        return {}
    w = Decimal(1) / Decimal(len(names))
    return dict.fromkeys(names, w)


def adv_cap_proxy_weights(
    asof: date,
    panel: dict[str, list[PricePoint]],
    lookback: int = _DEFAULT_ADV_LOOKBACK,
) -> dict[str, Decimal]:
    """Cap-weight PROXY: weight by trailing average daily value traded (close*volume).

    A candle-only, as-of, look-ahead-free stand-in for free-float cap weight (the
    repo has no historical constituent-weight panel). For the liquid MOEX universe
    the most-traded names ARE the mega-caps, so ADV-weight tracks cap-weight
    directionally — enough to isolate the WEIGHTING decision against a tilt on an
    identical cost/tax basis. Falls back to equal-weight if no traded value.
    """
    advs: dict[str, Decimal] = {}
    for s in _available(asof, panel):
        recent = [p for p in panel[s] if p[0] <= asof][-lookback:]
        if not recent:
            continue
        adv = sum((close * vol for _, close, vol in recent), Decimal(0)) / Decimal(len(recent))
        if adv > 0:
            advs[s] = adv
    total = sum(advs.values(), Decimal(0))
    if total <= 0:
        return equal_weights(asof, panel)
    return {s: adv / total for s, adv in advs.items()}


def inverse_vol_weights(
    asof: date,
    panel: dict[str, list[PricePoint]],
    lookback: int = _DEFAULT_VOL_LOOKBACK,
) -> dict[str, Decimal]:
    """Low-volatility tilt: weight each name inversely to its trailing realized vol.

    Candle-only and as-of (uses only closes dated ``<=`` asof). Names with fewer
    than :data:`_MIN_VOL_RETURNS` trailing returns or zero vol are excluded that
    rebalance (held at 0). This is the classic low-vol anomaly expressed as a pure
    inverse-vol weighting — no cap-weight blend, so it needs no constituent panel.
    """
    vols = _trailing_vols(asof, panel, lookback)
    inv = {s: Decimal(1) / Decimal(str(v)) for s, v in vols.items()}
    total = sum(inv.values(), Decimal(0))
    if total <= 0:
        return equal_weights(asof, panel)
    return {s: v / total for s, v in inv.items()}


def _trailing_vols(
    asof: date, panel: dict[str, list[PricePoint]], lookback: int
) -> dict[str, float]:
    """Trailing realized vol (stdev of daily log returns) per available name, as-of."""
    vols: dict[str, float] = {}
    for s in _available(asof, panel):
        closes = [float(close) for d, close, _ in panel[s] if d <= asof][-(lookback + 1) :]
        rets = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0
        ]
        if len(rets) >= _MIN_VOL_RETURNS:
            vol = statistics.stdev(rets)
            if vol > 0:
                vols[s] = vol
    return vols


def make_index_cap_weight_policy(
    weights_by_date: dict[date, dict[str, Decimal]],
) -> WeightPolicy:
    """REAL cap-weight policy from the committed IMOEX index-weight panel (step 3).

    As-of: at rebalance date ``d`` it takes the latest snapshot date ``<= d``,
    restricts to constituents that have a candle available as-of ``d`` (de-listed/
    renamed names without candles drop out), and RENORMALIZES over the covered set.
    The per-date coverage (share of index weight retained) is the honesty metric
    reported alongside the cert. Falls back to equal-weight if no snapshot/coverage.
    """
    sorted_dates = sorted(weights_by_date)

    def policy(asof: date, panel: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        prior = [wd for wd in sorted_dates if wd <= asof]
        if not prior:
            return equal_weights(asof, panel)
        raw = weights_by_date[prior[-1]]
        avail = set(_available(asof, panel))
        covered = {t: w for t, w in raw.items() if t in avail}
        total = sum(covered.values(), Decimal(0))
        if total <= 0:
            return equal_weights(asof, panel)
        return {t: w / total for t, w in covered.items()}

    return policy


def make_low_vol_blend_policy(
    weights_by_date: dict[date, dict[str, Decimal]],
    *,
    lam: Decimal = _DEFAULT_LOWVOL_LAMBDA,
    vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
) -> WeightPolicy:
    """Low-vol BLEND overlay on the real cap-weight (step 3, pre-registered lambda).

    ``FINAL_i = (1-lambda)*cap_weight_i + lambda*inverse_vol_i`` where the
    inverse-vol leg spans only the LOWEST-VOL HALF of the cap-weight constituents
    (the rest get 0 on the tilt leg). ``lambda`` and the lookback are fixed a priori.
    At ``lambda == 0`` this returns the cap-weight vector byte-for-byte (the
    data-correctness control: the tilt must reduce to the baseline).
    """
    cap_policy = make_index_cap_weight_policy(weights_by_date)

    def policy(asof: date, panel: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        cap = cap_policy(asof, panel)
        if lam == 0 or not cap:
            return cap
        vols = _trailing_vols(asof, {s: panel[s] for s in cap if s in panel}, vol_lookback)
        if not vols:
            return cap
        ranked = sorted(vols, key=lambda s: vols[s])
        half = ranked[: max(1, len(ranked) // 2)]  # lowest-vol half = the tilt set L
        inv = {s: Decimal(1) / Decimal(str(vols[s])) for s in half}
        inv_total = sum(inv.values(), Decimal(0))
        w_iv = {s: v / inv_total for s, v in inv.items()}
        blended = {
            t: (Decimal(1) - lam) * cap.get(t, Decimal(0)) + lam * w_iv.get(t, Decimal(0))
            for t in cap
        }
        total = sum(blended.values(), Decimal(0))
        return {t: v / total for t, v in blended.items()} if total > 0 else cap

    return policy


def make_dividend_yield_policy(
    dividend_schedule: dict[tuple[str, date], Decimal],
    *,
    k: Decimal = _DIV_TILT_K,
    lookback_days: int = _DIV_YIELD_LOOKBACK_DAYS,
) -> WeightPolicy:
    """Build a dividend-yield tilt policy on an equal-weight base (pre-registered params).

    Yield_i = (sum of PAID dividends with ex-date in (asof - lookback, asof]) /
    close_asof — strictly trailing, realized only (the committed schedule already
    drops cancelled/reduced). Tilt: w_i = (1/N) * (1 + k * z_i) where z_i is the
    cross-sectional standardized yield, clipped to ``[lo/N, hi/N]`` and renormalized
    so no single name dominates. ``k`` and the clip are fixed a priori — never
    fitted to the verdict (the anti-overfit discipline from the R&D consilium).
    """

    def policy(asof: date, panel: dict[str, list[PricePoint]]) -> dict[str, Decimal]:
        names = _available(asof, panel)
        if not names:
            return {}
        n = Decimal(len(names))
        base = Decimal(1) / n
        cutoff = asof - timedelta(days=lookback_days)
        yields: dict[str, Decimal] = {}
        for s in names:
            pts = [p for p in panel[s] if p[0] <= asof]
            price = pts[-1][1] if pts else Decimal(0)
            if price <= 0:
                yields[s] = Decimal(0)
                continue
            div = sum(
                (
                    amt
                    for (sym, ex), amt in dividend_schedule.items()
                    if sym == s and cutoff < ex <= asof
                ),
                Decimal(0),
            )
            yields[s] = div / price
        vals = [float(y) for y in yields.values()]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        raw: dict[str, Decimal] = {}
        lo, hi = _DIV_CLIP_LO * base, _DIV_CLIP_HI * base
        for s in names:
            z = Decimal(str((float(yields[s]) - mean) / std)) if std > 0 else Decimal(0)
            w = base * (Decimal(1) + k * z)
            raw[s] = min(hi, max(lo, w))
        total = sum(raw.values(), Decimal(0))
        return {s: w / total for s, w in raw.items()}

    return policy


class _PriceLookup:
    """Forward-filling as-of price lookup for one symbol (last close on/before d)."""

    def __init__(self, points: list[PricePoint]) -> None:
        ordered = sorted(points, key=lambda p: p[0])
        self._dates = [p[0] for p in ordered]
        self._closes = [p[1] for p in ordered]

    def price_on_or_before(self, d: date) -> Decimal | None:
        idx = bisect.bisect_right(self._dates, d) - 1
        if idx < 0:
            return None
        return self._closes[idx]


def simulate_basket(
    *,
    panel: dict[str, list[PricePoint]],
    dividend_schedule: dict[tuple[str, date], Decimal],
    weight_policy: WeightPolicy,
    rebalance_dates: list[date],
    costs: TransactionCosts,
    initial_nav: Decimal = _DEFAULT_INITIAL_NAV,
) -> BasketResult:
    """Simulate one weighted-basket arm to a daily NAV curve, net of cost and NDFL.

    Per trading day (the sorted union of all panel dates), in order:
      1. accrue dividends on held shares whose ex-date == today, NET-of-NDFL;
      2. if today is a rebalance date, trade every name's delta to its target
         weight (target = NAV * weight), charging ``costs`` on each traded leg;
      3. mark NAV = cash + Σ shares * as-of price.

    Being fully invested minus costs leaves a small negative cash residue equal to
    the cost drag — this is exact NAV accounting (NAV = nav_before - costs), not a
    bug; dividends and later rebalances absorb it.
    """
    lookups = {s: _PriceLookup(pts) for s, pts in panel.items()}
    all_dates = sorted({d for pts in panel.values() for d, _, _ in pts})
    rebal = set(rebalance_dates)

    holdings: dict[str, Decimal] = {}
    cash = initial_nav
    tax_acc = YtdTaxAccumulator()
    total_cost = Decimal(0)
    total_tax = Decimal(0)
    dividend_gross = Decimal(0)

    nav_curve: list[Decimal] = []
    for d in all_dates:
        # 1. dividends on shares held going into today (must hold before ex-date)
        for name, shares in holdings.items():
            if shares <= 0:
                continue
            per_share = dividend_schedule.get((name, d))
            if per_share is None:
                continue
            gross = per_share * shares
            tax = tax_acc.tax(gross, d.year)
            cash += gross - tax
            dividend_gross += gross
            total_tax += tax

        # 2. rebalance to target weights
        if d in rebal:
            nav_now = cash + sum(
                (shares * (lookups[name].price_on_or_before(d) or Decimal(0)))
                for name, shares in holdings.items()
            )
            targets = weight_policy(d, panel)
            names = set(targets) | {n for n, s in holdings.items() if s > 0}
            for name in names:
                price = lookups[name].price_on_or_before(d)
                if price is None or price <= 0:
                    continue
                target_value = nav_now * targets.get(name, Decimal(0))
                cur_value = holdings.get(name, Decimal(0)) * price
                delta_value = target_value - cur_value
                delta_shares = delta_value / price
                if delta_shares == 0:
                    continue
                cost = costs.total_cost(price, abs(delta_shares))
                cash -= delta_value
                cash -= cost
                holdings[name] = holdings.get(name, Decimal(0)) + delta_shares
                total_cost += cost

        # 3. mark NAV
        nav = cash + sum(
            (shares * (lookups[name].price_on_or_before(d) or Decimal(0)))
            for name, shares in holdings.items()
        )
        nav_curve.append(nav)

    return BasketResult(
        dates=all_dates,
        nav_curve=nav_curve,
        total_cost=total_cost,
        total_tax=total_tax,
        dividend_gross=dividend_gross,
        rebalance_dates=sorted(rebal),
    )


def max_drawdown_pct(curve: list[float]) -> float:
    """Peak-to-trough max drawdown of an equity curve, as a PERCENT (e.g. 25.0)."""
    if not curve:
        return 0.0
    peak = curve[0]
    worst = 0.0
    for x in curve:
        peak = max(peak, x)
        if peak > 0:
            dd = (peak - x) / peak
            worst = max(worst, dd)
    return worst * float(_PERCENT)
