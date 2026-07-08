"""Pure, tested primitives for the MOEX fundamental cross-sectional factor study.

Companion to ``jump_response_lab.py`` (same discipline: small, side-effect-free,
unit-pinned functions so the research cert cannot hide a look-ahead or an
arithmetic bug behind a plausible number).

Scope: given per-fiscal-year IFRS fundamentals (SmartLab MSFO) and a forward
price/total return per name, rank names cross-sectionally by a factor and measure
the top-minus-bottom spread against the risk-free deposit anchor.

NOT in scope: data fetching (see ``scripts/research/fetch_moex_fundamental_panel.py``)
or the study driver (``scripts/research/run_moex_fundamental_factor_study.py``).
"""

from __future__ import annotations

from decimal import Decimal

# A cross-sectional factor needs at least one name per tercile bucket.
_MIN_CROSS_SECTION = 3


def gross_profit_to_assets(
    revenue: Decimal, cost_of_sales: Decimal, total_assets: Decimal
) -> Decimal | None:
    """Novy-Marx gross profitability GP/A = (revenue - COGS) / total assets.

    Returns ``None`` if inputs are missing or assets are non-positive (a firm
    cannot be ranked on a factor it has no data for — never fabricate a 0.0).
    """
    if total_assets is None or total_assets <= 0:
        return None
    if revenue is None or cost_of_sales is None:
        return None
    return (revenue - cost_of_sales) / total_assets


def tercile_labels(values: list[tuple[str, Decimal]]) -> dict[str, str]:
    """Rank ``(name, factor)`` pairs into terciles: 'top' / 'mid' / 'bottom'.

    Higher factor value -> 'top'. Names with an equal factor fall on the natural
    sort boundary. Requires >= 3 names (one per bucket) else raises ValueError —
    a cross-sectional factor is undefined on a degenerate cross-section, and
    silently returning a 1- or 2-bucket split would fabricate a spread.
    """
    if len(values) < _MIN_CROSS_SECTION:
        msg = f"tercile ranking needs >=3 names, got {len(values)}"
        raise ValueError(msg)
    ordered = sorted(values, key=lambda kv: kv[1])  # ascending -> bottom first
    n = len(ordered)
    cut = n // 3
    out: dict[str, str] = {}
    for i, (name, _) in enumerate(ordered):
        if i < cut:
            out[name] = "bottom"
        elif i >= n - cut:
            out[name] = "top"
        else:
            out[name] = "mid"
    return out


def forward_total_return(
    price_entry: Decimal,
    price_exit: Decimal,
    dividends_paid: Decimal = Decimal(0),
) -> Decimal | None:
    """Simple forward total return = (exit - entry + dividends) / entry.

    ``dividends_paid`` is the per-share cash distributed between entry and exit
    (ex-dates strictly inside the window). Returns ``None`` if entry price is
    missing/non-positive (cannot form a return).
    """
    if price_entry is None or price_entry <= 0 or price_exit is None:
        return None
    return (price_exit - price_entry + dividends_paid) / price_entry


_SPLIT_RATIO = 3.0  # adjacent-day close ratio beyond this = a split, never a real move


def detect_splits(prices: dict[str, float]) -> list[tuple[str, float]]:
    """Find stock splits as ``(split_date, factor)``; pre-split values are multiplied by factor.

    A >3x or <1/3 adjacent-trading-day close ratio is a split (forward or reverse),
    never a real one-day move (e.g. PLZL 1:10.18, GMKN 1:98.4, VTBR ~5000:1 reverse).
    ``factor = post/pre`` (<1 for a forward split, >1 for a reverse). Dates are ISO
    strings so lexicographic ``<`` equals chronological ``<``.
    """
    dates = sorted(prices)
    out: list[tuple[str, float]] = []
    for i in range(1, len(dates)):
        pre, post = prices[dates[i - 1]], prices[dates[i]]
        if pre > 0 and post > 0:
            r = pre / post
            if r > _SPLIT_RATIO or r < 1.0 / _SPLIT_RATIO:
                out.append((dates[i], post / pre))
    return out


def split_factor_at(splits: list[tuple[str, float]], day: str) -> float:
    """Cumulative back-adjust multiplier for a value dated ``day`` (splits strictly AFTER it apply).

    MUST be applied to BOTH prices AND dividends: scaling prices without scaling the
    per-share dividend fabricates a return for any window crossing a split.
    """
    f = 1.0
    for sdate, ratio in splits:
        if day < sdate:
            f *= ratio
    return f


def deposit_accrual(annual_rate_pct: Decimal, days: int) -> Decimal:
    """Risk-free deposit total return over ``days`` at ``annual_rate_pct`` (simple, ACT/365).

    The honest bar the factor must beat. Simple (not compounded) to stay
    conservative for the factor — a lower bar, not a higher one.
    """
    if days < 0:
        msg = f"days must be non-negative, got {days}"
        raise ValueError(msg)
    return (annual_rate_pct / Decimal(100)) * (Decimal(days) / Decimal(365))


def long_short_spread(
    labels: dict[str, str],
    forward_returns: dict[str, Decimal],
) -> Decimal | None:
    """Mean forward return of 'top' tercile minus mean of 'bottom' tercile.

    Only names present in BOTH ``labels`` and ``forward_returns`` count (a name
    with a factor but no forward price, or vice-versa, is dropped — never
    imputed). Returns ``None`` if either leg ends up empty.
    """

    def _leg(bucket: str) -> list[Decimal]:
        return [
            forward_returns[n]
            for n, lab in labels.items()
            if lab == bucket and n in forward_returns
        ]

    top = _leg("top")
    bot = _leg("bottom")
    if not top or not bot:
        return None
    mean_top = sum(top, Decimal(0)) / Decimal(len(top))
    mean_bot = sum(bot, Decimal(0)) / Decimal(len(bot))
    return mean_top - mean_bot


def excess_over_deposit(portfolio_return: Decimal, deposit_return: Decimal) -> Decimal:
    """Factor/portfolio return minus the deposit anchor over the same window."""
    return portfolio_return - deposit_return
