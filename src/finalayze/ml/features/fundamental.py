"""Look-ahead-safe fundamental feature computation (FUND-02, Layer 3).

Mirrors ``macro.py``: every branch returns a defined ``0.0`` default (never NaN),
and all inputs are filtered to ``as_of <= D`` so a fundamental published after the
bar date can never leak into the feature vector.

Sector peers are resolved INSIDE this module from ``config/segments.py`` (keyed off
the snapshot's own symbol) so ``compute_features`` keeps its exact signature — no
``segment`` parameter is threaded through the 47 call sites (BLOCKER 2 decision).
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from config.segments import DEFAULT_SEGMENTS

from finalayze.ml.features.zscore import MOEX_ZSCORE_CLIP, safe_zscore

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import FundamentalSnapshot, MoexMarketData

# Minimum number of valid peer values required before a cross-sectional z-score is
# trusted. ru_* segments are small (ru_blue_chips=3); a z-score over 2-3 peers is
# noise, so below this threshold the z-score features return their 0.0 default.
_MIN_PEERS_FOR_ZSCORE = 4

# ~1 year tolerance window for the YoY revenue-growth lookback (snapshots are not
# guaranteed to land exactly 365 days apart).
_YOY_MIN_DAYS = 300
_YOY_MAX_DAYS = 430

_DEFAULT: dict[str, float] = {
    "earnings_yield": 0.0,
    "pe_zscore_vs_sector": 0.0,
    "revenue_growth_yoy": 0.0,
    "net_margin_trend": 0.0,
    "dividend_yield_z": 0.0,
}


def _resolve_sector_peer_symbols(symbol: str) -> tuple[str, ...]:
    """Sector peers = the symbols of the segment that owns ``symbol`` (config/segments.py).

    No ``compute_features`` signature change: the segment is derived from the snapshot
    symbol, not passed in. Returns ``()`` when the symbol is not in any segment (the
    0.0 defaults handle that case).
    """
    for seg in DEFAULT_SEGMENTS:
        if symbol in seg.symbols:
            return tuple(seg.symbols)
    return ()


def _filter_as_of(
    snapshots: tuple[FundamentalSnapshot, ...],
    as_of: datetime | None,
) -> list[FundamentalSnapshot]:
    """Keep only snapshots with ``as_of <= D`` (look-ahead guard)."""
    if as_of is None:
        return list(snapshots)
    return [s for s in snapshots if s.as_of <= as_of]


def _latest_per_symbol(
    snapshots: list[FundamentalSnapshot],
) -> dict[str, FundamentalSnapshot]:
    """Pick the most recent (max ``as_of``) snapshot per symbol."""
    latest: dict[str, FundamentalSnapshot] = {}
    for s in snapshots:
        cur = latest.get(s.symbol)
        if cur is None or s.as_of > cur.as_of:
            latest[s.symbol] = s
    return latest


def _earnings_yield(snapshot: FundamentalSnapshot) -> float:
    pe = snapshot.pe_ratio
    if pe is None or pe <= 0.0:
        return 0.0
    return 1.0 / pe


def _peer_zscore(
    target_value: float | None,
    peer_values: list[float],
) -> float:
    """Cross-sectional z-score of ``target_value`` against ``peer_values``.

    Guarded by ``_MIN_PEERS_FOR_ZSCORE``; returns 0.0 when too few valid peers,
    when the target value is missing, or when the dispersion is degenerate.
    """
    if target_value is None:
        return 0.0
    if len(peer_values) < _MIN_PEERS_FOR_ZSCORE:
        return 0.0
    population = [*peer_values, target_value]
    mean = statistics.fmean(population)
    std = statistics.pstdev(population)
    z = safe_zscore(target_value, mean, std)
    return max(-MOEX_ZSCORE_CLIP, min(MOEX_ZSCORE_CLIP, z))


def _revenue_growth_yoy(
    symbol: str,
    target: FundamentalSnapshot,
    history: list[FundamentalSnapshot],
) -> float:
    """YoY revenue growth using a snapshot ~1 year before ``target.as_of``.

    0.0 when there is no prior snapshot in the [300, 430]-day window or revenue is
    missing/non-positive.
    """
    rev_now = target.revenue_ttm
    if rev_now is None or rev_now <= 0.0:
        return 0.0
    best: FundamentalSnapshot | None = None
    for s in history:
        if s.symbol != symbol or s.revenue_ttm is None or s.revenue_ttm <= 0.0:
            continue
        age_days = (target.as_of - s.as_of).days
        if _YOY_MIN_DAYS <= age_days <= _YOY_MAX_DAYS and (best is None or s.as_of > best.as_of):
            best = s
    if best is None or best.revenue_ttm is None or best.revenue_ttm <= 0.0:
        return 0.0
    return (rev_now - best.revenue_ttm) / best.revenue_ttm


def _net_margin_trend(
    symbol: str,
    history: list[FundamentalSnapshot],
) -> float:
    """Change in net margin between the latest and the earliest prior snapshot.

    0.0 when fewer than two snapshots carry a net_margin value.
    """
    margins = [
        (s.as_of, s.net_margin) for s in history if s.symbol == symbol and s.net_margin is not None
    ]
    if len(margins) < 2:  # noqa: PLR2004 — need at least two points for a trend
        return 0.0
    margins.sort(key=lambda pair: pair[0])
    earliest = margins[0][1]
    latest = margins[-1][1]
    if earliest is None or latest is None:
        return 0.0
    return float(latest) - float(earliest)


def compute_fundamental_features(
    moex_data: MoexMarketData | None,
    *,
    as_of: datetime | None = None,
    sector_peers: tuple[FundamentalSnapshot, ...] | None = None,
) -> dict[str, float]:
    """Compute look-ahead-safe fundamental features for the snapshot's symbol.

    Keys: earnings_yield, pe_zscore_vs_sector, revenue_growth_yoy, net_margin_trend,
    dividend_yield_z. Returns the all-0.0 ``_DEFAULT`` when no usable snapshot exists.
    Every branch falls back to the default — NaN is never returned.

    ``sector_peers`` is an optional test/override hook. In production it is ``None`` and
    the peer universe is resolved internally from ``config/segments.py`` (BLOCKER 2).
    """
    if moex_data is None or not moex_data.fundamentals:
        return dict(_DEFAULT)

    in_window = _filter_as_of(moex_data.fundamentals, as_of)
    if not in_window:
        return dict(_DEFAULT)

    # The target symbol is read from the latest in-window snapshot (NOT a param).
    target = max(in_window, key=lambda s: s.as_of)
    symbol = target.symbol

    features = dict(_DEFAULT)
    features["earnings_yield"] = _earnings_yield(target)
    features["revenue_growth_yoy"] = _revenue_growth_yoy(symbol, target, in_window)
    features["net_margin_trend"] = _net_margin_trend(symbol, in_window)

    # Resolve the peer snapshots: explicit override hook, else internal resolution.
    if sector_peers is not None:
        peer_snapshots = [s for s in _filter_as_of(sector_peers, as_of) if s.symbol != symbol]
    else:
        peer_symbols = _resolve_sector_peer_symbols(symbol)
        peer_pool = [s for s in in_window if s.symbol in peer_symbols and s.symbol != symbol]
        peer_snapshots = list(_latest_per_symbol(peer_pool).values())

    pe_peers = [s.pe_ratio for s in peer_snapshots if s.pe_ratio is not None and s.pe_ratio > 0.0]
    features["pe_zscore_vs_sector"] = _peer_zscore(target.pe_ratio, pe_peers)

    div_peers = [s.dividend_yield for s in peer_snapshots if s.dividend_yield is not None]
    features["dividend_yield_z"] = _peer_zscore(target.dividend_yield, div_peers)

    return features
