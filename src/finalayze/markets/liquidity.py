"""MOEX liquidity primitive -- turnover helper, fail-closed loader, as-of selector (Layer 2).

Turns the static MOEX share snapshot into a turnover-rankable, point-in-time-safe
selection primitive (Phase 66, sub-area 2). Three concerns:

1. ``median_rub_turnover`` -- per-symbol median daily RUB turnover (close x volume) over a
   trailing window (D-01 metric, D-02 60-day window). Returns ``None`` for short history.
2. ``_load_liquidity_snapshot`` -- fail-closed committed-snapshot loader (D-04). Raises
   ``ConfigurationError`` on a missing/corrupt/unknown-sector file; NEVER falls back to a
   stale list. The committed JSON is the trust boundary (IN-05 pattern).
3. ``top_n_per_sector`` / ``eligible_universe_as_of`` -- Top-N-per-sector selector (D-03 /
   LIQ-06) and the CARDINAL point-in-time as-of universe function (D-05): selection at a
   rebalance uses only candles dated ``<= rebalance_ts`` -- zero look-ahead, survivorship-safe.

Layer-2 purity (CLAUDE.md invariant #1): stdlib + structlog + Layer-0 ``ConfigurationError`` /
``Candle`` only. No ``scripts.*``, no grpc/Tinkoff, no DB imports at runtime.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import json
import statistics
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_log = structlog.get_logger()

# D-02 trailing window (named constant -- no magic number).
_MIN_BARS_FOR_LIQUIDITY = 60

# Survivorship guard (D-05 / LIQ-04): a name's most-recent visible candle must fall within
# this many calendar days of the rebalance, else it is treated as inactive (delisted /
# stale) and excluded. Daily bars over a 60-trading-day window span ~84 calendar days;
# a generous multiple of the window tolerates weekends/holidays/gaps without admitting a
# long-dead name that merely "had >= 60 candles" somewhere in the distant past.
_MAX_STALENESS_DAYS = _MIN_BARS_FOR_LIQUIDITY * 3

# The committed liquidity snapshot -- read fail-closed, written ONLY by the
# generator (scripts/generate_liquidity_universe.py). Package-relative data asset,
# mirroring markets/data/moex_universe.json.
_LIQ_SNAPSHOT = Path(__file__).parent / "data" / "moex_liquidity_universe.json"

# Curated valid-sector set (V5 / IN-05 trust boundary). Plan 02 replaces this with the
# keys of config.segments.SECTOR_TO_SEGMENT (single source); until that downward Layer-1
# import is wired, this module-level fallback keeps the loader independently green.
_VALID_SECTORS: frozenset[str] = frozenset(
    {
        "oil_gas",
        "banks",
        "metals_mining",
        "utilities",
        "telecom",
        "consumer",
        "transport",
        "chemicals",
        "tech",
        "real_estate",
        "diversified",
    }
)


def median_rub_turnover(
    candles: list[Candle], window: int = _MIN_BARS_FOR_LIQUIDITY
) -> Decimal | None:
    """Median daily RUB turnover (close x volume) over the trailing ``window`` bars.

    ``Candle.close`` is ``Decimal`` and ``Candle.volume`` is ``int``, so the per-bar
    product is exact RUB turnover. The median is computed in ``float`` for speed and
    coerced back to ``Decimal`` via ``str`` (avoids binary-float artifacts).

    Returns ``None`` for a short-history name (< ``window`` bars) so the caller excludes
    it (Pitfall 7 / LIQ-01). The caller is responsible for the ``<= as_of`` filter and
    any ``MOEX_2022_BREAK`` exclusion -- this helper does not look at timestamps.
    """
    recent = candles[-window:]
    if len(recent) < window:
        return None
    daily = [float(c.close) * c.volume for c in recent]
    return Decimal(str(statistics.median(daily)))


def _load_liquidity_snapshot() -> dict[str, list[str]]:
    """Read the committed liquidity snapshot, fail-closed (D-04 / LIQ-02).

    Returns the ``sectors`` map (sector -> ranked symbols). Raises ``ConfigurationError``
    on a missing/corrupt file, an absent ``sectors`` key, or an unknown sector key
    (V5 / IN-05 trust boundary -- no silent drop). NEVER falls back to a stale list.
    """
    try:
        raw: Any = json.loads(_LIQ_SNAPSHOT.read_text(encoding="utf-8"))
        sectors: dict[str, list[str]] = raw["sectors"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        msg = f"liquidity snapshot missing/corrupt at {_LIQ_SNAPSHOT}: {exc}"
        raise ConfigurationError(msg) from exc  # NO fallback to a stale list (D-04)

    # IN-05: validate every sector key against the curated set. The committed file is
    # attacker-influenceable; an unknown sector must surface, not be silently traded.
    for sector in sectors:
        if sector not in _VALID_SECTORS:
            msg = (
                f"unknown sector {sector!r} in liquidity snapshot at {_LIQ_SNAPSHOT} "
                f"(valid: {sorted(_VALID_SECTORS)})"
            )
            raise ConfigurationError(msg)
    return sectors


def select_segment_symbols(segment_id: str, sector_to_segment: dict[str, str]) -> list[str]:
    """Thin LIVE selector: ranked symbols for ``segment_id`` from the committed snapshot.

    Maps each snapshot sector to its segment via ``sector_to_segment`` (the curated D-08
    map, supplied by the caller -- this Layer-2 module does not import Layer-1 config).
    No live calls. Returns the concatenated ranked symbol list for the segment (LIQ-03).
    """
    sectors = _load_liquidity_snapshot()
    out: list[str] = []
    for sector, symbols in sectors.items():
        if sector_to_segment.get(sector) == segment_id:
            out.extend(symbols)
    return out


def top_n_per_sector(
    scores: dict[str, Decimal],
    sector_map: dict[str, str],
    top_n: int,
) -> dict[str, list[str]]:
    """Top-N highest-turnover symbols per sector (D-03 / LIQ-06).

    Groups ``scores`` by ``sector_map`` sector, sorts each group by score descending, and
    keeps the top ``top_n``. The total eligible size is bounded by ``top_n * sector_count``,
    balancing across sectors instead of letting one sector dominate.
    """
    grouped: dict[str, list[str]] = {}
    for symbol, sector in sector_map.items():
        if symbol in scores:
            grouped.setdefault(sector, []).append(symbol)

    result: dict[str, list[str]] = {}
    for sector, symbols in grouped.items():
        ranked = sorted(symbols, key=lambda s: scores[s], reverse=True)
        result[sector] = ranked[:top_n]
    return result


def eligible_universe_as_of(
    candles_by_symbol: dict[str, list[Candle]],
    rebalance_ts: Any,
    sector_map: dict[str, str],
    top_n: int,
) -> set[str]:
    """CARDINAL point-in-time eligible universe at ``rebalance_ts`` (D-05 / LIQ-04).

    For each symbol, only candles dated ``timestamp <= rebalance_ts`` are visible -- a
    future-dated candle CANNOT influence a past selection (look-ahead guard, mirroring the
    ``as_of <= cutoff`` idiom in ``data/loader.py``). Symbols with < 60 visible bars score
    ``None`` and are excluded, which makes the function survivorship-safe: a delisted name
    is eligible only for dates it actually had >= 60 bars.

    Returns the union of ``top_n_per_sector`` over the as-of turnover scores.
    """
    staleness_cutoff = rebalance_ts - timedelta(days=_MAX_STALENESS_DAYS)
    scores: dict[str, Decimal] = {}
    for symbol, candles in candles_by_symbol.items():
        past = [c for c in candles if c.timestamp <= rebalance_ts]
        if not past:
            continue
        # Survivorship guard: a delisted/stale name whose most-recent visible candle is
        # older than the staleness horizon is no longer actively traded -- exclude it even
        # though it once "had >= 60 candles" (LIQ-04). The <= rebalance_ts filter above
        # already forbids look-ahead; this forbids look-BACK at a dead name.
        if past[-1].timestamp < staleness_cutoff:
            continue
        liq = median_rub_turnover(past)
        if liq is not None:
            scores[symbol] = liq

    selected = top_n_per_sector(scores, sector_map, top_n)
    eligible: set[str] = set()
    for symbols in selected.values():
        eligible.update(symbols)
    return eligible
