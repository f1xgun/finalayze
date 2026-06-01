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

# ── Top-N / RUB-floor selection parameters (SINGLE SOURCE -- D-03 / D-11) ────────
# The operator-chosen Top-N-per-sector and RUB turnover sanity floor, derived from the
# Task-2 live turnover distribution (Plan 66-04) and recorded in
# docs/operations/liquidity_universe_runbook.md. These live HERE (Layer 2) as named
# constants so BOTH the runtime selector (``eligible_universe_as_of`` default ``top_n``)
# AND the one-shot generator (``scripts/generate_liquidity_universe.py`` argparse defaults)
# read the SAME values -- no magic numbers, no two divergent copies. The committed snapshot
# was generated with exactly these (see its ``params`` block); a quarterly regeneration that
# changes N / the floor edits THESE constants (and re-runs the generator), keeping the
# committed file and the selector in lock-step.
_TOP_N_PER_SECTOR = 10  # D-03: Top-10 highest-turnover names per sector.
_MIN_TURNOVER_FLOOR_RUB = Decimal(1_000_000)  # 1M RUB/day median sanity floor (drop below).

# ── D-11 portfolio no-regression acceptance tolerances (SINGLE SOURCE) ───────────
# The operator-chosen bar the expanded-universe backtest-iteration is judged against vs the
# current curated baseline (Task-4 checkpoint / LIQ-10). Recorded here so the runbook and the
# Task-4 acceptance read identical numbers. Relative tolerances: the expanded universe is
# accepted iff PF is not worse by more than 5%, MaxDD is not worse (larger) by more than 15%,
# and WF-Sharpe is not worse by more than 10% -- AND the per-segment WF gate still passes.
_D11_PF_REGRESSION_TOLERANCE_PCT = Decimal(5)  # PF >= -5% vs baseline.
_D11_MAXDD_REGRESSION_TOLERANCE_PCT = Decimal(15)  # MaxDD <= +15% (relative) vs baseline.
_D11_WF_SHARPE_REGRESSION_TOLERANCE_PCT = Decimal(10)  # WF-Sharpe >= -10% vs baseline.

# ── Universal safety filters (single source) ─────────────────────────────────────
# Applied to the FINAL selected universe (snapshot OR bootstrap) so the exclusion holds
# regardless of source. Pre-66 these filters lived only on the run_iteration bootstrap
# path; centralising them HERE makes every seam (config.segments.DEFAULT_SEGMENTS,
# run_iteration.UNIVERSE, training.cli.SEGMENT_SYMBOLS) inherit identical behaviour from
# the one selector. scripts.run_iteration._drop_toxic re-uses _TOXIC_SYMBOLS (no divergent
# duplicate). Layer-2 pure: stdlib only, no scripts.* / grpc / DB imports.
#
# Toxic / sanctioned / structurally-illiquid MOEX names the backtest harness has always
# excluded from every ru_* segment (pre-66 invariant -- see tests/unit/test_run_iteration_
# universe.py and tests/unit/test_config.py::TestToxicSymbolsExcluded). SNGS/SNGSP are the
# sanctioned Surgutneftegaz pair; GAZP/VTBR/ALRS/IRAO are sanctioned/illiquid.
_TOXIC_SYMBOLS: frozenset[str] = frozenset({"GAZP", "VTBR", "ALRS", "SNGS", "SNGSP", "IRAO"})


# MOEX preferred shares carry a trailing ``P`` on the common ticker (SBERP=SBER pref,
# TATNP=TATN pref, ...). A preferred share that is rho>0.95-redundant with its common
# counterpart already in the SAME selected set adds no diversification and degrades ML
# (Phase-48 rule: SBERP removed from ru_finance because rho>0.95 with SBER). This drops a
# ``<X>P`` ONLY when its common ``<X>`` is present in the same set -- it does NOT exclude a
# preferred share whose common is absent (e.g. TRNFP stays when TRNF is not traded), so
# legitimately-traded standalone preferreds are preserved (objective constraint).
def _drop_preferred_duplicates(symbols: list[str]) -> list[str]:
    """Drop ``<X>P`` preferred shares whose common ``<X>`` is in ``symbols`` (order-preserving)."""
    present = set(symbols)
    return [s for s in symbols if not (len(s) > 1 and s.endswith("P") and s[:-1] in present)]


def _apply_safety_filters(symbols: list[str]) -> list[str]:
    """Universal post-filter: drop toxic/sanctioned names then preferred-share duplicates.

    Order-preserving (keeps the ranked sequence). Applied to the FINAL selected universe so
    the exclusion holds whether the symbols came from the committed snapshot or the bootstrap
    fallback -- the single-source guarantee the three seams rely on.
    """
    deduped = _drop_preferred_duplicates(symbols)
    return [s for s in deduped if s not in _TOXIC_SYMBOLS]


def _valid_sectors() -> frozenset[str]:
    """Curated valid-sector set (V5 / IN-05 trust boundary), SINGLE-SOURCED from config.

    Plan 02 makes ``config.segments.SECTOR_TO_SEGMENT`` the one authoritative sector
    source: its keys ARE the set of valid snapshot sectors. The import is done lazily
    here (not at module top) to avoid a circular import at boot -- ``config.segments``
    imports ``select_segment_symbols`` from THIS module to populate ``DEFAULT_SEGMENTS``,
    while THIS module reads ``SECTOR_TO_SEGMENT`` from config. A top-level import either
    way would form a cycle; the lazy call here runs only after both modules have finished
    importing. (config Layer 1 <-> markets Layer 2 -- see config/segments.py for the
    documented layer note.)
    """
    from config.segments import SECTOR_TO_SEGMENT  # noqa: PLC0415

    return frozenset(SECTOR_TO_SEGMENT.keys())


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
    valid = _valid_sectors()
    for sector in sectors:
        if sector not in valid:
            msg = (
                f"unknown sector {sector!r} in liquidity snapshot at {_LIQ_SNAPSHOT} "
                f"(valid: {sorted(valid)})"
            )
            raise ConfigurationError(msg)
    return sectors


def select_segment_symbols(
    segment_id: str,
    sector_to_segment: dict[str, str] | None = None,
    bootstrap: list[str] | None = None,
) -> list[str]:
    """Thin LIVE selector: ranked symbols for ``segment_id`` from the committed snapshot.

    Maps each snapshot sector to its segment via ``sector_to_segment`` (the curated D-08
    map). When ``sector_to_segment`` is ``None`` (the LIVE config-construction call from
    ``config.segments.DEFAULT_SEGMENTS``), the curated map is read lazily from
    ``config.segments.SECTOR_TO_SEGMENT`` -- the single D-08 source. No live calls.
    Returns the concatenated ranked symbol list for the segment (LIQ-03), AFTER the
    universal safety post-filter (``_apply_safety_filters``): toxic/sanctioned names and
    preferred-share duplicates are dropped from the FINAL set regardless of source, so the
    pre-66 ru_* exclusion invariant holds for both the committed snapshot and the bootstrap.

    Bootstrap fallback (pre-66-04 compat shim): the committed snapshot artifact is written
    by ``scripts/generate_liquidity_universe.py`` (Plan 66-04). Until it is generated the
    FILE does not exist yet -- this is the expected bootstrap state, NOT tampering. In that
    single case the selector returns ``bootstrap`` (the segment's PRIOR hardcoded symbol
    list, supplied by the config-construction call site) so pre-66 behaviour is preserved
    and the whole boot path -- plus every ``required_symbols()``-derived consumer -- stays
    intact before the artifact lands. Once 66-04 generates the snapshot, the liquid set
    replaces the bootstrap (Phase-65 compat-shim philosophy). ``bootstrap`` defaults to
    ``[]`` only when no list is passed.

    This is NOT a stale-list fallback for tampering: a snapshot that EXISTS but is corrupt
    / has an unknown sector STILL fails-closed (the loader raises ``ConfigurationError``,
    propagated here) -- only true FILE ABSENCE bootstraps, per D-04 / T-66-08.
    """
    if sector_to_segment is None:
        from config.segments import SECTOR_TO_SEGMENT  # noqa: PLC0415

        sector_to_segment = SECTOR_TO_SEGMENT

    if not _LIQ_SNAPSHOT.exists():
        # Bootstrap: artifact not yet generated (Plan 66-04). Return the segment's prior
        # hardcoded list (pre-66 behaviour) -- NOT empty, which would clobber every ru_*
        # universe and the Phase-65 generator's required_symbols(). A corrupt/tampered
        # EXISTING file is still fail-closed below (only true ABSENCE bootstraps).
        fallback = _apply_safety_filters(list(bootstrap) if bootstrap is not None else [])
        _log.warning(
            "liquidity_snapshot_absent_bootstrap",
            segment_id=segment_id,
            path=str(_LIQ_SNAPSHOT),
            bootstrap_count=len(fallback),
            note="prior hardcoded list until generate_liquidity_universe.py runs (Plan 66-04)",
        )
        return fallback

    sectors = _load_liquidity_snapshot()
    out: list[str] = []
    for sector, symbols in sectors.items():
        if sector_to_segment.get(sector) == segment_id:
            out.extend(symbols)
    # Universal safety post-filter (single source): toxic/sanctioned + preferred-share
    # duplicates dropped from the FINAL selected set, regardless of source (D-04 trust
    # boundary returns the snapshot verbatim EXCEPT for these always-on safety filters,
    # which were a pre-66 invariant on every ru_* universe).
    return _apply_safety_filters(out)


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
    top_n: int = _TOP_N_PER_SECTOR,
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
