"""Segment definitions (stocks, bonds, ETFs).

See docs/design/SEGMENTS.md for the full segment system design.

Layer note (Phase 66 Plan 02): this module is Layer 1 (config) yet imports
``select_segment_symbols`` from ``finalayze.markets.liquidity`` (Layer 2) to populate the
LIVE ``DEFAULT_SEGMENTS`` ru_* share universes from the committed liquidity snapshot
(D-07). Per the Phase-66 plan this cross-layer wiring is an explicit, sanctioned decision
(the curated ``SECTOR_TO_SEGMENT`` map below is the single D-08 sector source consumed by
both the selector and the Plan-04 generator). ``SECTOR_TO_SEGMENT`` is defined as a plain
module constant ABOVE/independent of the liquidity import so ``liquidity.py`` can read it
back without triggering ``DEFAULT_SEGMENTS`` construction -- breaking the import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Curated sector -> segment map (D-08 / LIQ-05) ────────────────────────────────
# SINGLE source of the sector->segment assignment. The Phase-65 share snapshot carries
# NO sector field, so this curated map is the authoritative, reviewable assignment that
# BOTH the LIVE selector (finalayze.markets.liquidity.select_segment_symbols) and the
# Plan-04 generator (scripts/generate_liquidity_universe.py) consult -- never duplicate
# the sector literal anywhere else. Keys are the curated MOEX sector tags; values are the
# ru_* SHARE segment_ids in DEFAULT_SEGMENTS below (NO bond segment is mapped --
# ru_ofz_pd / ru_ofz_pk are out of scope for the shares-only liquidity filter).
#
# Sector tags are seeded per A3 (manual seed now; a one-off T-Invest sector pull at
# generation time, Plan 04, may refresh/extend them deterministically).
# Defined ABOVE the liquidity import (cycle-break -- see module docstring).
SECTOR_TO_SEGMENT: dict[str, str] = {
    "oil_gas": "ru_energy",
    "banks": "ru_finance",
    "metals_mining": "ru_metals",
    "utilities": "ru_utilities",
    "telecom": "ru_telecom",
    "consumer": "ru_consumer",
    "transport": "ru_transport",
    "chemicals": "ru_chemicals",
    "tech": "ru_tech",
    "real_estate": "ru_construction",
}

# ── Bootstrap ru_* SHARE universes (pre-66-04 compat shim) ───────────────────────
# The PRIOR (pre-66-02) hardcoded ru_* SHARE symbol lists, captured verbatim from the
# segment definitions that existed before the liquidity selector was wired in. These are
# the bootstrap fallback the selector returns when the committed liquidity snapshot FILE
# is still ABSENT (Plan 66-04 has not yet generated it). Returning these preserves
# pre-66 behaviour so DEFAULT_SEGMENTS, run_iteration.UNIVERSE, training.SEGMENT_SYMBOLS
# and every required_symbols()-derived consumer stay populated. Once the snapshot lands,
# the liquid selected set replaces this bootstrap (D-07 / Phase-65 compat-shim philosophy).
# A snapshot that EXISTS but is corrupt still fails-closed -- only ABSENCE bootstraps.
# Bond segments (ru_ofz_pd / ru_ofz_pk) and US segments are NOT bootstrapped here -- they
# keep their own hardcoded lists below.
_BOOTSTRAP_SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "ru_energy": ["ROSN", "TATN", "NVTK", "SIBN", "TATNP", "TRNFP"],
    "ru_tech": ["YDEX", "OZON", "VKCO", "HEAD", "POSI", "ASTR", "DIAS", "SOFL"],
    "ru_finance": ["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR", "AFKS", "RENI"],
    "ru_metals": ["GMKN", "CHMF", "NLMK", "MAGN", "PLZL", "RUAL", "MTLR"],
    "ru_consumer": ["MGNT", "X5", "LENT"],
    "ru_telecom": ["MTSS", "RTKM"],
    "ru_utilities": ["HYDR", "FEES", "MSNG", "UPRO"],
    "ru_construction": ["PIKK", "SMLT"],
    "ru_chemicals": ["PHOR", "AKRN"],
    "ru_transport": ["AFLT", "FLOT", "NMTP"],
}

# Downward-wired LIVE selector (see module docstring for the Layer-1<->Layer-2 note).
# Imported AFTER SECTOR_TO_SEGMENT so liquidity.py's lazy read of that constant is safe.
from finalayze.markets.liquidity import select_segment_symbols  # noqa: E402


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for a market segment (stock, bond, ETF)."""

    segment_id: str
    market: str
    broker: str
    currency: str
    instrument_type: str = "stock"
    symbols: list[str] = field(default_factory=list)
    active_strategies: list[str] = field(default_factory=list)
    strategy_params: dict[str, dict[str, object]] = field(default_factory=dict)
    ml_model_id: str | None = None
    news_sources: list[str] = field(default_factory=list)
    news_languages: list[str] = field(default_factory=lambda: ["en"])
    max_allocation_pct: float = 0.30
    # D-09 / LIQ-07: cap on simultaneous open positions per segment. Additive -- the
    # default of 10 preserves current behaviour at every existing construction site so a
    # wider liquid universe cannot fragment capital across too many tiny positions.
    max_concurrent_positions: int = 10
    trading_hours: str = ""
    enabled: bool = True


# Default segment definitions -- loaded at startup, overridable via DB
DEFAULT_SEGMENTS: list[SegmentConfig] = [
    # ── US segments — frozen 2026-05-25 ─────────────────────────────────
    # Kept in DEFAULT_SEGMENTS for history; filtered out by every consumer
    # via `enabled=False`. Reason: no foreign card from RF + non-RF exchanges
    # are unavailable to RF residents in the current regulatory regime.
    SegmentConfig(
        segment_id="us_tech",
        market="us",
        broker="alpaca",
        currency="USD",
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"],
        active_strategies=["momentum", "mean_reversion", "event_driven"],
        news_languages=["en"],
        max_allocation_pct=0.30,
        trading_hours="14:30-21:00 UTC",
        enabled=False,
    ),
    SegmentConfig(
        segment_id="us_healthcare",
        market="us",
        broker="alpaca",
        currency="USD",
        symbols=["JNJ", "PFE", "UNH", "ABBV", "MRK"],
        active_strategies=["event_driven", "mean_reversion", "momentum"],
        news_languages=["en"],
        max_allocation_pct=0.25,
        trading_hours="14:30-21:00 UTC",
        enabled=False,
    ),
    SegmentConfig(
        segment_id="us_finance",
        market="us",
        broker="alpaca",
        currency="USD",
        symbols=["JPM", "BAC", "GS", "MS", "WFC"],
        active_strategies=["mean_reversion", "momentum", "event_driven"],
        news_languages=["en"],
        max_allocation_pct=0.25,
        trading_hours="14:30-21:00 UTC",
        enabled=False,
    ),
    SegmentConfig(
        segment_id="us_broad",
        market="us",
        broker="alpaca",
        currency="USD",
        symbols=["SPY", "QQQ", "DIA"],
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["en"],
        max_allocation_pct=0.30,
        trading_hours="14:30-21:00 UTC",
        enabled=False,
    ),
    SegmentConfig(
        segment_id="ru_energy",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_energy", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_energy"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "event_driven", "mean_reversion"],
        news_languages=["ru", "en"],
        max_allocation_pct=0.25,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_tech",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_tech", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_tech"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion", "event_driven"],
        news_languages=["ru"],
        max_allocation_pct=0.20,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_finance",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_finance", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_finance"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["mean_reversion", "event_driven", "momentum"],
        news_languages=["ru", "en"],
        max_allocation_pct=0.25,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_ofz_pd",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        instrument_type="bond",
        symbols=[
            "SU26239RMFS2",
            "SU26241RMFS8",
            "SU26243RMFS4",
            "SU26244RMFS2",
            "SU26246RMFS7",
            "SU26252RMFS5",
            "SU26253RMFS3",
        ],
        active_strategies=["bond_duration_rotation"],
        news_languages=["ru"],
        max_allocation_pct=0.30,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_ofz_pk",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        instrument_type="bond",
        symbols=["SU29007RMFS0", "SU29008RMFS8", "SU29009RMFS6", "SU29010RMFS4"],
        active_strategies=["bond_carry"],
        news_languages=["ru"],
        max_allocation_pct=0.50,
        trading_hours="07:00-15:40 UTC",
    ),
    # ── Additional MOEX sector segments ──────────────────────────────────
    SegmentConfig(
        segment_id="ru_metals",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_metals", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_metals"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru", "en"],
        max_allocation_pct=0.20,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_consumer",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_consumer", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_consumer"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.15,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_telecom",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_telecom", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_telecom"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.15,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_utilities",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_utilities", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_utilities"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.15,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_construction",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_construction", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_construction"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.10,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_chemicals",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_chemicals", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_chemicals"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.10,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_transport",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=select_segment_symbols(
            "ru_transport", bootstrap=_BOOTSTRAP_SEGMENT_SYMBOLS["ru_transport"]
        ),  # LIVE seam (D-07): selector-fed, prior-list bootstrap pre-snapshot
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.10,
        trading_hours="07:00-15:40 UTC",
    ),
]
