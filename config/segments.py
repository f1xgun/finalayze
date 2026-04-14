"""Segment definitions (stocks, bonds, ETFs).

See docs/design/SEGMENTS.md for the full segment system design.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    trading_hours: str = ""


# Default segment definitions -- loaded at startup, overridable via DB
DEFAULT_SEGMENTS: list[SegmentConfig] = [
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
    ),
    SegmentConfig(
        segment_id="ru_blue_chips",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=["SBER", "LKOH", "GMKN"],
        active_strategies=["momentum", "event_driven", "mean_reversion"],
        news_languages=["ru", "en"],
        max_allocation_pct=0.30,
        trading_hours="07:00-15:40 UTC",
    ),
    SegmentConfig(
        segment_id="ru_energy",
        market="moex",
        broker="tinkoff",
        currency="RUB",
        symbols=["ROSN", "TATN", "NVTK", "SIBN", "TATNP", "TRNFP"],
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
        symbols=["YDEX", "OZON", "VKCO", "HEAD", "POSI"],
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
        symbols=["SBER", "T", "CBOM", "BSPB", "MOEX", "VTBR"],
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
        symbols=["GMKN", "CHMF", "NLMK", "MAGN", "PLZL", "RUAL", "MTLR"],
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
        symbols=["MGNT", "X5", "LENT"],
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
        symbols=["MTSS", "RTKM"],
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
        symbols=["HYDR", "FEES", "MSNG", "UPRO"],
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
        symbols=["PIKK", "SMLT"],
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
        symbols=["PHOR", "AKRN"],
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
        symbols=["AFLT", "FLOT", "NMTP"],
        active_strategies=["momentum", "mean_reversion"],
        news_languages=["ru"],
        max_allocation_pct=0.10,
        trading_hours="07:00-15:40 UTC",
    ),
]
