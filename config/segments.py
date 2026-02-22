"""Stock segment definitions.

See docs/design/SEGMENTS.md for the full segment system design.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for a stock segment."""

    segment_id: str
    market: str
    broker: str
    currency: str
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
        symbols=["SBER", "GAZP", "LKOH", "GMKN"],
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
        symbols=["ROSN", "TATN", "NVTK"],
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
        symbols=["YNDX", "OZON", "VKCO"],
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
        symbols=["SBER", "VTBR", "TCSG"],
        active_strategies=["mean_reversion", "event_driven", "momentum"],
        news_languages=["ru", "en"],
        max_allocation_pct=0.25,
        trading_hours="07:00-15:40 UTC",
    ),
]
