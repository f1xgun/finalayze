"""Live geopolitical-risk monitor (Layer 5) — aggregates the sentiment store.

Reads rolling per-ticker sentiment across the active MOEX equity universe from the
live ``SentimentStore`` continuous aggregate, folds it into a single market-wide
:class:`GeoRiskInputs`, and runs the pure
:func:`finalayze.analysis.geopolitical_risk.assess_geopolitical_risk` brain.

Forward-only and ADVISORY (see the analysis module's disclaimer): it informs an
alert / dashboard, it does NOT auto-trade. Real money stays behind the hard-stop.

The reader is a narrow :class:`SentimentReader` Protocol (``SentimentStore``
satisfies it), so the aggregation is unit-testable with a fake — no DB required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from finalayze.analysis.geopolitical_risk import (
    GeoRiskInputs,
    assess_geopolitical_risk,
)

if TYPE_CHECKING:
    from finalayze.analysis.geopolitical_risk import GeoRiskAssessment
    from finalayze.data.sentiment_store import SentimentRow

_DEFAULT_WINDOW = "7d"
_MOEX = "moex"

# Market-sentiment proxy: the IMOEX heavyweights dominate Russian market news flow,
# so their aggregate sentiment is a faithful, cheap stand-in for market-wide
# geopolitical stress (avoids enumerating the full universe per call). Names absent
# from the sentiment store simply contribute no rows.
MOEX_BELLWETHERS: list[str] = [
    "SBER",
    "GAZP",
    "LKOH",
    "GMKN",
    "NVTK",
    "ROSN",
    "TATN",
    "PLZL",
    "VTBR",
    "MGNT",
    "MOEX",
    "MTSS",
    "SNGS",
    "CHMF",
    "NLMK",
]


class SentimentReader(Protocol):
    """The slice of ``SentimentStore`` the monitor needs (for testability)."""

    async def get_rolling(
        self, ticker: str, *, window: str = ..., market_id: str = ...
    ) -> list[SentimentRow]: ...


async def aggregate_inputs(
    reader: SentimentReader,
    tickers: list[str],
    *,
    window: str = _DEFAULT_WINDOW,
    market_id: str = _MOEX,
    sanctions_event_count: int = 0,
    geopolitical_event_count: int = 0,
) -> GeoRiskInputs:
    """Fold per-ticker rolling sentiment into one market-wide ``GeoRiskInputs``.

    ``mean_sentiment`` is article-count-weighted across every ticker/bucket;
    ``article_volume`` is the total article count over the window. Event counts
    are passed through (best-effort from an event-type-aware source; default 0 —
    the score degrades gracefully to sentiment + volume).
    """
    weighted_sum = 0.0
    total_weight = 0
    total_articles = 0
    for ticker in tickers:
        rows = await reader.get_rolling(ticker, window=window, market_id=market_id)
        for row in rows:
            total_articles += row.article_count
            if row.avg_score is not None and row.article_count > 0:
                weighted_sum += row.avg_score * row.article_count
                total_weight += row.article_count
    mean_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
    return GeoRiskInputs(
        mean_sentiment=mean_sentiment,
        article_volume=total_articles,
        sanctions_event_count=sanctions_event_count,
        geopolitical_event_count=geopolitical_event_count,
    )


async def assess_live(
    reader: SentimentReader,
    tickers: list[str],
    *,
    window: str = _DEFAULT_WINDOW,
    market_id: str = _MOEX,
    sanctions_event_count: int = 0,
    geopolitical_event_count: int = 0,
) -> GeoRiskAssessment:
    """Aggregate the live sentiment store and return the advisory assessment."""
    inputs = await aggregate_inputs(
        reader,
        tickers,
        window=window,
        market_id=market_id,
        sanctions_event_count=sanctions_event_count,
        geopolitical_event_count=geopolitical_event_count,
    )
    return assess_geopolitical_risk(inputs)
