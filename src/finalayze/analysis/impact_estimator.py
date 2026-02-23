"""Impact estimator — routes news to affected segments (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from finalayze.analysis.event_classifier import EventType

if TYPE_CHECKING:
    from finalayze.core.schemas import NewsArticle, SentimentResult

_PRIMARY = 1.0
_SECONDARY_OIL = 0.5
_SECONDARY = 0.3
_HALF = 0.5
_GEOPOLITICAL_WEIGHT = 0.3

# EventType → {primary segments, secondary segments}
# For MACRO and GEOPOLITICAL, routing is handled specially in _build_sector_impacts
_EVENT_ROUTING: dict[EventType, tuple[list[str], list[str]]] = {
    EventType.OIL_PRICE: (["ru_energy"], ["ru_blue_chips"]),
    EventType.CBR_RATE: (["ru_finance"], ["ru_blue_chips"]),
    EventType.SANCTIONS: (["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"], []),
    EventType.FDA: (["us_healthcare"], []),
    EventType.EARNINGS: ([], []),  # handled by caller with specific symbol
    EventType.MACRO: ([], []),  # all segments at 0.5 — handled separately
    EventType.GEOPOLITICAL: ([], []),  # all segments at 0.3 — handled separately
    EventType.OTHER: ([], []),
}


class SegmentImpact(BaseModel):
    """Impact of a news event on a specific segment."""

    model_config = ConfigDict(frozen=True)

    segment_id: str
    weight: float  # 1.0 = primary, secondary varies by event
    sentiment: float  # from SentimentResult


def _build_impacts_by_prefix(prefix: str, active_segments: list[str]) -> dict[str, float]:
    """Return primary-weight impacts for all segments matching prefix."""
    return {seg: _PRIMARY for seg in active_segments if seg.startswith(prefix)}


def _build_sector_impacts(event: EventType, active_segments: list[str]) -> dict[str, float]:
    """Return primary/secondary impacts for sector-scoped events."""
    # MACRO and GEOPOLITICAL affect all active segments with fixed weights
    if event == EventType.MACRO:
        return dict.fromkeys(active_segments, _HALF)
    if event == EventType.GEOPOLITICAL:
        return dict.fromkeys(active_segments, _GEOPOLITICAL_WEIGHT)

    primary_segs, secondary_segs = _EVENT_ROUTING.get(event, ([], []))
    impacts: dict[str, float] = {}
    for seg in primary_segs:
        if seg in active_segments:
            impacts[seg] = _PRIMARY
    for seg in secondary_segs:
        if seg in active_segments and seg not in impacts:
            # OIL_PRICE secondary weight is 0.5; all others use 0.3
            secondary_weight = _SECONDARY_OIL if event == EventType.OIL_PRICE else _SECONDARY
            impacts[seg] = secondary_weight
    return impacts


class ImpactEstimator:
    """Routes news impact to affected segments based on scope and event type.

    No LLM needed — pure rule-based routing.
    """

    def estimate(
        self,
        article: NewsArticle,
        event: EventType,
        sentiment: SentimentResult,
        active_segments: list[str],
    ) -> list[SegmentImpact]:
        """Estimate which segments are affected and by how much.

        Args:
            article: The news article being analyzed (scope is extracted from article.scope).
            event: Classified event type.
            sentiment: Sentiment result from NewsAnalyzer.
            active_segments: List of segment IDs currently active in the system.

        Returns:
            List of SegmentImpact — may be empty if no matching segments.
        """
        sent = sentiment.sentiment
        scope = article.scope

        if scope == "global":
            impacts = dict.fromkeys(active_segments, _PRIMARY)
        elif scope == "us":
            impacts = _build_impacts_by_prefix("us_", active_segments)
        elif scope == "russia":
            impacts = _build_impacts_by_prefix("ru_", active_segments)
        elif scope == "sector":
            impacts = _build_sector_impacts(event, active_segments)
        else:
            impacts = {}

        return [
            SegmentImpact(segment_id=seg, weight=w, sentiment=sent * w)
            for seg, w in impacts.items()
        ]
