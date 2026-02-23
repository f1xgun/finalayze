"""Impact estimator — routes news to affected segments (Layer 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from finalayze.analysis.event_classifier import EventType

if TYPE_CHECKING:
    from finalayze.core.schemas import SentimentResult

_PRIMARY = 1.0
_SECONDARY = 0.3

# EventType → {primary segments, secondary segments}
_EVENT_ROUTING: dict[EventType, tuple[list[str], list[str]]] = {
    EventType.OIL_PRICE: (["ru_energy"], ["ru_blue_chips"]),
    EventType.CBR_RATE: (["ru_finance"], ["ru_blue_chips"]),
    EventType.SANCTIONS: (["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"], []),
    EventType.FDA: (["us_healthcare"], []),
    EventType.EARNINGS: ([], []),  # handled by caller with specific symbol
    EventType.MACRO: ([], []),  # global event — caller uses scope="global"
    EventType.GEOPOLITICAL: ([], []),  # global event
    EventType.OTHER: ([], []),
}


@dataclass(frozen=True)
class SegmentImpact:
    """Impact of a news event on a specific segment."""

    segment_id: str
    weight: float  # 1.0 = primary, 0.3 = secondary
    sentiment: float  # from SentimentResult


def _build_impacts_by_prefix(prefix: str, active_segments: list[str]) -> dict[str, float]:
    """Return primary-weight impacts for all segments matching prefix."""
    return {seg: _PRIMARY for seg in active_segments if seg.startswith(prefix)}


def _build_sector_impacts(event: EventType, active_segments: list[str]) -> dict[str, float]:
    """Return primary/secondary impacts for sector-scoped events."""
    primary_segs, secondary_segs = _EVENT_ROUTING.get(event, ([], []))
    impacts: dict[str, float] = {}
    for seg in primary_segs:
        if seg in active_segments:
            impacts[seg] = _PRIMARY
    for seg in secondary_segs:
        if seg in active_segments and seg not in impacts:
            impacts[seg] = _SECONDARY
    return impacts


class ImpactEstimator:
    """Routes news impact to affected segments based on scope and event type.

    No LLM needed — pure rule-based routing.
    """

    def estimate(
        self,
        scope: str,
        event: EventType,
        sentiment: SentimentResult,
        active_segments: list[str],
    ) -> list[SegmentImpact]:
        """Estimate which segments are affected and by how much.

        Args:
            scope: Geographic scope — "global", "us", "russia", or "sector".
            event: Classified event type.
            sentiment: Sentiment result from NewsAnalyzer.
            active_segments: List of segment IDs currently active in the system.

        Returns:
            List of SegmentImpact — may be empty if no matching segments.
        """
        sent = sentiment.sentiment

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
