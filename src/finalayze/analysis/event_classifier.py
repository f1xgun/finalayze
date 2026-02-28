"""News event type classifier using an LLM client (Layer 3)."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient
    from finalayze.core.schemas import NewsArticle

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class EventType(StrEnum):
    """Classification categories for news events."""

    EARNINGS = "earnings"
    FDA = "fda"
    MACRO = "macro"
    GEOPOLITICAL = "geopolitical"
    CBR_RATE = "cbr_rate"
    OIL_PRICE = "oil_price"
    SANCTIONS = "sanctions"
    OTHER = "other"


# Map prompt event types (extended vocabulary) to our internal EventType enum.
_PROMPT_TO_EVENT_TYPE: dict[str, EventType] = {
    "earnings": EventType.EARNINGS,
    "fda": EventType.FDA,
    "macro": EventType.MACRO,
    "geopolitical": EventType.GEOPOLITICAL,
    "cbr_rate": EventType.CBR_RATE,
    "oil_price": EventType.OIL_PRICE,
    "sanctions": EventType.SANCTIONS,
    "regulatory": EventType.OTHER,
    "merger_acquisition": EventType.OTHER,
    "product_launch": EventType.OTHER,
    "interest_rate": EventType.MACRO,
    "opec": EventType.OIL_PRICE,
    "commodity_price": EventType.OTHER,
    "clinical_trial": EventType.FDA,
    "bankruptcy": EventType.OTHER,
    "ipo": EventType.OTHER,
    "dividend": EventType.EARNINGS,
    "stock_split": EventType.OTHER,
    "other": EventType.OTHER,
}


class EventClassifier:
    """Classifies news articles into EventType categories using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system: str | None = None

    def _load_system(self) -> str:
        if self._system is None:
            self._system = (_PROMPTS_DIR / "classify_event.txt").read_text(encoding="utf-8").strip()
        return self._system

    def _parse_response(self, raw: str) -> EventType:
        """Parse the LLM response — handles both JSON and plain-text formats.

        The classify_event.txt prompt instructs the model to return a JSON object
        with an ``event_types`` list.  This method extracts the first recognised
        event type from that list.  If the response cannot be parsed as JSON, it
        falls back to treating the raw string as a bare event-type label (backwards
        compatibility).  (#143)
        """
        stripped = raw.strip()
        # Try JSON parsing first (expected format from the prompt)
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                event_types_raw = data.get("event_types", [])
                if isinstance(event_types_raw, list):
                    for et in event_types_raw:
                        candidate = str(et).strip().lower()
                        if candidate in _PROMPT_TO_EVENT_TYPE:
                            return _PROMPT_TO_EVENT_TYPE[candidate]
                # Fallback: try "event_type" single-value field
                single = data.get("event_type", "")
                candidate = str(single).strip().lower()
                if candidate in _PROMPT_TO_EVENT_TYPE:
                    return _PROMPT_TO_EVENT_TYPE[candidate]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat the raw response as a plain event-type string
        value = stripped.lower()
        try:
            return EventType(value)
        except ValueError:
            return _PROMPT_TO_EVENT_TYPE.get(value, EventType.OTHER)

    async def classify(self, article: NewsArticle) -> EventType:
        """Classify a news article into an EventType.

        Args:
            article: The news article to classify.

        Returns:
            EventType value. Returns ``EventType.OTHER`` for unrecognised responses.
        """
        system = self._load_system()
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"
        raw = await self._llm.complete(user_prompt, system)
        return self._parse_response(raw)
