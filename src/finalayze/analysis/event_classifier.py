"""News event type classifier using an LLM client (Layer 3)."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient
    from finalayze.core.schemas import NewsArticle

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_LLM_TIMEOUT_SECONDS = 5.0

_log = structlog.get_logger()


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


class EventClassifierResult(BaseModel):
    """Structured LLM response for event classification."""

    model_config = ConfigDict(frozen=True)

    event_types: list[str] = []


class EventClassifier:
    """Classifies news articles into EventType categories using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system: str | None = None

    def _load_system(self) -> str:
        if self._system is None:
            self._system = (_PROMPTS_DIR / "classify_event.txt").read_text(encoding="utf-8").strip()
        return self._system

    @staticmethod
    def _resolve_event_type(result: EventClassifierResult) -> EventType:
        """Extract the first recognised EventType from a parsed result."""
        for et in result.event_types:
            candidate = str(et).strip().lower()
            if candidate in _PROMPT_TO_EVENT_TYPE:
                return _PROMPT_TO_EVENT_TYPE[candidate]
        return EventType.OTHER

    async def classify(self, article: NewsArticle) -> EventType:
        """Classify a news article into an EventType.

        Args:
            article: The news article to classify.

        Returns:
            EventType value. Returns ``EventType.OTHER`` on timeouts or parse errors.
        """
        system = self._load_system()
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"

        try:
            result = await asyncio.wait_for(
                self._llm.parse_structured(user_prompt, system, EventClassifierResult),
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            return self._resolve_event_type(result)
        except TimeoutError:
            _log.warning("llm_timeout", analyzer="EventClassifier", article_url=article.url)
            return EventType.OTHER
        except Exception:
            _log.warning("llm_parse_error", analyzer="EventClassifier", article_url=article.url)
            return EventType.OTHER
