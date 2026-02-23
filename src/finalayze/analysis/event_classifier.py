"""News event type classifier using an LLM client (Layer 3)."""

from __future__ import annotations

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


class EventClassifier:
    """Classifies news articles into EventType categories using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system: str | None = None

    def _load_system(self) -> str:
        if self._system is None:
            self._system = (_PROMPTS_DIR / "classify_event.txt").read_text(encoding="utf-8").strip()
        return self._system

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
        value = raw.strip().lower()
        try:
            return EventType(value)
        except ValueError:
            return EventType.OTHER
