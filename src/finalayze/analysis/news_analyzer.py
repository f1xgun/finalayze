"""News sentiment analyzer using an LLM client (Layer 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.core.schemas import NewsArticle, SentimentResult

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_FALLBACK = SentimentResult(sentiment=0.0, confidence=0.0, reasoning="parse_error")


class NewsAnalyzer:
    """Analyzes news articles for financial sentiment using an LLM.

    Selects EN or RU prompt based on article language.
    Falls back to neutral sentiment on parse errors.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._prompts: dict[str, str] = {}

    def _load_prompt(self, language: str) -> str:
        """Load and cache the system prompt for the given language."""
        if language not in self._prompts:
            lang = language if language in ("en", "ru") else "en"
            prompt_path = _PROMPTS_DIR / f"sentiment_{lang}.txt"
            self._prompts[language] = prompt_path.read_text(encoding="utf-8").strip()
        return self._prompts[language]

    async def analyze(self, article: NewsArticle) -> SentimentResult:
        """Analyze an article and return a SentimentResult.

        Args:
            article: The news article to analyze.

        Returns:
            SentimentResult with sentiment [-1.0, 1.0], confidence, and reasoning.
            Returns neutral result (0.0 sentiment, 0.0 confidence) on parse errors.
        """
        system = self._load_prompt(article.language)
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"

        raw = await self._llm.complete(user_prompt, system)

        try:
            data = json.loads(raw)
            return SentimentResult(
                sentiment=float(data["sentiment"]),
                confidence=float(data["confidence"]),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return _FALLBACK
