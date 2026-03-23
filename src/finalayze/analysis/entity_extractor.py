"""LLM-based MOEX ticker extraction from Russian news text (Layer 3).

Uses an LLM to identify company mentions in Russian-language financial news
and map them to MOEX ticker symbols.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient
    from finalayze.core.schemas import NewsArticle

log = structlog.get_logger(__name__)

# Circuit breaker: stop calling LLM after N consecutive failures
_CIRCUIT_BREAKER_THRESHOLD = 5
_CIRCUIT_BREAKER_RESET_SECONDS = 300  # 5 min cooldown before retrying

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Known MOEX tickers from all ru_* segments
_VALID_TICKERS: frozenset[str] = frozenset(
    {
        "SBER",
        "GAZP",
        "LKOH",
        "YDEX",
        "ROSN",
        "GMKN",
        "MGNT",
        "VTBR",
        "MTSS",
        "PLZL",
        "NVTK",
        "SNGS",
        "TATN",
        "CHMF",
        "NLMK",
        "MAGN",
        "ALRS",
        "IRAO",
        "RUAL",
        "MOEX",
        "TCSG",
        "OZON",
        "PIKK",
        "AFKS",
        "TRNFP",
        "AFLT",
        "MSNG",
        "HYDR",
        "PHOR",
    }
)

# Regex to strip markdown code fences from LLM output
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class EntityExtractor:
    """Extracts MOEX ticker symbols from Russian news articles via LLM.

    Filters extracted tickers against a known set of valid MOEX symbols.
    Returns an empty list on malformed LLM responses.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._system_prompt = self._load_prompt()
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0  # monotonic timestamp

    async def extract(self, article: NewsArticle) -> list[str]:
        """Extract MOEX ticker symbols mentioned in the article.

        Args:
            article: A Russian-language news article.

        Returns:
            List of valid MOEX ticker symbols found in the article.
            Empty list if no companies identified or on parse errors.
        """
        # Circuit breaker: skip LLM calls when circuit is open
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            now = time.monotonic()
            if now < self._circuit_open_until:
                return []
            # Half-open: allow one retry attempt
            log.info("entity_extraction_circuit_half_open")

        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"

        try:
            raw = await self._llm.complete(user_prompt, self._system_prompt)
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures == _CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_RESET_SECONDS
                log.warning(
                    "entity_extraction_circuit_opened",
                    failures=self._consecutive_failures,
                    cooldown_seconds=_CIRCUIT_BREAKER_RESET_SECONDS,
                    exc_info=True,
                )
            else:
                # Quiet: individual failures are expected during rate limiting
                log.debug("entity_extraction_llm_failed", article_url=article.url)
            return []

        # Success: reset circuit breaker
        if self._consecutive_failures > 0:
            log.info(
                "entity_extraction_circuit_closed",
                previous_failures=self._consecutive_failures,
            )
        self._consecutive_failures = 0
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> list[str]:
        """Parse LLM JSON response and filter to valid MOEX tickers."""
        # Strip markdown code fences if present
        match = _CODE_FENCE_RE.match(raw.strip())
        if match:
            raw = match.group(1)

        try:
            data = json.loads(raw)
            tickers = data.get("tickers", [])
        except (json.JSONDecodeError, AttributeError, TypeError):
            log.warning("entity_extraction_parse_failed", raw_response=raw[:200])
            return []

        return [t for t in tickers if t in _VALID_TICKERS]

    def _load_prompt(self) -> str:
        """Load the entity extraction system prompt from disk."""
        prompt_path = _PROMPTS_DIR / "entity_extraction.txt"
        return prompt_path.read_text(encoding="utf-8").strip()
