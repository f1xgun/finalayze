"""Unified news impact analyzer -- single LLM call per article (Layer 3).

Replaces the 2-call EntityExtractor + CombinedAnalyzer pipeline with a single
intelligent call that returns event_type, sentiment, sectors, and tickers.
Implements NEWS-05 (sector-aware impact) and NEWS-09 (single LLM call).
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict

from finalayze.analysis.entity_extractor import _VALID_TICKERS
from finalayze.analysis.event_classifier import _PROMPT_TO_EVENT_TYPE, EventType

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient
    from finalayze.core.schemas import NewsArticle

log = structlog.get_logger(__name__)

# Circuit breaker constants (same as EntityExtractor)
_CIRCUIT_BREAKER_THRESHOLD = 5
_CIRCUIT_BREAKER_RESET_SECONDS = 300  # 5 min cooldown

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Regex to strip markdown code fences from LLM output
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class SectorImpactDetail(BaseModel):
    """Impact detail for a single sector."""

    model_config = ConfigDict(frozen=True)

    sector: str
    direction: int  # -1 or +1
    magnitude: float  # 0.0-1.0
    reasoning: str


class NewsImpactResult(BaseModel):
    """Complete impact analysis result from a single LLM call."""

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    sentiment: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    affected_sectors: list[SectorImpactDetail]
    direct_tickers: list[str]


_FALLBACK_RESULT = NewsImpactResult(
    event_type=EventType.OTHER,
    sentiment=0.0,
    confidence=0.0,
    reasoning="parse_error",
    affected_sectors=[],
    direct_tickers=[],
)


class NewsImpactAnalyzer:
    """Analyzes news articles for market impact using a single LLM call.

    Produces event_type, sentiment, confidence, affected sectors, and
    directly mentioned tickers from one LLM invocation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._prompts: dict[str, str] = {}
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0  # monotonic timestamp

    async def analyze(self, article: NewsArticle) -> NewsImpactResult:
        """Analyze a news article for market impact.

        Args:
            article: The news article to analyze.

        Returns:
            NewsImpactResult with event_type, sentiment, confidence,
            affected_sectors, and direct_tickers.
            Returns fallback on parse errors or circuit breaker open.
        """
        # Circuit breaker check
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            now = time.monotonic()
            if now < self._circuit_open_until:
                log.debug(
                    "news_impact_circuit_skipped",
                    article_url=article.url,
                    remaining_seconds=round(self._circuit_open_until - now, 1),
                )
                return _FALLBACK_RESULT
            # Half-open: allow one retry
            log.info("news_impact_circuit_half_open")

        system_prompt = self._load_prompt(article.language)
        user_prompt = f"Title: {article.title}\n\nContent: {article.content}"

        try:
            raw = await self._llm.complete(
                user_prompt, system_prompt, json_mode=True, max_tokens=2048
            )
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures == _CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_RESET_SECONDS
                log.warning(
                    "news_impact_circuit_opened",
                    failures=self._consecutive_failures,
                    cooldown_seconds=_CIRCUIT_BREAKER_RESET_SECONDS,
                    exc_info=True,
                )
            else:
                log.debug("news_impact_llm_failed", article_url=article.url)
            return _FALLBACK_RESULT

        # Success: reset circuit breaker
        if self._consecutive_failures > 0:
            log.info(
                "news_impact_circuit_closed",
                previous_failures=self._consecutive_failures,
            )
        self._consecutive_failures = 0
        result = self._parse_response(raw)
        if result is _FALLBACK_RESULT:
            log.warning(
                "news_impact_llm_parse_fallback",
                article_url=article.url,
            )
        else:
            log.debug(
                "news_impact_llm_success",
                article_url=article.url,
                event_type=result.event_type,
                sentiment=round(result.sentiment, 3),
                confidence=round(result.confidence, 3),
            )
        return result

    def _load_prompt(self, language: str) -> str:
        """Load and cache prompt for the given language."""
        lang = language if language in ("ru", "en") else "en"
        if lang not in self._prompts:
            prompt_path = _PROMPTS_DIR / f"analyze_impact_{lang}.txt"
            self._prompts[lang] = prompt_path.read_text(encoding="utf-8").strip()
        return self._prompts[lang]

    def _parse_response(self, raw: str) -> NewsImpactResult:  # noqa: PLR0912
        """Parse LLM JSON response into NewsImpactResult.

        Handles common LLM output issues:
        - Markdown code fences around JSON
        - JavaScript-style comments (// ...) inside JSON
        - Extra text after the JSON object (e.g., "Примечание: ...")
        - Double-escaped JSON strings ("{\"key\": ...}")
        - Truncated JSON (attempts to close braces)
        """
        stripped = raw.strip()

        # Strip code fences
        match = _CODE_FENCE_RE.match(stripped)
        if match:
            stripped = match.group(1).strip()

        # Unescape double-escaped JSON: "{\"key\": ...}" -> {"key": ...}
        if stripped.startswith('"') and '\\"' in stripped:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                stripped = json.loads(stripped)  # parse the outer string

        # Remove JavaScript-style single-line comments (// ...)
        stripped = re.sub(r"//[^\n]*", "", stripped)

        # Extract first JSON object — ignore trailing text after closing brace
        brace_start = stripped.find("{")
        if brace_start >= 0:
            depth, i = 0, brace_start
            in_string = False
            for i, ch in enumerate(stripped[brace_start:], start=brace_start):
                if ch == '"' and (i == 0 or stripped[i - 1] != "\\"):
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            stripped = stripped[brace_start : i + 1]
                            break
            else:
                # Truncated JSON — try to close open braces/brackets
                stripped = stripped[brace_start:]
                stripped += "]" * (stripped.count("[") - stripped.count("]"))
                stripped += "}" * (stripped.count("{") - stripped.count("}"))

        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            log.warning("news_impact_parse_failed", raw_response=raw[:200])
            return _FALLBACK_RESULT

        if not isinstance(data, dict):
            log.warning("news_impact_parse_failed", raw_response=raw[:200])
            return _FALLBACK_RESULT

        # Event type
        event_type_str = str(data.get("event_type", "other")).strip().lower()
        event_type = _PROMPT_TO_EVENT_TYPE.get(event_type_str, EventType.OTHER)

        # Sentiment and confidence with clamping
        sentiment = _clamp(float(data.get("sentiment", 0.0)), -1.0, 1.0)
        confidence = _clamp(float(data.get("confidence", 0.0)), 0.0, 1.0)
        reasoning = str(data.get("reasoning", ""))

        # Affected sectors (guard against null from LLM)
        affected_sectors: list[SectorImpactDetail] = []
        for entry in data.get("affected_sectors") or []:
            if not isinstance(entry, dict):
                continue
            try:
                direction_raw = float(entry.get("direction", 1))
                direction = 1 if direction_raw >= 0 else -1
                magnitude = _clamp(float(entry.get("magnitude", 0.0)), 0.0, 1.0)
                affected_sectors.append(
                    SectorImpactDetail(
                        sector=str(entry.get("sector", "")),
                        direction=direction,
                        magnitude=magnitude,
                        reasoning=str(entry.get("reasoning", "")),
                    )
                )
            except (ValueError, TypeError):
                continue

        # Direct tickers filtered against valid set (guard against null from LLM)
        raw_tickers = data.get("direct_tickers") or []
        direct_tickers = [t for t in raw_tickers if isinstance(t, str) and t in _VALID_TICKERS]

        return NewsImpactResult(
            event_type=event_type,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=reasoning,
            affected_sectors=affected_sectors,
            direct_tickers=direct_tickers,
        )


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))
