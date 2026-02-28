"""Financial fact-checker using an LLM client (Layer 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.analysis.llm_client import LLMClient

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_DEFAULT_CREDIBILITY = 0.5


class FactChecker:
    """Cross-references news claims against multiple sources using an LLM.

    The fact_check.txt prompt instructs the model to return a JSON object
    containing a ``credibility_score`` field in [0, 1] along with lists of
    confirmed claims, unverified claims, and contradictions.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client
        self._system: str | None = None

    def _load_system(self) -> str:
        if self._system is None:
            self._system = (_PROMPTS_DIR / "fact_check.txt").read_text(encoding="utf-8").strip()
        return self._system

    async def check(self, claim: str, sources: list[str]) -> float:
        """Cross-reference *claim* against *sources* and return a credibility score.

        Args:
            claim: The primary news claim or article text to fact-check.
            sources: A list of related article texts from other sources.

        Returns:
            Credibility score in ``[0.0, 1.0]``.  Higher means more credible.
            Returns ``0.5`` when the LLM response cannot be parsed.
        """
        system = self._load_system()
        sources_block = "\n\n---\n\n".join(sources) if sources else "(no additional sources)"
        user_prompt = f"Primary claim:\n{claim}\n\nRelated sources:\n{sources_block}"
        raw = await self._client.complete(user_prompt, system)
        return self._parse_credibility(raw)

    @staticmethod
    def _parse_credibility(raw: str) -> float:
        """Extract ``credibility_score`` from the LLM JSON response."""
        try:
            data = json.loads(raw.strip())
            score = float(data.get("credibility_score", _DEFAULT_CREDIBILITY))
            # Clamp to [0, 1] in case the model returns an out-of-range value
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            return _DEFAULT_CREDIBILITY
