"""Unit tests for FactChecker (#179)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from finalayze.analysis.fact_checker import FactChecker

_CLAIM = "The Federal Reserve raised interest rates by 25 basis points."
_SOURCE_1 = "Fed hikes rates 0.25% — Bloomberg"
_SOURCE_2 = "Federal Reserve increases benchmark rate — Reuters"

# Credibility bounds (no magic numbers per ruff PLR2004)
_MIN_CREDIBILITY = 0.0
_MAX_CREDIBILITY = 1.0
_DEFAULT_CREDIBILITY = 0.5
_HIGH_CREDIBILITY = 0.9
_LOW_CREDIBILITY = 0.2


class TestFactCheckerInit:
    def test_instantiation_succeeds(self) -> None:
        mock_llm = AsyncMock()
        checker = FactChecker(llm_client=mock_llm)
        assert checker is not None


class TestFactCheckerCheck:
    @pytest.mark.asyncio
    async def test_check_returns_float_in_range(self) -> None:
        """check() must return a float in [0.0, 1.0]."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "credibility_score": _HIGH_CREDIBILITY,
                "confirmed_claims": ["Fed raised rates"],
                "unverified_claims": [],
                "contradictions": [],
                "reasoning": "Multiple sources confirm.",
            }
        )
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [_SOURCE_1, _SOURCE_2])

        assert isinstance(score, float)
        assert _MIN_CREDIBILITY <= score <= _MAX_CREDIBILITY

    @pytest.mark.asyncio
    async def test_check_extracts_credibility_score_from_json(self) -> None:
        """check() must return the credibility_score value from the LLM JSON response."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "credibility_score": _HIGH_CREDIBILITY,
                "confirmed_claims": [],
                "unverified_claims": [],
                "contradictions": [],
                "reasoning": "Confirmed.",
            }
        )
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [_SOURCE_1])
        assert score == pytest.approx(_HIGH_CREDIBILITY)

    @pytest.mark.asyncio
    async def test_check_low_credibility_score(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps(
            {
                "credibility_score": _LOW_CREDIBILITY,
                "confirmed_claims": [],
                "unverified_claims": [_CLAIM],
                "contradictions": ["Source says rates unchanged"],
                "reasoning": "Contradicted by primary source.",
            }
        )
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [_SOURCE_1])
        assert score == pytest.approx(_LOW_CREDIBILITY)

    @pytest.mark.asyncio
    async def test_check_invalid_json_returns_default(self) -> None:
        """When LLM returns non-JSON, check() falls back to 0.5."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "This is not JSON."
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [_SOURCE_1])
        assert score == pytest.approx(_DEFAULT_CREDIBILITY)

    @pytest.mark.asyncio
    async def test_check_out_of_range_score_clamped(self) -> None:
        """Scores outside [0, 1] returned by the LLM must be clamped."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({"credibility_score": 1.5})
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [])
        assert score == pytest.approx(_MAX_CREDIBILITY)

    @pytest.mark.asyncio
    async def test_check_no_sources_still_calls_llm(self) -> None:
        """check() with an empty sources list must still call the LLM."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({"credibility_score": _DEFAULT_CREDIBILITY})
        checker = FactChecker(llm_client=mock_llm)
        score = await checker.check(_CLAIM, [])
        mock_llm.complete.assert_called_once()
        assert _MIN_CREDIBILITY <= score <= _MAX_CREDIBILITY

    @pytest.mark.asyncio
    async def test_check_passes_claim_and_sources_to_llm(self) -> None:
        """The user prompt sent to the LLM must contain the claim and sources."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({"credibility_score": _DEFAULT_CREDIBILITY})
        checker = FactChecker(llm_client=mock_llm)
        await checker.check(_CLAIM, [_SOURCE_1, _SOURCE_2])

        call_args = mock_llm.complete.call_args
        user_prompt: str = call_args[0][0]
        assert _CLAIM in user_prompt
        assert _SOURCE_1 in user_prompt
        assert _SOURCE_2 in user_prompt
