"""Unit tests for abstract LLM client and implementations."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import openai
import pytest
from config.settings import Settings
from pydantic import ValidationError

from finalayze.analysis.llm_client import (
    AnthropicClient,
    DeepSeekClient,
    GroqClient,
    LLMClient,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
)
from finalayze.core.exceptions import LLMError, LLMRateLimitError
from finalayze.core.schemas import AgentOutput, Claim, FileLineSource

_SYSTEM = "You are a financial analyst."
_PROMPT = "Analyze this news: Fed raises rates."
_RESPONSE = "Positive for USD, negative for bonds."

_MAX_RETRIES = 3
_EXPECTED_CALLS_ON_SECOND_ATTEMPT = 2


def _make_mock_openai_client(response: str = _RESPONSE) -> MagicMock:
    """Build a fully configured mock openai client that returns response."""
    mock_choice = MagicMock()
    mock_choice.message.content = response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_openai = MagicMock()
    mock_openai.chat = MagicMock()
    mock_openai.chat.completions = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_openai


class TestLLMClientIsAbstract:
    def test_cannot_instantiate_base_class(self) -> None:
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


class TestOpenRouterClient:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = _make_mock_openai_client()
            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result = await client.complete(_PROMPT, _SYSTEM)

        assert result == _RESPONSE

    @pytest.mark.asyncio
    async def test_caches_identical_prompts(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = _make_mock_openai_client()
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result1 = await client.complete(_PROMPT, _SYSTEM)
            result2 = await client.complete(_PROMPT, _SYSTEM)

        assert result1 == result2
        # create called only once (second call hits cache)
        assert mock_openai.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_client_created_once_not_per_request(self) -> None:
        """SDK client must be created in __init__, not on every _complete_once call."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = _make_mock_openai_client()
            client = OpenRouterClient(api_key="test-key", model="llama-3")
            # make two different requests (different prompts → no cache hit)
            await client.complete("prompt1", _SYSTEM)
            await client.complete("prompt2", _SYSTEM)

        # AsyncOpenAI constructor must be called only ONCE (in __init__)
        assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates_immediately(self) -> None:
        """RateLimitError propagates without retry (FallbackLLMClient handles it)."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()

            rate_resp = MagicMock(status_code=429, headers={})
            rate_err = openai.RateLimitError(message="rate limited", response=rate_resp, body=None)
            mock_openai.chat.completions.create = AsyncMock(side_effect=rate_err)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            with pytest.raises(LLMRateLimitError):
                await client.complete(_PROMPT, _SYSTEM)

        # Only 1 call — no retries, fast-fail for FallbackLLMClient to handle
        assert mock_openai.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_llm_error_immediately_no_retry(self) -> None:
        """Non-rate-limit errors (402, 500) fail fast without retry."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=openai.OpenAIError("persistent error")
            )
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            with pytest.raises(LLMError):
                await client.complete(_PROMPT, _SYSTEM)

        # Only 1 call — no retries for non-rate-limit errors
        assert mock_openai.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_raises_immediately_no_retry(self) -> None:
        """LLMRateLimitError propagates immediately — no retries."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError("rate limited", response=MagicMock(), body=None)
            )
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            with pytest.raises(LLMRateLimitError):
                await client.complete(_PROMPT, _SYSTEM)

        # Single call, no retries
        assert mock_openai.chat.completions.create.call_count == 1


# ── #147: Bounded LRU cache ──────────────────────────────────────────────────


class TestBoundedLRUCache:
    """The in-memory cache must not grow beyond _CACHE_MAX_SIZE entries (#147)."""

    @pytest.mark.asyncio
    async def test_cache_evicts_oldest_entry_when_full(self) -> None:
        from finalayze.analysis.llm_client import _CACHE_MAX_SIZE

        with patch("openai.AsyncOpenAI") as mock_cls:
            # Each unique prompt returns its index as a string
            call_count = 0

            async def _side_effect(*_args: object, **_kwargs: object) -> object:
                nonlocal call_count
                mock_choice = MagicMock()
                mock_choice.message.content = str(call_count)
                call_count += 1
                mock_completion = MagicMock()
                mock_completion.choices = [mock_choice]
                return mock_completion

            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = _side_effect
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")

            # Fill the cache to exactly its maximum
            for i in range(_CACHE_MAX_SIZE):
                await client.complete(f"unique_prompt_{i}", _SYSTEM)

            assert len(client._cache) == _CACHE_MAX_SIZE  # noqa: SLF001

            # Adding one more entry must evict the oldest
            await client.complete("overflow_prompt", _SYSTEM)
            assert len(client._cache) == _CACHE_MAX_SIZE  # noqa: SLF001
            # The very first prompt should have been evicted
            first_key = client._cache_key("unique_prompt_0", _SYSTEM)  # noqa: SLF001
            assert first_key not in client._cache  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_grow_cache(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = _make_mock_openai_client()
            client = OpenRouterClient(api_key="test-key", model="llama-3")

            await client.complete(_PROMPT, _SYSTEM)
            size_after_first = len(client._cache)  # noqa: SLF001
            # Same prompt — must hit cache, not add a new entry
            await client.complete(_PROMPT, _SYSTEM)
            assert len(client._cache) == size_after_first  # noqa: SLF001


class TestCreateLLMClientFactory:
    def test_openrouter_provider_returns_openrouter_client(self) -> None:
        with patch("openai.AsyncOpenAI"):
            settings = Settings(
                llm_provider="openrouter",
                llm_api_key="key",
                llm_model="model",
                llm_fallback_provider="",
                llm_fallback_api_key="",
            )
            client = create_llm_client(settings)
        assert isinstance(client, OpenRouterClient)

    def test_openai_provider_returns_openai_client(self) -> None:
        with patch("openai.AsyncOpenAI"):
            settings = Settings(
                llm_provider="openai",
                llm_api_key="key",
                llm_model="gpt-4o",
                llm_fallback_provider="",
                llm_fallback_api_key="",
            )
            client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self) -> None:
        with patch("anthropic.AsyncAnthropic"):
            settings = Settings(
                llm_provider="anthropic",
                llm_api_key="key",
                llm_model="claude-3",
                llm_fallback_provider="",
                llm_fallback_api_key="",
            )
            client = create_llm_client(settings)
        assert isinstance(client, AnthropicClient)

    def test_unknown_provider_rejected_by_settings_validation(self) -> None:
        """Settings must reject invalid llm_provider values via Literal validation."""
        with pytest.raises(ValidationError):
            Settings(llm_provider="unknown", llm_api_key="key", llm_model="model")  # type: ignore[arg-type]


# ── parse_structured() helpers ───────────────────────────────────────────────

_DT = datetime(2026, 4, 12, tzinfo=UTC)

_AGENT_OUTPUT = AgentOutput(
    agent_name="quant-analyst",
    recommendation="Enable dual_momentum",
    claims=[
        Claim(
            statement="PF is 1.29",
            source=FileLineSource(
                kind="file",
                path="src/finalayze/strategies/combiner.py",
                line=142,
                excerpt="class StrategyCombiner",
            ),
            confidence=0.9,
        )
    ],
    timestamp=_DT,
)

_AGENT_OUTPUT_JSON = _AGENT_OUTPUT.model_dump_json()


def _make_anthropic_parse_response(parsed_obj: object) -> MagicMock:
    """Build a mock response from anthropic messages.parse()."""
    mock_message = MagicMock()
    mock_message.parsed_output = parsed_obj
    return mock_message


def _make_openai_parse_response(parsed_obj: object) -> MagicMock:
    """Build a mock response from openai beta.chat.completions.parse()."""
    mock_choice = MagicMock()
    mock_choice.message.parsed = parsed_obj
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


# ── AnthropicClient.parse_structured() ──────────────────────────────────────


class TestAnthropicClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_calls_messages_parse(self) -> None:
        """AnthropicClient.parse_structured() calls self._client.messages.parse()."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_anthropic = MagicMock()
            mock_anthropic.messages.parse = AsyncMock(
                return_value=_make_anthropic_parse_response(_AGENT_OUTPUT)
            )
            mock_cls.return_value = mock_anthropic

            client = AnthropicClient(api_key="test-key", model="claude-3")
            result = await client.parse_structured(
                prompt=_PROMPT,
                system=_SYSTEM,
                response_model=AgentOutput,
            )

        assert result == _AGENT_OUTPUT
        mock_anthropic.messages.parse.assert_called_once()
        call_kwargs = mock_anthropic.messages.parse.call_args.kwargs
        assert call_kwargs["output_format"] is AgentOutput

    @pytest.mark.asyncio
    async def test_parse_structured_raises_rate_limit_error(self) -> None:
        """AnthropicClient.parse_structured() raises LLMRateLimitError on RateLimitError."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_anthropic = MagicMock()
            rate_resp = MagicMock(status_code=429, headers={})
            mock_anthropic.messages.parse = AsyncMock(
                side_effect=anthropic.RateLimitError(
                    message="rate limited",
                    response=rate_resp,
                    body=None,
                )
            )
            mock_cls.return_value = mock_anthropic

            client = AnthropicClient(api_key="test-key", model="claude-3")
            with pytest.raises(LLMRateLimitError):
                await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)

    @pytest.mark.asyncio
    async def test_parse_structured_raises_llm_error_on_api_error(self) -> None:
        """AnthropicClient.parse_structured() raises LLMError on APIError."""
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_anthropic = MagicMock()
            mock_anthropic.messages.parse = AsyncMock(
                side_effect=anthropic.APIStatusError(
                    message="api error",
                    response=MagicMock(status_code=500, headers={}),
                    body=None,
                )
            )
            mock_cls.return_value = mock_anthropic

            client = AnthropicClient(api_key="test-key", model="claude-3")
            with pytest.raises(LLMError):
                await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)


# ── OpenAIClient.parse_structured() ─────────────────────────────────────────


class TestOpenAIClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_calls_beta_parse(self) -> None:
        """OpenAIClient.parse_structured() calls beta.chat.completions.parse()."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.beta = MagicMock()
            mock_openai.beta.chat = MagicMock()
            mock_openai.beta.chat.completions = MagicMock()
            mock_openai.beta.chat.completions.parse = AsyncMock(
                return_value=_make_openai_parse_response(_AGENT_OUTPUT)
            )
            mock_cls.return_value = mock_openai

            client = OpenAIClient(api_key="test-key", model="gpt-4o")
            result = await client.parse_structured(
                prompt=_PROMPT,
                system=_SYSTEM,
                response_model=AgentOutput,
            )

        assert result == _AGENT_OUTPUT
        mock_openai.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_openai.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["response_format"] is AgentOutput

    @pytest.mark.asyncio
    async def test_parse_structured_raises_llm_error_when_parsed_is_none(self) -> None:
        """OpenAIClient.parse_structured() raises LLMError when parsed is None."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.beta = MagicMock()
            mock_openai.beta.chat = MagicMock()
            mock_openai.beta.chat.completions = MagicMock()
            mock_openai.beta.chat.completions.parse = AsyncMock(
                return_value=_make_openai_parse_response(None)
            )
            mock_cls.return_value = mock_openai

            client = OpenAIClient(api_key="test-key", model="gpt-4o")
            with pytest.raises(LLMError):
                await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)


# ── OpenRouterClient.parse_structured() ─────────────────────────────────────


class TestOpenRouterClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_falls_back_on_bad_request(self) -> None:
        """OpenRouterClient falls back to complete(json_mode=True) on BadRequestError."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.beta = MagicMock()
            mock_openai.beta.chat = MagicMock()
            mock_openai.beta.chat.completions = MagicMock()
            # Structured parse fails with BadRequestError
            bad_req = openai.BadRequestError(
                message="unsupported",
                response=MagicMock(status_code=400, headers={}),
                body=None,
            )
            mock_openai.beta.chat.completions.parse = AsyncMock(side_effect=bad_req)
            # Fallback to regular chat completions returning JSON
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = _AGENT_OUTPUT_JSON
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result = await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)

        assert result.agent_name == _AGENT_OUTPUT.agent_name
        assert result.recommendation == _AGENT_OUTPUT.recommendation


# ── GroqClient.parse_structured() ───────────────────────────────────────────


class TestGroqClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_uses_openai_compatible_path(self) -> None:
        """GroqClient.parse_structured() uses beta.chat.completions.parse()."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.beta = MagicMock()
            mock_openai.beta.chat = MagicMock()
            mock_openai.beta.chat.completions = MagicMock()
            mock_openai.beta.chat.completions.parse = AsyncMock(
                return_value=_make_openai_parse_response(_AGENT_OUTPUT)
            )
            mock_cls.return_value = mock_openai

            client = GroqClient(api_key="test-key", model="llama3-8b-8192")
            result = await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)

        assert result == _AGENT_OUTPUT


# ── DeepSeekClient.parse_structured() ───────────────────────────────────────


class TestDeepSeekClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_uses_openai_compatible_path(self) -> None:
        """DeepSeekClient.parse_structured() uses beta.chat.completions.parse()."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.beta = MagicMock()
            mock_openai.beta.chat = MagicMock()
            mock_openai.beta.chat.completions = MagicMock()
            mock_openai.beta.chat.completions.parse = AsyncMock(
                return_value=_make_openai_parse_response(_AGENT_OUTPUT)
            )
            mock_cls.return_value = mock_openai

            client = DeepSeekClient(api_key="test-key", model="deepseek-chat")
            result = await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)

        assert result == _AGENT_OUTPUT
