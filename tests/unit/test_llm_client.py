"""Unit tests for abstract LLM client and implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from config.settings import Settings
from pydantic import ValidationError

from finalayze.analysis.llm_client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
)
from finalayze.core.exceptions import LLMError, LLMRateLimitError

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
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """When SDK raises LLMError once, client retries and returns on second attempt."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()

            mock_choice = MagicMock()
            mock_choice.message.content = _RESPONSE
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]

            # First call raises openai.OpenAIError, second succeeds
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=[openai.OpenAIError("transient"), mock_completion]
            )
            mock_cls.return_value = mock_openai

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = OpenRouterClient(api_key="test-key", model="llama-3")
                result = await client.complete(_PROMPT, _SYSTEM)

        assert result == _RESPONSE
        assert mock_openai.chat.completions.create.call_count == _EXPECTED_CALLS_ON_SECOND_ATTEMPT

    @pytest.mark.asyncio
    async def test_raises_llm_error_after_all_retries_exhausted(self) -> None:
        """When SDK raises on every attempt, LLMError is raised after 3 attempts."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=openai.OpenAIError("persistent error")
            )
            mock_cls.return_value = mock_openai

            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = OpenRouterClient(api_key="test-key", model="llama-3")
                with pytest.raises(LLMError):
                    await client.complete(_PROMPT, _SYSTEM)

        assert mock_openai.chat.completions.create.call_count == _MAX_RETRIES

    @pytest.mark.asyncio
    async def test_rate_limit_retries_and_raises_after_exhaustion(self) -> None:
        """LLMRateLimitError is retried and re-raised after all attempts fail."""
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError("rate limited", response=MagicMock(), body=None)
            )
            mock_cls.return_value = mock_openai

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                client = OpenRouterClient(api_key="test-key", model="llama-3")
                with pytest.raises(LLMRateLimitError):
                    await client.complete(_PROMPT, _SYSTEM)

        # Sleep called between retries: _MAX_RETRIES - 1 times
        assert mock_sleep.call_count == _MAX_RETRIES - 1
        assert mock_openai.chat.completions.create.call_count == _MAX_RETRIES


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
            settings = Settings(llm_provider="openrouter", llm_api_key="key", llm_model="model")
            client = create_llm_client(settings)
        assert isinstance(client, OpenRouterClient)

    def test_openai_provider_returns_openai_client(self) -> None:
        with patch("openai.AsyncOpenAI"):
            settings = Settings(llm_provider="openai", llm_api_key="key", llm_model="gpt-4o")
            client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self) -> None:
        with patch("anthropic.AsyncAnthropic"):
            settings = Settings(llm_provider="anthropic", llm_api_key="key", llm_model="claude-3")
            client = create_llm_client(settings)
        assert isinstance(client, AnthropicClient)

    def test_unknown_provider_rejected_by_settings_validation(self) -> None:
        """Settings must reject invalid llm_provider values via Literal validation."""
        with pytest.raises(ValidationError):
            Settings(llm_provider="unknown", llm_api_key="key", llm_model="model")  # type: ignore[arg-type]
