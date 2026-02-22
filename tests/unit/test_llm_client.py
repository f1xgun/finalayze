"""Unit tests for abstract LLM client and implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.settings import Settings

from finalayze.analysis.llm_client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
)
from finalayze.core.exceptions import ConfigurationError

_SYSTEM = "You are a financial analyst."
_PROMPT = "Analyze this news: Fed raises rates."
_RESPONSE = "Positive for USD, negative for bonds."


class TestLLMClientIsAbstract:
    def test_cannot_instantiate_base_class(self) -> None:
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


class TestOpenRouterClient:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = _RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat = MagicMock()
            mock_openai.chat.completions = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result = await client.complete(_PROMPT, _SYSTEM)

        assert result == _RESPONSE

    @pytest.mark.asyncio
    async def test_caches_identical_prompts(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = _RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_openai

            client = OpenRouterClient(api_key="test-key", model="llama-3")
            result1 = await client.complete(_PROMPT, _SYSTEM)
            result2 = await client.complete(_PROMPT, _SYSTEM)

        assert result1 == result2
        # create called only once (second call hits cache)
        assert mock_openai.chat.completions.create.call_count == 1


class TestCreateLLMClientFactory:
    def test_openrouter_provider_returns_openrouter_client(self) -> None:
        settings = Settings(llm_provider="openrouter", llm_api_key="key", llm_model="model")
        client = create_llm_client(settings)
        assert isinstance(client, OpenRouterClient)

    def test_openai_provider_returns_openai_client(self) -> None:
        settings = Settings(llm_provider="openai", llm_api_key="key", llm_model="gpt-4o")
        client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self) -> None:
        settings = Settings(llm_provider="anthropic", llm_api_key="key", llm_model="claude-3")
        client = create_llm_client(settings)
        assert isinstance(client, AnthropicClient)

    def test_unknown_provider_raises_configuration_error(self) -> None:
        settings = Settings(llm_provider="unknown", llm_api_key="key", llm_model="model")
        with pytest.raises(ConfigurationError):
            create_llm_client(settings)
