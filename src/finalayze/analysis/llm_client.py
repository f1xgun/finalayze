"""Abstract LLM client and provider implementations (Layer 3).

Supports OpenRouter (default), OpenAI, and Anthropic as providers.
Select provider via ``config/settings.py`` ``llm_provider`` field.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import anthropic
import openai

from finalayze.core.exceptions import ConfigurationError, LLMError, LLMRateLimitError

if TYPE_CHECKING:
    from config.settings import Settings

# ── Retry configuration ─────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2
# Maximum number of responses held in the in-memory LRU cache (#147).
# Older entries are evicted when the limit is reached.
_CACHE_MAX_SIZE = 1000


class LLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user message / question.
            system: The system instruction for the model.

        Returns:
            Model response as a plain string.

        Raises:
            LLMRateLimitError: When provider rate limit is hit.
            LLMError: On any other LLM API failure.
        """
        ...


class _CachingLLMClient(LLMClient, ABC):
    """Mixin that adds SHA-256 bounded LRU in-memory caching and exponential backoff retry.

    The cache is an ``OrderedDict``-based LRU store capped at ``_CACHE_MAX_SIZE``
    entries so that the process memory does not grow without bound in long-running
    deployments (#147).
    """

    def __init__(self) -> None:
        # OrderedDict used as a bounded LRU cache: oldest entry evicted when full.
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _cache_key(self, prompt: str, system: str) -> str:
        payload = f"{system}\n{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(self, prompt: str, system: str) -> str:
        """Complete with bounded LRU caching and retry."""
        key = self._cache_key(prompt, system)
        if key in self._cache:
            # Move to end to mark as most-recently used
            self._cache.move_to_end(key)
            return self._cache[key]

        for attempt in range(_MAX_RETRIES):
            try:
                result = await self._complete_once(prompt, system)
                # Evict oldest entry when cache is full (#147)
                if len(self._cache) >= _CACHE_MAX_SIZE:
                    self._cache.popitem(last=False)
                self._cache[key] = result
                return result
            except LLMRateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BASE_SECONDS ** (attempt + 1)
                await asyncio.sleep(wait)
            except LLMError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BASE_SECONDS ** (attempt + 1)
                await asyncio.sleep(wait)

        msg = "LLM request failed after all retries"  # unreachable but satisfies mypy
        raise LLMError(msg)

    @abstractmethod
    async def _complete_once(self, prompt: str, system: str) -> str:
        """Single attempt at completion — no retry logic here."""
        ...


class OpenRouterClient(_CachingLLMClient):
    """LLM client using OpenRouter API (OpenAI-compatible, many models)."""

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)

    async def _complete_once(self, prompt: str, system: str) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
        except openai.RateLimitError as exc:
            msg = f"OpenRouter rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"OpenRouter API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "OpenRouter returned empty response"
            raise LLMError(msg)
        return content


class OpenAIClient(_CachingLLMClient):
    """LLM client using OpenAI API directly."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def _complete_once(self, prompt: str, system: str) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
        except openai.RateLimitError as exc:
            msg = f"OpenAI rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"OpenAI API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "OpenAI returned empty response"
            raise LLMError(msg)
        return content


class AnthropicClient(_CachingLLMClient):
    """LLM client using Anthropic API (requires console API key)."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _complete_once(self, prompt: str, system: str) -> str:
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as exc:
            msg = f"Anthropic rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except anthropic.APIError as exc:
            msg = f"Anthropic API error: {exc}"
            raise LLMError(msg) from exc

        block = message.content[0]
        if not hasattr(block, "text"):
            msg = "Anthropic returned non-text content block"
            raise LLMError(msg)
        return block.text


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory — instantiates the correct LLM client from settings.

    Args:
        settings: Application settings with ``llm_provider``, ``llm_api_key``,
            and ``llm_model`` fields.

    Returns:
        Configured LLMClient implementation.

    Raises:
        ConfigurationError: When ``llm_provider`` is not a recognised value.
    """
    provider = settings.llm_provider
    key = settings.llm_api_key
    model = settings.llm_model

    if provider == "openrouter":
        return OpenRouterClient(api_key=key, model=model)
    if provider == "openai":
        return OpenAIClient(api_key=key, model=model)
    if provider == "anthropic":
        return AnthropicClient(api_key=key, model=model)

    msg = f"Unknown llm_provider {provider!r}. Choose: openrouter, openai, anthropic"
    raise ConfigurationError(msg)
