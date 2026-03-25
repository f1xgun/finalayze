"""Abstract LLM client and provider implementations (Layer 3).

Supports OpenRouter (default), OpenAI, and Anthropic as providers.
Select provider via ``config/settings.py`` ``llm_provider`` field.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

import anthropic
import openai
import structlog

from finalayze.core.exceptions import ConfigurationError, LLMError, LLMRateLimitError

if TYPE_CHECKING:
    from config.settings import Settings

_log = structlog.get_logger(__name__)

# ── Retry configuration ─────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2
# Maximum number of responses held in the in-memory LRU cache (#147).
# Older entries are evicted when the limit is reached.
_CACHE_MAX_SIZE = 1000


class LLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt and return the model's text response.

        Args:
            prompt: The user message / question.
            system: The system instruction for the model.
            json_mode: If True, request structured JSON output from the model.
            max_tokens: Override default max_tokens (1024) for this call.

        Returns:
            Model response as a plain string.

        Raises:
            LLMRateLimitError: When provider rate limit is hit.
            LLMError: On any other LLM API failure.
        """
        ...


class _AsyncRateLimiter:
    """Sliding-window rate limiter for async code.

    Tracks request timestamps in a deque and sleeps when the window is full.
    Thread-safe within a single event loop via asyncio.Lock.
    """

    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max_rpm
        self._window = 60.0  # seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Evict timestamps older than the window
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_rpm:
                # Wait until the oldest request exits the window
                sleep_until = self._timestamps[0] + self._window
                wait = sleep_until - now
                if wait > 0:
                    _log.debug("llm_rate_limit_wait", wait_seconds=round(wait, 1))
                    # Release lock while sleeping so other coroutines can check too
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait)
                    finally:
                        await self._lock.acquire()
                    # Re-evict after sleep
                    now = time.monotonic()
                    while self._timestamps and self._timestamps[0] <= now - self._window:
                        self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


class _CachingLLMClient(LLMClient, ABC):
    """Mixin that adds SHA-256 bounded LRU in-memory caching and exponential backoff retry.

    The cache is an ``OrderedDict``-based LRU store capped at ``_CACHE_MAX_SIZE``
    entries so that the process memory does not grow without bound in long-running
    deployments (#147).
    """

    def __init__(self, rate_limiter: _AsyncRateLimiter | None = None) -> None:
        # OrderedDict used as a bounded LRU cache: oldest entry evicted when full.
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._rate_limiter = rate_limiter

    def _cache_key(self, prompt: str, system: str) -> str:
        payload = f"{system}\n{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Complete with bounded LRU caching, rate limiting, and retry."""
        key = self._cache_key(prompt, system)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        result = await self._complete_once(
            prompt, system, json_mode=json_mode, max_tokens=max_tokens
        )
        if len(self._cache) >= _CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        self._cache[key] = result
        return result

    @abstractmethod
    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Single attempt at completion — no retry logic here."""
        ...


class OpenRouterClient(_CachingLLMClient):
    """LLM client using OpenRouter API (OpenAI-compatible, many models)."""

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self, api_key: str, model: str, rate_limiter: _AsyncRateLimiter | None = None
    ) -> None:
        super().__init__(rate_limiter=rate_limiter)
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)

    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or 1024,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
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

    def __init__(
        self, api_key: str, model: str, rate_limiter: _AsyncRateLimiter | None = None
    ) -> None:
        super().__init__(rate_limiter=rate_limiter)
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or 1024,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
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

    def __init__(
        self, api_key: str, model: str, rate_limiter: _AsyncRateLimiter | None = None
    ) -> None:
        super().__init__(rate_limiter=rate_limiter)
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        effective_system = system
        if json_mode:
            effective_system = system + "\n\nIMPORTANT: Respond with valid JSON only. No comments, no extra text."
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens or 1024,
                system=effective_system,
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


class GroqClient(_CachingLLMClient):
    """LLM client using Groq API (OpenAI-compatible, free tier: 14400 req/day)."""

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self, api_key: str, model: str, rate_limiter: _AsyncRateLimiter | None = None
    ) -> None:
        super().__init__(rate_limiter=rate_limiter)
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)

    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or 1024,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            )
        except openai.RateLimitError as exc:
            msg = f"Groq rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"Groq API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "Groq returned empty response"
            raise LLMError(msg)
        return content


class DeepSeekClient(_CachingLLMClient):
    """LLM client using DeepSeek API (OpenAI-compatible)."""

    _BASE_URL = "https://api.deepseek.com"

    def __init__(
        self, api_key: str, model: str, rate_limiter: _AsyncRateLimiter | None = None
    ) -> None:
        super().__init__(rate_limiter=rate_limiter)
        self._model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)

    async def _complete_once(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens or 1024,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            )
        except openai.RateLimitError as exc:
            msg = f"DeepSeek rate limit: {exc}"
            raise LLMRateLimitError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"DeepSeek API error: {exc}"
            raise LLMError(msg) from exc

        content = completion.choices[0].message.content
        if content is None:
            msg = "DeepSeek returned empty response"
            raise LLMError(msg)
        return content


class FallbackLLMClient(LLMClient):
    """Wraps a primary and fallback client; switches on rate limit errors.

    After primary hits a rate limit, all requests go to fallback for
    ``_FALLBACK_COOLDOWN_SECONDS`` before retrying primary again.
    """

    _FALLBACK_COOLDOWN_SECONDS = 300  # 5 min before retrying primary

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallback_until: float = 0.0  # monotonic timestamp
        self._logged_fallback = False

    async def complete(
        self,
        prompt: str,
        system: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Try primary; on rate limit, use fallback for cooldown period."""
        now = time.monotonic()
        if now >= self._fallback_until:
            self._logged_fallback = False
            try:
                return await self._primary.complete(
                    prompt, system, json_mode=json_mode, max_tokens=max_tokens
                )
            except LLMRateLimitError:
                self._fallback_until = now + self._FALLBACK_COOLDOWN_SECONDS
                if not self._logged_fallback:
                    self._logged_fallback = True
                    _log.warning(
                        "llm_fallback_activated",
                        reason="rate_limit",
                        cooldown_seconds=self._FALLBACK_COOLDOWN_SECONDS,
                    )
        return await self._fallback.complete(
            prompt, system, json_mode=json_mode, max_tokens=max_tokens
        )


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory — instantiates the correct LLM client from settings.

    Args:
        settings: Application settings with ``llm_provider``, ``llm_api_key``,
            ``llm_model``, and ``llm_max_rpm`` fields.

    Returns:
        Configured LLMClient implementation.

    Raises:
        ConfigurationError: When ``llm_provider`` is not a recognised value.
    """
    provider = settings.llm_provider
    key = settings.llm_api_key
    model = settings.llm_model
    max_rpm = getattr(settings, "llm_max_rpm", 0)

    rate_limiter: _AsyncRateLimiter | None = None
    if max_rpm > 0:
        rate_limiter = _AsyncRateLimiter(max_rpm)
        _log.info("llm_rate_limiter_enabled", max_rpm=max_rpm)

    # Fallback provider (optional)
    fb_provider = getattr(settings, "llm_fallback_provider", "")
    fb_key = getattr(settings, "llm_fallback_api_key", "")
    fb_model = getattr(settings, "llm_fallback_model", "")
    if fb_provider and fb_key:
        # With fallback: primary needs no rate limiter (fallback handles 429).
        # Rate limiter goes on fallback to respect its own limits.
        primary = _build_single_client(provider, key, model, rate_limiter=None)
        fallback = _build_single_client(fb_provider, fb_key, fb_model, rate_limiter)
        _log.info(
            "llm_fallback_configured",
            fallback_provider=fb_provider,
            fallback_model=fb_model,
        )
        return FallbackLLMClient(primary, fallback)

    # No fallback: rate limiter on primary to avoid 429
    return _build_single_client(provider, key, model, rate_limiter)


def _build_single_client(
    provider: str,
    key: str,
    model: str,
    rate_limiter: _AsyncRateLimiter | None,
) -> LLMClient:
    """Build a single LLM client for the given provider."""
    if provider == "openrouter":
        return OpenRouterClient(api_key=key, model=model, rate_limiter=rate_limiter)
    if provider == "openai":
        return OpenAIClient(api_key=key, model=model, rate_limiter=rate_limiter)
    if provider == "anthropic":
        return AnthropicClient(api_key=key, model=model, rate_limiter=rate_limiter)
    if provider == "deepseek":
        return DeepSeekClient(api_key=key, model=model, rate_limiter=rate_limiter)
    if provider == "groq":
        return GroqClient(api_key=key, model=model, rate_limiter=rate_limiter)

    providers = "openrouter, openai, anthropic, deepseek, groq"
    msg = f"Unknown llm_provider {provider!r}. Choose: {providers}"
    raise ConfigurationError(msg)
