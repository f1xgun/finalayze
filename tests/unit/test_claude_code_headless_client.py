"""Unit tests for ClaudeCodeHeadlessClient (subscription-backed via `claude -p`)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.settings import Settings

from finalayze.analysis.llm_client import (
    ClaudeCodeHeadlessClient,
    create_llm_client,
)
from finalayze.core.exceptions import LLMError, LLMRateLimitError
from finalayze.core.schemas import AgentOutput, Claim, FileLineSource

_SYSTEM = "You are a financial analyst."
_PROMPT = "Analyze this news: Fed raises rates."
_RESULT_TEXT = "Positive for USD, negative for bonds."


def _envelope(result_text: str, *, is_error: bool = False) -> bytes:
    """Build a Claude Code JSON envelope mimicking what the real CLI emits."""
    payload = {
        "type": "result",
        "subtype": "success" if not is_error else "error",
        "is_error": is_error,
        "result": result_text,
        "total_cost_usd": 0.001,
        "session_id": "test-session-id",
    }
    return (json.dumps(payload) + "\n").encode()


def _patch_spawn(stdout: bytes, *, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    """Return an AsyncMock substitute for asyncio.create_subprocess_exec."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return AsyncMock(return_value=proc)


# ── complete() ──────────────────────────────────────────────────────────────


class TestClaudeCodeHeadlessClientComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_envelope_result_field(self) -> None:
        spawn = _patch_spawn(_envelope(_RESULT_TEXT))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            result = await client.complete(_PROMPT, _SYSTEM)
        assert result == _RESULT_TEXT

    @pytest.mark.asyncio
    async def test_complete_passes_required_cli_args(self) -> None:
        spawn = _patch_spawn(_envelope("OK"))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet", max_budget_usd=0.5)
            await client.complete(_PROMPT, _SYSTEM)
        args = list(spawn.call_args.args)
        assert args[0] == "claude"
        assert "-p" in args
        assert _PROMPT in args
        # Replace, NOT append — otherwise project CLAUDE.md + skills + memory
        # leak into every call (44k+ tokens of overhead).
        assert "--system-prompt" in args
        assert _SYSTEM in args
        assert "--append-system-prompt" not in args
        assert "--model" in args
        assert "sonnet" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--no-session-persistence" in args
        assert "--max-budget-usd" in args
        assert "0.5" in args

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_llm_error(self) -> None:
        spawn = _patch_spawn(b"", stderr=b"unknown error", returncode=2)
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            with pytest.raises(LLMError):
                await client.complete(_PROMPT, _SYSTEM)

    @pytest.mark.asyncio
    async def test_rate_limit_in_stderr_raises_rate_limit_error(self) -> None:
        spawn = _patch_spawn(b"", stderr=b"5-hour usage limit reached", returncode=1)
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            with pytest.raises(LLMRateLimitError):
                await client.complete(_PROMPT, _SYSTEM)

    @pytest.mark.asyncio
    async def test_envelope_is_error_true_raises_llm_error(self) -> None:
        spawn = _patch_spawn(_envelope("model refused", is_error=True))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            with pytest.raises(LLMError):
                await client.complete(_PROMPT, _SYSTEM)

    @pytest.mark.asyncio
    async def test_invalid_json_envelope_raises_llm_error(self) -> None:
        spawn = _patch_spawn(b"not json at all")
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            with pytest.raises(LLMError):
                await client.complete(_PROMPT, _SYSTEM)

    @pytest.mark.asyncio
    async def test_caches_identical_prompts(self) -> None:
        spawn = _patch_spawn(_envelope("OK"))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            r1 = await client.complete(_PROMPT, _SYSTEM)
            r2 = await client.complete(_PROMPT, _SYSTEM)
        assert r1 == r2 == "OK"
        # Subprocess spawned only once due to LRU cache (second call hits cache).
        assert spawn.call_count == 1


# ── parse_structured() ──────────────────────────────────────────────────────

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


class TestClaudeCodeHeadlessClientParseStructured:
    @pytest.mark.asyncio
    async def test_parse_structured_passes_json_schema_arg(self) -> None:
        spawn = _patch_spawn(_envelope(_AGENT_OUTPUT.model_dump_json()))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            result = await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)

        assert result.agent_name == _AGENT_OUTPUT.agent_name
        assert result.recommendation == _AGENT_OUTPUT.recommendation

        args = list(spawn.call_args.args)
        assert "--json-schema" in args
        idx = args.index("--json-schema")
        schema_doc = json.loads(args[idx + 1])
        # Pydantic-generated schema has agent_name as a property
        assert "agent_name" in schema_doc.get("properties", {})

    @pytest.mark.asyncio
    async def test_parse_structured_invalid_response_raises_llm_error(self) -> None:
        spawn = _patch_spawn(_envelope('{"agent_name": "q", "missing_required_fields": true}'))
        with patch("finalayze.analysis.llm_client.asyncio.create_subprocess_exec", spawn):
            client = ClaudeCodeHeadlessClient(model="sonnet")
            with pytest.raises(LLMError):
                await client.parse_structured(_PROMPT, _SYSTEM, AgentOutput)


# ── Factory ─────────────────────────────────────────────────────────────────


class TestCreateLLMClientFactoryHeadless:
    def test_claude_code_headless_provider_returns_headless_client(self) -> None:
        settings = Settings(
            llm_provider="claude_code_headless",
            llm_api_key="",  # subscription auth — no key required
            llm_model="sonnet",
            llm_fallback_provider="",
            llm_fallback_api_key="",
        )
        client = create_llm_client(settings)
        assert isinstance(client, ClaudeCodeHeadlessClient)

    def test_headless_primary_with_anthropic_fallback(self) -> None:
        """Subscription primary + API fallback survives 5-hour rate windows."""
        with patch("anthropic.AsyncAnthropic"):
            settings = Settings(
                llm_provider="claude_code_headless",
                llm_api_key="",
                llm_model="sonnet",
                llm_fallback_provider="anthropic",
                llm_fallback_api_key="key",
                llm_fallback_model="claude-sonnet-4-6",
            )
            client = create_llm_client(settings)
        # Wrapped in FallbackLLMClient (not directly a headless client).
        from finalayze.analysis.llm_client import AnthropicClient, FallbackLLMClient

        assert isinstance(client, FallbackLLMClient)
        assert isinstance(client._primary, ClaudeCodeHeadlessClient)  # noqa: SLF001
        assert isinstance(client._fallback, AnthropicClient)  # noqa: SLF001
