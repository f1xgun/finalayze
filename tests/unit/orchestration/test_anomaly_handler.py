"""Tests for AnomalyHandler — extracted anomaly raw + LLM-enrichment orchestration.

Tests verify:
  - handle() returns raw_alert_id and schedules enrich() when llm_client present
  - handle() skips enrich() when provider returns None
  - handle() swallows send_async failure (returns None)
  - enrich() threads parent_id into the anomaly_llm send
  - enrich() swallows all exceptions
  - lazy provider is read at call time (set provider to return None then a client)
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from finalayze.api.alerts import TelegramAlerter
from finalayze.orchestration.anomaly_handler import AnomalyHandler

_SYMBOL = "SBER"
_MARKET_ID = "ru_blue_chips"


def _make_anomaly() -> SimpleNamespace:
    """Build a duck-typed AnomalyResult."""
    return SimpleNamespace(
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        price_move_pct=15.0,
        sigma=5.0,
        volume_ratio=1.2,
        anomaly_type="price",
    )


class TestAnomalyHandlerHandleMethod:
    """Test AnomalyHandler.handle() orchestration logic."""

    @pytest.mark.asyncio
    async def test_handle_returns_raw_alert_id(self) -> None:
        """handle() returns the UUID from the raw alert send."""
        raw_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

        handler = AnomalyHandler(alerter, lambda: None)
        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        result = await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        assert result == raw_uuid

    @pytest.mark.asyncio
    async def test_handle_schedules_enrich_when_llm_present(self) -> None:
        """handle() schedules enrich() task when llm_client provider returns a client."""
        raw_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

        llm_client = AsyncMock()
        handler = AnomalyHandler(alerter, lambda: llm_client)

        # Monkey-patch enrich to track if it was called
        enrich_called = []

        async def mock_enrich(*args: Any, **kwargs: Any) -> None:
            enrich_called.append((args, kwargs))

        handler.enrich = mock_enrich  # type: ignore[method-assign]

        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        result = await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        # Yield to let the task run
        await asyncio.sleep(0)

        assert result == raw_uuid
        assert len(enrich_called) == 1, f"enrich() must be called once, got {len(enrich_called)}"

    @pytest.mark.asyncio
    async def test_handle_skips_enrich_when_llm_none(self) -> None:
        """handle() does not schedule enrich() when provider returns None."""
        raw_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

        handler = AnomalyHandler(alerter, lambda: None)  # no LLM

        # Monkey-patch enrich to track if it was called
        enrich_called = []

        async def mock_enrich(*args: Any, **kwargs: Any) -> None:
            enrich_called.append((args, kwargs))

        handler.enrich = mock_enrich  # type: ignore[method-assign]

        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        result = await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        # Yield once to give any task a chance to run
        await asyncio.sleep(0)

        assert result == raw_uuid
        assert len(enrich_called) == 0, "enrich() must NOT be called when llm_client is None"

    @pytest.mark.asyncio
    async def test_handle_returns_none_on_send_failure(self) -> None:
        """handle() returns None and logs when send_async fails."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(side_effect=RuntimeError("Telegram down"))

        handler = AnomalyHandler(alerter, lambda: None)
        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        with structlog.testing.capture_logs() as captured:
            result = await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        assert result is None
        failure_logs = [log for log in captured if log.get("event") == "anomaly_raw_send_failed"]
        assert len(failure_logs) >= 1, f"Expected anomaly_raw_send_failed log, got: {captured}"

    @pytest.mark.asyncio
    async def test_handle_passes_anomaly_raw_type(self) -> None:
        """handle() sends with alert_type='anomaly_raw' and parent_id=None."""
        raw_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

        handler = AnomalyHandler(alerter, lambda: None)
        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        alerter.send_async.assert_awaited_once()
        kwargs = alerter.send_async.await_args.kwargs
        assert kwargs.get("alert_type") == "anomaly_raw"
        assert kwargs.get("parent_id") is None

    @pytest.mark.asyncio
    async def test_handle_calls_self_enrich_for_patchability(self) -> None:
        """handle() calls self.enrich() (not a function) so instance patches work."""
        raw_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, raw_uuid))

        llm_client = AsyncMock()
        handler = AnomalyHandler(alerter, lambda: llm_client)

        captured: dict[str, Any] = {}

        async def _capture_enrich(
            symbol: str,
            market_id: str,
            anomaly: object,
            *,
            parent_id: uuid.UUID | None = None,
        ) -> None:
            captured["symbol"] = symbol
            captured["market_id"] = market_id
            captured["parent_id"] = parent_id

        handler.enrich = _capture_enrich  # type: ignore[method-assign]

        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"

        await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)
        await asyncio.sleep(0)

        assert captured["symbol"] == _SYMBOL
        assert captured["parent_id"] == raw_uuid


class TestAnomalyHandlerEnrichMethod:
    """Test AnomalyHandler.enrich() LLM logic."""

    @pytest.mark.asyncio
    async def test_enrich_threads_parent_id_into_send(self) -> None:
        """enrich() forwards parent_id into send_async(alert_type='anomaly_llm')."""
        parent_uuid = uuid.uuid4()
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="LLM explanation")

        handler = AnomalyHandler(alerter, lambda: llm_client)
        anomaly = _make_anomaly()

        await handler.enrich(
            _SYMBOL,
            _MARKET_ID,
            anomaly,
            parent_id=parent_uuid,
        )

        alerter.send_async.assert_awaited_once()
        kwargs = alerter.send_async.await_args.kwargs
        assert kwargs.get("alert_type") == "anomaly_llm"
        assert kwargs.get("parent_id") == parent_uuid

    @pytest.mark.asyncio
    async def test_enrich_constructs_llm_prompt(self) -> None:
        """enrich() builds LLM prompt with anomaly details."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="explanation")

        handler = AnomalyHandler(alerter, lambda: llm_client)
        anomaly = _make_anomaly()

        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)

        prompt_arg = llm_client.complete.await_args[0][0]
        assert _SYMBOL in prompt_arg
        assert _MARKET_ID in prompt_arg
        assert "15.0" in prompt_arg or "+15.0" in prompt_arg  # price_move_pct

    @pytest.mark.asyncio
    async def test_enrich_swallows_llm_timeout(self) -> None:
        """enrich() swallows TimeoutError from LLM, logs, does not send follow-up."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(side_effect=TimeoutError("LLM timeout"))

        handler = AnomalyHandler(alerter, lambda: llm_client)
        anomaly = _make_anomaly()

        with structlog.testing.capture_logs() as captured:
            await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)

        # No follow-up sent
        alerter.send_async.assert_not_called()
        # Logs failure
        failure_logs = [log for log in captured if log.get("event") == "anomaly_llm_failure"]
        assert len(failure_logs) >= 1

    @pytest.mark.asyncio
    async def test_enrich_swallows_all_exceptions(self) -> None:
        """enrich() swallows ALL exceptions -- never re-raises."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(side_effect=RuntimeError("Telegram down"))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="explanation")

        handler = AnomalyHandler(alerter, lambda: llm_client)
        anomaly = _make_anomaly()

        # Must not raise
        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)

    @pytest.mark.asyncio
    async def test_enrich_uses_lazy_provider(self) -> None:
        """enrich() reads llm_client via provider at call time, not construction."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        # Provider returns None initially
        provider_state: dict[str, Any] = {"client": None}

        def provider() -> Any:
            return provider_state["client"]

        handler = AnomalyHandler(alerter, provider)
        anomaly = _make_anomaly()

        # First call: no LLM, so send_async is NOT called for follow-up
        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)
        alerter.send_async.assert_not_called()

        # Now set LLM client
        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="explanation")
        provider_state["client"] = llm_client

        # Second call: LLM is present, so send_async IS called for follow-up
        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)
        alerter.send_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_enrich_formats_follow_up_with_unverified_label(self) -> None:
        """enrich() follow-up message includes 'AI interpretation (unverified):' label."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="Likely driven by CBR decision")

        handler = AnomalyHandler(alerter, lambda: llm_client)
        anomaly = _make_anomaly()

        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)

        sent_text = alerter.send_async.await_args[0][0]
        assert sent_text.startswith("AI interpretation (unverified):")
        assert "Likely driven by CBR decision" in sent_text


class TestAnomalyHandlerLazyProvider:
    """Test lazy LLM client provider behavior."""

    @pytest.mark.asyncio
    async def test_provider_called_at_handle_time(self) -> None:
        """handle() calls provider at call time, not construction."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        call_count = [0]

        def provider() -> Any:
            call_count[0] += 1
            return None

        handler = AnomalyHandler(alerter, provider)
        assert call_count[0] == 0, "Provider should not be called during construction"

        anomaly = _make_anomaly()
        raw_text = "ANOMALY raw"
        await handler.handle(_SYMBOL, _MARKET_ID, anomaly, raw_text)

        assert call_count[0] >= 1, "Provider should be called during handle()"

    @pytest.mark.asyncio
    async def test_provider_called_at_enrich_time(self) -> None:
        """enrich() calls provider at call time, not construction."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_async = AsyncMock(return_value=(True, uuid.uuid4()))

        call_count = [0]

        def provider() -> Any:
            call_count[0] += 1
            return None

        handler = AnomalyHandler(alerter, provider)
        assert call_count[0] == 0, "Provider should not be called during construction"

        anomaly = _make_anomaly()
        await handler.enrich(_SYMBOL, _MARKET_ID, anomaly)

        assert call_count[0] >= 1, "Provider should be called during enrich()"
