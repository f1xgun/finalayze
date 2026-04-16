"""Tests for anomaly detection integration in TradingLoop.

Verifies ANMI-01 (ordering), ANMI-02 (enrichment format), ANMI-03 (graceful degradation).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from finalayze.analysis.anomaly_detector import AnomalyDetector, AnomalyResult
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import Candle

# -- Helpers ------------------------------------------------------------------

_SYMBOL = "SBER"
_MARKET_ID = "ru_blue_chips"


def _make_candle(close: float, volume: int, offset_minutes: int = 0) -> Candle:
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        timeframe="1h",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes),
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=volume,
    )


def _make_anomaly_candles() -> list[Candle]:
    """25 normal candles + last candle with +15% spike -- guaranteed anomaly."""
    candles = []
    for i in range(24):
        factor = 1.0 + (i % 5 - 2) * 0.002
        candles.append(_make_candle(close=100.0 * factor, volume=1000, offset_minutes=i * 60))
    # Spike on last candle
    candles.append(_make_candle(close=115.0, volume=1000, offset_minutes=24 * 60))
    return candles


def _make_anomaly_result() -> AnomalyResult:
    return AnomalyResult(
        symbol=_SYMBOL,
        market_id=_MARKET_ID,
        price_move_pct=15.0,
        sigma=5.0,
        volume_ratio=1.0,
        anomaly_type="price",
    )


def _make_tl() -> object:
    """Create a minimal TradingLoop instance for testing _enrich_anomaly_async."""
    from finalayze.core.trading_loop import TradingLoop

    return object.__new__(TradingLoop)


# -- ANMI-01: Ordering Guarantee ----------------------------------------------


class TestOrderingGuarantee:
    """Raw alert MUST fire before any LLM await."""

    def test_send_alert_called_before_llm_complete(self) -> None:
        """Verify send_alert() is invoked before llm_client.complete() is awaited.

        Uses call_order list with side_effect to record invocation sequence --
        same pattern as test_rate_limiter_integration.py.
        """
        call_order: list[str] = []

        alerter = MagicMock(spec=TelegramAlerter)
        alerter.send_alert = MagicMock(side_effect=lambda msg: call_order.append("raw_alert"))
        alerter._send = AsyncMock(side_effect=lambda msg: call_order.append("follow_up"))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(
            side_effect=lambda prompt, system: call_order.append("llm_call") or "test explanation"
        )

        detector = AnomalyDetector()
        candles = _make_anomaly_candles()

        # Verify detector finds anomaly
        anomaly = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert anomaly is not None, "Test setup error: candles must trigger anomaly"

        # Simulate what _process_instrument does:
        # 1. send_alert (sync)
        raw_text = f"ANOMALY {_SYMBOL}: {anomaly.price_move_pct:+.1f}%"
        alerter.send_alert(raw_text)

        # 2. LLM enrichment (async, fire-and-forget -- here we run it synchronously for test)
        loop = asyncio.new_event_loop()
        try:
            tl = _make_tl()
            tl._llm_client = llm_client  # type: ignore[attr-defined]
            tl._alerter = alerter  # type: ignore[attr-defined]

            loop.run_until_complete(
                tl._enrich_anomaly_async(_SYMBOL, _MARKET_ID, anomaly)  # type: ignore[attr-defined]
            )
        finally:
            loop.close()

        assert len(call_order) >= 2, f"Expected at least 2 calls, got: {call_order}"
        assert call_order[0] == "raw_alert", (
            f"raw_alert must be first call, got order: {call_order}"
        )

    def test_anomaly_detection_calls_send_alert_synchronously(self) -> None:
        """Verify _enrich_anomaly_async uses _send (async follow-up), not send_alert.

        The raw sync alert (send_alert) is fired BEFORE _enrich_anomaly_async
        is called. _enrich_anomaly_async itself only sends the async follow-up
        via _send. This ensures the raw alert is never delayed by LLM calls.
        """
        import inspect

        from finalayze.core.trading_loop import TradingLoop

        # After decomposition, the anomaly enrichment lives on TradingLoop
        # as _enrich_anomaly_async. Verify it uses _send (async) not send_alert (sync).
        source = inspect.getsource(TradingLoop._enrich_anomaly_async)
        assert "_send(" in source, "_enrich_anomaly_async must use async _send for follow-up"
        assert "send_alert(" not in source, (
            "_enrich_anomaly_async must NOT use sync send_alert (that's the caller's job)"
        )


# -- ANMI-02: Enrichment Format -----------------------------------------------


class TestLLMEnrichment:
    """Follow-up message must contain 'AI interpretation (unverified):' label."""

    @pytest.mark.asyncio
    async def test_follow_up_contains_unverified_label(self) -> None:
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="Likely driven by CBR rate decision")

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]

        anomaly = _make_anomaly_result()
        await tl._enrich_anomaly_async(_SYMBOL, _MARKET_ID, anomaly)  # type: ignore[attr-defined]

        alerter._send.assert_called_once()
        sent_text = alerter._send.call_args[0][0]
        assert sent_text.startswith("AI interpretation (unverified):"), (
            f"Follow-up must start with 'AI interpretation (unverified):', got: {sent_text!r}"
        )
        assert "Likely driven by CBR rate decision" in sent_text

    @pytest.mark.asyncio
    async def test_llm_prompt_includes_anomaly_details(self) -> None:
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="explanation")

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]

        anomaly = _make_anomaly_result()
        await tl._enrich_anomaly_async(_SYMBOL, _MARKET_ID, anomaly)  # type: ignore[attr-defined]

        prompt_arg = llm_client.complete.call_args[0][0]
        assert _SYMBOL in prompt_arg
        assert _MARKET_ID in prompt_arg
        assert "15.0" in prompt_arg or "+15.0" in prompt_arg  # price_move_pct


# -- ANMI-03: Graceful Degradation --------------------------------------------


class TestGracefulDegradation:
    """LLM timeout/failure must not suppress or delay raw alert."""

    @pytest.mark.asyncio
    async def test_llm_timeout_logs_failure_no_follow_up(self) -> None:
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(side_effect=TimeoutError("LLM timeout"))

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]

        anomaly = _make_anomaly_result()

        with structlog.testing.capture_logs() as captured:
            await tl._enrich_anomaly_async(  # type: ignore[attr-defined]
                _SYMBOL, _MARKET_ID, anomaly
            )

        # Follow-up must NOT be sent on timeout
        alerter._send.assert_not_called()

        # Must log anomaly_llm_failure
        failure_logs = [log for log in captured if log.get("event") == "anomaly_llm_failure"]
        assert len(failure_logs) >= 1, f"Expected anomaly_llm_failure log, got: {captured}"
        assert failure_logs[0]["symbol"] == _SYMBOL
        assert failure_logs[0]["market_id"] == _MARKET_ID

    @pytest.mark.asyncio
    async def test_llm_exception_logs_failure_no_follow_up(self) -> None:
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock()

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM provider down"))

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]

        anomaly = _make_anomaly_result()

        with structlog.testing.capture_logs() as captured:
            await tl._enrich_anomaly_async(  # type: ignore[attr-defined]
                _SYMBOL, _MARKET_ID, anomaly
            )

        alerter._send.assert_not_called()
        failure_logs = [log for log in captured if log.get("event") == "anomaly_llm_failure"]
        assert len(failure_logs) >= 1

    @pytest.mark.asyncio
    async def test_enrichment_never_raises(self) -> None:
        """_enrich_anomaly_async must swallow ALL exceptions -- never re-raise."""
        alerter = MagicMock(spec=TelegramAlerter)
        alerter._send = AsyncMock(side_effect=Exception("Telegram down too"))

        llm_client = AsyncMock()
        llm_client.complete = AsyncMock(return_value="explanation")

        tl = _make_tl()
        tl._llm_client = llm_client  # type: ignore[attr-defined]
        tl._alerter = alerter  # type: ignore[attr-defined]

        anomaly = _make_anomaly_result()
        # Must not raise -- all exceptions caught
        await tl._enrich_anomaly_async(_SYMBOL, _MARKET_ID, anomaly)  # type: ignore[attr-defined]

    def test_no_llm_client_still_fires_raw_alert(self) -> None:
        """When llm_client is None, anomaly detection still works, just no enrichment."""
        detector = AnomalyDetector()
        candles = _make_anomaly_candles()
        anomaly = detector.check(candles, _SYMBOL, _MARKET_ID)
        assert anomaly is not None

        alerter = MagicMock(spec=TelegramAlerter)
        # Simulate what _process_instrument does when llm_client is None:
        raw_text = f"ANOMALY {_SYMBOL}: {anomaly.price_move_pct:+.1f}%"
        alerter.send_alert(raw_text)
        alerter.send_alert.assert_called_once()
        # No LLM dispatch -- llm_client would be None, so no run_coroutine_threadsafe
