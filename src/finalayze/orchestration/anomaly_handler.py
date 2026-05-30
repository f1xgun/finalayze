"""Anomaly detection raw + LLM-enrichment orchestrator.

Extracted from TradingLoop to improve modularity and testability.
Handles the two-phase anomaly alert workflow:
  1. Fire raw anomaly alert via send_async (capture alert_id)
  2. Schedule async LLM enrichment with parent_id FK threading

Phase 57-04 Task 3 (D-04): parent_id FK threading into anomaly_llm alerts.
Moved from core/ to orchestration/ in Phase 22 (dependency layer cleanup).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from finalayze.api.alerts import TelegramAlerter

_log = structlog.get_logger()

_ANOMALY_SYSTEM_PROMPT = (
    "You are a financial market analyst. Explain the likely cause of "
    "the following statistical anomaly in 2-3 sentences. Be specific "
    "about potential catalysts (earnings, macro events, sector rotation). "
    "Do not give trading advice."
)
_ANOMALY_LLM_TIMEOUT = 30.0


class AnomalyHandler:
    """Orchestrate the anomaly raw + LLM-enrichment pair.

    The handler sends the raw alert via _alerter.send_async() to capture
    the raw_alert_id, then schedules the async LLM follow-up via
    asyncio.create_task(self.enrich(..., parent_id=raw_alert_id)) so the
    follow-up's persisted row carries the parent_id FK.

    The llm_client is read lazily via the provider function at call time,
    NOT captured at construction, so TradingLoop's late-initialization of
    llm_client (which is also used by _run_portfolio_review_async) is supported.
    """

    def __init__(
        self,
        alerter: TelegramAlerter,
        llm_client_provider: Callable[[], Any],
    ) -> None:
        """Initialize AnomalyHandler with dependencies.

        Args:
            alerter: TelegramAlerter for sending alerts.
            llm_client_provider: Callable that returns the current LLM client (or None).
                Invoked at handle/enrich call time, NOT at construction.
        """
        self._alerter = alerter
        self._llm_client_provider = llm_client_provider
        # GC anchor for the enrich task (was TradingLoop._anomaly_enrich_task)
        self._enrich_task: asyncio.Task[None] | None = None

    async def handle(
        self,
        symbol: str,
        market_id: str,
        anomaly: object,
        raw_text: str,
    ) -> uuid.UUID | None:
        """Orchestrate the anomaly raw + LLM-enrichment pair.

        Sends the raw alert via send_async(alert_type='anomaly_raw') to capture
        raw_alert_id, then schedules the async LLM follow-up via
        asyncio.create_task(self.enrich(..., parent_id=raw_alert_id)) so the
        follow-up's persisted row carries the parent_id FK.

        Returns:
            The captured raw alert id for caller-side tracking; None if raw send fails.
            Never raises.
        """
        try:
            _ok, raw_alert_id = await self._alerter.send_async(
                raw_text,
                alert_type="anomaly_raw",
                symbol=symbol,
                market_id=market_id,
                parent_id=None,
            )
        except Exception:
            _log.warning(
                "anomaly_raw_send_failed",
                symbol=symbol,
                market_id=market_id,
            )
            return None

        if self._llm_client_provider() is not None:
            # Fire-and-forget: store the task on the instance so the asyncio
            # GC doesn't drop it mid-flight (ruff RUF006). The reference is
            # purposely overwritten on the next anomaly so we don't leak a
            # growing list across cycles.
            self._enrich_task = asyncio.create_task(
                self.enrich(
                    symbol,
                    market_id,
                    anomaly,
                    parent_id=raw_alert_id,
                ),
            )
        return raw_alert_id

    async def enrich(
        self,
        symbol: str,
        market_id: str,
        anomaly: object,
        *,
        parent_id: uuid.UUID | None = None,
    ) -> None:
        """Fire-and-forget LLM enrichment -- never raises, never blocks raw alert.

        When parent_id is supplied (the alert_id returned by the prior raw-alert
        handle() call), the LLM follow-up send_async threads it through
        alert_type='anomaly_llm' so the persisted child row's FK to the parent
        anomaly_raw row is populated at insert time.
        """
        try:
            llm_client = self._llm_client_provider()
            if llm_client is None:
                return

            prompt = (
                f"Ticker: {symbol} ({market_id})\n"
                f"Price move: {anomaly.price_move_pct:+.1f}%\n"  # type: ignore[attr-defined]
                f"Sigma: {anomaly.sigma:.1f}\n"  # type: ignore[attr-defined]
                f"Volume ratio: {anomaly.volume_ratio:.1f}x average\n"  # type: ignore[attr-defined]
                f"Anomaly type: {anomaly.anomaly_type}"  # type: ignore[attr-defined]
            )
            explanation = await asyncio.wait_for(
                llm_client.complete(prompt, _ANOMALY_SYSTEM_PROMPT),
                timeout=_ANOMALY_LLM_TIMEOUT,
            )
            follow_up = f"AI interpretation (unverified): {explanation}"
            await self._alerter.send_async(
                follow_up,
                alert_type="anomaly_llm",
                symbol=symbol,
                market_id=market_id,
                parent_id=parent_id,
            )
        except Exception:
            _log.warning(
                "anomaly_llm_failure",
                symbol=symbol,
                market_id=market_id,
            )
