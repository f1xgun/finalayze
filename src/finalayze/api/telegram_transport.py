"""Telegram HTTP transport (Layer 6).

Owns the persistent httpx.AsyncClient and DB alert persistence.
Has no knowledge of queues, priorities, or message formatting.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from finalayze.orchestration.db_persistence import TradingPersistence

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"
_SEND_MESSAGE_PATH = "/sendMessage"

_log = structlog.get_logger()


class TelegramTransport:
    """Async HTTP transport to the Telegram Bot API.

    Responsibilities:
      - Persistent httpx.AsyncClient (one per process lifetime)
      - DB alert persistence (queued → sent/failed envelope)
      - One async send() method; no thread bridging, no priority logic
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        persistence: TradingPersistence | None = None,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=10)
        self._persistence: TradingPersistence | None = persistence
        self._last_alert_ts: dict[uuid.UUID, datetime] = {}

    async def send(
        self,
        text: str,
        *,
        parse_mode: str = "HTML",
        alert_type: str = "generic",
        priority_name: str = "INFO",
        symbol: str | None = None,
        market_id: str | None = None,
        parent_id: uuid.UUID | None = None,
        alert_metadata: dict[str, object] | None = None,
    ) -> tuple[bool, uuid.UUID | None]:
        """POST a message to Telegram. Returns (ok, alert_id).

        Never raises — HTTP errors are caught and logged.
        Persists an alerts row before sending and updates status after.
        """
        alert_id = self._persist_before_send(
            text, alert_type, priority_name, symbol, market_id, parent_id, alert_metadata
        )

        if not self._token:
            self._update_status(alert_id, "sent")
            return (True, alert_id)

        url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
        payload: dict[str, str] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            resp = await self._client.post(url, json=payload)
            if resp.status_code == 429:  # noqa: PLR2004
                retry_after = resp.json().get("parameters", {}).get("retry_after", 30)
                _log.warning("telegram_rate_limited", retry_after=retry_after)
                self._update_status(alert_id, "failed")
                return (False, alert_id)
            self._update_status(alert_id, "sent")
            return (True, alert_id)
        except Exception:
            _log.exception("telegram_transport_send_failed")
            self._update_status(alert_id, "failed")
            return (False, alert_id)

    async def close(self) -> None:
        """Shut down the persistent httpx client. Idempotent."""
        await self._client.aclose()

    def _persist_before_send(
        self,
        text: str,
        alert_type: str,
        priority_name: str,
        symbol: str | None,
        market_id: str | None,
        parent_id: uuid.UUID | None,
        alert_metadata: dict[str, object] | None,
    ) -> uuid.UUID | None:
        if self._persistence is None:
            return None
        alert_id = uuid.uuid4()
        timestamp = datetime.now(tz=UTC)
        try:
            self._persistence.persist_alert(
                alert_id,
                timestamp,
                alert_type,
                priority_name,
                text,
                symbol=symbol,
                market_id=market_id,
                parent_id=parent_id,
                delivery_status="queued",
                alert_metadata=alert_metadata,
            )
            self._last_alert_ts[alert_id] = timestamp
        except Exception:
            _log.warning("alert_persist_before_send_failed", exc_info=True)
            return None
        return alert_id

    def _update_status(self, alert_id: uuid.UUID | None, delivery_status: str) -> None:
        if self._persistence is None or alert_id is None:
            return
        ts = self._last_alert_ts.pop(alert_id, None)
        if ts is None:
            return
        try:
            self._persistence.update_alert_status(alert_id, ts, delivery_status)
        except Exception:
            _log.warning("alert_status_update_failed", exc_info=True)
