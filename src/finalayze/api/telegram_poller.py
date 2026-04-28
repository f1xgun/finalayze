"""Long-polling Telegram update receiver.

Runs as a background asyncio task inside the FastAPI lifespan.
Calls ``getUpdates`` with timeout=30 (long-polling) and dispatches
each update to ``TelegramBotHandler.handle_update``.

Calls ``deleteWebhook`` on start so getUpdates and webhook don't conflict.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from finalayze.api.telegram_bot import TelegramBotHandler

_log = structlog.get_logger()

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_LONG_POLL_TIMEOUT = 30  # seconds — Telegram holds connection open
_CLIENT_TIMEOUT = _LONG_POLL_TIMEOUT + 10  # httpx must be longer than poll timeout
_ERROR_BACKOFF = 5  # seconds to wait after network error


class TelegramPoller:
    """Background long-poller for Telegram bot commands.

    Usage::
        poller = TelegramPoller(token, bot_handler)
        await poller.start()   # inside lifespan
        ...
        await poller.stop()    # on shutdown
    """

    def __init__(self, token: str, handler: TelegramBotHandler) -> None:
        self._token = token
        self._handler = handler
        self._offset = 0
        self._task: asyncio.Task[None] | None = None

    def _url(self, method: str) -> str:
        return _API_BASE.format(token=self._token, method=method)

    async def start(self) -> None:
        """Delete any registered webhook, then start the poll loop."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(self._url("deleteWebhook"))
        except Exception:
            _log.warning("telegram_poller_delete_webhook_failed", exc_info=True)

        self._task = asyncio.create_task(self._poll_loop(), name="telegram-poller")
        _log.info("telegram_poller_started")

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        _log.info("telegram_poller_stopped")

    async def _poll_loop(self) -> None:
        async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
            while True:
                try:
                    resp = await client.get(
                        self._url("getUpdates"),
                        params={"offset": self._offset, "timeout": _LONG_POLL_TIMEOUT},
                    )
                    data: dict[str, Any] = resp.json()
                    for update in data.get("result", []):
                        self._offset = update["update_id"] + 1
                        try:
                            await self._handler.handle_update(update)
                        except Exception:
                            _log.warning("telegram_poller_dispatch_failed", exc_info=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _log.warning("telegram_poller_getUpdates_failed", exc_info=True)
                    await asyncio.sleep(_ERROR_BACKOFF)
