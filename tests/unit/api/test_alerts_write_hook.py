"""Tests for TelegramTransport.send() persistence write-hook (Phase 57-02).

After the AlertQueue refactor, persistence lives in TelegramTransport.send(),
not in TelegramAlerter. These tests validate D-03/D-05/D-06:
  - persist row with delivery_status='queued' BEFORE httpx.post
  - update delivery_status='sent' or 'failed' AFTER the response
  - persistence failure NEVER blocks the Telegram send
  - send_alert routes only through queue.post, never calls persist_alert directly
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from finalayze.api.alerts import AlertPriority, AlertQueue, TelegramAlerter
from finalayze.api.telegram_transport import TelegramTransport

_FAKE_TOKEN = "fake-bot-token"  # noqa: S105
_CHAT_ID = "987654"


def _make_transport(
    *,
    persistence: MagicMock | None = None,
    token: str = _FAKE_TOKEN,
) -> TelegramTransport:
    return TelegramTransport(bot_token=token, chat_id=_CHAT_ID, persistence=persistence)


def _make_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    return resp


# ── TelegramTransport.send() persistence write-hook ─────────────────────────


def test_send_persists_before_httpx_and_updates_sent_after() -> None:
    """200 OK path: persist queued BEFORE post, update sent AFTER."""
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence)

    call_order: list[str] = []

    def _persist_side_effect(*_args: object, **_kwargs: object) -> None:
        call_order.append("persist_alert")

    def _update_side_effect(*_args: object, **_kwargs: object) -> None:
        call_order.append("update_alert_status")

    persistence.persist_alert.side_effect = _persist_side_effect
    persistence.update_alert_status.side_effect = _update_side_effect

    async def _post_side_effect(*_args: object, **_kwargs: object) -> MagicMock:
        call_order.append("httpx.post")
        return _make_response(200)

    with patch.object(transport._client, "post", new=AsyncMock(side_effect=_post_side_effect)):
        result = asyncio.run(
            transport.send("hi", alert_type="signal", symbol="SBER", market_id="moex"),
        )

    assert call_order == ["persist_alert", "httpx.post", "update_alert_status"]
    persistence.persist_alert.assert_called_once()
    kw = persistence.persist_alert.call_args.kwargs
    assert kw["delivery_status"] == "queued"
    assert kw["symbol"] == "SBER"
    assert kw["market_id"] == "moex"
    persistence.update_alert_status.assert_called_once()
    update_args = persistence.update_alert_status.call_args.args
    assert update_args[2] == "sent"

    ok, alert_id = result
    assert ok is True
    assert isinstance(alert_id, uuid.UUID)


def test_send_updates_failed_on_httpx_exception() -> None:
    """Transport failure: update delivery_status='failed', return (False, alert_id)."""
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence)

    with patch.object(
        transport._client,
        "post",
        new=AsyncMock(side_effect=httpx.RequestError("DNS failed")),
    ):
        result = asyncio.run(transport.send("hi", alert_type="signal"))

    ok, alert_id = result
    assert ok is False
    assert isinstance(alert_id, uuid.UUID)
    persistence.persist_alert.assert_called_once()
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "failed"


def test_send_updates_failed_on_rate_limit_429() -> None:
    """429 rate limit: update delivery_status='failed', return (False, alert_id)."""
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence)

    with patch.object(
        transport._client,
        "post",
        new=AsyncMock(return_value=_make_response(429, {"parameters": {"retry_after": 30}})),
    ):
        result = asyncio.run(transport.send("hi", alert_type="signal"))

    ok, alert_id = result
    assert ok is False
    assert isinstance(alert_id, uuid.UUID)
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "failed"


def test_send_survives_persistence_failure() -> None:
    """persist_alert raising MUST NOT block the httpx send path."""
    persistence = MagicMock()
    persistence.persist_alert.side_effect = RuntimeError("DB unavailable")
    transport = _make_transport(persistence=persistence)

    with patch.object(
        transport._client, "post", new=AsyncMock(return_value=_make_response(200))
    ) as mock_post:
        result = asyncio.run(transport.send("hi", alert_type="signal"))

    ok, alert_id = result
    assert ok is True
    assert alert_id is None
    mock_post.assert_awaited_once()


def test_send_passes_parent_id_to_persist() -> None:
    """anomaly_llm: parent_id flows through to persist_alert kwarg."""
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence)
    parent_id = uuid.uuid4()

    with patch.object(transport._client, "post", new=AsyncMock(return_value=_make_response(200))):
        asyncio.run(
            transport.send(
                "LLM interpretation",
                alert_type="anomaly_llm",
                parent_id=parent_id,
            ),
        )

    persistence.persist_alert.assert_called_once()
    assert persistence.persist_alert.call_args.kwargs["parent_id"] == parent_id


def test_send_no_token_short_circuits_with_persistence() -> None:
    """No token: persist still records the attempt; status='sent' (no-op transport)."""
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence, token="")

    with patch.object(
        transport._client, "post", new=AsyncMock(return_value=_make_response(200))
    ) as mock_post:
        result = asyncio.run(transport.send("hi", alert_type="signal"))

    ok, alert_id = result
    assert ok is True
    assert isinstance(alert_id, uuid.UUID)
    persistence.persist_alert.assert_called_once()
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "sent"
    mock_post.assert_not_awaited()


# ── send_alert wrapper invariant: no double-write ────────────────────────────


def test_send_alert_does_not_call_persist_directly() -> None:
    """send_alert routes through queue.post(), never calls persist_alert itself.

    After the AlertQueue refactor, TelegramAlerter.send_alert() only enqueues
    via AlertQueue.post(). Persistence is exclusively TelegramTransport's
    concern. Verify that calling send_alert with a mock queue never touches
    persistence directly.
    """
    persistence = MagicMock()
    transport = _make_transport(persistence=persistence)

    loop = asyncio.new_event_loop()
    try:
        alerter = TelegramAlerter(bot_token=_FAKE_TOKEN, chat_id=_CHAT_ID)
        queue = AlertQueue(loop=loop, transport=transport)
        alerter.set_queue(queue)

        mock_queue = MagicMock()
        alerter.set_queue(mock_queue)

        alerter.send_alert("test", priority=AlertPriority.INFO)

        mock_queue.post.assert_called_once_with("test", AlertPriority.INFO)
        persistence.persist_alert.assert_not_called()
    finally:
        loop.close()
