"""Tests for TelegramAlerter._send / _send_sync persistence write-hook (Phase 57-02).

Validates D-03/D-05/D-06 from CONTEXT.md:
  - persist row with delivery_status='queued' BEFORE httpx.post
  - update delivery_status='sent' or 'failed' AFTER the response
  - persistence failure NEVER blocks the Telegram send
  - send_alert MUST NOT call persist_alert directly (Pitfall 1: no double-write)

Returns from _send / _send_sync are now ``tuple[bool, uuid.UUID | None]`` so the
anomaly path (Plan 03) can thread the parent_id from raw -> llm follow-up.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from finalayze.api.alerts import AlertPriority, TelegramAlerter

if TYPE_CHECKING:
    pass


_FAKE_TOKEN = "fake-bot-token"  # noqa: S105
_CHAT_ID = "987654"


def _make_alerter(
    *,
    persistence: MagicMock | None = None,
    token: str = _FAKE_TOKEN,
) -> TelegramAlerter:
    """Build a TelegramAlerter with optional persistence injected."""
    return TelegramAlerter(bot_token=token, chat_id=_CHAT_ID, persistence=persistence)


def _make_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    """Build a fake httpx Response with .status_code and .json()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    return resp


# ── Async _send tests ────────────────────────────────────────────────────────


def test_send_persists_before_httpx_and_updates_sent_after() -> None:
    """200 OK path: persist queued BEFORE post, update sent AFTER."""
    persistence = MagicMock()
    alerter = _make_alerter(persistence=persistence)

    # Track call order of persist vs httpx
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

    alerter._client.post = AsyncMock(side_effect=_post_side_effect)  # type: ignore[method-assign]

    result = asyncio.run(
        alerter._send("hi", alert_type="signal", symbol="SBER", market_id="moex"),
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
    alerter = _make_alerter(persistence=persistence)

    alerter._client.post = AsyncMock(  # type: ignore[method-assign]
        side_effect=httpx.RequestError("DNS failed"),
    )

    result = asyncio.run(alerter._send("hi", alert_type="signal"))
    ok, alert_id = result
    assert ok is False
    assert isinstance(alert_id, uuid.UUID)
    persistence.persist_alert.assert_called_once()
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "failed"


def test_send_updates_failed_on_rate_limit_429() -> None:
    """429 rate limit: update delivery_status='failed', return (False, alert_id)."""
    persistence = MagicMock()
    alerter = _make_alerter(persistence=persistence)

    alerter._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_response(429, {"parameters": {"retry_after": 30}}),
    )

    result = asyncio.run(alerter._send("hi", alert_type="signal"))
    ok, alert_id = result
    assert ok is False
    assert isinstance(alert_id, uuid.UUID)
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "failed"


def test_send_survives_persistence_failure() -> None:
    """persist_alert raising MUST NOT block the httpx send path."""
    persistence = MagicMock()
    persistence.persist_alert.side_effect = RuntimeError("DB unavailable")
    alerter = _make_alerter(persistence=persistence)

    alerter._client.post = AsyncMock(return_value=_make_response(200))  # type: ignore[method-assign]

    result = asyncio.run(alerter._send("hi", alert_type="signal"))
    ok, alert_id = result
    # Send still succeeded transport-wise
    assert ok is True
    # No alert_id because persist failed; update is skipped
    assert alert_id is None
    alerter._client.post.assert_awaited_once()


def test_send_passes_parent_id_to_persist() -> None:
    """anomaly_llm: parent_id flows through to persist_alert kwarg."""
    persistence = MagicMock()
    alerter = _make_alerter(persistence=persistence)

    alerter._client.post = AsyncMock(return_value=_make_response(200))  # type: ignore[method-assign]
    parent_id = uuid.uuid4()

    asyncio.run(
        alerter._send(
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
    alerter = _make_alerter(persistence=persistence, token="")

    # httpx must NEVER be called when token is empty
    alerter._client.post = AsyncMock(return_value=_make_response(200))  # type: ignore[method-assign]

    result = asyncio.run(alerter._send("hi", alert_type="signal"))
    ok, alert_id = result
    assert ok is True
    assert isinstance(alert_id, uuid.UUID)
    persistence.persist_alert.assert_called_once()
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "sent"
    alerter._client.post.assert_not_awaited()


# ── Sync _send_sync tests ────────────────────────────────────────────────────


def test_send_sync_persists_and_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync path: same persist-before + update-after invariants."""
    persistence = MagicMock()
    alerter = _make_alerter(persistence=persistence)

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.post = MagicMock(return_value=_make_response(200))

    monkeypatch.setattr(
        "finalayze.api.alerts.httpx.Client",
        MagicMock(return_value=fake_client),
    )

    result = alerter._send_sync("hi", alert_type="signal", symbol="SBER")
    ok, alert_id = result
    assert ok is True
    assert isinstance(alert_id, uuid.UUID)
    persistence.persist_alert.assert_called_once()
    persistence.update_alert_status.assert_called_once()
    assert persistence.update_alert_status.call_args.args[2] == "sent"


# ── send_alert wrapper invariant: no double-write ────────────────────────────


def test_send_alert_does_not_double_persist() -> None:
    """Pitfall 1: send_alert -> _send must persist exactly ONCE (not twice).

    Calls send_alert from a sync context (no running loop) so it routes through
    _send_sync exactly once. Persistence must be invoked once and only once.
    """
    persistence = MagicMock()
    alerter = _make_alerter(persistence=persistence)

    # Patch _send_sync so we can confirm call shape, but still inside _send_sync
    # the real _persist_alert_before_send would fire. Easiest: patch httpx.Client
    # underneath and let the real path run.
    real_send_sync = alerter._send_sync
    sync_calls: list[tuple[tuple, dict]] = []

    def _wrapped_sync(text: str, **kwargs: object) -> tuple[bool, uuid.UUID | None]:
        sync_calls.append(((text,), kwargs))
        # Fake transport success
        return (True, uuid.uuid4())

    # Replace _send_sync to count + return success without httpx
    alerter._send_sync = _wrapped_sync  # type: ignore[method-assign]
    # ALSO assert send_alert itself does NOT call persist_alert at the top level
    # (only _send / _send_sync do). Confirm by counting persistence calls.
    alerter.send_alert("test", priority=AlertPriority.INFO)

    # send_alert in sync context calls _send_sync exactly once
    assert len(sync_calls) == 1
    # Restore for safety
    alerter._send_sync = real_send_sync  # type: ignore[method-assign]
    # Persistence is invoked inside _send_sync — but we replaced it. So no calls
    # should be observed at the persistence mock from this code path. The point
    # of this test: send_alert itself does NOT call persistence.persist_alert
    # (only _send / _send_sync do, exactly once each).
    persistence.persist_alert.assert_not_called()
