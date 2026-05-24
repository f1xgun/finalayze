"""Tests for TelegramAlerter resilience (OPS-04)."""

from __future__ import annotations

from unittest.mock import MagicMock

from finalayze.api.alerts import AlertPriority, TelegramAlerter


def test_send_alert_noop_when_no_token() -> None:
    """OPS-04: alerter with empty token is a no-op, never raises."""
    alerter = TelegramAlerter(bot_token="", chat_id="123")
    alerter.send_alert("test message")


def test_send_alert_suppresses_queue_error() -> None:
    """OPS-04: exceptions from queue.post() must not propagate out of send_alert."""
    alerter = TelegramAlerter(bot_token="fake:token", chat_id="123")
    bad_queue = MagicMock()
    bad_queue.post.side_effect = RuntimeError("loop closed")
    alerter.set_queue(bad_queue)
    # Must not raise
    alerter.send_alert("test message", priority=AlertPriority.INFO)
