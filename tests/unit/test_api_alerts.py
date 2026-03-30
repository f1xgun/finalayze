"""Tests for TelegramAlerter resilience (OPS-04)."""

from __future__ import annotations

from unittest.mock import patch

from finalayze.api.alerts import TelegramAlerter


def test_send_alert_noop_when_no_token() -> None:
    """OPS-04: alerter with empty token is a no-op, never raises."""
    alerter = TelegramAlerter(bot_token="", chat_id="123")
    # Should return immediately without error
    alerter.send_alert("test message")


def test_send_alert_suppresses_network_error() -> None:
    """OPS-04: network errors in send_alert must not propagate."""
    _fake_token = "fake:token"  # noqa: S105
    alerter = TelegramAlerter(bot_token=_fake_token, chat_id="123")
    with patch.object(alerter, "_send_sync", side_effect=ConnectionError("DNS failed")):
        # Must not raise
        alerter.send_alert("test message")
