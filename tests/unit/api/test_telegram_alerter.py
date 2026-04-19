"""Tests for the enriched TelegramAlerter methods (Phase 57-02).

Covers:
  - on_stop_loss_triggered (D-09 enrichment): pnl_amount, pnl_pct, hold_bars,
    currency kw-only params with '—' rendering for None fields.
  - on_signal_generated (D-14 new method): top-3 strategy breakdown +
    (+N more) truncation + (NEW|ADD|FLIP) position context suffix.

Backwards compatibility: legacy positional-arg call to on_stop_loss_triggered
still works (covered by Test 1 + tests/unit/test_api_alerts.py).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.api.alerts import AlertPriority, TelegramAlerter

_FAKE_TOKEN = "fake-bot-token"  # noqa: S105
_CHAT_ID = "987654"


def _make_alerter() -> TelegramAlerter:
    """Build a TelegramAlerter with patched send_alert so we can inspect calls."""
    alerter = TelegramAlerter(bot_token=_FAKE_TOKEN, chat_id=_CHAT_ID)
    alerter.send_alert = MagicMock()  # type: ignore[method-assign]
    return alerter


# ── on_stop_loss_triggered (D-09 enrichment) ────────────────────────────────


def test_on_stop_loss_triggered_legacy_signature() -> None:
    """Legacy positional-arg call still works; missing P&L renders as '—'."""
    alerter = _make_alerter()

    alerter.on_stop_loss_triggered(
        "SBER",
        Decimal("100"),
        Decimal("95"),
        Decimal("92"),
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "SBER" in text
    assert "100" in text
    assert "95" in text
    assert "92" in text
    # Em-dash rendering for missing fields
    assert "—" in text


def test_on_stop_loss_triggered_enriched() -> None:
    """All kwargs present: pnl_amount, pnl_pct, hold_bars, currency rendered."""
    alerter = _make_alerter()

    alerter.on_stop_loss_triggered(
        "SBER",
        Decimal("100"),
        Decimal("95"),
        Decimal("92"),
        pnl_amount=Decimal("-80.50"),
        pnl_pct=-0.0805,
        hold_bars=12,
        currency="RUB",
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "₽-80.50" in text
    assert "-8.05%" in text
    assert "12 bars" in text


def test_on_stop_loss_triggered_pnl_positive_sign() -> None:
    """Positive pnl_amount renders explicit + sign; USD gets $ symbol."""
    alerter = _make_alerter()

    alerter.on_stop_loss_triggered(
        "AAPL",
        Decimal("100"),
        Decimal("95"),
        Decimal("96"),
        pnl_amount=Decimal("50.00"),
        pnl_pct=0.05,
        currency="USD",
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "$+50.00" in text
    assert "+5.00%" in text


def test_on_stop_loss_triggered_mixed_none() -> None:
    """Some fields present, some None: each missing field renders '—' independently."""
    alerter = _make_alerter()

    alerter.on_stop_loss_triggered(
        "AAPL",
        Decimal("100"),
        Decimal("95"),
        Decimal("96"),
        pnl_amount=Decimal("20"),
        pnl_pct=None,
        hold_bars=5,
        currency="USD",
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "$+20.00" in text
    # P&L pct slot is em-dash since pnl_pct=None
    assert "—" in text
    assert "5 bars" in text


def test_on_stop_loss_triggered_priority_important() -> None:
    """Stop-loss alerts route at AlertPriority.IMPORTANT."""
    alerter = _make_alerter()

    alerter.on_stop_loss_triggered(
        "SBER",
        Decimal("100"),
        Decimal("95"),
        Decimal("92"),
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    assert (
        alerter.send_alert.call_args.kwargs["priority"]  # type: ignore[attr-defined]
        is AlertPriority.IMPORTANT
    )


# ── on_signal_generated (D-14 new method) ───────────────────────────────────


def test_on_signal_generated_buy_body() -> None:
    """BUY signal renders green emoji + symbol + market_id + top strategies + conf + context."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "SBER",
        "moex",
        "BUY",
        0.58,
        [("momentum", 0.72), ("macd", 0.64), ("rsi", 0.51)],
        "NEW",
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    # Green circle U+1F7E2
    assert "\U0001f7e2" in text
    assert "BUY" in text
    assert "<b>SBER</b>" in text
    assert "[moex]" in text
    assert "momentum 0.72" in text
    assert "macd 0.64" in text
    assert "rsi 0.51" in text
    assert "0.58" in text
    assert "(NEW)" in text
    assert (
        alerter.send_alert.call_args.kwargs["priority"]  # type: ignore[attr-defined]
        is AlertPriority.INFO
    )


def test_on_signal_generated_sell_body() -> None:
    """SELL signal renders red emoji + SELL label."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "GAZP",
        "moex",
        "SELL",
        0.62,
        [("mean_reversion", 0.71)],
        "NEW",
    )

    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    # Red circle U+1F534
    assert "\U0001f534" in text
    assert "SELL" in text


def test_on_signal_generated_truncates_to_top3_plus_more() -> None:
    """5 strategies truncate to top-3 + '(+2 more)' marker."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "SBER",
        "moex",
        "BUY",
        0.55,
        [
            ("momentum", 0.72),
            ("macd", 0.64),
            ("rsi", 0.51),
            ("ichimoku", 0.45),
            ("hurst", 0.42),
        ],
        "NEW",
    )

    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "momentum 0.72" in text
    assert "macd 0.64" in text
    assert "rsi 0.51" in text
    assert "(+2 more)" in text
    # 4th and 5th strategies must NOT appear inline (they're behind +N)
    assert "ichimoku 0.45" not in text
    assert "hurst 0.42" not in text


def test_on_signal_generated_add_context() -> None:
    """ADD position context appears as suffix."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "SBER",
        "moex",
        "BUY",
        0.60,
        [("momentum", 0.72)],
        "ADD",
    )

    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert text.endswith("(ADD)")


def test_on_signal_generated_flip_context() -> None:
    """FLIP position context appears as suffix."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "SBER",
        "moex",
        "SELL",
        0.65,
        [("mean_reversion", 0.71)],
        "FLIP",
    )

    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert text.endswith("(FLIP)")


def test_on_signal_generated_empty_breakdown() -> None:
    """Empty strategy_breakdown does not crash; conf + context still rendered."""
    alerter = _make_alerter()

    alerter.on_signal_generated(
        "SBER",
        "moex",
        "BUY",
        0.58,
        [],
        "NEW",
    )

    alerter.send_alert.assert_called_once()  # type: ignore[attr-defined]
    text = alerter.send_alert.call_args.args[0]  # type: ignore[attr-defined]
    assert "0.58" in text
    assert "(NEW)" in text
