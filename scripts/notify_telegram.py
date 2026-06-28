"""Minimal Telegram notifier CLI for the autonomous audit loop.

Sends a single message to the operator's Telegram chat via the Bot API, using the
same credentials as the trading system (``FINALAYZE_TELEGRAM_BOT_TOKEN`` +
``FINALAYZE_TELEGRAM_CHAT_ID``/``FINALAYZE_TELEGRAM_ADMIN_CHAT_ID``). Standalone +
dependency-light so cron/headless runs can escalate without constructing the full
``TelegramAlerter``. Fail-soft: exits non-zero (never raises) when Telegram is
unconfigured or unreachable, so the caller can fall back to the report.

Usage:
    uv run python scripts/notify_telegram.py --title "..." --body "..." [--priority high]
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

_TIMEOUT_S = 10.0
_OK = 200
_PRIORITY_PREFIX = {"normal": "", "high": "⚠️ "}  # warning sign for high


def _resolve_chat_id() -> str | None:
    return (
        os.environ.get("FINALAYZE_TELEGRAM_CHAT_ID")
        or os.environ.get("FINALAYZE_TELEGRAM_ADMIN_CHAT_ID")
        or None
    )


def send(title: str, body: str, priority: str = "normal") -> int:
    """Send one Telegram message. Returns 0 on success, 1 on any failure (fail-soft)."""
    load_dotenv()
    token = os.environ.get("FINALAYZE_TELEGRAM_BOT_TOKEN")
    chat_id = _resolve_chat_id()
    if not token or not chat_id:
        print("telegram not configured (FINALAYZE_TELEGRAM_BOT_TOKEN/CHAT_ID unset) -- skipped")
        return 1
    prefix = _PRIORITY_PREFIX.get(priority, "")
    text = f"{prefix}*{title}*\n{body}"
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"telegram send failed: {exc}")
        return 1
    if resp.status_code != _OK:
        print(f"telegram send failed: HTTP {resp.status_code} {resp.text[:200]}")
        return 1
    print("telegram sent")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Send a Telegram message (autonomous-audit).")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--priority", choices=["normal", "high"], default="normal")
    args = parser.parse_args(argv)
    return send(args.title, args.body, args.priority)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
