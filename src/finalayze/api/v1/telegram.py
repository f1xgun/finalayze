"""Telegram webhook endpoint for bot commands (Layer 6 - API).

Single POST endpoint that validates the webhook secret token
and delegates to TelegramBotHandler for command processing.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    from finalayze.api.telegram_bot import TelegramBotHandler


def create_telegram_router(
    bot_handler: TelegramBotHandler,
    webhook_secret: str,
) -> APIRouter:
    """Create a FastAPI router for the Telegram webhook.

    Args:
        bot_handler: The TelegramBotHandler instance for command dispatch.
        webhook_secret: Expected value of X-Telegram-Bot-Api-Secret-Token header.

    Returns:
        APIRouter with /api/telegram/webhook POST endpoint.
    """
    router = APIRouter()

    @router.post("/api/telegram/webhook")
    async def telegram_webhook(request: Request) -> dict[str, str]:
        """Receive Telegram webhook updates.

        Validates the secret token header. Parses the JSON body.
        Delegates to bot_handler.handle_update().
        """
        secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if secret != webhook_secret:
            raise HTTPException(status_code=403, detail="Invalid secret token")

        try:
            update = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        return await bot_handler.handle_update(update)

    return router
