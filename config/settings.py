"""Application settings loaded from environment variables.

See docs/architecture/OVERVIEW.md for configuration details.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings

from config.modes import WorkMode


class Settings(BaseSettings):
    """Global application settings.

    All values can be overridden via environment variables
    prefixed with ``FINALAYZE_``.
    """

    # Core
    mode: WorkMode = WorkMode.DEBUG
    base_currency: str = "USD"
    database_url: str = "postgresql+asyncpg://finalayze:secret@localhost:5432/finalayze"
    redis_url: str = "redis://localhost:6379/0"

    # API Keys
    finnhub_api_key: str = ""
    newsapi_api_key: str = ""
    anthropic_api_key: str = ""

    # Alpaca (US)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # Tinkoff (MOEX)
    tinkoff_token: str = ""
    tinkoff_sandbox: bool = True

    # Per-market trading limits
    alpaca_max_portfolio_value: float = 10_000
    tinkoff_max_portfolio_value: float = 500_000

    # Global risk
    max_positions_per_market: int = 10
    max_position_pct: float = 0.20
    daily_loss_limit_pct: float = 0.02
    max_cross_market_exposure_pct: float = 0.80

    # Risk
    kelly_fraction: float = 0.5
    stop_loss_atr_multiplier: float = 2.0
    circuit_breaker_l1: float = 0.05
    circuit_breaker_l2: float = 0.10
    circuit_breaker_l3: float = 0.15

    # LLM
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    llm_provider: Literal["openrouter", "openai", "anthropic"] = "openrouter"
    llm_api_key: str = ""  # API key for selected provider

    # Cycle intervals (restart required to apply changes)
    news_cycle_minutes: int = 30  # FINALAYZE_NEWS_CYCLE_MINUTES
    strategy_cycle_minutes: int = 60  # FINALAYZE_STRATEGY_CYCLE_MINUTES
    daily_reset_hour_utc: int = 0  # FINALAYZE_DAILY_RESET_HOUR_UTC

    # Telegram alerting
    telegram_bot_token: str = ""  # FINALAYZE_TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""  # FINALAYZE_TELEGRAM_CHAT_ID

    # Safety
    real_confirmed: bool = False

    # API auth
    api_key: str = "change-me-in-production"  # FINALAYZE_API_KEY
    real_token: str = ""  # FINALAYZE_REAL_TOKEN — required to switch to REAL mode via API

    model_config = {"env_prefix": "FINALAYZE_", "env_file": ".env"}
