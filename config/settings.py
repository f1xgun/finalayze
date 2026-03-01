"""Application settings loaded from environment variables.

See docs/architecture/OVERVIEW.md for configuration details.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

from finalayze.core.modes import WorkMode


class Settings(BaseSettings):
    """Global application settings.

    All values can be overridden via environment variables
    prefixed with ``FINALAYZE_``.
    """

    # Core
    mode: WorkMode = WorkMode.DEBUG
    base_currency: str = "USD"
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"

    # DB pool
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

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

    # ML
    ml_enabled: bool = False  # opt-in
    ml_retrain_interval_hours: int = 168  # weekly
    ml_model_dir: str = "models/"
    ml_min_train_samples: int = 252  # ~1 year of daily bars
    ml_model_hmac_key: str = ""  # FINALAYZE_ML_MODEL_HMAC_KEY — for model integrity

    # FX
    fx_update_interval_minutes: int = 60  # FINALAYZE_FX_UPDATE_INTERVAL_MINUTES

    # Safety
    real_confirmed: bool = False

    # CORS
    cors_origins: list[str] = []  # FINALAYZE_CORS_ORIGINS (comma-separated)

    # API auth
    api_key: str = ""  # FINALAYZE_API_KEY — set in production
    real_token: str = ""  # FINALAYZE_REAL_TOKEN — required to switch to REAL mode via API

    model_config = {"env_prefix": "FINALAYZE_", "env_file": ".env"}

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> Settings:
        """Ensure required keys are set for non-DEBUG/TEST modes."""
        # DEBUG and TEST modes skip credential validation (no live services needed)
        if self.mode in (WorkMode.DEBUG, WorkMode.TEST):
            if not self.database_url:
                self.database_url = "postgresql+asyncpg://finalayze:secret@localhost:5432/finalayze"
            return self
        # Non-DEBUG/TEST modes require an explicit database URL
        if not self.database_url:
            raise ValueError("FINALAYZE_DATABASE_URL is required for non-DEBUG/TEST modes")
        # All non-DEBUG modes need a live LLM
        if not self.llm_api_key and not self.anthropic_api_key:
            raise ValueError("llm_api_key (or anthropic_api_key) is required for non-DEBUG mode")
        # REAL mode additionally requires broker credentials
        if self.mode == WorkMode.REAL:
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                raise ValueError("alpaca_api_key and alpaca_secret_key are required for REAL mode")
            if not self.real_confirmed:
                raise ValueError("real_confirmed must be True for REAL mode")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide Settings singleton (cached after first call).

    Use this instead of instantiating ``Settings()`` directly at module import
    time, so that environment variables injected by tests or deployment tools
    are picked up correctly.
    """
    return Settings()
