"""Application settings loaded from environment variables.

See docs/architecture/OVERVIEW.md for configuration details.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

from finalayze.core.modes import RolloutPhase, WorkMode

if TYPE_CHECKING:
    from finalayze.risk.rollout import RolloutLimits


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

    # Analytics (Phase 55 TRAD-02 D-03 cost-threshold win definition).
    # Commission expressed in basis points per notional. Source: backtest/costs.py
    # MOEX_COSTS.commission_rate=0.0004 (=4 bps), US per-share rate normalizes
    # to ~1 bps at $500 avg fill (RESEARCH.md Open Q1).
    default_commission_bps_us: float = 1.0
    default_commission_bps_moex: float = 4.0
    default_slippage_cost_bps: float = 5.0

    # Risk
    kelly_fraction: float = 0.5
    stop_loss_atr_multiplier: float = 2.0
    circuit_breaker_l1: float = 0.05
    circuit_breaker_l2: float = 0.10
    circuit_breaker_l3: float = 0.15

    # LLM
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    llm_provider: Literal[
        "openrouter",
        "openai",
        "anthropic",
        "deepseek",
        "groq",
        "claude_code_headless",  # uses local `claude -p` CLI + Claude.ai subscription
    ] = "openrouter"
    llm_api_key: str = ""  # API key for selected provider
    llm_max_rpm: int = 0  # FINALAYZE_LLM_MAX_RPM — max requests/min (0 = unlimited)
    # Fallback LLM: used when primary provider returns rate limit errors
    llm_fallback_provider: str = ""  # FINALAYZE_LLM_FALLBACK_PROVIDER
    llm_fallback_model: str = ""  # FINALAYZE_LLM_FALLBACK_MODEL
    llm_fallback_api_key: str = ""  # FINALAYZE_LLM_FALLBACK_API_KEY

    # Cycle intervals (restart required to apply changes)
    news_cycle_minutes: int = 2  # FINALAYZE_NEWS_CYCLE_MINUTES  # TODO: revert to 30
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

    # Bond layers
    bond_capital: float = 1_000_000.0  # FINALAYZE_BOND_CAPITAL (RUB)
    bond_cycle_enabled: bool = True  # FINALAYZE_BOND_CYCLE_ENABLED
    bond_cycle_minutes: int = 1440  # FINALAYZE_BOND_CYCLE_MINUTES (default daily)

    # Telegram extensions (Plan 03 prep)
    telegram_webhook_secret: str = ""  # FINALAYZE_TELEGRAM_WEBHOOK_SECRET
    telegram_allowed_chat_ids: list[str] = []  # FINALAYZE_TELEGRAM_ALLOWED_CHAT_IDS
    telegram_admin_chat_id: str = ""  # FINALAYZE_TELEGRAM_ADMIN_CHAT_ID
    weekly_digest_hour_utc: int = 16  # FINALAYZE_WEEKLY_DIGEST_HOUR_UTC (Sunday 19:00 MSK)

    # Kill switch
    kill_switch_flag_path: str = "/tmp/finalayze_killed"  # noqa: S108  # FINALAYZE_KILL_SWITCH_FLAG_PATH

    # News pipeline (Phase 7)
    news_rss_urls: list[str] = [
        # Official RU
        "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
        "https://www.interfax.ru/rss.asp",
        "https://tass.com/rss/v2.xml",
        # Business / finance RU
        "https://www.banki.ru/xml/news.rss",
        "https://www.vedomosti.ru/rss/news",
        "https://www.kommersant.ru/RSS/news.xml",
        # Global / US markets
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.investing.com/rss/news.rss",
    ]
    news_poll_interval_minutes: int = 5  # FINALAYZE_NEWS_POLL_INTERVAL_MINUTES
    telegram_channels: list[str] = [  # FINALAYZE_TELEGRAM_CHANNELS
        "@markettwits",
        "@AK47pfl",
        "@cbrstocks",
        "@investorbiz",
        "@raborynok",
    ]

    # Rollout
    rollout_phase: RolloutPhase = RolloutPhase.FULL  # FINALAYZE_ROLLOUT_PHASE

    # Safety
    real_confirmed: bool = False

    # CORS
    cors_origins: list[str] = []  # FINALAYZE_CORS_ORIGINS (comma-separated)

    # API auth
    api_key: str = ""  # FINALAYZE_API_KEY — set in production
    real_token: str = ""  # FINALAYZE_REAL_TOKEN — required to switch to REAL mode via API

    model_config = {"env_prefix": "FINALAYZE_", "env_file": ".env", "extra": "ignore"}

    def effective_risk_limits(self) -> RolloutLimits:
        """Return risk limits for the current rollout phase."""
        from finalayze.risk.rollout import ROLLOUT_LIMITS  # noqa: PLC0415

        return ROLLOUT_LIMITS[self.rollout_phase]

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> Settings:
        """Ensure required keys are set for non-DEBUG/TEST modes."""
        import os  # noqa: PLC0415

        # Sandbox safety: default to MINIMAL rollout unless explicitly overridden
        if self.mode == WorkMode.SANDBOX and not os.environ.get("FINALAYZE_ROLLOUT_PHASE"):
            self.rollout_phase = RolloutPhase.MINIMAL

        # DEBUG, TEST, and SANDBOX modes skip credential validation
        # (SANDBOX uses stubs for missing services)
        if self.mode in (WorkMode.DEBUG, WorkMode.TEST, WorkMode.SANDBOX):
            if not self.database_url:
                self.database_url = "postgresql+asyncpg://finalayze:secret@localhost:5432/finalayze"
            return self
        # Non-DEBUG/TEST/SANDBOX modes require an explicit database URL
        if not self.database_url:
            raise ValueError("FINALAYZE_DATABASE_URL is required for non-DEBUG/TEST modes")
        # REAL mode needs a live LLM
        if not self.llm_api_key and not self.anthropic_api_key:
            raise ValueError("llm_api_key (or anthropic_api_key) is required for REAL mode")
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
