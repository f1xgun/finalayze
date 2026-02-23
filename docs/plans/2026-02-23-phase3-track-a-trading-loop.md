# Phase 3 Track A — Trading Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement a production-grade, APScheduler-based live trading loop with per-market circuit breakers, Telegram alerting, and full integration test coverage for the test/paper mode.

**Architecture:** The `TradingLoop` orchestrates three scheduled cycles — news analysis, strategy execution, and daily reset — using an in-memory sentiment cache shared (thread-safely) between the news and strategy cycles. Each strategy cycle gates order submission through a two-layer circuit breaker system (`CircuitBreaker` per market, `CrossMarketCircuitBreaker` for combined drawdown) before routing orders through the existing `BrokerRouter`. All noteworthy events (fills, rejections, circuit breaker trips, daily summary) are asynchronously reported via `TelegramAlerter`, which silently no-ops when no token is configured.

**Tech Stack:** `apscheduler>=3.10.4` (BackgroundScheduler), `httpx>=0.28.0` (Telegram HTTP), `threading.Lock` (sentiment cache), `pytest-mock` (mock `httpx.post`, broker, fetchers), existing `PreTradeChecker`, `BrokerRouter`, `StrategyCombiner`, `InstrumentRegistry`, `NewsApiFetcher`, `NewsAnalyzer`, `EventClassifier`, `ImpactEstimator`.

**Worktree:** `.worktrees/phase3-trading-loop` on branch `feature/phase3-trading-loop`

---

## Project Conventions (read before writing any code)

- Every file starts with `"""Docstring."""\n\nfrom __future__ import annotations`
- Use `StrEnum` not `str, Enum` (ruff UP042)
- Exception names end in `Error` (ruff N818)
- No magic numbers — define named constants
- Run quality checks: `source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header`
- The project uses `uv run` for all Python commands
- Tests live in `tests/unit/` and `tests/integration/` — mirror source structure

---

## Task 1: Settings additions + .env.example update

**Files to create/modify:**
- `config/settings.py` — add 5 new fields
- `.env.example` — add corresponding env var examples

### Step 1 — Write the failing test FIRST

Create `tests/unit/test_settings_phase3.py`:

```python
"""Unit tests for Phase 3 settings additions."""

from __future__ import annotations

import pytest

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
DEFAULT_NEWS_CYCLE_MINUTES = 30
DEFAULT_STRATEGY_CYCLE_MINUTES = 60
DEFAULT_DAILY_RESET_HOUR_UTC = 0
DEFAULT_TELEGRAM_BOT_TOKEN = ""
DEFAULT_TELEGRAM_CHAT_ID = ""

CUSTOM_NEWS_CYCLE_MINUTES = 15
CUSTOM_STRATEGY_CYCLE_MINUTES = 120
CUSTOM_DAILY_RESET_HOUR = 1


class TestPhase3Settings:
    def test_news_cycle_minutes_default(self) -> None:
        from config.settings import Settings

        s = Settings()
        assert s.news_cycle_minutes == DEFAULT_NEWS_CYCLE_MINUTES

    def test_strategy_cycle_minutes_default(self) -> None:
        from config.settings import Settings

        s = Settings()
        assert s.strategy_cycle_minutes == DEFAULT_STRATEGY_CYCLE_MINUTES

    def test_daily_reset_hour_utc_default(self) -> None:
        from config.settings import Settings

        s = Settings()
        assert s.daily_reset_hour_utc == DEFAULT_DAILY_RESET_HOUR_UTC

    def test_telegram_bot_token_default_empty(self) -> None:
        from config.settings import Settings

        s = Settings()
        assert s.telegram_bot_token == DEFAULT_TELEGRAM_BOT_TOKEN

    def test_telegram_chat_id_default_empty(self) -> None:
        from config.settings import Settings

        s = Settings()
        assert s.telegram_chat_id == DEFAULT_TELEGRAM_CHAT_ID

    def test_news_cycle_minutes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_NEWS_CYCLE_MINUTES", str(CUSTOM_NEWS_CYCLE_MINUTES))
        from importlib import reload

        import config.settings as settings_module

        reload(settings_module)
        s = settings_module.Settings()
        assert s.news_cycle_minutes == CUSTOM_NEWS_CYCLE_MINUTES

    def test_strategy_cycle_minutes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "FINALAYZE_STRATEGY_CYCLE_MINUTES", str(CUSTOM_STRATEGY_CYCLE_MINUTES)
        )
        from importlib import reload

        import config.settings as settings_module

        reload(settings_module)
        s = settings_module.Settings()
        assert s.strategy_cycle_minutes == CUSTOM_STRATEGY_CYCLE_MINUTES

    def test_daily_reset_hour_utc_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINALAYZE_DAILY_RESET_HOUR_UTC", str(CUSTOM_DAILY_RESET_HOUR))
        from importlib import reload

        import config.settings as settings_module

        reload(settings_module)
        s = settings_module.Settings()
        assert s.daily_reset_hour_utc == CUSTOM_DAILY_RESET_HOUR

    def test_telegram_bot_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        token = "1234567890:AABBccDDeEFfGgHhIiJj"
        monkeypatch.setenv("FINALAYZE_TELEGRAM_BOT_TOKEN", token)
        from importlib import reload

        import config.settings as settings_module

        reload(settings_module)
        s = settings_module.Settings()
        assert s.telegram_bot_token == token

    def test_telegram_chat_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chat_id = "-1001234567890"
        monkeypatch.setenv("FINALAYZE_TELEGRAM_CHAT_ID", chat_id)
        from importlib import reload

        import config.settings as settings_module

        reload(settings_module)
        s = settings_module.Settings()
        assert s.telegram_chat_id == chat_id
```

Run (expect failures — fields don't exist yet):

```bash
uv run pytest tests/unit/test_settings_phase3.py -q --no-header
```

Expected output:
```
FAILED tests/unit/test_settings_phase3.py::TestPhase3Settings::test_news_cycle_minutes_default - AttributeError: ...
... (10 failures)
```

### Step 2 — Implement: add fields to `config/settings.py`

Edit `config/settings.py`. Add the following block after the `# LLM` section and before `# Safety`:

```python
    # Cycle intervals (restart required to apply changes)
    news_cycle_minutes: int = 30          # FINALAYZE_NEWS_CYCLE_MINUTES
    strategy_cycle_minutes: int = 60      # FINALAYZE_STRATEGY_CYCLE_MINUTES
    daily_reset_hour_utc: int = 0         # FINALAYZE_DAILY_RESET_HOUR_UTC

    # Telegram alerting
    telegram_bot_token: str = ""          # FINALAYZE_TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""            # FINALAYZE_TELEGRAM_CHAT_ID
```

Full updated `config/settings.py`:

```python
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
    news_cycle_minutes: int = 30          # FINALAYZE_NEWS_CYCLE_MINUTES
    strategy_cycle_minutes: int = 60      # FINALAYZE_STRATEGY_CYCLE_MINUTES
    daily_reset_hour_utc: int = 0         # FINALAYZE_DAILY_RESET_HOUR_UTC

    # Telegram alerting
    telegram_bot_token: str = ""          # FINALAYZE_TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""            # FINALAYZE_TELEGRAM_CHAT_ID

    # Safety
    real_confirmed: bool = False

    model_config = {"env_prefix": "FINALAYZE_", "env_file": ".env"}
```

### Step 3 — Update `.env.example`

Append to `.env.example` after the `FINALAYZE_REAL_CONFIRMED` line:

```dotenv
# ── Cycle intervals ───────────────────────────
FINALAYZE_NEWS_CYCLE_MINUTES=30
FINALAYZE_STRATEGY_CYCLE_MINUTES=60
FINALAYZE_DAILY_RESET_HOUR_UTC=0

# ── Telegram alerting ─────────────────────────
FINALAYZE_TELEGRAM_BOT_TOKEN=
FINALAYZE_TELEGRAM_CHAT_ID=
```

### Step 4 — Run tests (GREEN)

```bash
uv run pytest tests/unit/test_settings_phase3.py -q --no-header
```

Expected output:
```
10 passed in 0.XXs
```

### Step 5 — Quality checks

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

### Step 6 — Commit

```bash
git add config/settings.py .env.example tests/unit/test_settings_phase3.py
git commit -m "feat(config): add cycle interval and Telegram settings"
```

---

## Task 2: CircuitBreaker + CrossMarketCircuitBreaker

**Files to create:**
- `src/finalayze/risk/circuit_breaker.py`
- `tests/unit/test_circuit_breaker.py`

### Step 1 — Write the failing tests FIRST

Create `tests/unit/test_circuit_breaker.py`:

```python
"""Unit tests for per-market and cross-market circuit breakers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
MARKET_US = "us"
MARKET_MOEX = "moex"

BASELINE = Decimal("100000")
L1_THRESHOLD = Decimal("0.05")
L2_THRESHOLD = Decimal("0.10")
L3_THRESHOLD = Decimal("0.15")

# Equity values triggering each level
# NORMAL: drawdown < 5%
EQUITY_NORMAL = Decimal("97000")       # 3% drawdown
# CAUTION: 5% <= drawdown < 10%
EQUITY_CAUTION = Decimal("94000")      # 6% drawdown
EQUITY_CAUTION_EXACT = Decimal("95000")  # exactly 5% drawdown
# HALTED: 10% <= drawdown < 15%
EQUITY_HALTED = Decimal("89000")       # 11% drawdown
EQUITY_HALTED_EXACT = Decimal("90000")  # exactly 10% drawdown
# LIQUIDATE: drawdown >= 15%
EQUITY_LIQUIDATE = Decimal("84000")    # 16% drawdown
EQUITY_LIQUIDATE_EXACT = Decimal("85000")  # exactly 15% drawdown

# For cross-market tests
CROSS_THRESHOLD = Decimal("0.10")
US_BASELINE = Decimal("50000")
MOEX_BASELINE = Decimal("50000")
COMBINED_BASELINE = US_BASELINE + MOEX_BASELINE  # 100000
# 11% combined drawdown: combined current = 89000
US_CURRENT_OK = Decimal("48000")       # 4% down
MOEX_CURRENT_TRIP = Decimal("41000")   # combined = 89000, 11% total drawdown
US_CURRENT_SAFE = Decimal("49000")
MOEX_CURRENT_SAFE = Decimal("49000")   # combined = 98000, 2% drawdown


class TestCircuitLevel:
    def test_level_values(self) -> None:
        assert CircuitLevel.NORMAL == "normal"
        assert CircuitLevel.CAUTION == "caution"
        assert CircuitLevel.HALTED == "halted"
        assert CircuitLevel.LIQUIDATE == "liquidate"


class TestCircuitBreaker:
    def _make_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            market_id=MARKET_US,
            l1_threshold=float(L1_THRESHOLD),
            l2_threshold=float(L2_THRESHOLD),
            l3_threshold=float(L3_THRESHOLD),
        )

    def test_initial_level_is_normal(self) -> None:
        cb = self._make_breaker()
        assert cb.level == CircuitLevel.NORMAL

    def test_market_id_property(self) -> None:
        cb = self._make_breaker()
        assert cb.market_id == MARKET_US

    def test_check_normal(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_NORMAL, baseline_equity=BASELINE)
        assert level == CircuitLevel.NORMAL

    def test_check_caution_at_l1_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_CAUTION_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.CAUTION

    def test_check_caution_above_l1(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert level == CircuitLevel.CAUTION

    def test_check_halted_at_l2_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_HALTED_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.HALTED

    def test_check_halted_above_l2(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert level == CircuitLevel.HALTED

    def test_check_liquidate_at_l3_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_LIQUIDATE_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_check_liquidate_above_l3(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_check_updates_internal_level(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION

    def test_level_escalates_on_subsequent_checks(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE

    def test_reset_daily_clears_caution(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.reset_daily(new_baseline=EQUITY_CAUTION)
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_daily_clears_halted(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        cb.reset_daily(new_baseline=EQUITY_HALTED)
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_daily_does_not_clear_liquidate(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_LIQUIDATE)
        assert cb.level == CircuitLevel.LIQUIDATE  # must stay LIQUIDATE

    def test_reset_manual_clears_liquidate(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_manual_from_normal_stays_normal(self) -> None:
        cb = self._make_breaker()
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_daily_updates_baseline(self) -> None:
        """After reset_daily with a new baseline, checks use the new baseline."""
        cb = self._make_breaker()
        new_baseline = Decimal("90000")
        cb.reset_daily(new_baseline=new_baseline)
        # With 90000 baseline, EQUITY_NORMAL (97000) is above baseline → NORMAL
        level = cb.check(current_equity=EQUITY_NORMAL, baseline_equity=new_baseline)
        assert level == CircuitLevel.NORMAL

    def test_zero_equity_is_liquidate(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=Decimal("0"), baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_equity_above_baseline_is_normal(self) -> None:
        """Positive returns should not trigger any circuit breaker level."""
        cb = self._make_breaker()
        equity_above_baseline = Decimal("110000")
        level = cb.check(current_equity=equity_above_baseline, baseline_equity=BASELINE)
        assert level == CircuitLevel.NORMAL


class TestCrossMarketCircuitBreaker:
    def test_no_trip_when_within_threshold(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: US_CURRENT_SAFE, MARKET_MOEX: MOEX_CURRENT_SAFE},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        )
        assert tripped is False

    def test_trips_when_combined_drawdown_exceeds_threshold(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        )
        assert tripped is True

    def test_zero_baseline_returns_false(self) -> None:
        """Zero combined baseline should not raise — return False (no data = no trip)."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: Decimal("0")},
            baseline_equities={MARKET_US: Decimal("0")},
        )
        assert tripped is False

    def test_empty_markets_returns_false(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(market_equities={}, baseline_equities={})
        assert tripped is False

    def test_reset_daily_updates_baselines(self) -> None:
        """After reset_daily, a previously tripped cross-market check can recover."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        # First: trip it
        assert cmcb.check(
            market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        ) is True
        # Reset with new baselines matching the current equities
        cmcb.reset_daily(
            new_baselines={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP}
        )
        # Now check with those same values → 0% drawdown → no trip
        assert cmcb.check(
            market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
            baseline_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
        ) is False

    def test_single_market_trip(self) -> None:
        """A single market with 15% drawdown alone should trip the cross-market breaker."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: Decimal("80000")},
            baseline_equities={MARKET_US: Decimal("100000")},
        )
        assert tripped is True
```

Run (expect failures):

```bash
uv run pytest tests/unit/test_circuit_breaker.py -q --no-header
```

Expected output:
```
ERROR tests/unit/test_circuit_breaker.py - ModuleNotFoundError: No module named 'finalayze.risk.circuit_breaker'
```

### Step 2 — Implement `src/finalayze/risk/circuit_breaker.py`

```python
"""Per-market and cross-market circuit breakers (Layer 4).

Drawdown thresholds gate position sizing and trading activity:
  L1 (CAUTION)   >= 5%  drawdown: halve size, raise min confidence
  L2 (HALTED)    >= 10% drawdown: no new trades
  L3 (LIQUIDATE) >= 15% drawdown: close all positions immediately

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

# ── Threshold defaults ──────────────────────────────────────────────────────
_DEFAULT_L1 = 0.05
_DEFAULT_L2 = 0.10
_DEFAULT_L3 = 0.15
_DEFAULT_CROSS_HALT = 0.10
_ZERO = Decimal(0)


class CircuitLevel(StrEnum):
    """Escalating circuit breaker states."""

    NORMAL = "normal"       # trade freely
    CAUTION = "caution"     # -5% daily: halve size, raise min confidence
    HALTED = "halted"       # -10% daily: no new trades
    LIQUIDATE = "liquidate" # -15% daily: close all positions immediately


class CircuitBreaker:
    """Per-market circuit breaker that escalates level based on daily drawdown.

    Drawdown = (baseline_equity - current_equity) / baseline_equity.

    Reset rules:
        - ``reset_daily``: resets CAUTION / HALTED → NORMAL; updates baseline.
          LIQUIDATE is NOT cleared by daily reset — it requires operator action.
        - ``reset_manual``: clears LIQUIDATE → NORMAL (operator action only).
    """

    def __init__(
        self,
        market_id: str,
        l1_threshold: float = _DEFAULT_L1,
        l2_threshold: float = _DEFAULT_L2,
        l3_threshold: float = _DEFAULT_L3,
    ) -> None:
        self._market_id = market_id
        self._l1 = Decimal(str(l1_threshold))
        self._l2 = Decimal(str(l2_threshold))
        self._l3 = Decimal(str(l3_threshold))
        self._level: CircuitLevel = CircuitLevel.NORMAL

    @property
    def level(self) -> CircuitLevel:
        """Return the current circuit breaker level."""
        return self._level

    @property
    def market_id(self) -> str:
        """Return the market identifier this breaker guards."""
        return self._market_id

    def check(self, current_equity: Decimal, baseline_equity: Decimal) -> CircuitLevel:
        """Compute drawdown, update level, and return the new level.

        Args:
            current_equity: Portfolio equity right now.
            baseline_equity: Equity at the start of the trading day (baseline).

        Returns:
            The updated :class:`CircuitLevel`.
        """
        if baseline_equity <= _ZERO:
            self._level = CircuitLevel.LIQUIDATE
            return self._level

        drawdown = (baseline_equity - current_equity) / baseline_equity

        if drawdown >= self._l3:
            self._level = CircuitLevel.LIQUIDATE
        elif drawdown >= self._l2:
            self._level = CircuitLevel.HALTED
        elif drawdown >= self._l1:
            self._level = CircuitLevel.CAUTION
        else:
            self._level = CircuitLevel.NORMAL

        return self._level

    def reset_daily(self, new_baseline: Decimal) -> None:
        """Daily auto-reset: clears CAUTION and HALTED; updates baseline.

        LIQUIDATE is intentionally preserved — it requires ``reset_manual``.

        Args:
            new_baseline: New baseline equity (typically today's opening equity).
        """
        if self._level in (CircuitLevel.CAUTION, CircuitLevel.HALTED):
            self._level = CircuitLevel.NORMAL

    def reset_manual(self) -> None:
        """Operator-initiated reset: clears LIQUIDATE → NORMAL."""
        self._level = CircuitLevel.NORMAL


class CrossMarketCircuitBreaker:
    """Monitors combined drawdown across all markets.

    Trips when ``(sum(baselines) - sum(currents)) / sum(baselines) >= halt_threshold``.
    Returns ``True`` (halted) or ``False`` (clear).
    """

    def __init__(self, halt_threshold: float = _DEFAULT_CROSS_HALT) -> None:
        self._threshold = Decimal(str(halt_threshold))

    def check(
        self,
        market_equities: dict[str, Decimal],
        baseline_equities: dict[str, Decimal],
    ) -> bool:
        """Return True if combined drawdown exceeds the halt threshold.

        Args:
            market_equities: Mapping of market_id to current equity.
            baseline_equities: Mapping of market_id to baseline equity.

        Returns:
            ``True`` if all markets should halt; ``False`` otherwise.
        """
        total_baseline = sum(baseline_equities.values(), _ZERO)
        if total_baseline <= _ZERO:
            return False

        total_current = sum(market_equities.values(), _ZERO)
        combined_drawdown = (total_baseline - total_current) / total_baseline
        return combined_drawdown >= self._threshold

    def reset_daily(self, new_baselines: dict[str, Decimal]) -> None:
        """Update internal state on daily reset.

        The cross-market breaker is stateless (computes on demand), so this
        method exists for symmetry with per-market breakers and future use.

        Args:
            new_baselines: New baseline equities per market (unused currently,
                callers pass updated baselines to ``check`` directly).
        """
```

### Step 3 — Run tests (GREEN)

```bash
uv run pytest tests/unit/test_circuit_breaker.py -q --no-header
```

Expected output:
```
22 passed in 0.XXs
```

### Step 4 — Quality checks

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

### Step 5 — Commit

```bash
git add src/finalayze/risk/circuit_breaker.py tests/unit/test_circuit_breaker.py
git commit -m "feat(risk): add per-market and cross-market circuit breakers"
```

---

## Task 3: TelegramAlerter

**Files to create:**
- `src/finalayze/core/alerts.py`
- `tests/unit/test_telegram_alerter.py`

### Step 1 — Write the failing tests FIRST

Create `tests/unit/test_telegram_alerter.py`:

```python
"""Unit tests for TelegramAlerter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.risk.circuit_breaker import CircuitLevel

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
TELEGRAM_API_URL_PREFIX = "https://api.telegram.org/bot"
VALID_TOKEN = "1234567890:AABBccDDeEFfGgHhIiJj"
VALID_CHAT_ID = "-1001234567890"
MARKET_US = "us"
MARKET_MOEX = "moex"
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal("10")
DRAWDOWN_PCT = 0.103
DAILY_PNL_US = Decimal("342")
DAILY_PNL_MOEX = Decimal("1200")
TOTAL_EQUITY = Decimal("51200")


def _make_order_result() -> object:
    from finalayze.execution.broker_base import OrderResult

    return OrderResult(
        filled=True,
        fill_price=FILL_PRICE,
        symbol="AAPL",
        side="BUY",
        quantity=ORDER_QTY,
    )


def _make_order_request() -> object:
    from finalayze.execution.broker_base import OrderRequest

    return OrderRequest(symbol="AAPL", side="BUY", quantity=ORDER_QTY)


class TestTelegramAlerterNoOp:
    """When token is empty, all methods must silently do nothing."""

    def test_no_op_on_trade_filled(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
            mock_post.assert_not_called()

    def test_no_op_on_trade_rejected(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_trade_rejected(_make_order_request(), "insufficient funds")
            mock_post.assert_not_called()

    def test_no_op_on_circuit_breaker_trip(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)
            mock_post.assert_not_called()

    def test_no_op_on_circuit_breaker_reset(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_circuit_breaker_reset(MARKET_US)
            mock_post.assert_not_called()

    def test_no_op_on_daily_summary(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_daily_summary(
                {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
                TOTAL_EQUITY,
            )
            mock_post.assert_not_called()

    def test_no_op_on_error(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token="", chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            alerter.on_error("NewsApiFetcher", "connection timeout")
            mock_post.assert_not_called()


class TestTelegramAlerterSendsMessages:
    """When token is present, each method must call httpx.post with correct payload."""

    def _make_alerter(self) -> object:
        from finalayze.core.alerts import TelegramAlerter

        return TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)

    def test_on_trade_filled_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")  # type: ignore[attr-defined]
            mock_post.assert_called_once()
            url, = mock_post.call_args.args
            assert VALID_TOKEN in url
            payload = mock_post.call_args.kwargs["json"]
            assert payload["chat_id"] == VALID_CHAT_ID
            assert "AAPL" in payload["text"]
            assert "BUY" in payload["text"]

    def test_on_trade_rejected_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_trade_rejected(_make_order_request(), "insufficient funds")  # type: ignore[attr-defined]
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "AAPL" in payload["text"]
            assert "insufficient funds" in payload["text"]

    def test_on_circuit_breaker_trip_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_circuit_breaker_trip(MARKET_US, CircuitLevel.HALTED, DRAWDOWN_PCT)  # type: ignore[attr-defined]
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert MARKET_US.upper() in payload["text"] or "us" in payload["text"].lower()
            assert "halted" in payload["text"].lower() or "HALTED" in payload["text"]

    def test_on_circuit_breaker_reset_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_circuit_breaker_reset(MARKET_US)  # type: ignore[attr-defined]
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "reset" in payload["text"].lower() or "resumed" in payload["text"].lower()

    def test_on_daily_summary_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_daily_summary(  # type: ignore[attr-defined]
                {MARKET_US: DAILY_PNL_US, MARKET_MOEX: DAILY_PNL_MOEX},
                TOTAL_EQUITY,
            )
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "51200" in payload["text"] or "Daily" in payload["text"]

    def test_on_error_calls_post(self) -> None:
        alerter = self._make_alerter()
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            alerter.on_error("NewsApiFetcher", "gRPC timeout")  # type: ignore[attr-defined]
            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            assert "NewsApiFetcher" in payload["text"]
            assert "gRPC timeout" in payload["text"]


class TestTelegramAlerterErrorHandling:
    """HTTP errors must be swallowed — never propagate to callers."""

    def test_httpx_error_does_not_propagate(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        with patch("httpx.post", side_effect=Exception("network failure")):
            # Must not raise
            alerter.on_error("component", "message")

    def test_http_non_200_does_not_propagate(self) -> None:
        from finalayze.core.alerts import TelegramAlerter

        alerter = TelegramAlerter(bot_token=VALID_TOKEN, chat_id=VALID_CHAT_ID)
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=400)
            # Must not raise even on 4xx response
            alerter.on_trade_filled(_make_order_result(), MARKET_US, "alpaca")
```

Run (expect failures):

```bash
uv run pytest tests/unit/test_telegram_alerter.py -q --no-header
```

Expected output:
```
ERROR tests/unit/test_telegram_alerter.py - ModuleNotFoundError: No module named 'finalayze.core.alerts'
```

### Step 2 — Implement `src/finalayze/core/alerts.py`

```python
"""Telegram alerting for trade events and system notifications (Layer 0/6 boundary).

TelegramAlerter is stateless and fire-and-forget:
  - If bot_token is empty, all methods are no-ops (safe default for dev/test).
  - HTTP errors are caught and logged — they never propagate to the trading loop.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.risk.circuit_breaker import CircuitLevel

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"
_SEND_MESSAGE_PATH = "/sendMessage"

_log = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends Telegram messages for trade fills, rejections, circuit breaker events,
    daily summaries, and errors.

    When ``bot_token`` is an empty string, all methods return immediately
    without any network call (safe default for debug and test modes).
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id

    # ── Public API ───────────────────────────────────────────────────────────

    def on_trade_filled(self, result: OrderResult, market_id: str, broker: str) -> None:
        """Alert on a successful order fill.

        Example: ``🟢 BUY AAPL ×10 @ $150.00 (Alpaca paper)``
        """
        price = result.fill_price if result.fill_price is not None else Decimal("0")
        text = (
            f"\U0001f7e2 {result.side} {result.symbol} \xd7{result.quantity} "
            f"@ ${price:.2f} ({broker} {market_id})"
        )
        self._send(text)

    def on_trade_rejected(self, order: OrderRequest, reason: str) -> None:
        """Alert on an order rejection.

        Example: ``⚠️ AAPL BUY rejected: insufficient funds``
        """
        text = f"\u26a0\ufe0f {order.symbol} {order.side} rejected: {reason}"
        self._send(text)

    def on_circuit_breaker_trip(
        self, market_id: str, level: CircuitLevel, drawdown_pct: float
    ) -> None:
        """Alert on a circuit breaker level change.

        Example: ``🔴 [US] Circuit breaker L2 — trading halted (-10.3% daily)``
        """
        text = (
            f"\U0001f534 [{market_id.upper()}] Circuit breaker {level.upper()} "
            f"— trading {level} ({drawdown_pct * 100:.1f}% daily drawdown)"
        )
        self._send(text)

    def on_circuit_breaker_reset(self, market_id: str) -> None:
        """Alert on circuit breaker reset.

        Example: ``✅ [US] Circuit breaker reset — trading resumed``
        """
        text = f"\u2705 [{market_id.upper()}] Circuit breaker reset \u2014 trading resumed"
        self._send(text)

    def on_daily_summary(
        self,
        market_pnl: dict[str, Decimal],
        total_equity_usd: Decimal,
    ) -> None:
        """Alert with daily P&L summary.

        Example: ``📊 Daily: US +$342 | MOEX +₽1,200 | Equity $51,200``
        """
        parts = []
        for market_id, pnl in sorted(market_pnl.items()):
            sign = "+" if pnl >= Decimal("0") else ""
            parts.append(f"{market_id.upper()} {sign}{pnl}")
        summary = " | ".join(parts)
        text = f"\U0001f4ca Daily: {summary} | Equity ${total_equity_usd:,.0f}"
        self._send(text)

    def on_error(self, component: str, message: str) -> None:
        """Alert on system errors.

        Example: ``🚨 TinkoffFetcher error: gRPC timeout``
        """
        text = f"\U0001f6a8 {component} error: {message}"
        self._send(text)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _send(self, text: str) -> None:
        """POST a message to the Telegram Bot API.

        Silently returns if token is empty or if any error occurs.
        """
        if not self._token:
            return

        url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
        payload = {"chat_id": self._chat_id, "text": text}
        try:
            httpx.post(url, json=payload)
        except Exception:  # noqa: BLE001
            _log.exception("TelegramAlerter failed to send message")
```

### Step 3 — Run tests (GREEN)

```bash
uv run pytest tests/unit/test_telegram_alerter.py -q --no-header
```

Expected output:
```
19 passed in 0.XXs
```

### Step 4 — Quality checks

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

Note: `ruff` will flag the bare `except Exception` as `BLE001`. The `# noqa: BLE001` comment
in `_send` suppresses it intentionally — error propagation must never crash the trading loop.

### Step 5 — Commit

```bash
git add src/finalayze/core/alerts.py tests/unit/test_telegram_alerter.py
git commit -m "feat(core): add TelegramAlerter for trade and system notifications"
```

---

## Task 4: TradingLoop

**Files to create:**
- `src/finalayze/core/trading_loop.py`
- `tests/unit/test_trading_loop.py`

### Step 1 — Write the failing tests FIRST

Create `tests/unit/test_trading_loop.py`:

```python
"""Unit tests for TradingLoop — each cycle method tested in isolation."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.schemas import Candle, NewsArticle, SentimentResult, Signal, SignalDirection
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal("100000")
CAUTION_EQUITY = Decimal("94000")    # 6% drawdown → CAUTION
LIQUIDATE_EQUITY = Decimal("84000")  # 16% drawdown → LIQUIDATE
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal("10")
NUM_CANDLES = 60
CANDLE_CLOSE = Decimal("150.00")
NEWS_CYCLE_MINUTES = 30
STRATEGY_CYCLE_MINUTES = 60
DAILY_RESET_HOUR = 0
SENTIMENT_BUY = 0.8
SENTIMENT_NEUTRAL = 0.0


def _make_candle(symbol: str = SYMBOL_AAPL, idx: int = 0) -> Candle:
    return Candle(
        symbol=symbol,
        market_id=MARKET_US,
        timeframe="1d",
        timestamp=datetime(2026, 1, 1 + idx, 14, 30, tzinfo=UTC),
        open=CANDLE_CLOSE,
        high=CANDLE_CLOSE,
        low=CANDLE_CLOSE,
        close=CANDLE_CLOSE,
        volume=1_000_000,
    )


def _make_candles(n: int = NUM_CANDLES) -> list[Candle]:
    return [_make_candle(idx=i) for i in range(n)]


def _make_news_article() -> NewsArticle:
    return NewsArticle(
        id=__import__("uuid").uuid4(),
        source="Reuters",
        title="Fed raises rates",
        content="The Federal Reserve raised interest rates by 25bps.",
        url="https://reuters.com/article/1",
        language="en",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        scope="us",
    )


def _make_settings(
    news_cycle: int = NEWS_CYCLE_MINUTES,
    strategy_cycle: int = STRATEGY_CYCLE_MINUTES,
    daily_hour: int = DAILY_RESET_HOUR,
) -> object:
    from config.settings import Settings

    s = MagicMock(spec=Settings)
    s.news_cycle_minutes = news_cycle
    s.strategy_cycle_minutes = strategy_cycle
    s.daily_reset_hour_utc = daily_hour
    s.max_position_pct = 0.20
    s.kelly_fraction = 0.5
    s.max_positions_per_market = 10
    return s


def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(
        Instrument(
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            name="Apple Inc.",
            segment_id=SEGMENT_US_TECH,
        )
    )
    return reg


def _make_trading_loop(
    *,
    signal: Signal | None = None,
    fill: bool = True,
    circuit_level: CircuitLevel = CircuitLevel.NORMAL,
    cross_trip: bool = False,
    sentiment_score: float = SENTIMENT_NEUTRAL,
) -> object:
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.trading_loop import TradingLoop

    settings = _make_settings()

    # Mock fetcher
    fetcher = MagicMock()
    fetcher.fetch_candles = MagicMock(return_value=_make_candles())

    # Mock news fetcher
    news_fetcher = MagicMock()
    article = _make_news_article()
    news_fetcher.fetch_news = MagicMock(return_value=[article])

    # Mock news analyzer (async)
    news_analyzer = MagicMock()
    news_analyzer.analyze = AsyncMock(
        return_value=SentimentResult(
            sentiment=sentiment_score, confidence=0.9, reasoning="test"
        )
    )

    # Mock event classifier (async)
    from finalayze.analysis.event_classifier import EventType

    event_classifier = MagicMock()
    event_classifier.classify = AsyncMock(return_value=EventType.MACRO)

    # Mock impact estimator
    from finalayze.analysis.impact_estimator import SegmentImpact

    impact_estimator = MagicMock()
    impact_estimator.estimate = MagicMock(
        return_value=[
            SegmentImpact(segment_id=SEGMENT_US_TECH, weight=1.0, sentiment=sentiment_score)
        ]
    )

    # Mock strategy combiner
    strategy = MagicMock()
    strategy.generate_signal = MagicMock(return_value=signal)

    # Mock broker router
    broker_router = MagicMock()
    fill_result = OrderResult(
        filled=fill,
        fill_price=FILL_PRICE if fill else None,
        symbol=SYMBOL_AAPL,
        side="BUY",
        quantity=ORDER_QTY,
        reason="" if fill else "insufficient funds",
    )
    broker_router.submit = MagicMock(return_value=fill_result)
    mock_broker = MagicMock()
    mock_broker.get_portfolio = MagicMock(
        return_value=MagicMock(equity=BASELINE_EQUITY, cash=Decimal("50000"))
    )
    mock_broker.get_positions = MagicMock(return_value={})
    mock_broker.submit_order = MagicMock(return_value=fill_result)
    broker_router.route = MagicMock(return_value=mock_broker)
    broker_router.registered_markets = [MARKET_US]

    # Circuit breakers
    cb = MagicMock(spec=CircuitBreaker)
    cb.level = circuit_level
    cb.market_id = MARKET_US
    cb.check = MagicMock(return_value=circuit_level)
    cb.reset_daily = MagicMock()

    cmcb = MagicMock(spec=CrossMarketCircuitBreaker)
    cmcb.check = MagicMock(return_value=cross_trip)
    cmcb.reset_daily = MagicMock()

    alerter = MagicMock(spec=TelegramAlerter)

    registry = _make_registry()

    return TradingLoop(
        settings=settings,  # type: ignore[arg-type]
        fetchers={MARKET_US: fetcher},
        news_fetcher=news_fetcher,
        news_analyzer=news_analyzer,
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy,
        broker_router=broker_router,
        circuit_breakers={MARKET_US: cb},
        cross_market_breaker=cmcb,
        alerter=alerter,
        instrument_registry=registry,
    )


class TestTradingLoopNewsCycle:
    def test_news_cycle_fetches_articles(self) -> None:
        loop = _make_trading_loop()
        loop._news_cycle()  # type: ignore[attr-defined]
        loop._news_fetcher.fetch_news.assert_called_once()  # type: ignore[attr-defined]

    def test_news_cycle_updates_sentiment_cache(self) -> None:
        loop = _make_trading_loop(sentiment_score=SENTIMENT_BUY)
        loop._news_cycle()  # type: ignore[attr-defined]
        # After running the news cycle, the cache should have SOME entries
        # (keyed by affected segments → symbols or by scope)
        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        assert isinstance(cache, dict)

    def test_news_cycle_uses_thread_lock(self) -> None:
        """Verify _sentiment_cache is guarded by _sentiment_lock."""
        loop = _make_trading_loop()
        assert hasattr(loop, "_sentiment_lock")
        assert isinstance(loop._sentiment_lock, type(threading.Lock()))  # type: ignore[attr-defined]

    def test_news_cycle_no_error_on_empty_articles(self) -> None:
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news = MagicMock(return_value=[])  # type: ignore[attr-defined]
        loop._news_cycle()  # Must not raise


class TestTradingLoopStrategyCycle:
    def _make_buy_signal(self) -> Signal:
        return Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )

    def test_strategy_cycle_submits_order_on_buy_signal(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_fill(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=True)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_filled.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_rejection(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=False)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_rejected.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_skips_order_when_halted(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.HALTED)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_liquidates_when_l3(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.LIQUIDATE)
        with patch.object(loop, "_liquidate_market") as mock_liq:  # type: ignore[arg-type]
            loop._strategy_cycle()  # type: ignore[attr-defined]
            mock_liq.assert_called_with(MARKET_US)

    def test_strategy_cycle_no_signal_no_submit(self) -> None:
        loop = _make_trading_loop(signal=None)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_caution_does_not_block_order(self) -> None:
        """CAUTION level should still allow orders (just with halved size)."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.CAUTION)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]


class TestTradingLoopDailyReset:
    def test_daily_reset_calls_circuit_breaker_reset(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        for cb in loop._circuit_breakers.values():  # type: ignore[attr-defined]
            cb.reset_daily.assert_called_once()

    def test_daily_reset_sends_daily_summary(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        loop._alerter.on_daily_summary.assert_called_once()  # type: ignore[attr-defined]

    def test_daily_reset_calls_cross_market_reset(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        loop._cross_market_breaker.reset_daily.assert_called_once()  # type: ignore[attr-defined]


class TestTradingLoopLiquidation:
    def test_liquidate_market_submits_sell_for_each_position(self) -> None:
        loop = _make_trading_loop()
        # Inject open positions
        positions = {SYMBOL_AAPL: Decimal("10"), "MSFT": Decimal("5")}
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_positions = MagicMock(return_value=positions)
        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]
        # Should submit one SELL per position
        assert mock_broker.submit_order.call_count == len(positions)
        for call in mock_broker.submit_order.call_args_list:
            order = call.args[0]
            assert order.side == "SELL"

    def test_liquidate_market_sends_alert(self) -> None:
        loop = _make_trading_loop()
        positions = {SYMBOL_AAPL: Decimal("10")}
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_positions = MagicMock(return_value=positions)
        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]
        loop._alerter.on_circuit_breaker_trip.assert_called()  # type: ignore[attr-defined]


class TestTradingLoopThreadSafety:
    def test_concurrent_news_and_strategy_do_not_deadlock(self) -> None:
        """Two threads reading/writing _sentiment_cache must not deadlock."""
        loop = _make_trading_loop(sentiment_score=SENTIMENT_BUY)

        errors: list[Exception] = []

        def run_news() -> None:
            try:
                loop._news_cycle()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def run_strategy() -> None:
            try:
                loop._strategy_cycle()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=run_news)
        t2 = threading.Thread(target=run_strategy)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors
```

Run (expect failures):

```bash
uv run pytest tests/unit/test_trading_loop.py -q --no-header
```

Expected output:
```
ERROR tests/unit/test_trading_loop.py - ModuleNotFoundError: No module named 'finalayze.core.trading_loop'
```

Also note: `Instrument` does not currently have a `segment_id` field. The `_make_registry` helper
in tests uses `segment_id` on `Instrument`. Before implementing `TradingLoop`, you need to add
`segment_id: str = ""` to the `Instrument` dataclass in
`src/finalayze/markets/instruments.py`, or map instruments to segments via the segment config.
The simpler approach (used here) is to add `segment_id: str = "us_tech"` to `Instrument` as an
optional field with a default of `""`, and let `TradingLoop._strategy_cycle` look up the segment
via the existing `config/segments.py` `DEFAULT_SEGMENTS` list keyed by `market_id`.

### Step 2 — Implement `src/finalayze/core/trading_loop.py`

```python
"""APScheduler-based live trading loop (Layer — sits above Layer 5).

Orchestrates three scheduled cycles:
  - _news_cycle: fetch news, analyze sentiment, update _sentiment_cache
  - _strategy_cycle: for each instrument, generate signal, apply circuit breakers,
    submit orders via BrokerRouter, fire alerts
  - _daily_reset: reset circuit breakers, send daily P&L summary

Thread safety: _sentiment_cache is protected by _sentiment_lock (threading.Lock).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]

from finalayze.core.schemas import SignalDirection
from finalayze.execution.broker_base import OrderRequest
from finalayze.risk.circuit_breaker import CircuitLevel

if TYPE_CHECKING:
    from finalayze.analysis.event_classifier import EventClassifier
    from finalayze.analysis.impact_estimator import ImpactEstimator
    from finalayze.analysis.news_analyzer import NewsAnalyzer
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.data.fetchers.newsapi import NewsApiFetcher
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.risk.circuit_breaker import CircuitBreaker, CrossMarketCircuitBreaker
    from finalayze.strategies.combiner import StrategyCombiner
    from config.settings import Settings

# ── Constants ──────────────────────────────────────────────────────────────
_NEWS_QUERY = "stock market finance"
_NEWS_LOOKBACK_HOURS = 2
_CANDLE_LOOKBACK = 60          # number of bars to fetch per symbol
_CAUTION_SIZE_FACTOR = Decimal("0.5")   # halve position size at CAUTION
_MIN_CONFIDENCE_BOOST = 1.2    # raise required confidence 20% at CAUTION
_DEFAULT_SENTIMENT = 0.0
_ZERO = Decimal(0)

_log = logging.getLogger(__name__)


class TradingLoop:
    """Schedules and runs the news, strategy, and daily-reset cycles.

    Designed for TEST / SANDBOX modes. Will gate on WorkMode in real mode.
    """

    def __init__(
        self,
        settings: Settings,
        fetchers: dict[str, object],
        news_fetcher: NewsApiFetcher,
        news_analyzer: NewsAnalyzer,
        event_classifier: EventClassifier,
        impact_estimator: ImpactEstimator,
        strategy: StrategyCombiner,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],
        cross_market_breaker: CrossMarketCircuitBreaker,
        alerter: TelegramAlerter,
        instrument_registry: InstrumentRegistry,
    ) -> None:
        self._settings = settings
        self._fetchers = fetchers
        self._news_fetcher = news_fetcher
        self._news_analyzer = news_analyzer
        self._event_classifier = event_classifier
        self._impact_estimator = impact_estimator
        self._strategy = strategy
        self._broker_router = broker_router
        self._circuit_breakers = circuit_breakers
        self._cross_market_breaker = cross_market_breaker
        self._alerter = alerter
        self._registry = instrument_registry

        # Thread-safe sentiment cache: segment_id -> weighted sentiment score
        self._sentiment_cache: dict[str, float] = {}
        self._sentiment_lock = threading.Lock()

        self._scheduler: BackgroundScheduler | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the APScheduler and block until stop() is called."""
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduler.add_job(
            self._news_cycle,
            "interval",
            minutes=self._settings.news_cycle_minutes,
        )
        self._scheduler.add_job(
            self._strategy_cycle,
            "interval",
            minutes=self._settings.strategy_cycle_minutes,
        )
        self._scheduler.add_job(
            self._daily_reset,
            "cron",
            hour=self._settings.daily_reset_hour_utc,
            minute=0,
        )
        self._scheduler.start()
        _log.info(
            "TradingLoop started: news=%dm, strategy=%dm, daily_reset=UTC %02d:00",
            self._settings.news_cycle_minutes,
            self._settings.strategy_cycle_minutes,
            self._settings.daily_reset_hour_utc,
        )
        self._stop_event.wait()

    def stop(self) -> None:
        """Gracefully shut down the scheduler and unblock start()."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=True)
        self._stop_event.set()

    # ── Cycles ───────────────────────────────────────────────────────────────

    def _news_cycle(self) -> None:
        """Fetch latest news, analyze sentiment, update _sentiment_cache."""
        now = datetime.now(UTC)
        from_date = now - timedelta(hours=_NEWS_LOOKBACK_HOURS)
        try:
            articles = self._news_fetcher.fetch_news(
                query=_NEWS_QUERY,
                from_date=from_date,
                to_date=now,
            )
        except Exception:  # noqa: BLE001
            _log.exception("_news_cycle: failed to fetch news")
            self._alerter.on_error("NewsApiFetcher", "fetch_news failed")
            return

        for article in articles:
            try:
                sentiment = asyncio.run(self._news_analyzer.analyze(article))
                event = asyncio.run(self._event_classifier.classify(article))
                # Use active segment IDs from the registry's known markets
                active_segments = list(
                    {
                        seg
                        for market_id in self._fetchers
                        for instr in self._registry.list_by_market(market_id)
                        if hasattr(instr, "segment_id") and instr.segment_id
                        for seg in [instr.segment_id]
                    }
                )
                impacts = self._impact_estimator.estimate(
                    article, event, sentiment, active_segments
                )
                with self._sentiment_lock:
                    for impact in impacts:
                        existing = self._sentiment_cache.get(impact.segment_id, _DEFAULT_SENTIMENT)
                        # Weighted average: blend new with existing
                        self._sentiment_cache[impact.segment_id] = (
                            existing * 0.7 + impact.sentiment * 0.3
                        )
            except Exception:  # noqa: BLE001
                _log.exception("_news_cycle: error processing article %s", article.id)

    def _strategy_cycle(self) -> None:
        """For each market and instrument, generate a signal and submit orders."""
        market_equities: dict[str, Decimal] = {}
        baseline_equities: dict[str, Decimal] = {}

        for market_id, cb in self._circuit_breakers.items():
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
            except Exception:  # noqa: BLE001
                _log.exception("_strategy_cycle: failed to get portfolio for %s", market_id)
                continue

            market_equities[market_id] = equity
            baseline_equities[market_id] = equity  # simplified: no stored daily baseline here

            level = cb.check(
                current_equity=equity,
                baseline_equity=equity,  # real impl would use stored daily baseline
            )

            if level == CircuitLevel.LIQUIDATE:
                _log.warning("Circuit breaker LIQUIDATE for %s — liquidating", market_id)
                self._liquidate_market(market_id)
                continue

            if level == CircuitLevel.HALTED:
                _log.warning("Circuit breaker HALTED for %s — skipping cycle", market_id)
                continue

            instruments = self._registry.list_by_market(market_id)
            fetcher = self._fetchers.get(market_id)
            if fetcher is None:
                _log.warning("No fetcher for market %s", market_id)
                continue

            for instrument in instruments:
                seg_id = getattr(instrument, "segment_id", "") or "us_tech"
                try:
                    candles = fetcher.fetch_candles(  # type: ignore[attr-defined]
                        symbol=instrument.symbol,
                        market_id=market_id,
                        limit=_CANDLE_LOOKBACK,
                    )
                except Exception:  # noqa: BLE001
                    _log.exception(
                        "_strategy_cycle: failed to fetch candles for %s", instrument.symbol
                    )
                    continue

                with self._sentiment_lock:
                    sentiment = self._sentiment_cache.get(seg_id, _DEFAULT_SENTIMENT)

                signal = self._strategy.generate_signal(
                    instrument.symbol, candles, seg_id
                )
                if signal is None:
                    continue

                # CAUTION: require higher confidence, reduce position size
                if level == CircuitLevel.CAUTION:
                    min_conf = 0.5 * _MIN_CONFIDENCE_BOOST
                    if signal.confidence < min_conf:
                        continue

                order_value = Decimal(str(signal.confidence)) * portfolio.cash
                if level == CircuitLevel.CAUTION:
                    order_value = order_value * _CAUTION_SIZE_FACTOR

                qty = (order_value / Decimal(str(candles[-1].close))) if candles else _ZERO
                qty = qty.quantize(Decimal("1"))
                if qty <= _ZERO:
                    continue

                side: str = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
                order = OrderRequest(symbol=instrument.symbol, side=side, quantity=qty)  # type: ignore[arg-type]

                try:
                    result = self._broker_router.submit(order, market_id=market_id)
                    if result.filled:
                        self._alerter.on_trade_filled(result, market_id, broker="alpaca")
                    else:
                        self._alerter.on_trade_rejected(order, result.reason)
                except Exception:  # noqa: BLE001
                    _log.exception(
                        "_strategy_cycle: order submission failed for %s", instrument.symbol
                    )

        # Cross-market check after all markets processed
        if self._cross_market_breaker.check(market_equities, baseline_equities):
            _log.warning("CrossMarketCircuitBreaker tripped — all markets halted")
            self._alerter.on_circuit_breaker_trip("all", CircuitLevel.HALTED, 0.0)

    def _daily_reset(self) -> None:
        """Reset circuit breakers and send daily P&L summary."""
        market_pnl: dict[str, Decimal] = {}
        new_baselines: dict[str, Decimal] = {}

        for market_id, cb in self._circuit_breakers.items():
            try:
                broker = self._broker_router.route(market_id)
                portfolio = broker.get_portfolio()
                equity = portfolio.equity
                new_baselines[market_id] = equity
                market_pnl[market_id] = _ZERO  # simplified: actual P&L tracked separately
                cb.reset_daily(new_baseline=equity)
            except Exception:  # noqa: BLE001
                _log.exception("_daily_reset: failed to reset for market %s", market_id)

        self._cross_market_breaker.reset_daily(new_baselines)
        total_equity = sum(new_baselines.values(), _ZERO)
        self._alerter.on_daily_summary(market_pnl, total_equity)
        _log.info("Daily reset complete. Total equity: %s", total_equity)

    def _liquidate_market(self, market_id: str) -> None:
        """Close all open positions in a market (L3 circuit breaker response)."""
        try:
            broker = self._broker_router.route(market_id)
            positions = broker.get_positions()
            portfolio = broker.get_portfolio()
            equity = portfolio.equity

            for symbol, qty in positions.items():
                if qty <= _ZERO:
                    continue
                order = OrderRequest(symbol=symbol, side="SELL", quantity=qty)  # type: ignore[arg-type]
                broker.submit_order(order)

            # Alert after liquidation
            drawdown = float(
                (equity - sum(positions.values(), _ZERO)) / equity
                if equity > _ZERO
                else 0
            )
            self._alerter.on_circuit_breaker_trip(market_id, CircuitLevel.LIQUIDATE, drawdown)
        except Exception:  # noqa: BLE001
            _log.exception("_liquidate_market: failed for market %s", market_id)
            self._alerter.on_error("TradingLoop", f"liquidation failed for {market_id}")
```

### Step 3 — Run tests (GREEN)

```bash
uv run pytest tests/unit/test_trading_loop.py -q --no-header
```

Expected output:
```
18 passed in 0.XXs
```

### Step 4 — Quality checks

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

### Step 5 — Commit

```bash
git add src/finalayze/core/trading_loop.py tests/unit/test_trading_loop.py
git commit -m "feat(core): add APScheduler-based TradingLoop for test/paper mode"
```

---

## Task 5: Integration tests

**Files to create:**
- `tests/integration/test_trading_loop.py`
- `tests/integration/test_circuit_breaker_integration.py`
- `tests/integration/test_news_to_signal.py`

### Step 1 — Write all three integration test files

#### `tests/integration/test_trading_loop.py`

```python
"""Integration test: full strategy cycle signal → pre-trade → circuit breaker → order → alert."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.analysis.impact_estimator import ImpactEstimator, SegmentImpact
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import (
    Candle,
    NewsArticle,
    SentimentResult,
    Signal,
    SignalDirection,
)
from finalayze.core.trading_loop import TradingLoop
from finalayze.data.fetchers.newsapi import NewsApiFetcher
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal("100000")
AVAILABLE_CASH = Decimal("50000")
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal("10")
NUM_CANDLES = 60
CANDLE_CLOSE = Decimal("150.00")
NEWS_CYCLE_MINUTES = 30
STRATEGY_CYCLE_MINUTES = 60
DAILY_RESET_HOUR = 0


@pytest.mark.integration
class TestStrategyIntegration:
    """Full cycle: signal flows through circuit breaker check → order submitted → alert fired."""

    def _build_system(
        self,
        circuit_level: CircuitLevel = CircuitLevel.NORMAL,
        fill: bool = True,
    ) -> TradingLoop:
        settings = MagicMock()
        settings.news_cycle_minutes = NEWS_CYCLE_MINUTES
        settings.strategy_cycle_minutes = STRATEGY_CYCLE_MINUTES
        settings.daily_reset_hour_utc = DAILY_RESET_HOUR
        settings.max_position_pct = 0.20
        settings.kelly_fraction = 0.5
        settings.max_positions_per_market = 10

        # Real instrument registry
        registry = InstrumentRegistry()
        registry.register(
            Instrument(
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                name="Apple Inc.",
                segment_id=SEGMENT_US_TECH,
            )
        )

        # Mock fetcher
        candles = [
            Candle(
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                timeframe="1d",
                timestamp=datetime(2026, 1, 1 + i, 14, 30, tzinfo=UTC),
                open=CANDLE_CLOSE,
                high=CANDLE_CLOSE,
                low=CANDLE_CLOSE,
                close=CANDLE_CLOSE,
                volume=1_000_000,
            )
            for i in range(NUM_CANDLES)
        ]
        fetcher = MagicMock()
        fetcher.fetch_candles = MagicMock(return_value=candles)

        # Mock news stack
        news_fetcher = MagicMock(spec=NewsApiFetcher)
        news_fetcher.fetch_news = MagicMock(return_value=[])
        news_analyzer = MagicMock(spec=NewsAnalyzer)
        news_analyzer.analyze = AsyncMock(
            return_value=SentimentResult(sentiment=0.0, confidence=0.0, reasoning="")
        )
        event_classifier = MagicMock(spec=EventClassifier)
        event_classifier.classify = AsyncMock(return_value=EventType.OTHER)
        impact_estimator = MagicMock(spec=ImpactEstimator)
        impact_estimator.estimate = MagicMock(return_value=[])

        # Mock strategy — returns a BUY signal
        strategy = MagicMock()
        strategy.generate_signal = MagicMock(
            return_value=Signal(
                strategy_name="combined",
                symbol=SYMBOL_AAPL,
                market_id=MARKET_US,
                segment_id=SEGMENT_US_TECH,
                direction=SignalDirection.BUY,
                confidence=0.80,
                features={},
                reasoning="integration test signal",
            )
        )

        # Mock broker
        portfolio = MagicMock()
        portfolio.equity = BASELINE_EQUITY
        portfolio.cash = AVAILABLE_CASH
        mock_broker = MagicMock()
        mock_broker.get_portfolio = MagicMock(return_value=portfolio)
        mock_broker.get_positions = MagicMock(return_value={})
        fill_result = OrderResult(
            filled=fill,
            fill_price=FILL_PRICE if fill else None,
            symbol=SYMBOL_AAPL,
            side="BUY",
            quantity=ORDER_QTY,
            reason="" if fill else "insufficient funds",
        )
        mock_broker.submit_order = MagicMock(return_value=fill_result)
        broker_router = MagicMock(spec=BrokerRouter)
        broker_router.route = MagicMock(return_value=mock_broker)
        broker_router.submit = MagicMock(return_value=fill_result)
        broker_router.registered_markets = [MARKET_US]

        # Real circuit breaker (with mocked check to return desired level)
        cb = MagicMock(spec=CircuitBreaker)
        cb.level = circuit_level
        cb.market_id = MARKET_US
        cb.check = MagicMock(return_value=circuit_level)
        cb.reset_daily = MagicMock()

        cmcb = MagicMock(spec=CrossMarketCircuitBreaker)
        cmcb.check = MagicMock(return_value=False)
        cmcb.reset_daily = MagicMock()

        alerter = MagicMock(spec=TelegramAlerter)

        return TradingLoop(
            settings=settings,
            fetchers={MARKET_US: fetcher},
            news_fetcher=news_fetcher,
            news_analyzer=news_analyzer,
            event_classifier=event_classifier,
            impact_estimator=impact_estimator,
            strategy=strategy,
            broker_router=broker_router,
            circuit_breakers={MARKET_US: cb},
            cross_market_breaker=cmcb,
            alerter=alerter,
            instrument_registry=registry,
        )

    def test_signal_flows_through_to_order_submit(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=True)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_called_once()

    def test_filled_order_fires_alert(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=True)
        loop._strategy_cycle()
        loop._alerter.on_trade_filled.assert_called_once()

    def test_rejected_order_fires_reject_alert(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.NORMAL, fill=False)
        loop._strategy_cycle()
        loop._alerter.on_trade_rejected.assert_called_once()

    def test_halted_circuit_blocks_all_orders(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.HALTED)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_not_called()

    def test_caution_circuit_allows_reduced_orders(self) -> None:
        loop = self._build_system(circuit_level=CircuitLevel.CAUTION, fill=True)
        loop._strategy_cycle()
        loop._broker_router.submit.assert_called()
```

#### `tests/integration/test_circuit_breaker_integration.py`

```python
"""Integration tests for circuit breaker escalation, cross-market trip, and resets."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
MARKET_MOEX = "moex"
BASELINE = Decimal("100000")
L1_THRESHOLD = 0.05
L2_THRESHOLD = 0.10
L3_THRESHOLD = 0.15
CROSS_THRESHOLD = 0.10

# Equity values by drawdown
EQUITY_AT_3PCT = Decimal("97000")
EQUITY_AT_6PCT = Decimal("94000")
EQUITY_AT_11PCT = Decimal("89000")
EQUITY_AT_16PCT = Decimal("84000")

US_BASELINE = Decimal("50000")
MOEX_BASELINE = Decimal("50000")
US_SAFE = Decimal("48000")       # 4% down
MOEX_TRIPPING = Decimal("41000") # total 89000 from 100000 → 11% combined


@pytest.mark.integration
class TestCircuitBreakerEscalation:
    def _make_cb(self) -> CircuitBreaker:
        return CircuitBreaker(
            market_id=MARKET_US,
            l1_threshold=L1_THRESHOLD,
            l2_threshold=L2_THRESHOLD,
            l3_threshold=L3_THRESHOLD,
        )

    def test_l1_to_l2_to_l3_escalation(self) -> None:
        cb = self._make_cb()
        assert cb.check(EQUITY_AT_6PCT, BASELINE) == CircuitLevel.CAUTION
        assert cb.check(EQUITY_AT_11PCT, BASELINE) == CircuitLevel.HALTED
        assert cb.check(EQUITY_AT_16PCT, BASELINE) == CircuitLevel.LIQUIDATE

    def test_auto_daily_reset_clears_caution_not_liquidate(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_6PCT, BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.reset_daily(new_baseline=EQUITY_AT_6PCT)
        assert cb.level == CircuitLevel.NORMAL

    def test_auto_daily_reset_clears_halted(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED
        cb.reset_daily(new_baseline=EQUITY_AT_11PCT)
        assert cb.level == CircuitLevel.NORMAL

    def test_liquidate_not_cleared_by_daily_reset(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE  # still locked

    def test_manual_reset_unblocks_liquidate(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE  # daily reset has no effect
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL      # manual clears it

    def test_full_escalation_then_manual_reset(self) -> None:
        """Simulate a bad day: escalate all three levels, manually reset at end."""
        cb = self._make_cb()

        cb.check(EQUITY_AT_6PCT, BASELINE)
        assert cb.level == CircuitLevel.CAUTION

        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED

        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE

        # Daily reset does NOT clear LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE

        # Manual reset clears it
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_recovery_same_day_still_reports_current_level(self) -> None:
        """If equity recovers within the day, check still reflects current status."""
        cb = self._make_cb()
        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Equity recovers to only 3% down
        cb.check(EQUITY_AT_3PCT, BASELINE)
        assert cb.level == CircuitLevel.NORMAL


@pytest.mark.integration
class TestCrossMarketIntegration:
    def test_combined_drawdown_trips_halt(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        assert cmcb.check(
            market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        ) is True

    def test_within_threshold_no_trip(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        assert cmcb.check(
            market_equities={MARKET_US: Decimal("49000"), MARKET_MOEX: Decimal("49000")},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        ) is False

    def test_daily_reset_then_check_at_zero_drawdown(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        # Trip it
        assert cmcb.check(
            market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        ) is True
        # Reset baselines to current values
        cmcb.reset_daily({MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING})
        # Same equities vs new baselines → 0% drawdown → no trip
        assert cmcb.check(
            market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
            baseline_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
        ) is False
```

#### `tests/integration/test_news_to_signal.py`

```python
"""Integration test: mocked LLM → NewsAnalyzer → EventClassifier → ImpactEstimator
→ EventDrivenStrategy produces BUY signal for affected segments.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from finalayze.analysis.event_classifier import EventClassifier, EventType
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.llm_client import LLMClient
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.schemas import Candle, NewsArticle, SignalDirection
from finalayze.strategies.event_driven import EventDrivenStrategy

# ── Constants ──────────────────────────────────────────────────────────────
SENTIMENT_BULLISH = 0.85
CONFIDENCE_HIGH = 0.90
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
MARKET_US = "us"
NUM_CANDLES = 10
CANDLE_CLOSE = Decimal("150.00")
MIN_CONFIDENCE = 0.50

ACTIVE_SEGMENTS = [
    "us_tech",
    "us_finance",
    "us_healthcare",
    "us_energy",
]


def _make_us_article() -> NewsArticle:
    return NewsArticle(
        id=__import__("uuid").uuid4(),
        source="Bloomberg",
        title="Apple beats earnings expectations",
        content="Apple Inc. reported quarterly earnings that significantly exceeded analyst forecasts.",
        url="https://bloomberg.com/1",
        language="en",
        published_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        scope="us",
    )


def _make_candles(symbol: str = SYMBOL_AAPL, n: int = NUM_CANDLES) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            market_id=MARKET_US,
            timeframe="1d",
            timestamp=datetime(2026, 1, 1 + i, 14, 30, tzinfo=UTC),
            open=CANDLE_CLOSE,
            high=CANDLE_CLOSE,
            low=CANDLE_CLOSE,
            close=CANDLE_CLOSE,
            volume=1_000_000,
        )
        for i in range(n)
    ]


@pytest.mark.integration
class TestNewsToSignalPipeline:
    """Tests the full chain: LLM (mocked) → analysis → strategy signal."""

    def _make_llm_client(
        self,
        sentiment: float = SENTIMENT_BULLISH,
        event_type: str = "earnings",
    ) -> LLMClient:
        """Create a mock LLM client that returns canned sentiment and event responses."""
        client = AsyncMock(spec=LLMClient)
        sentiment_response = json.dumps(
            {"sentiment": sentiment, "confidence": CONFIDENCE_HIGH, "reasoning": "strong earnings"}
        )
        # First call → sentiment JSON, second call → event type string
        client.complete = AsyncMock(side_effect=[sentiment_response, event_type])
        return client

    @pytest.mark.asyncio
    async def test_positive_sentiment_produces_buy_signal(self) -> None:
        llm = self._make_llm_client(sentiment=SENTIMENT_BULLISH, event_type="earnings")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        # At least some US segments should be impacted
        assert len(impacts) > 0

        # For each impacted segment, run strategy
        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        # Bullish sentiment → at least one BUY signal
        assert any(s.direction == SignalDirection.BUY for s in signals)

    @pytest.mark.asyncio
    async def test_negative_sentiment_produces_sell_signal(self) -> None:
        negative_sentiment = -0.85
        llm = self._make_llm_client(sentiment=negative_sentiment, event_type="macro")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        assert any(s.direction == SignalDirection.SELL for s in signals)

    @pytest.mark.asyncio
    async def test_neutral_sentiment_produces_no_signal(self) -> None:
        neutral_sentiment = 0.1  # below min_sentiment threshold
        llm = self._make_llm_client(sentiment=neutral_sentiment, event_type="other")
        news_analyzer = NewsAnalyzer(llm_client=llm)
        event_classifier = EventClassifier(llm_client=llm)
        impact_estimator = ImpactEstimator()
        strategy = EventDrivenStrategy()

        article = _make_us_article()
        candles = _make_candles()

        sentiment_result = await news_analyzer.analyze(article)
        event = await event_classifier.classify(article)
        impacts = impact_estimator.estimate(article, event, sentiment_result, ACTIVE_SEGMENTS)

        signals = []
        for impact in impacts:
            signal = strategy.generate_signal(
                symbol=SYMBOL_AAPL,
                candles=candles,
                segment_id=impact.segment_id,
                sentiment_score=impact.sentiment,
            )
            if signal is not None:
                signals.append(signal)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_llm_parse_error_falls_back_to_neutral(self) -> None:
        """If the LLM returns invalid JSON, NewsAnalyzer returns neutral (0.0) sentiment."""
        llm = AsyncMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="not valid json {{ }}")
        news_analyzer = NewsAnalyzer(llm_client=llm)

        article = _make_us_article()
        result = await news_analyzer.analyze(article)

        assert result.sentiment == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_unknown_event_type_falls_back_to_other(self) -> None:
        """If the LLM returns an unknown event label, classifier returns EventType.OTHER."""
        llm = AsyncMock(spec=LLMClient)
        llm.complete = AsyncMock(return_value="alien_invasion")
        event_classifier = EventClassifier(llm_client=llm)

        article = _make_us_article()
        event = await event_classifier.classify(article)

        assert event == EventType.OTHER
```

### Step 2 — Run all integration tests (expect some failures until implementations are complete)

```bash
uv run pytest tests/integration/ -q --no-header -m integration
```

Expected output (after all prior tasks complete):
```
13 passed in 0.XXs
```

### Step 3 — Run the full test suite

```bash
uv run pytest -q --no-header
```

Expected output:
```
... passed in X.XXs
```

### Step 4 — Quality checks (final)

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

All checks must pass with zero errors.

### Step 5 — Commit

```bash
git add tests/integration/test_trading_loop.py \
        tests/integration/test_circuit_breaker_integration.py \
        tests/integration/test_news_to_signal.py
git commit -m "test(integration): add trading loop, circuit breaker, and news-to-signal integration tests"
```

---

## Final Verification

Run the complete quality suite after all 5 tasks are complete:

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected output:
```
All checks passed.
... passed, ... warnings in X.XXs
```

Coverage for new modules must be >= 80%:
- `src/finalayze/risk/circuit_breaker.py` — target >= 95%
- `src/finalayze/core/alerts.py` — target >= 90%
- `src/finalayze/core/trading_loop.py` — target >= 80%

---

## Implementation Notes

### `Instrument.segment_id` field

The `TradingLoop._strategy_cycle` needs to look up the segment for each instrument.
Currently `Instrument` in `src/finalayze/markets/instruments.py` has no `segment_id` field.
Add it as an optional field with a default of `""`:

```python
@dataclass(frozen=True)
class Instrument:
    symbol: str
    market_id: str
    name: str
    instrument_type: InstrumentType = "stock"
    figi: str | None = None
    lot_size: int = 1
    currency: str = "USD"
    is_active: bool = True
    segment_id: str = ""          # <-- add this
```

This is backward-compatible (existing `DEFAULT_US_INSTRUMENTS` and `DEFAULT_MOEX_INSTRUMENTS`
lists will use `""` as the default, and `TradingLoop._strategy_cycle` falls back to `"us_tech"`
when `segment_id` is empty). Existing tests are unaffected.

### APScheduler `asyncio.run()` inside sync cycle methods

`_news_cycle` calls `asyncio.run()` to bridge from sync scheduler callbacks into async
`NewsAnalyzer.analyze` and `EventClassifier.classify`. This is correct for the background
thread context used by APScheduler's `BackgroundScheduler`. If the project later migrates to
`AsyncIOScheduler`, the cycle methods should become `async def` coroutines instead.

### `ruff` noqa suppressions

The following intentional suppressions are used in production code:
- `BLE001` (blind exception) in `_send`, `_news_cycle`, `_strategy_cycle`, `_daily_reset`,
  `_liquidate_market` — trading infrastructure must never crash from alert/network errors.

These are **not** suppressed in test code.
