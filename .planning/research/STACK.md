# Stack Research: v3.0 Production Readiness

**Domain:** Sandbox monitoring, go/no-go gate automation, gradual rollout, production health monitoring
**Researched:** 2026-03-21
**Confidence:** HIGH (based on direct codebase audit + verified library docs)

## Context: What Already Exists (v2.0 stack)

The system already provides — do NOT re-implement or re-add:

- `prometheus-client` + `prometheus-fastapi-instrumentator` — Prometheus metrics (MetricsCollector with 15+ business metrics already defined)
- `structlog` — structured JSON logging (already used everywhere)
- `apscheduler>=3.10.4` — APScheduler (BackgroundScheduler for news/strategy/daily-reset cycles in TradingLoop)
- `streamlit>=1.41.0` — Dashboard with 5 pages (portfolio, risk, signals, trades, system_status)
- `FastAPI` + 20+ REST endpoints — health, portfolio, trades, risk, signals, ML, system
- `redis>=5.2.0` — Redis Streams event bus (EventBus) + RedisCache
- `sqlalchemy[asyncio]` + `asyncpg` + TimescaleDB — async ORM with CycleLogEntry in JSONL
- `httpx` — HTTP client (used for Telegram alerts)
- `python-telegram-bot` (via TelegramAlerter) — priority-queue alerting already live
- `CircuitBreaker` — 3-level escalation (CAUTION 5%, HALTED 10%, LIQUIDATE 15%)
- `PreTradeChecker` — 11-check pre-trade risk gate
- `ValidationLogger` — append-only JSONL cycle log (`results/validation/cycles.jsonl`)
- `core/modes.py` — WorkMode enum (DEBUG/SANDBOX/TEST/REAL) + real_confirmed guard

---

## Key Finding: Minimal New Dependencies

v3.0 requires exactly **three new pip packages** and **one optional infrastructure component** (Grafana). All monitoring primitives already exist. New work is: report generation, automated gate evaluation logic, and rollout configuration — not new observability infrastructure.

---

## New Dependencies Required

### Core Technologies (New)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `Jinja2` | `>=3.1.4` | HTML template rendering for go/no-go gate reports | Already a transitive dep via FastAPI/Starlette; making it explicit. Jinja2 is the standard Python templating library — no learning curve, familiar to all Python devs. Renders HTML gate reports with pass/fail table, metric charts embeds. Latest stable is 3.1.6. |
| `weasyprint` | `>=68.1` | Convert Jinja2-rendered HTML → PDF for go/no-go archive | Best HTML-to-PDF for Python in 2025: CSS3 support, no Chromium required, BSD license, Python-native. v68.1 released 2025-01-30. Requires Python 3.10+. Alternative `reportlab` requires building PDFs programmatically — far higher development cost for formatted reports. |
| `aiogram` | `>=3.17.0` | Telegram bot interactive commands for kill switch + gate triggers | aiogram v3 is async-native, strictly typed, pydantic v2 compatible. Current TelegramAlerter uses raw `httpx` calls (fire-and-forget); aiogram adds proper command handling (/kill, /rollout, /gonogo) with FSM. aiogram v3 superseded v2 in 2023 and is now the community standard. **Note:** If only kill switch needs bot command (not conversation FSM), raw httpx POST to setWebhook is sufficient and aiogram can be deferred. |

### Supporting Libraries (New)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `types-Jinja2` | `>=3.1.0` | mypy stubs for Jinja2 | Add to dev dependencies; required for strict mypy compliance |

### Infrastructure Components (No New Python Package)

| Component | How Deployed | Purpose | Why Not a Pip Package |
|-----------|-------------|---------|----------------------|
| Grafana | Docker Compose service | Dashboard for Prometheus metrics (already being scraped) | Grafana is a standalone service, not a Python library. Prometheus already exports metrics — Grafana just adds a visualization layer. Configure with pre-built dashboards. |
| Grafana Loki + Alloy | Docker Compose service (optional) | Centralized log aggregation for structlog JSON output | Optional for v3.0. structlog already writes JSON; Alloy (replacement for Promtail) tails and ships to Loki. No Python code changes required — pure infrastructure. Defer if operational complexity exceeds value at launch scale. |

---

## Existing Libraries That Cover New v3.0 Needs

No new library needed for these — the existing stack already supports them:

| v3.0 Need | Existing Capability | Where |
|-----------|-------------------|-------|
| Sandbox metric collection | `MetricsCollector` + `ValidationLogger` | `api/metrics.py`, `core/validation_logger.py` |
| Uptime tracking | `prometheus-client` `Counter`/`Gauge` | `api/metrics.py` |
| Fill rate monitoring | `trades_total` + `order_rejection_total` Counters | `api/metrics.py` |
| Slippage measurement | `trade_slippage_bps` Histogram | `api/metrics.py` |
| Signal divergence (sandbox vs backtest) | `strategy_signal_count` Counter | `api/metrics.py` |
| Drawdown monitoring | `drawdown_pct` / `max_drawdown_pct` Gauges | `api/metrics.py` + `risk/drawdown_monitor.py` |
| Kill switch command | Extend existing `CircuitBreaker.reset_manual()` + Telegram `/stop` in `telegram_bot.py` | `core/telegram_bot.py`, `risk/circuit_breaker.py` |
| Tightened risk limits (gradual rollout) | `PreTradeChecker` + `CircuitBreaker` thresholds configurable in `Settings` | `risk/pre_trade_check.py`, `config/settings.py` |
| Scheduled metric evaluation | `apscheduler` `BackgroundScheduler` (already running) | `core/trading_loop.py` |
| Health check endpoints | `/api/v1/health` already exists | `api/v1/system.py` |
| Anomaly detection for trading metrics | `statsmodels` STL decomposition OR simple z-score from `numpy` | `statsmodels` already installed |
| Go/no-go threshold evaluation | Pure Python — compare metric values to configured thresholds | New module `core/gonogo.py` |
| Capital scaling configuration | `Settings` + `SegmentConfig` already pydantic-configurable | `config/settings.py`, `config/segments.py` |
| Database persistence for gate results | `SQLAlchemy` async ORM + existing alembic migrations | `core/db.py` |

---

## Recommended Architecture for New Modules

### Module Map for v3.0

```
core/gonogo.py           — GateEvaluator: threshold checks, pass/fail, result schema
core/sandbox_reporter.py — SandboxMetricCollector: queries ValidationLogger + Prometheus API
core/rollout_config.py   — RolloutPhase: capital limits, position limits, tightened circuit breaker thresholds
api/v1/gonogo.py         — REST endpoint: GET /api/v1/gonogo/status, POST /api/v1/gonogo/evaluate
```

### New Library Integration Points

**Jinja2** — in `core/sandbox_reporter.py`:
```python
from jinja2 import Environment, FileSystemLoader
# Render HTML from templates/gonogo_report.html.j2 with metric dict
# Then pass to weasyprint for PDF archive
```

**WeasyPrint** — in `core/sandbox_reporter.py`:
```python
from weasyprint import HTML
# HTML(string=rendered_html).write_pdf("reports/gonogo_2026-03-21.pdf")
```

**aiogram (if adopted)** — in `core/telegram_bot.py`:
```python
# Replace raw httpx Telegram calls with aiogram Router
# Add CommandHandler for /kill, /rollout <phase>, /gonogo
```

---

## Alternatives Considered

| Recommended | Alternative | Why Alternative Was Rejected |
|-------------|-------------|------------------------------|
| `weasyprint` for PDF | `reportlab` | ReportLab requires building PDFs programmatically (canvas API). For formatted HTML reports with tables and conditional colors, HTML→PDF via WeasyPrint takes 1/5 the code. ReportLab excels at programmatic chart-heavy documents — not our use case. |
| `weasyprint` for PDF | `playwright` (headless Chrome) | Playwright gives pixel-perfect Chrome rendering but adds a 150MB+ browser dependency, complexity for CI, and is overkill for a daily/weekly text report. WeasyPrint's CSS3 support covers all needed formatting. |
| `Jinja2` for templates | `mako` or `chameleon` | Jinja2 is already a transitive dependency, has the largest community, and is the FastAPI template standard. No reason to add a second templating engine. |
| Simple Python z-score for anomaly detection | `adtk` (0.6.2) or `darts` | ADTK last released 2020, no longer actively maintained. Darts is a heavy ML forecasting library (adds >500MB). Trading metric anomalies (drawdown spikes, slippage outliers) can be detected reliably with a rolling z-score using `numpy` + `pandas` — both already installed. Custom implementation is 20 lines and has no dependency risk. |
| APScheduler 3.x (existing) | APScheduler 4.x | APScheduler 4.x is still in alpha/pre-release as of 2025-03-21 (v4.0a4). API changed completely (Task/Schedule/Job split), async context manager required, job stores redesigned. Stable 3.x (3.11.2) works and is already integrated in TradingLoop. Upgrade is non-trivial and adds no value for v3.0 goals. |
| `aiogram` v3 for bot | python-telegram-bot | python-telegram-bot v20+ is also async-native and well maintained. Either works. aiogram is lighter. The choice matters only if interactive conversation FSM is needed (unlikely for v3.0). |
| Grafana for dashboards | Add new Streamlit pages | The Streamlit dashboard is appropriate for ad-hoc operator views. Grafana excels at time-series panels with alerting rules driven by existing Prometheus metrics — no code changes required on the Python side, just dashboard JSON configuration. Both can coexist. |

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `opentelemetry` SDK | OpenTelemetry traces are overkill for a single-process trading system. Adds 10+ packages, significant config complexity. | `structlog` + Prometheus metrics already cover all observability needs |
| `celery` (already installed but unused) | Celery adds a distributed task queue architecture that mismatches the single-process APScheduler loop. Would require separate worker processes. | `apscheduler` `BackgroundScheduler` (already integrated in TradingLoop) |
| Feature flag libraries (Unleash, LaunchDarkly, Statsig) | External feature flag services introduce an external runtime dependency. "Gradual rollout" for a trading system means capital/risk limit progression, not code path toggling. | `rollout_config.py` with `RolloutPhase` enum (MINIMAL / STANDARD / FULL) controlled via Settings |
| `datadog`, `newrelic`, `sentry` APMs | External SaaS observability platforms add cost and data egress of sensitive trading data. | `prometheus-client` + Grafana (self-hosted) |
| `pandas-profiling` / `ydata-profiling` | Heavy profiling tool for data science EDA, not live trading metrics. | Custom metric summaries in `sandbox_reporter.py` |
| `darts` or `prophet` for anomaly detection | Forecasting frameworks with heavy dependencies (PyTorch, Stan). Rolling z-score with numpy is sufficient and has no new dep. | `numpy` + `pandas` rolling statistics (already installed) |
| `fastapi-health` / `fastapi-healthchecks` | Thin wrappers that add minimal value over the existing `/api/v1/health` endpoint. | Extend `api/v1/system.py` directly |

---

## Stack Patterns by Variant

**If go/no-go gate runs daily (automated, no human trigger):**
- Use APScheduler to call `GateEvaluator.evaluate()` each morning before market open
- Results stored in TimescaleDB via alembic migration (new `gonogo_results` table)
- Telegram alert dispatched via existing TelegramAlerter

**If go/no-go gate requires human confirmation before capital scaling:**
- Add Telegram bot command `/gonogo confirm` (requires aiogram OR raw httpx webhook)
- GateEvaluator returns PENDING state until operator responds
- Capital scaling in `RolloutPhase` only advances after confirmation

**If PDF report archiving is required:**
- Use `weasyprint` + `Jinja2` (2 new deps)
- Reports saved to `reports/` directory and linked in Telegram message

**If PDF report is not required (HTML summary in Telegram only):**
- Skip `weasyprint` entirely — render Jinja2 HTML, strip to plaintext for Telegram
- Zero new pip packages for v3.0

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `weasyprint>=68.1` | Python 3.10+ | Confirmed. Our baseline is Python 3.12 — no issue. |
| `Jinja2>=3.1.4` | FastAPI 0.115+ (uses Jinja2 as dep) | Already transitively installed. Just make explicit in pyproject.toml. |
| `aiogram>=3.17.0` | Python 3.9+, asyncio | Not conflicting with existing async stack. aiogram v3 uses aiohttp internally (does NOT conflict with httpx). |
| All existing deps | Unchanged | v3.0 adds at most 2-3 pip packages. No version bumps to existing packages required. |

---

## Installation

```bash
# New production dependencies
uv add "jinja2>=3.1.4" "weasyprint>=68.1"

# New dev dependencies (mypy stubs)
uv add --dev "types-Jinja2>=3.1.0"

# Optional: Telegram bot interactive commands
uv add "aiogram>=3.17.0"

# Infrastructure (Docker Compose additions — no Python package)
# Add grafana service to docker-compose.yml (points at existing Prometheus)
# Add loki + alloy services if log aggregation is required (optional)
```

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| Zero new deps for monitoring core | HIGH | Direct codebase audit: MetricsCollector, ValidationLogger, CircuitBreaker, PreTradeChecker all cover monitoring needs |
| Jinja2 is already a transitive dep | HIGH | FastAPI 0.115 depends on Starlette which depends on Jinja2 — it is already installed, just needs to be made explicit |
| WeasyPrint v68.1 for PDF | HIGH | Official PyPI page confirmed current version (2025-01-30 release), Python 3.10+ requirement met |
| APScheduler 3.x is stable choice | HIGH | 4.x is pre-release alpha; 3.x is proven and already integrated |
| aiogram v3 async compatibility | MEDIUM | No conflict found, but not verified in this codebase's asyncio event loop setup. May need testing with APScheduler's BackgroundScheduler thread model. |
| Grafana infrastructure (no code) | HIGH | Standard Prometheus→Grafana pipeline, no Python changes needed |
| Rolling z-score sufficient for anomaly detection | HIGH | Trading metric anomalies (drawdown, slippage) are step-function events detectable by simple threshold; no ML-based anomaly detection needed |
| RolloutPhase config vs feature flags | HIGH | Capital/position limit progression is configuration, not code path toggling — external feature flag services are mismatched for this use case |

---

## Sources

- Codebase audit: `src/finalayze/api/metrics.py`, `src/finalayze/core/validation_logger.py`, `src/finalayze/risk/circuit_breaker.py`, `src/finalayze/core/telegram_bot.py`, `pyproject.toml`
- WeasyPrint current version: https://pypi.org/project/weasyprint/ (v68.1, released 2025-01-30)
- Jinja2 current version: https://pypi.org/project/Jinja2/ (v3.1.6)
- APScheduler 4.x status: https://github.com/agronholm/apscheduler/issues/465 (still pre-release as of 2025)
- Grafana Loki + Alloy: https://grafana.com/docs/loki/latest/ (Alloy replaced Promtail for log collection)
- aiogram v3: https://docs.aiogram.dev/en/latest/ (async-native, Pydantic v2 compatible)
- ADTK maintenance status: https://github.com/arundo/adtk (last release 2020 — not recommended)
- WeasyPrint vs ReportLab comparison: https://dev.to/claudeprime/generate-pdfs-in-python-weasyprint-vs-reportlab-ifi

---
*Stack research for: v3.0 Production Readiness — sandbox monitoring, go/no-go gates, gradual rollout, production health*
*Researched: 2026-03-21*
