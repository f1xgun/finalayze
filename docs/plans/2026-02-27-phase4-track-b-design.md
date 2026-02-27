# Phase 4 Track B — Observability & Dashboard Design

**Version:** 1.0 | **Date:** 2026-02-27 | **Status:** Approved

---

## Goal

Give the sole operator full visibility into the running system from any device without
touching logs or the database. Cover: REST API with auth, Streamlit operator dashboard,
Prometheus metrics with Alertmanager rules.

---

## Architecture

```
Browser / external client
        │  HTTPS
        ▼
  ┌──────────────┐       ┌─────────────────┐
  │  Streamlit   │──────▶│  FastAPI :8000  │
  │  :8501       │  API  │  (X-API-Key)    │
  └──────────────┘  key  └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              PostgreSQL       Redis       BrokerRouter
              (history)     (live state)  (Alpaca/Tinkoff)

  Prometheus :9090 ──scrapes──▶ FastAPI /metrics (no auth, internal)
  Alertmanager ────rules──────▶ fires alerts (email / Telegram)
```

**Auth layers:**
- FastAPI: `X-API-Key` header (`FINALAYZE_API_KEY` env var). All endpoints except `/api/v1/health` require it.
- Streamlit: `st.secrets` password check on every page load. Calls FastAPI with the API key stored in `secrets.toml`.
- Prometheus `/metrics`: no auth, bind to internal Docker network only (not exposed externally).

---

## PR Split

| PR | Contents | Dependency |
|----|----------|------------|
| **B-1** | Core API — all endpoints + X-API-Key auth + Alembic migration 003 + integration tests | none |
| **B-2** | Streamlit dashboard — 5 pages, calls B-1 API, st.secrets password | B-1 merged |
| **B-3** | Prometheus metrics middleware + business metrics + Alertmanager rules + docker-compose.monitoring.yml | B-1 merged |

B-2 and B-3 can be developed in parallel after B-1 merges.

---

## API Design (PR B-1)

### Auth

```python
# src/finalayze/api/v1/auth.py
async def verify_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401)
```

`FINALAYZE_API_KEY` added to `config/settings.py` and `.env.example`.

### Endpoints

#### Health (no auth)
```
GET  /api/v1/health          # liveness: 200 if all critical components up, 503 otherwise
                             # response: {status, mode, version, components: {db, redis, alpaca, tinkoff, llm}}
GET  /api/v1/health/feeds    # feed health: last-seen timestamp per source + latency
```

#### System (auth required)
```
GET  /api/v1/system/status   # mode, uptime, component health details
GET  /api/v1/system/errors   # recent exceptions: [{timestamp, component, message, traceback_excerpt}]
POST /api/v1/system/mode     # body: {mode, confirm_token?}
                             # sandbox→real requires confirm_token = FINALAYZE_REAL_TOKEN env var
```

#### Portfolio
```
GET  /api/v1/portfolio                    # unified across all markets in base currency
GET  /api/v1/portfolio/{market_id}        # per-market (us / moex)
GET  /api/v1/portfolio/positions          # all open positions + unrealized P&L + stop distance (ATR)
GET  /api/v1/portfolio/positions/{symbol} # single position detail
GET  /api/v1/portfolio/history            # equity curve: [{timestamp, equity, drawdown_pct}]
GET  /api/v1/portfolio/performance        # Sharpe (30d), Sortino (30d), max_drawdown, win_rate, profit_factor
```

#### Trades
```
GET  /api/v1/trades                  # history — ?market=us&symbol=AAPL&limit=100&from=ISO&to=ISO
GET  /api/v1/trades/{trade_id}       # single trade detail for audit drill-down
GET  /api/v1/trades/analytics        # slippage stats, fill latency — ?market=moex&period=7d
```

#### Signals & Strategies
```
GET  /api/v1/signals                 # recent signals — ?market=us&segment=us_tech&limit=50
GET  /api/v1/strategies/performance  # per-strategy: {win_rate, profit_factor, trade_count_today, last_signal_at}
```

#### Risk
```
GET  /api/v1/risk/status    # circuit breaker level per market + level_since timestamp
GET  /api/v1/risk/exposure  # per-segment exposure: {segment_id, value_usd, pct_of_portfolio}
POST /api/v1/risk/override  # body: {market_id, level} — emergency manual circuit breaker control
```

#### ML
```
GET  /api/v1/ml/status      # per model: {segment, model_type, last_retrain, prediction_latency_p50}
```

#### News
```
GET  /api/v1/news           # recent articles — ?scope=global&limit=20
```

### Data sources per endpoint group

| Group | Source |
|-------|--------|
| Portfolio / positions | BrokerRouter → live Alpaca / Tinkoff calls |
| Portfolio history | `portfolio_snapshots` table (TimescaleDB) — requires migration 003 |
| Trades / signals / news | PostgreSQL (`orders`, `signals`, `news_articles` tables) |
| Risk status | In-memory `CircuitBreaker` state injected via FastAPI app state |
| System errors | In-memory ring buffer (last 100 exceptions) populated by logging handler |
| ML status | `MLModelRegistry` injected via app state |
| Feed health | In-memory timestamps updated by fetchers |

### Alembic migration 003

New table required:
```sql
CREATE TABLE portfolio_snapshots (
    timestamp   TIMESTAMPTZ NOT NULL,
    market_id   VARCHAR(10) NOT NULL,
    equity      NUMERIC(14,4),
    cash        NUMERIC(14,4),
    daily_pnl   NUMERIC(14,4),
    drawdown_pct NUMERIC(7,4),
    mode        VARCHAR(10),
    PRIMARY KEY (timestamp, market_id)
);
SELECT create_hypertable('portfolio_snapshots', 'timestamp');
```

`TradingLoop._strategy_cycle` writes a snapshot row after each cycle.

---

## Streamlit Dashboard (PR B-2)

**Entry point:** `src/finalayze/dashboard/app.py`

**Auth:** On every page load, check `st.secrets["password"]`. If not set or wrong, show login form. API key stored in `st.secrets["api_key"]`, base URL in `st.secrets["api_url"]`.

**HTTP client:** `httpx` with a shared client; calls wrapped in `asyncio.run()` inside Streamlit callbacks.

### Page 1: System Status
- Mode badge (REAL = red, TEST = yellow, SANDBOX = green, DEBUG = grey)
- Component health table: Alpaca, Tinkoff, LLM, DB, Redis — green/red + last heartbeat
- Recent errors list (last 10, collapsible tracebacks)
- Quick action: mode switcher with two-step confirmation for REAL

### Page 2: Portfolio
- Equity curve chart with drawdown shading below (Altair/Plotly)
- Summary row: total equity (USD), daily P&L %, cash %
- Position heatmap by segment — rows = segments, cells = unrealized P&L %
- Per-market equity table (USD equivalent)

### Page 3: Trades
- Filterable table: market, symbol, side, qty, filled price, slippage bps, timestamp
- Slippage scatter: x = time-of-day, y = slippage bps, color = market

### Page 4: Signals
- Strategy performance matrix: strategy × {win_rate, profit_factor, trades_today, last_signal_at}
- Recent signals table: symbol, strategy, direction, confidence, segment

### Page 5: Risk
- Circuit breaker badges per market with "since [timestamp]"
- Per-segment exposure bar chart (% of total portfolio)
- Emergency override form: select market + level (requires confirmation)

---

## Prometheus Metrics (PR B-3)

**Implementation:** `prometheus-fastapi-instrumentator` for HTTP metrics + custom `MetricsCollector` class updated by `TradingLoop` after each cycle.

**Endpoint:** `GET /metrics` — bound to internal Docker network, not authenticated.

### Metric definitions

```python
# Portfolio
finalayze_portfolio_equity_usd{market}                    Gauge
finalayze_portfolio_equity_pct_change{market, period}     Gauge   # period: 1d, 7d, 30d
finalayze_daily_pnl_usd{market}                           Gauge
finalayze_drawdown_pct{market}                            Gauge   # current drawdown 0.0–1.0
finalayze_max_drawdown_pct{market}                        Gauge   # rolling 30d

# Positions
finalayze_open_positions_count{market}                    Gauge

# Circuit breakers
finalayze_circuit_breaker_level{market}                   Gauge   # 0=normal 1=caution 2=halted 3=liquidate

# Execution
finalayze_trades_total{market, side}                      Counter
finalayze_trade_slippage_bps{market}                      Histogram
finalayze_order_fill_latency_seconds{market}              Histogram
finalayze_order_rejection_total{market, reason}           Counter

# Strategy
finalayze_strategy_win_rate{market, strategy}             Gauge   # rolling 100 trades
finalayze_strategy_signal_count{market, strategy, dir}    Counter

# ML model health
finalayze_ml_model_last_retrain_timestamp{model}          Gauge   # Unix timestamp
finalayze_ml_model_prediction_latency_seconds{model}      Histogram

# Data feed health
finalayze_market_data_feed_latency_seconds{market, src}   Histogram
finalayze_news_feed_last_article_timestamp{scope}         Gauge   # Unix timestamp

# Currency
finalayze_usd_rub_rate                                    Gauge
finalayze_portfolio_equity_rub                            Gauge   # MOEX book in RUB
```

### Alertmanager rules

```yaml
groups:
  - name: finalayze_trading
    rules:
      - alert: CircuitBreakerHalted
        expr: finalayze_circuit_breaker_level >= 2
        for: 1m
        annotations:
          summary: "Circuit breaker {{ $labels.market }} level {{ $value }}"

      - alert: HighDrawdown
        expr: finalayze_drawdown_pct > 0.08
        for: 5m
        annotations:
          summary: "Drawdown {{ $value | humanizePercentage }} on {{ $labels.market }}"

      - alert: NewsFeedSilent
        expr: time() - finalayze_news_feed_last_article_timestamp > 1800
        annotations:
          summary: "No news articles for 30+ minutes (scope={{ $labels.scope }})"

      - alert: MLModelStale
        expr: time() - finalayze_ml_model_last_retrain_timestamp > 259200  # 3 days
        annotations:
          summary: "ML model {{ $labels.model }} not retrained in 3+ days"

      - alert: NoTradesDeadMan
        expr: increase(finalayze_trades_total[2h]) == 0
        annotations:
          summary: "No trades in 2 hours — system may be stuck"
```

### Docker Compose additions (`docker-compose.monitoring.yml`)

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.51.0
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml

  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports: ["9093:9093"]
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

**Note:** Grafana deferred to a later observability hardening track.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `portfolio_snapshots` table missing | Migration 003 is task 1 of B-1 |
| Streamlit async threading | Use `httpx` + `asyncio.run()`, single shared client |
| Prometheus multi-worker double-count | Set `PROMETHEUS_MULTIPROC_DIR` env var, use `CollectorRegistry(multiprocess)` |
| Mode change to REAL by accident | `POST /system/mode` to `real` requires `confirm_token` matching `FINALAYZE_REAL_TOKEN` env var |
| API key leaked in logs | `verify_api_key` dependency redacts key from all log output; tested explicitly |

---

## Acceptance Criteria

Track B is done when an operator can answer all of the following using only the dashboard + API:

1. `GET /health` returns 200 with component status; 503 if DB or any broker unreachable
2. Portfolio page shows equity < 30s stale for both US and MOEX in USD
3. All open positions visible with unrealized P&L and stop distance
4. Circuit breaker level-2+ state is visually prominent (red badge with timestamp)
5. Last 100 trades visible and filterable; each trade has enough detail to reconstruct what happened
6. Switching to REAL mode requires two-step confirmation; wrong token is rejected
7. At least one Alertmanager alert fires in a test scenario (circuit breaker trip to level 2)
8. All new API endpoints have integration tests; Streamlit pages have smoke tests
9. `GET /metrics` scraped by Prometheus in docker-compose stack
10. API key never logged or shown in dashboard UI
11. Full stack starts with `docker compose up`
12. All endpoints respond < 500ms under normal load
