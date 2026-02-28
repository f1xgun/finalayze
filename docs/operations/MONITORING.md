# Monitoring & Alerting

## Prometheus Metrics

The application exposes metrics at `/metrics` (no auth, blocked by nginx externally).
Prometheus scrapes this endpoint on the internal Docker network.

### Metrics Reference

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `finalayze_trades_total` | Counter | market, side | Total trades executed |
| `finalayze_trades_filled_total` | Counter | market, side | Successfully filled trades |
| `finalayze_trades_rejected_total` | Counter | market, reason | Rejected trades |
| `finalayze_trade_latency_seconds` | Histogram | market | Order submission to fill latency |
| `finalayze_signals_total` | Counter | strategy, direction | Signals generated |
| `finalayze_portfolio_equity` | Gauge | market | Current portfolio equity |
| `finalayze_portfolio_cash` | Gauge | market | Available cash |
| `finalayze_drawdown_pct` | Gauge | market | Current drawdown percentage |
| `finalayze_circuit_breaker_level` | Gauge | market | Circuit breaker level (0-3) |
| `finalayze_news_articles_total` | Counter | source | Articles fetched |
| `finalayze_news_sentiment_score` | Gauge | segment | Latest sentiment score |
| `finalayze_news_feed_last_article_timestamp` | Gauge | scope | Timestamp of last article |
| `finalayze_ml_predictions_total` | Counter | model | ML predictions made |
| `finalayze_ml_model_last_retrain_timestamp` | Gauge | model | Last retrain timestamp |
| `finalayze_api_requests_total` | Counter | method, endpoint, status | HTTP requests |
| `finalayze_api_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `finalayze_data_fetch_errors_total` | Counter | source | Data fetch failures |
| `finalayze_rate_limit_waits_total` | Counter | limiter | Rate limiter wait events |

### Viewing Metrics

```bash
# From inside the Docker network
curl http://app:8000/metrics

# Prometheus UI
open http://localhost:9090
```

## Alert Rules

Defined in `monitoring/alerts.yml`, loaded by Prometheus.

| Alert | Expression | For | Severity | Action |
|-------|-----------|-----|----------|--------|
| `CircuitBreakerHalted` | `finalayze_circuit_breaker_level >= 2` | 1m | critical | Trading halted — check drawdown, see Runbook |
| `HighDrawdown` | `finalayze_drawdown_pct > 0.08` | 5m | warning | Review positions, consider manual liquidation |
| `NewsFeedSilent` | `time() - finalayze_news_feed_last_article_timestamp > 1800` | — | warning | Check API key, NewsAPI status |
| `MLModelStale` | `time() - finalayze_ml_model_last_retrain_timestamp > 259200` | — | warning | Run model retrain |
| `NoTradesDeadMan` | `increase(finalayze_trades_total[2h]) == 0` | 2h | warning | Check if market is open, review logs |

## Alertmanager

Configured via `monitoring/alertmanager.yml`. Sends alerts to configured
receivers (Telegram by default via `TelegramAlerter`).

```bash
# Start monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# Check Alertmanager UI
open http://localhost:9093
```

## Daily Monitoring Checklist

1. Check `/api/v1/health` — all components should be `ok`
2. Review Prometheus alerts — no firing critical alerts
3. Verify `finalayze_trades_total` is incrementing during market hours
4. Check `finalayze_drawdown_pct` is within acceptable range (< 5%)
5. Verify news feeds are active (`NewsFeedSilent` not firing)
6. Review `/api/v1/system/errors` for any recurring errors
7. Check `finalayze_circuit_breaker_level` is 0 for all markets
8. Verify ML model freshness (retrained within 3 days)
