# Requirements: Finalayze

**Defined:** 2026-03-30
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v6.0 Requirements

Requirements for Sandbox Stability & Observability milestone.

### gRPC Stability

- [ ] **GRPC-01**: gRPC PollerCompletionQueue runs on a dedicated event loop isolated from APScheduler — no BlockingIOError flooding the main asyncio loop, strategy cycles fire within 5 min of scheduled time
- [ ] **GRPC-02**: TinkoffBroker reconnects gRPC channel on StatusCode.INTERNAL (error 70001) — automatic recovery within 1 retry cycle, no multi-hour outage windows
- [ ] **GRPC-03**: Portfolio fetch failure falls back to last-known portfolio state — strategy cycle continues with cached positions instead of skipping entirely

### Data Persistence

- [ ] **PERSIST-01**: Executed orders persisted to `orders` table after fill — symbol, side, quantity, fill_price, order_id, timestamp stored
- [ ] **PERSIST-02**: Generated signals persisted to `signals` table — strategy, symbol, direction, confidence, reasoning stored
- [ ] **PERSIST-03**: Processed news articles persisted to `news_articles` table — title, source, published_at, content hash stored
- [ ] **PERSIST-04**: Sentiment scores persisted to `sentiment_scores` table — ticker, score, source, timestamp stored
- [ ] **PERSIST-05**: DB write failures are fire-and-forget with structured logging — never crash the trading loop or increment consecutive error counter

### Observability

- [ ] **OBS-01**: Promtail ships Docker container logs to Loki — `/var/lib/docker/containers` mounted, JSON log format parsed correctly
- [ ] **OBS-02**: Loki retains queryable logs for 30 days — dashboard queries return results for all 7 containers
- [ ] **OBS-03**: FX rate (USD/RUB) fetched from CBR XML API as fallback when gRPC FX fetch fails — `finalayze_usd_rub_rate` metric is non-zero

### Operational Hygiene

- [ ] **OPS-01**: Strategy cycle skips execution when MOEX market is closed — no wasted cycles with 0 instruments processed
- [ ] **OPS-02**: Stale tickers removed/updated in config/segments.py — FIVE, FIXP, POLY removed; YNDX→YDEX; HHRU→HH (if valid)
- [ ] **OPS-03**: LLM article deduplication via content hash — seen articles skipped within 24h TTL window, reducing rate-limit fallbacks
- [ ] **OPS-04**: Telegram alerter startup failure does not block trading loop launch — alert sent on next successful cycle instead

## Future Requirements

### News Pipeline Enhancement

- **NEWS-F01**: Article persistence to database with queryable API endpoint
- **NEWS-F02**: Prompt injection sanitization for LLM inputs

### Code Quality

- **QUAL-01**: Migrate 99 test files from core.trading_loop shim to canonical imports
- **QUAL-02**: Inject _alerter_ref via TradingLoop constructor parameter

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full gRPC SDK replacement | t-tech-investments SDK works, only isolation needed |
| Real-time WebSocket data feeds | Current polling interval matches strategy timeframe |
| Multi-provider LLM orchestration | Single primary + fallback is sufficient |
| Automated go/no-go from DB metrics | Advisory report pattern proven in v3.0 |
| Log-based alerting (Loki alerts) | Prometheus alerts already cover critical paths |
| DB-level trade analytics/reporting | Out of scope — persist first, analyze later |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GRPC-01 | Phase 29 | Pending |
| GRPC-02 | Phase 30 | Pending |
| GRPC-03 | Phase 30 | Pending |
| PERSIST-01 | Phase 31 | Pending |
| PERSIST-02 | Phase 31 | Pending |
| PERSIST-03 | Phase 31 | Pending |
| PERSIST-04 | Phase 31 | Pending |
| PERSIST-05 | Phase 31 | Pending |
| OBS-01 | Phase 29 | Pending |
| OBS-02 | Phase 29 | Pending |
| OBS-03 | Phase 30 | Pending |
| OPS-01 | Phase 28 | Pending |
| OPS-02 | Phase 28 | Pending |
| OPS-03 | Phase 28 | Pending |
| OPS-04 | Phase 28 | Pending |

**Coverage:**
- v6.0 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-03-30 after roadmap creation*
