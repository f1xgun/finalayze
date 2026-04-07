# Requirements: Finalayze

**Defined:** 2026-03-30
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v6.0 Requirements

Requirements for Sandbox Stability & Observability milestone.

### gRPC Stability

- [x] **GRPC-01**: gRPC PollerCompletionQueue runs on a dedicated event loop isolated from APScheduler — no BlockingIOError flooding the main asyncio loop, strategy cycles fire within 5 min of scheduled time
- [x] **GRPC-02**: TinkoffBroker reconnects gRPC channel on StatusCode.INTERNAL (error 70001) — automatic recovery within 1 retry cycle, no multi-hour outage windows
- [x] **GRPC-03**: Portfolio fetch failure falls back to last-known portfolio state — strategy cycle continues with cached positions instead of skipping entirely

### Data Persistence

- [x] **PERSIST-01**: Executed orders persisted to `orders` table after fill — symbol, side, quantity, fill_price, order_id, timestamp stored
- [x] **PERSIST-02**: Generated signals persisted to `signals` table — strategy, symbol, direction, confidence, reasoning stored
- [x] **PERSIST-03**: Processed news articles persisted to `news_articles` table — title, source, published_at, content hash stored
- [x] **PERSIST-04**: Sentiment scores persisted to `sentiment_scores` table — ticker, score, source, timestamp stored
- [x] **PERSIST-05**: DB write failures are fire-and-forget with structured logging — never crash the trading loop or increment consecutive error counter

### Observability

- [x] **OBS-01**: Promtail ships Docker container logs to Loki — `/var/lib/docker/containers` mounted, JSON log format parsed correctly
- [x] **OBS-02**: Loki retains queryable logs for 30 days — dashboard queries return results for all 7 containers
- [x] **OBS-03**: FX rate (USD/RUB) fetched from CBR XML API as fallback when gRPC FX fetch fails — `finalayze_usd_rub_rate` metric is non-zero

### Operational Hygiene

- [x] **OPS-01**: Strategy cycle skips execution when MOEX market is closed — no wasted cycles with 0 instruments processed
- [x] **OPS-02**: Stale tickers removed/updated in config/segments.py — FIVE, FIXP, POLY removed; YNDX→YDEX; HHRU→HH (if valid)
- [x] **OPS-03**: LLM article deduplication via content hash — seen articles skipped within 24h TTL window, reducing rate-limit fallbacks
- [x] **OPS-04**: Telegram alerter startup failure does not block trading loop launch — alert sent on next successful cycle instead

## v7.0 Requirements

Requirements for Agent Intelligence & Experiment Framework milestone.

### Sandbox Signal Fixes

- [ ] **SANDBOX-FIX-01**: `_CANDLE_LOOKBACK >= 210` in trading loop — RSI2 Connors (SMA-200), dual_momentum (126 bars), OU mean reversion (126 bars) all receive sufficient data in live mode
- [ ] **SANDBOX-FIX-02**: `TradingLoop.start()` checks `KillSwitch.is_killed` before starting scheduler — killed system does not resume on Docker restart
- [ ] **SANDBOX-FIX-03**: When `FINALAYZE_MODE=sandbox` and `rollout_phase` not explicitly set, effective rollout is MINIMAL — sandbox starts with conservative risk limits
- [ ] **SANDBOX-FIX-04**: Staleness threshold handles weekends and MOEX holidays — Monday morning and post-holiday cycles not blocked by 48h threshold
- [x] **SANDBOX-FIX-05**: TinkoffFetcher wrapped in CachingFetcher in sandbox mode — repeated API calls for same data eliminated
- [x] **SANDBOX-FIX-06**: RateLimiter passed to TinkoffFetcher in sandbox — API throttling prevented for large instrument universes
- [x] **SANDBOX-FIX-07**: `FINALAYZE_LLM_API_KEY` documented and event_driven enabled for ru_blue_chips, ru_energy, ru_finance — news pipeline activated for MOEX
- [x] **SANDBOX-FIX-08**: Per-gate signal drop counters in ValidationLogger — instruments_no_bars, signals_below_threshold, signals_pre_trade_rejected tracked separately
- [ ] **SANDBOX-FIX-09**: ML quality gate bug: profit_factor gate populated with actual PF from fold predictions — gate no longer always fails with default 1.0
- [ ] **SANDBOX-FIX-10**: ML quality gate: Brier score evaluated on calibrated probabilities — calibrator applied during walk-forward evaluation

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
| GRPC-01 | Phase 29 | Complete |
| GRPC-02 | Phase 30 | Complete |
| GRPC-03 | Phase 30 | Complete |
| PERSIST-01 | Phase 31 | Complete |
| PERSIST-02 | Phase 31 | Complete |
| PERSIST-03 | Phase 31 | Complete |
| PERSIST-04 | Phase 31 | Complete |
| PERSIST-05 | Phase 31 | Complete |
| OBS-01 | Phase 29 | Complete |
| OBS-02 | Phase 29 | Complete |
| OBS-03 | Phase 30 | Complete |
| OPS-01 | Phase 28 | Complete |
| OPS-02 | Phase 28 | Complete |
| OPS-03 | Phase 28 | Complete |
| OPS-04 | Phase 28 | Complete |

**Coverage:**
- v6.0 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-03-30 after roadmap creation*
