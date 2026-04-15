# Requirements: Finalayze v10.0

**Defined:** 2026-04-15
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v10.0 Requirements

Requirements for runtime LLM trading agents milestone. Each maps to roadmap phases.

### News Pipeline

- [x] **NEWS-01**: News pipeline processes MOEX RSS feeds (RBC, Interfax, TASS, MOEX ISS) with 5-second per-article LLM timeout
- [x] **NEWS-02**: NewsAnalyzer migrated from json.loads() to parse_structured() for reliable sentiment parsing
- [x] **NEWS-03**: Source credibility map wired (RSS: 0.8, Telegram: 0.7) and passed to EventDrivenStrategy
- [x] **NEWS-04**: Ticker whitelist validation filters LLM-extracted entities against InstrumentRegistry
- [x] **NEWS-05**: LLM liveness check added to HealthMonitor with Telegram alert on sustained failure
- [x] **NEWS-06**: Article budget cap (max 20 articles/cycle) prevents LLM cost explosion

### EventDriven Activation

- [x] **EVNT-01**: EventDrivenStrategy enabled on all ru_* segments with weight 0.15
- [x] **EVNT-02**: CBR/dividend duplicate signal guard prevents double-weight with cbr_calendar strategy
- [x] **EVNT-03**: Sentiment decay respects market hours (freeze during MOEX close, resume on open)

### Portfolio Review Agent

- [x] **PFRA-01**: Daily LLM portfolio review runs outside market hours with structured PortfolioReviewResult output
- [x] **PFRA-02**: Review results delivered via Telegram with concentration risk and upcoming catalyst analysis
- [x] **PFRA-03**: Advisory-only enforcement — schema has no trade-directive fields, no write access to order pipeline

### Anomaly Interpreter

- [x] **ANMI-01**: AnomalyDetector fires raw alert immediately, then async LLM enrichment follows
- [x] **ANMI-02**: LLM explanation appended to Telegram alert labeled "AI interpretation (unverified)"
- [x] **ANMI-03**: Graceful degradation — LLM timeout/failure does not suppress or delay raw statistical alert

### Sentiment ML Infrastructure

- [x] **STML-01**: TimescaleDB continuous aggregate for rolling sentiment (1d/7d/30d buckets)
- [x] **STML-02**: SentimentStore reader (Layer 2) provides rolling aggregation query for future ML feature extraction

## v11.0 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Sentiment as ML Features

- **STML-10**: Sentiment rolling features (news_sentiment_ema_24h, news_sentiment_ema_7d, event_count_24h) added to XGBoost feature set
- **STML-11**: Sentiment features pass quality gates (Brier validation, feature importance budget, CPCV)

### Cached Reasoning Overlay

- **OVRL-01**: Async pre-computation of LLM trade reasoning in news_cycle → Redis cache
- **OVRL-02**: Lightweight sync ReasoningOverlayStep reads cached reasoning in PositionSizingPipeline

## Out of Scope

| Feature | Reason |
|---------|--------|
| Pre-Trade Reasoning Agent (LLM modifier in sizing pipeline) | Unanimous REJECT by 5 domain experts: non-determinism, uncalibrated output, irreproducible backtests, sync/async mismatch |
| Live A/B testing | 4/5 REJECT: formally underpowered (need 6+ years data), backtest via CPCV instead |
| T-Pulse API integration | T-Invest SDK has no news service; unofficial REST endpoint has uncertain post-2024 auth status |
| Autonomous rebalancing from Portfolio Review | Risk Officer condition: suggestions only, never autonomous execution |
| LLM control of CircuitBreaker | Risk Officer condition: read-only access, no override of automated risk systems |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| NEWS-01 | Phase 49 | Complete |
| NEWS-02 | Phase 49 | Complete |
| NEWS-03 | Phase 49 | Complete |
| NEWS-04 | Phase 49 | Complete |
| NEWS-05 | Phase 49 | Complete |
| NEWS-06 | Phase 49 | Complete |
| EVNT-01 | Phase 50 | Complete |
| EVNT-02 | Phase 50 | Complete |
| EVNT-03 | Phase 50 | Complete |
| ANMI-01 | Phase 51 | Complete |
| ANMI-02 | Phase 51 | Complete |
| ANMI-03 | Phase 51 | Complete |
| PFRA-01 | Phase 52 | Complete |
| PFRA-02 | Phase 52 | Complete |
| PFRA-03 | Phase 52 | Complete |
| STML-01 | Phase 53 | Complete |
| STML-02 | Phase 53 | Complete |

**Coverage:**
- v10.0 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-15*
*Last updated: 2026-04-15 — traceability filled after roadmap creation*
