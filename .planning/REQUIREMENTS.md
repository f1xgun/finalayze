# Requirements: Finalayze

**Defined:** 2026-03-23
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v5.0 Requirements

Requirements for Data Flow Correctness & Live-Backtest Parity milestone.

### Order Sizing Bugs

- [x] **SIZE-01**: SELL orders use actual held position quantity, not Kelly-computed amount — no over/under-sell
- [x] **SIZE-02**: Sector exposure calculation uses each position's own last price, not current instrument's price
- [x] **SIZE-03**: CAUTION confidence threshold computed as `segment.min_combined_confidence * 1.2`, not hardcoded 0.6

### Live-Backtest Parity

- [x] **PARITY-01**: Live trading loop uses PositionSizingPipeline with all steps (VolTarget, Regime, MetaLabel, Copula, EVT, HardCaps) — matching backtest engine
- [x] **PARITY-02**: Live trailing stops ratchet upward after activation threshold, matching SimulatedBroker trailing stop behavior
- [x] **PARITY-03**: All 14 pre-trade checks receive required parameters in live path — stop_loss_price, has_pending_order, regime_state, strategy_name, correlations are passed
- [x] **PARITY-04**: Stop-loss exit in a cycle prevents same-cycle re-entry for the same symbol

### Data Validation

- [x] **DATA-01**: DataNormalizer.validate() runs on fetched candles before strategy processing — rejects negative prices, low > high, zero volume
- [x] **DATA-02**: Candle staleness detection active — configurable threshold (default: 2x timeframe interval), warning logged and instrument skipped when stale
- [x] **DATA-03**: IMOEX index candles use share volume (row[5]), not turnover value (row[4])

### News Pipeline

- [x] **NEWS-01**: News cycle skipped entirely when no segment has event_driven enabled — no LLM calls wasted
- [x] **NEWS-02**: Sentiment cache has time-based exponential decay (configurable half-life, default 4 hours) — stale sentiment decays to zero
- [ ] **NEWS-03**: Entity extractor _VALID_TICKERS contains "TCSG" (not "T") matching the extraction prompt
- [ ] **NEWS-04**: Telegram reader deduplicates messages by message link URL — no repeated processing within time window

### Data Infrastructure

- [x] **INFRA-01**: TinkoffFetcher reuses a persistent gRPC channel across calls (like TinkoffBroker pattern) — no per-call channel churn
- [x] **INFRA-02**: Brent crude candles cached via _cached_fetch() in MarketDataLoader — not re-downloaded on every backtest

### Intelligent News Impact

- [x] **NEWS-05**: NewsImpactAnalyzer replaces EntityExtractor + CombinedAnalyzer — single LLM call returns event_type, sentiment, confidence, affected_sectors (with direction/magnitude/reasoning), and optional direct_tickers
- [x] **NEWS-06**: SectorTickerMapper maps sector names (oil_gas, banking, metals, etc.) to MOEX tickers via static registry — no LLM for ticker resolution
- [x] **NEWS-07**: Per-ticker sentiment stored in _sentiment_cache as (segment_id, ticker) key — not just segment-level flat score
- [x] **NEWS-08**: Articles without explicit company mentions produce non-zero sentiment for affected tickers via sector mapping (e.g., "ЦБ повысил ставку" → banking tickers get negative sentiment)
- [x] **NEWS-09**: LLM calls per article reduced from 2 to 1 — NewsImpactAnalyzer prompt combines sentiment analysis, event classification, and sector impact prediction

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
| Full news pipeline rewrite | Only fix bugs and disable no-op; v6.0 can redesign |
| Event-driven strategy enablement | Requires live news feed validation first |
| Multi-source data fusion | Current single-source per data type is sufficient |
| Real-time WebSocket data feeds | Current polling interval (60 min) matches strategy timeframe |
| bond_cycle time.sleep fix | Covered by v4.0 pattern; bond fill timeout is short (2 min) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SIZE-01 | Phase 23 | Complete |
| SIZE-02 | Phase 23 | Complete |
| SIZE-03 | Phase 23 | Complete |
| PARITY-01 | Phase 24 | Complete |
| PARITY-02 | Phase 24 | Complete |
| PARITY-03 | Phase 24 | Complete |
| PARITY-04 | Phase 24 | Complete |
| DATA-01 | Phase 25 | Complete |
| DATA-02 | Phase 25 | Complete |
| DATA-03 | Phase 25 | Complete |
| NEWS-01 | Phase 26 | Complete |
| NEWS-02 | Phase 26 | Complete |
| NEWS-03 | Phase 26 | Pending |
| NEWS-04 | Phase 26 | Pending |
| INFRA-01 | Phase 25 | Complete |
| INFRA-02 | Phase 25 | Complete |
| NEWS-05 | Phase 27 | Complete |
| NEWS-06 | Phase 27 | Complete |
| NEWS-07 | Phase 27 | Complete |
| NEWS-08 | Phase 27 | Complete |
| NEWS-09 | Phase 27 | Complete |

**Coverage:**
- v5.0 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-23*
*Last updated: 2026-03-23 after initial definition*
