# Requirements: Finalayze v10.1

**Defined:** 2026-04-14
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v10.1 Requirements

Requirements for Dashboard & Monitoring milestone. Each maps to roadmap phases 54-57.

### Position Monitor & Stop-Loss

- [ ] **STOP-01**: REST endpoint `/api/v1/positions/stops` returns current stop-loss state per position (entry_price, highest_price, current_stop, trail_activated, distance_pct)
- [ ] **STOP-02**: Dashboard positions table shows stop_level and distance_to_stop columns alongside existing price data
- [ ] **STOP-03**: Stop-loss history chart per position (entry to trail progression to current) on position detail view
- [ ] **STOP-04**: Position risk heatmap color-codes positions by distance to stop (green >5%, yellow 2-5%, red <2%)

### Trades Analytics

- [ ] **TRAD-01**: Trades endpoint computes slippage_bps by comparing fill_price to signal price at signal time
- [ ] **TRAD-02**: Trade analytics endpoint returns win_rate, avg_win, avg_loss, profit_factor computed from filled orders

### Strategy Performance

- [ ] **SIGP-01**: Strategy performance endpoint returns per-strategy win_rate, profit_factor, signal_count, last_signal computed from signals+orders join
- [ ] **SIGP-02**: Dashboard signals page shows strategy performance heatmap with green/red coloring by win rate

### Equity Curve & Performance

- [ ] **EQTY-01**: Daily equity snapshots table records equity per market per day; TradingLoop persists snapshot at end of each strategy cycle
- [ ] **EQTY-02**: Portfolio `/history` endpoint returns 30-day equity curve from snapshots with drawdown overlay

### Performance Metrics

- [ ] **PERF-01**: Portfolio `/performance` endpoint computes rolling 30d Sharpe, Sortino, max drawdown from equity snapshots
- [ ] **PERF-02**: Dashboard equity curve uses Plotly with drawdown area chart (same as sandbox page pattern)

### Alerting & Notifications

- [ ] **ALRT-01**: Stop-loss triggered alert includes symbol, entry_price, stop_price, P&L amount and %, hold duration
- [ ] **ALRT-02**: New signal alert includes strategy name, confidence, current position status (new/add/flip)
- [ ] **ALRT-03**: Dashboard `/alerts` page shows chronological alert history from DB (persist alerts to alerts table)
- [ ] **ALRT-04**: Daily summary alert at market close with total P&L, positions opened/closed, equity change

## v10.0 Requirements (prior milestone)

Retained for reference. See v10.0 REQUIREMENTS.md in main worktree.

### News Pipeline

- [ ] **NEWS-01**: News pipeline processes MOEX RSS feeds with 5-second per-article LLM timeout
- [ ] **NEWS-02**: NewsAnalyzer migrated from json.loads() to parse_structured()
- [ ] **NEWS-03**: Source credibility map wired (RSS: 0.8, Telegram: 0.7)
- [ ] **NEWS-04**: Ticker whitelist validation against InstrumentRegistry
- [ ] **NEWS-05**: LLM liveness check with Telegram alert on sustained failure
- [ ] **NEWS-06**: Article budget cap (max 20 articles/cycle)

### EventDriven Activation

- [ ] **EVNT-01**: EventDrivenStrategy enabled on all ru_* segments with weight 0.15
- [ ] **EVNT-02**: CBR/dividend duplicate signal guard
- [ ] **EVNT-03**: Sentiment decay respects market hours

### Portfolio Review Agent

- [ ] **PFRA-01**: Daily LLM portfolio review with structured output
- [ ] **PFRA-02**: Telegram delivery with concentration risk analysis
- [ ] **PFRA-03**: Advisory-only enforcement

### Anomaly Interpreter

- [ ] **ANMI-01**: Raw alert fires immediately, async LLM enrichment follows
- [ ] **ANMI-02**: LLM explanation labeled "AI interpretation (unverified)"
- [ ] **ANMI-03**: LLM failure does not suppress raw alert

### Sentiment ML Infrastructure

- [ ] **STML-01**: TimescaleDB continuous aggregate for rolling sentiment
- [ ] **STML-02**: SentimentStore reader for future ML features

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time WebSocket streaming to dashboard | Polling/refresh sufficient for daily trading cadence |
| Custom alert routing rules (per-strategy, per-severity) | Overkill for single-operator system; all alerts go to one Telegram chat |
| Multi-user dashboard auth | Single operator, localhost access only |
| Backtest equity curves in dashboard | Already in Streamlit sandbox page; this milestone covers live equity only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STOP-01 | Phase 54 | Pending |
| STOP-02 | Phase 54 | Pending |
| STOP-03 | Phase 54 | Pending |
| STOP-04 | Phase 54 | Pending |
| TRAD-01 | Phase 55 | Pending |
| TRAD-02 | Phase 55 | Pending |
| SIGP-01 | Phase 55 | Pending |
| SIGP-02 | Phase 55 | Pending |
| EQTY-01 | Phase 56 | Pending |
| EQTY-02 | Phase 56 | Pending |
| PERF-01 | Phase 56 | Pending |
| PERF-02 | Phase 56 | Pending |
| ALRT-01 | Phase 57 | Pending |
| ALRT-02 | Phase 57 | Pending |
| ALRT-03 | Phase 57 | Pending |
| ALRT-04 | Phase 57 | Pending |

**Coverage:**
- v10.1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-04-14*
*Last updated: 2026-04-14 -- traceability filled after roadmap creation*
