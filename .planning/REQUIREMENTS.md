# Requirements: Finalayze v3.0 Production Readiness

**Defined:** 2026-03-21
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v3.0 Requirements

Requirements for production readiness milestone. Each maps to roadmap phases.

### Sandbox Monitoring

- [x] **MON-01**: SandboxMonitorService collects per-cycle metrics (trades, P&L, equity, fill rate, uptime) and persists to TimescaleDB
- [x] **MON-02**: Slippage capture records expected_price at signal time vs fill_price at execution, computes realized slippage in bps
- [x] **MON-03**: Streamlit sandbox dashboard page shows real-time trade log, equity curve, uptime %, fill rate, slippage histogram
- [x] **MON-04**: Anomaly detector triggers Telegram alerts on drawdown spikes (>2σ), fill rate drops (<90%), and slippage outliers (>50bps)

### Go/No-Go Gate

- [x] **GATE-01**: GoNoGoReporter evaluates formalized thresholds (uptime ≥99%, fill rate ≥95%, max DD <5%, trades ≥5/5days, signal divergence <50%)
- [x] **GATE-02**: Gate thresholds derived from walk-forward backtest distribution percentiles, not hardcoded round numbers
- [x] **GATE-03**: REST endpoint `/sandbox/gonogo` returns structured pass/fail report with per-criterion breakdown

### Gradual Rollout

- [x] **ROLL-01**: RolloutPhase enum (MINIMAL/STANDARD/FULL) with per-phase capital and position limits in Settings
- [x] **ROLL-02**: PreTradeChecker and CircuitBreaker respect RolloutPhase limits (3% max position at MINIMAL, 1% daily loss, 2% DD auto-stop)
- [x] **ROLL-03**: Capital ladder validation confirms position sizing produces valid lot sizes at each tier (50K/150K/500K/2.5M RUB)

### Production Operations

- [x] **OPS-01**: Kill switch cancels all open orders at broker, stops TradingLoop, sends Telegram critical alert — response time <30 seconds
- [x] **OPS-02**: Health check heartbeat every 5 minutes, REST `/health/production` endpoint, auto-alert on 2 missed heartbeats
- [x] **OPS-03**: 3-tier alert taxonomy (critical/warning/info) integrated into TelegramMonitor priority queue to prevent alert fatigue
- [x] **OPS-04**: Telegram bot `/kill` command triggers kill switch, `/gonogo` command runs gate report

## Future Requirements

Deferred to v4.0+.

### Expansion

- **EXP-01**: Multi-account support (multiple Tinkoff portfolios)
- **EXP-02**: Tax optimization (NDFL calculation, IIS deductions)
- **EXP-03**: Cross-market correlations (MOEX vs US for hedging)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Automated live promotion without human checkpoint | Go/no-go is advisory — human decides |
| Per-cycle shadow backtest | Compute cost too high for real-time operation |
| Multi-account rollout | Tinkoff API doesn't support it |
| Mobile monitoring app | Telegram + Streamlit dashboard sufficient |
| Advanced ML anomaly detection (LSTM/Prophet) | Rolling z-score sufficient for step-function events |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MON-01 | Phase 16 | Complete |
| MON-02 | Phase 16 | Complete |
| MON-03 | Phase 18 | Complete |
| MON-04 | Phase 16 | Complete |
| GATE-01 | Phase 16 | Complete |
| GATE-02 | Phase 16 | Complete |
| GATE-03 | Phase 18 | Complete |
| ROLL-01 | Phase 15 | Complete |
| ROLL-02 | Phase 15 | Complete |
| ROLL-03 | Phase 15 | Complete |
| OPS-01 | Phase 17 | Complete |
| OPS-02 | Phase 17 | Complete |
| OPS-03 | Phase 17 | Complete |
| OPS-04 | Phase 17 | Complete |

**Coverage:**
- v3.0 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-03-21*
*Traceability updated: 2026-03-21*
