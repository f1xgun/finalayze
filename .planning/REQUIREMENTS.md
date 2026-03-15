# Requirements: Finalayze MOEX MVP

**Defined:** 2026-03-14
**Core Value:** Autonomous profitable MOEX trading (stocks + bonds + coupons) with risk limits, operating without human intervention

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### MOEX Equity Foundation

- [x] **EQF-01**: Position sizing uses RUB denomination for MOEX segments (not USD)
- [x] **EQF-02**: MOEX backtest produces positive PnL with walk-forward validation
- [x] **EQF-03**: MOEX-specific strategy parameters tuned (ru_* YAML presets calibrated)
- [x] **EQF-04**: MOEX holiday calendar integrated (14-20 non-weekend non-trading days/year)
- [x] **EQF-05**: MOEX costs (commissions, slippage) fully wired in backtest engine

### Bond Data Pipeline

- [x] **BDP-01**: Bond candle data fetched via T-Invest API (GetBonds, GetBondCoupons)
- [x] **BDP-02**: NKD (accrued coupon interest) and dirty price computed correctly
- [x] **BDP-03**: MacroCacheService provides real-time CBR key rate and FX data
- [x] **BDP-04**: QuantLib integration for YTM, modified duration, convexity calculations
- [x] **BDP-05**: Bond instrument registry with FIGI mapping for OFZ and corporate bonds

### Bond Execution

- [x] **BEX-01**: `BondCycleProcessor._size_and_execute()` completes real order submission
- [x] **BEX-02**: `YieldStop._process_yield_stops()` computes current YTM and exits positions
- [x] **BEX-03**: Separate `moex_bonds` TinkoffBroker instance in BrokerRouter
- [x] **BEX-04**: DV01BudgetStep uses dirty price (not face_value) for cash calculations
- [x] **BEX-05**: Bond backtest shows positive carry PnL with walk-forward validation
- [x] **BEX-06**: LayerLedger reconciliation on startup (sync with broker state)

### News Pipeline

- [x] **NWS-01**: Russian news RSS feed reader (RBC, Interfax, TASS, Kommersant)
- [x] **NWS-02**: LLM analysis of Russian news via existing NewsAnalyzer + Russian prompts
- [x] **NWS-03**: Telegram channel reading for financial sentiment (Telethon)
- [x] **NWS-04**: News-driven signal generation (event impact → trading decision)
- [x] **NWS-05**: `event_driven` strategy enabled on MOEX segments

### Monitoring & Alerts

- [x] **MON-01**: Telegram bot sends trade alerts (fill, stop-loss, circuit breaker)
- [x] **MON-02**: Daily P&L summary fixed (currently shows zero)
- [x] **MON-03**: Telegram priority message queue (prevent loss during circuit breaker bursts)
- [x] **MON-04**: Coupon receipt alerts via Telegram
- [x] **MON-05**: CBR meeting alerts with impact analysis

### Autonomous Operation

- [x] **AUT-01**: BondCycleProcessor integrated into TradingLoop scheduler
- [x] **AUT-02**: MOEX trading schedule gate (skip non-trading days, respect hours)
- [x] **AUT-03**: All circuit breakers verified (equity + bond layers)
- [x] **AUT-04**: T-Invest sandbox validation: 5+ days autonomous operation without critical errors
- [x] **AUT-05**: Real money deployment on small account (first real MOEX trades)
- [x] **AUT-06**: Graceful error recovery (network, API, market data gaps)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### ML Enhancement

- **ML-01**: ML ensemble enabled for ru_* segments with MOEX-specific features
- **ML-02**: Cross-market correlations (MOEX vs US, Brent impact)
- **ML-03**: OFZ yield curve bootstrapping from CBR zero-coupon curve

### Advanced News

- **NWS-06**: T-Invest API news/events stream (if available)
- **NWS-07**: Telegram Pulse sentiment aggregation
- **NWS-08**: Multi-LLM comparison (Claude vs GPT for Russian news accuracy)

### Portfolio

- **PRT-01**: Multi-account support
- **PRT-02**: Cross-asset correlation-based allocation (stocks vs bonds)
- **PRT-03**: Tax optimization (NDFL, IIS deductions)

## Out of Scope

| Feature | Reason |
|---------|--------|
| US market changes in this milestone | Already works, not MVP focus |
| Derivatives/futures | Complexity too high for MVP |
| High-frequency trading | System operates on daily/intraday bars |
| Mobile app | Streamlit dashboard + Telegram sufficient |
| Cryptocurrency | Not available on MOEX |
| Custom ML training UI | CLI scripts sufficient |
| Python 3.13+ upgrade | Separate task, not blocking |
| Multi-broker support | T-Invest only for MVP |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EQF-01 | Phase 1 | Complete |
| EQF-02 | Phase 2 | Complete |
| EQF-03 | Phase 2 | Complete |
| EQF-04 | Phase 1 | Complete |
| EQF-05 | Phase 1 | Complete |
| BDP-01 | Phase 3 | Complete |
| BDP-02 | Phase 3 | Complete |
| BDP-03 | Phase 3 | Complete |
| BDP-04 | Phase 3 | Complete |
| BDP-05 | Phase 3 | Complete |
| BEX-01 | Phase 4 | Complete |
| BEX-02 | Phase 4 | Complete |
| BEX-03 | Phase 4 | Complete |
| BEX-04 | Phase 4 | Complete |
| BEX-05 | Phase 4 | Complete |
| BEX-06 | Phase 4 | Complete |
| NWS-01 | Phase 7 | Complete |
| NWS-02 | Phase 7 | Complete |
| NWS-03 | Phase 7 | Complete |
| NWS-04 | Phase 7 | Complete |
| NWS-05 | Phase 7 | Complete |
| MON-01 | Phase 5 | Complete |
| MON-02 | Phase 5 | Complete |
| MON-03 | Phase 5 | Complete |
| MON-04 | Phase 5 | Complete |
| MON-05 | Phase 5 | Complete |
| AUT-01 | Phase 5 | Complete |
| AUT-02 | Phase 5 | Complete |
| AUT-03 | Phase 5 | Complete |
| AUT-04 | Phase 6 | Complete |
| AUT-05 | Phase 7 | Complete |
| AUT-06 | Phase 6 | Complete |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 32
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after roadmap creation*
