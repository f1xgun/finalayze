# Roadmap: Finalayze MOEX MVP

## Overview

This roadmap takes Finalayze from a working US equity trading system to a fully autonomous MOEX trading platform covering stocks, OFZ bonds, and LLM-driven news signals. The build order follows a strict dependency chain: data correctness (RUB sizing) must come first, then equity validation, then bond data, then bond execution, then full integration, then sandbox proof, and finally news pipeline with real money deployment. Each phase delivers a verifiable capability that unblocks the next.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: MOEX Equity Foundation** - Fix RUB sizing, wire MOEX costs and holiday calendar into backtest engine (completed 2026-03-14)
- [ ] **Phase 2: MOEX Equity Validation** - Tune ru_* strategy presets and achieve positive walk-forward backtest PnL
- [ ] **Phase 3: Bond Data Pipeline** - Wire bond instrument discovery, NKD/dirty price math, and MacroCacheService
- [ ] **Phase 4: Bond Execution** - Complete BondCycleProcessor stubs, YieldStop, and achieve positive bond backtest PnL
- [ ] **Phase 5: Integration and Telegram** - Wire equity+bond cycles into TradingLoop, build Telegram alerting with rate limiting
- [ ] **Phase 6: Sandbox Validation** - Prove 5+ days autonomous operation in T-Invest sandbox without critical errors
- [ ] **Phase 7: News Pipeline and Go-Live** - Connect Russian news sources, enable event_driven strategy, deploy real money

## Phase Details

### Phase 1: MOEX Equity Foundation
**Goal**: MOEX equity positions are correctly sized in RUB with accurate costs and calendar awareness
**Depends on**: Nothing (first phase)
**Requirements**: EQF-01, EQF-04, EQF-05
**Success Criteria** (what must be TRUE):
  1. MOEX backtest positions are sized at 10-20% of RUB equity (not 0.02%)
  2. Backtest engine skips MOEX non-trading days (public holidays, not just weekends)
  3. MOEX commission and slippage costs are deducted in backtest PnL calculations
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Fix MOEX commission rate (0.04%) and add transferred holidays with TradingLoop integration
- [ ] 01-02-PLAN.md — Fix RUB position sizing (1M RUB capital) and validate MOEX backtest on all segments

### Phase 2: MOEX Equity Validation
**Goal**: MOEX equity strategies produce profitable results in walk-forward backtests
**Depends on**: Phase 1
**Requirements**: EQF-02, EQF-03
**Success Criteria** (what must be TRUE):
  1. Walk-forward backtest shows positive PnL on at least 2 MOEX segments (ru_blue_chips, ru_energy)
  2. ru_* YAML strategy presets are calibrated with MOEX-specific parameters (not US defaults)
  3. Walk-forward out-of-sample Sharpe > 0 over 2022-2025 period
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Bond Data Pipeline
**Goal**: Bond instruments are discoverable and all bond math (YTM, duration, NKD, dirty price) computes correctly
**Depends on**: Phase 1
**Requirements**: BDP-01, BDP-02, BDP-03, BDP-04, BDP-05
**Success Criteria** (what must be TRUE):
  1. TinkoffFetcher returns non-empty candle series for OFZ-PD and OFZ-PK instruments
  2. NKD (accrued coupon interest) and dirty price match known OFZ settlement examples
  3. MacroCacheService provides CBR key rate within 24 hours of actual CBR value
  4. QuantLib YTM and modified duration calculations match manual bond math for test OFZ bonds
  5. Bond instrument registry contains OFZ and corporate bonds with correct FIGI mappings
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Bond Execution
**Goal**: BondCycleProcessor executes the full 4-layer bond pipeline without stubs, with proven positive PnL
**Depends on**: Phase 3
**Requirements**: BEX-01, BEX-02, BEX-03, BEX-04, BEX-05, BEX-06
**Success Criteria** (what must be TRUE):
  1. BondCycleProcessor._size_and_execute() submits real orders to T-Invest sandbox without errors
  2. YieldStop exits bond positions when current YTM crosses regime-adaptive thresholds
  3. Separate "moex_bonds" TinkoffBroker instance handles bond orders (no shared gRPC client with equity cycle)
  4. DV01BudgetStep uses dirty price (clean price + NKD) for cash sufficiency checks
  5. Bond backtest shows positive carry PnL with walk-forward validation on OFZ instruments
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Integration and Telegram
**Goal**: Equity and bond cycles run together in TradingLoop with reliable Telegram alerting for all trade events
**Depends on**: Phase 2, Phase 4
**Requirements**: AUT-01, AUT-02, AUT-03, MON-01, MON-02, MON-03, MON-04, MON-05
**Success Criteria** (what must be TRUE):
  1. TradingLoop runs concurrent equity and bond APScheduler cycles without gRPC errors
  2. Bond cycle is skipped on MOEX holidays; macro refresh runs 7 days/week regardless
  3. All circuit breakers fire correctly for both equity and bond layers
  4. Telegram bot delivers trade fill, stop-loss, and circuit breaker alerts within 60 seconds (even during 20-fill bursts)
  5. Daily P&L summary shows correct RUB amounts (not zero)
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Sandbox Validation
**Goal**: System proves autonomous operation capability in T-Invest sandbox over multiple trading days
**Depends on**: Phase 5
**Requirements**: AUT-04, AUT-06
**Success Criteria** (what must be TRUE):
  1. System runs 5+ consecutive trading days in T-Invest sandbox without critical errors
  2. LayerLedger reconciles with broker state on every startup (no ghost positions after restarts)
  3. Sandbox drawdown stays below 5%
  4. System recovers gracefully from network interruptions, API errors, and market data gaps
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: News Pipeline and Go-Live
**Goal**: Russian news feeds drive event-based trading signals and the system executes first real MOEX trades
**Depends on**: Phase 6
**Requirements**: NWS-01, NWS-02, NWS-03, NWS-04, NWS-05, AUT-05
**Success Criteria** (what must be TRUE):
  1. Russian news RSS reader fetches articles from at least 3 sources (RBC, Interfax, TASS/Kommersant)
  2. LLM analyzes Russian news and produces sentiment scores that influence combined MOEX signal
  3. event_driven strategy is enabled on MOEX segments and generates signals from news events
  4. Telegram channel reader ingests financial sentiment from configured channels
  5. First real MOEX trades execute on a small account (500K RUB) after sandbox validation passes
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD
- [ ] 07-03: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

Note: Phases 2 and 3 depend only on Phase 1 (not on each other) and could theoretically run in parallel, but sequential execution is recommended for a solo developer.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. MOEX Equity Foundation | 2/2 | Complete   | 2026-03-14 |
| 2. MOEX Equity Validation | 0/2 | Not started | - |
| 3. Bond Data Pipeline | 0/3 | Not started | - |
| 4. Bond Execution | 0/3 | Not started | - |
| 5. Integration and Telegram | 0/3 | Not started | - |
| 6. Sandbox Validation | 0/2 | Not started | - |
| 7. News Pipeline and Go-Live | 0/3 | Not started | - |
