# Roadmap: Finalayze

## Milestones

- ✅ **v1.0 MOEX MVP** -- Phases 1-7 (shipped 2026-03-19)
- **v2.0 MOEX Profitability** -- Phases 8-12 (in progress)

## Phases

<details>
<summary>v1.0 MOEX MVP (Phases 1-7) -- SHIPPED 2026-03-19</summary>

- [x] Phase 1: MOEX Equity Foundation (2/2 plans) -- completed 2026-03-14
- [x] Phase 2: MOEX Equity Validation (3/3 plans) -- completed 2026-03-14
- [x] Phase 3: Bond Data Pipeline (3/3 plans) -- completed 2026-03-14
- [x] Phase 4: Bond Execution (3/3 plans) -- completed 2026-03-14
- [x] Phase 5: Integration and Telegram (4/4 plans) -- completed 2026-03-14
- [x] Phase 6: Sandbox Validation (4/4 plans) -- completed 2026-03-15
- [x] Phase 7: News Pipeline and Go-Live (3/3 plans) -- completed 2026-03-15

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v2.0 MOEX Profitability

**Milestone Goal:** Transform MOEX equity from negative Sharpe to profitable operation through universe cleanup, MOEX-native strategies, macro regime gating, and portfolio-level OFZ/equity allocation.

- [x] **Phase 8: Data Foundation** - Fix vol target, universe, dividend calendar, and 2022 structural break so all MOEX backtests produce valid results (completed 2026-03-20)
- [x] **Phase 9: Strategy Wiring** - Connect existing DividendGap, CBR, and RUB/oil strategies to backtest engine and sizing pipeline (completed 2026-03-20)
- [x] **Phase 10: Macro Regime** - Add CBR regime sizing, OFZ rotation trigger, and sector allocation overlay to position sizing pipeline (completed 2026-03-20)
- [x] **Phase 11: Advanced Strategies and ML** - Preferred share arbitrage and ML ensemble with Russian macro features for ru_* segments (completed 2026-03-21)
- [x] **Phase 12: Portfolio Assembly** - Joint OFZ + equity backtest with 40/60 allocation and RUB crisis brake (completed 2026-03-21)

## Phase Details

### Phase 8: Data Foundation
**Goal**: All MOEX backtests produce valid, unbiased results with properly sized positions
**Depends on**: Nothing (first phase of v2.0; builds on v1.0 infrastructure)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. Running a walk-forward backtest on ru_blue_chips produces position sizes of 5-15% of equity (not 0.5-2% as currently)
  2. GAZP, VTBR, SNGS, IRAO, ALRS are excluded from all ru_* segment backtests and no trades are generated for them
  3. Dividend calendar contains 150+ events including cancelled/reduced dividends with a status field distinguishing paid/cancelled/reduced
  4. Walk-forward backtest on any ru_* segment with training window crossing Feb-Mar 2022 excludes the structural break period from vol and ATR calculations
**Plans:** 3/3 plans complete

Plans:
- [ ] 08-01-PLAN.md -- Vol target recalibration (0.40) and toxic symbol removal from MOEX universe
- [ ] 08-02-PLAN.md -- Feb-Mar 2022 structural break exclusion from vol/ATR calculations
- [ ] 08-03-PLAN.md -- Dividend calendar expansion to 150+ events with status field

### Phase 9: Strategy Wiring
**Goal**: Existing but unconnected strategies generate real trades in MOEX backtests, establishing a positive equity baseline
**Depends on**: Phase 8
**Requirements**: STRAT-01, STRAT-02, STRAT-03, STRAT-04
**Success Criteria** (what must be TRUE):
  1. DividendGapStrategy generates trades on ex-dividend dates using the expanded calendar, with per-symbol max_hold_bars (not force-closed at 15 bars)
  2. Dividend gap signals bypass ADX combiner routing and are not diluted below min_combined_confidence by other strategies
  3. CBRStrategyWrapper is registered in the combiner and generates signals around CBR rate decision dates
  4. BrentGateStep in the sizing pipeline reduces energy sector position sizes when Brent-in-RUB is below threshold
  5. RubOilRegimeStep in the sizing pipeline scales equity positions based on RUB/oil decorrelation state
**Plans:** 2/2 plans complete

Plans:
- [x] 09-01-PLAN.md -- DividendGap/CBR combiner wiring with event strategy bypass and yield-based hold bars
- [x] 09-02-PLAN.md -- RubOilRegimeStep and BrentGateStep sizing pipeline integration

### Phase 10: Macro Regime
**Goal**: MOEX equity positions are sized according to CBR rate regime, sector allocation rotates based on macro conditions, and OFZ allocation shifts when CBR cutting cycle begins
**Depends on**: Phase 9
**Requirements**: MACRO-01, MACRO-02, MACRO-03
**Success Criteria** (what must be TRUE):
  1. CBRRegimeStep in the sizing pipeline scales equity positions down during hiking cycles and up during cutting cycles, using OFZ yield curve slope as a leading indicator (not raw CBR announcements)
  2. OFZ PK-to-PD rotation triggers when CBR cutting cycle is detected (2+ consecutive cuts), shifting bond allocation from floating-rate PK to fixed-rate PD
  3. SectorAllocationStep in the sizing pipeline (not in combiner) adjusts sector weights using MOEX sector indices -- energy overweight when Brent elevated, financials sensitive to CBR direction
**Plans:** 2/2 plans complete

Plans:
- [x] 10-01-PLAN.md -- CBRRegimeStep and SectorAllocationStep in equity sizing pipeline
- [x] 10-02-PLAN.md -- OFZ PK-to-PD rotation trigger in BondCycleProcessor

### Phase 11: Advanced Strategies and ML
**Goal**: Preferred share arbitrage captures pref/ord spread convergence, and ML ensemble operates on ru_* segments with Russian macro features
**Depends on**: Phase 10 (requires MacroSnapshot extensions and positive equity baseline)
**Requirements**: ADV-01, ADV-02, ADV-03
**Success Criteria** (what must be TRUE):
  1. Preferred share arbitrage strategy generates long-only trades on SBER/SBERP and TATN/TATNP pairs when spread z-score exceeds 2.0, with cointegration validated on post-2022 data
  2. 10 Russian macro ML features (CBR rate level/delta/direction, USDRUB return/zscore/vol, Brent return, IMOEX relative, turnover zscore) are computed per bar and available to the ML pipeline
  3. ML ensemble is enabled for at least one ru_* segment in reinforcer-only mode, with quality gates passing on 2024-2025 calm-period validation data
**Plans:** 3/4 plans complete

Plans:
- [x] 11-01-PLAN.md -- Preferred share arbitrage (allow_short, pairs preset, TATNP FIGI)
- [x] 11-02-PLAN.md -- 7 new Russian macro ML features (CBR/FX/Brent) and schema version bump
- [x] 11-03-PLAN.md -- ML ensemble enablement for ru_blue_chips (training + quality gates)
- [ ] 11-04-PLAN.md -- Gap closure: retrain ML models with optimized hyperparameters to pass quality gates

### Phase 12: Portfolio Assembly
**Goal**: Combined OFZ + equity portfolio operates as a single system with aggregate risk management and walk-forward Sharpe >= +0.10
**Depends on**: Phase 11 (requires working equity strategies and ML)
**Requirements**: PORT-01, PORT-02, PORT-03
**Success Criteria** (what must be TRUE):
  1. PortfolioBacktestOrchestrator runs bond and equity engines jointly, producing a merged equity curve with aggregate Sharpe, max drawdown, and profit factor metrics
  2. Portfolio allocation enforces 40% OFZ carry + 60% equity split with monthly rebalancing, and RUB crisis brake shifts to 80% OFZ when USD/RUB spikes more than 15% over 20 bars
  3. Blended MOEX portfolio walk-forward Sharpe is >= +0.10 across the combined OFZ + equity curve
**Plans:** 2/2 plans complete

Plans:
- [x] 12-01-PLAN.md -- PortfolioBacktestOrchestrator with 40/60 allocation, rebalancing, and USDRUB crisis brake
- [x] 12-02-PLAN.md -- Walk-forward Sharpe on merged curve and portfolio backtest CLI script

## Progress

**Execution Order:**
Phases execute in numeric order: 8 -> 8.1 -> ... -> 9 -> ... -> 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. MOEX Equity Foundation | v1.0 | 2/2 | Complete | 2026-03-14 |
| 2. MOEX Equity Validation | v1.0 | 3/3 | Complete | 2026-03-14 |
| 3. Bond Data Pipeline | v1.0 | 3/3 | Complete | 2026-03-14 |
| 4. Bond Execution | v1.0 | 3/3 | Complete | 2026-03-14 |
| 5. Integration and Telegram | v1.0 | 4/4 | Complete | 2026-03-14 |
| 6. Sandbox Validation | v1.0 | 4/4 | Complete | 2026-03-15 |
| 7. News Pipeline and Go-Live | v1.0 | 3/3 | Complete | 2026-03-15 |
| 8. Data Foundation | v2.0 | 3/3 | Complete | 2026-03-20 |
| 9. Strategy Wiring | v2.0 | 2/2 | Complete | 2026-03-20 |
| 10. Macro Regime | v2.0 | 2/2 | Complete | 2026-03-20 |
| 11. Advanced Strategies and ML | v2.0 | 4/4 | Complete | 2026-03-21 |
| 12. Portfolio Assembly | v2.0 | 2/2 | Complete | 2026-03-21 |
