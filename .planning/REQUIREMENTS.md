# Requirements: Finalayze v2.0 MOEX Profitability

**Defined:** 2026-03-20
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v2.0 Requirements

Requirements for MOEX profitability milestone. Each maps to roadmap phases.

### Data Foundation

- [ ] **DATA-01**: Vol target recalibrated for MOEX segments (0.35-0.45 instead of US-calibrated 0.19)
- [ ] **DATA-02**: Toxic symbols removed from universe (GAZP, VTBR, SNGS, IRAO, ALRS), confidence thresholds raised to 0.38+
- [ ] **DATA-03**: Dividend calendar expanded to 150+ events including cancelled/reduced dividends via T-Invest API
- [x] **DATA-04**: Feb-Mar 2022 structural break excluded from vol/ATR calculations, separate regime classification

### Strategy Wiring

- [x] **STRAT-01**: DividendGapStrategy calendar populated from expanded YAML, `_EVENT_STRATEGIES` bypass added to combiner ADX routing
- [x] **STRAT-02**: CBRStrategyWrapper wired into combiner for trading around CBR rate decisions
- [x] **STRAT-03**: rub_oil_regime.py integrated into position sizing pipeline as RubOilRegimeStep
- [x] **STRAT-04**: BrentGateStep added to sizing pipeline — gates energy sector positions when Brent below threshold

### Macro Regime

- [x] **MACRO-01**: CBRRegimeStep in sizing pipeline — CBR rate level + direction affects equity allocation sizing
- [ ] **MACRO-02**: OFZ PK-to-PD rotation trigger — detects CBR cutting cycle start for bond allocation shift
- [x] **MACRO-03**: SectorAllocationStep in sizing pipeline for sector rotation using MOEX sector indices (MOEXOG, MOEXFN, etc.)

### Advanced Strategies

- [x] **ADV-01**: Preferred share arbitrage (SBER/SBERP, TATN/TATNP) via adapted PairsStrategy with Kalman filter
- [x] **ADV-02**: 10 Russian macro ML features (CBR rate/delta/direction, USDRUB return/zscore/vol, Brent return, IMOEX relative, turnover zscore)
- [x] **ADV-03**: ML ensemble enabled for ru_* segments with macro features, reinforcer-only mode

### Portfolio Assembly

- [x] **PORT-01**: PortfolioBacktestOrchestrator for joint equity + OFZ backtest with merged equity curve
- [x] **PORT-02**: Portfolio allocation 40% OFZ carry + 60% equity with RUB crisis brake (USD/RUB +15% over 20 bars -> freeze equity)
- [x] **PORT-03**: Blended MOEX portfolio walk-forward Sharpe >= +0.10 (combined equity + OFZ)

## Future Requirements

Deferred to v3.0+.

### Expansion

- **EXP-01**: Multi-account support (multiple Tinkoff portfolios)
- **EXP-02**: Tax optimization (NDFL calculation, IIS deductions)
- **EXP-03**: Cross-market correlations (MOEX vs US for hedging)
- **EXP-04**: OFZ yield curve bootstrapping from CBR zero-coupon curve

## Out of Scope

| Feature | Reason |
|---------|--------|
| US market development | MOEX-only focus for v2.0 |
| Derivatives/futures | Complexity too high for current stage |
| Intraday trading (< daily bars) | System architecture is daily/swing |
| New pip dependencies | Research confirmed zero needed |
| Full sector rotation optimizer | Simple allocation step sufficient; optimizer is overfit risk |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 13 (gap closure) | Pending |
| DATA-02 | Phase 13 (gap closure) | Pending |
| DATA-03 | Phase 13 (gap closure) | Pending |
| DATA-04 | Phase 8 | Complete |
| STRAT-01 | Phase 9 | Complete |
| STRAT-02 | Phase 9 | Complete |
| STRAT-03 | Phase 9 | Complete |
| STRAT-04 | Phase 9 | Complete |
| MACRO-01 | Phase 10 | Complete |
| MACRO-02 | Phase 14 (gap closure) | Pending |
| MACRO-03 | Phase 10 | Complete |
| ADV-01 | Phase 11 | Complete |
| ADV-02 | Phase 11 | Complete |
| ADV-03 | Phase 11 | Complete |
| PORT-01 | Phase 12 | Complete |
| PORT-02 | Phase 12 | Complete |
| PORT-03 | Phase 12 | Complete |

**Coverage:**
- v2.0 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after roadmap creation*
