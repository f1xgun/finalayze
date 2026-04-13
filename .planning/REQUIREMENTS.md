# Requirements: Finalayze

**Defined:** 2026-04-13
**Core Value:** Autonomous profitable MOEX trading with acceptable risk limits

## v9.0 Requirements

Requirements for ML AutoResearch & MOEX Adaptation milestone.

### MOEX Data Integration

- [x] **MOEX-01**: `auto_ml_research.py` fetches MOEX candles via TinkoffFetcher for ru_blue_chips, ru_energy, ru_finance, ru_tech segments — yfinance not used for any ru_* segment
- [x] **MOEX-02**: MOEX segment symbols defined in `_SEGMENT_SYMBOLS` matching production `config/segments.py` universe
- [x] **MOEX-03**: MOEX macro features (CBR rate, USDRUB, IMOEX, Brent) passed via `MoexMarketData` to `build_full_dataset()` — all 10 macro features non-zero in MOEX experiments

### Quality Gate Adaptation

- [x] **GATE-01**: `evaluate_fold()` accepts `min_signals` parameter — MOEX experiments use n_eff-scaled threshold instead of hardcoded 50
- [ ] **GATE-02**: MOEX-specific walk-forward fold constants produce 3+ folds on 730-day dataset — no single-fold trivial pass
- [x] **GATE-03**: Degenerate predictor guard rejects all-BUY/all-SELL models (buy_ratio outside 0.15–0.85 range fails gate)

### Experiment Infrastructure

- [ ] **EXPINT-01**: `--experiment-id` flag creates ExperimentManager entry at loop start, links results per experiment, records ACCEPT/REJECT/INCONCLUSIVE verdict at end
- [ ] **EXPINT-02**: JSONL log preserved as audit trail alongside ExperimentManager integration — backward compatible when `--experiment-id` not provided

### New Search Strategies

- [ ] **STRAT-01**: Ensemble weight optimization strategy searches bounded weight grid for XGB/LGBM/CatBoost — weights sum to 1.0, no single model >0.7
- [ ] **STRAT-02**: Cross-segment transfer strategy reads best US experiment features and filters to market-neutral intersection (excludes VIX-only and MOEX-only features)
- [ ] **STRAT-03**: Feature engineering strategy generates domain-motivated combinations (lag ratios, rolling z-scores, cross-feature interactions) with hard cap on generated feature count to prevent overfitting

## Future Requirements

### ML Pipeline Enhancement

- **MLPIPE-F01**: Auto-apply best experiment results to production model via PresetApplicator
- **MLPIPE-F02**: Scheduled auto_ml_research runs (cron/APScheduler)
- **MLPIPE-F03**: Dashboard page for autoresearch results visualization

## Out of Scope

| Feature | Reason |
|---------|--------|
| Neural architecture search (NAS) | Overkill for tree-based ensemble; complexity not justified |
| AutoML frameworks (AutoGluon, H2O) | Would replace existing pipeline rather than enhance it |
| Real-time model retraining | System operates on daily bars; batch retraining sufficient |
| GPU-accelerated training | Dataset size too small to benefit; CPU training completes in seconds |
| Multi-objective optimization (NSGA-II) | Single composite score sufficient for current experiment loop |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MOEX-01 | Phase 40 | Complete |
| MOEX-02 | Phase 40 | Complete |
| MOEX-03 | Phase 40 | Complete |
| GATE-01 | Phase 41 | Complete |
| GATE-02 | Phase 41 | Pending |
| GATE-03 | Phase 41 | Complete |
| EXPINT-01 | Phase 42 | Pending |
| EXPINT-02 | Phase 42 | Pending |
| STRAT-01 | Phase 43 | Pending |
| STRAT-02 | Phase 44 | Pending |
| STRAT-03 | Phase 44 | Pending |

**Coverage:**
- v9.0 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-13*
*Last updated: 2026-04-13 after roadmap creation*
