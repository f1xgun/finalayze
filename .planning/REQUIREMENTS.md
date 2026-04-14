# Requirements: Finalayze v9.1 MOEX ML Model Quality

**Defined:** 2026-04-14
**Core Value:** Raise ML model quality on failing MOEX segments to pass quality gates

## v9.1 Requirements

### Model Complexity

- [ ] **MCPX-01**: Autoresearch uses reduced model complexity for MOEX segments (max_depth=3, n_estimators=100, min_child_weight=20)
- [ ] **MCPX-02**: MOEX-specific hyperparameter defaults are separate from US defaults in autoresearch config

### Cross-Asset Features

- [ ] **FEAT-01**: Brent crude return features (ret_5d, ret_21d) available in technical feature set for MOEX segments
- [ ] **FEAT-02**: Brent features wired from existing `_fetch_moex_macro_data()` into feature engineering pipeline

### Ensemble Consistency

- [ ] **ENSM-01**: XGBoost sets scale_pos_weight=1.0 when sample_weight is provided (matching LightGBM behavior)
- [ ] **ENSM-02**: All 3 ensemble members (XGB, LGBM, CatBoost) use consistent class rebalancing strategy

### Feature Selection Stability

- [ ] **FSEL-01**: Feature selection runs once on full pre-test dataset, not per-fold, in autoresearch pipeline
- [ ] **FSEL-02**: Selected feature set is stable across walk-forward folds (same features used in all folds)

### Segment Restructuring

- [ ] **SEGM-01**: SBERP removed from ru_finance segment (rho > 0.95 with SBER adds noise without signal)
- [ ] **SEGM-02**: Minimum history check (500 trading days) gates ML eligibility per symbol in autoresearch
- [ ] **SEGM-03**: ru_tech segment has defined ML policy (disabled, merged, or min-history filtered)

### Asymmetric Barriers

- [ ] **BARR-01**: Energy stocks use asymmetric triple barrier (wider lower ATR multiplier for commodity-linked volatility)
- [ ] **BARR-02**: Barrier asymmetry configurable per segment in autoresearch

## Future Requirements

### Advanced Model Improvements

- **ADVML-01**: Per-segment feature engineering (sector-specific feature sets)
- **ADVML-02**: Calibration improvement via isotonic regression post-hoc on MOEX folds
- **ADVML-03**: Cross-segment transfer learning (US→MOEX feature transfer with domain adaptation)

## Out of Scope

| Feature | Reason |
|---------|--------|
| US market ML changes | MOEX-only focus per user directive |
| New model architectures (LSTM, Transformer) | Current tree ensembles sufficient, complexity not warranted |
| Live trading integration of ML | Separate milestone — v9.1 is autoresearch pipeline only |
| Quality gates relaxation beyond current fixes | Already adapted in v9.0+; further relaxation masks real model weakness |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MCPX-01 | — | Pending |
| MCPX-02 | — | Pending |
| FEAT-01 | — | Pending |
| FEAT-02 | — | Pending |
| ENSM-01 | — | Pending |
| ENSM-02 | — | Pending |
| FSEL-01 | — | Pending |
| FSEL-02 | — | Pending |
| SEGM-01 | — | Pending |
| SEGM-02 | — | Pending |
| SEGM-03 | — | Pending |
| BARR-01 | — | Pending |
| BARR-02 | — | Pending |

**Coverage:**
- v9.1 requirements: 13 total
- Mapped to phases: 0
- Unmapped: 13 ⚠️

---
*Requirements defined: 2026-04-14*
*Last updated: 2026-04-14 after initial definition*
