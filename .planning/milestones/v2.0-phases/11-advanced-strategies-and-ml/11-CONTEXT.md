# Phase 11: Advanced Strategies and ML - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable preferred share arbitrage on MOEX (SBER/SBERP, TATN/TATNP pairs), add 10 Russian macro ML features, and enable ML ensemble for at least one ru_* segment in reinforcer-only mode with quality gates passing on 2024-2025 calm-period validation data.

</domain>

<decisions>
## Implementation Decisions

### Preferred Share Arbitrage
- Cointegration validation period: post-2022 data only (2023-2025) — avoids sanction-era structural break
- Z-score thresholds: z_entry=2.0, z_exit=0.5 — matches success criteria (z>2.0 entry), 0.5 for profit capture
- Long-only constraint: set `allow_short=False` on PairsStrategy for MOEX pairs — success criteria says "long-only trades"
- Target preset: ru_blue_chips (both SBER and TATN are blue chips)
- Uses existing PairsStrategy with Kalman filter support already implemented

### Russian Macro ML Features (10 features)
- Feature list: CBR rate level, CBR rate delta, CBR direction (one-hot: cut/hike binary, hold=baseline), USDRUB return, USDRUB zscore, USDRUB vol, Brent return, IMOEX relative strength, turnover zscore, realized_vol_ratio
- Computation: extend existing `_compute_moex_features()` in technical.py — already has 7 MOEX features, add remaining 3 (CBR rate level, CBR delta, CBR direction one-hot)
- CBR encoding: one-hot binary columns (`cbr_direction_cut`, `cbr_direction_hike`), "hold" is baseline (0, 0)
- Look-ahead safety: all macro features lagged by 2 bars (matching existing MOEX features)

### ML Ensemble for ru_* Segments
- Training data period: 2023-01 to 2025-12 (post-sanctions calm period)
- Target segment for initial enablement: ru_blue_chips (most liquid, highest data quality)
- Feature selection: 10 features (fewer than US's 15 — smaller MOEX dataset, avoid overfitting)
- Quality gates: same dynamic gates as US with 2024-2025 validation window
- Mode: reinforcer-only (weight=0.10, same as us_tech enablement pattern)
- Training command pattern: `scripts/train_models.py --segment ru_blue_chips --walk-forward --excess-returns --sequential-bootstrap --force-save`

### Claude's Discretion
- TATNP FIGI registration (needs lookup or placeholder)
- Feature schema version bump (v2 → v3 for new features)
- Model hyperparameter tuning for MOEX (tree depth, learning rate)
- Training script MOEX-specific flags

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PairsStrategy` in `strategies/pairs.py` — full cointegration + Kalman filter, z-score entry/exit
- `EnsembleModel` in `ml/models/ensemble.py` — XGB+LGBM+CatBoost+meta-learner
- `_compute_moex_features()` in `ml/features/technical.py` — 7 existing MOEX features
- `train_models.py` — walk-forward training with sequential bootstrap
- `quality_gates.py` — 6 dynamic AFML gates
- `MLStrategy` in `strategies/ml_strategy.py` — reinforcer-only wrapper
- `EnsembleCalibrator` in `ml/calibration.py` — Platt + conformal calibration

### Established Patterns
- Pairs config in YAML: `pairs.params.pairs: [[SBER, SBERP]]`, `pairs.params.z_entry`, etc.
- ML enablement via preset: `ml_ensemble.enabled: true`, `ml_ensemble.weight: 0.10`
- Feature schema versioning: `FEATURE_SCHEMA_VERSION` constant in technical.py
- Model persistence in `models/<segment>/` directory

### Integration Points
- `strategies/presets/ru_blue_chips.yaml` — add pairs config and ml_ensemble enablement
- `ml/features/technical.py` — extend `_compute_moex_features()` with 3 CBR features
- `scripts/train_models.py` — may need MOEX-specific training flags
- `markets/instruments.py` — TATNP FIGI registration

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard wiring using established patterns from us_tech ML enablement.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
