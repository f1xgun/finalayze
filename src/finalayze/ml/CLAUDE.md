# ML

## Purpose
Machine learning pipeline: feature engineering (45 features), model training (XGBoost, LightGBM, CatBoost, LSTM), ensemble prediction, probability calibration, meta-labeling, and model persistence.

## Layer
Layer 3 -- Analysis / ML. Can import from layers 0-2. Never import from layers 4-6.

## Key Files
- `models/ensemble.py` -- EnsembleModel: weighted average of sub-model probabilities with optional stacking meta-learner and conformal calibrator
- `models/base.py` -- BaseMLModel ABC for individual models
- `models/xgboost_model.py`, `models/lightgbm_model.py`, `models/catboost_model.py` -- Tree-based model wrappers
- `models/lstm_model.py` -- PyTorch LSTM model
- `models/stacking.py` -- StackingEnsemble (LogisticRegression meta-learner)
- `calibration.py` -- EnsembleCalibrator (Platt scaling + isotonic fallback), ConformalCalibrator (prediction sets with coverage guarantees)
- `features/technical.py` -- 45 technical features (RSI, MACD, Bollinger, cross-asset correlations, regime, calendar, z-scores)
- `features/corporate_actions.py` -- Dividend and corporate action features
- `features/multi_timeframe.py` -- Weekly/monthly context features
- `meta_labeler.py` -- MetaLabeler: P(profitable) gating for position sizing
- `loader.py` -- Model persistence: load/save EnsembleModel per segment from `models/<segment>/`
- `registry.py` -- MLModelRegistry: segment -> EnsembleModel lookup
- `integrity.py` -- HMAC verification for serialized model files
- `staleness.py` -- Model staleness detection
- `training/labeling.py` -- Binary labels (market-neutral via benchmark alignment)
- `training/sample_weights.py` -- Sequential bootstrapping (Lopez de Prado)
- `training/trend_scanning.py` -- Trend-scanning labels
- `training/feature_selection.py` -- Feature importance budget and selection pipeline
- `training/quality_gates.py` -- Brier score, accuracy, calibration quality gates
- `training/splitter.py` -- Purged K-fold and walk-forward splitting
- `training/cpcv.py` -- Combinatorial purged cross-validation

## Public API
- `EnsembleModel` -- `predict_proba(features, symbol) -> float` (BUY probability)
- `MLModelRegistry` -- `get(segment_id) -> EnsembleModel | None`
- `load_registry(model_dir, segments)` -- load all segment models
- `MetaLabeler` -- P(profitable) for position sizing pipeline

## Contracts
- Input: `dict[str, float]` feature vectors, `list[Candle]` for feature computation
- Output: BUY probability in [0.0, 1.0]. Returns 0.5 when no models are trained.
- Invariants: `FEATURE_SCHEMA_VERSION` must match between saved models and current code (v2). Models with mismatched versions are rejected at load time. ml_ensemble is reinforcer-only (can boost signals but never create standalone trades). Only enabled for us_tech segment.

## Testing
- Test location: `tests/unit/test_ml_*.py`, `tests/unit/test_ensemble*.py`, `tests/unit/test_calibration.py`
- Run: `uv run pytest tests/unit/ -k ml -v`

## Common Patterns
- Models stored as pickle files in `models/<segment>/` (xgb.pkl, lgbm.pkl, catboost.pkl, calibrator.pkl, meta_learner.pkl)
- `selected_features.json` controls which features are used at inference time
- Training uses `scripts/train_models.py --segment X --walk-forward --excess-returns --sequential-bootstrap`
- Conformal calibrator produces prediction sets: singleton {1} = BUY, singleton {0} = SELL, {0,1} = abstain
