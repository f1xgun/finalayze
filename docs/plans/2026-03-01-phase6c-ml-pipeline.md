# Phase 6C: ML Pipeline Fixes

**Date:** 2026-03-01
**Status:** NOT STARTED
**Owner:** ml-agent
**Scope:** `src/finalayze/ml/`, `src/finalayze/core/trading_loop.py`, `scripts/train_models.py`

---

## Overview

Nine targeted improvements to the ML pipeline addressing feature diversity,
normalization, regularization, error handling, LSTM training stability,
validation gating, corporate action handling, and atomic persistence.

## Execution Order

Tasks are grouped into three waves based on dependencies. Within each wave,
tasks can be executed in parallel.

```
Wave 1 (no dependencies):
  6C.1  Feature diversity          (M)  -- technical.py
  6C.2  ATR/MACD normalization     (S)  -- technical.py
  6C.3  Tree regularization        (S)  -- xgboost_model.py, lightgbm_model.py
  6C.5  LSTM early stopping + clip (M)  -- lstm_model.py
  6C.6  LSTM dropout + decay       (S)  -- lstm_model.py
  6C.9  LSTM atomic save           (S)  -- loader.py

Wave 2 (depends on Wave 1):
  6C.4  Ensemble exception guard   (S)  -- ensemble.py  (depends on 6C.1 for new feature names)

Wave 3 (depends on Wave 1 + 2):
  6C.7  Validation gate metrics    (M)  -- trading_loop.py, train_models.py
  6C.8  Corporate action handling  (M)  -- technical.py or new file, training/__init__.py
```

---

## Task Details

### 6C.1 -- Feature Diversity (M)

**Problem:** Only 6 features (RSI, MACD histogram, Bollinger %B, volume ratio,
ATR, sentiment). Insufficient for tree models to find diverse splits.

**Files to modify:**
- `src/finalayze/ml/features/technical.py` (lines 18-90)

**Changes:**
1. Add the following new features after the existing ATR block (after line 73):
   - **ROC(10):** `ta.roc(close_s, length=10)` -- rate of change
   - **Williams %R(14):** `ta.willr(high_s, low_s, close_s, length=14)`
   - **ADX(14):** `ta.adx(high_s, low_s, close_s, length=14)` -- extract the ADX column
   - **MA slope (20-bar SMA):** compute `sma_20 = ta.sma(close_s, length=20)`, then
     `ma_slope = (sma_20 - sma_20.shift(1)) / close_s` (normalized by price)
   - **Historical volatility (20):** `ta.stdev(close_s.pct_change(), length=20)`
   - **Garman-Klass volatility (20):** manual formula using high/low/open/close:
     `0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2`, rolled over 20 bars
   - **Day-of-week:** extract from candle timestamp; encode as `sin(2*pi*dow/5)` and
     `cos(2*pi*dow/5)` (two features, cyclical encoding)
   - **OBV slope (10):** `ta.obv(close_s, volume_s)` then slope over last 10 bars,
     normalized by volume mean
   - **RSI divergence:** difference between price ROC and RSI ROC over 14 bars
2. Update `_MIN_CANDLES` from 30 to 30 (unchanged -- 30 bars still covers all
   indicator warmup periods since max lookback is 26 for MACD slow).
3. Add all new features to the `feature_df` DataFrame (lines 78-87).
4. Update `feature_df` column list to include all ~16 features.

**Tests to write** (`tests/unit/test_ml_features.py` -- new file):
- `test_compute_features_returns_all_expected_keys`: assert >= 16 keys returned
- `test_compute_features_no_nans`: all values are finite floats
- `test_compute_features_day_of_week_cyclical`: sin/cos values in [-1, 1]
- `test_compute_features_garman_klass_non_negative`: GK vol >= 0
- `test_compute_features_minimum_candles_unchanged`: 30 candles still works

**Dependencies:** None.

---

### 6C.2 -- ATR/MACD Normalization (S)

**Problem:** ATR and MACD histogram are in absolute price units. A $200 stock
produces ATR ~5 while a $20 stock produces ATR ~0.5, making cross-asset
comparison meaningless for ML.

**Files to modify:**
- `src/finalayze/ml/features/technical.py` (lines 72-73, 51-55, 78-87)

**Changes:**
1. After computing `atr_val` (line 73), normalize:
   ```python
   last_close = closes[-1]
   atr_pct = atr_val / last_close if last_close > 0 else 0.0
   ```
2. After computing `macd_hist` (line 55), normalize:
   ```python
   macd_hist_pct = macd_hist / last_close if last_close > 0 else 0.0
   ```
3. Replace `"atr_14"` with `"atr_14_pct"` and `"macd_hist"` with
   `"macd_hist_pct"` in the feature dict (lines 80-81).
4. Keep raw `atr_14` and `macd_hist` available only if needed for backward
   compat -- but since all models retrain, the rename is safe.

**IMPORTANT:** This is a breaking change for feature names. All saved models
will need retraining. The feature name validation in XGBoostModel/LightGBMModel/
LSTMModel will reject old feature sets automatically.

**Tests to write** (`tests/unit/test_ml_features.py`):
- `test_atr_pct_scales_with_price`: two candle sets at different price levels
  should produce similar `atr_14_pct` values
- `test_macd_hist_pct_scales_with_price`: same logic for MACD
- `test_old_feature_names_absent`: assert `"atr_14"` and `"macd_hist"` are
  NOT in the returned dict

**Dependencies:** Should be implemented alongside 6C.1 to avoid two rounds of
feature name changes.

---

### 6C.3 -- Tree Model Regularization (S)

**Problem:** XGBoost and LightGBM use no explicit regularization beyond
`max_depth=4`. Risk of overfitting on small per-segment datasets.

**Files to modify:**
- `src/finalayze/ml/models/xgboost_model.py` (lines 74-80)
- `src/finalayze/ml/models/lightgbm_model.py` (lines 74-76)

**Changes in `xgboost_model.py`:**
Replace the `XGBClassifier` constructor (lines 74-80) with:
```python
self._model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    subsample=0.8,       # row subsampling
    colsample_bytree=0.8,  # column subsampling
    eval_metric="logloss",
    verbosity=0,
)
```

**Changes in `lightgbm_model.py`:**
Replace the `LGBMClassifier` constructor (lines 74-76) with:
```python
self._model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    subsample=0.8,       # row subsampling
    colsample_bytree=0.8,  # column subsampling
    verbosity=-1,
)
```

**Tests to write** (`tests/unit/test_ml_models.py` or new `tests/unit/test_ml_regularization.py`):
- `test_xgboost_regularization_params`: after `fit()`, inspect
  `model._model.get_params()` and assert the 4 new params match
- `test_lightgbm_regularization_params`: same for LightGBM
- `test_xgboost_overfit_less_than_unregularized`: train on small noisy data,
  verify regularized model has lower variance across seeds (optional)

**Dependencies:** None.

---

### 6C.4 -- Ensemble predict_proba Exception Handling (S)

**Problem:** `EnsembleModel.predict_proba()` (lines 33-50 of `ensemble.py`)
calls each model's `predict_proba()` without try/except. If one model raises
(e.g., `InsufficientDataError` on feature mismatch), the entire ensemble fails
instead of gracefully degrading.

**Files to modify:**
- `src/finalayze/ml/models/ensemble.py` (lines 33-50)
- `src/finalayze/core/exceptions.py` (add `PredictionError`)

**Changes:**

1. Add `PredictionError` to `src/finalayze/core/exceptions.py`:
   ```python
   class PredictionError(FinalayzeError):
       """All ensemble sub-models failed to produce a prediction."""
   ```

2. Rewrite `predict_proba()` in `ensemble.py`:
   ```python
   import logging
   _log = logging.getLogger(__name__)

   def predict_proba(self, features: dict[str, float], *, symbol: str = "__default__") -> float:
       probs: list[float] = []

       for m in self._models:
           if getattr(m, "_model", None) is None:
               continue
           try:
               probs.append(m.predict_proba(features))
           except Exception:
               _log.warning("Ensemble: %s failed, skipping", type(m).__name__, exc_info=True)

       if self._lstm_model is not None and getattr(self._lstm_model, "_trained", False):
           try:
               probs.append(self._lstm_model.predict_proba(features, symbol=symbol))
           except Exception:
               _log.warning("Ensemble: LSTM failed, skipping", exc_info=True)

       if not probs:
           raise PredictionError("All ensemble sub-models failed to produce a prediction")
       return sum(probs) / len(probs)
   ```

**Note on behavior change:** Currently, when no models are trained, the method
returns 0.5 (`_DEFAULT_PROB`). After this change, if all models are trained but
ALL raise exceptions, a `PredictionError` is raised. If no models are trained
at all (all `_model is None`), the behavior should also raise `PredictionError`
since there are no predictions to average. Update calling code to handle this.
However -- the existing behavior of returning 0.5 for untrained models is
arguably safer for the trading loop. Decision: raise `PredictionError` only
when trained models all fail. Keep returning 0.5 when no models are trained
(check `if not probs and not any_trained`).

**Tests to write** (`tests/unit/test_ml_ensemble.py` -- new or extend existing):
- `test_ensemble_skips_failing_model`: one model raises, others succeed; average
  is computed from surviving models
- `test_ensemble_all_fail_raises_prediction_error`: all trained models raise;
  `PredictionError` is raised
- `test_ensemble_untrained_returns_default`: no trained models; returns 0.5
- `test_ensemble_partial_failure_logged`: verify log output contains warning

**Dependencies:** 6C.1 (feature name changes may trigger the mismatch path).

---

### 6C.5 -- LSTM Early Stopping + Gradient Clipping (M)

**Problem:** LSTM trains for a fixed 50 epochs with no early stopping. On small
datasets this leads to overfitting; on large datasets it may undertrain.
No gradient clipping risks exploding gradients on volatile financial data.

**Files to modify:**
- `src/finalayze/ml/models/lstm_model.py` (lines 155-175, constants at top)

**Changes:**

1. Add constants at the top of the file (after line 26):
   ```python
   _PATIENCE = 5           # early stopping patience (epochs without improvement)
   _MAX_GRAD_NORM = 1.0    # gradient clipping max norm
   ```

2. Rewrite the training loop (lines 169-175) to include:
   - Split training data further into train/val (e.g., last 10% of training
     sequences as validation for early stopping -- separate from the calibration
     holdout which is already split off).
   - Track best validation loss; stop if no improvement for `_PATIENCE` epochs.
   - Add `torch.nn.utils.clip_grad_norm_(self._model.parameters(), _MAX_GRAD_NORM)`
     after `loss.backward()` and before `optimizer.step()`.
   - Restore best model weights when early stopping triggers.

   Revised loop structure:
   ```python
   # Further split train into train_inner + val_inner for early stopping
   n_inner_val = max(int(n_train * 0.1), 1)
   n_inner_train = n_train - n_inner_val
   x_inner_train = x_train[:n_inner_train]
   y_inner_train = y_train[:n_inner_train]
   x_inner_val = x_train[n_inner_train:]
   y_inner_val = y_train[n_inner_train:]

   best_val_loss = float("inf")
   best_state = None
   patience_counter = 0

   self._model.train()
   for epoch in range(_TRAIN_EPOCHS):
       optimizer.zero_grad()
       output = self._model(x_inner_train)
       loss = criterion(output, y_inner_train)
       loss.backward()
       torch.nn.utils.clip_grad_norm_(self._model.parameters(), _MAX_GRAD_NORM)
       optimizer.step()

       # Validation loss for early stopping
       self._model.eval()
       with torch.no_grad():
           val_output = self._model(x_inner_val)
           val_loss = float(criterion(val_output, y_inner_val))
       self._model.train()

       if val_loss < best_val_loss:
           best_val_loss = val_loss
           best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
           patience_counter = 0
       else:
           patience_counter += 1
           if patience_counter >= _PATIENCE:
               break

   # Restore best weights
   if best_state is not None:
       self._model.load_state_dict(best_state)
   ```

**Tests to write** (`tests/unit/test_lstm_training.py` -- new file):
- `test_lstm_early_stopping_triggers`: provide data that converges quickly;
  assert training stops before `_TRAIN_EPOCHS` (check epoch count via mock or
  by observing that fit completes faster with early stopping)
- `test_lstm_gradient_clipping_applied`: mock `clip_grad_norm_` and verify it
  is called each epoch
- `test_lstm_best_weights_restored`: train, then verify the model state matches
  the epoch with best validation loss (not the last epoch)
- `test_lstm_fit_still_works_small_data`: 25 samples still trains without error

**Dependencies:** None. Can be combined with 6C.6.

---

### 6C.6 -- LSTM Dropout + Weight Decay (S)

**Problem:** No dropout between LSTM layers and no weight decay in Adam. Both
are standard regularization techniques for LSTMs on small financial datasets.

**Files to modify:**
- `src/finalayze/ml/models/lstm_model.py`
  - `_LSTMNet.__init__` (lines 32-41): add dropout
  - `LSTMModel.fit` (line 156): add weight_decay to Adam

**Changes:**

1. In `_LSTMNet.__init__` (line 32), add `dropout` parameter:
   ```python
   def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                dropout: float = 0.0) -> None:
       super().__init__()
       self._lstm = nn.LSTM(
           input_size,
           hidden_size,
           num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0.0,  # PyTorch requires num_layers > 1
       )
       self._dropout = nn.Dropout(dropout)  # applied after LSTM, before linear
       self._linear = nn.Linear(hidden_size, 1)
       self._sigmoid = nn.Sigmoid()
   ```

2. In `_LSTMNet.forward` (lines 43-48), add dropout before linear:
   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       lstm_out, _ = self._lstm(x)
       last_hidden = lstm_out[:, -1, :]
       dropped = self._dropout(last_hidden)
       result: torch.Tensor = self._sigmoid(self._linear(dropped))
       return result
   ```

3. Add constants:
   ```python
   _DROPOUT = 0.2
   _WEIGHT_DECAY = 1e-4
   ```

4. In `LSTMModel.fit` (line 155), pass dropout to `_LSTMNet`:
   ```python
   self._model = _LSTMNet(n_features, self._hidden_size, self._num_layers, dropout=_DROPOUT)
   ```

5. In `LSTMModel.fit` (line 156), add weight_decay:
   ```python
   optimizer = torch.optim.Adam(self._model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
   ```

**Tests to write** (`tests/unit/test_lstm_training.py`):
- `test_lstm_dropout_present`: after `fit()`, verify `self._model._dropout.p == 0.2`
- `test_lstm_weight_decay_effect`: train two models (one with decay, one
  without); verify weight magnitudes are smaller with decay
- `test_lstm_dropout_disabled_during_eval`: call `model.eval()`, verify dropout
  is bypassed (standard PyTorch behavior, but good to confirm)

**Dependencies:** None. Should be implemented together with 6C.5 since both
modify `lstm_model.py`.

---

### 6C.7 -- Validation Gate Metrics (M)

**Problem:** The retrain validation gate in `_retrain_segment()` (trading_loop.py
lines 715-727) uses only accuracy at a 52% threshold. This is a weak gate:
accuracy is insensitive to probability calibration and can pass poorly
calibrated models that make confident wrong predictions.

**Files to modify:**
- `src/finalayze/core/trading_loop.py` (lines 715-727)
- `scripts/train_models.py` (lines 192-213 -- add Brier/log-loss to output)
- `src/finalayze/ml/training/__init__.py` (add `validate_model()` utility)

**Changes:**

1. Add a `validate_model()` function in `src/finalayze/ml/training/__init__.py`:
   ```python
   from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
   from dataclasses import dataclass

   @dataclass
   class ValidationResult:
       accuracy: float
       brier_score: float
       log_loss_val: float
       n_samples: int
       passed: bool

   _MIN_ACCURACY = 0.52
   _MAX_BRIER_SCORE = 0.25     # perfect = 0.0, coin flip = 0.25
   _MAX_LOG_LOSS = 0.69        # coin flip = ln(2) ~ 0.693

   def validate_ensemble(
       ensemble: EnsembleModel,
       val_features: list[dict[str, float]],
       val_labels: list[int],
   ) -> ValidationResult:
       """Evaluate an ensemble on validation data and return metrics + pass/fail."""
       probas = [ensemble.predict_proba(f) for f in val_features]
       preds = [round(p) for p in probas]
       acc = accuracy_score(val_labels, preds)
       brier = brier_score_loss(val_labels, probas)
       ll = log_loss(val_labels, probas, labels=[0, 1])
       passed = acc >= _MIN_ACCURACY and brier <= _MAX_BRIER_SCORE and ll <= _MAX_LOG_LOSS
       return ValidationResult(
           accuracy=acc, brier_score=brier, log_loss_val=ll,
           n_samples=len(val_labels), passed=passed,
       )
   ```

2. In `trading_loop.py`, replace lines 715-727 with a call to `validate_ensemble()`:
   ```python
   from finalayze.ml.training import validate_ensemble

   result = validate_ensemble(ensemble, val_features, val_labels)
   if not result.passed:
       _log.warning(
           "_retrain: validation failed for %s — acc=%.3f brier=%.3f logloss=%.3f",
           segment_id, result.accuracy, result.brier_score, result.log_loss_val,
       )
       return
   ```

3. In `scripts/train_models.py`, add Brier score and log-loss to the printed
   summary alongside accuracy (lines 192-213).

**Tests to write** (`tests/unit/test_ml_validation.py` -- new file):
- `test_validate_ensemble_passes_good_model`: synthetic data where model
  achieves > 52% accuracy and Brier < 0.25
- `test_validate_ensemble_fails_bad_accuracy`: accuracy below 52% -> not passed
- `test_validate_ensemble_fails_bad_brier`: accuracy OK but Brier > 0.25
  (overconfident wrong predictions) -> not passed
- `test_validate_ensemble_fails_bad_logloss`: log-loss above threshold -> not passed
- `test_validation_result_dataclass_fields`: verify all fields present

**Dependencies:** 6C.4 (ensemble may raise `PredictionError` inside validation;
the `validate_ensemble` function should handle this).

---

### 6C.8 -- Corporate Action Handling (M)

**Problem:** No detection or handling of stock splits and dividends. A 2:1 split
creates a 50% price gap that poisons technical indicators (RSI spikes to 100,
ATR explodes, etc.).

**Files to modify:**
- `src/finalayze/ml/features/technical.py` (add validation)
- `src/finalayze/ml/training/__init__.py` (add split detection in `build_windows`)
- New file: `src/finalayze/ml/features/corporate_actions.py`

**Changes:**

1. Create `src/finalayze/ml/features/corporate_actions.py`:
   ```python
   """Corporate action detection for candle data (Layer 3)."""

   _SPLIT_THRESHOLD = 0.40  # 40% single-bar move likely indicates split/reverse-split

   def detect_splits(candles: list[Candle]) -> list[int]:
       """Return indices where a suspected split/reverse-split occurred.

       A split is detected when the close-to-close return exceeds the threshold
       AND the high-low range of the suspect bar is small relative to the gap
       (ruling out genuine crash/rally bars).
       """
       suspect_indices: list[int] = []
       for i in range(1, len(candles)):
           prev_close = float(candles[i - 1].close)
           if prev_close == 0:
               continue
           ret = abs(float(candles[i].close) - prev_close) / prev_close
           bar_range = abs(float(candles[i].high) - float(candles[i].low))
           gap = abs(float(candles[i].close) - prev_close)
           # Split: large gap but small intraday range (price just shifted)
           if ret > _SPLIT_THRESHOLD and bar_range < gap * 0.5:
               suspect_indices.append(i)
       return suspect_indices

   def adjust_for_splits(candles: list[Candle]) -> list[Candle]:
       """Return candles with suspected splits adjusted via ratio.

       Uses backward adjustment: multiply all bars before the split by
       the ratio new_close / old_close.
       """
       # Implementation: iterate splits in reverse, adjust all prior bars
       ...
   ```

2. In `build_windows()` (`training/__init__.py`), add an optional
   `skip_split_windows: bool = True` parameter. When a window spans a detected
   split index, skip that sample.

3. In `compute_features()` (`technical.py`), add an optional sanity check:
   if the max single-bar return in the candle window exceeds 40%, log a warning
   (do not raise -- the caller decides whether to skip).

4. Document that yfinance returns split-adjusted data by default (`auto_adjust=True`),
   so the primary risk is with database-stored raw candles from Alpaca/Tinkoff.

**Tests to write** (`tests/unit/test_corporate_actions.py` -- new file):
- `test_detect_splits_finds_2_for_1`: synthetic candles with a 2:1 split
- `test_detect_splits_ignores_normal_volatility`: 5% daily move not flagged
- `test_detect_splits_finds_reverse_split`: 1:2 reverse split detected
- `test_adjust_for_splits_corrects_prices`: adjusted candles have smooth prices
- `test_build_windows_skips_split_window`: window spanning a split is excluded

**Dependencies:** 6C.1 (new features make split artifacts more visible).

---

### 6C.9 -- LSTM Atomic Save in Loader (S)

**Problem:** In `loader.py` line 98, `ensemble._lstm_model.save(segment_dir / "lstm.pkl")`
calls `LSTMModel.save()` directly, which writes to the target path without
atomic temp+rename. If the process is interrupted mid-write, the LSTM checkpoint
is corrupted. XGB and LGBM models already use `_atomic_save()`.

**Files to modify:**
- `src/finalayze/ml/loader.py` (lines 97-98)
- `src/finalayze/ml/models/lstm_model.py` (lines 194-222 -- `save()` method)

**Changes:**

Option A (preferred): Make `LSTMModel.save()` itself atomic:
```python
def save(self, path: Path) -> None:
    if self._model is None:
        msg = "Cannot save an untrained LSTMModel"
        raise ValueError(msg)

    payload: dict[str, Any] = {
        "state_dict": self._model.state_dict(),
        "config": { ... },  # same as current
    }

    # Atomic save: write to temp file, then rename
    import tempfile
    import os

    # Main weights file
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    tmp_path = Path(tmp_str)
    try:
        os.close(fd)
        torch.save(payload, tmp_path)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Scaler file -- also atomic
    scaler_path = path.parent / (path.name + ".scaler.pkl")
    fd2, tmp_str2 = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="scaler")
    tmp_path2 = Path(tmp_str2)
    try:
        os.close(fd2)
        with tmp_path2.open("wb") as fh:
            pickle.dump(self._scaler, fh)
        tmp_path2.rename(scaler_path)
    except Exception:
        tmp_path2.unlink(missing_ok=True)
        raise

    # Platt scaler file -- also atomic
    platt_path = path.parent / (path.name + ".platt.pkl")
    fd3, tmp_str3 = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="platt")
    tmp_path3 = Path(tmp_str3)
    try:
        os.close(fd3)
        with tmp_path3.open("wb") as fh:
            pickle.dump(self._platt_scaler, fh)
        tmp_path3.rename(platt_path)
    except Exception:
        tmp_path3.unlink(missing_ok=True)
        raise
```

Option B: Change `loader.py` to wrap the LSTM save in the same `_atomic_save`
pattern. Less clean because LSTM writes 3 files.

**Recommendation:** Option A -- keep atomicity inside `LSTMModel.save()` since
it knows about its companion files.

**Tests to write** (`tests/unit/test_ml_loader.py` -- extend existing):
- `test_lstm_save_atomic_no_corrupt_on_interrupt`: mock `torch.save` to raise
  after writing; verify no partial file remains at the target path
- `test_lstm_save_creates_all_three_files`: after save, all 3 files exist
- `test_lstm_save_scaler_atomic`: mock pickle.dump to raise; verify no
  partial scaler file

**Dependencies:** None.

---

## File Summary Table

| File | Tasks | Type |
|------|-------|------|
| `src/finalayze/ml/features/technical.py` | 6C.1, 6C.2, 6C.8 | Modify |
| `src/finalayze/ml/features/corporate_actions.py` | 6C.8 | New |
| `src/finalayze/ml/models/xgboost_model.py` | 6C.3 | Modify |
| `src/finalayze/ml/models/lightgbm_model.py` | 6C.3 | Modify |
| `src/finalayze/ml/models/lstm_model.py` | 6C.5, 6C.6, 6C.9 | Modify |
| `src/finalayze/ml/models/ensemble.py` | 6C.4 | Modify |
| `src/finalayze/ml/loader.py` | 6C.9 | Modify (minor) |
| `src/finalayze/ml/training/__init__.py` | 6C.7, 6C.8 | Modify |
| `src/finalayze/core/exceptions.py` | 6C.4 | Modify |
| `src/finalayze/core/trading_loop.py` | 6C.7 | Modify |
| `scripts/train_models.py` | 6C.7 | Modify |
| `tests/unit/test_ml_features.py` | 6C.1, 6C.2 | New |
| `tests/unit/test_ml_ensemble.py` | 6C.4 | New |
| `tests/unit/test_lstm_training.py` | 6C.5, 6C.6 | New |
| `tests/unit/test_ml_validation.py` | 6C.7 | New |
| `tests/unit/test_corporate_actions.py` | 6C.8 | New |
| `tests/unit/test_ml_loader.py` | 6C.9 | Extend |

---

## Test Plan

All tests use synthetic data (50 rows, 5-16 features) for speed. No network
calls or real market data.

```bash
# Run all ML tests
uv run pytest tests/unit/test_ml_features.py tests/unit/test_ml_ensemble.py \
  tests/unit/test_lstm_training.py tests/unit/test_ml_validation.py \
  tests/unit/test_corporate_actions.py tests/unit/test_ml_loader.py -v

# Lint + type check
uv run ruff check src/finalayze/ml/ tests/unit/test_ml_*.py tests/unit/test_lstm_*.py tests/unit/test_corporate_*.py
uv run ruff format --check src/finalayze/ml/ tests/unit/
uv run mypy src/finalayze/ml/
```

**TDD flow per task:**
1. Write failing test
2. `uv run pytest <test_file> -v` -- confirm RED
3. Implement the change
4. `uv run pytest <test_file> -v` -- confirm GREEN
5. `uv run ruff check . && uv run mypy src/`
6. Commit: `git commit -m "feat(ml): <task description>"`

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Feature name changes break saved models | All models must be retrained after 6C.1+6C.2. Document in release notes. |
| Early stopping on tiny datasets (< 30 sequences) | Inner validation split minimum is 1 sample; fallback to full training if split too small. |
| Garman-Klass NaN on zero open price | Guard with `if open > 0` check, default to 0. |
| Corporate action detector false positives | Conservative threshold (40%); log warnings rather than silently dropping data. |
| `PredictionError` propagation to trading loop | Trading loop must catch `PredictionError` and skip the signal (same as current `InsufficientDataError` handling). |

---

## Estimated Effort

| Task | Complexity | Estimate |
|------|-----------|----------|
| 6C.1 | M | 2-3 hours |
| 6C.2 | S | 30 min |
| 6C.3 | S | 30 min |
| 6C.4 | S | 1 hour |
| 6C.5 | M | 2 hours |
| 6C.6 | S | 30 min |
| 6C.7 | M | 2 hours |
| 6C.8 | M | 2-3 hours |
| 6C.9 | S | 1 hour |
| **Total** | | **~12 hours** |
