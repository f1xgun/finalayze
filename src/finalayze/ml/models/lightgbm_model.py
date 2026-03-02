"""LightGBM per-segment model (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5
_CALIBRATION_HOLDOUT_FRACTION = 0.2
_MIN_CALIBRATION_SAMPLES = 50


class LightGBMModel(BaseMLModel):
    """LightGBM classifier for directional prediction per segment.

    After fitting, an isotonic regression calibrator is trained on a holdout
    set so that ``predict_proba`` returns well-calibrated probabilities.
    """

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: IsotonicRegression | LogisticRegression | None = None
        self._feature_names: list[str] | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return calibrated BUY probability (0.0-1.0). Returns 0.5 when untrained."""
        if self._model is None:
            return _UNTRAINED_PROB
        if self._feature_names is not None:
            incoming = sorted(features)
            if incoming != self._feature_names:
                msg = (
                    f"Feature mismatch for segment {self.segment_id!r}: "
                    f"expected {self._feature_names}, got {incoming}"
                )
                raise InsufficientDataError(msg)
        features_arr = np.array([[features[k] for k in sorted(features)]], dtype=float)
        raw_proba = float(self._model.predict_proba(features_arr)[0][1])
        if self._calibrator is not None:
            if isinstance(self._calibrator, LogisticRegression):
                calibrated = float(self._calibrator.predict_proba(np.array([[raw_proba]]))[0][1])
            else:
                calibrated = float(self._calibrator.predict([raw_proba])[0])
            return max(0.0, min(1.0, calibrated))
        return raw_proba

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the model on feature dicts and binary labels.

        Splits the last 20% of data as a calibration holdout, fits LightGBM on
        the training portion, then fits a calibrator on the holdout's raw
        probabilities.  Uses isotonic regression when there are at least
        ``_MIN_CALIBRATION_SAMPLES`` calibration samples, otherwise falls
        back to Platt scaling (logistic regression).
        """
        if X:
            self._feature_names = sorted(X[0])
        x_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)

        # Split: train on first 80%, calibrate on last 20%
        n_total = len(X)
        n_cal = max(int(n_total * _CALIBRATION_HOLDOUT_FRACTION), 1)
        n_train = n_total - n_cal

        x_train, x_cal = x_arr[:n_train], x_arr[n_train:]
        y_train, y_cal = y_arr[:n_train], y_arr[n_train:]

        self._model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            is_unbalance=True,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
        )
        self._model.fit(x_train, y_train)

        # Fit calibrator on holdout if both classes are present
        n_cal_actual = len(x_cal)
        if n_cal_actual > 0 and len(np.unique(y_cal)) > 1:
            raw_probas = self._model.predict_proba(x_cal)[:, 1]
            if n_cal_actual >= _MIN_CALIBRATION_SAMPLES:
                self._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                self._calibrator.fit(raw_probas, y_cal)
            else:
                # Platt scaling fallback for small calibration sets
                lr = LogisticRegression()
                lr.fit(raw_probas.reshape(-1, 1), y_cal)
                self._calibrator = lr
        else:
            self._calibrator = None

    def save(self, path: Path) -> None:
        """Persist model to disk using joblib."""
        import joblib  # noqa: PLC0415, import-untyped

        joblib.dump(self, path)

    @classmethod
    def load_from(cls, path: Path) -> LightGBMModel:
        """Load a previously saved LightGBMModel.

        If an HMAC key is configured, verifies file integrity before loading.
        """
        import joblib  # noqa: PLC0415, import-untyped

        from finalayze.ml.loader import _get_hmac_key  # noqa: PLC0415

        key = _get_hmac_key()
        if key:
            from finalayze.ml.integrity import verify_model  # noqa: PLC0415

            verify_model(path, key.encode())

        return joblib.load(path)  # type: ignore[no-any-return]
