"""XGBoost per-segment model (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5
_CALIBRATION_HOLDOUT_FRACTION = 0.2
_MIN_CALIBRATION_SAMPLES = 10


class XGBoostModel(BaseMLModel):
    """XGBoost classifier for directional prediction per segment.

    After fitting, an isotonic regression calibrator is trained on a holdout
    set so that ``predict_proba`` returns well-calibrated probabilities.
    """

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: xgb.XGBClassifier | None = None
        self._calibrator: IsotonicRegression | None = None
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
            calibrated = float(self._calibrator.predict([raw_proba])[0])
            return max(0.0, min(1.0, calibrated))
        return raw_proba

    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train the model on feature dicts and binary labels.

        Splits the last 20% of data as a calibration holdout, fits XGBoost on
        the training portion, then fits an isotonic regression calibrator on
        the holdout's raw probabilities.
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

        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(x_train, y_train)

        # Fit isotonic calibrator on holdout if enough samples with both classes
        if len(x_cal) >= _MIN_CALIBRATION_SAMPLES and len(np.unique(y_cal)) > 1:
            raw_probas = self._model.predict_proba(x_cal)[:, 1]
            self._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._calibrator.fit(raw_probas, y_cal)
        else:
            self._calibrator = None

    def save(self, path: Path) -> None:
        """Persist model to disk using joblib."""
        import joblib  # noqa: PLC0415, import-untyped

        joblib.dump(self, path)

    @classmethod
    def load_from(cls, path: Path) -> XGBoostModel:
        """Load a previously saved XGBoostModel.

        If an HMAC key is configured, verifies file integrity before loading.
        """
        import joblib  # noqa: PLC0415, import-untyped

        from finalayze.ml.loader import _get_hmac_key  # noqa: PLC0415

        key = _get_hmac_key()
        if key:
            from finalayze.ml.integrity import verify_model  # noqa: PLC0415

            verify_model(path, key.encode())

        return joblib.load(path)  # type: ignore[no-any-return]
