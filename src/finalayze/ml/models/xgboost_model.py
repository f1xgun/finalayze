"""XGBoost per-segment model (Layer 3).

Returns raw (uncalibrated) probabilities. Calibration is applied at the
ensemble level by ``EnsembleCalibrator`` (see ``calibration.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class XGBoostModel(BaseMLModel):
    """XGBoost classifier for directional prediction per segment.

    Returns raw probabilities; calibration is handled at the ensemble level.
    """

    def __init__(
        self,
        segment_id: str,
        max_depth: int = 5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ) -> None:
        self.segment_id = segment_id
        self._max_depth = max_depth
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._min_child_weight = min_child_weight
        self._gamma = gamma
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return raw BUY probability (0.0-1.0). Returns 0.5 when untrained."""
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
        return float(self._model.predict_proba(features_arr)[0][1])

    def fit(
        self,
        X: list[dict[str, float]],  # noqa: N803
        y: list[int],
        *,
        sample_weight: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Train the model on feature dicts and binary labels.

        Trains on the full dataset. Calibration is handled at the ensemble
        level by ``EnsembleCalibrator``.

        Args:
            X: Feature dictionaries.
            y: Binary labels (1=BUY, 0=SELL/HOLD).
            sample_weight: Optional per-sample weights (e.g. from uniqueness
                weighting).
        """
        if X:
            self._feature_names = sorted(X[0])
        x_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            scale_pos_weight=spw,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_weight=self._min_child_weight,
            gamma=self._gamma,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(x_arr, y_arr, sample_weight=sample_weight)

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
