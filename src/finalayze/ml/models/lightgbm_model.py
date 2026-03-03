"""LightGBM per-segment model (Layer 3).

Returns raw (uncalibrated) probabilities. Calibration is applied at the
ensemble level by ``EnsembleCalibrator`` (see ``calibration.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class LightGBMModel(BaseMLModel):
    """LightGBM classifier for directional prediction per segment.

    Returns raw probabilities; calibration is handled at the ensemble level.
    """

    def __init__(self, segment_id: str) -> None:
        self.segment_id = segment_id
        self._model: lgb.LGBMClassifier | None = None
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
        self._model.fit(x_arr, y_arr, sample_weight=sample_weight)

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
