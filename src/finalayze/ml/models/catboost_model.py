"""CatBoost per-segment model (Layer 3).

Returns raw (uncalibrated) probabilities. Calibration is applied at the
ensemble level by ``EnsembleCalibrator`` (see ``calibration.py``).

CatBoost's ordered boosting is specifically designed for small datasets,
making it a better fit than LSTM for financial data with ~3500 samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.models.base import BaseMLModel

_UNTRAINED_PROB = 0.5


class CatBoostModel(BaseMLModel):
    """CatBoost classifier with ordered boosting for small financial datasets.

    Returns raw probabilities; calibration is handled at the ensemble level.
    """

    def __init__(
        self,
        segment_id: str,
        iterations: int = 300,
        depth: int = 4,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 5.0,
        random_strength: float = 2.0,
        bagging_temperature: float = 1.0,
    ) -> None:
        self.segment_id = segment_id
        self._iterations = iterations
        self._depth = depth
        self._learning_rate = learning_rate
        self._l2_leaf_reg = l2_leaf_reg
        self._random_strength = random_strength
        self._bagging_temperature = bagging_temperature
        self._model: object | None = None  # CatBoostClassifier (lazy import)
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
        proba = self._model.predict_proba(features_arr)  # type: ignore[union-attr]
        return float(proba[0][1])

    def fit(
        self,
        X: list[dict[str, float]],  # noqa: N803
        y: list[int],
        *,
        sample_weight: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Train the model on feature dicts and binary labels.

        Uses ordered boosting and early stopping on a temporal validation split
        (last 10% of data).

        Args:
            X: Feature dictionaries.
            y: Binary labels (1=BUY, 0=SELL/HOLD).
            sample_weight: Optional per-sample weights (e.g. from uniqueness
                weighting).
        """
        from catboost import CatBoostClassifier  # noqa: PLC0415

        if X:
            self._feature_names = sorted(X[0])
        x_arr = np.array([[row[k] for k in sorted(row)] for row in X], dtype=float)
        y_arr = np.array(y, dtype=int)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        # Temporal validation split: last 10% for early stopping monitoring
        n_val = max(int(len(x_arr) * 0.1), 1)
        x_train, x_val = x_arr[:-n_val], x_arr[-n_val:]
        y_train, y_val = y_arr[:-n_val], y_arr[-n_val:]
        sw_train = sample_weight[:-n_val] if sample_weight is not None else None

        self._model = CatBoostClassifier(
            iterations=self._iterations,
            depth=self._depth,
            learning_rate=self._learning_rate,
            l2_leaf_reg=self._l2_leaf_reg,
            random_strength=self._random_strength,
            bagging_temperature=self._bagging_temperature,
            scale_pos_weight=spw,
            eval_metric="Logloss",
            early_stopping_rounds=25,
            verbose=0,
            random_seed=42,
        )
        self._model.fit(  # type: ignore[union-attr]
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            sample_weight=sw_train,
        )

    def save(self, path: Path) -> None:
        """Persist model to disk using joblib (atomic write)."""
        from finalayze.ml.loader import _atomic_save  # noqa: PLC0415

        _atomic_save(self, path)

    @classmethod
    def load_from(cls, path: Path) -> CatBoostModel:
        """Load a previously saved CatBoostModel.

        If an HMAC key is configured, verifies file integrity before loading.
        """
        import joblib  # noqa: PLC0415, import-untyped

        from finalayze.ml.loader import _get_hmac_key  # noqa: PLC0415

        key = _get_hmac_key()
        if key:
            from finalayze.ml.integrity import verify_model  # noqa: PLC0415

            verify_model(path, key.encode())

        return joblib.load(path)  # type: ignore[no-any-return]
