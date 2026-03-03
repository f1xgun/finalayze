"""Meta-labeling: predict P(signal profitable) using XGBoost (Layer 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from finalayze.core.exceptions import InsufficientDataError

_UNTRAINED_PROB = 0.5
_MIN_SAMPLES = 30


@dataclass
class MetaSample:
    """Single sample for meta-labeling."""

    features: dict[str, float]
    signal_direction: float
    strategy_name: str
    confidence: float
    profitable: bool | None


class MetaLabeler:
    """XGBoost meta-labeler: predicts P(signal is profitable).

    Builds a feature vector from technical features, signal direction,
    confidence, and one-hot encoded strategy name.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self._strategy_vocab: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the meta-labeler has been trained."""
        return self._model is not None

    def fit(self, samples: list[MetaSample]) -> None:
        """Train XGBoost classifier on labeled MetaSamples.

        Raises ``InsufficientDataError`` if fewer than 30 samples are provided.
        """
        if len(samples) < _MIN_SAMPLES:
            msg = f"MetaLabeler requires at least {_MIN_SAMPLES} samples, got {len(samples)}"
            raise InsufficientDataError(msg)

        # Build strategy vocabulary (sorted for determinism)
        self._strategy_vocab = sorted({s.strategy_name for s in samples})

        # Build feature names: sorted feature keys + signal_direction + confidence + one-hot
        feature_keys = sorted(samples[0].features)
        self._feature_names = (
            feature_keys
            + ["_confidence", "_signal_direction"]
            + [f"_strategy_{name}" for name in self._strategy_vocab]
        )

        x_arr = np.array([self._sample_to_vector(s) for s in samples], dtype=float)
        y_arr = np.array([1 if s.profitable else 0 for s in samples], dtype=np.intp)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=spw,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(x_arr, y_arr)

    def predict_proba(self, sample: MetaSample) -> float:
        """Return probability that the signal is profitable (0.0-1.0).

        Returns 0.5 if the model has not been fitted yet.
        """
        if self._model is None:
            return _UNTRAINED_PROB

        vec = np.array([self._sample_to_vector(sample)], dtype=float)
        return float(self._model.predict_proba(vec)[0][1])

    def _sample_to_vector(self, sample: MetaSample) -> list[float]:
        """Convert a MetaSample into a flat feature vector."""
        feature_keys = sorted(sample.features)
        vec: list[float] = [sample.features[k] for k in feature_keys]

        # Append signal metadata
        vec.append(sample.confidence)
        vec.append(sample.signal_direction)

        # One-hot encode strategy name
        if self._strategy_vocab is not None:
            vec.extend(
                1.0 if sample.strategy_name == name else 0.0 for name in self._strategy_vocab
            )

        return vec
