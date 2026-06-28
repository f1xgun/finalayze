"""Ensemble model combining XGBoost + LightGBM + optional LSTM (Layer 3)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import structlog

from finalayze.core.exceptions import InsufficientDataError, PredictionError

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from finalayze.ml.calibration import EnsembleCalibrator
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.lstm_model import LSTMModel
    from finalayze.ml.models.stacking import StackingEnsemble

_DEFAULT_PROB = 0.5
_BYPASS_CLAMP_LOWER = 0.30
_BYPASS_CLAMP_UPPER = 0.70
_CLAMP_RATE_WARNING_THRESHOLD = 0.50
_MIN_PREDICTIONS_FOR_CLAMP_WARNING = 10
_log = structlog.get_logger()

# Mapping from normalized class name to all known aliases used by training scripts.
# Walk-forward path saves "xgboost"/"lightgbm"/"catboost";
# single-split path saves "xgb"/"lgbm"/"catboost" (after lowercasing).
_KEY_MAP: dict[str, list[str]] = {
    "xgboostmodel": ["xgboostmodel", "xgboost", "xgb"],
    "lightgbmmodel": ["lightgbmmodel", "lightgbm", "lgbm"],
    "catboostmodel": ["catboostmodel", "catboost"],
}


class EnsembleModel:
    """Averages probability predictions from multiple trained BaseMLModel instances.

    Only models that are trained contribute to the average.  Untrained models
    are skipped, so the denominator always reflects active models.  When no
    models are trained, returns 0.5 (neutral probability).  When trained models
    all raise exceptions, raises ``PredictionError``.
    """

    def __init__(
        self,
        models: list[BaseMLModel],
        lstm_model: LSTMModel | None = None,
        stacking: StackingEnsemble | None = None,
        calibrator: EnsembleCalibrator | None = None,
        selected_features: list[str] | None = None,
        model_weights: dict[str, float] | None = None,
    ) -> None:
        self._models = models
        self._lstm_model = lstm_model
        self._stacking = stacking
        self._calibrator = calibrator
        self._meta_learner: LogisticRegression | None = None
        self.selected_features = selected_features
        self._model_weights = model_weights
        self.base_rate: float | None = None
        self.last_model_probas: dict[str, float] = {}
        self._total_predictions: int = 0
        self._clamped_predictions: int = 0

    @property
    def calibrator_active(self) -> bool:
        """Whether the calibrator is active (fitted and not bypassed)."""
        if self._calibrator is None:
            return False
        if not self._calibrator.is_fitted:
            return False
        return not getattr(self._calibrator, "calibrator_bypassed", False)

    def predict_proba(self, features: dict[str, float], *, symbol: str = "__default__") -> float:
        """Return mean BUY probability across all *trained* models.

        Falls back to 0.5 when no models are trained.
        Raises PredictionError when all trained models fail.

        After calling, ``last_model_probas`` contains per-model outputs
        keyed by class name (e.g. ``{"XGBoostModel": 0.8, "LSTMModel": 0.6}``).
        """
        probs: list[float] = []
        model_probas: dict[str, float] = {}
        any_trained = False

        for m in self._models:
            if getattr(m, "_model", None) is None:
                continue
            any_trained = True
            try:
                p = m.predict_proba(features)
                probs.append(p)
                model_probas[type(m).__name__] = p
            except Exception:
                _log.warning(
                    "ensemble_model_failed",
                    model=type(m).__name__,
                    exc_info=True,
                )

        if self._lstm_model is not None and getattr(self._lstm_model, "_trained", False):
            any_trained = True
            try:
                p = self._lstm_model.predict_proba(features, symbol=symbol)
                probs.append(p)
                model_probas["LSTMModel"] = p
            except Exception:
                _log.warning("ensemble_lstm_failed", exc_info=True)

        self.last_model_probas = model_probas

        if not probs:
            if any_trained:
                raise PredictionError("All ensemble sub-models failed to produce a prediction")
            return _DEFAULT_PROB

        # Stacking output is *already* calibrated -- never pass it through the
        # EnsembleCalibrator (would double-calibrate). Meta-learner output is
        # NOT pre-calibrated (audit #20), so it must go through the calibrator
        # just like the raw-average path.
        if self._stacking is not None and self._stacking.is_fitted:
            return self._stacking.predict_proba(probs)

        if self._meta_learner is not None:
            raw = self._predict_via_meta_learner(probs)
        else:
            raw = self._compute_raw_average(probs, model_probas)

        if self._calibrator is not None and self._calibrator.is_fitted:
            if getattr(self._calibrator, "calibrator_bypassed", False):
                return self._clamp_bypassed_prob(raw)
            return self._calibrator.calibrate(raw)
        return raw

    def _resolve_weight(self, class_name: str) -> float:
        """Look up model weight, tolerating key format differences.

        Training scripts save weights with keys like ``"xgboost"`` or ``"xgb"``,
        but ``predict_proba`` keys results by ``type(m).__name__`` (e.g.
        ``"XGBoostModel"``).  This method bridges the gap by trying exact match,
        lowercase match, and known alias lookup via ``_KEY_MAP``.
        """
        if self._model_weights is None:
            return 0.0
        # Try exact match first
        w = self._model_weights.get(class_name)
        if w is not None:
            return w
        # Try lowercase
        lower = class_name.lower()
        w = self._model_weights.get(lower)
        if w is not None:
            return w
        # Try known aliases
        for aliases in _KEY_MAP.values():
            if lower in aliases:
                for alias in aliases:
                    w = self._model_weights.get(alias)
                    if w is not None:
                        return w
        return 0.0

    def _compute_raw_average(
        self,
        probs: list[float],
        model_probas: dict[str, float],
    ) -> float:
        """Compute raw average: weighted if weights provided, else equal."""
        model_names = list(model_probas.keys())
        if self._model_weights and model_names:
            weighted_sum = 0.0
            weight_sum = 0.0
            for name, prob in zip(model_names, probs, strict=False):
                w = self._resolve_weight(name)
                weighted_sum += w * prob
                weight_sum += w
            return weighted_sum / weight_sum if weight_sum > 0 else _DEFAULT_PROB
        return sum(probs) / len(probs)

    def _clamp_bypassed_prob(self, raw: float) -> float:
        """Clamp raw probability to safe range when calibrator is bypassed.

        Tracks clamping rate and logs a critical warning when >50% of recent
        predictions require clamping.
        """
        clamped = max(_BYPASS_CLAMP_LOWER, min(_BYPASS_CLAMP_UPPER, raw))
        was_clamped = clamped != raw
        self._total_predictions += 1
        if was_clamped:
            self._clamped_predictions += 1
            _log.warning(
                "calibrator_bypassed_clamped",
                raw_prob=round(raw, 4),
                clamped_prob=round(clamped, 4),
            )
        if (
            self._total_predictions >= _MIN_PREDICTIONS_FOR_CLAMP_WARNING
            and self._clamped_predictions / self._total_predictions > _CLAMP_RATE_WARNING_THRESHOLD
        ):
            _log.critical(
                "calibrator_high_clamp_rate",
                total=self._total_predictions,
                clamped=self._clamped_predictions,
                rate=round(
                    self._clamped_predictions / self._total_predictions,
                    3,
                ),
            )
        return clamped

    def _predict_via_meta_learner(self, probs: list[float]) -> float:
        """Stack base model probabilities and pass through the meta-learner.

        Returns the class-1 (BUY) probability from the fitted LogisticRegression.
        """
        import numpy as np  # noqa: PLC0415

        x = np.array([probs], dtype=np.float64)
        return float(self._meta_learner.predict_proba(x)[0, 1])  # type: ignore[union-attr]

    def fit_meta_learner(
        self,
        base_model_oof_probs: np.ndarray,  # type: ignore[type-arg]
        labels: np.ndarray,  # type: ignore[type-arg]
    ) -> None:
        """Fit LogisticRegression meta-learner on out-of-fold base model predictions.

        The meta-learner learns optimal combination weights from OOF predictions,
        which prevents data leakage (base models never see the same data they predict on).

        Parameters
        ----------
        base_model_oof_probs:
            Shape (n_samples, n_models) -- out-of-fold probability predictions.
        labels:
            Shape (n_samples,) -- binary outcome labels.
        """
        import numpy as np  # noqa: PLC0415
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        x = np.asarray(base_model_oof_probs, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
        meta.fit(x, y)
        self._meta_learner = meta
        _log.info(
            "meta_learner_fitted",
            n_samples=len(y),
            n_models=x.shape[1],
        )

    def save_meta_learner(self, path: Path) -> None:
        """Persist the fitted meta-learner to disk using joblib.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``segment_dir / "meta_learner.pkl"``).
        """
        import joblib  # noqa: PLC0415

        if self._meta_learner is None:
            _log.warning("save_meta_learner called but no meta-learner is fitted")
            return
        joblib.dump(self._meta_learner, path)
        _log.info("meta_learner_saved", path=str(path))

    def load_meta_learner(self, path: Path) -> None:
        """Load a previously saved meta-learner from disk.

        Parameters
        ----------
        path:
            Source file path containing the joblib-serialized LogisticRegression.
        """
        # Key-gated HMAC verification before deserializing (audit 2026-06-28, HIGH:
        # the meta-learner pickle was loaded with bare joblib.load, bypassing the
        # integrity check the boosting models enforce). Function-local import keeps
        # the loader<->ensemble dependency one-directional.
        from finalayze.ml.loader import _verified_joblib_load  # noqa: PLC0415

        self._meta_learner = _verified_joblib_load(path)
        _log.info("meta_learner_loaded", path=str(path))

    @property
    def prediction_uncertainty(self) -> float:
        """Standard deviation of per-model probabilities (epistemic uncertainty proxy)."""
        if not self.last_model_probas or len(self.last_model_probas) < 2:  # noqa: PLR2004
            return 0.0
        import numpy as np  # noqa: PLC0415

        return float(np.std(list(self.last_model_probas.values())))

    def fit(
        self,
        X: list[dict[str, float]],  # noqa: N803
        y: list[int],
        *,
        sample_weight: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Train all constituent models (including LSTM if present).

        Each model is trained independently. If a model raises InsufficientDataError
        (e.g. LSTM when len(X) < sequence_length), it is left untrained and will
        return 0.5 in predict_proba -- graceful degradation.

        Args:
            X: Feature dictionaries.
            y: Binary labels.
            sample_weight: Optional per-sample weights passed through to each
                constituent model's ``fit`` method.
        """
        for model in self._models:
            with contextlib.suppress(InsufficientDataError):
                model.fit(X, y, sample_weight=sample_weight)
        if self._lstm_model is not None:
            with contextlib.suppress(InsufficientDataError):
                self._lstm_model.fit(X, y, sample_weight=sample_weight)
