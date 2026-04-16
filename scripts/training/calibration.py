"""Calibration and meta-learner fitting for the training pipeline.

Fits EnsembleCalibrator and stacking meta-learner on out-of-sample
predictions, then saves them alongside the trained models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel

if TYPE_CHECKING:
    from pathlib import Path

_MIN_META_LEARNER_SAMPLES = 20


def fit_and_save_calibrator(
    segment_id: str,
    segment_dir: Path,
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> None:
    """Fit EnsembleCalibrator on out-of-sample ensemble probabilities and save it.

    Uses the TEST split to avoid data leakage: the calibrator sees the model's
    out-of-sample probability distribution, not the training distribution.
    """
    import numpy as _np  # noqa: PLC0415

    from finalayze.ml.calibration import EnsembleCalibrator  # noqa: PLC0415
    from finalayze.ml.loader import _atomic_save  # noqa: PLC0415

    raw_probas: list[float] = []
    for feat in test_features:
        probs: list[float] = []
        for m in models:
            trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
            if trained is None:
                continue
            try:
                probs.append(m.predict_proba(feat))
            except Exception:
                continue
        if probs:
            raw_probas.append(sum(probs) / len(probs))
        else:
            raw_probas.append(0.5)

    calibrator = EnsembleCalibrator()
    calibrator.fit(_np.array(raw_probas), _np.array(test_labels))

    if calibrator.is_fitted:
        _atomic_save(calibrator, segment_dir / "calibrator.pkl")
        # Show calibration effect
        cal_low = calibrator.calibrate(0.2)
        cal_high = calibrator.calibrate(0.8)
        print(
            f"[{segment_id}] Calibrator fitted on {len(test_features)} OOS samples: "
            f"raw 0.2 -> {cal_low:.3f}, raw 0.8 -> {cal_high:.3f}"
        )
    else:
        print(f"[{segment_id}] Calibrator skipped (insufficient OOS data or single class)")


def fit_and_save_meta_learner(
    segment_id: str,
    segment_dir: Path,
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    oof_features: list[dict[str, float]],
    oof_labels: list[int],
) -> None:
    """Fit a stacking meta-learner on out-of-fold base model predictions and save it.

    Generates per-model probability predictions on the OOF set (data the base models
    were NOT trained on), stacks them into a matrix, and trains a LogisticRegression
    meta-learner to learn optimal combination weights.
    """
    import numpy as _np  # noqa: PLC0415

    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415

    if len(oof_features) < _MIN_META_LEARNER_SAMPLES:
        print(
            f"[{segment_id}] Too few OOF samples ({len(oof_features)}) for meta-learner, skipping."
        )
        return

    # Collect per-model OOF probabilities
    model_proba_columns: list[list[float]] = []
    model_names: list[str] = []

    for m in models:
        trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
        if trained is None:
            continue
        probas: list[float] = []
        for feat in oof_features:
            try:
                probas.append(m.predict_proba(feat))
            except Exception:
                probas.append(0.5)
        model_proba_columns.append(probas)
        model_names.append(type(m).__name__)

    if not model_proba_columns:
        print(f"[{segment_id}] No trained models for meta-learner OOF predictions, skipping.")
        return

    oof_matrix = _np.column_stack(model_proba_columns)
    labels_arr = _np.array(oof_labels, dtype=_np.int64)

    # Fit meta-learner via EnsembleModel helper
    ensemble = EnsembleModel(models=[])
    ensemble.fit_meta_learner(oof_matrix, labels_arr)

    meta_path = segment_dir / "meta_learner.pkl"
    ensemble.save_meta_learner(meta_path)
    print(
        f"[{segment_id}] Saved meta_learner.pkl "
        f"(trained on {len(oof_features)} OOF samples, {len(model_names)} models: "
        f"{', '.join(model_names)})"
    )
