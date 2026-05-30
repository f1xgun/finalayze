"""Model persistence: load/save EnsembleModel per segment (Layer 3)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from finalayze.ml.registry import MLModelRegistry

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.ensemble import EnsembleModel

_log = structlog.get_logger()

# Increment this whenever the feature set changes in an incompatible way.
# Saved models whose feature_schema_version differs from this value will be
# rejected at load time (graceful degradation — registry returns None for
# that segment rather than producing silently wrong predictions).
# History: 1 = original; 2 = +4 MOEX features; 3 = +7 MOEX macro (cbr/fx/brent).
FEATURE_SCHEMA_VERSION: int = 3


def load_registry(model_dir: Path, segments: list[str]) -> MLModelRegistry:
    """Load saved models for each segment, returning a populated registry.

    Missing or corrupt model files are logged as warnings and skipped —
    the registry will simply return ``None`` for those segments.
    """
    registry = MLModelRegistry()

    if not model_dir.is_dir():
        _log.warning("Model directory %s does not exist — returning empty registry", model_dir)
        return registry

    for segment_id in segments:
        segment_dir = model_dir / segment_id
        if not segment_dir.is_dir():
            _log.debug("No model directory for segment %s", segment_id)
            continue
        try:
            ensemble = _load_segment(segment_id, segment_dir)
            registry.register(segment_id, ensemble)
            _log.info("Loaded ML ensemble for segment %s", segment_id)
        except Exception:
            _log.warning(
                "Failed to load models for segment %s — skipping",
                segment_id,
                exc_info=True,
            )

    return registry


def _load_segment(segment_id: str, segment_dir: Path) -> EnsembleModel:  # noqa: PLR0912, PLR0915
    """Load individual model files and assemble an EnsembleModel."""
    from finalayze.ml.models.catboost_model import CatBoostModel  # noqa: PLC0415
    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415
    from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415
    from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

    # S2.3: warn loudly if the segment was force-saved or its quality gate
    # failed. Logged BEFORE any model loading so the warning fires even when
    # downstream loads raise (e.g. missing files).
    gate_results_path = segment_dir / "wf_gate_results.json"
    if gate_results_path.exists():
        gate_data = json.loads(gate_results_path.read_text())
        if gate_data.get("force_saved") is True or gate_data.get("overall_passed") is False:
            _log.warning(
                "ml_force_saved_artifact_loaded",
                segment_id=segment_id,
                overall_passed=gate_data.get("overall_passed"),
                bh_passed=gate_data.get("bh_passed"),
                best_accuracy=gate_data.get("best_accuracy"),
                force_saved=gate_data.get("force_saved", False),
            )

    models: list[BaseMLModel] = []

    xgb_path = segment_dir / "xgb.pkl"
    if xgb_path.exists():
        models.append(XGBoostModel.load_from(xgb_path))

    lgbm_path = segment_dir / "lgbm.pkl"
    if lgbm_path.exists():
        models.append(LightGBMModel.load_from(lgbm_path))

    # CatBoost (primary third model); fall back to LSTM for backward compat
    catboost_path = segment_dir / "catboost.pkl"
    lstm_path = segment_dir / "lstm.pkl"
    lstm_model = None  # kept for backward compat with EnsembleModel API

    if catboost_path.exists():
        models.append(CatBoostModel.load_from(catboost_path))
    elif lstm_path.exists():
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415

        lstm_model = LSTMModel(segment_id=segment_id)
        lstm_model.load(lstm_path)

    if not models and lstm_model is None:
        msg = f"No model files found in {segment_dir}"
        raise FileNotFoundError(msg)

    # Load MI-selected feature list if available (feature mismatch fix)
    selected_features: list[str] | None = None
    features_path = segment_dir / "selected_features.json"
    if features_path.exists():
        selected_features = json.loads(features_path.read_text())

    # Load fitted EnsembleCalibrator if available. S6.2: warn when missing so
    # operators can spot segments running uncalibrated inference (the meta-
    # learner path used to silently bypass the calibrator before Sprint 6).
    from finalayze.ml.calibration import EnsembleCalibrator  # noqa: PLC0415

    calibrator: EnsembleCalibrator | None = None
    calibrator_path = segment_dir / "calibrator.pkl"
    if calibrator_path.exists():
        import joblib  # noqa: PLC0415

        loaded_cal = joblib.load(calibrator_path)
        if isinstance(loaded_cal, EnsembleCalibrator) and loaded_cal.is_fitted:
            calibrator = loaded_cal
            _log.debug("Loaded fitted calibrator for segment %s", segment_id)
        else:
            _log.warning(
                "ml_calibrator_unfit",
                segment_id=segment_id,
                reason="calibrator.pkl present but not fitted; inference will be uncalibrated",
            )
    else:
        _log.warning(
            "ml_calibrator_missing",
            segment_id=segment_id,
            reason="no calibrator.pkl; inference probabilities are uncalibrated",
        )

    # Load performance-weighted model weights if available
    model_weights: dict[str, float] | None = None
    weights_path = segment_dir / "model_weights.json"
    if weights_path.exists():
        model_weights = json.loads(weights_path.read_text())
        _log.debug("Loaded model_weights for segment %s", segment_id)

    # Load segment metadata (base_rate, feature_schema_version, etc.) if available
    base_rate: float | None = None
    meta_path = segment_dir / "segment_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

        # Version guard: reject models trained with a different feature schema.
        # Default to 1 (pre-MOEX) when the field is absent (legacy models).
        saved_version: int = meta.get("feature_schema_version", 1)
        if saved_version != FEATURE_SCHEMA_VERSION:
            _log.warning(
                "feature_schema_version_mismatch",
                segment_id=segment_id,
                saved=saved_version,
                current=FEATURE_SCHEMA_VERSION,
            )
            msg = (
                f"Feature schema version mismatch for segment '{segment_id}': "
                f"saved={saved_version}, current={FEATURE_SCHEMA_VERSION}. "
                "Retrain models to resolve."
            )
            raise ValueError(msg)

        base_rate = meta.get("base_rate")
        _log.debug("Loaded segment_meta for segment %s (base_rate=%s)", segment_id, base_rate)

    ensemble = EnsembleModel(
        models=models,
        lstm_model=lstm_model,
        selected_features=selected_features,
        calibrator=calibrator,
        model_weights=model_weights,
    )
    ensemble.base_rate = base_rate

    # Load stacking meta-learner if available
    meta_learner_path = segment_dir / "meta_learner.pkl"
    if meta_learner_path.exists():
        ensemble.load_meta_learner(meta_learner_path)
        _log.debug("Loaded meta-learner for segment %s", segment_id)

    return ensemble


def save_ensemble(model_dir: Path, segment_id: str, ensemble: EnsembleModel) -> None:
    """Save all constituent models of an ensemble atomically.

    Uses temp files + rename to avoid leaving corrupt files if the process
    is interrupted mid-write.
    """
    segment_dir = model_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    for model in ensemble._models:
        model_type = type(model).__name__.lower()
        if "xgboost" in model_type:
            _atomic_save(model, segment_dir / "xgb.pkl")
        elif "lightgbm" in model_type:
            _atomic_save(model, segment_dir / "lgbm.pkl")
        elif "catboost" in model_type:
            _atomic_save(model, segment_dir / "catboost.pkl")

    if ensemble._lstm_model is not None:
        ensemble._lstm_model.save(segment_dir / "lstm.pkl")

    # Persist MI-selected feature list alongside models (feature mismatch fix)
    if ensemble.selected_features is not None:
        features_path = segment_dir / "selected_features.json"
        features_path.write_text(json.dumps(ensemble.selected_features))

    # Persist performance-weighted model weights alongside models
    if ensemble._model_weights is not None:
        weights_path = segment_dir / "model_weights.json"
        weights_path.write_text(json.dumps(ensemble._model_weights, indent=2))

    # Persist segment metadata (feature_schema_version + base_rate).
    # Always written so that the version guard in _load_segment can reject
    # stale models after a feature-set change.
    meta: dict[str, object] = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
    if getattr(ensemble, "base_rate", None) is not None:
        meta["base_rate"] = ensemble.base_rate
    meta_path = segment_dir / "segment_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Persist fitted EnsembleCalibrator alongside models
    if ensemble._calibrator is not None and getattr(ensemble._calibrator, "is_fitted", False):
        _atomic_save(ensemble._calibrator, segment_dir / "calibrator.pkl")

    # Persist stacking meta-learner alongside models
    if ensemble._meta_learner is not None:
        ensemble.save_meta_learner(segment_dir / "meta_learner.pkl")


def _get_hmac_key() -> str:
    """Return the ML model HMAC key from settings, or empty string."""
    try:
        from config.settings import get_settings  # noqa: PLC0415

        return getattr(get_settings(), "ml_model_hmac_key", "")
    except Exception:
        return ""


def _atomic_save(model: object, target: Path) -> None:
    """Save a model to *target* atomically via temp + rename."""
    import joblib  # noqa: PLC0415

    fd, tmp_path_str = tempfile.mkstemp(dir=target.parent, suffix=".tmp", prefix=target.stem)
    tmp_path = Path(tmp_path_str)
    try:
        os.close(fd)
        joblib.dump(model, tmp_path)
        tmp_path.rename(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Sign model after successful save
    key = _get_hmac_key()
    if key:
        from finalayze.ml.integrity import sign_model  # noqa: PLC0415

        sign_model(target, key.encode())
