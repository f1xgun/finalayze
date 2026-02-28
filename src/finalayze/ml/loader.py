"""Model persistence: load/save EnsembleModel per segment (Layer 3)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.ml.registry import MLModelRegistry

if TYPE_CHECKING:
    from finalayze.ml.models.base import BaseMLModel
    from finalayze.ml.models.ensemble import EnsembleModel

_log = logging.getLogger(__name__)


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


def _load_segment(segment_id: str, segment_dir: Path) -> EnsembleModel:
    """Load individual model files and assemble an EnsembleModel."""
    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415
    from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415
    from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415
    from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

    models: list[BaseMLModel] = []

    xgb_path = segment_dir / "xgb.pkl"
    if xgb_path.exists():
        models.append(XGBoostModel.load_from(xgb_path))

    lgbm_path = segment_dir / "lgbm.pkl"
    if lgbm_path.exists():
        models.append(LightGBMModel.load_from(lgbm_path))

    lstm_model: LSTMModel | None = None
    lstm_path = segment_dir / "lstm.pkl"
    if lstm_path.exists():
        lstm_model = LSTMModel(segment_id=segment_id)
        lstm_model.load(lstm_path)

    if not models and lstm_model is None:
        msg = f"No model files found in {segment_dir}"
        raise FileNotFoundError(msg)

    return EnsembleModel(models=models, lstm_model=lstm_model)


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

    if ensemble._lstm_model is not None:
        ensemble._lstm_model.save(segment_dir / "lstm.pkl")


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
