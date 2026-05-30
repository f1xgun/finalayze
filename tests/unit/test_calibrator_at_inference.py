"""S6.2 — conformal calibrator must run on every inference path.

Sprint 2 added a meta-learner (LogisticRegression over base-model OOF probs)
and an EnsembleCalibrator that maps raw ensemble probability to a frequency-
calibrated one. The training pipeline saves both ``meta_learner.pkl`` and
``calibrator.pkl``, but ``EnsembleModel.predict_proba`` short-circuited the
meta-learner output and returned it raw — the calibrator was loaded into
memory but never invoked.

Audit #20: meta-learner output is *not* pre-calibrated (unlike the stacking
classifier whose ``predict_proba`` already produces calibrated values), so
this is a true correctness bug: probabilities passed downstream to the risk
sizer were uncalibrated whenever a meta-learner was active.

Contract:
  S6.2-01: predict_proba(meta-learner + calibrator) routes through calibrator.
  S6.2-02: predict_proba(no meta-learner + calibrator) still routes through it
           (regression guard — the raw-average path was already correct).
  S6.2-03: stacking classifier stays bypassed even when a calibrator is set
           (no double-calibration; stacking output already calibrated).
  S6.2-04: loader logs ``ml_calibrator_missing`` at WARNING when calibrator.pkl
           absent for a non-empty model dir (operator visibility, no crash).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from finalayze.ml.calibration import EnsembleCalibrator
from finalayze.ml.models.ensemble import EnsembleModel


class _StubModel:
    """Minimal BaseMLModel stand-in that returns a fixed probability."""

    def __init__(self, prob: float, *, trained: bool = True) -> None:
        self._prob = prob
        self._model: object | None = object() if trained else None

    def predict_proba(self, features: dict[str, float]) -> float:  # noqa: ARG002
        return self._prob


def _fitted_calibrator(*, output: float) -> EnsembleCalibrator:
    """Build a calibrator whose ``calibrate()`` returns a known sentinel.

    We don't need to actually fit on real data — patching ``calibrate`` lets
    us prove that the inference path invokes it regardless of meta-learner.
    """
    cal = EnsembleCalibrator()
    cal._fitted = True  # type: ignore[reportPrivateUsage]
    cal.calibrate = MagicMock(return_value=output)  # type: ignore[method-assign]
    return cal


def _fit_meta_learner(ensemble: EnsembleModel, *, n_base_models: int) -> None:
    """Fit a tiny meta-learner so the inference branch is taken."""
    rng = np.random.default_rng(0)
    n_samples = 100
    oof = rng.uniform(0.0, 1.0, size=(n_samples, n_base_models))
    labels = (oof.mean(axis=1) > 0.5).astype(np.int64)
    ensemble.fit_meta_learner(oof, labels)


# ─── S6.2-01 ────────────────────────────────────────────────────────────────
def test_meta_learner_output_passes_through_calibrator() -> None:
    """When meta-learner is set, predict_proba must still call the calibrator."""
    base_models = [_StubModel(prob=0.7), _StubModel(prob=0.8)]
    calibrator = _fitted_calibrator(output=0.42)
    ensemble = EnsembleModel(models=base_models, calibrator=calibrator)
    _fit_meta_learner(ensemble, n_base_models=2)

    result = ensemble.predict_proba({"feat": 1.0})

    assert calibrator.calibrate.called, (  # type: ignore[attr-defined]
        "Meta-learner path must invoke EnsembleCalibrator.calibrate "
        "(audit #20: meta-learner output is uncalibrated)"
    )
    assert result == pytest.approx(0.42)


# ─── S6.2-02 ────────────────────────────────────────────────────────────────
def test_raw_average_still_routes_through_calibrator() -> None:
    """Regression guard for the path that already worked correctly."""
    base_models = [_StubModel(prob=0.7), _StubModel(prob=0.8)]
    calibrator = _fitted_calibrator(output=0.33)
    ensemble = EnsembleModel(models=base_models, calibrator=calibrator)
    # No meta-learner fit → falls through to raw average

    result = ensemble.predict_proba({"feat": 1.0})

    assert calibrator.calibrate.called  # type: ignore[attr-defined]
    assert result == pytest.approx(0.33)


# ─── S6.2-03 ────────────────────────────────────────────────────────────────
def test_stacking_classifier_bypasses_calibrator() -> None:
    """Stacking output is already calibrated — calibrator must NOT be applied."""
    base_models = [_StubModel(prob=0.7), _StubModel(prob=0.8)]
    calibrator = _fitted_calibrator(output=0.99)  # would override if applied

    stacking = MagicMock()
    stacking.is_fitted = True
    stacking.predict_proba = MagicMock(return_value=0.55)

    ensemble = EnsembleModel(
        models=base_models,
        stacking=stacking,
        calibrator=calibrator,
    )

    result = ensemble.predict_proba({"feat": 1.0})

    assert result == pytest.approx(0.55), "Stacking output must pass through unchanged"
    assert not calibrator.calibrate.called, (  # type: ignore[attr-defined]
        "Stacking is pre-calibrated; applying calibrator would double-calibrate"
    )


# ─── S6.2-04 ────────────────────────────────────────────────────────────────
def test_loader_logs_when_calibrator_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operator visibility: warn when a segment dir has models but no calibrator.

    Tests the ``ml_calibrator_missing`` structlog event emitted by
    ``loader._load_segment``.
    """
    import logging

    import joblib
    import structlog

    from finalayze.ml.loader import FEATURE_SCHEMA_VERSION, _load_segment
    from finalayze.ml.models.xgboost_model import XGBoostModel

    # Configure structlog to feed into stdlib logging so caplog sees it.
    structlog.configure(
        processors=[structlog.stdlib.add_log_level, structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    caplog.set_level(logging.WARNING)

    segment_dir = tmp_path / "ru_energy"
    segment_dir.mkdir()
    # Drop a real (untrained) XGBoostModel via joblib so loader has *something*
    # to load. We aren't asserting on the model output here.
    model = XGBoostModel(segment_id="ru_energy")
    joblib.dump(model, segment_dir / "xgb.pkl")
    # Mandatory meta to clear the version guard
    (segment_dir / "segment_meta.json").write_text(
        json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION})
    )
    # Deliberately no calibrator.pkl

    _load_segment("ru_energy", segment_dir)

    assert any("ml_calibrator_missing" in rec.getMessage() for rec in caplog.records), (
        "Loader must emit ``ml_calibrator_missing`` warning when calibrator.pkl "
        f"is absent; observed: {[r.getMessage() for r in caplog.records]}"
    )
