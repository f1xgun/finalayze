"""S6.3 — segment_meta.json: loud warning + offline restore helper.

Audit #4 reported ru_energy missing ``segment_meta.json`` in production.
The original report claimed ``ModelLoader`` raises ``FileNotFoundError``;
in fact the loader silently treats a missing meta file as a legacy model
(``base_rate=None``, version guard never fires). That silent path lets
stale-schema artefacts boot indefinitely, hiding the cause from the
operator.

Contract:
  S6.3-01: loader emits ``ml_segment_meta_missing`` at WARNING when a
           segment directory has model files but no segment_meta.json.
  S6.3-02: the legacy ``base_rate=None`` behaviour stays — loader does
           NOT crash (audit #4 wording was wrong; preserving the working
           code path avoids breaking still-running setups).
  S6.3-03: ``scripts.restore_segment_meta`` regenerates segment_meta.json
           from neighbouring training artefacts (model_weights.json
           + selected_features.json + optional base_rate flag).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
import structlog


@pytest.fixture(autouse=True)
def _structlog_to_stdlib() -> None:
    """Route structlog warnings into stdlib logging so caplog sees them."""
    structlog.configure(
        processors=[structlog.stdlib.add_log_level, structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def _stage_segment_with_models(segment_dir: Path) -> None:
    """Drop an untrained XGBoost model into the dir so the loader has work to do."""
    from finalayze.ml.models.xgboost_model import XGBoostModel

    segment_dir.mkdir(parents=True)
    xgb = XGBoostModel(segment_id=segment_dir.name)
    joblib.dump(xgb, segment_dir / "xgb.pkl")


# ─── S6.3-01 ────────────────────────────────────────────────────────────────
def test_loader_warns_when_segment_meta_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing segment_meta.json must surface as ``ml_segment_meta_missing``."""
    from finalayze.ml.loader import _load_segment

    segment_dir = tmp_path / "ru_energy"
    _stage_segment_with_models(segment_dir)

    caplog.set_level(logging.WARNING)
    _load_segment("ru_energy", segment_dir)

    assert any("ml_segment_meta_missing" in rec.getMessage() for rec in caplog.records), (
        f"expected ``ml_segment_meta_missing`` WARNING, observed: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ─── S6.3-02 ────────────────────────────────────────────────────────────────
def test_loader_still_returns_ensemble_when_meta_missing(tmp_path: Path) -> None:
    """Loader must not crash — base_rate stays None, ensemble loads."""
    from finalayze.ml.loader import _load_segment

    segment_dir = tmp_path / "ru_energy"
    _stage_segment_with_models(segment_dir)

    ensemble = _load_segment("ru_energy", segment_dir)
    assert ensemble is not None
    assert ensemble.base_rate is None


# ─── S6.3-03 ────────────────────────────────────────────────────────────────
def test_restore_segment_meta_cli_regenerates_file(tmp_path: Path) -> None:
    """``scripts/restore_segment_meta.py`` writes a valid segment_meta.json.

    Operator workflow: when models/<segment>/segment_meta.json is missing
    (e.g. ru_energy after audit #4), the helper rebuilds it from sibling
    artefacts so the loader's version guard works again.
    """
    from finalayze.ml.loader import FEATURE_SCHEMA_VERSION

    segment_dir = tmp_path / "ru_energy"
    _stage_segment_with_models(segment_dir)
    (segment_dir / "model_weights.json").write_text(
        json.dumps({"xgboost": 0.4, "lightgbm": 0.35, "catboost": 0.25})
    )
    (segment_dir / "selected_features.json").write_text(json.dumps(["rsi_14", "macd_diff"]))

    script = Path(__file__).resolve().parents[2] / "scripts" / "restore_segment_meta.py"
    result = subprocess.run(  # noqa: S603 — test fixture, args are hermetic
        [
            sys.executable,
            str(script),
            "--models-dir",
            str(tmp_path),
            "--segment",
            "ru_energy",
            "--base-rate",
            "0.4923",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"helper failed: {result.stderr}"

    meta_path = segment_dir / "segment_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert meta["base_rate"] == pytest.approx(0.4923)
