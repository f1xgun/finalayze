"""Tests for walk-forward training, calendar-date split, and BH correction (D1/D3/D4)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


class TestCalendarDateSplit:
    """D4: Walk-forward folds split by calendar date."""

    def test_generates_folds_from_timestamps(self) -> None:
        """Timestamps spanning 3+ years produce multiple folds."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        # 5 years of daily timestamps
        timestamps = [base + timedelta(days=i) for i in range(1825)]
        folds = _generate_walk_forward_folds(timestamps)
        assert len(folds) >= 3  # noqa: PLR2004

    def test_folds_have_three_splits(self) -> None:
        """Each fold has train, cal, test index lists."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(1825)]
        folds = _generate_walk_forward_folds(timestamps)
        assert len(folds) > 0
        for train_idx, _cal_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_no_temporal_overlap_between_splits(self) -> None:
        """Train, cal, and test indices must not overlap within a fold."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(1825)]
        folds = _generate_walk_forward_folds(timestamps)
        for train_idx, cal_idx, test_idx in folds:
            train_set = set(train_idx)
            cal_set = set(cal_idx)
            test_set = set(test_idx)
            assert len(train_set & cal_set) == 0, "Train and cal must not overlap"
            assert len(train_set & test_set) == 0, "Train and test must not overlap"
            assert len(cal_set & test_set) == 0, "Cal and test must not overlap"

    def test_train_before_test_temporally(self) -> None:
        """All train timestamps must be before all test timestamps."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(1825)]
        folds = _generate_walk_forward_folds(timestamps)
        for train_idx, _cal_idx, test_idx in folds:
            max_train_ts = max(timestamps[i] for i in train_idx)
            min_test_ts = min(timestamps[i] for i in test_idx)
            assert max_train_ts < min_test_ts

    def test_empty_timestamps_returns_empty(self) -> None:
        """Empty timestamps produce no folds."""
        from scripts.train_models import _generate_walk_forward_folds

        folds = _generate_walk_forward_folds([])
        assert folds == []

    def test_short_period_returns_empty(self) -> None:
        """Less than 18 months of data produces no folds."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(365)]
        folds = _generate_walk_forward_folds(timestamps)
        assert folds == []

    def test_folds_step_forward(self) -> None:
        """Consecutive folds step forward in time."""
        from scripts.train_models import _generate_walk_forward_folds

        base = datetime(2020, 1, 1, tzinfo=UTC)
        timestamps = [base + timedelta(days=i) for i in range(1825)]
        folds = _generate_walk_forward_folds(timestamps)
        if len(folds) >= 2:  # noqa: PLR2004
            # First test index of fold 1 should be after fold 0
            min_test_0 = min(folds[0][2])
            min_test_1 = min(folds[1][2])
            assert min_test_1 > min_test_0


class TestBHCorrection:
    """D3: Benjamini-Hochberg multiple testing correction."""

    def test_all_significant_pass(self) -> None:
        """All very small p-values pass."""
        from scripts.train_models import _apply_bh_correction

        p_values = [0.001, 0.002, 0.003]
        results = _apply_bh_correction(p_values, fdr=0.10)
        assert all(results)

    def test_all_nonsignificant_fail(self) -> None:
        """All large p-values fail."""
        from scripts.train_models import _apply_bh_correction

        p_values = [0.50, 0.60, 0.70]
        results = _apply_bh_correction(p_values, fdr=0.10)
        assert not any(results)

    def test_mixed_p_values(self) -> None:
        """Some pass, some fail with mixed p-values."""
        from scripts.train_models import _apply_bh_correction

        p_values = [0.001, 0.50, 0.90]
        results = _apply_bh_correction(p_values, fdr=0.10)
        # First should pass, others should fail
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False

    def test_empty_p_values(self) -> None:
        """Empty input returns empty."""
        from scripts.train_models import _apply_bh_correction

        results = _apply_bh_correction([], fdr=0.10)
        assert results == []

    def test_single_p_value_significant(self) -> None:
        """Single significant p-value passes."""
        from scripts.train_models import _apply_bh_correction

        results = _apply_bh_correction([0.05], fdr=0.10)
        assert results == [True]

    def test_single_p_value_not_significant(self) -> None:
        """Single non-significant p-value fails."""
        from scripts.train_models import _apply_bh_correction

        results = _apply_bh_correction([0.50], fdr=0.10)
        assert results == [False]

    def test_fdr_threshold_respected(self) -> None:
        """BH threshold = (rank/n) * fdr. With n=3, fdr=0.10:
        rank 1: threshold=0.033, rank 2: 0.067, rank 3: 0.10."""
        from scripts.train_models import _apply_bh_correction

        p_values = [0.03, 0.06, 0.09]
        results = _apply_bh_correction(p_values, fdr=0.10)
        assert results[0] is True  # 0.03 <= 0.033
        assert results[1] is True  # 0.06 <= 0.067
        assert results[2] is True  # 0.09 <= 0.10


class TestBuildDatasetWithTimestamps:
    """D4: Dataset building returns timestamps."""

    def test_returns_five_tuple(self) -> None:
        """_build_dataset_with_timestamps returns 5 elements."""
        from unittest.mock import patch

        from scripts.train_models import _build_dataset_with_timestamps

        # Mock to avoid actual data fetching
        fake_ts = [datetime(2023, 1, i + 1, tzinfo=UTC) for i in range(5)]
        fake_return = (
            [{"feat": float(i)} for i in range(5)],
            [0, 1, 0, 1, 0],
            None,
            None,
            fake_ts,
        )
        with patch(
            "scripts.training.dataset_builder._build_dataset_direction",
            return_value=fake_return,
        ):
            result = _build_dataset_with_timestamps("us_tech", ["AAPL"], label_mode="direction")

        assert len(result) == 5  # noqa: PLR2004
        features, _labels, _weights, _hold_bars, timestamps = result
        assert len(features) == 5  # noqa: PLR2004
        assert len(timestamps) == 5  # noqa: PLR2004
        assert all(isinstance(ts, datetime) for ts in timestamps)


class TestWalkForwardParams:
    """Walk-forward parameter sanity checks."""

    def test_wf_params_exist(self) -> None:
        """Walk-forward parameters are defined."""
        from scripts.train_models import (
            _WF_CAL_MONTHS,
            _WF_STEP_MONTHS,
            _WF_TEST_MONTHS,
            _WF_TRAIN_MONTHS,
        )

        assert _WF_TRAIN_MONTHS == 12  # noqa: PLR2004
        assert _WF_CAL_MONTHS == 2  # noqa: PLR2004
        assert _WF_TEST_MONTHS == 4  # noqa: PLR2004
        assert _WF_STEP_MONTHS == 3  # noqa: PLR2004

    def test_parse_args_walk_forward_default_false(self) -> None:
        """--walk-forward defaults to False."""
        from scripts.train_models import _parse_args

        args = _parse_args([])
        assert args.walk_forward is False

    def test_parse_args_walk_forward_flag(self) -> None:
        """--walk-forward flag sets True."""
        from scripts.train_models import _parse_args

        args = _parse_args(["--walk-forward"])
        assert args.walk_forward is True

    def test_parse_args_force_save_default_false(self) -> None:
        """--force-save defaults to False."""
        from scripts.train_models import _parse_args

        args = _parse_args([])
        assert args.force_save is False

    def test_parse_args_force_save_flag(self) -> None:
        """--force-save flag sets True."""
        from scripts.train_models import _parse_args

        args = _parse_args(["--force-save"])
        assert args.force_save is True


class TestQualityGateEnforcement:
    """Models must not be saved when quality gates fail (unless --force-save)."""

    @staticmethod
    def _make_fake_dataset(n: int = 200) -> tuple:
        """Build a fake dataset with timestamps spanning 3+ years for WF folds."""
        base = datetime(2020, 1, 1, tzinfo=UTC)
        features = [{"feat_a": float(i), "feat_b": float(i * 2)} for i in range(n)]
        labels = [i % 2 for i in range(n)]
        timestamps = [base + timedelta(days=i * 5) for i in range(n)]
        return features, labels, None, None, timestamps

    @staticmethod
    def _make_fake_fold_results(passed: bool) -> list:
        """Create fake fold results that either all pass or all fail."""
        from finalayze.ml.training.quality_gates import QualityGateResult

        gates = [
            "accuracy",
            "brier",
            "profit_factor",
            "signal_count",
            "class_balance",
            "sensitivity",
            "specificity",
        ]
        fold = [
            QualityGateResult(gate_name=g, passed=passed, value=0.5, threshold=0.5) for g in gates
        ]
        return [fold, fold]  # Two folds

    def test_models_not_saved_when_gates_fail(self, tmp_path: Path) -> None:
        """When quality gates fail and force_save=False, model files must not exist."""
        from scripts.train_models import train_walk_forward

        from finalayze.ml.training.quality_gates import FoldMetrics

        features, labels, bw, hb, timestamps = self._make_fake_dataset()
        failing_folds = self._make_fake_fold_results(passed=False)
        fake_fold_metrics = FoldMetrics(accuracy=0.48, brier_score=0.26, log_loss=0.70, n_test=60)

        with (
            patch(
                "scripts.training.walk_forward.build_dataset_with_timestamps",
                return_value=(features, labels, bw, hb, timestamps),
            ),
            patch(
                "scripts.training.walk_forward.generate_walk_forward_folds",
                return_value=[
                    (list(range(100)), list(range(100, 140)), list(range(140, 200))),
                ],
            ),
            patch(
                "scripts.training.walk_forward.select_features",
                return_value=["feat_a", "feat_b"],
            ),
            patch("scripts.training.walk_forward.compute_decay_weights") as mock_decay,
            patch(
                "scripts.training.walk_forward.evaluate_fold_metrics",
                return_value=fake_fold_metrics,
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_fold",
                return_value=failing_folds[0],
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_walk_forward",
                return_value=(False, {"accuracy": 0.3, "brier": 0.4}),
            ),
            patch("scripts.training.walk_forward.XGBoostModel") as mock_xgb_cls,
            patch("scripts.training.walk_forward.LightGBMModel") as mock_lgbm_cls,
            patch("scripts.training.walk_forward.CatBoostModel") as mock_cat_cls,
        ):
            import numpy as np

            mock_decay.return_value = np.ones(100)

            mock_xgb = MagicMock()
            mock_lgbm = MagicMock()
            mock_cat = MagicMock()
            mock_xgb_cls.return_value = mock_xgb
            mock_lgbm_cls.return_value = mock_lgbm
            mock_cat_cls.return_value = mock_cat

            result = train_walk_forward(
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
                force_save=False,
            )

        # Gate pass rates should still be returned
        assert result is not None
        assert "accuracy" in result

        # Gate results JSON should be saved for diagnostics
        gate_results_path = tmp_path / "us_tech" / "wf_gate_results.json"
        assert gate_results_path.exists()
        gate_data = json.loads(gate_results_path.read_text())
        assert gate_data["overall_passed"] is False

        # Model files must NOT be saved
        mock_xgb.save.assert_not_called()
        mock_lgbm.save.assert_not_called()
        mock_cat.save.assert_not_called()

    def test_models_saved_when_gates_fail_with_force_save(self, tmp_path: Path) -> None:
        """When quality gates fail but force_save=True, model files must be saved."""
        from scripts.train_models import train_walk_forward

        from finalayze.ml.training.quality_gates import FoldMetrics

        features, labels, bw, hb, timestamps = self._make_fake_dataset()
        failing_folds = self._make_fake_fold_results(passed=False)
        fake_fold_metrics = FoldMetrics(accuracy=0.48, brier_score=0.26, log_loss=0.70, n_test=60)

        with (
            patch(
                "scripts.training.walk_forward.build_dataset_with_timestamps",
                return_value=(features, labels, bw, hb, timestamps),
            ),
            patch(
                "scripts.training.walk_forward.generate_walk_forward_folds",
                return_value=[
                    (list(range(100)), list(range(100, 140)), list(range(140, 200))),
                ],
            ),
            patch(
                "scripts.training.walk_forward.select_features",
                return_value=["feat_a", "feat_b"],
            ),
            patch("scripts.training.walk_forward.compute_decay_weights") as mock_decay,
            patch(
                "scripts.training.walk_forward.evaluate_fold_metrics",
                return_value=fake_fold_metrics,
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_fold",
                return_value=failing_folds[0],
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_walk_forward",
                return_value=(False, {"accuracy": 0.3, "brier": 0.4}),
            ),
            patch("scripts.training.walk_forward.XGBoostModel") as mock_xgb_cls,
            patch("scripts.training.walk_forward.LightGBMModel") as mock_lgbm_cls,
            patch("scripts.training.walk_forward.CatBoostModel") as mock_cat_cls,
        ):
            import numpy as np

            mock_decay.return_value = np.ones(100)

            mock_xgb = MagicMock()
            mock_lgbm = MagicMock()
            mock_cat = MagicMock()
            mock_xgb.predict_proba.return_value = 0.6
            mock_lgbm.predict_proba.return_value = 0.6
            mock_cat.predict_proba.return_value = 0.6
            mock_xgb_cls.return_value = mock_xgb
            mock_lgbm_cls.return_value = mock_lgbm
            mock_cat_cls.return_value = mock_cat

            result = train_walk_forward(
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
                force_save=True,
            )

        # Gate pass rates should still be returned
        assert result is not None

        # Model files MUST be saved because force_save=True
        mock_xgb.save.assert_called_once()
        mock_lgbm.save.assert_called_once()
        mock_cat.save.assert_called_once()

    def test_models_saved_when_gates_pass(self, tmp_path: Path) -> None:
        """When quality gates pass, model files must be saved (force_save irrelevant)."""
        from scripts.train_models import train_walk_forward

        from finalayze.ml.training.quality_gates import FoldMetrics

        features, labels, bw, hb, timestamps = self._make_fake_dataset()
        passing_folds = self._make_fake_fold_results(passed=True)
        fake_fold_metrics = FoldMetrics(accuracy=0.65, brier_score=0.20, log_loss=0.60, n_test=60)

        with (
            patch(
                "scripts.training.walk_forward.build_dataset_with_timestamps",
                return_value=(features, labels, bw, hb, timestamps),
            ),
            patch(
                "scripts.training.walk_forward.generate_walk_forward_folds",
                return_value=[
                    (list(range(100)), list(range(100, 140)), list(range(140, 200))),
                ],
            ),
            patch(
                "scripts.training.walk_forward.select_features",
                return_value=["feat_a", "feat_b"],
            ),
            patch("scripts.training.walk_forward.compute_decay_weights") as mock_decay,
            patch(
                "scripts.training.walk_forward.evaluate_fold_metrics",
                return_value=fake_fold_metrics,
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_fold",
                return_value=passing_folds[0],
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_walk_forward",
                return_value=(True, {"accuracy": 0.8, "brier": 0.9}),
            ),
            patch("scripts.training.walk_forward.XGBoostModel") as mock_xgb_cls,
            patch("scripts.training.walk_forward.LightGBMModel") as mock_lgbm_cls,
            patch("scripts.training.walk_forward.CatBoostModel") as mock_cat_cls,
        ):
            import numpy as np

            mock_decay.return_value = np.ones(100)

            mock_xgb = MagicMock()
            mock_lgbm = MagicMock()
            mock_cat = MagicMock()
            mock_xgb.predict_proba.return_value = 0.7
            mock_lgbm.predict_proba.return_value = 0.7
            mock_cat.predict_proba.return_value = 0.7
            mock_xgb_cls.return_value = mock_xgb
            mock_lgbm_cls.return_value = mock_lgbm
            mock_cat_cls.return_value = mock_cat

            result = train_walk_forward(
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
                force_save=False,
            )

        assert result is not None

        # Model files MUST be saved because gates passed
        mock_xgb.save.assert_called_once()
        mock_lgbm.save.assert_called_once()
        mock_cat.save.assert_called_once()
