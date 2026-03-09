"""Tests for temporal gap in train_models.py three-way split (5.8)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


class TestTrainGap:
    def test_three_way_split_produces_nonempty_sets(self) -> None:
        """Three-way split should produce non-empty train/cal/test sets."""
        from scripts.train_models import (
            _CALIBRATION_RATIO,
            _PURGE_GAP,
            _TRAIN_RATIO,
        )

        # Need enough samples so purge gaps don't consume cal/test entirely.
        # train_end = 0.70*n, cal_start = train_end + 100, cal_end = 0.85*n
        # test_start = cal_end + 100. For test to be non-empty: cal_end + 100 < n
        # -> 0.85n + 100 < n -> 100 < 0.15n -> n > 667
        n_samples = 800

        train_end = int(n_samples * _TRAIN_RATIO)
        cal_start = min(train_end + _PURGE_GAP, n_samples)
        cal_end = int(n_samples * (_TRAIN_RATIO + _CALIBRATION_RATIO))
        test_start = min(cal_end + _PURGE_GAP, n_samples)

        train_size = train_end
        cal_size = cal_end - cal_start
        test_size = n_samples - test_start

        assert train_size > 0, "Train set must be non-empty"
        assert cal_size > 0, "Calibration set must be non-empty"
        assert test_size > 0, "Test set must be non-empty"

    def test_purge_gaps_between_splits(self) -> None:
        """Purge gaps exist between train-cal and cal-test boundaries."""
        from scripts.train_models import (
            _CALIBRATION_RATIO,
            _PURGE_GAP,
            _TRAIN_RATIO,
        )

        n_samples = 800

        train_end = int(n_samples * _TRAIN_RATIO)
        cal_start = min(train_end + _PURGE_GAP, n_samples)
        cal_end = int(n_samples * (_TRAIN_RATIO + _CALIBRATION_RATIO))
        test_start = min(cal_end + _PURGE_GAP, n_samples)

        # Gap between train and cal
        gap1 = cal_start - train_end
        assert gap1 == _PURGE_GAP, f"Train-cal gap should be {_PURGE_GAP}, got {gap1}"

        # Gap between cal and test
        gap2 = test_start - cal_end
        assert gap2 == _PURGE_GAP, f"Cal-test gap should be {_PURGE_GAP}, got {gap2}"

    def test_no_temporal_overlap(self) -> None:
        """No overlap between any of the three splits."""
        from scripts.train_models import (
            _CALIBRATION_RATIO,
            _PURGE_GAP,
            _TRAIN_RATIO,
        )

        n_samples = 800

        train_end = int(n_samples * _TRAIN_RATIO)
        cal_start = min(train_end + _PURGE_GAP, n_samples)
        cal_end = int(n_samples * (_TRAIN_RATIO + _CALIBRATION_RATIO))
        test_start = min(cal_end + _PURGE_GAP, n_samples)

        # Define index ranges
        train_indices = set(range(0, train_end))
        cal_indices = set(range(cal_start, cal_end))
        test_indices = set(range(test_start, n_samples))

        assert len(train_indices & cal_indices) == 0, "Train and cal must not overlap"
        assert len(train_indices & test_indices) == 0, "Train and test must not overlap"
        assert len(cal_indices & test_indices) == 0, "Cal and test must not overlap"

    def test_purge_gap_equals_window_plus_max_hold(self) -> None:
        """The purge gap should be _WINDOW_SIZE + _TB_MAX_HOLD = 100."""
        from scripts.train_models import _PURGE_GAP, _TB_MAX_HOLD, _WINDOW_SIZE

        expected = _WINDOW_SIZE + _TB_MAX_HOLD  # 80 + 20 = 100
        assert expected == _PURGE_GAP

    def test_ratios_sum_to_one(self) -> None:
        """Train + calibration + test ratios should sum to 1.0."""
        from scripts.train_models import (
            _CALIBRATION_RATIO,
            _TEST_RATIO,
            _TRAIN_RATIO,
        )

        total = _TRAIN_RATIO + _CALIBRATION_RATIO + _TEST_RATIO
        assert abs(total - 1.0) < 1e-9, f"Ratios should sum to 1.0, got {total}"

    def test_train_one_segment_applies_three_way_split(self) -> None:
        """train_one_segment should apply a three-way split with purge gaps."""
        from scripts.train_models import train_one_segment

        n_samples = 800
        features = [{"feat": float(i)} for i in range(n_samples)]
        labels = [i % 2 for i in range(n_samples)]

        # Track what XGBoost.fit receives (train) and predict_proba receives (test)
        train_calls: list[int] = []
        test_calls: list[int] = []

        mock_model = MagicMock()

        def track_fit(x: list, y: list, **kwargs: object) -> None:
            train_calls.append(len(x))

        def track_predict(f: dict) -> float:
            test_calls.append(1)
            return 0.5

        mock_model.fit.side_effect = track_fit
        mock_model.predict_proba.side_effect = track_predict
        mock_model._model = None  # prevent feature importance logging
        mock_model._feature_names = None

        with (
            patch(
                "scripts.train_models._build_dataset",
                return_value=(features, labels, None, None),
            ),
            patch("scripts.train_models.XGBoostModel", return_value=mock_model),
            patch("scripts.train_models.LightGBMModel", return_value=mock_model),
            patch("scripts.train_models.CatBoostModel", return_value=mock_model),
        ):
            train_one_segment("us_tech", ["AAPL"], Path("/tmp/test_models"))

        # train_end = int(800 * 0.70) = 560
        expected_train = 560
        assert train_calls[0] == expected_train

        # cal: cal_start = 560+100 = 660, cal_end = int(800*0.85) = 680 -> 20 samples
        # test: test_start = 680+100 = 780, test_end = 800 -> 20 samples
        # predict_proba called for:
        #   calibrator fitting (20 cal * 3 models) +
        #   evaluation (20 test * 3 models) +
        #   meta-learner OOF predictions (20 test * 3 models)
        expected_cal_size = 20
        expected_test_size = 20
        expected_total = (expected_cal_size + expected_test_size * 2) * 3  # 3 models
        assert sum(test_calls) == expected_total

    def test_small_dataset_gap_clamps(self) -> None:
        """When dataset is small, purge gaps may consume cal/test entirely."""
        from scripts.train_models import (
            _CALIBRATION_RATIO,
            _PURGE_GAP,
            _TRAIN_RATIO,
        )

        n_samples = 100
        train_end = int(n_samples * _TRAIN_RATIO)  # 70
        cal_start = min(train_end + _PURGE_GAP, n_samples)  # min(170, 100) = 100
        cal_end = int(n_samples * (_TRAIN_RATIO + _CALIBRATION_RATIO))  # 85
        test_start = min(cal_end + _PURGE_GAP, n_samples)  # min(185, 100) = 100

        # cal_start (100) > cal_end (85) -> empty cal set
        cal_size = max(0, cal_end - cal_start)
        assert cal_size == 0

        # test_start (100) == n_samples -> empty test set
        test_size = n_samples - test_start
        assert test_size == 0
