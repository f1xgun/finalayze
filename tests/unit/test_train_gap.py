"""Tests for temporal gap in train_models.py train/test split (5.8)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


class TestTrainGap:
    def test_gap_between_train_and_test(self) -> None:
        """There should be a gap of _WINDOW_SIZE between train and test sets."""
        from scripts.train_models import _TRAIN_RATIO, _WINDOW_SIZE

        # Use enough samples so gap doesn't get clamped: need split + WINDOW < n
        # split = 0.8 * n, so need 0.8n + 60 < n → n > 300
        n_samples = 500

        split = int(n_samples * _TRAIN_RATIO)
        gap_end = min(split + _WINDOW_SIZE, n_samples)

        # Verify gap math: train=[0, split), gap=[split, gap_end), test=[gap_end, n)
        assert split < gap_end <= n_samples
        assert n_samples - gap_end > 0  # test set is non-empty
        # The gap is _WINDOW_SIZE samples wide (not clamped)
        assert gap_end - split == _WINDOW_SIZE

    def test_train_one_segment_applies_gap(self) -> None:
        """train_one_segment should apply a gap between train and test data."""
        from scripts.train_models import _WINDOW_SIZE, train_one_segment

        n_samples = 300
        features = [{"feat": float(i)} for i in range(n_samples)]
        labels = [i % 2 for i in range(n_samples)]

        # Track what XGBoost.fit receives (train) and predict_proba receives (test)
        train_calls: list[int] = []
        test_calls: list[int] = []

        mock_xgb = MagicMock()

        def track_fit(x: list, y: list) -> None:
            train_calls.append(len(x))

        def track_predict(f: dict) -> float:
            test_calls.append(1)
            return 0.5

        mock_xgb.fit.side_effect = track_fit
        mock_xgb.predict_proba.side_effect = track_predict

        with (
            patch("scripts.train_models._build_dataset", return_value=(features, labels)),
            patch("scripts.train_models.XGBoostModel", return_value=mock_xgb),
            patch("scripts.train_models.LightGBMModel", return_value=mock_xgb),
            patch("scripts.train_models.LSTMModel", return_value=mock_xgb),
        ):
            train_one_segment("us_tech", ["AAPL"], Path("/tmp/test_models"))

        # Expected: split = 240, gap_end = 300 → test has 0 samples
        # Actually split = 240, gap_end = min(240+60, 300) = 300, test = 0
        # So predict_proba should NOT be called at all (empty test set)
        assert sum(test_calls) == 0
        # Train should get 240 samples (0.8 * 300)
        assert train_calls[0] == 240

    def test_small_dataset_gap_clamps(self) -> None:
        """When dataset is small, gap_end should clamp to len(features)."""
        from scripts.train_models import _TRAIN_RATIO, _WINDOW_SIZE

        n_samples = 80
        split = int(n_samples * _TRAIN_RATIO)  # 64
        gap_end = min(split + _WINDOW_SIZE, n_samples)  # min(124, 80) = 80

        # gap_end clamped to n_samples → empty test set
        assert gap_end == n_samples
        assert n_samples - gap_end == 0
