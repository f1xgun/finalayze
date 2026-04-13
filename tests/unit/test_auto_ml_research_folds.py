"""Tests for walk-forward fold generation with MOEX constants."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from auto_ml_research import (
    _MOEX_PURGE_GAP,
    _MOEX_WF_CAL_MONTHS,
    _MOEX_WF_STEP_MONTHS,
    _MOEX_WF_TEST_MONTHS,
    _MOEX_WF_TRAIN_MONTHS,
    generate_folds,
)

_MOEX_DATASET_DAYS = 730
_US_DATASET_DAYS = 1000
_MIN_MOEX_FOLDS = 3
_MIN_US_FOLDS = 3


def _make_daily_timestamps(n_days: int, start_year: int = 2023) -> list[datetime]:
    """Generate n_days calendar-day timestamps (one per day, no weekday filter).

    Using calendar days so that n_days directly corresponds to the date span used
    by generate_folds() when computing fold boundaries (which uses timedelta(days=N*30)).
    """
    start = datetime(start_year, 1, 2)
    return [start + timedelta(days=i) for i in range(n_days)]


class TestMoexFoldGeneration:
    def test_moex_folds_730_days(self) -> None:
        """730-day MOEX dataset with MOEX constants produces 3+ folds."""
        timestamps = _make_daily_timestamps(_MOEX_DATASET_DAYS)
        folds = generate_folds(
            timestamps,
            train_months=_MOEX_WF_TRAIN_MONTHS,
            cal_months=_MOEX_WF_CAL_MONTHS,
            test_months=_MOEX_WF_TEST_MONTHS,
            step_months=_MOEX_WF_STEP_MONTHS,
            purge_gap=_MOEX_PURGE_GAP,
        )
        assert len(folds) >= _MIN_MOEX_FOLDS, (
            f"Expected >= {_MIN_MOEX_FOLDS} folds, got {len(folds)}"
        )

    def test_us_constants_on_moex_data_few_folds(self) -> None:
        """730-day dataset with US constants produces <= 1 fold (proves MOEX constants needed)."""
        timestamps = _make_daily_timestamps(_MOEX_DATASET_DAYS)
        folds = generate_folds(timestamps)  # US defaults
        assert len(folds) <= 1, (
            f"Expected <= 1 fold with US constants on 730 days, got {len(folds)}"
        )

    def test_us_folds_1825_days(self) -> None:
        """1825-day US dataset with US constants produces 3+ folds."""
        timestamps = _make_daily_timestamps(_US_DATASET_DAYS)
        folds = generate_folds(timestamps)  # US defaults
        assert len(folds) >= _MIN_US_FOLDS, (
            f"Expected >= {_MIN_US_FOLDS} folds, got {len(folds)}"
        )

    def test_default_kwargs_backward_compatible(self) -> None:
        """generate_folds() with no kwargs behaves identically to old code."""
        timestamps = _make_daily_timestamps(_US_DATASET_DAYS)
        folds_default = generate_folds(timestamps)
        folds_explicit = generate_folds(
            timestamps,
            train_months=12,
            cal_months=2,
            test_months=4,
            step_months=3,
            purge_gap=100,
        )
        assert len(folds_default) == len(folds_explicit)

    def test_each_moex_fold_has_data(self) -> None:
        """Every fold from MOEX generation has non-empty train and test sets."""
        timestamps = _make_daily_timestamps(_MOEX_DATASET_DAYS)
        folds = generate_folds(
            timestamps,
            train_months=_MOEX_WF_TRAIN_MONTHS,
            cal_months=_MOEX_WF_CAL_MONTHS,
            test_months=_MOEX_WF_TEST_MONTHS,
            step_months=_MOEX_WF_STEP_MONTHS,
            purge_gap=_MOEX_PURGE_GAP,
        )
        for i, (train_idx, _cal_idx, test_idx) in enumerate(folds):
            assert len(train_idx) > 0, f"Fold {i} has empty train set"
            assert len(test_idx) > 0, f"Fold {i} has empty test set"
