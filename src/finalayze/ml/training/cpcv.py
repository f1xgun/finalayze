"""Combinatorial Purged Cross-Validation (CPCV) for financial time series (Layer 3).

Generates C(n_groups, n_test_groups) splits with purge gaps to prevent
information leakage between train and test sets.

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import xgboost as xgb

_log = logging.getLogger(__name__)

_MIN_PURGE_WINDOW = 60  # Must be >= feature window size
_MIN_GROUPS = 2
_MIN_CLASSES = 2
_PROBA_THRESHOLD = 0.5
_PERFECT_SHARPE = 10.0


@dataclass(frozen=True)
class CPCVSplit:
    """A single CPCV train/test split with purged boundaries."""

    train_indices: list[int]
    test_indices: list[int]


def generate_cpcv_splits(
    n_samples: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    purge_window: int = 60,
) -> list[CPCVSplit]:
    """Generate C(n_groups, n_test_groups) splits with purge gaps.

    Splits data into n_groups contiguous blocks. For each combination of
    n_test_groups blocks as test set, the remaining blocks form the training
    set with purge_window samples removed at boundaries adjacent to test blocks.

    Args:
        n_samples: Total number of samples.
        n_groups: Number of contiguous groups to split into.
        n_test_groups: Number of groups to use as test in each fold.
        purge_window: Number of samples to purge at train/test boundaries.
            Must be >= 60 (feature window size).

    Returns:
        List of CPCVSplit, one per combination.

    Raises:
        ValueError: If purge_window < _MIN_PURGE_WINDOW or parameters invalid.
    """
    if purge_window < _MIN_PURGE_WINDOW:
        msg = f"purge_window must be >= {_MIN_PURGE_WINDOW}, got {purge_window}"
        raise ValueError(msg)
    if n_test_groups >= n_groups:
        msg = f"n_test_groups ({n_test_groups}) must be < n_groups ({n_groups})"
        raise ValueError(msg)
    if n_groups < _MIN_GROUPS:
        msg = f"n_groups must be >= {_MIN_GROUPS}, got {n_groups}"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    # Split into n_groups contiguous blocks
    group_boundaries = np.array_split(np.arange(n_samples), n_groups)
    groups: list[set[int]] = [set(g.tolist()) for g in group_boundaries]
    group_ranges: list[tuple[int, int]] = [
        (int(g.min()), int(g.max())) for g in group_boundaries if len(g) > 0
    ]

    splits: list[CPCVSplit] = []

    for test_group_indices in combinations(range(n_groups), n_test_groups):
        test_set: set[int] = set()
        for gi in test_group_indices:
            test_set |= groups[gi]

        # Build purge set: indices within purge_window of any test block boundary
        purge_set: set[int] = set()
        for gi in test_group_indices:
            g_min, g_max = group_ranges[gi]
            # Purge before test block
            purge_start = max(0, g_min - purge_window)
            for idx in range(purge_start, g_min):
                purge_set.add(idx)
            # Purge after test block
            purge_end = min(n_samples - 1, g_max + purge_window)
            for idx in range(g_max + 1, purge_end + 1):
                purge_set.add(idx)

        # Train = all indices not in test and not in purge
        train_set = set(range(n_samples)) - test_set - purge_set

        splits.append(
            CPCVSplit(
                train_indices=sorted(train_set),
                test_indices=sorted(test_set),
            )
        )

    return splits


def evaluate_cpcv(
    features: list[dict[str, float]],
    labels: list[int],
    model_class: type | None = None,  # noqa: ARG001
    n_groups: int = 5,
    n_test_groups: int = 2,
    purge_window: int = 60,
) -> dict[str, Any]:
    """Run CPCV and return per-fold metrics.

    Uses XGBoost only for screening (fast evaluation). The model_class parameter
    is accepted for API flexibility but defaults to XGBoost internally.

    Args:
        features: List of feature dicts.
        labels: Binary labels (0/1).
        model_class: Ignored; XGBoost is always used for screening.
        n_groups: Number of groups for CPCV.
        n_test_groups: Number of test groups per fold.
        purge_window: Purge window size (must be >= 60).

    Returns:
        Dict with keys: fold_sharpes, median_sharpe, negative_folds_pct, accepted.
        Rejection criteria: median_sharpe < 0.3 OR negative_folds_pct > 0.40
    """
    splits = generate_cpcv_splits(
        n_samples=len(features),
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_window=purge_window,
    )

    if not features:
        return {
            "fold_sharpes": [],
            "median_sharpe": 0.0,
            "negative_folds_pct": 1.0,
            "accepted": False,
        }

    feature_names = sorted(features[0].keys())
    x_arr = np.array([[row[k] for k in feature_names] for row in features], dtype=float)
    y_arr = np.array(labels, dtype=int)

    fold_sharpes: list[float] = []

    for split in splits:
        if not split.train_indices or not split.test_indices:
            fold_sharpes.append(0.0)
            continue

        x_train = x_arr[split.train_indices]
        y_train = y_arr[split.train_indices]
        x_test = x_arr[split.test_indices]
        y_test = y_arr[split.test_indices]

        # Need both classes in training set
        if len(np.unique(y_train)) < _MIN_CLASSES:
            fold_sharpes.append(0.0)
            continue

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train)

        # Compute fold "Sharpe" as proxy: (accuracy - 0.5) / std(predictions)
        # This approximates strategy Sharpe from prediction quality
        probas = model.predict_proba(x_test)[:, 1]
        preds = (probas > _PROBA_THRESHOLD).astype(int)

        # Simulated returns: +1 for correct, -1 for incorrect
        returns = np.where(preds == y_test, 1.0, -1.0)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        if std_ret > 0:
            sharpe = mean_ret / std_ret
        else:
            # All predictions same outcome: positive if all correct, else negative
            sharpe = float(np.sign(mean_ret)) * _PERFECT_SHARPE if mean_ret != 0 else 0.0
        fold_sharpes.append(sharpe)

    median_sharpe = float(np.median(fold_sharpes)) if fold_sharpes else 0.0
    n_negative = sum(1 for s in fold_sharpes if s < 0)
    negative_pct = n_negative / len(fold_sharpes) if fold_sharpes else 1.0

    # Rejection criteria
    accepted = median_sharpe >= 0.3 and negative_pct <= 0.40  # noqa: PLR2004

    return {
        "fold_sharpes": fold_sharpes,
        "median_sharpe": median_sharpe,
        "negative_folds_pct": negative_pct,
        "accepted": accepted,
    }
