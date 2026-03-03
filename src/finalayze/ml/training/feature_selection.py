"""Feature selection by importance and correlation deduplication (Layer 3).

Drops low-importance features and deduplicates highly correlated ones
to reduce overfitting and improve model generalization.

Provides two approaches:
- Pearson correlation-based (original): ``select_features``
- Mutual Information-based: ``select_features_mi``, ``compute_feature_mi``

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _deduplicate_correlated(
    feature_names: list[str],
    corr_matrix: np.ndarray,
    importance_map: dict[str, float],
    threshold: float,
) -> set[str]:
    """Find features to drop based on pairwise correlation.

    For each pair with abs(correlation) > threshold, drops the less important one.
    """
    to_drop: set[str] = set()
    n_feats = len(feature_names)
    for i in range(n_feats):
        if feature_names[i] in to_drop:
            continue
        for j in range(i + 1, n_feats):
            if feature_names[j] in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                imp_i = importance_map.get(feature_names[i], 0.0)
                imp_j = importance_map.get(feature_names[j], 0.0)
                if imp_i >= imp_j:
                    to_drop.add(feature_names[j])
                else:
                    to_drop.add(feature_names[i])
    return to_drop


def select_features(
    features: list[dict[str, float]],
    labels: list[int],
    importance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
) -> tuple[list[dict[str, float]], list[str]]:
    """Select features by importance and deduplication.

    Steps:
        1. Train a quick XGBoost model on the full dataset.
        2. Get feature importances (gain-based).
        3. Drop features with importance < threshold (default 1%).
        4. Among remaining, find pairs with abs(correlation) > 0.85.
        5. Drop the less important feature in each correlated pair.

    Args:
        features: List of feature dicts.
        labels: Binary labels (0/1).
        importance_threshold: Min normalized importance to keep (default 0.01).
        correlation_threshold: Max abs correlation before dedup (default 0.85).

    Returns:
        Tuple of (filtered_features, selected_feature_names).
    """
    if not features:
        return [], []

    feature_names = sorted(features[0].keys())
    x_arr = np.array([[row[k] for k in feature_names] for row in features], dtype=float)
    y_arr = np.array(labels, dtype=int)

    # Step 1: Train quick XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(x_arr, y_arr)

    # Step 2: Get gain-based importances and normalize
    importance_map = _build_importance_map(model, feature_names)

    # Step 3: Drop low-importance features
    important_features = [
        name for name in feature_names if importance_map.get(name, 0.0) >= importance_threshold
    ]

    if not important_features:
        # Keep at least one feature (the most important)
        best = max(feature_names, key=lambda n: importance_map.get(n, 0.0))
        important_features = [best]

    # Step 4-5: Deduplicate correlated features
    important_indices = [feature_names.index(n) for n in important_features]
    x_important = x_arr[:, important_indices]

    if x_important.shape[1] > 1:
        corr_matrix = np.corrcoef(x_important, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    else:
        corr_matrix = np.array([[1.0]])

    to_drop = _deduplicate_correlated(
        important_features, corr_matrix, importance_map, correlation_threshold
    )
    selected = [n for n in important_features if n not in to_drop]

    # Build filtered feature dicts
    filtered = [{k: row[k] for k in selected} for row in features]
    return filtered, selected


# ---------------------------------------------------------------------------
# Mutual Information-based feature selection
# ---------------------------------------------------------------------------

_MI_RANDOM_STATE = 42
_MI_N_NEIGHBORS = 3


def compute_feature_mi(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute Mutual Information between each feature and the target.

    Args:
        x: Feature matrix (n_samples, n_features).
        y: Binary target labels.

    Returns:
        pd.Series indexed by feature names with MI scores (non-negative).
    """
    if x.empty:
        return pd.Series(dtype=float)

    mi_values = mutual_info_classif(
        x.values,
        y.values,
        discrete_features=False,
        random_state=_MI_RANDOM_STATE,
        n_neighbors=_MI_N_NEIGHBORS,
    )
    return pd.Series(mi_values, index=x.columns)


def _pairwise_mi_matrix(x: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
    """Compute pairwise MI between features using discretised target trick.

    Approximates MI(feature_i, feature_j) by treating feature_j as a
    continuous regression target and computing MI via k-NN estimation.
    """
    n_features = x.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for j in range(n_features):
        mi_row = mutual_info_regression(
            x.values,
            x.iloc[:, j].values,
            discrete_features=False,
            random_state=_MI_RANDOM_STATE,
            n_neighbors=_MI_N_NEIGHBORS,
        )
        mi_matrix[:, j] = mi_row

    # Symmetrise: MI(i,j) = (MI_ij + MI_ji) / 2
    return (mi_matrix + mi_matrix.T) / 2.0


def select_features_mi(
    x: pd.DataFrame,
    y: pd.Series,
    max_features: int = 15,
    mi_threshold: float = 0.05,
) -> list[str]:
    """Select features using Mutual Information with greedy deduplication.

    Steps:
        1. Compute MI between each feature and the target.
        2. Remove features with MI < mi_threshold (uninformative).
        3. Among remaining, greedily deduplicate: starting from the highest
           target-MI feature, add features one-by-one; skip a feature if its
           pairwise MI with any already-selected feature exceeds the median
           pairwise MI (i.e. it is redundant).
        4. Return up to max_features by target MI.

    Args:
        x: Feature matrix (n_samples, n_features).
        y: Binary target labels.
        max_features: Maximum number of features to return.
        mi_threshold: Minimum MI with target to keep a feature.

    Returns:
        List of selected feature names, ordered by descending target MI.
    """
    if x.empty or x.shape[1] == 0:
        return []

    # Step 1: compute MI with target
    mi_scores = compute_feature_mi(x, y)

    # Step 2: filter uninformative features
    informative = mi_scores[mi_scores >= mi_threshold]
    if informative.empty:
        return []

    # Sort by MI descending
    informative = informative.sort_values(ascending=False)
    candidates = list(informative.index)

    if len(candidates) <= 1:
        return candidates[:max_features]

    # Step 3: greedy deduplication via pairwise MI
    x_candidates = x[candidates]
    pairwise_mi = _pairwise_mi_matrix(x_candidates)

    # Use median of off-diagonal pairwise MI as redundancy threshold
    n_cand = len(candidates)
    off_diag = [pairwise_mi[i, j] for i in range(n_cand) for j in range(i + 1, n_cand)]
    redundancy_threshold = float(np.median(off_diag)) if off_diag else 0.0

    selected: list[str] = [candidates[0]]  # best feature always selected
    selected_indices: list[int] = [0]

    for idx in range(1, n_cand):
        if len(selected) >= max_features:
            break
        # Check if this candidate is redundant with any already-selected feature
        is_redundant = False
        for sel_idx in selected_indices:
            if pairwise_mi[idx, sel_idx] > redundancy_threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(candidates[idx])
            selected_indices.append(idx)

    return selected[:max_features]


def _build_importance_map(model: xgb.XGBClassifier, feature_names: list[str]) -> dict[str, float]:
    """Extract and normalize gain-based feature importances from XGBoost."""
    raw_importances = model.get_booster().get_score(importance_type="gain")

    importance_map: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        xgb_name = f"f{i}"
        importance_map[name] = raw_importances.get(xgb_name, 0.0)

    total_importance = sum(importance_map.values())
    if total_importance > 0:
        for name in importance_map:
            importance_map[name] /= total_importance

    return importance_map
