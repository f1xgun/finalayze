"""Feature selection by importance and correlation deduplication (Layer 3).

Drops low-importance features and deduplicates highly correlated ones
to reduce overfitting and improve model generalization.

See docs/plans/2026-03-02-enhanced-improvement-plan.md, task B.7.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb


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
