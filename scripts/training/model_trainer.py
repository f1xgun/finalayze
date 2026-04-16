"""Model training and evaluation for the training pipeline.

Handles training XGBoost, LightGBM, and CatBoost models, evaluating them,
and saving results for a single segment (non-walk-forward path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.training.calibration import fit_and_save_calibrator, fit_and_save_meta_learner
from scripts.training.data_loader import is_moex_segment
from scripts.training.dataset_builder import (
    _WINDOW_SIZE,
    LABEL_MODE_TRIPLE_BARRIER,
    PURGE_GAP,
    build_dataset,
    compute_uniqueness_from_hold_bars,
)
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training.sample_weights import compute_decay_weights

if TYPE_CHECKING:
    import numpy as np
    from config.settings import Settings

    from finalayze.core.schemas import MarketContext

# Split ratios
TRAIN_RATIO = 0.70
CALIBRATION_RATIO = 0.15
TEST_RATIO = 0.15

_TUNED_PARAMS_DIR = Path(__file__).parent.parent.parent / "results" / "tuned_params"

# XGBoost max_depth: shallower for MOEX (smaller dataset, prevent overfit)
_US_MAX_DEPTH = 5
_MOEX_MAX_DEPTH = 3

# CatBoost depth: shallower for MOEX (smaller dataset)
_US_CATBOOST_DEPTH = 4
_MOEX_CATBOOST_DEPTH = 3

# MOEX-specific ensemble hyperparameters: more trees + lower LR for small dataset
MOEX_N_ESTIMATORS = 300
MOEX_LEARNING_RATE = 0.03

# MI feature selection: fewer features for MOEX (smaller dataset, 50:1 sample-to-feature ratio)
_US_MAX_FEATURES = 15
_MOEX_MAX_FEATURES = 10

# Feature selection mode choices
FEAT_SEL_MI = "mi"
FEAT_SEL_EFFICIENT = "efficient"

# Model key normalization for weights
_MODEL_KEY_NORMALIZE: dict[str, str] = {
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "catboost": "catboost",
}


def load_tuned_params(segment_id: str, model_type: str) -> dict | None:
    """Load Optuna-tuned params if available, else return None."""
    path = _TUNED_PARAMS_DIR / segment_id / f"{model_type}.json"
    if path.exists():
        with path.open() as f:
            return json.loads(f.read())
    return None


def get_max_features(segment_id: str) -> int:
    """Return max MI-selected features: 10 for MOEX, 15 for US."""
    return _MOEX_MAX_FEATURES if is_moex_segment(segment_id) else _US_MAX_FEATURES


def get_xgboost_max_depth(segment_id: str) -> int:
    """Return XGBoost max_depth: 3 for MOEX, 5 for US."""
    return _MOEX_MAX_DEPTH if is_moex_segment(segment_id) else _US_MAX_DEPTH


def get_catboost_depth(segment_id: str) -> int:
    """Return CatBoost depth: 3 for MOEX, 4 for US."""
    return _MOEX_CATBOOST_DEPTH if is_moex_segment(segment_id) else _US_CATBOOST_DEPTH


def select_features(
    train_df: object,
    train_series: object,
    max_feats: int,
    mode: str = FEAT_SEL_EFFICIENT,
) -> list[str]:
    """Dispatch feature selection by mode.

    Args use ``object`` to avoid top-level pandas import; callers already pass
    ``pd.DataFrame`` / ``pd.Series``.
    """
    import pandas as _pd  # noqa: PLC0415

    from finalayze.ml.training.feature_selection import (  # noqa: PLC0415
        select_features_efficient,
        select_features_mi,
    )

    df = _pd.DataFrame(train_df) if not isinstance(train_df, _pd.DataFrame) else train_df
    s = _pd.Series(train_series) if not isinstance(train_series, _pd.Series) else train_series
    if mode == FEAT_SEL_EFFICIENT:
        return select_features_efficient(df, s, max_features=max_feats)
    return select_features_mi(df, s, max_features=max_feats)


def compute_model_weights(
    results: dict[str, str | float],
) -> dict[str, float]:
    """Compute performance-weighted averaging weights from accuracy results.

    Weight = max(0, accuracy - 0.50)^2. Auto-excludes coin-flip models.

    Accepts either pre-computed accuracy floats or formatted result strings
    (e.g. ``"acc=0.620 brier=0.230 logloss=0.650"``).
    """
    weights: dict[str, float] = {}
    for name, result in results.items():
        if isinstance(result, (int, float)):
            acc = float(result)
        else:
            acc = 0.5
            for part in result.split():
                if part.startswith("acc="):
                    acc = float(part.split("=")[1])
        normalized = _MODEL_KEY_NORMALIZE.get(name.lower(), name.lower())
        weights[normalized] = max(0.0, acc - 0.50) ** 2
    return weights


def evaluate_model(
    model: XGBoostModel | LightGBMModel | CatBoostModel,
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> str:
    """Evaluate a model and return a formatted summary string."""
    probas = [model.predict_proba(f) for f in test_features]
    preds = [round(p) for p in probas]
    acc = float(accuracy_score(test_labels, preds))
    brier = float(brier_score_loss(test_labels, probas))
    ll = float(log_loss(test_labels, probas, labels=[0, 1]))
    return f"acc={acc:.3f} brier={brier:.3f} logloss={ll:.3f}"


def train_and_evaluate_models(  # noqa: PLR0912, PLR0915
    segment_id: str,
    segment_dir: Path,
    train_features: list[dict[str, float]],
    train_labels: list[int],
    cal_features: list[dict[str, float]],
    cal_labels: list[int],
    test_features: list[dict[str, float]],
    test_labels: list[int],
    sample_weights: np.ndarray | None = None,  # type: ignore[type-arg]
) -> dict[str, str]:
    """Train XGBoost, LightGBM, and CatBoost; return evaluation results."""
    results: dict[str, str] = {}

    # Check for Optuna-tuned hyperparameters
    xgb_tuned = load_tuned_params(segment_id, "xgboost")
    lgbm_tuned = load_tuned_params(segment_id, "lightgbm")
    if xgb_tuned:
        print(f"[{segment_id}] Using tuned XGBoost params: {xgb_tuned}")
    if lgbm_tuned:
        print(f"[{segment_id}] Using tuned LightGBM params: {lgbm_tuned}")

    default_depth = get_xgboost_max_depth(segment_id)
    xgb_kwargs: dict[str, int | float] = {"max_depth": default_depth}
    if xgb_tuned:
        xgb_kwargs["max_depth"] = xgb_tuned.get("max_depth", default_depth)
        for key in (
            "n_estimators",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ):
            if key in xgb_tuned:
                xgb_kwargs[key] = xgb_tuned[key]
    elif is_moex_segment(segment_id):
        xgb_kwargs["n_estimators"] = MOEX_N_ESTIMATORS
        xgb_kwargs["learning_rate"] = MOEX_LEARNING_RATE
    xgb = XGBoostModel(segment_id=segment_id, **xgb_kwargs)  # type: ignore[arg-type]
    xgb.fit(train_features, train_labels, sample_weight=sample_weights)
    xgb.save(segment_dir / "xgb.pkl")

    # Log top-10 feature importances from XGBoost
    if xgb._model is not None and xgb._feature_names is not None:
        importances = xgb._model.feature_importances_
        feat_imp = sorted(
            zip(xgb._feature_names, importances, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"[{segment_id}] XGBoost top-10 feature importances:")
        for name, score in feat_imp[:10]:
            print(f"  {name:>25s}: {score:.4f}")

    if test_features:
        results["XGB"] = evaluate_model(xgb, test_features, test_labels)

    lgbm_kwargs: dict[str, int | float] = {}
    if lgbm_tuned:
        for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
        ):
            if key in lgbm_tuned:
                lgbm_kwargs[key] = lgbm_tuned[key]
        # LightGBM uses feature_fraction/bagging_fraction in Optuna search space
        if "feature_fraction" in lgbm_tuned:
            lgbm_kwargs["colsample_bytree"] = lgbm_tuned["feature_fraction"]
        if "bagging_fraction" in lgbm_tuned:
            lgbm_kwargs["subsample"] = lgbm_tuned["bagging_fraction"]
    elif is_moex_segment(segment_id):
        lgbm_kwargs["n_estimators"] = MOEX_N_ESTIMATORS
        lgbm_kwargs["learning_rate"] = MOEX_LEARNING_RATE
    lgbm = LightGBMModel(segment_id=segment_id, **lgbm_kwargs)  # type: ignore[arg-type]
    lgbm.fit(train_features, train_labels, sample_weight=sample_weights)
    lgbm.save(segment_dir / "lgbm.pkl")
    if test_features:
        results["LGBM"] = evaluate_model(lgbm, test_features, test_labels)

    catboost_depth = get_catboost_depth(segment_id)
    catboost = CatBoostModel(segment_id=segment_id, depth=catboost_depth)
    catboost.fit(train_features, train_labels, sample_weight=sample_weights)
    catboost.save(segment_dir / "catboost.pkl")
    if test_features:
        results["CATBOOST"] = evaluate_model(catboost, test_features, test_labels)

    # Fit EnsembleCalibrator on CALIBRATION set raw probabilities (out-of-sample)
    if cal_features and cal_labels:
        fit_and_save_calibrator(
            segment_id,
            segment_dir,
            [xgb, lgbm, catboost],
            cal_features,
            cal_labels,
        )

    # Fit stacking meta-learner on TEST set OOF predictions (out-of-sample)
    if test_features and test_labels:
        fit_and_save_meta_learner(
            segment_id,
            segment_dir,
            [xgb, lgbm, catboost],
            test_features,
            test_labels,
        )

    return results


def train_one_segment(  # noqa: PLR0915
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
    seq_bootstrap: bool = True,
    market_context: MarketContext | None = None,
    feat_sel_mode: str = FEAT_SEL_EFFICIENT,
) -> None:
    """Train and save models for a single segment.

    When market_context is provided, ambient MOEX/cross-asset data is sliced per
    training window to prevent look-ahead bias.
    """
    import numpy as _np  # noqa: PLC0415
    from config.settings import Settings as _Settings  # noqa: PLC0415

    from finalayze.ml.training.sample_weights import sequential_bootstrap  # noqa: PLC0415

    if settings is None:
        settings = _Settings()
    print(f"\n[{segment_id}] Fetching candles for {symbols} (label_mode={label_mode})...")

    features_list, label_list, barrier_weights, hold_bars = build_dataset(
        segment_id,
        symbols,
        settings,
        label_mode=label_mode,
        excess_returns=excess_returns,
        market_context=market_context,
    )
    if not features_list:
        print(f"[{segment_id}] No samples -- skipping.")
        return

    if len(features_list) < _WINDOW_SIZE:
        print(f"[{segment_id}] Only {len(features_list)} samples, need {_WINDOW_SIZE}+, skipping.")
        return

    print(
        f"[{segment_id}] Total samples: {len(features_list)} "
        f"(label balance: {sum(label_list)}/{len(label_list)} positive)"
    )

    # Three-way temporal split: train / calibration / test
    # Purge gaps between each split prevent label leakage
    n = len(features_list)
    train_end = int(n * TRAIN_RATIO)
    cal_start = min(train_end + PURGE_GAP, n)
    cal_end = int(n * (TRAIN_RATIO + CALIBRATION_RATIO))
    test_start = min(cal_end + PURGE_GAP, n)

    train_features = features_list[:train_end]
    train_labels = label_list[:train_end]
    cal_features = features_list[cal_start:cal_end]
    cal_labels = label_list[cal_start:cal_end]
    test_features = features_list[test_start:]
    test_labels = label_list[test_start:]

    print(
        f"[{segment_id}] Split: train={len(train_features)}, "
        f"cal={len(cal_features)}, test={len(test_features)} "
        f"(purge_gap={PURGE_GAP})"
    )

    # Feature selection on TRAIN data only (no leakage -- design doc 2.3)
    selected_features: list[str] | None = None
    if train_features:
        import pandas as pd  # noqa: PLC0415

        feature_names = sorted(train_features[0].keys())
        train_df = pd.DataFrame(train_features)
        train_series = pd.Series(train_labels)
        max_feats = get_max_features(segment_id)
        selected_features = select_features(train_df, train_series, max_feats, mode=feat_sel_mode)
        if selected_features:
            print(
                f"[{segment_id}] Selected {len(selected_features)}/{len(feature_names)} "
                f"features (max={max_feats})"
            )
            train_features = [{k: row[k] for k in selected_features} for row in train_features]
            cal_features = [{k: row[k] for k in selected_features} for row in cal_features]
            test_features = [{k: row[k] for k in selected_features} for row in test_features]

    # Compute sample weights: combine decay, uniqueness, and barrier weights
    decay_weights = compute_decay_weights(len(train_features))

    # Uniqueness from overlapping labels (A6)
    if hold_bars is not None and len(hold_bars) >= train_end:
        train_hold_bars = hold_bars[:train_end]
        uniqueness = compute_uniqueness_from_hold_bars(train_hold_bars)
        # Normalize to mean=1
        u_mean = float(uniqueness.mean()) if len(uniqueness) > 0 else 1.0
        uniqueness = uniqueness / u_mean if u_mean > 0 else uniqueness
    else:
        uniqueness = _np.ones(len(train_features), dtype=_np.float64)

    # Barrier weights: use sqrt to dampen extreme PnL values (A6)
    if barrier_weights is not None and len(barrier_weights) > 0:
        train_bw = barrier_weights[:train_end]
        dampened_bw = _np.sqrt(_np.abs(train_bw))
        bw_mean = float(dampened_bw.mean()) if len(dampened_bw) > 0 else 1.0
        normalized_bw = dampened_bw / bw_mean if bw_mean > 0 else dampened_bw
    else:
        normalized_bw = _np.ones(len(train_features), dtype=_np.float64)

    sample_weights = decay_weights * uniqueness * normalized_bw

    # Sequential bootstrapping: debias overlapping labels (AFML Ch. 4)
    if seq_bootstrap and hold_bars is not None and len(hold_bars) >= train_end:
        train_hold_bars_sb = hold_bars[:train_end]
        sb_starts = _np.arange(len(train_features), dtype=_np.int64)
        sb_holds = _np.array(train_hold_bars_sb[: len(train_features)], dtype=_np.int64)
        sb_n = len(train_features)
        sb_indices = sequential_bootstrap(sb_starts, sb_holds, sb_n)
        train_features = [train_features[i] for i in sb_indices]
        train_labels = [train_labels[i] for i in sb_indices]
        sample_weights = sample_weights[sb_indices]
        print(f"[{segment_id}] Sequential bootstrap: {sb_n} draws, {len(set(sb_indices))} unique")

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Persist MI-selected features for inference-time filtering (feature mismatch fix)
    if selected_features:
        features_path = segment_dir / "selected_features.json"
        features_path.write_text(json.dumps(selected_features))
        print(f"[{segment_id}] Saved selected_features.json ({len(selected_features)} features)")

    results = train_and_evaluate_models(
        segment_id,
        segment_dir,
        train_features,
        train_labels,
        cal_features,
        cal_labels,
        test_features,
        test_labels,
        sample_weights=sample_weights,
    )
    summary = " | ".join(f"{k}: {v}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")

    # Compute and save performance-weighted model weights
    model_weights = compute_model_weights(results)
    weights_path = segment_dir / "model_weights.json"
    weights_path.write_text(json.dumps(model_weights, indent=2))
    print(f"[{segment_id}] Saved model_weights.json: {model_weights}")

    # Compute and save base_rate from training label distribution
    positive_count = sum(1 for y in train_labels if y > 0)
    base_rate = positive_count / len(train_labels) if len(train_labels) > 0 else 0.50
    meta = {"base_rate": round(base_rate, 4)}
    meta_path = segment_dir / "segment_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{segment_id}] Saved segment_meta.json: base_rate={base_rate:.4f}")
