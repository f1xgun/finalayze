"""Optuna hyperparameter tuning for XGBoost and LightGBM per segment.

Usage:
    uv run python scripts/tune_hyperparams.py --segment us_tech
    uv run python scripts/tune_hyperparams.py --segment us_tech --n-trials 100
    uv run python scripts/tune_hyperparams.py  # all segments
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

# Ensure src/ and project root are importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# torch must be imported before lightgbm to prevent OpenMP conflicts
import torch  # noqa: F401, I001
import lightgbm as lgb
import optuna
import xgboost as xgb
from sklearn.metrics import brier_score_loss

from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows

optuna.logging.set_verbosity(optuna.logging.WARNING)

_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_DEFAULT_N_TRIALS = 50
_DEFAULT_OUTPUT_DIR = "results/tuned_params"
_LOOKBACK_DAYS = 1825
_MOEX_LOOKBACK_DAYS = 730
_N_FOLDS = 5
_PURGE_WINDOW = 60
_MIN_UNIQUE_CLASSES = 2

_SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "us_tech": ["AAPL", "MSFT", "GOOGL"],
    "us_healthcare": ["JNJ", "PFE", "UNH"],
    "us_finance": ["JPM", "BAC", "GS"],
    "us_broad": ["SPY", "QQQ", "IWM"],
    "us_industrial": ["CAT", "DE", "HON"],
    "ru_blue_chips": ["SBER.ME", "GAZP.ME", "LKOH.ME"],
    "ru_energy": ["NVTK.ME", "ROSN.ME"],
    "ru_tech": ["YNDX.ME"],
    "ru_finance": ["VTBR.ME"],
}


def _xgboost_search_space(trial: optuna.Trial) -> dict:
    """Define XGBoost hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lightgbm_search_space(trial: optuna.Trial) -> dict:
    """Define LightGBM hyperparameter search space."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_dataset(segment_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch data and build feature/label arrays for a segment."""
    is_moex = segment_id.startswith("ru_")
    lookback = _MOEX_LOOKBACK_DAYS if is_moex else _LOOKBACK_DAYS
    symbols = _SEGMENT_SYMBOLS.get(segment_id, [])

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
    fetcher = YFinanceFetcher(market_id="moex" if is_moex else "us")

    rows: list[tuple[datetime, dict[str, float], int]] = []
    for symbol in symbols:
        try:
            candles = fetcher.fetch_candles(symbol, start, end)
            if not candles or len(candles) < _WINDOW_SIZE + 1:
                continue
            feat_list, lbl_list, ts_list = build_windows(candles, _WINDOW_SIZE)
            for ts, feat, lbl in zip(ts_list, feat_list, lbl_list, strict=True):
                rows.append((ts, feat, lbl))
        except Exception as e:
            print(f"  [warn] {symbol}: {e}")
            continue

    if not rows:
        msg = f"No data for segment {segment_id}"
        raise ValueError(msg)

    rows.sort(key=lambda r: r[0])

    # Convert list of dicts to numpy array (all dicts have same keys)
    feature_names = sorted(rows[0][1].keys())
    features = np.array([[r[1][k] for k in feature_names] for r in rows], dtype=float)
    labels = np.array([r[2] for r in rows], dtype=int)
    return features, labels


def _temporal_cv_brier(
    model_type: str,
    params: dict,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Evaluate params using temporal cross-validation with Brier score."""
    n = len(features)
    fold_size = n // (_N_FOLDS + 1)
    scores: list[float] = []

    for fold in range(_N_FOLDS):
        val_start = (fold + 1) * fold_size
        val_end = min(val_start + fold_size, n)
        train_end = val_start - _PURGE_WINDOW

        if train_end <= 0 or val_end <= val_start:
            continue

        x_train = features[:train_end]
        y_train = labels[:train_end]
        x_val = features[val_start:val_end]
        y_val = labels[val_start:val_end]

        train_classes = len(np.unique(y_train))
        val_classes = len(np.unique(y_val))
        if train_classes < _MIN_UNIQUE_CLASSES or val_classes < _MIN_UNIQUE_CLASSES:
            continue

        if model_type == "xgboost":
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            spw = n_neg / n_pos if n_pos > 0 else 1.0
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=spw,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(x_train, y_train)
        else:
            model = lgb.LGBMClassifier(
                **params,
                is_unbalance=True,
                verbosity=-1,
            )
            model.fit(x_train, y_train)

        probas = model.predict_proba(x_val)[:, 1]
        scores.append(brier_score_loss(y_val, probas))

    if not scores:
        return 1.0
    return float(np.mean(scores))


def _save_best_params(
    segment_id: str,
    model_type: str,
    params: dict,
    output_dir: Path,
) -> Path:
    """Save best params as JSON."""
    seg_dir = output_dir / segment_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    path = seg_dir / f"{model_type}.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def _tune_segment(
    segment_id: str,
    model_type: str,
    n_trials: int,
    output_dir: Path,
) -> dict:
    """Run Optuna tuning for one segment and model type."""
    print(f"\n  Tuning {model_type} for {segment_id} ({n_trials} trials)...")
    features, labels = _build_dataset(segment_id)
    print(f"    Dataset: {len(features)} samples, {features.shape[1]} features")

    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            params = _xgboost_search_space(trial)
        else:
            params = _lightgbm_search_space(trial)
        return _temporal_cv_brier(model_type, params, features, labels)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_score = study.best_value
    print(f"    Best Brier score: {best_score:.4f}")
    print(f"    Best params: {best}")

    path = _save_best_params(segment_id, model_type, best, output_dir)
    print(f"    Saved to: {path}")
    return best


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--segment", default=None, help="Single segment to tune")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=_DEFAULT_N_TRIALS)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run Optuna hyperparameter tuning."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    segments = [args.segment] if args.segment else list(_SEGMENT_SYMBOLS.keys())
    model_types = ["xgboost", "lightgbm"] if args.model_type == "both" else [args.model_type]

    print(f"Optuna tuning: {len(segments)} segments x {len(model_types)} model types")
    print(f"  Trials per run: {args.n_trials}")
    print(f"  Output: {output_dir}")

    for segment in segments:
        for mt in model_types:
            try:
                _tune_segment(segment, mt, args.n_trials, output_dir)
            except Exception as e:
                print(f"  [error] {segment}/{mt}: {e}")
                continue


if __name__ == "__main__":
    main()
