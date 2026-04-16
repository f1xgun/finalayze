"""Train XGBoost + LightGBM + CatBoost models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
    uv run python scripts/train_models.py --label-mode direction  # old next-bar labels
    uv run python scripts/train_models.py --label-mode triple_barrier  # default
    uv run python scripts/train_models.py --label-mode trend_scanning  # Prado 2020
    uv run python scripts/train_models.py --walk-forward --force-save  # save despite gate failures

This module is a thin wrapper that delegates to scripts.training.*.
All symbols are re-exported here for backward compatibility with existing
tests and scripts that import from scripts.train_models.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: must happen BEFORE any project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch
from scripts.training.calibration import fit_and_save_calibrator as _fit_and_save_calibrator
from scripts.training.calibration import fit_and_save_meta_learner as _fit_and_save_meta_learner
from scripts.training.cli import DEFAULT_OUTPUT_DIR as _DEFAULT_OUTPUT_DIR
from scripts.training.cli import SEGMENT_SYMBOLS as _SEGMENT_SYMBOLS
from scripts.training.cli import main
from scripts.training.cli import parse_args as _parse_args
from scripts.training.data_loader import LOOKBACK_DAYS as _LOOKBACK_DAYS
from scripts.training.data_loader import MOEX_BENCHMARK as _MOEX_BENCHMARK
from scripts.training.data_loader import MOEX_LOOKBACK_DAYS as _MOEX_LOOKBACK_DAYS
from scripts.training.data_loader import US_BENCHMARK as _US_BENCHMARK
from scripts.training.data_loader import VIX_TICKER as _VIX_TICKER
from scripts.training.data_loader import align_benchmark_candles as _align_benchmark_candles
from scripts.training.data_loader import build_market_data_loader as _build_market_data_loader
from scripts.training.data_loader import fetch_benchmark_candles as _fetch_benchmark_candles
from scripts.training.data_loader import fetch_candles as _fetch_candles
from scripts.training.data_loader import fetch_from_db as _fetch_from_db
from scripts.training.data_loader import fetch_moex_benchmark as _fetch_moex_benchmark
from scripts.training.data_loader import fetch_symbol_candles as _fetch_symbol_candles
from scripts.training.data_loader import fetch_tinkoff_candles as _fetch_tinkoff_candles
from scripts.training.data_loader import fetch_us_benchmark as _fetch_us_benchmark
from scripts.training.data_loader import fetch_vix_candles as _fetch_vix_candles
from scripts.training.data_loader import get_lookback_days as _get_lookback_days
from scripts.training.data_loader import is_moex_segment as _is_moex_segment
from scripts.training.data_loader import orm_to_candle as _orm_to_candle
from scripts.training.dataset_builder import (
    LABEL_MODE_DIRECTION,
    LABEL_MODE_TREND_SCANNING,
    LABEL_MODE_TRIPLE_BARRIER,
    _build_dataset_direction,
)
from scripts.training.dataset_builder import MOEX_ATR_UPLIFT as _MOEX_ATR_UPLIFT
from scripts.training.dataset_builder import PURGE_GAP as _PURGE_GAP
from scripts.training.dataset_builder import SEGMENT_BARRIER_CONFIG as _SEGMENT_BARRIER_CONFIG
from scripts.training.dataset_builder import TB_ATR_PERIOD as _TB_ATR_PERIOD
from scripts.training.dataset_builder import TB_LOWER_ATR_MULT as _TB_LOWER_ATR_MULT
from scripts.training.dataset_builder import TB_MAX_HOLD as _TB_MAX_HOLD
from scripts.training.dataset_builder import TB_UPPER_ATR_MULT as _TB_UPPER_ATR_MULT
from scripts.training.dataset_builder import build_dataset as _build_dataset
from scripts.training.dataset_builder import (
    build_dataset_with_timestamps as _build_dataset_with_timestamps,
)
from scripts.training.dataset_builder import (
    compute_uniqueness_from_hold_bars as _compute_uniqueness_from_hold_bars,
)
from scripts.training.dataset_builder import get_barrier_params as _get_barrier_params
from scripts.training.dataset_builder import (
    get_triple_barrier_params as _get_triple_barrier_params,
)
from scripts.training.model_trainer import (
    _MOEX_CATBOOST_DEPTH,
    _MOEX_MAX_DEPTH,
    _MOEX_MAX_FEATURES,
    _US_CATBOOST_DEPTH,
    _US_MAX_DEPTH,
    _US_MAX_FEATURES,
    FEAT_SEL_EFFICIENT,
    FEAT_SEL_MI,
    train_one_segment,
)
from scripts.training.model_trainer import CALIBRATION_RATIO as _CALIBRATION_RATIO
from scripts.training.model_trainer import MOEX_LEARNING_RATE as _MOEX_LEARNING_RATE
from scripts.training.model_trainer import MOEX_N_ESTIMATORS as _MOEX_N_ESTIMATORS
from scripts.training.model_trainer import TEST_RATIO as _TEST_RATIO
from scripts.training.model_trainer import TRAIN_RATIO as _TRAIN_RATIO
from scripts.training.model_trainer import compute_model_weights as _compute_model_weights
from scripts.training.model_trainer import evaluate_model as _evaluate_model
from scripts.training.model_trainer import get_catboost_depth as _get_catboost_depth
from scripts.training.model_trainer import get_max_features as _get_max_features
from scripts.training.model_trainer import get_xgboost_max_depth as _get_xgboost_max_depth
from scripts.training.model_trainer import load_tuned_params as _load_tuned_params
from scripts.training.model_trainer import select_features as _select_features
from scripts.training.model_trainer import (
    train_and_evaluate_models as _train_and_evaluate_models,
)
from scripts.training.quality import (
    compute_accuracy_threshold,
    compute_brier_threshold,
    compute_n_eff,
)
from scripts.training.walk_forward import BH_FDR as _BH_FDR
from scripts.training.walk_forward import (
    MOEX_MIN_PASSING_FOLDS_RATIO as _MOEX_MIN_PASSING_FOLDS_RATIO,
)
from scripts.training.walk_forward import MOEX_PURGE_GAP as _MOEX_PURGE_GAP
from scripts.training.walk_forward import MOEX_WF_CAL_MONTHS as _MOEX_WF_CAL_MONTHS
from scripts.training.walk_forward import MOEX_WF_STEP_MONTHS as _MOEX_WF_STEP_MONTHS
from scripts.training.walk_forward import MOEX_WF_TEST_MONTHS as _MOEX_WF_TEST_MONTHS
from scripts.training.walk_forward import MOEX_WF_TRAIN_MONTHS as _MOEX_WF_TRAIN_MONTHS
from scripts.training.walk_forward import WF_CAL_MONTHS as _WF_CAL_MONTHS
from scripts.training.walk_forward import WF_STEP_MONTHS as _WF_STEP_MONTHS
from scripts.training.walk_forward import WF_TEST_MONTHS as _WF_TEST_MONTHS
from scripts.training.walk_forward import WF_TRAIN_MONTHS as _WF_TRAIN_MONTHS
from scripts.training.walk_forward import (
    apply_bh_across_segments as _apply_bh_across_segments,
)
from scripts.training.walk_forward import apply_bh_correction as _apply_bh_correction
from scripts.training.walk_forward import evaluate_fold_metrics as _evaluate_fold_metrics
from scripts.training.walk_forward import (
    generate_walk_forward_folds as _generate_walk_forward_folds,
)
from scripts.training.walk_forward import train_walk_forward

# ---------------------------------------------------------------------------
# Re-export everything for backward compatibility.
# Tests and scripts import from scripts.train_models using private names.
# ruff: noqa: F401
# ---------------------------------------------------------------------------
from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training import DEFAULT_WINDOW_SIZE
from finalayze.ml.training.feature_selection import select_features_mi
from finalayze.ml.training.sample_weights import compute_decay_weights

# Constants that don't come from the submodules
_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_MIN_HISTORY_DAYS = 500
_MIN_CANDLES = _WINDOW_SIZE + 1

if __name__ == "__main__":
    main()
