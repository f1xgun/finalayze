"""Unit tests for the train_models.py training script."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from finalayze.core.schemas import Candle

# Constants
N_CANDLES = 200  # enough for 80-candle windows + max_hold for triple barrier
WINDOW_SIZE = 80
EXPECTED_MODEL_FILES = {"xgb.pkl", "lgbm.pkl", "catboost.pkl"}


def _make_candles(n: int = N_CANDLES, symbol: str = "AAPL") -> list[Candle]:
    """Build synthetic candle list with enough volatility for triple barrier labels."""
    rng = np.random.default_rng(42)
    prices = 100.0 + rng.standard_normal(n).cumsum()
    base = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id="us",
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
            high=Decimal(str(round(float(prices[i]) * 1.01, 2))),
            low=Decimal(str(round(float(prices[i]) * 0.99, 2))),
            close=Decimal(str(round(float(prices[i]), 2))),
            volume=int(1000 + rng.integers(0, 500)),
        )
        for i in range(n)
    ]


def _load_script_module() -> object:
    """Load scripts/train_models.py as a module without executing __main__."""
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "scripts" / "train_models.py"
    spec = importlib.util.spec_from_file_location("train_models", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.unit
@pytest.mark.slow
class TestTrainModelsScript:
    def test_script_creates_output_files_direction_mode(self, tmp_path: Path) -> None:
        """train_one_segment() produces xgb.pkl, lgbm.pkl, lstm.pkl with direction labels."""
        mod = _load_script_module()
        candles = _make_candles()

        with patch("scripts.training.dataset_builder.fetch_symbol_candles", return_value=candles):
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
                label_mode="direction",
            )

        segment_dir = tmp_path / "us_tech"
        assert segment_dir.is_dir()
        created = {p.name for p in segment_dir.iterdir()}
        assert EXPECTED_MODEL_FILES.issubset(created)

    def test_script_creates_output_files_triple_barrier_mode(self, tmp_path: Path) -> None:
        """train_one_segment() produces model files with triple barrier labels."""
        mod = _load_script_module()
        # Need >= _MIN_HISTORY_DAYS=500 candles to pass the history gate
        candles = _make_candles(n=500)

        with patch("scripts.training.dataset_builder.fetch_symbol_candles", return_value=candles):
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
                label_mode="triple_barrier",
            )

        segment_dir = tmp_path / "us_tech"
        assert segment_dir.is_dir()
        created = {p.name for p in segment_dir.iterdir()}
        assert EXPECTED_MODEL_FILES.issubset(created)

    def test_script_handles_insufficient_candles_gracefully(self, tmp_path: Path) -> None:
        """train_one_segment() skips segments with too few candles without raising."""
        mod = _load_script_module()
        short_candles = _make_candles(n=30)  # too few for 80-candle windows

        with patch(
            "scripts.training.dataset_builder.fetch_symbol_candles",
            return_value=short_candles,
        ):
            # Should complete without raising
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

    def test_parse_args_defaults(self) -> None:
        """CLI defaults: segment=None, output_dir='models/', label_mode='triple_barrier'."""
        mod = _load_script_module()
        args = mod._parse_args([])  # type: ignore[union-attr]
        assert args.segment is None
        assert args.output_dir == "models/"
        assert args.label_mode == "triple_barrier"

    def test_parse_args_with_segment(self) -> None:
        mod = _load_script_module()
        args = mod._parse_args(["--segment", "us_tech", "--output-dir", "/tmp/out"])  # type: ignore[union-attr]
        assert args.segment == "us_tech"
        assert args.output_dir == "/tmp/out"

    def test_parse_args_label_mode_direction(self) -> None:
        """--label-mode direction is accepted."""
        mod = _load_script_module()
        args = mod._parse_args(["--label-mode", "direction"])  # type: ignore[union-attr]
        assert args.label_mode == "direction"

    def test_parse_args_label_mode_triple_barrier(self) -> None:
        """--label-mode triple_barrier is accepted (and is the default)."""
        mod = _load_script_module()
        args = mod._parse_args(["--label-mode", "triple_barrier"])  # type: ignore[union-attr]
        assert args.label_mode == "triple_barrier"

    def test_moex_segment_gets_fewer_max_features(self) -> None:
        """MOEX segments use max_features=10, US segments use 15."""
        mod = _load_script_module()
        assert mod._get_max_features("us_tech") == 15  # type: ignore[union-attr]
        assert mod._get_max_features("ru_blue_chips") == 10  # type: ignore[union-attr]
        assert mod._get_max_features("ru_energy") == 10  # type: ignore[union-attr]
        assert mod._get_max_features("us_broad") == 15  # type: ignore[union-attr]

    def test_moex_segment_uses_atr_uplift(self) -> None:
        """MOEX segments get 1.2x ATR uplift for triple barrier params."""
        mod = _load_script_module()
        us_params = mod._get_triple_barrier_params("us_tech")  # type: ignore[union-attr]
        ru_params = mod._get_triple_barrier_params("ru_blue_chips")  # type: ignore[union-attr]

        us_upper = 2.0
        moex_upper = 2.4  # 2.0 * 1.2

        assert us_params["upper_atr_mult"] == pytest.approx(us_upper)
        assert us_params["lower_atr_mult"] == pytest.approx(us_upper)
        assert ru_params["upper_atr_mult"] == pytest.approx(moex_upper)
        assert ru_params["lower_atr_mult"] == pytest.approx(moex_upper)

    def test_build_dataset_direction_returns_none_weights(self) -> None:
        """Direction mode returns None for barrier_weights."""
        mod = _load_script_module()
        candles = _make_candles()

        with patch("scripts.training.dataset_builder.fetch_symbol_candles", return_value=candles):
            features, _labels, weights, hold_bars = mod._build_dataset(  # type: ignore[union-attr]
                "us_tech",
                ["AAPL"],
                label_mode="direction",
            )
        assert len(features) > 0
        assert weights is None
        assert hold_bars is None

    def test_build_dataset_triple_barrier_returns_weights(self) -> None:
        """Triple barrier mode returns non-None barrier_weights array."""
        mod = _load_script_module()
        # Need >= _MIN_HISTORY_DAYS=500 candles to pass the history gate
        candles = _make_candles(n=500)

        with patch("scripts.training.dataset_builder.fetch_symbol_candles", return_value=candles):
            features, _labels, weights, hold_bars = mod._build_dataset(  # type: ignore[union-attr]
                "us_tech",
                ["AAPL"],
                label_mode="triple_barrier",
            )
        # Triple barrier may produce fewer samples due to noise filtering,
        # but with 200 volatile candles we should get some
        if len(features) > 0:
            assert weights is not None
            assert len(weights) == len(features)
            assert all(w >= 0 for w in weights)
            assert hold_bars is not None
            assert len(hold_bars) == len(features)


@pytest.mark.unit
class TestWalkForwardUsesLastFold:
    """Verify that train_walk_forward always saves the last fold's models,
    not the fold with the highest accuracy (no cherry-picking / selection bias)."""

    def test_last_fold_models_saved_not_best_accuracy(self, tmp_path: Path) -> None:  # noqa: PLR0915
        """Even when an earlier fold has higher accuracy, the last fold's models are saved."""
        mod = _load_script_module()

        # Build 500 synthetic features + labels (enough for 3 folds of 100+ train samples)
        n_samples = 500
        n_features = 5
        feature_names = [f"feat_{i}" for i in range(n_features)]
        features = [
            {name: float(i * 10 + j) for j, name in enumerate(feature_names)}
            for i in range(n_samples)
        ]
        labels = [i % 2 for i in range(n_samples)]
        timestamps = [
            datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_samples)
        ]

        # Define 3 folds; train must have >= 80 samples (_WINDOW_SIZE)
        fold_size = 100
        folds = [
            (
                list(range(0, fold_size)),  # train (100)
                list(range(fold_size, fold_size + 10)),  # cal (10)
                list(range(fold_size + 10, fold_size + 30)),  # test (20)
            ),
            (
                list(range(100, 100 + fold_size)),
                list(range(100 + fold_size, 100 + fold_size + 10)),
                list(range(100 + fold_size + 10, 100 + fold_size + 30)),
            ),
            (
                list(range(200, 200 + fold_size)),
                list(range(200 + fold_size, 200 + fold_size + 10)),
                list(range(200 + fold_size + 10, 200 + fold_size + 30)),
            ),
        ]

        # Create distinguishable mock model triplets for each fold
        fold_model_sets: list[list[MagicMock]] = []
        fold_accuracies = [0.90, 0.60, 0.70]  # Fold 0 has highest accuracy

        for fold_i in range(len(folds)):
            mock_models = []
            for model_name in ["xgb", "lgbm", "cat"]:
                m = MagicMock()
                m._trained = True
                m._model = True
                m.predict_proba.return_value = 0.6
                # Tag each model so we can identify which fold it came from
                m._fold_tag = fold_i
                m._model_name = model_name
                mock_models.append(m)
            fold_model_sets.append(mock_models)

        # Track which fold's models are constructed via side_effect on model classes
        model_call_counts = {"xgb": 0, "lgbm": 0, "cat": 0}

        def make_xgb_factory(fold_models: list[list[MagicMock]]) -> object:
            def factory(*args: object, **kwargs: object) -> MagicMock:
                idx = model_call_counts["xgb"]
                model_call_counts["xgb"] += 1
                return fold_models[min(idx, len(fold_models) - 1)][0]

            return factory

        def make_lgbm_factory(fold_models: list[list[MagicMock]]) -> object:
            def factory(*args: object, **kwargs: object) -> MagicMock:
                idx = model_call_counts["lgbm"]
                model_call_counts["lgbm"] += 1
                return fold_models[min(idx, len(fold_models) - 1)][1]

            return factory

        def make_cat_factory(fold_models: list[list[MagicMock]]) -> object:
            def factory(*args: object, **kwargs: object) -> MagicMock:
                idx = model_call_counts["cat"]
                model_call_counts["cat"] += 1
                return fold_models[min(idx, len(fold_models) - 1)][2]

            return factory

        # Build FoldMetrics with different accuracies per fold
        from finalayze.ml.training.quality_gates import FoldMetrics, QualityGateResult

        fold_metrics_list = [
            FoldMetrics(
                accuracy=acc,
                brier_score=0.20,
                log_loss=0.50,
                n_test=10,
                mean_uniqueness=1.0,
                buy_ratio=0.5,
                sensitivity=0.5,
                specificity=0.5,
                signal_count=10,
            )
            for acc in fold_accuracies
        ]

        eval_call_count = {"idx": 0}

        def mock_evaluate_fold_metrics(
            models: list[object],
            test_features: list[dict[str, float]],
            test_labels: list[int],
            mean_uniqueness: float = 1.0,
            avg_hold_bars: float = 1.0,
            calibrator: object | None = None,
        ) -> FoldMetrics:
            idx = eval_call_count["idx"]
            eval_call_count["idx"] += 1
            return fold_metrics_list[min(idx, len(fold_metrics_list) - 1)]

        # Mock gate results -- all passing
        mock_gate_result = QualityGateResult(
            gate_name="accuracy", passed=True, value=0.70, threshold=0.55
        )

        with (
            patch(
                "scripts.training.walk_forward.build_dataset_with_timestamps",
                return_value=(features, labels, None, None, timestamps),
            ),
            patch(
                "scripts.training.walk_forward.generate_walk_forward_folds",
                return_value=folds,
            ),
            patch(
                "scripts.training.walk_forward.XGBoostModel",
                side_effect=make_xgb_factory(fold_model_sets),
            ),
            patch(
                "scripts.training.walk_forward.LightGBMModel",
                side_effect=make_lgbm_factory(fold_model_sets),
            ),
            patch(
                "scripts.training.walk_forward.CatBoostModel",
                side_effect=make_cat_factory(fold_model_sets),
            ),
            patch(
                "scripts.training.walk_forward.evaluate_fold_metrics",
                side_effect=mock_evaluate_fold_metrics,
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_fold",
                return_value=[mock_gate_result],
            ),
            patch(
                "finalayze.ml.training.quality_gates.evaluate_walk_forward",
                return_value=(True, {"accuracy": 1.0}),
            ),
            patch("scripts.training.walk_forward.select_features", return_value=[]),
        ):
            result = mod.train_walk_forward(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

        assert result is not None

        # The saved models should be from fold 2 (last), not fold 0 (highest accuracy)
        segment_dir = tmp_path / "us_tech"
        assert segment_dir.is_dir()

        # Verify that save() was called on the last fold's models (fold index 2)
        last_fold_models = fold_model_sets[2]
        last_fold_models[0].save.assert_called_once()  # xgb from fold 2
        last_fold_models[1].save.assert_called_once()  # lgbm from fold 2
        last_fold_models[2].save.assert_called_once()  # catboost from fold 2

        # Verify that the first fold's models (highest accuracy=0.90) were NOT saved
        first_fold_models = fold_model_sets[0]
        first_fold_models[0].save.assert_not_called()
        first_fold_models[1].save.assert_not_called()
        first_fold_models[2].save.assert_not_called()

        # Verify gate results JSON records the last fold's accuracy, not the best
        gate_results_path = segment_dir / "wf_gate_results.json"
        assert gate_results_path.exists()
        wf_data = json.loads(gate_results_path.read_text())
        last_fold_accuracy = 0.70
        assert wf_data["best_accuracy"] == pytest.approx(last_fold_accuracy)


# --- Dynamic quality gates ---

# Constants for readability (ruff PLR2004)
_N_SAMPLES_2000 = 2000
_N_SAMPLES_100 = 100
_HOLD_BARS_20 = 20.0
_HOLD_BARS_1 = 1.0
_HOLD_BARS_100 = 100.0
_N_EFF_100 = 100
_N_EFF_2000 = 2000
_N_EFF_5 = 5
_N_EFF_25 = 25
_N_EFF_400 = 400
_ACCURACY_0_60 = 0.60
_ACCURACY_0_56 = 0.56
_ACCURACY_0_75 = 0.75


@pytest.mark.unit
class TestDynamicQualityGates:
    """Quality gates should adjust thresholds based on effective sample size."""

    def test_n_eff_with_overlap(self) -> None:
        """n_eff should be n/hold_bars."""
        mod = _load_script_module()
        assert mod.compute_n_eff(_N_SAMPLES_2000, _HOLD_BARS_20) == _N_EFF_100  # type: ignore[union-attr]
        assert mod.compute_n_eff(_N_SAMPLES_2000, _HOLD_BARS_1) == _N_EFF_2000  # type: ignore[union-attr]
        assert mod.compute_n_eff(_N_SAMPLES_100, _HOLD_BARS_20) == _N_EFF_5  # type: ignore[union-attr]

    def test_accuracy_gate_scales_with_n_eff(self) -> None:
        """Smaller n_eff -> higher accuracy threshold."""
        mod = _load_script_module()

        threshold_small = mod.compute_accuracy_threshold(n_eff=_N_EFF_25)  # type: ignore[union-attr]
        threshold_large = mod.compute_accuracy_threshold(n_eff=_N_EFF_400)  # type: ignore[union-attr]

        assert threshold_small > _ACCURACY_0_60  # ~0.66 for n_eff=25
        assert threshold_large < _ACCURACY_0_56  # ~0.54 for n_eff=400
        assert threshold_small > threshold_large  # Monotonically decreasing

    def test_brier_gate_scales_with_n_eff(self) -> None:
        """Smaller n_eff -> tighter (lower) Brier threshold."""
        mod = _load_script_module()

        brier_small = mod.compute_brier_threshold(n_eff=_N_EFF_25)  # type: ignore[union-attr]
        brier_large = mod.compute_brier_threshold(n_eff=_N_EFF_400)  # type: ignore[union-attr]
        assert brier_small < brier_large

    def test_accuracy_threshold_capped(self) -> None:
        """Accuracy threshold should be capped at 0.75."""
        mod = _load_script_module()
        assert mod.compute_accuracy_threshold(n_eff=_N_EFF_5) <= _ACCURACY_0_75  # type: ignore[union-attr]

    def test_n_eff_floor_at_one(self) -> None:
        """n_eff should never be < 1."""
        mod = _load_script_module()
        assert mod.compute_n_eff(1, _HOLD_BARS_100) >= 1  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Minimum history gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMinHistoryGate:
    def test_min_history_days_constant(self) -> None:
        from scripts.train_models import _MIN_HISTORY_DAYS

        assert _MIN_HISTORY_DAYS == 500
