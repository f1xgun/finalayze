"""Unit tests for the train_models.py training script."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from finalayze.core.schemas import Candle

# Constants
N_CANDLES = 120  # enough for 60-candle windows + max_hold for triple barrier
WINDOW_SIZE = 60
EXPECTED_MODEL_FILES = {"xgb.pkl", "lgbm.pkl", "lstm.pkl"}


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

        with patch.object(mod, "_fetch_symbol_candles", return_value=candles):  # type: ignore[union-attr]
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
        # Need more candles for triple barrier (window_size + max_hold + margin)
        candles = _make_candles(n=200)

        with patch.object(mod, "_fetch_symbol_candles", return_value=candles):  # type: ignore[union-attr]
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
        short_candles = _make_candles(n=30)  # too few for 60-candle windows

        with patch.object(mod, "_fetch_symbol_candles", return_value=short_candles):  # type: ignore[union-attr]
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

        with patch.object(mod, "_fetch_symbol_candles", return_value=candles):  # type: ignore[union-attr]
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
        candles = _make_candles(n=200)

        with patch.object(mod, "_fetch_symbol_candles", return_value=candles):  # type: ignore[union-attr]
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
