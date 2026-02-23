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
N_CANDLES = 120  # enough for 60-candle windows
WINDOW_SIZE = 60
EXPECTED_MODEL_FILES = {"xgb.pkl", "lgbm.pkl", "lstm.pkl"}


def _make_candles(n: int = N_CANDLES, symbol: str = "AAPL") -> list[Candle]:
    """Build synthetic candle list."""
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
            high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
            low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
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
    def test_script_creates_output_files(self, tmp_path: Path) -> None:
        """train_one_segment() produces xgb.pkl, lgbm.pkl, lstm.pkl."""
        mod = _load_script_module()
        candles = _make_candles()

        with patch.object(mod, "_fetch_candles", return_value=candles):  # type: ignore[union-attr]
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

        segment_dir = tmp_path / "us_tech"
        assert segment_dir.is_dir()
        created = {p.name for p in segment_dir.iterdir()}
        assert EXPECTED_MODEL_FILES.issubset(created)

    def test_script_handles_insufficient_candles_gracefully(self, tmp_path: Path) -> None:
        """train_one_segment() skips segments with too few candles without raising."""
        mod = _load_script_module()
        short_candles = _make_candles(n=30)  # too few for 60-candle windows

        with patch.object(mod, "_fetch_candles", return_value=short_candles):  # type: ignore[union-attr]
            # Should complete without raising
            mod.train_one_segment(  # type: ignore[union-attr]
                segment_id="us_tech",
                symbols=["AAPL"],
                output_dir=tmp_path,
            )

    def test_parse_args_defaults(self) -> None:
        """CLI defaults: segment=None, output_dir='models/'."""
        mod = _load_script_module()
        args = mod._parse_args([])  # type: ignore[union-attr]
        assert args.segment is None
        assert args.output_dir == "models/"

    def test_parse_args_with_segment(self) -> None:
        mod = _load_script_module()
        args = mod._parse_args(["--segment", "us_tech", "--output-dir", "/tmp/out"])  # type: ignore[union-attr]
        assert args.segment == "us_tech"
        assert args.output_dir == "/tmp/out"
