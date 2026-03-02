"""Tests for Phase 0.6 (temporal ordering) and 0.7 (calibration thresholds)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from finalayze.ml.models.lightgbm_model import (
    _MIN_CALIBRATION_SAMPLES as LGBM_MIN_CAL,
)
from finalayze.ml.models.lightgbm_model import (
    LightGBMModel,
)
from finalayze.ml.models.xgboost_model import (
    _MIN_CALIBRATION_SAMPLES as XGB_MIN_CAL,
)
from finalayze.ml.models.xgboost_model import (
    XGBoostModel,
)
from finalayze.ml.training import build_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RNG_SEED = 42
_N_FEATURES = 5
_EXPECTED_MIN_CALIBRATION = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_candles(
    symbol: str,
    n: int,
    start: datetime,
    *,
    base_price: float = 100.0,
) -> list:
    """Create synthetic Candle objects for a symbol.

    Uses small price changes (0.5% of price) to avoid triggering the
    corporate-action split detector (40% threshold).
    """
    from finalayze.core.schemas import Candle

    rng = np.random.default_rng(_RNG_SEED + hash(symbol) % 1000)
    candles = []
    price = base_price
    for i in range(n):
        ts = start + timedelta(days=i)
        # Small moves: ~0.5% of price to stay far below split threshold
        change = rng.standard_normal() * 0.5
        o = price
        c = price + change
        h = max(o, c) + abs(rng.standard_normal() * 0.3)
        l_val = min(o, c) - abs(rng.standard_normal() * 0.3)  # noqa: E741
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=ts,
                open=round(o, 2),
                high=round(h, 2),
                low=round(l_val, 2),
                close=round(c, 2),
                volume=int(rng.integers(1000, 10000)),
            )
        )
        price = c
    return candles


def _make_synthetic_data(
    n_samples: int = 100,
    n_features: int = _N_FEATURES,
) -> tuple[list[dict[str, float]], list[int]]:
    """Create synthetic training data with separable classes."""
    rng = np.random.default_rng(_RNG_SEED)
    feature_keys = [f"feat_{i}" for i in range(n_features)]
    features: list[dict[str, float]] = []
    labels: list[int] = []
    for i in range(n_samples):
        label = 1 if i % 2 == 0 else 0
        row = {
            k: float(rng.standard_normal() + (1.0 if label == 1 else -1.0)) for k in feature_keys
        }
        features.append(row)
        labels.append(label)
    return features, labels


# ---------------------------------------------------------------------------
# Phase 0.6 — Temporal ordering
# ---------------------------------------------------------------------------
class TestBuildDatasetTemporalOrdering:
    """Verify build_dataset sorts output by timestamp across symbols."""

    def test_build_dataset_sorted_by_timestamp(self) -> None:
        """Multi-symbol build produces temporally ordered output."""
        window_size = 30
        n_candles = 100

        # Symbol A starts earlier, symbol B starts later
        sym_a = _make_candles("AAA", n_candles, datetime(2020, 1, 1, tzinfo=UTC))
        sym_b = _make_candles("BBB", n_candles, datetime(2020, 3, 1, tzinfo=UTC))

        candles_by_symbol = {"AAA": sym_a, "BBB": sym_b}
        _features, _labels, timestamps = build_dataset(candles_by_symbol, window_size)

        assert len(timestamps) > 0
        # Timestamps must be monotonically non-decreasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp at index {i} ({timestamps[i]}) is before "
                f"index {i - 1} ({timestamps[i - 1]})"
            )

    def test_no_future_leakage_multi_symbol(self) -> None:
        """No test sample has a timestamp before any train sample."""
        window_size = 30
        n_candles = 100
        train_ratio = 0.8

        sym_a = _make_candles("AAA", n_candles, datetime(2020, 1, 1, tzinfo=UTC))
        sym_b = _make_candles("BBB", n_candles, datetime(2020, 3, 1, tzinfo=UTC))

        candles_by_symbol = {"AAA": sym_a, "BBB": sym_b}
        _features, _labels, timestamps = build_dataset(candles_by_symbol, window_size)

        n_total = len(timestamps)
        assert n_total > 0
        split = int(n_total * train_ratio)
        assert split > 0
        assert split < n_total

        train_ts = timestamps[:split]
        test_ts = timestamps[split:]

        max_train_ts = max(train_ts)
        for i, ts in enumerate(test_ts):
            assert ts >= max_train_ts, (
                f"Test sample {i} (ts={ts}) is before max train timestamp ({max_train_ts})"
            )


# ---------------------------------------------------------------------------
# Phase 0.7 — Calibration thresholds
# ---------------------------------------------------------------------------
class TestMinCalibrationSamples:
    """Verify _MIN_CALIBRATION_SAMPLES is 50 in both model files."""

    def test_min_calibration_samples_is_50(self) -> None:
        assert XGB_MIN_CAL == _EXPECTED_MIN_CALIBRATION
        assert LGBM_MIN_CAL == _EXPECTED_MIN_CALIBRATION


class TestXGBoostCalibratorSelection:
    """XGBoost uses isotonic for large samples, Platt for small."""

    def test_calibrator_uses_isotonic_when_large_sample(self) -> None:
        """>=50 calibration samples uses IsotonicRegression."""
        # 300 samples => 20% holdout = 60 cal samples (>= 50)
        n_samples = 300
        model = XGBoostModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        assert isinstance(model._calibrator, IsotonicRegression)  # noqa: SLF001

    def test_calibrator_uses_platt_when_small_sample(self) -> None:
        """<50 calibration samples uses LogisticRegression (Platt scaling)."""
        # 60 samples => 20% holdout = 12 cal samples (< 50)
        n_samples = 60
        model = XGBoostModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        assert isinstance(model._calibrator, LogisticRegression)  # noqa: SLF001

    def test_platt_calibrated_proba_in_range(self) -> None:
        """Platt-calibrated probabilities are in [0, 1]."""
        n_samples = 60
        model = XGBoostModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        for sample in features[:10]:
            proba = model.predict_proba(sample)
            assert 0.0 <= proba <= 1.0


class TestLightGBMCalibratorSelection:
    """LightGBM uses isotonic for large samples, Platt for small."""

    def test_calibrator_uses_isotonic_when_large_sample(self) -> None:
        """>=50 calibration samples uses IsotonicRegression."""
        n_samples = 300
        model = LightGBMModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        assert isinstance(model._calibrator, IsotonicRegression)  # noqa: SLF001

    def test_calibrator_uses_platt_when_small_sample(self) -> None:
        """<50 calibration samples uses LogisticRegression (Platt scaling)."""
        n_samples = 60
        model = LightGBMModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        assert isinstance(model._calibrator, LogisticRegression)  # noqa: SLF001

    def test_platt_calibrated_proba_in_range(self) -> None:
        """Platt-calibrated probabilities are in [0, 1]."""
        n_samples = 60
        model = LightGBMModel("test-seg")
        features, labels = _make_synthetic_data(n_samples=n_samples)
        model.fit(features, labels)
        for sample in features[:10]:
            proba = model.predict_proba(sample)
            assert 0.0 <= proba <= 1.0
