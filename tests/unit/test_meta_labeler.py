"""Tests for MetaLabeler (meta-labeling with XGBoost)."""

from __future__ import annotations

import pytest

from finalayze.core.exceptions import InsufficientDataError
from finalayze.ml.meta_labeler import MetaLabeler, MetaSample

_MIN_SAMPLES = 30


def _make_samples(n: int, profitable_ratio: float = 0.6) -> list[MetaSample]:
    """Generate synthetic MetaSamples for testing."""
    samples: list[MetaSample] = []
    for i in range(n):
        profitable = i < int(n * profitable_ratio)
        samples.append(
            MetaSample(
                features={"rsi": float(i % 100), "macd": float(i * 0.1)},
                signal_direction=1.0 if i % 2 == 0 else -1.0,
                strategy_name="momentum" if i % 3 != 0 else "mean_reversion",
                confidence=0.5 + (i % 50) * 0.01,
                profitable=profitable,
            )
        )
    return samples


class TestMetaLabeler:
    """MetaLabeler unit tests."""

    def test_predict_before_fit_returns_half(self) -> None:
        labeler = MetaLabeler()
        sample = MetaSample(
            features={"rsi": 50.0, "macd": 0.1},
            signal_direction=1.0,
            strategy_name="momentum",
            confidence=0.7,
            profitable=None,
        )
        assert labeler.predict_proba(sample) == 0.5

    def test_fit_and_predict_range(self) -> None:
        labeler = MetaLabeler()
        samples = _make_samples(50)
        labeler.fit(samples)

        inference_sample = MetaSample(
            features={"rsi": 45.0, "macd": 0.3},
            signal_direction=1.0,
            strategy_name="momentum",
            confidence=0.6,
            profitable=None,
        )
        prob = labeler.predict_proba(inference_sample)
        assert 0.0 <= prob <= 1.0

    def test_minimum_samples_guard(self) -> None:
        labeler = MetaLabeler()
        samples = _make_samples(_MIN_SAMPLES - 1)
        with pytest.raises(InsufficientDataError, match="at least 30"):
            labeler.fit(samples)

    def test_unknown_strategy_at_inference(self) -> None:
        labeler = MetaLabeler()
        samples = _make_samples(50)
        labeler.fit(samples)

        inference_sample = MetaSample(
            features={"rsi": 45.0, "macd": 0.3},
            signal_direction=1.0,
            strategy_name="totally_new_strategy",
            confidence=0.6,
            profitable=None,
        )
        prob = labeler.predict_proba(inference_sample)
        assert 0.0 <= prob <= 1.0

    def test_is_fitted_property(self) -> None:
        labeler = MetaLabeler()
        assert labeler.is_fitted is False

        samples = _make_samples(50)
        labeler.fit(samples)
        assert labeler.is_fitted is True
