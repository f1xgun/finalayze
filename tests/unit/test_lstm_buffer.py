"""Tests for per-symbol LSTM feature buffer isolation (issue 5.6)."""

from __future__ import annotations

import numpy as np
import pytest

from finalayze.ml.models.lstm_model import LSTMModel

_SEQ_LEN = 5
_N_FEATURES = 3
_FEATURE_KEYS = ["f_a", "f_b", "f_c"]


def _make_features(seed: float = 1.0) -> dict[str, float]:
    """Return a feature dict with deterministic values based on seed."""
    return {k: seed * (i + 1) for i, k in enumerate(_FEATURE_KEYS)}


def _make_training_data(
    n_samples: int = 50,
) -> tuple[list[dict[str, float]], list[int]]:
    """Create minimal training data for the LSTM."""
    rng = np.random.default_rng(42)
    X = [{k: float(rng.standard_normal()) for k in _FEATURE_KEYS} for _ in range(n_samples)]
    y = [int(rng.integers(0, 2)) for _ in range(n_samples)]
    return X, y


class TestPerSymbolBuffers:
    """Different symbols must get independent feature buffers."""

    def test_different_symbols_get_independent_buffers(self) -> None:
        """Predictions for symbol A must not be influenced by symbol B data."""
        model = LSTMModel(segment_id="test", sequence_length=_SEQ_LEN)
        X, y = _make_training_data()
        model.fit(X, y)

        features_a = _make_features(seed=1.0)
        features_b = _make_features(seed=100.0)

        # Feed symbol A several times to build its buffer
        for _ in range(_SEQ_LEN):
            model.predict_proba(features_a, symbol="AAPL")

        prob_a_before = model.predict_proba(features_a, symbol="AAPL")

        # Feed symbol B with very different data
        for _ in range(_SEQ_LEN):
            model.predict_proba(features_b, symbol="GOOG")

        # Symbol A prediction must be unchanged (buffer not contaminated)
        prob_a_after = model.predict_proba(features_a, symbol="AAPL")
        assert prob_a_before == pytest.approx(prob_a_after, abs=1e-6), (
            "Symbol A prediction changed after feeding symbol B data — cross-contamination detected"
        )

    def test_symbols_have_separate_buffer_lengths(self) -> None:
        """Each symbol buffer fills independently."""
        model = LSTMModel(segment_id="test", sequence_length=_SEQ_LEN)
        X, y = _make_training_data()
        model.fit(X, y)

        # Feed 3 data points to AAPL only
        for i in range(3):
            model.predict_proba(_make_features(seed=float(i)), symbol="AAPL")

        # GOOG buffer should be empty (length 0), AAPL should have 3
        assert len(model._feature_buffers.get("AAPL", [])) == 3
        assert len(model._feature_buffers.get("GOOG", [])) == 0


class TestBackwardCompatibility:
    """Calling predict_proba without symbol kwarg uses '__default__'."""

    def test_no_symbol_kwarg_uses_default(self) -> None:
        model = LSTMModel(segment_id="test", sequence_length=_SEQ_LEN)
        X, y = _make_training_data()
        model.fit(X, y)

        model.predict_proba(_make_features(seed=1.0))
        assert "__default__" in model._feature_buffers


class TestFitClearsAllBuffers:
    """fit() must reset all per-symbol buffers."""

    def test_fit_clears_all_symbol_buffers(self) -> None:
        model = LSTMModel(segment_id="test", sequence_length=_SEQ_LEN)
        X, y = _make_training_data()
        model.fit(X, y)

        # Populate buffers for two symbols
        model.predict_proba(_make_features(seed=1.0), symbol="AAPL")
        model.predict_proba(_make_features(seed=2.0), symbol="GOOG")
        assert len(model._feature_buffers) == 2  # noqa: PLR2004

        # Re-train — all buffers must be wiped
        model.fit(X, y)
        assert len(model._feature_buffers) == 0
