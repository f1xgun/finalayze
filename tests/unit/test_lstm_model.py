"""Unit tests for LSTMModel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finalayze.core.exceptions import InsufficientDataError

# Constants — no magic numbers
N_TRAIN_SAMPLES = 60
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 2
N_FEATURES = 6
HALF_PROB = 0.5


def _make_feature_dict(seed: int = 0) -> dict[str, float]:
    """Return a synthetic feature dict with N_FEATURES keys."""
    rng = np.random.default_rng(seed)
    return {f"feat_{i:02d}": float(rng.standard_normal()) for i in range(N_FEATURES)}


def _make_dataset(
    n: int = N_TRAIN_SAMPLES,
) -> tuple[list[dict[str, float]], list[int]]:
    rng = np.random.default_rng(1)
    X = [_make_feature_dict(i) for i in range(n)]
    y = [int(rng.integers(0, 2)) for _ in range(n)]
    return X, y


@pytest.mark.unit
class TestLSTMModelUntrained:
    def test_untrained_returns_half(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        result = model.predict_proba(_make_feature_dict())
        assert result == pytest.approx(HALF_PROB)

    def test_untrained_is_false(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        assert model._trained is False  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.slow
class TestLSTMModelFit:
    def test_fit_trains_without_error(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)
        assert model._trained is True  # noqa: SLF001

    def test_fit_insufficient_data_raises(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        X, y = _make_dataset(n=SEQUENCE_LENGTH - 1)  # one short
        with pytest.raises(InsufficientDataError):
            model.fit(X, y)

    def test_predict_after_fit_returns_float_in_range(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)
        result = model.predict_proba(_make_feature_dict())
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_feature_order_consistency(self) -> None:
        """Same features in different dict order produce the same prediction."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)

        features = _make_feature_dict(seed=99)
        features_reversed = dict(reversed(list(features.items())))

        # Flush buffer so both calls start from the same state
        model._feature_buffer.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        model._feature_buffer.clear()  # noqa: SLF001
        r2 = model.predict_proba(features_reversed)
        assert r1 == pytest.approx(r2)


@pytest.mark.unit
@pytest.mark.slow
class TestLSTMModelSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="us_tech",
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        X, y = _make_dataset()
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)
        assert save_path.exists()

        model2 = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        model2.load(save_path)
        assert model2._trained is True  # noqa: SLF001

        features = _make_feature_dict()
        model._feature_buffer.clear()  # noqa: SLF001
        model2._feature_buffer.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        r2 = model2.predict_proba(features)
        assert r1 == pytest.approx(r2, abs=1e-5)
