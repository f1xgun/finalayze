"""Unit tests for LSTMModel."""

from __future__ import annotations

import threading
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
# Small dataset constants for fast tests (#175)
N_SMALL_SAMPLES = 10
SMALL_SEQ_LEN = 3
SMALL_N_FEATURES = 5
N_CONCURRENT_THREADS = 8
N_PREDICTIONS_PER_THREAD = 5


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
        model._feature_buffers.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        model._feature_buffers.clear()  # noqa: SLF001
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
        model._feature_buffers.clear()  # noqa: SLF001
        model2._feature_buffers.clear()  # noqa: SLF001
        r1 = model.predict_proba(features)
        r2 = model2.predict_proba(features)
        assert r1 == pytest.approx(r2, abs=1e-5)

    def test_save_creates_companion_scaler_file(self, tmp_path: Path) -> None:
        """save() must write a companion .scaler.pkl file alongside the checkpoint (#152/#169)."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        X, y = _make_dataset()
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)

        scaler_path = tmp_path / "lstm.pkl.scaler.pkl"
        assert scaler_path.exists()

    def test_load_uses_weights_only_safe_deserialization(self, tmp_path: Path) -> None:
        """load() must use weights_only=True for the torch checkpoint (#169)."""
        from unittest import mock

        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
        X, y = _make_dataset()
        model.fit(X, y)

        save_path = tmp_path / "lstm.pkl"
        model.save(save_path)

        # Patch torch.load to verify it is called with weights_only=True
        import torch

        original_load = torch.load

        calls: list[dict[str, object]] = []

        def _patched_load(*args: object, **kwargs: object) -> object:
            calls.append({"args": args, "kwargs": kwargs})
            return original_load(*args, **kwargs)

        with mock.patch("torch.load", side_effect=_patched_load):
            model2 = LSTMModel(segment_id="us_tech", sequence_length=SEQUENCE_LENGTH)
            model2.load(save_path)

        assert len(calls) >= 1
        # weights_only=True must be passed
        assert calls[0]["kwargs"].get("weights_only") is True


# ── #175: Small-data LSTM tests ──────────────────────────────────────────────


def _make_small_dataset() -> tuple[list[dict[str, float]], list[int]]:
    """Create a minimal synthetic dataset for fast LSTM tests (#175)."""
    rng = np.random.default_rng(42)
    X = [  # noqa: N806
        {f"f{i}": float(rng.standard_normal()) for i in range(SMALL_N_FEATURES)}
        for _ in range(N_SMALL_SAMPLES)
    ]
    y = [int(rng.integers(0, 2)) for _ in range(N_SMALL_SAMPLES)]
    return X, y


@pytest.mark.unit
class TestLSTMModelSmallData:
    """Fast tests using a tiny synthetic dataset (#175)."""

    def test_fit_trains_without_error_small_data(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="seg",
            sequence_length=SMALL_SEQ_LEN,
            hidden_size=8,
            num_layers=1,
        )
        X, y = _make_small_dataset()
        model.fit(X, y)
        assert model._trained is True  # noqa: SLF001

    def test_predict_proba_after_fit_in_zero_one(self) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="seg",
            sequence_length=SMALL_SEQ_LEN,
            hidden_size=8,
            num_layers=1,
        )
        X, y = _make_small_dataset()
        model.fit(X, y)
        result = model.predict_proba(X[0])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_save_load_roundtrip_small_data(self, tmp_path: Path) -> None:
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="seg",
            sequence_length=SMALL_SEQ_LEN,
            hidden_size=8,
            num_layers=1,
        )
        X, y = _make_small_dataset()
        model.fit(X, y)

        save_path = tmp_path / "small_lstm.pkl"
        model.save(save_path)

        model2 = LSTMModel(segment_id="seg", sequence_length=SMALL_SEQ_LEN)
        model2.load(save_path)

        model._feature_buffers.clear()  # noqa: SLF001
        model2._feature_buffers.clear()  # noqa: SLF001

        r1 = model.predict_proba(X[0])
        r2 = model2.predict_proba(X[0])
        assert r1 == pytest.approx(r2, abs=1e-4)

    def test_scaler_applied_consistently_after_load(self, tmp_path: Path) -> None:
        """Loaded model must apply the same scaler as the original (#152)."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="seg",
            sequence_length=SMALL_SEQ_LEN,
            hidden_size=8,
            num_layers=1,
        )
        X, y = _make_small_dataset()
        model.fit(X, y)
        assert model._scaler is not None  # noqa: SLF001

        save_path = tmp_path / "scaler_test.pkl"
        model.save(save_path)

        model2 = LSTMModel(segment_id="seg", sequence_length=SMALL_SEQ_LEN)
        model2.load(save_path)
        assert model2._scaler is not None  # noqa: SLF001


# ── #138: Buffer copy under lock ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.slow
class TestLSTMBufferCopyUnderLock:
    """predict_proba must not mutate the shared buffer when padding (#138)."""

    def test_concurrent_predict_proba_does_not_corrupt_buffer(self) -> None:
        """Multiple threads calling predict_proba concurrently must not cause data races."""
        from finalayze.ml.models.lstm_model import LSTMModel

        model = LSTMModel(
            segment_id="seg",
            sequence_length=SMALL_SEQ_LEN,
            hidden_size=8,
            num_layers=1,
        )
        X, y = _make_small_dataset()
        model.fit(X, y)

        errors: list[Exception] = []
        results: list[float] = []
        lock = threading.Lock()

        def _predict() -> None:
            try:
                for _ in range(N_PREDICTIONS_PER_THREAD):
                    r = model.predict_proba(X[0])
                    with lock:
                        results.append(r)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_predict) for _ in range(N_CONCURRENT_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent predict_proba raised: {errors}"
        assert len(results) == N_CONCURRENT_THREADS * N_PREDICTIONS_PER_THREAD
        # All returned probabilities must be in [0, 1]
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_buffer_size_never_exceeds_sequence_length_under_concurrent_access(self) -> None:
        """Per-symbol buffers must never grow beyond sequence_length (#138)."""
        from finalayze.ml.models.lstm_model import LSTMModel

        seq_len = SMALL_SEQ_LEN
        model = LSTMModel(segment_id="seg", sequence_length=seq_len, hidden_size=8, num_layers=1)
        X, y = _make_small_dataset()
        model.fit(X, y)

        def _predict_many() -> None:
            for _ in range(N_PREDICTIONS_PER_THREAD):
                model.predict_proba(X[0])

        threads = [threading.Thread(target=_predict_many) for _ in range(N_CONCURRENT_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each per-symbol buffer is a deque with maxlen — size must never exceed seq_len
        for buf in model._feature_buffers.values():  # noqa: SLF001
            assert len(buf) <= seq_len
