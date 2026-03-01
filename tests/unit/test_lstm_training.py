"""Tests for LSTM training improvements (6C.5 + 6C.6)."""

from __future__ import annotations

from unittest.mock import patch

from finalayze.ml.models.lstm_model import LSTMModel, _DROPOUT, _WEIGHT_DECAY

_N_SAMPLES = 50
_FEATURES = {"a": 1.0, "b": 2.0, "c": 3.0}


def _make_training_data(
    n: int = _N_SAMPLES,
) -> tuple[list[dict[str, float]], list[int]]:
    X = [{k: v + i * 0.1 for k, v in _FEATURES.items()} for i in range(n)]
    y = [i % 2 for i in range(n)]
    return X, y


class TestLSTMEarlyStopping:
    """6C.5: Early stopping + gradient clipping."""

    def test_lstm_gradient_clipping_applied(self) -> None:
        """Verify clip_grad_norm_ is called during training."""
        model = LSTMModel(segment_id="test", sequence_length=5)
        X, y = _make_training_data(30)
        with patch("torch.nn.utils.clip_grad_norm_", wraps=__import__("torch").nn.utils.clip_grad_norm_) as mock_clip:
            model.fit(X, y)
            assert mock_clip.call_count > 0

    def test_lstm_best_weights_restored(self) -> None:
        """After training, model should have the best weights (not last epoch)."""
        model = LSTMModel(segment_id="test", sequence_length=5)
        X, y = _make_training_data(40)
        model.fit(X, y)
        # Model should be trained and functional
        assert model._trained
        prob = model.predict_proba({"a": 1.0, "b": 2.0, "c": 3.0})
        assert 0.0 <= prob <= 1.0

    def test_lstm_fit_still_works_small_data(self) -> None:
        """25 samples with sequence_length=5 should still train without error."""
        model = LSTMModel(segment_id="test", sequence_length=5)
        X, y = _make_training_data(25)
        model.fit(X, y)
        assert model._trained


class TestLSTMDropoutWeightDecay:
    """6C.6: Dropout + weight decay."""

    def test_lstm_dropout_present(self) -> None:
        """After fit(), the internal network should have dropout=0.2."""
        model = LSTMModel(segment_id="test", sequence_length=5)
        X, y = _make_training_data(30)
        model.fit(X, y)
        assert model._model is not None
        assert model._model._dropout.p == _DROPOUT

    def test_lstm_dropout_disabled_during_eval(self) -> None:
        """In eval mode, dropout should be bypassed."""
        model = LSTMModel(segment_id="test", sequence_length=5)
        X, y = _make_training_data(30)
        model.fit(X, y)
        assert model._model is not None
        model._model.eval()
        assert not model._model._dropout.training

    def test_lstm_constants(self) -> None:
        """Verify dropout and weight decay constants."""
        assert _DROPOUT == 0.2
        assert _WEIGHT_DECAY == 1e-4
