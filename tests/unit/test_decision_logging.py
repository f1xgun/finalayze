"""Tests for PR-2: Decision Logging Enhancement.

Task 2.1 — MomentumStrategy enriched Signal.features + last_skip_reason
Task 2.2 — MeanReversionStrategy enriched Signal.features
Task 2.3 — EnsembleModel.last_model_probas
Task 2.4 — JournalingStrategyCombiner.last_features + model_probas capture
Task 2.5 — DecisionRecord strategy_features / model_probas fields
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VOLUME = 1_000_000
_BASE_DT = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)


def _make_candles(
    prices: list[float],
    *,
    symbol: str = "AAPL",
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> list[Candle]:
    candles: list[Candle] = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        h = Decimal(str(highs[i])) if highs else p + Decimal(1)
        lo = Decimal(str(lows[i])) if lows else p - Decimal(1)
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=_BASE_DT + timedelta(days=i),
                open=p,
                high=h,
                low=lo,
                close=p,
                volume=_VOLUME,
            )
        )
    return candles


# ===== Price sequences that produce BUY / SELL signals =====


def _buy_prices() -> list[float]:
    """Crash + recovery -> BUY signal."""
    stable_price = 200.0
    prices: list[float] = [stable_price] * 40
    crash_bottom = stable_price - 4.0 * 16
    prices.extend([stable_price - 4.0 * (i + 1) for i in range(16)])
    prices.extend([crash_bottom] * 3)
    prices.extend([crash_bottom + 2.0 * (i + 1) for i in range(4)])
    return prices


def _sell_prices() -> list[float]:
    """Rally + decline -> SELL signal."""
    stable_price = 100.0
    rally_top = stable_price + 4.0 * 16
    prices: list[float] = [stable_price] * 40
    prices.extend([stable_price + 4.0 * (i + 1) for i in range(16)])
    prices.extend([rally_top] * 3)
    prices.extend([rally_top - 2.0 * (i + 1) for i in range(7)])
    return prices


_MOMENTUM_PARAMS: dict[str, object] = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "min_confidence": 0.6,
    "macd_hist_lookback": 3,
    "lookback_bars": 5,
    "trend_filter": False,
    "trend_sma_period": 50,
    "trend_sma_buffer_pct": 2.0,
    "adx_filter": False,
    "adx_period": 14,
    "adx_threshold": 25,
    "volume_filter": False,
    "volume_sma_period": 20,
    "volume_min_ratio": 1.0,
    "neutral_reset_bars": 20,
}

_MR_PARAMS: dict[str, object] = {
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "min_confidence": 0.0,
    "squeeze_threshold": 0.0,
    "min_band_distance_pct": 0.0,
    "rsi_oversold_mr": 100,
    "rsi_overbought_mr": 0,
    "rsi_period": 14,
    "exit_at_mean": False,
}


# ===================================================================
# Task 2.1 — MomentumStrategy enriched Signal.features + last_skip_reason
# ===================================================================


class TestMomentumSignalFeatures:
    """Signal.features must contain indicator values for decision logging."""

    def test_buy_signal_features_include_indicators(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A BUY signal must include rsi_value, macd_hist, sma_trend,
        adx_value, volume_ratio in its features dict."""
        from finalayze.strategies.momentum import MomentumStrategy

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        candles = _make_candles(_buy_prices())
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected a BUY signal"
        assert signal.direction == SignalDirection.BUY

        # Required feature keys
        for key in ("rsi_value", "macd_hist", "sma_trend", "adx_value", "volume_ratio"):
            assert key in signal.features, f"Missing feature key: {key}"

        # rsi_value should be a number
        assert isinstance(signal.features["rsi_value"], float)
        # macd_hist should be a number
        assert isinstance(signal.features["macd_hist"], float)

    def test_sell_signal_features_include_indicators(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SELL signal also includes indicator features."""
        from finalayze.strategies.momentum import MomentumStrategy

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        candles = _make_candles(_sell_prices())
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected a SELL signal"
        for key in ("rsi_value", "macd_hist", "sma_trend", "adx_value", "volume_ratio"):
            assert key in signal.features, f"Missing feature key: {key}"

    def test_last_skip_reason_set_on_insufficient_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When returning None due to insufficient data, last_skip_reason is set."""
        from finalayze.strategies.momentum import MomentumStrategy

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        short = _make_candles([100.0] * 5)
        result = strategy.generate_signal("AAPL", short, "us_tech")
        assert result is None
        assert strategy.last_skip_reason is not None
        assert "insufficient" in strategy.last_skip_reason.lower()

    def test_last_skip_reason_none_on_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a signal is produced, last_skip_reason should be None."""
        from finalayze.strategies.momentum import MomentumStrategy

        strategy = MomentumStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MOMENTUM_PARAMS)
        candles = _make_candles(_buy_prices())
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None
        assert strategy.last_skip_reason is None


# ===================================================================
# Task 2.2 — MeanReversionStrategy enriched Signal.features
# ===================================================================


class TestMeanReversionSignalFeatures:
    """Signal.features must contain bb_pct_b, rsi_value, squeeze_active, band_distance."""

    def test_buy_signal_features_include_bb_indicators(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finalayze.strategies.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MR_PARAMS)

        # Stable high then crash below lower band
        prices = [200.0 + (i % 3 - 1) * 0.5 for i in range(25)]
        prices.append(50.0)
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected a BUY signal below lower band"
        assert signal.direction == SignalDirection.BUY

        for key in ("bb_pct_b", "rsi_value", "squeeze_active", "band_distance"):
            assert key in signal.features, f"Missing feature key: {key}"

        # bb_pct_b below 0 for below lower band
        assert signal.features["bb_pct_b"] < 0.0

    def test_sell_signal_features_include_bb_indicators(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finalayze.strategies.mean_reversion import MeanReversionStrategy

        strategy = MeanReversionStrategy()
        monkeypatch.setattr(strategy, "get_parameters", lambda _seg: _MR_PARAMS)

        # Stable then spike above upper band
        prices = [100.0 + (i % 3 - 1) * 0.5 for i in range(25)]
        prices.append(300.0)
        candles = _make_candles(prices)

        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is not None, "Expected a SELL signal above upper band"
        assert signal.direction == SignalDirection.SELL

        for key in ("bb_pct_b", "rsi_value", "squeeze_active", "band_distance"):
            assert key in signal.features, f"Missing feature key: {key}"

        # bb_pct_b > 1.0 for above upper band
        assert signal.features["bb_pct_b"] > 1.0


# ===================================================================
# Task 2.3 — EnsembleModel.last_model_probas
# ===================================================================


class TestEnsembleModelProbas:
    """EnsembleModel must expose per-model probabilities after predict_proba."""

    def _make_model(
        self, *, trained: bool = True, proba: float = 0.7, name: str = "xgboost"
    ) -> MagicMock:
        model = MagicMock()
        model._model = MagicMock() if trained else None
        model.predict_proba.return_value = proba
        type(model).__name__ = name  # type: ignore[assignment]
        return model

    def _make_lstm(self, *, trained: bool = True, proba: float = 0.6) -> MagicMock:
        lstm = MagicMock()
        lstm._trained = trained
        lstm.predict_proba.return_value = proba
        return lstm

    def test_last_model_probas_populated(self) -> None:
        """After predict_proba, last_model_probas has per-model entries."""
        from finalayze.ml.models.ensemble import EnsembleModel

        xgb = self._make_model(proba=0.8, name="XGBoostModel")
        lgbm = self._make_model(proba=0.6, name="LightGBMModel")
        ensemble = EnsembleModel(models=[xgb, lgbm], lstm_model=None)

        result = ensemble.predict_proba({"a": 1.0})
        assert result == pytest.approx(0.7)

        assert hasattr(ensemble, "last_model_probas")
        assert isinstance(ensemble.last_model_probas, dict)
        assert len(ensemble.last_model_probas) == 2
        # Should have entries keyed by model class name
        assert "XGBoostModel" in ensemble.last_model_probas
        assert "LightGBMModel" in ensemble.last_model_probas
        assert ensemble.last_model_probas["XGBoostModel"] == pytest.approx(0.8)
        assert ensemble.last_model_probas["LightGBMModel"] == pytest.approx(0.6)

    def test_last_model_probas_includes_lstm(self) -> None:
        """LSTM probability is also captured in last_model_probas."""
        from finalayze.ml.models.ensemble import EnsembleModel

        xgb = self._make_model(proba=0.8, name="XGBoostModel")
        lstm = self._make_lstm(proba=0.55)
        ensemble = EnsembleModel(models=[xgb], lstm_model=lstm)

        ensemble.predict_proba({"a": 1.0})

        assert "LSTMModel" in ensemble.last_model_probas
        assert ensemble.last_model_probas["LSTMModel"] == pytest.approx(0.55)

    def test_last_model_probas_empty_when_untrained(self) -> None:
        """Untrained models produce empty last_model_probas."""
        from finalayze.ml.models.ensemble import EnsembleModel

        untrained = self._make_model(trained=False, name="XGBoostModel")
        ensemble = EnsembleModel(models=[untrained], lstm_model=None)

        ensemble.predict_proba({"a": 1.0})
        assert ensemble.last_model_probas == {}

    def test_last_model_probas_initial_state(self) -> None:
        """Before calling predict_proba, last_model_probas is empty dict."""
        from finalayze.ml.models.ensemble import EnsembleModel

        xgb = self._make_model(proba=0.8, name="XGBoostModel")
        ensemble = EnsembleModel(models=[xgb], lstm_model=None)
        assert ensemble.last_model_probas == {}


# ===================================================================
# Task 2.4 — JournalingStrategyCombiner.last_features
# ===================================================================


class _MockStrategy:
    """Minimal strategy mock for combiner tests."""

    def __init__(self, name: str, signal: Signal | None) -> None:
        self._name = name
        self._signal = signal

    @property
    def name(self) -> str:
        return self._name

    def supported_segments(self) -> list[str]:
        return ["us_tech"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        return self._signal


class TestJournalingCombinerFeatures:
    """JournalingStrategyCombiner.last_features aggregates strategy features."""

    def _make_signal(
        self,
        name: str,
        direction: SignalDirection = SignalDirection.BUY,
        confidence: float = 0.8,
        features: dict[str, float] | None = None,
    ) -> Signal:
        return Signal(
            strategy_name=name,
            symbol="AAPL",
            market_id="us",
            segment_id="us_tech",
            direction=direction,
            confidence=confidence,
            features=features or {},
            reasoning="test",
        )

    def test_last_features_merges_strategy_features(self) -> None:
        """last_features aggregates features from each strategy, prefixed by name."""
        from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner

        sig_a = self._make_signal("momentum", features={"rsi_value": 25.0, "macd_hist": 0.5})
        sig_b = self._make_signal(
            "mean_reversion",
            direction=SignalDirection.BUY,
            features={"bb_pct_b": -0.1, "band_distance": 0.05},
        )
        strat_a = _MockStrategy("momentum", sig_a)
        strat_b = _MockStrategy("mean_reversion", sig_b)

        combiner = JournalingStrategyCombiner([strat_a, strat_b])  # type: ignore[arg-type]
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
                "mean_reversion": {"enabled": True, "weight": 0.5},
            }
        }
        candles = _make_candles([100.0] * 5)
        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal("AAPL", candles, "us_tech")

        assert hasattr(combiner, "last_features")
        feats = combiner.last_features
        assert isinstance(feats, dict)

        # Features are prefixed by strategy name
        assert "momentum.rsi_value" in feats
        assert feats["momentum.rsi_value"] == pytest.approx(25.0)
        assert "momentum.macd_hist" in feats
        assert "mean_reversion.bb_pct_b" in feats
        assert "mean_reversion.band_distance" in feats

    def test_last_features_empty_when_no_signals(self) -> None:
        """When no strategies fire, last_features is empty."""
        from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner

        strat = _MockStrategy("momentum", None)
        combiner = JournalingStrategyCombiner([strat])  # type: ignore[arg-type]
        config: dict[str, Any] = {
            "strategies": {
                "momentum": {"enabled": True, "weight": 0.5},
            }
        }
        candles = _make_candles([100.0] * 5)
        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal("AAPL", candles, "us_tech")

        assert combiner.last_features == {}

    def test_last_model_probas_captured_from_ml_strategy(self) -> None:
        """When MLStrategy with EnsembleModel is present, capture last_model_probas."""
        from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner

        # Create a mock MLStrategy with a mock ensemble that has last_model_probas
        mock_ml_sig = self._make_signal("ml_ensemble", features={"ml_prob": 0.75})
        ml_strat = _MockStrategy("ml_ensemble", mock_ml_sig)

        # Simulate that MLStrategy has a _registry with get() returning an ensemble
        mock_ensemble = MagicMock()
        mock_ensemble.last_model_probas = {
            "XGBoostModel": 0.8,
            "LightGBMModel": 0.7,
        }
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_ensemble
        ml_strat._registry = mock_registry  # type: ignore[attr-defined]

        combiner = JournalingStrategyCombiner([ml_strat])  # type: ignore[arg-type]
        config: dict[str, Any] = {
            "strategies": {
                "ml_ensemble": {"enabled": True, "weight": 1.0},
            }
        }
        candles = _make_candles([100.0] * 5)
        with patch.object(combiner, "_load_config", return_value=config):
            combiner.generate_signal("AAPL", candles, "us_tech")

        assert hasattr(combiner, "last_model_probas")
        assert combiner.last_model_probas is not None
        assert "XGBoostModel" in combiner.last_model_probas


# ===================================================================
# Task 2.5 — DecisionRecord strategy_features / model_probas
# ===================================================================


class TestDecisionRecordEnrichment:
    """DecisionRecord includes strategy_features and model_probas fields."""

    def test_decision_record_has_strategy_features_field(self) -> None:
        """DecisionRecord accepts strategy_features as an optional dict."""
        from finalayze.backtest.decision_journal import DecisionRecord, FinalAction

        record = DecisionRecord(
            record_id="00000000-0000-0000-0000-000000000001",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_tech",
            final_action=FinalAction.BUY,
            strategy_features={"momentum.rsi_value": 25.0, "momentum.macd_hist": 0.5},
        )
        assert record.strategy_features is not None
        assert record.strategy_features["momentum.rsi_value"] == pytest.approx(25.0)

    def test_decision_record_has_model_probas_field(self) -> None:
        """DecisionRecord accepts model_probas as an optional dict."""
        from finalayze.backtest.decision_journal import DecisionRecord, FinalAction

        record = DecisionRecord(
            record_id="00000000-0000-0000-0000-000000000002",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_tech",
            final_action=FinalAction.SKIP,
            model_probas={"XGBoostModel": 0.8, "LightGBMModel": 0.7},
        )
        assert record.model_probas is not None
        assert record.model_probas["XGBoostModel"] == pytest.approx(0.8)

    def test_decision_record_fields_default_none(self) -> None:
        """strategy_features and model_probas default to None."""
        from finalayze.backtest.decision_journal import DecisionRecord, FinalAction

        record = DecisionRecord(
            record_id="00000000-0000-0000-0000-000000000003",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_tech",
            final_action=FinalAction.SKIP,
        )
        assert record.strategy_features is None
        assert record.model_probas is None

    def test_make_record_passes_strategy_features(self) -> None:
        """DecisionJournal.make_record accepts strategy_features kwarg."""
        from finalayze.backtest.decision_journal import DecisionJournal, FinalAction

        record = DecisionJournal.make_record(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_tech",
            final_action=FinalAction.BUY,
            strategy_features={"momentum.rsi_value": 28.0},
            model_probas={"XGBoostModel": 0.75},
        )
        assert record.strategy_features is not None
        assert record.model_probas is not None
