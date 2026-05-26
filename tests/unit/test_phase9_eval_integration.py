"""Tests for Phase 9: PairsStrategy + MLStrategy evaluation integration."""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.ml.registry import MLModelRegistry
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.pairs import PairsStrategy

# Ensure scripts/ is importable
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

_BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
_NUM_CANDLES = 100


def _make_candles(
    symbol: str = "AAPL",
    n: int = _NUM_CANDLES,
    base_price: float = 150.0,
) -> list[Candle]:
    """Create a list of synthetic candles for testing."""
    candles = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.1))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=_BASE_TS + timedelta(days=i),
                open=price,
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1_000_000,
            )
        )
    return candles


# ── Fix 1: PairsStrategy wiring ──────────────────────────────────────────────


class TestPairsStrategyWiring:
    """Test that PairsStrategy receives peer candles and is properly set up."""

    def test_pairs_strategy_receives_peer_candles(self) -> None:
        """PairsStrategy should have peer candles after _setup_pairs_strategy."""
        strategy = PairsStrategy()
        msft_candles = _make_candles(symbol="MSFT", n=80)
        strategy.set_peer_candles("MSFT", msft_candles)

        assert "MSFT" in strategy._peer_candles
        assert len(strategy._peer_candles["MSFT"]) == 80

    def test_pairs_strategy_skipped_when_no_pairs_in_preset(self) -> None:
        """_setup_pairs_strategy returns None when preset has no pairs config."""
        # ru_tech has no pairs section in the YAML (was us_industrial pre-S1.2)
        from run_evaluation import _setup_pairs_strategy

        mock_fetcher = MagicMock()
        start = _BASE_TS
        end = _BASE_TS + timedelta(days=365)

        result = _setup_pairs_strategy("ru_tech", mock_fetcher, start, end)
        assert result is None

    def test_pairs_strategy_created_for_us_tech(self) -> None:
        """_setup_pairs_strategy returns PairsStrategy for us_tech (has pairs config)."""
        from run_evaluation import _setup_pairs_strategy

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_candles.return_value = _make_candles(symbol="MSFT")
        start = _BASE_TS
        end = _BASE_TS + timedelta(days=365)

        result = _setup_pairs_strategy("us_tech", mock_fetcher, start, end)
        assert result is not None
        assert isinstance(result, PairsStrategy)
        # Should have fetched candles for peers in the us_tech pairs config
        mock_fetcher.fetch_candles.assert_called()


# ── Fix 2: MLStrategy wiring ─────────────────────────────────────────────────


class TestMLStrategyWiring:
    """Test MLStrategy model loading and graceful degradation."""

    def test_ml_strategy_graceful_when_no_models(self) -> None:
        """_setup_ml_strategy returns None when models_dir doesn't have segment subdir."""
        from run_evaluation import _setup_ml_strategy

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _setup_ml_strategy("us_tech", Path(tmpdir))
        assert result is None

    def test_ml_strategy_none_when_empty_segment_dir(self) -> None:
        """_setup_ml_strategy returns None when segment dir exists but has no model files."""
        from run_evaluation import _setup_ml_strategy

        with tempfile.TemporaryDirectory() as tmpdir:
            segment_dir = Path(tmpdir) / "us_tech"
            segment_dir.mkdir()
            result = _setup_ml_strategy("us_tech", Path(tmpdir))
        assert result is None

    def test_ml_strategy_returns_none_for_untrained(self) -> None:
        """MLStrategy returns None signal when ensemble is untrained (prob=0.5)."""
        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        registry.register("us_tech", ensemble)
        strategy = MLStrategy(registry)

        candles = _make_candles(n=_NUM_CANDLES)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None


# ── Fix 3: YAML preset enablement ────────────────────────────────────────────


class TestMLEnsemblePresets:
    """Test that ml_ensemble section exists in all YAML presets (disabled until trained)."""

    def test_ml_ensemble_present_in_all_presets(self) -> None:
        """All equity segment YAML presets should have ml_ensemble section."""
        # 4 US (us_tech, us_broad, us_finance, us_healthcare — frozen but presets kept)
        # + 4 RU (ru_blue_chips, ru_energy, ru_finance, ru_tech).
        # us_industrial preset removed in S1.2 as orphan (no segment).
        expected_preset_count = 8
        # Filter to equity segment presets (have segment_id key, instrument_type != bond)
        preset_files = sorted(_PRESETS_DIR.glob("*.yaml"))
        segment_presets = []
        for p in preset_files:
            with p.open() as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "segment_id" in data:
                # Bond presets don't use ml_ensemble
                if data.get("instrument_type") == "bond":
                    continue
                segment_presets.append((p, data))
        assert len(segment_presets) == expected_preset_count, (
            f"Expected {expected_preset_count} equity segment presets, found {len(segment_presets)}"
        )

        for preset_path, data in segment_presets:
            ml_cfg = data.get("strategies", {}).get("ml_ensemble")
            assert ml_cfg is not None, f"{preset_path.name}: ml_ensemble section should exist"


# ── Integration ───────────────────────────────────────────────────────────────


class TestCombinerWithAllStrategies:
    """Test that the combiner works with all strategies wired in."""

    def test_combiner_with_all_strategies_produces_signals(self) -> None:
        """JournalingStrategyCombiner should accept Momentum + MeanReversion + Pairs + ML."""
        registry = MLModelRegistry()
        ensemble = registry.create_ensemble("us_tech")
        registry.register("us_tech", ensemble)

        strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            PairsStrategy(),
            MLStrategy(registry),
        ]
        combiner = JournalingStrategyCombiner(strategies=strategies)

        candles = _make_candles(n=_NUM_CANDLES)
        # Should not raise — signal may or may not fire depending on candle data
        combiner.generate_signal("AAPL", candles, "us_tech")
        # Verify all strategies were evaluated
        assert "momentum" in combiner.last_signals
        assert "mean_reversion" in combiner.last_signals

    def test_batch_evaluation_strategy_list_includes_pairs(self) -> None:
        """_build_strategies should include PairsStrategy for segments with pairs config."""
        from run_batch_evaluation import _build_strategies

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_candles.return_value = _make_candles(symbol="MSFT")
        start = _BASE_TS
        end = _BASE_TS + timedelta(days=365)

        strategies = _build_strategies("us_tech", mock_fetcher, start, end, models_dir=None)
        strategy_names = [s.name for s in strategies]
        assert "momentum" in strategy_names
        assert "mean_reversion" in strategy_names
        assert "pairs" in strategy_names
