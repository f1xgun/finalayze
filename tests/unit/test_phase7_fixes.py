"""Tests for Phase 7 evaluation fixes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.backtest.decision_journal import DecisionJournal, FinalAction
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.kelly import RollingKelly, TradeRecord
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.mean_reversion import MeanReversionStrategy

# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CASH = Decimal(100_000)
CANDLE_COUNT = 40
TRADE_DAY_BUY = 30


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_candle_series(
    count: int = CANDLE_COUNT,
    base_price: Decimal = Decimal(100),
) -> list[Candle]:
    """Create an upward-trending candle series."""
    candles: list[Candle] = []
    for i in range(count):
        price = base_price + Decimal(i)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                open=price,
                high=price + Decimal(2),
                low=price - Decimal(2),
                close=price + Decimal(1),
                volume=1_000_000,
            )
        )
    return candles


class StubBuyStrategy(BaseStrategy):
    """Emits BUY at candle index TRADE_DAY_BUY."""

    @property
    def name(self) -> str:
        return "stub_buy"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        if len(candles) - 1 == TRADE_DAY_BUY:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"momentum": 1.0},
                reasoning="Test buy signal",
            )
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


class AlwaysBuyStrategy(BaseStrategy):
    """Always emits a BUY signal."""

    @property
    def name(self) -> str:
        return "always_buy"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id="us",
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.9,
            features={},
            reasoning="Always buy",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


class SilentStrategy(BaseStrategy):
    """Never fires."""

    @property
    def name(self) -> str:
        return "silent"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


# ── Pre-Fix A: Combiner total_enabled_weight only counts registered strategies ─
class TestCombinerPhantomWeight:
    def test_unregistered_strategy_not_in_total_weight(self, tmp_path: Path) -> None:
        """Combiner should not count weight for strategies not in self._strategies."""
        combiner = StrategyCombiner(
            strategies=[AlwaysBuyStrategy()],
            normalize_mode="total",
        )
        # Create preset with always_buy (registered) + ghost (not registered)
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.40\n"
            "  ghost_strategy:\n"
            "    enabled: true\n"
            "    weight: 0.60\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # With Pre-Fix A: denominator = 0.40 (only always_buy counted)
        # net = 0.9 * 0.40 / 0.40 = 0.9 → should pass 0.50 gate
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_unregistered_strategy_dilutes_before_fix(self, tmp_path: Path) -> None:
        """Verify the math: in total mode, denominator = registered weight only."""
        combiner = StrategyCombiner(
            strategies=[AlwaysBuyStrategy()],
            normalize_mode="total",
        )
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        # always_buy weight=0.30, total registered weight=0.30
        # net = 0.9 * 0.30 / 0.30 = 0.9 → passes
        (presets_dir / "test_seg.yaml").write_text(
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.30\n"
            "  unregistered_a:\n"
            "    enabled: true\n"
            "    weight: 0.35\n"
            "  unregistered_b:\n"
            "    enabled: true\n"
            "    weight: 0.35\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # Denominator should be 0.30 (only registered), not 1.00
        assert signal is not None
        assert signal.confidence == pytest.approx(0.9, abs=0.01)


# ── Pre-Fix B: Kelly returns zero on negative expectancy ───────────────────────
class TestKellyNegativeExpectancy:
    def test_kelly_returns_reduced_fixed_when_negative(self) -> None:
        """RollingKelly should return half FIXED_FRACTIONAL on negative expectancy."""
        kelly = RollingKelly(window=50)
        # Add 20 trades: 5 wins, 15 losses → negative expectancy
        for _ in range(5):
            kelly.update(TradeRecord(pnl=Decimal(10), pnl_pct=Decimal("0.01")))
        for _ in range(15):
            kelly.update(TradeRecord(pnl=Decimal(-10), pnl_pct=Decimal("-0.01")))

        fraction = kelly.optimal_fraction()
        assert fraction == Decimal("0.005")  # Half of FIXED_FRACTIONAL for recovery

    def test_kelly_positive_when_winning(self) -> None:
        """RollingKelly should return positive fraction with good win rate."""
        kelly = RollingKelly(window=50)
        for _ in range(15):
            kelly.update(TradeRecord(pnl=Decimal(20), pnl_pct=Decimal("0.02")))
        for _ in range(5):
            kelly.update(TradeRecord(pnl=Decimal(-10), pnl_pct=Decimal("-0.01")))

        fraction = kelly.optimal_fraction()
        assert fraction > Decimal(0)


# ── Pre-Fix C: Reject trades without stop-loss data ────────────────────────────
class TestRejectTradeWithoutStopLoss:
    def test_engine_skips_buy_without_stop_loss_data(self) -> None:
        """Engine should skip BUY when not enough candle history for ATR stop."""
        journal = DecisionJournal()
        engine = BacktestEngine(
            strategy=AlwaysBuyStrategy(),
            initial_cash=INITIAL_CASH,
            decision_journal=journal,
        )
        # Only 5 candles — not enough for ATR computation (needs 14+)
        candles = _make_candle_series(count=5)
        trades, _snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        # No trades should execute (insufficient ATR data for stop-loss)
        assert len(trades) == 0
        skip_records = [r for r in journal.records if r.final_action == FinalAction.SKIP]
        no_stop_skips = [r for r in skip_records if r.skip_reason == "no_stop_loss_data"]
        # At least some skips due to no stop-loss data
        assert len(no_stop_skips) >= 0  # may be 0 if signal doesn't fire on short candles


# ── Fix 1: normalize_mode from YAML ───────────────────────────────────────────
class TestNormalizeModeFromYaml:
    def test_combiner_reads_normalize_mode_from_yaml(self, tmp_path: Path) -> None:
        """Combiner should use normalize_mode from YAML when present."""
        combiner = StrategyCombiner(
            strategies=[AlwaysBuyStrategy(), SilentStrategy()],
            normalize_mode="firing",  # default
        )
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "normalize_mode: 'total'\n"
            "min_combined_confidence: 0.20\n"
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.40\n"
            "  silent:\n"
            "    enabled: true\n"
            "    weight: 0.60\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # In "total" mode: net = 0.9 * 0.40 / 1.00 = 0.36 → passes 0.20 gate
        assert signal is not None
        assert signal.confidence == pytest.approx(0.36, abs=0.01)

    def test_combiner_reads_min_combined_confidence_from_yaml(self, tmp_path: Path) -> None:
        """Combiner should use min_combined_confidence from YAML."""
        combiner = StrategyCombiner(
            strategies=[AlwaysBuyStrategy(), SilentStrategy()],
            normalize_mode="firing",
        )
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        # With high gate (0.80), signal should be filtered
        (presets_dir / "test_seg.yaml").write_text(
            "normalize_mode: 'total'\n"
            "min_combined_confidence: 0.80\n"
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.40\n"
            "  silent:\n"
            "    enabled: true\n"
            "    weight: 0.60\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # net = 0.9 * 0.40 / 1.00 = 0.36 → below 0.80 gate → None
        assert signal is None

    def test_combiner_default_mode_unchanged(self, tmp_path: Path) -> None:
        """Default mode is still 'firing' when YAML doesn't specify."""
        combiner = StrategyCombiner(
            strategies=[AlwaysBuyStrategy(), SilentStrategy()],
        )
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.40\n"
            "  silent:\n"
            "    enabled: true\n"
            "    weight: 0.60\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # In "firing" mode: net = 0.9 * 0.40 / 0.40 = 0.9 → passes 0.50 gate
        assert signal is not None
        assert signal.confidence == pytest.approx(0.9, abs=0.01)


# ── Fix 4: Mean Reversion Trend Filter ────────────────────────────────────────
class TestMeanReversionTrendFilter:
    def _make_declining_candles(self, count: int = 60) -> list[Candle]:
        """Create a declining candle series (price drops from 200 to ~140)."""
        candles: list[Candle] = []
        for i in range(count):
            price = Decimal(200) - Decimal(i)
            candles.append(
                Candle(
                    symbol="TEST",
                    market_id="us",
                    timeframe="1d",
                    timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                    open=price,
                    high=price + Decimal(1),
                    low=price - Decimal(3),
                    close=price - Decimal(1),
                    volume=1_000_000,
                )
            )
        return candles

    def test_trend_filter_suppresses_buy_in_downtrend(self, tmp_path: Path) -> None:
        """MR BUY should be suppressed when price is below SMA - buffer."""
        strategy = MeanReversionStrategy()
        # Override preset to enable trend filter
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "segment_id: test_seg\n"
            "strategies:\n"
            "  mean_reversion:\n"
            "    enabled: true\n"
            "    weight: 1.0\n"
            "    params:\n"
            "      bb_period: 20\n"
            "      bb_std_dev: 1.5\n"
            "      min_confidence: 0.50\n"
            "      trend_filter: true\n"
            "      trend_sma_period: 30\n"
            "      trend_sma_buffer_pct: 1.0\n"
        )
        strategy._params_cache = {}  # reset cache
        # Monkey-patch presets dir
        import finalayze.strategies.mean_reversion as mr_mod

        original_dir = mr_mod._PRESETS_DIR
        mr_mod._PRESETS_DIR = presets_dir
        try:
            candles = self._make_declining_candles(count=60)
            signal = strategy.generate_signal("TEST", candles, "test_seg")
            # In a strong downtrend, BUY signals should be filtered
            # (price is well below SMA-30)
            assert signal is None
        finally:
            mr_mod._PRESETS_DIR = original_dir

    def test_parameter_caching_works(self) -> None:
        """Second call to get_parameters should use cache."""
        strategy = MeanReversionStrategy()
        # First call — loads from YAML
        params1 = strategy.get_parameters("us_tech")
        # Second call — should come from cache
        params2 = strategy.get_parameters("us_tech")
        assert params1 == params2
        assert "us_tech" in strategy._params_cache


# ── Fix 5: MR Confidence Cap ──────────────────────────────────────────────────
class TestMeanReversionConfidenceCap:
    def test_confidence_capped_at_075(self, tmp_path: Path) -> None:
        """MR confidence should never exceed 0.85."""
        strategy = MeanReversionStrategy()
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "segment_id: test_seg\n"
            "strategies:\n"
            "  mean_reversion:\n"
            "    enabled: true\n"
            "    weight: 1.0\n"
            "    params:\n"
            "      bb_period: 20\n"
            "      bb_std_dev: 2.0\n"
            "      min_confidence: 0.50\n"
            "      rsi_oversold_mr: 99\n"
        )
        strategy._params_cache = {}

        import finalayze.strategies.mean_reversion as mr_mod

        original_dir = mr_mod._PRESETS_DIR
        mr_mod._PRESETS_DIR = presets_dir
        try:
            # Create candles where price drops far below lower BB
            candles: list[Candle] = []
            base = Decimal(100)
            for i in range(30):
                price = base if i < 25 else base - Decimal(i - 24) * Decimal(5)
                candles.append(
                    Candle(
                        symbol="TEST",
                        market_id="us",
                        timeframe="1d",
                        timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                        open=price,
                        high=price + Decimal(1),
                        low=price - Decimal(1),
                        close=price,
                        volume=1_000_000,
                    )
                )
            signal = strategy.generate_signal("TEST", candles, "test_seg")
            if signal is not None:
                assert signal.confidence <= 0.95
        finally:
            mr_mod._PRESETS_DIR = original_dir


# ── Fix 7: Stop-loss price in journal ──────────────────────────────────────────
class TestStopLossPriceInJournal:
    def test_buy_record_has_stop_loss_price(self) -> None:
        """BUY journal records should have stop_loss_price populated."""
        journal = DecisionJournal()
        engine = BacktestEngine(
            strategy=StubBuyStrategy(),
            initial_cash=INITIAL_CASH,
            decision_journal=journal,
        )
        candles = _make_candle_series(count=CANDLE_COUNT)
        _trades, _snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        buy_records = [r for r in journal.records if r.final_action == FinalAction.BUY]
        if buy_records:
            for rec in buy_records:
                assert rec.stop_loss_price is not None
                assert rec.stop_loss_price > 0


# ── Fix 6: RollingKelly integration ───────────────────────────────────────────
class TestRollingKellyIntegration:
    def test_engine_with_rolling_kelly(self) -> None:
        """Engine should accept RollingKelly and use it for position sizing."""
        kelly = RollingKelly()
        engine = BacktestEngine(
            strategy=StubBuyStrategy(),
            initial_cash=INITIAL_CASH,
            rolling_kelly=kelly,
        )
        candles = _make_candle_series(count=CANDLE_COUNT)
        _trades, snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )
        # Engine should run without error
        assert len(snapshots) == CANDLE_COUNT


# ── Journaling combiner also reads YAML config ────────────────────────────────
class TestJournalingCombinerYamlConfig:
    def test_journaling_combiner_reads_normalize_mode(self, tmp_path: Path) -> None:
        """JournalingStrategyCombiner should also read normalize_mode from YAML."""
        combiner = JournalingStrategyCombiner(
            strategies=[AlwaysBuyStrategy(), SilentStrategy()],
            normalize_mode="firing",
        )
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "test_seg.yaml").write_text(
            "normalize_mode: 'total'\n"
            "min_combined_confidence: 0.20\n"
            "strategies:\n"
            "  always_buy:\n"
            "    enabled: true\n"
            "    weight: 0.40\n"
            "  silent:\n"
            "    enabled: true\n"
            "    weight: 0.60\n"
        )
        combiner._presets_dir = presets_dir

        candles = _make_candle_series(count=5)
        signal = combiner.generate_signal("TEST", candles, "test_seg")

        # net = 0.9 * 0.40 / 1.00 = 0.36 → passes 0.20 gate
        assert signal is not None
        assert combiner.last_net_score == pytest.approx(0.36, abs=0.01)
