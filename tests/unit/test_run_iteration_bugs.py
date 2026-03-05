"""Unit tests for run_iteration.py bug fixes and strategy wiring.

Tests cover:
- Bug 2: MLStrategy gated on --models-dir CLI flag
- UNIVERSE has real MOEX tickers
- _build_strategies includes all wired strategies
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Insert project root so scripts/ and config/ are importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the script as a module (it guards main() behind __name__ == "__main__")
# We need to patch argparse before import so it doesn't call sys.exit
with patch("sys.argv", ["run_iteration.py", "--name", "test", "--description", "test"]):
    import scripts.run_iteration as ri  # noqa: E402

from finalayze.strategies.dividend_gap import DividendGapStrategy  # noqa: E402
from finalayze.strategies.dual_momentum import DualMomentumStrategy  # noqa: E402
from finalayze.strategies.event_driven import EventDrivenStrategy  # noqa: E402
from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy  # noqa: E402

# ── Bug 2: _setup_ml_strategy handles missing models ──────────────────────────


class TestBug2MLStrategyNoModelsDir:
    """MLStrategy should be skipped when models_dir is None or no segment dir."""

    def test_build_strategies_without_models_dir_does_not_crash(self) -> None:
        """_build_strategies should work when models_dir=None (no gate)."""
        from datetime import UTC, datetime  # noqa: PLC0415

        from finalayze.data.fetchers.yfinance import YFinanceFetcher  # noqa: PLC0415

        fetcher = YFinanceFetcher(market_id="us")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        # Should not crash -- MLStrategy path is gated by models_dir is not None
        strategies = ri._build_strategies(
            "us_tech", fetcher, start, end, models_dir=None, symbols=["AAPL", "MSFT"]
        )
        assert isinstance(strategies, list)
        assert len(strategies) >= 1  # At least the base strategies


# ── UNIVERSE has real MOEX tickers ───────────────────────────────────────────


class TestUniverseMOEXTickers:
    """UNIVERSE dict must contain real MOEX symbols, not ETF proxies."""

    def test_ru_blue_chips_has_real_moex_tickers(self) -> None:
        """ru_blue_chips should contain SBER, GAZP, LKOH, GMKN."""
        symbols = ri.UNIVERSE["ru_blue_chips"]
        assert "SBER" in symbols
        assert "GAZP" in symbols
        assert "LKOH" in symbols
        assert "GMKN" in symbols

    def test_ru_blue_chips_no_etf_proxies(self) -> None:
        """ru_blue_chips must NOT contain ETF proxies like RSX, ERUS."""
        symbols = ri.UNIVERSE["ru_blue_chips"]
        for proxy in ("RSX", "ERUS", "FLRU.L", "TUR", "EWZ", "INDA"):
            assert proxy not in symbols

    def test_ru_energy_has_real_moex_tickers(self) -> None:
        """ru_energy should contain ROSN, TATN, NVTK."""
        symbols = ri.UNIVERSE["ru_energy"]
        assert "ROSN" in symbols
        assert "TATN" in symbols
        assert "NVTK" in symbols

    def test_ru_energy_no_etf_proxies(self) -> None:
        """ru_energy must NOT contain ETF proxies like XLE, BP."""
        symbols = ri.UNIVERSE["ru_energy"]
        for proxy in ("XLE", "BP", "SHEL", "TTE", "ENB"):
            assert proxy not in symbols


# ── _build_strategies includes all wired strategies ──────────────────────────

_MIN_BASE_STRATEGIES = 5  # Momentum, MeanReversion, RSI2Connors, OU, DualMom


class TestBuildStrategiesExpanded:
    """_build_strategies returns all wired strategies including newly added ones."""

    def _build(self, segment: str = "us_tech") -> list[object]:
        from datetime import UTC, datetime  # noqa: PLC0415

        from finalayze.data.fetchers.yfinance import YFinanceFetcher  # noqa: PLC0415

        fetcher = YFinanceFetcher(market_id="us")
        # Wide range to cover static YAML dividend dates (Jun-Jul 2024)
        start = datetime(2022, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        symbols = ri.UNIVERSE.get(segment, ["AAPL", "MSFT"])
        return ri._build_strategies(segment, fetcher, start, end, models_dir=None, symbols=symbols)

    def test_us_segment_has_at_least_5_strategies(self) -> None:
        strategies = self._build("us_tech")
        assert len(strategies) >= _MIN_BASE_STRATEGIES

    def test_ru_segment_has_dividend_gap(self) -> None:
        strategies = self._build("ru_blue_chips")
        types = [type(s) for s in strategies]
        assert DividendGapStrategy in types

    def test_us_segment_no_dividend_gap(self) -> None:
        strategies = self._build("us_tech")
        types = [type(s) for s in strategies]
        assert DividendGapStrategy not in types

    def test_ou_mean_reversion_present(self) -> None:
        strategies = self._build("us_tech")
        types = [type(s) for s in strategies]
        assert OUMeanReversionStrategy in types

    def test_dual_momentum_present(self) -> None:
        strategies = self._build("us_tech")
        types = [type(s) for s in strategies]
        assert DualMomentumStrategy in types

    def test_ru_segment_has_at_least_6_strategies(self) -> None:
        """RU segments get DividendGap extra, so >= 6."""
        strategies = self._build("ru_blue_chips")
        assert len(strategies) >= _MIN_BASE_STRATEGIES + 1
