"""Tests for the Layer-2 MOEX liquidity primitive (LIQ-01/02/04/06).

RED-first TDD coverage for ``src/finalayze/markets/liquidity.py``:
  - ``median_rub_turnover`` per-symbol median RUB turnover helper (LIQ-01).
  - ``_load_liquidity_snapshot`` fail-closed committed-snapshot loader (LIQ-02).
  - ``eligible_universe_as_of`` point-in-time as-of selection (LIQ-04 / D-05 CARDINAL).
  - ``top_n_per_sector`` Top-N-per-sector selector (LIQ-06).

The D-05 no-look-ahead proof mirrors ``test_pead_sue_proxy.py`` ``TestA4LookAhead``:
appending a FUTURE high-turnover candle is a no-op; moving that same candle to <= T
DOES change the selection (proves the filter is live, not a global no-op).

All inputs are constructed in-test; no live data / token needed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import Candle
from finalayze.markets import liquidity

# ---------------------------------------------------------------------------
# Constants (ruff PLR2004 -- no magic numbers)
# ---------------------------------------------------------------------------
_MARKET_ID_MOEX = "MOEX"
_TIMEFRAME = "1d"
_WINDOW = 60  # D-02 trailing trading days
_SHORT_HISTORY_BARS = 30  # < _WINDOW -> ineligible

_LIQUID_SYMBOL = "SBER"
_THIN_SYMBOL = "THINCO"
_DELISTED_SYMBOL = "OLDCO"
_OTHER_LIQUID_SYMBOL = "GAZP"

_LIQUID_SECTOR = "oil_gas"
_BANK_SECTOR = "banks"
_UNKNOWN_SECTOR = "crypto_mining_moon"

_LIQUID_CLOSE = Decimal("250.00")
_LIQUID_VOLUME = 1_000_000
_THIN_CLOSE = Decimal("10.00")
_THIN_VOLUME = 1_000

_BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
_REBALANCE_TS = _BASE_TS + timedelta(days=_WINDOW + 10)
_FUTURE_OFFSET_DAYS = 5

_TOP_N = 1
_TOP_N_TWO = 2

# Known per-bar turnover so median is exactly predictable.
_EXPECTED_MEDIAN_TURNOVER = Decimal(str(float(_LIQUID_CLOSE) * _LIQUID_VOLUME))


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
def _candle(symbol: str, ts: datetime, close: Decimal, volume: int) -> Candle:
    return Candle(
        symbol=symbol,
        market_id=_MARKET_ID_MOEX,
        timeframe=_TIMEFRAME,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        source="test",
    )


def _series(
    symbol: str,
    *,
    n_bars: int,
    close: Decimal,
    volume: int,
    start: datetime = _BASE_TS,
) -> list[Candle]:
    return [_candle(symbol, start + timedelta(days=i), close, volume) for i in range(n_bars)]


# ===================================================================
# LIQ-01: median_rub_turnover
# ===================================================================
class TestMedianTurnover:
    def test_median_turnover(self) -> None:
        candles = _series(
            _LIQUID_SYMBOL, n_bars=_WINDOW, close=_LIQUID_CLOSE, volume=_LIQUID_VOLUME
        )
        result = liquidity.median_rub_turnover(candles, window=_WINDOW)
        assert result == _EXPECTED_MEDIAN_TURNOVER

    def test_median_turnover_short_history_none(self) -> None:
        candles = _series(
            _THIN_SYMBOL,
            n_bars=_SHORT_HISTORY_BARS,
            close=_THIN_CLOSE,
            volume=_THIN_VOLUME,
        )
        assert liquidity.median_rub_turnover(candles, window=_WINDOW) is None


# ===================================================================
# LIQ-02: fail-closed snapshot loader
# ===================================================================
class TestLoaderFailClosed:
    def test_loader_missing_file(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        missing = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", missing)
        with pytest.raises(ConfigurationError):
            liquidity._load_liquidity_snapshot()

    def test_loader_corrupt_json(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", corrupt)
        with pytest.raises(ConfigurationError):
            liquidity._load_liquidity_snapshot()

    def test_loader_missing_sectors_key(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        no_sectors = tmp_path / "no_sectors.json"
        no_sectors.write_text(json.dumps({"generated_at": "x"}), encoding="utf-8")
        monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", no_sectors)
        with pytest.raises(ConfigurationError):
            liquidity._load_liquidity_snapshot()

    def test_loader_rejects_unknown_sector(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        bad = tmp_path / "unknown_sector.json"
        bad.write_text(
            json.dumps({"sectors": {_UNKNOWN_SECTOR: [_LIQUID_SYMBOL]}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(liquidity, "_LIQ_SNAPSHOT", bad)
        with pytest.raises(ConfigurationError):
            liquidity._load_liquidity_snapshot()


# ===================================================================
# LIQ-04 / D-05: as-of point-in-time selection (CARDINAL look-ahead proof)
# ===================================================================
class TestAsOfLookAhead:
    def _candles_by_symbol(self) -> dict[str, list[Candle]]:
        # One liquid name in oil_gas, one thin name in banks. Both have >= 60 bars
        # up to the rebalance so the selection is well-defined.
        liquid = _series(_LIQUID_SYMBOL, n_bars=_WINDOW, close=_LIQUID_CLOSE, volume=_LIQUID_VOLUME)
        thin = _series(_THIN_SYMBOL, n_bars=_WINDOW, close=_THIN_CLOSE, volume=_THIN_VOLUME)
        return {_LIQUID_SYMBOL: liquid, _THIN_SYMBOL: thin}

    @property
    def _sector_map(self) -> dict[str, str]:
        return {_LIQUID_SYMBOL: _LIQUID_SECTOR, _THIN_SYMBOL: _BANK_SECTOR}

    def test_as_of_no_lookahead(self) -> None:
        candles = self._candles_by_symbol()
        baseline = liquidity.eligible_universe_as_of(
            candles, _REBALANCE_TS, self._sector_map, top_n=_TOP_N
        )

        # Append a FUTURE high-turnover candle to the thin name AFTER the rebalance.
        future_ts = _REBALANCE_TS + timedelta(days=_FUTURE_OFFSET_DAYS)
        with_future = dict(candles)
        with_future[_THIN_SYMBOL] = [
            *candles[_THIN_SYMBOL],
            _candle(_THIN_SYMBOL, future_ts, _LIQUID_CLOSE, _LIQUID_VOLUME),
        ]
        result = liquidity.eligible_universe_as_of(
            with_future, _REBALANCE_TS, self._sector_map, top_n=_TOP_N
        )

        # The future-dated candle was filtered by the <= rebalance_ts cutoff: identical.
        assert result == baseline

    def test_as_of_same_candle_in_window_changes(self) -> None:
        """Moving that candle to <= T (in-window) MUST change the result."""
        candles = self._candles_by_symbol()
        baseline = liquidity.eligible_universe_as_of(
            candles, _REBALANCE_TS, self._sector_map, top_n=_TOP_N
        )

        # Replace the thin name's whole window with high-turnover bars (<= T):
        # it now out-ranks within its own sector and (with top_n across both
        # sectors) the eligible set differs from the baseline thin-excluded set.
        in_window = dict(candles)
        in_window[_THIN_SYMBOL] = _series(
            _THIN_SYMBOL,
            n_bars=_WINDOW,
            close=_LIQUID_CLOSE,
            volume=_LIQUID_VOLUME,
        )
        result = liquidity.eligible_universe_as_of(
            in_window, _REBALANCE_TS, self._sector_map, top_n=_TOP_N
        )

        assert _THIN_SYMBOL not in baseline
        assert _THIN_SYMBOL in result
        assert result != baseline

    def test_survivorship(self) -> None:
        # Delisted name has 60 bars ONLY in an early window; none near a later rebalance.
        delisted = _series(
            _DELISTED_SYMBOL,
            n_bars=_WINDOW,
            close=_LIQUID_CLOSE,
            volume=_LIQUID_VOLUME,
            start=_BASE_TS,
        )
        candles = {_DELISTED_SYMBOL: delisted}
        sector_map = {_DELISTED_SYMBOL: _LIQUID_SECTOR}

        early_rebalance = _BASE_TS + timedelta(days=_WINDOW)
        late_rebalance = _BASE_TS + timedelta(days=_WINDOW * 10)

        eligible_early = liquidity.eligible_universe_as_of(
            candles, early_rebalance, sector_map, top_n=_TOP_N
        )
        eligible_late = liquidity.eligible_universe_as_of(
            candles, late_rebalance, sector_map, top_n=_TOP_N
        )

        assert _DELISTED_SYMBOL in eligible_early
        assert _DELISTED_SYMBOL not in eligible_late


# ===================================================================
# LIQ-06: Top-N-per-sector
# ===================================================================
class TestTopNPerSector:
    def test_top_n_per_sector(self) -> None:
        scores: dict[str, Decimal] = {
            _LIQUID_SYMBOL: Decimal(1000),
            _OTHER_LIQUID_SYMBOL: Decimal(500),
            _THIN_SYMBOL: Decimal(10),
        }
        sector_map = {
            _LIQUID_SYMBOL: _LIQUID_SECTOR,
            _OTHER_LIQUID_SYMBOL: _LIQUID_SECTOR,
            _THIN_SYMBOL: _BANK_SECTOR,
        }
        result = liquidity.top_n_per_sector(scores, sector_map, top_n=_TOP_N)

        # Top-1 of oil_gas is the highest-turnover SBER; GAZP excluded.
        assert result[_LIQUID_SECTOR] == [_LIQUID_SYMBOL]
        assert result[_BANK_SECTOR] == [_THIN_SYMBOL]

        # Size is bounded by top_n * sector_count.
        total = sum(len(v) for v in result.values())
        sector_count = len(set(sector_map.values()))
        assert total <= _TOP_N * sector_count

    def test_top_n_two_keeps_both(self) -> None:
        scores: dict[str, Decimal] = {
            _LIQUID_SYMBOL: Decimal(1000),
            _OTHER_LIQUID_SYMBOL: Decimal(500),
        }
        sector_map = {
            _LIQUID_SYMBOL: _LIQUID_SECTOR,
            _OTHER_LIQUID_SYMBOL: _LIQUID_SECTOR,
        }
        result = liquidity.top_n_per_sector(scores, sector_map, top_n=_TOP_N_TWO)
        # Ranked by turnover desc.
        assert result[_LIQUID_SECTOR] == [_LIQUID_SYMBOL, _OTHER_LIQUID_SYMBOL]
