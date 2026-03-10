"""Tests for MOEX-specific ML features."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import (
    Candle,
    FXRate,
    KeyRateRecord,
    MarketContext,
    MoexMarketData,
    TurnoverRecord,
)
from finalayze.ml.features.technical import (
    _EXTERNAL_DATA_LAG_BARS,
    _compute_commodity_features,
    _compute_fx_features,
    _compute_macro_features,
    _compute_turnover_features,
    compute_features,
)


def _make_candles(n: int = 100, symbol: str = "SBER") -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            market_id="moex",
            timeframe="1d",
            timestamp=datetime(2024, 1, 1, 7, 0, tzinfo=UTC) + timedelta(days=i),
            open=Decimal(str(100 + i * 0.1)),
            high=Decimal(str(101 + i * 0.1)),
            low=Decimal(str(99 + i * 0.1)),
            close=Decimal(str(100.5 + i * 0.1)),
            volume=1000000 + i * 1000,
        )
        for i in range(n)
    ]


def _make_fx_rates(n: int = 80) -> tuple[FXRate, ...]:
    return tuple(
        FXRate(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(days=i),
            pair="USDRUB",
            rate=Decimal(str(85 + i * 0.1)),
        )
        for i in range(n)
    )


def _make_key_rates(n: int = 5) -> tuple[KeyRateRecord, ...]:
    return tuple(
        KeyRateRecord(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(days=i * 30),
            rate=Decimal("0.16"),  # 16% as decimal fraction
        )
        for i in range(n)
    )


def _make_turnover(n: int = 80) -> tuple[TurnoverRecord, ...]:
    return tuple(
        TurnoverRecord(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(days=i),
            volume_rub=Decimal(str(1_500_000_000_000 + i * 10_000_000_000)),
        )
        for i in range(n)
    )


class TestExternalDataLag:
    def test_lag_constant_exists(self) -> None:
        assert _EXTERNAL_DATA_LAG_BARS == 2  # noqa: PLR2004


class TestFXFeatures:
    def test_returns_default_when_none(self) -> None:
        assert _compute_fx_features(None) == {"usdrub_zscore_60d": 0.0}

    def test_returns_float_with_data(self) -> None:
        moex = MoexMarketData(fx_rates=_make_fx_rates(80))
        result = _compute_fx_features(moex)
        assert isinstance(result["usdrub_zscore_60d"], float)

    def test_returns_default_insufficient_data(self) -> None:
        moex = MoexMarketData(fx_rates=_make_fx_rates(5))
        assert _compute_fx_features(moex)["usdrub_zscore_60d"] == 0.0

    def test_structural_break_circuit_breaker(self) -> None:
        """Extreme std (>20% of mean) -> returns 0.0 (neutral)."""
        rates = list(_make_fx_rates(80))
        # Inject multiple massive spikes to push std > 20% of mean
        for i in [-3, -5, -7, -10]:
            rates[i] = FXRate(
                timestamp=rates[i].timestamp,
                pair="USDRUB",
                rate=Decimal("300.00"),
            )
        moex = MoexMarketData(fx_rates=tuple(rates))
        result = _compute_fx_features(moex)
        assert result["usdrub_zscore_60d"] == 0.0


class TestCommodityFeatures:
    def test_returns_default_when_none(self) -> None:
        assert _compute_commodity_features(None) == {"brent_zscore_60d": 0.0}

    def test_returns_float_with_data(self) -> None:
        brent = tuple(_make_candles(80, "BZ=F"))
        moex = MoexMarketData(commodity_candles={"BZ=F": brent})
        result = _compute_commodity_features(moex)
        assert isinstance(result["brent_zscore_60d"], float)


class TestMacroFeatures:
    def test_returns_default_when_none(self) -> None:
        assert _compute_macro_features(None) == {"real_rate_zscore": 0.0}

    def test_returns_float_with_data(self) -> None:
        moex = MoexMarketData(key_rates=_make_key_rates(5))
        result = _compute_macro_features(moex)
        assert isinstance(result["real_rate_zscore"], float)


class TestTurnoverFeatures:
    def test_returns_default_when_none(self) -> None:
        assert _compute_turnover_features(None) == {"market_turnover_zscore": 0.0}

    def test_returns_float_with_data(self) -> None:
        moex = MoexMarketData(turnover=_make_turnover(80))
        result = _compute_turnover_features(moex)
        assert isinstance(result["market_turnover_zscore"], float)

    def test_zscore_clipped(self) -> None:
        records = list(_make_turnover(80))
        records[-3] = TurnoverRecord(
            timestamp=records[-3].timestamp,
            volume_rub=Decimal("99e12"),
        )
        moex = MoexMarketData(turnover=tuple(records))
        result = _compute_turnover_features(moex)
        assert -3.0 <= result["market_turnover_zscore"] <= 3.0


class TestLagEnforcement:
    def test_fx_lag_excludes_last_1(self) -> None:
        """Spike at position [-1] should NOT affect z-score (lagged by 2)."""
        rates = list(_make_fx_rates(80))
        rates[-1] = FXRate(
            timestamp=rates[-1].timestamp,
            pair="USDRUB",
            rate=Decimal("500.00"),
        )
        moex_with_spike = MoexMarketData(fx_rates=tuple(rates))
        moex_no_spike = MoexMarketData(fx_rates=_make_fx_rates(80))

        result_spike = _compute_fx_features(moex_with_spike)
        result_clean = _compute_fx_features(moex_no_spike)
        assert result_spike["usdrub_zscore_60d"] == result_clean["usdrub_zscore_60d"]

    def test_fx_lag_excludes_last_2(self) -> None:
        """Spike at position [-2] should also NOT affect z-score (lag = 2)."""
        rates = list(_make_fx_rates(80))
        rates[-2] = FXRate(
            timestamp=rates[-2].timestamp,
            pair="USDRUB",
            rate=Decimal("500.00"),
        )
        moex_with_spike = MoexMarketData(fx_rates=tuple(rates))
        moex_no_spike = MoexMarketData(fx_rates=_make_fx_rates(80))

        result_spike = _compute_fx_features(moex_with_spike)
        result_clean = _compute_fx_features(moex_no_spike)
        assert result_spike["usdrub_zscore_60d"] == result_clean["usdrub_zscore_60d"]

    def test_fx_lag_includes_position_minus_3(self) -> None:
        """Spike at position [-3] SHOULD affect z-score (inside lag window)."""
        rates = list(_make_fx_rates(80))
        rates[-3] = FXRate(
            timestamp=rates[-3].timestamp,
            pair="USDRUB",
            rate=Decimal("150.00"),
        )
        moex_with_spike = MoexMarketData(fx_rates=tuple(rates))
        moex_no_spike = MoexMarketData(fx_rates=_make_fx_rates(80))

        result_spike = _compute_fx_features(moex_with_spike)
        result_clean = _compute_fx_features(moex_no_spike)
        assert result_spike["usdrub_zscore_60d"] != result_clean["usdrub_zscore_60d"]


class TestMacroFeaturesEdgeCases:
    def test_missing_cpi_all_months_returns_default(self) -> None:
        """If CPI table has no matching entries even with 6-month fallback -> 0.0."""
        rates = tuple(
            KeyRateRecord(
                timestamp=datetime(2030, 1 + i, 1, 0, 0, tzinfo=UTC),
                rate=Decimal("0.20"),
            )
            for i in range(5)
        )
        moex = MoexMarketData(key_rates=rates)
        result = _compute_macro_features(moex)
        assert result["real_rate_zscore"] == 0.0


class TestComputeFeaturesMarketContext:
    def test_market_context_parameter(self) -> None:
        features = compute_features(_make_candles(100), market_context=MarketContext())
        assert isinstance(features, dict)

    def test_deprecated_kwargs_warn(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_features(_make_candles(100), benchmark_candles=_make_candles(100, "IMOEX"))
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_moex_features_present(self) -> None:
        moex = MoexMarketData(
            fx_rates=_make_fx_rates(80),
            turnover=_make_turnover(80),
            key_rates=_make_key_rates(5),
            commodity_candles={"BZ=F": tuple(_make_candles(80, "BZ=F"))},
        )
        ctx = MarketContext(moex_data=moex)
        features = compute_features(_make_candles(100), market_context=ctx)
        moex_keys = (
            "usdrub_zscore_60d",
            "brent_zscore_60d",
            "real_rate_zscore",
            "market_turnover_zscore",
        )
        for feat in moex_keys:
            assert feat in features

    def test_moex_features_default_without_data(self) -> None:
        features = compute_features(_make_candles(100), market_context=MarketContext())
        moex_keys = (
            "usdrub_zscore_60d",
            "brent_zscore_60d",
            "real_rate_zscore",
            "market_turnover_zscore",
        )
        for feat in moex_keys:
            assert features.get(feat, 0.0) == 0.0


class TestBrentHolidaySuppression:
    # Base date 2024-02-01 (Thursday) — no Russian holidays in the 80-day range
    _BASE = datetime(2024, 2, 1, 14, 30, tzinfo=UTC)
    # 16 extra calendar days from a weekday gives gap=5 non-trading days (> 3 threshold)
    _HOLIDAY_EXTRA_DAYS = 16
    # 2 extra calendar days from a weekday gives gap=0 non-trading days (< threshold)
    _WEEKEND_EXTRA_DAYS = 2

    def _make_brent_candles_with_gap(
        self,
        n: int = 80,
        gap_start_idx: int = 77,
        extra_calendar_days: int = 16,
        vary_prices: bool = False,
    ) -> tuple[Candle, ...]:
        """Make Brent candles with an artificial date gap at gap_start_idx.

        Candles 0..gap_start_idx-1 are spaced 1 day apart.
        Candle gap_start_idx onward is shifted by extra_calendar_days to simulate
        MOEX returning from an extended closure.
        When vary_prices=True, close prices follow a trend so z-score != 0.
        """
        candles: list[Candle] = []
        offset = 0
        for i in range(n):
            if i == gap_start_idx:
                offset += extra_calendar_days
            close = Decimal(str(70 + i * 0.5)) if vary_prices else Decimal(81)
            candles.append(
                Candle(
                    symbol="BZ=F",
                    market_id="us",
                    timeframe="1d",
                    timestamp=self._BASE + timedelta(days=i + offset),
                    open=close - Decimal(1),
                    high=close + Decimal(1),
                    low=close - Decimal(1),
                    close=close,
                    volume=1000,
                )
            )
        return tuple(candles)

    def test_suppressed_when_last_pair_in_lagged_has_extended_gap(self) -> None:
        """Gap at lagged[-1] position (last pair) → brent_zscore_60d=0.0."""
        # brent has 80 candles; lagged = brent[:-2] has 78 (indices 0..77).
        # gap_start_idx=77: gap is between lagged[76] and lagged[77] (i=-1 in check loop).
        # With _HOLIDAY_EXTRA_DAYS=16 from a Wednesday (2024-04-17), gap=5 > 3 → suppressed.
        # vary_prices ensures z-score would be non-zero absent suppression.
        brent = self._make_brent_candles_with_gap(
            n=80, gap_start_idx=77, extra_calendar_days=self._HOLIDAY_EXTRA_DAYS, vary_prices=True
        )
        moex = MoexMarketData(commodity_candles={"BZ=F": brent})
        result = _compute_commodity_features(moex)
        assert result["brent_zscore_60d"] == 0.0

    def test_suppressed_when_second_to_last_pair_in_lagged_has_extended_gap(self) -> None:
        """Gap at lagged[-2] position → brent_zscore_60d=0.0 (2-bar suppression window)."""
        # gap_start_idx=76: gap is between lagged[75] and lagged[76] (i=-2 in check loop).
        # With _HOLIDAY_EXTRA_DAYS=16 from a Tuesday (2024-04-16), gap=5 > 3 → suppressed.
        brent = self._make_brent_candles_with_gap(
            n=80, gap_start_idx=76, extra_calendar_days=self._HOLIDAY_EXTRA_DAYS, vary_prices=True
        )
        moex = MoexMarketData(commodity_candles={"BZ=F": brent})
        result = _compute_commodity_features(moex)
        assert result["brent_zscore_60d"] == 0.0

    def test_not_suppressed_when_gap_is_far_from_end(self) -> None:
        """Gap at index 10 (far from lagged window end) → suppression does NOT fire."""
        # Neither pair (75,76) nor (76,77) has the holiday gap → z-score is non-zero
        brent = self._make_brent_candles_with_gap(
            n=80, gap_start_idx=10, extra_calendar_days=self._HOLIDAY_EXTRA_DAYS, vary_prices=True
        )
        moex = MoexMarketData(commodity_candles={"BZ=F": brent})
        result = _compute_commodity_features(moex)
        # With trending prices, z-score should be non-zero (not suppressed to 0.0)
        assert result["brent_zscore_60d"] != 0.0

    def test_not_suppressed_for_regular_two_day_gap(self) -> None:
        """A 2-day calendar gap (0 non-trading days) never triggers suppression."""
        # extra_calendar_days=2 → gap=0 non-trading days → below threshold of 3 → not suppressed
        brent = self._make_brent_candles_with_gap(
            n=80, gap_start_idx=77, extra_calendar_days=self._WEEKEND_EXTRA_DAYS, vary_prices=True
        )
        moex = MoexMarketData(commodity_candles={"BZ=F": brent})
        result = _compute_commodity_features(moex)
        # With a 2-day gap the signal is not suppressed → non-zero z-score
        assert result["brent_zscore_60d"] != 0.0
