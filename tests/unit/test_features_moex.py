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
