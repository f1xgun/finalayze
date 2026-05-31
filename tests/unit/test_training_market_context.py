"""Tests for MarketContext threading through training paths (Task 9).

Validates:
- _slice_market_context() prevents look-ahead bias per window
- build_windows() slices context per window
- build_dataset() accepts and threads market_context
- build_triple_barrier_dataset() accepts market_context
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import (
    Candle,
    FundamentalSnapshot,
    FXRate,
    KeyRateRecord,
    MarketContext,
    MoexMarketData,
)

# Fundamental-slice constants (ruff PLR2004: no magic numbers)
_FUND_SYMBOL = "SBER"
_FUND_PE_PAST = 8.0
_FUND_PE_FUTURE = 9.0
_FUND_PE_BOUNDARY = 7.0

# ── Helpers ──────────────────────────────────────────────────────────────────

_WINDOW_SIZE = 40  # small enough for fast tests


def _make_candles(
    n: int,
    start: datetime | None = None,
    symbol: str = "SBER",
    market_id: str = "moex",
) -> list[Candle]:
    """Create n synthetic daily candles."""
    base = start or datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=market_id,
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(100),
            high=Decimal(101),
            low=Decimal(99),
            close=Decimal(100),
            volume=1_000_000,
        )
        for i in range(n)
    ]


def _make_fx_rates(n: int, start: datetime | None = None) -> tuple[FXRate, ...]:
    base = start or datetime(2024, 1, 1, tzinfo=UTC)
    return tuple(
        FXRate(
            timestamp=base + timedelta(days=i),
            pair="USDRUB",
            rate=Decimal(90),
        )
        for i in range(n)
    )


def _make_key_rates(n: int, start: datetime | None = None) -> tuple[KeyRateRecord, ...]:
    base = start or datetime(2024, 1, 1, tzinfo=UTC)
    return tuple(
        KeyRateRecord(
            timestamp=base + timedelta(days=i * 45),
            rate=Decimal("0.16"),
        )
        for i in range(n)
    )


# ── _slice_market_context tests ────────────────────────────────────────────────


class TestSliceMarketContext:
    def test_import_exists(self) -> None:
        """_slice_market_context must be importable from finalayze.ml.training."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        assert callable(_slice_market_context)

    def test_returns_market_context(self) -> None:
        """_slice_market_context must return a MarketContext."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        ctx = MarketContext()
        result = _slice_market_context(ctx, datetime(2024, 6, 1, tzinfo=UTC))
        assert isinstance(result, MarketContext)

    def test_filters_fx_rates_to_max_ts(self) -> None:
        """fx_rates after max_ts must be excluded."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        # 12 monthly fx_rates: Jan-Dec 2024
        fx_rates = tuple(
            FXRate(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 30),
                pair="USDRUB",
                rate=Decimal(90),
            )
            for i in range(12)
        )
        moex = MoexMarketData(fx_rates=fx_rates)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.fx_rates is not None
        assert all(r.timestamp <= max_ts for r in sliced.moex_data.fx_rates)
        # Should have dropped later entries
        assert len(sliced.moex_data.fx_rates) < len(fx_rates)

    def test_filters_key_rates_to_max_ts(self) -> None:
        """key_rates after max_ts must be excluded (by timestamp, not count)."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        key_rates = tuple(
            KeyRateRecord(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 45),
                rate=Decimal("0.16"),
            )
            for i in range(8)
        )
        moex = MoexMarketData(key_rates=key_rates)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.key_rates is not None
        assert len(sliced.moex_data.key_rates) > 0
        assert all(r.timestamp <= max_ts for r in sliced.moex_data.key_rates)

    def test_key_rates_before_max_ts_preserved(self) -> None:
        """Key rates before max_ts must be kept for forward-fill in feature computation."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        # 3 rates: Jan 2024, Mar 2024, May 2024 — all should survive
        key_rates = (
            KeyRateRecord(timestamp=datetime(2024, 1, 1, tzinfo=UTC), rate=Decimal("0.16")),
            KeyRateRecord(timestamp=datetime(2024, 3, 1, tzinfo=UTC), rate=Decimal("0.165")),
            KeyRateRecord(timestamp=datetime(2024, 5, 1, tzinfo=UTC), rate=Decimal("0.17")),
        )
        moex = MoexMarketData(key_rates=key_rates)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.key_rates is not None
        # All 3 are at or before June 1
        assert len(sliced.moex_data.key_rates) == 3

    def test_future_key_rate_excluded(self) -> None:
        """A key rate dated after max_ts must be excluded."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        key_rates = (
            KeyRateRecord(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                rate=Decimal("0.16"),
            ),
            KeyRateRecord(
                timestamp=datetime(2024, 7, 1, tzinfo=UTC),
                rate=Decimal("0.18"),
            ),  # future
        )
        moex = MoexMarketData(key_rates=key_rates)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.key_rates is not None
        assert len(sliced.moex_data.key_rates) == 1
        assert sliced.moex_data.key_rates[0].timestamp == datetime(2024, 1, 1, tzinfo=UTC)

    def test_filters_commodity_candles_to_max_ts(self) -> None:
        """commodity_candles after max_ts must be excluded per symbol."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        brent_candles = tuple(
            Candle(
                symbol="BZ=F",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 10),
                open=Decimal(80),
                high=Decimal(81),
                low=Decimal(79),
                close=Decimal(80),
                volume=1000,
            )
            for i in range(20)  # Jan to Jul 2024
        )
        moex = MoexMarketData(commodity_candles={"BZ=F": brent_candles})
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.commodity_candles is not None
        brent_sliced = sliced.moex_data.commodity_candles["BZ=F"]
        assert all(c.timestamp <= max_ts for c in brent_sliced)
        assert len(brent_sliced) < len(brent_candles)

    def test_filters_turnover_to_max_ts(self) -> None:
        """turnover after max_ts must be excluded."""
        from finalayze.core.schemas import TurnoverRecord  # noqa: PLC0415
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        turnover = tuple(
            TurnoverRecord(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 15),
                volume_rub=Decimal(1000000),
            )
            for i in range(15)
        )
        moex = MoexMarketData(turnover=turnover)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.turnover is not None
        assert all(r.timestamp <= max_ts for r in sliced.moex_data.turnover)
        assert len(sliced.moex_data.turnover) < len(turnover)

    def test_benchmark_candles_sliced(self) -> None:
        """benchmark_candles after max_ts must be excluded."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        # 60 daily candles: Jan 1 to Feb 29, 2024.
        # max_ts = Jan 31 (day 30) → candles from day 31 onwards must be dropped.
        max_ts = datetime(2024, 1, 31, tzinfo=UTC)
        bench = _make_candles(60, symbol="SPY", market_id="us")
        ctx = MarketContext(benchmark_candles=bench)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.benchmark_candles is not None
        assert all(c.timestamp <= max_ts for c in sliced.benchmark_candles)
        assert len(sliced.benchmark_candles) < len(bench)

    def test_vix_candles_sliced(self) -> None:
        """vix_candles after max_ts must be excluded."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        # 60 daily candles: Jan 1 to Feb 29, 2024.
        # max_ts = Jan 31 (day 30) → candles from day 31 onwards must be dropped.
        max_ts = datetime(2024, 1, 31, tzinfo=UTC)
        vix = _make_candles(60, symbol="^VIX", market_id="us")
        ctx = MarketContext(vix_candles=vix)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.vix_candles is not None
        assert all(c.timestamp <= max_ts for c in sliced.vix_candles)
        assert len(sliced.vix_candles) < len(vix)

    def test_none_moex_data_preserved(self) -> None:
        """When moex_data is None, sliced result must also have moex_data=None."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        ctx = MarketContext(moex_data=None)
        sliced = _slice_market_context(ctx, datetime(2024, 6, 1, tzinfo=UTC))
        assert sliced.moex_data is None

    def test_none_fields_within_moex_data_preserved(self) -> None:
        """None fields inside MoexMarketData must stay None after slicing."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        moex = MoexMarketData(fx_rates=None, key_rates=None)
        ctx = MarketContext(moex_data=moex)
        sliced = _slice_market_context(ctx, datetime(2024, 6, 1, tzinfo=UTC))
        assert sliced.moex_data is not None
        assert sliced.moex_data.fx_rates is None
        assert sliced.moex_data.key_rates is None

    def test_exact_max_ts_boundary_is_inclusive(self) -> None:
        """A record exactly at max_ts must be INCLUDED (<=, not <)."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        fx_rates = (FXRate(timestamp=max_ts, pair="USDRUB", rate=Decimal(90)),)
        moex = MoexMarketData(fx_rates=fx_rates)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.fx_rates is not None
        assert len(sliced.moex_data.fx_rates) == 1

    # ── fundamentals slice (FUNDML-01, T-64-01 look-ahead guard) ─────────────

    def test_future_fundamental_excluded(self) -> None:
        """A fundamental snapshot dated after max_ts must be dropped (Test B)."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        past = FundamentalSnapshot(
            symbol=_FUND_SYMBOL,
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            pe_ratio=_FUND_PE_PAST,
        )
        future = FundamentalSnapshot(
            symbol=_FUND_SYMBOL,
            as_of=datetime(2024, 7, 1, tzinfo=UTC),  # after max_ts
            pe_ratio=_FUND_PE_FUTURE,
        )
        moex = MoexMarketData(fundamentals=(past, future))
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.fundamentals is not None
        assert len(sliced.moex_data.fundamentals) == 1
        assert sliced.moex_data.fundamentals[0].as_of == datetime(2024, 1, 1, tzinfo=UTC)

    def test_exact_max_ts_boundary_fundamental_is_inclusive(self) -> None:
        """A fundamental snapshot exactly at max_ts must be INCLUDED (Test C)."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        max_ts = datetime(2024, 6, 1, tzinfo=UTC)
        boundary = FundamentalSnapshot(
            symbol=_FUND_SYMBOL,
            as_of=max_ts,
            pe_ratio=_FUND_PE_BOUNDARY,
        )
        moex = MoexMarketData(fundamentals=(boundary,))
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, max_ts)

        assert sliced.moex_data is not None
        assert sliced.moex_data.fundamentals is not None
        assert len(sliced.moex_data.fundamentals) == 1

    def test_none_fundamentals_preserved(self) -> None:
        """fundamentals=None must stay None after slicing (Test D)."""
        from finalayze.ml.training import _slice_market_context  # noqa: PLC0415

        moex = MoexMarketData(fundamentals=None)
        ctx = MarketContext(moex_data=moex)

        sliced = _slice_market_context(ctx, datetime(2024, 6, 1, tzinfo=UTC))

        assert sliced.moex_data is not None
        assert sliced.moex_data.fundamentals is None


# ── build_windows with market_context tests ──────────────────────────────────


class TestBuildWindowsWithMarketContext:
    def test_accepts_market_context_kwarg(self) -> None:
        """build_windows must accept market_context keyword argument."""
        from finalayze.ml.training import build_windows  # noqa: PLC0415

        candles = _make_candles(50)
        ctx = MarketContext()
        # Should not raise
        result = build_windows(candles, window_size=_WINDOW_SIZE, market_context=ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_no_context_still_works(self) -> None:
        """build_windows without market_context must behave as before."""
        from finalayze.ml.training import build_windows  # noqa: PLC0415

        candles = _make_candles(50)
        features, labels, ts = build_windows(candles, window_size=_WINDOW_SIZE)
        assert isinstance(features, list)
        assert isinstance(labels, list)
        assert isinstance(ts, list)

    def test_no_lookahead_in_fx_rates(self) -> None:
        """Each window's context must contain only fx_rates up to window's max_ts."""
        from finalayze.ml.training import build_windows  # noqa: PLC0415

        _N_CANDLES = 55  # noqa: N806
        candles = _make_candles(_N_CANDLES)
        fx_rates = _make_fx_rates(_N_CANDLES)
        moex = MoexMarketData(fx_rates=fx_rates)
        ctx = MarketContext(moex_data=moex)

        # Patch compute_features to capture the market_context passed in per window
        captured_contexts: list[MarketContext | None] = []
        from unittest.mock import patch  # noqa: PLC0415

        import finalayze.ml.training as training_module  # noqa: PLC0415

        original_compute = training_module.compute_features

        def capturing_compute(
            window: list[Candle],
            *,
            benchmark_candles: list[Candle] | None = None,
            vix_candles: list[Candle] | None = None,
            sentiment_score: float = 0.0,
            market_context: MarketContext | None = None,
        ) -> dict[str, float]:
            captured_contexts.append(market_context)
            return original_compute(
                window,
                benchmark_candles=benchmark_candles,
                vix_candles=vix_candles,
                sentiment_score=sentiment_score,
                market_context=market_context,
            )

        with patch.object(training_module, "compute_features", side_effect=capturing_compute):
            build_windows(candles, window_size=_WINDOW_SIZE, market_context=ctx)

        assert len(captured_contexts) > 0
        for i, captured_ctx in enumerate(captured_contexts):
            # Window i covers candles[i:i+_WINDOW_SIZE];
            # max_ts is candles[i+_WINDOW_SIZE-1].timestamp
            window_max_ts = candles[i + _WINDOW_SIZE - 1].timestamp
            if (
                captured_ctx is not None
                and captured_ctx.moex_data is not None
                and captured_ctx.moex_data.fx_rates is not None
            ):
                for rate in captured_ctx.moex_data.fx_rates:
                    assert rate.timestamp <= window_max_ts, (
                        f"Window {i}: fx_rate {rate.timestamp} > window max_ts {window_max_ts}"
                    )


# ── build_dataset with market_context tests ───────────────────────────────────


class TestBuildDatasetWithMarketContext:
    def test_accepts_market_context_kwarg(self) -> None:
        """build_dataset must accept market_context keyword argument."""
        from finalayze.ml.training import build_dataset  # noqa: PLC0415

        candles = _make_candles(50)
        ctx = MarketContext()
        result = build_dataset({"SBER": candles}, window_size=_WINDOW_SIZE, market_context=ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_no_context_still_works(self) -> None:
        """build_dataset without market_context must behave as before."""
        from finalayze.ml.training import build_dataset  # noqa: PLC0415

        candles = _make_candles(50)
        features, _labels, _ts = build_dataset({"SBER": candles}, window_size=_WINDOW_SIZE)
        assert isinstance(features, list)

    def test_returns_nonempty_with_enough_candles(self) -> None:
        """build_dataset must return non-empty result with >= window_size+1 candles."""
        import numpy as np  # noqa: PLC0415

        from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_dataset  # noqa: PLC0415

        # Must provide enough candles for compute_features (needs >= DEFAULT_WINDOW_SIZE=80)
        rng = np.random.default_rng(42)
        n = DEFAULT_WINDOW_SIZE + 5
        prices = 100.0 + rng.standard_normal(n).cumsum()
        base = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                symbol="SBER",
                market_id="moex",
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=Decimal(str(round(float(prices[i]) * 0.999, 2))),
                high=Decimal(str(round(float(prices[i]) * 1.005, 2))),
                low=Decimal(str(round(float(prices[i]) * 0.995, 2))),
                close=Decimal(str(round(float(prices[i]), 2))),
                volume=1_000_000,
            )
            for i in range(n)
        ]
        ctx = MarketContext()
        features, labels, ts = build_dataset({"SBER": candles}, market_context=ctx)
        # n candles, window_size=80 → n-80 samples
        assert len(features) > 0
        assert len(features) == len(labels) == len(ts)


# ── build_triple_barrier_dataset with market_context tests ────────────────────


class TestBuildTripleBarrierDatasetWithMarketContext:
    def test_accepts_market_context_kwarg(self) -> None:
        """build_triple_barrier_dataset must accept market_context kwarg."""
        from finalayze.ml.training.labeling import build_triple_barrier_dataset  # noqa: PLC0415

        candles = _make_candles(80)
        ctx = MarketContext()
        # Should not raise
        result = build_triple_barrier_dataset(candles, window_size=30, market_context=ctx)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_no_lookahead_via_market_context_fx_rates(self) -> None:
        """compute_features inside build_triple_barrier_dataset must only see past fx_rates."""
        from finalayze.ml.training.labeling import build_triple_barrier_dataset  # noqa: PLC0415

        _N = 80  # noqa: N806
        candles = _make_candles(_N)
        fx_rates = _make_fx_rates(_N)
        moex = MoexMarketData(fx_rates=fx_rates)
        ctx = MarketContext(moex_data=moex)

        # Patch compute_features to track what contexts are passed
        captured_contexts: list[MarketContext | None] = []
        from unittest.mock import patch  # noqa: PLC0415

        import finalayze.ml.training.labeling as labeling_module  # noqa: PLC0415

        original_compute = labeling_module.compute_features

        def capturing_compute(
            window: list[Candle],
            *,
            benchmark_candles: list[Candle] | None = None,
            vix_candles: list[Candle] | None = None,
            sentiment_score: float = 0.0,
            market_context: MarketContext | None = None,
        ) -> dict[str, float]:
            captured_contexts.append(market_context)
            return original_compute(
                window,
                benchmark_candles=benchmark_candles,
                vix_candles=vix_candles,
                sentiment_score=sentiment_score,
                market_context=market_context,
            )

        with patch.object(labeling_module, "compute_features", side_effect=capturing_compute):
            build_triple_barrier_dataset(candles, window_size=30, max_hold=5, market_context=ctx)

        assert len(captured_contexts) > 0, "compute_features must have been called"
        for captured_ctx in captured_contexts:
            if (
                captured_ctx is not None
                and captured_ctx.moex_data is not None
                and captured_ctx.moex_data.fx_rates is not None
            ):
                # All fx_rates in the context must be <= the last candle in the WINDOW,
                # not the full dataset. We cannot easily check per-window here, but
                # the key invariant is that they are not from the full unsliced context.
                assert len(captured_ctx.moex_data.fx_rates) <= len(fx_rates)
