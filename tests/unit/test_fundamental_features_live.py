"""D-03 full-path fundamental guard (FUNDML-01).

This is the cardinal RED->GREEN guard for Phase 64 Plan 01. It proves that
``compute_features`` (NOT just ``compute_fundamental_features``) emits a NON-ZERO
fundamental feature once a backfilled ``MarketContext.moex_data.fundamentals`` is
carried through the loader/slice path.

Pre-fix (RED): the two frozen-dataclass reconstruction sites (``_load_moex`` and
``_slice_market_context``) drop ``fundamentals=``, so the per-window context the
training pipeline feeds into ``compute_features`` always sees the all-zero
``_DEFAULT`` and ``earnings_yield == 0.0``. This test routes a populated
``MarketContext`` through ``_slice_market_context`` (the genuine per-window
training path — exactly what ``build_windows`` does) BEFORE calling
``compute_features``, so it catches Pitfall 3 (the slice silently dropping
fundamentals). Pre-fix the slice returns ``fundamentals=None`` and the guard
fails.

Post-fix (GREEN): once the slice carries ``fundamentals``, ``compute_features``
delivers a live, non-zero ``earnings_yield``.

No live data / token required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from finalayze.core.schemas import FundamentalSnapshot, MarketContext, MoexMarketData
from finalayze.ml.features.technical import compute_features
from finalayze.ml.training import _slice_market_context

# ── Named constants (ruff PLR2004: no magic numbers) ─────────────────────────
_SYMBOL = "SBER"
_PEERS = ("LKOH", "GMKN", "ROSN", "NVTK", "MGNT")  # >= 4 ru_blue_chips peers
_MARKET_ID = "moex"
_TIMEFRAME = "1d"

# Candle window: 90 daily bars (>= compute_features' _MIN_CANDLES=80) ending on D.
_N_CANDLES = 90
_WINDOW_END = datetime(2023, 6, 30, tzinfo=UTC)
_ONE_DAY = timedelta(days=1)

# Fundamental snapshot is dated strictly before the last bar (as_of <= D).
_SNAPSHOT_AS_OF = datetime(2023, 6, 1, tzinfo=UTC)
_PE_RATIO = 8.0  # positive -> earnings_yield = 1/8 = 0.125 (non-zero)
_EXPECTED_EARNINGS_YIELD = 1.0 / _PE_RATIO
_EPS_TTM = 50.0
_REVENUE_TTM = 2.0e12
_DIVIDEND_YIELD = 0.07
_NET_MARGIN = 0.25

# Per-peer offsets so peers carry plausibly distinct fundamentals.
_PEER_PE_BASE = 6.0
_PEER_PE_STEP = 1.5
_PEER_DIV_BASE = 0.04
_PEER_DIV_STEP = 0.01

_FUNDAMENTAL_KEYS = (
    "earnings_yield",
    "pe_zscore_vs_sector",
    "revenue_growth_yoy",
    "net_margin_trend",
    "dividend_yield_z",
)

# Prices form a gentle uptrend; values are not load-bearing for this guard.
_BASE_PRICE = 250.0
_PRICE_STEP = 0.5


def _make_window() -> list:
    """90 ascending daily candles ending at _WINDOW_END."""
    from decimal import Decimal

    from finalayze.core.schemas import Candle

    start = _WINDOW_END - (_N_CANDLES - 1) * _ONE_DAY
    candles: list[Candle] = []
    for i in range(_N_CANDLES):
        price = _BASE_PRICE + i * _PRICE_STEP
        candles.append(
            Candle(
                symbol=_SYMBOL,
                market_id=_MARKET_ID,
                timeframe=_TIMEFRAME,
                timestamp=start + i * _ONE_DAY,
                open=Decimal(str(price)),
                high=Decimal(str(price + 1.0)),
                low=Decimal(str(price - 1.0)),
                close=Decimal(str(price)),
                volume=1_000_000,
            )
        )
    return candles


def _make_fundamentals() -> tuple[FundamentalSnapshot, ...]:
    """SBER target + >= 4 peer snapshots, all dated as_of <= D."""
    target = FundamentalSnapshot(
        symbol=_SYMBOL,
        as_of=_SNAPSHOT_AS_OF,
        pe_ratio=_PE_RATIO,
        eps_ttm=_EPS_TTM,
        revenue_ttm=_REVENUE_TTM,
        dividend_yield=_DIVIDEND_YIELD,
        net_margin=_NET_MARGIN,
        currency="RUB",
    )
    peers = tuple(
        FundamentalSnapshot(
            symbol=peer,
            as_of=_SNAPSHOT_AS_OF,
            pe_ratio=_PEER_PE_BASE + idx * _PEER_PE_STEP,
            dividend_yield=_PEER_DIV_BASE + idx * _PEER_DIV_STEP,
            currency="RUB",
        )
        for idx, peer in enumerate(_PEERS)
    )
    return (target, *peers)


class TestFundamentalFeaturesLive:
    def test_compute_features_emits_nonzero_earnings_yield(self) -> None:
        """D-03 guard: compute_features delivers all 5 fundamental keys AND a
        non-zero earnings_yield when moex_data.fundamentals is populated on a
        backfilled as_of (RED before the loader/slice fixes; GREEN after)."""
        window = _make_window()
        full_ctx = MarketContext(
            moex_data=MoexMarketData(fundamentals=_make_fundamentals()),
        )
        # Route through the genuine per-window training path: the slice runs
        # before compute_features inside build_windows. Pre-fix the slice drops
        # fundamentals (Pitfall 3) and earnings_yield collapses to the 0.0 default.
        window_max_ts = window[-1].timestamp
        ctx = _slice_market_context(full_ctx, window_max_ts)

        features = compute_features(window, market_context=ctx)

        # (1) All 5 fundamental keys present in the output dict.
        for key in _FUNDAMENTAL_KEYS:
            assert key in features, f"missing fundamental feature key: {key}"

        # (2) earnings_yield is LIVE (not the all-zero _DEFAULT).
        assert features["earnings_yield"] != 0.0
        assert features["earnings_yield"] == _EXPECTED_EARNINGS_YIELD
