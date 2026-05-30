"""Tests for look-ahead-safe fundamental features (FUND-02, Layer 3)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from finalayze.core.schemas import FundamentalSnapshot, MoexMarketData
from finalayze.ml.features.fundamental import (
    _MIN_PEERS_FOR_ZSCORE,
    _resolve_sector_peer_symbols,
    compute_fundamental_features,
)

_EXPECTED_KEYS = {
    "earnings_yield",
    "pe_zscore_vs_sector",
    "revenue_growth_yoy",
    "net_margin_trend",
    "dividend_yield_z",
}

_D = datetime(2025, 3, 1, tzinfo=UTC)


def _snapshot(
    symbol: str,
    as_of: datetime,
    *,
    pe_ratio: float | None = None,
    eps_ttm: float | None = None,
    net_margin: float | None = None,
    dividend_yield: float | None = None,
    revenue_ttm: float | None = None,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        as_of=as_of,
        pe_ratio=pe_ratio,
        eps_ttm=eps_ttm,
        net_margin=net_margin,
        dividend_yield=dividend_yield,
        revenue_ttm=revenue_ttm,
    )


class TestKeysPresent:
    def test_keys_present_and_earnings_yield(self) -> None:
        """All five keys present; earnings_yield == 1/pe_ratio."""
        moex = MoexMarketData(
            fundamentals=(
                _snapshot(
                    "SBER",
                    _D,
                    pe_ratio=5.0,
                    eps_ttm=120.0,
                    net_margin=0.3,
                    dividend_yield=0.11,
                    revenue_ttm=3e12,
                ),
            ),
        )
        result = compute_fundamental_features(moex, as_of=_D)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert result["earnings_yield"] == pytest.approx(1 / 5.0)


class TestDefaultNotNaN:
    def test_none_moex_data_returns_zero_defaults(self) -> None:
        result = compute_fundamental_features(None)
        assert set(result.keys()) == _EXPECTED_KEYS
        for value in result.values():
            assert value == 0.0
            assert not math.isnan(value)

    def test_none_fundamentals_returns_zero_defaults(self) -> None:
        result = compute_fundamental_features(MoexMarketData(fundamentals=None))
        assert set(result.keys()) == _EXPECTED_KEYS
        for value in result.values():
            assert value == 0.0
            assert not math.isnan(value)


class TestLookAhead:
    """A future-dated (as_of > D) snapshot must NOT influence features at D."""

    def test_spike_after_d_does_not_change_output(self) -> None:
        base = _snapshot("SBER", _D - timedelta(days=30), pe_ratio=5.0, revenue_ttm=2e12)
        spike_future = _snapshot(
            "SBER",
            _D + timedelta(days=30),
            pe_ratio=999.0,
            revenue_ttm=9e12,
        )

        moex_no_spike = MoexMarketData(fundamentals=(base,))
        moex_future_spike = MoexMarketData(fundamentals=(base, spike_future))

        result_clean = compute_fundamental_features(moex_no_spike, as_of=_D)
        result_future = compute_fundamental_features(moex_future_spike, as_of=_D)

        assert result_future == result_clean

    def test_spike_at_or_before_d_changes_output(self) -> None:
        base = _snapshot("SBER", _D - timedelta(days=400), pe_ratio=5.0, revenue_ttm=2e12)
        # Same spike snapshot, but now dated <= D — it must be picked up.
        spike_past = _snapshot("SBER", _D - timedelta(days=1), pe_ratio=50.0, revenue_ttm=9e12)

        moex_no_spike = MoexMarketData(fundamentals=(base,))
        moex_past_spike = MoexMarketData(fundamentals=(base, spike_past))

        result_clean = compute_fundamental_features(moex_no_spike, as_of=_D)
        result_past = compute_fundamental_features(moex_past_spike, as_of=_D)

        assert result_past != result_clean


class TestTinyPeerGuard:
    def test_two_peers_yields_default_zscore(self) -> None:
        target = _snapshot("SBER", _D, pe_ratio=5.0, dividend_yield=0.10)
        peers = (
            _snapshot("LKOH", _D, pe_ratio=8.0, dividend_yield=0.12),
            _snapshot("GMKN", _D, pe_ratio=10.0, dividend_yield=0.09),
        )
        moex = MoexMarketData(fundamentals=(target,))
        result = compute_fundamental_features(moex, as_of=_D, sector_peers=peers)
        assert len(peers) < _MIN_PEERS_FOR_ZSCORE
        assert result["pe_zscore_vs_sector"] == 0.0

    def test_four_plus_peers_yields_finite_nondefault_zscore(self) -> None:
        target = _snapshot("SBER", _D, pe_ratio=2.0, dividend_yield=0.30)
        peers = (
            _snapshot("LKOH", _D, pe_ratio=8.0, dividend_yield=0.05),
            _snapshot("GMKN", _D, pe_ratio=9.0, dividend_yield=0.06),
            _snapshot("ROSN", _D, pe_ratio=10.0, dividend_yield=0.07),
            _snapshot("TATN", _D, pe_ratio=11.0, dividend_yield=0.08),
        )
        moex = MoexMarketData(fundamentals=(target,))
        result = compute_fundamental_features(moex, as_of=_D, sector_peers=peers)
        assert len(peers) >= _MIN_PEERS_FOR_ZSCORE
        assert result["pe_zscore_vs_sector"] != 0.0
        assert math.isfinite(result["pe_zscore_vs_sector"])


class TestInternalSectorResolution:
    """Closes BLOCKER 2: peers resolved internally from config/segments.py."""

    def test_resolve_returns_owning_segment_symbols(self) -> None:
        from config.segments import DEFAULT_SEGMENTS

        peers = _resolve_sector_peer_symbols("ROSN")
        assert "ROSN" in peers
        owning = next(seg for seg in DEFAULT_SEGMENTS if "ROSN" in seg.symbols)
        assert peers == tuple(owning.symbols)

    def test_resolve_unknown_symbol_returns_empty(self) -> None:
        assert _resolve_sector_peer_symbols("NOT_A_TICKER") == ()

    def test_internal_resolution_produces_nondefault_zscore(self) -> None:
        # ROSN's segment (ru_energy) has >= 4 symbols. Build fundamentals that
        # include >= 4 peers so the INTERNAL resolution (sector_peers=None) yields
        # a non-default z-score without any peer set being passed in.
        target = _snapshot("ROSN", _D, pe_ratio=2.0)
        peers = (
            _snapshot("TATN", _D, pe_ratio=8.0),
            _snapshot("NVTK", _D, pe_ratio=9.0),
            _snapshot("SIBN", _D, pe_ratio=10.0),
            _snapshot("TATNP", _D, pe_ratio=11.0),
            _snapshot("TRNFP", _D, pe_ratio=12.0),
        )
        moex = MoexMarketData(fundamentals=(target, *peers))
        result = compute_fundamental_features(moex, as_of=_D)  # no sector_peers passed
        assert result["pe_zscore_vs_sector"] != 0.0
        assert math.isfinite(result["pe_zscore_vs_sector"])
