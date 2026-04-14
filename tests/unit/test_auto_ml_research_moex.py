"""Unit tests for MOEX segment detection and symbol loading in auto_ml_research.py.

Tests cover:
- _is_moex_segment helper
- _SEGMENT_SYMBOLS contains all 4 ru_* equity segments
- _SEGMENT_SYMBOLS does NOT contain bond segments
- _get_lookback_days returns segment-appropriate values
- _get_max_features returns segment-appropriate values
- argparse --segment choices include all 4 ru_* equity segments
- Macro shift(1+) no-lookahead bias: last FX spike not in feature
- Macro features non-zero when realistic MoexMarketData is supplied
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure scripts/ and project root are importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Constants (no magic numbers — ruff PLR2004)
_MOEX_LOOKBACK_DAYS_EXPECTED = 1095
_US_LOOKBACK_DAYS_EXPECTED = 1825
_MOEX_MAX_FEATURES_EXPECTED = 10
_US_MAX_FEATURES_EXPECTED = 15

_RU_EQUITY_SEGMENTS = ["ru_blue_chips", "ru_energy", "ru_tech", "ru_finance"]
_BOND_SEGMENTS = ["ru_ofz_pd", "ru_ofz_pk"]

# Expected symbols for ru_blue_chips from config/segments.py
_RU_BLUE_CHIPS_SYMBOLS = ["SBER", "LKOH", "GMKN"]


class TestBarrierConfig:
    """Tests for _SEGMENT_BARRIER_CONFIG and _get_barrier_params."""

    def test_ru_energy_asymmetric(self) -> None:
        from scripts.auto_ml_research import _get_barrier_params

        upper, lower = _get_barrier_params("ru_energy")
        assert upper == pytest.approx(1.8)   # 1.5 * 1.2
        assert lower == pytest.approx(2.4)   # 2.0 * 1.2

    def test_ru_energy_lower_wider_than_upper(self) -> None:
        from scripts.auto_ml_research import _get_barrier_params

        upper, lower = _get_barrier_params("ru_energy")
        assert lower > upper

    def test_ru_finance_symmetric(self) -> None:
        from scripts.auto_ml_research import _get_barrier_params

        upper, lower = _get_barrier_params("ru_finance")
        assert upper == pytest.approx(2.4)   # 2.0 * 1.2
        assert lower == pytest.approx(2.4)   # 2.0 * 1.2

    def test_us_tech_no_uplift(self) -> None:
        from scripts.auto_ml_research import _get_barrier_params

        upper, lower = _get_barrier_params("us_tech")
        assert upper == pytest.approx(2.0)
        assert lower == pytest.approx(2.0)

    def test_config_driven(self) -> None:
        """Changing _SEGMENT_BARRIER_CONFIG affects output."""
        from scripts.auto_ml_research import _SEGMENT_BARRIER_CONFIG, _get_barrier_params

        original = _SEGMENT_BARRIER_CONFIG.get("ru_energy")
        try:
            _SEGMENT_BARRIER_CONFIG["ru_energy"] = (1.0, 3.0)
            upper, lower = _get_barrier_params("ru_energy")
            assert upper == pytest.approx(1.2)   # 1.0 * 1.2
            assert lower == pytest.approx(3.6)   # 3.0 * 1.2
        finally:
            if original is not None:
                _SEGMENT_BARRIER_CONFIG["ru_energy"] = original


class TestIsModexSegment:
    """Test _is_moex_segment helper function."""

    def test_ru_segment_returns_true(self) -> None:
        """_is_moex_segment('ru_blue_chips') returns True."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("ru_blue_chips") is True

    def test_ru_energy_returns_true(self) -> None:
        """_is_moex_segment('ru_energy') returns True."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("ru_energy") is True

    def test_us_segment_returns_false(self) -> None:
        """_is_moex_segment('us_tech') returns False."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("us_tech") is False

    def test_us_healthcare_returns_false(self) -> None:
        """_is_moex_segment('us_healthcare') returns False."""
        from scripts.auto_ml_research import _is_moex_segment

        assert _is_moex_segment("us_healthcare") is False


class TestSegmentSymbols:
    """Test _SEGMENT_SYMBOLS dict contains correct ru_* segments."""

    def test_ru_blue_chips_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_blue_chips' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_blue_chips" in _SEGMENT_SYMBOLS

    def test_ru_energy_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_energy' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_energy" in _SEGMENT_SYMBOLS

    def test_ru_tech_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_tech' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_tech" in _SEGMENT_SYMBOLS

    def test_ru_finance_in_segment_symbols(self) -> None:
        """_SEGMENT_SYMBOLS contains 'ru_finance' key."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_finance" in _SEGMENT_SYMBOLS

    def test_ru_blue_chips_symbols_match_config(self) -> None:
        """_SEGMENT_SYMBOLS['ru_blue_chips'] matches DEFAULT_SEGMENTS symbols."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        symbols = _SEGMENT_SYMBOLS["ru_blue_chips"]
        assert symbols == _RU_BLUE_CHIPS_SYMBOLS

    def test_bond_segment_ofz_pd_excluded(self) -> None:
        """_SEGMENT_SYMBOLS does NOT contain 'ru_ofz_pd' (bond segment)."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_ofz_pd" not in _SEGMENT_SYMBOLS

    def test_bond_segment_ofz_pk_excluded(self) -> None:
        """_SEGMENT_SYMBOLS does NOT contain 'ru_ofz_pk' (bond segment)."""
        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        assert "ru_ofz_pk" not in _SEGMENT_SYMBOLS


class TestGetLookbackDays:
    """Test _get_lookback_days returns segment-appropriate values."""

    def test_ru_blue_chips_returns_1095(self) -> None:
        """_get_lookback_days('ru_blue_chips') returns 1095."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("ru_blue_chips") == _MOEX_LOOKBACK_DAYS_EXPECTED

    def test_ru_energy_returns_1095(self) -> None:
        """_get_lookback_days('ru_energy') returns 1095."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("ru_energy") == _MOEX_LOOKBACK_DAYS_EXPECTED

    def test_us_tech_returns_1825(self) -> None:
        """_get_lookback_days('us_tech') returns 1825."""
        from scripts.auto_ml_research import _get_lookback_days

        assert _get_lookback_days("us_tech") == _US_LOOKBACK_DAYS_EXPECTED


class TestGetMaxFeatures:
    """Test _get_max_features returns segment-appropriate values."""

    def test_ru_blue_chips_returns_10(self) -> None:
        """_get_max_features('ru_blue_chips') returns 10."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("ru_blue_chips") == _MOEX_MAX_FEATURES_EXPECTED

    def test_ru_tech_returns_10(self) -> None:
        """_get_max_features('ru_tech') returns 10."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("ru_tech") == _MOEX_MAX_FEATURES_EXPECTED

    def test_us_tech_returns_15(self) -> None:
        """_get_max_features('us_tech') returns 15."""
        from scripts.auto_ml_research import _get_max_features

        assert _get_max_features("us_tech") == _US_MAX_FEATURES_EXPECTED


class TestArgparseChoices:
    """Test argparse --segment choices include all 4 ru_* equity segments."""

    def test_ru_blue_chips_in_choices(self) -> None:
        """argparse choices include 'ru_blue_chips'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        # Should not raise
        args = parser.parse_args(["--segment", "ru_blue_chips"])
        assert args.segment == "ru_blue_chips"

    def test_ru_energy_in_choices(self) -> None:
        """argparse choices include 'ru_energy'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_energy"])
        assert args.segment == "ru_energy"

    def test_ru_tech_in_choices(self) -> None:
        """argparse choices include 'ru_tech'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_tech"])
        assert args.segment == "ru_tech"

    def test_ru_finance_in_choices(self) -> None:
        """argparse choices include 'ru_finance'."""
        import argparse

        from scripts.auto_ml_research import _SEGMENT_SYMBOLS

        parser = argparse.ArgumentParser()
        parser.add_argument("--segment", choices=list(_SEGMENT_SYMBOLS.keys()))
        args = parser.parse_args(["--segment", "ru_finance"])
        assert args.segment == "ru_finance"


# ---------------------------------------------------------------------------
# Look-ahead bias and macro feature plumbing tests (Plan 02)
# ---------------------------------------------------------------------------

# Constants for macro tests (no magic numbers — ruff PLR2004)
_CANDLE_COUNT = 200  # random-walk series; enough for _WINDOW_SIZE=80 + _TB_MAX_HOLD=20 + warmup
_MACRO_COUNT = 200  # one record per day matching candle count
_STABLE_FX_RATE = Decimal(80)
_SPIKE_FX_RATE = Decimal(200)  # large spike injected into the last 2 records
_SPIKE_ABS_ZSCORE_LIMIT = 3.0  # spike must NOT produce extreme z-score

_KEY_RATE_DECIMAL = Decimal("0.16")  # 16%
_BRENT_BASE_PRICE = Decimal(75)
_TURNOVER_RUB = Decimal(1000000)

# Minimum number of MOEX feature keys expected when macro data is wired
_MIN_MOEX_FEATURE_KEYS = 3

# Seed for reproducible random-walk candles
_RW_SEED = 42
_RW_DRIFT = 0.02  # daily volatility (2%) for non-zero ATR


def _make_candles(n: int, base_ts: datetime | None = None, symbol: str = "TEST") -> list:
    """Create n synthetic daily candles with a seeded random-walk price series.

    Uses non-zero ATR so build_triple_barrier_dataset produces labels.
    """
    import random

    from finalayze.core.schemas import Candle

    if base_ts is None:
        base_ts = datetime(2022, 1, 1, tzinfo=UTC)

    rng = random.Random(_RW_SEED)  # noqa: S311 (test data, not security)
    price = 100.0
    candles = []
    for i in range(n):
        price *= 1 + rng.gauss(0, _RW_DRIFT)
        open_ = Decimal(str(round(price * 0.99, 2)))
        close_ = Decimal(str(round(price, 2)))
        high_ = Decimal(str(round(price * 1.01, 2)))
        low_ = Decimal(str(round(price * 0.98, 2)))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="ru",
                timeframe="1d",
                timestamp=base_ts + timedelta(days=i),
                open=open_,
                high=high_,
                low=low_,
                close=close_,
                volume=1000 + i,
            )
        )
    return candles


def _make_fx_rates(n: int, rate: Decimal, base_ts: datetime | None = None) -> list:
    """Create n FXRate records with the given rate."""
    from finalayze.core.schemas import FXRate

    if base_ts is None:
        base_ts = datetime(2022, 1, 1, tzinfo=UTC)
    return [
        FXRate(
            timestamp=base_ts + timedelta(days=i),
            pair="USDRUB",
            rate=rate,
        )
        for i in range(n)
    ]


class TestMacroShift2NoLookahead:
    """Test that _EXTERNAL_DATA_LAG_BARS prevents future macro values leaking into features."""

    @pytest.mark.slow
    def test_macro_shift2_no_lookahead(self) -> None:
        """Spike in last 2 FX records must NOT appear in usdrub_zscore_60d.

        _EXTERNAL_DATA_LAG_BARS=2 excludes the last 2 records before computing
        z-score, so a spike injected at position [-1] and [-2] should be invisible.
        The z-score from all-stable lagged data must stay in normal range.
        """
        from scripts.auto_ml_research import build_full_dataset

        from finalayze.core.schemas import MoexMarketData

        candles = _make_candles(_CANDLE_COUNT)

        # Build FX rates: stable for first (n-2), then spike in last 2
        stable = _make_fx_rates(_MACRO_COUNT - 2, _STABLE_FX_RATE)
        spike = _make_fx_rates(
            2,
            _SPIKE_FX_RATE,
            base_ts=datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=_MACRO_COUNT - 2),
        )
        fx_rates = stable + spike

        moex_data = MoexMarketData(fx_rates=tuple(fx_rates))

        features, _labels, _w, _h, _ts = build_full_dataset(
            "ru_blue_chips",
            {"TEST": candles},
            None,
            None,
            moex_data=moex_data,
        )

        assert features, "Expected non-empty feature list"

        # The last sample should NOT reflect the spike (lag hides last 2 records)
        last_feat = features[-1]
        if "usdrub_zscore_60d" in last_feat:
            assert abs(last_feat["usdrub_zscore_60d"]) < _SPIKE_ABS_ZSCORE_LIMIT, (
                f"usdrub_zscore_60d={last_feat['usdrub_zscore_60d']:.2f} reflects spike — "
                f"look-ahead bias not prevented by lag"
            )


class TestMoexMacroFeaturesNonZero:
    """Test that MoexMarketData flows through build_full_dataset to produce non-zero features."""

    @pytest.mark.slow
    def test_moex_macro_features_nonzero(self) -> None:
        """With realistic macro data, at least 3 MOEX feature keys must be present
        and at least one must be non-zero.
        """
        from scripts.auto_ml_research import build_full_dataset

        from finalayze.core.schemas import FXRate, KeyRateRecord, MoexMarketData, TurnoverRecord

        _N = 200
        base_ts = datetime(2022, 1, 1, tzinfo=UTC)

        candles = _make_candles(_N, base_ts)

        # Realistic FX: slight upward drift 80..82
        fx_rates = [
            FXRate(
                timestamp=base_ts + timedelta(days=i),
                pair="USDRUB",
                rate=Decimal(80) + Decimal(str(round(i * 0.01, 2))),
            )
            for i in range(_N)
        ]

        # Key rates: 10 quarterly records (sparse — forward-filled internally)
        key_rates = [
            KeyRateRecord(
                timestamp=base_ts + timedelta(days=i * 30),
                rate=_KEY_RATE_DECIMAL,
            )
            for i in range(10)
        ]

        # Brent candles: _N records with slight upward drift for non-zero z-score
        brent_candles = _make_candles(_N, base_ts, symbol="BZ=F")

        # Turnover: _N records
        turnover = [
            TurnoverRecord(
                timestamp=base_ts + timedelta(days=i),
                volume_rub=_TURNOVER_RUB,
            )
            for i in range(_N)
        ]

        moex_data = MoexMarketData(
            fx_rates=tuple(fx_rates),
            key_rates=tuple(key_rates),
            commodity_candles={"BZ=F": tuple(brent_candles)},
            turnover=tuple(turnover),
        )

        features, _labels, _w, _h, _ts = build_full_dataset(
            "ru_blue_chips",
            {"TEST": candles},
            None,
            None,
            moex_data=moex_data,
        )

        assert features, "Expected non-empty feature list"

        _EXPECTED_MOEX_KEYS = {
            "usdrub_zscore_60d",
            "brent_zscore_60d",
            "cbr_rate_level",
            "cbr_rate_delta",
            "cbr_direction_cut",
            "cbr_direction_hike",
            "real_rate_zscore",
            "market_turnover_zscore",
            "usdrub_return",
            "usdrub_vol",
            "brent_return",
        }

        present_keys = _EXPECTED_MOEX_KEYS & set(features[0].keys())
        assert len(present_keys) >= _MIN_MOEX_FEATURE_KEYS, (
            f"Expected at least {_MIN_MOEX_FEATURE_KEYS} MOEX feature keys, "
            f"got {len(present_keys)}: {present_keys}"
        )

        # At least one MOEX feature must be non-zero
        any_nonzero = any(features[0].get(k, 0.0) != 0.0 for k in present_keys)
        assert any_nonzero, (
            f"All MOEX features are zero — MoexMarketData not flowing through pipeline. "
            f"Present keys: {present_keys}, "
            f"values: { {k: features[0].get(k) for k in present_keys} }"
        )


# ---------------------------------------------------------------------------
# MOEX hyperparameter routing tests (Plan 45-01)
# ---------------------------------------------------------------------------

# Constants for MOEX hparam tests (no magic numbers — ruff PLR2004)
_MOEX_EXPECTED_XGB_MAX_DEPTH = 3
_MOEX_EXPECTED_XGB_N_ESTIMATORS = 100
_MOEX_EXPECTED_XGB_MIN_CHILD_WEIGHT = 20
_MOEX_EXPECTED_LGBM_N_ESTIMATORS = 100
_MOEX_EXPECTED_LGBM_NUM_LEAVES = 15
_MOEX_EXPECTED_CAT_DEPTH = 3
_MOEX_EXPECTED_CAT_ITERATIONS = 100

_US_EXPECTED_XGB_MAX_DEPTH = 5
_US_EXPECTED_XGB_N_ESTIMATORS = 200


class TestMoexHparams:
    """Test MOEX-specific reduced-complexity hyperparameters and routing."""

    def test_moex_hparams_xgb_max_depth(self) -> None:
        """_MOEX_HPARAMS['xgb_max_depth'] == 3."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["xgb_max_depth"] == _MOEX_EXPECTED_XGB_MAX_DEPTH

    def test_moex_hparams_xgb_n_estimators(self) -> None:
        """_MOEX_HPARAMS['xgb_n_estimators'] == 100."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["xgb_n_estimators"] == _MOEX_EXPECTED_XGB_N_ESTIMATORS

    def test_moex_hparams_xgb_min_child_weight(self) -> None:
        """_MOEX_HPARAMS['xgb_min_child_weight'] == 20."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["xgb_min_child_weight"] == _MOEX_EXPECTED_XGB_MIN_CHILD_WEIGHT

    def test_moex_hparams_lgbm_n_estimators(self) -> None:
        """_MOEX_HPARAMS['lgbm_n_estimators'] == 100."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["lgbm_n_estimators"] == _MOEX_EXPECTED_LGBM_N_ESTIMATORS

    def test_moex_hparams_lgbm_num_leaves(self) -> None:
        """_MOEX_HPARAMS['lgbm_num_leaves'] == 15."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["lgbm_num_leaves"] == _MOEX_EXPECTED_LGBM_NUM_LEAVES

    def test_moex_hparams_cat_depth(self) -> None:
        """_MOEX_HPARAMS['cat_depth'] == 3."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["cat_depth"] == _MOEX_EXPECTED_CAT_DEPTH

    def test_moex_hparams_cat_iterations(self) -> None:
        """_MOEX_HPARAMS['cat_iterations'] == 100."""
        from scripts.auto_ml_research import _MOEX_HPARAMS

        assert _MOEX_HPARAMS["cat_iterations"] == _MOEX_EXPECTED_CAT_ITERATIONS

    def test_get_hparams_moex_segment_returns_moex_hparams(self) -> None:
        """_get_hparams('ru_energy') returns MOEX-profile values."""
        from scripts.auto_ml_research import _get_hparams

        hp = _get_hparams("ru_energy")
        assert hp["xgb_max_depth"] == _MOEX_EXPECTED_XGB_MAX_DEPTH
        assert hp["xgb_n_estimators"] == _MOEX_EXPECTED_XGB_N_ESTIMATORS

    def test_get_hparams_us_segment_returns_default_hparams(self) -> None:
        """_get_hparams('us_tech') returns US-profile (default) values."""
        from scripts.auto_ml_research import _get_hparams

        hp = _get_hparams("us_tech")
        assert hp["xgb_max_depth"] == _US_EXPECTED_XGB_MAX_DEPTH
        assert hp["xgb_n_estimators"] == _US_EXPECTED_XGB_N_ESTIMATORS

    def test_default_hparams_unchanged_max_depth(self) -> None:
        """_DEFAULT_HPARAMS still has xgb_max_depth == 5 (US segments unchanged)."""
        from scripts.auto_ml_research import _DEFAULT_HPARAMS

        assert _DEFAULT_HPARAMS["xgb_max_depth"] == _US_EXPECTED_XGB_MAX_DEPTH

    def test_default_hparams_unchanged_n_estimators(self) -> None:
        """_DEFAULT_HPARAMS still has xgb_n_estimators == 200 (US segments unchanged)."""
        from scripts.auto_ml_research import _DEFAULT_HPARAMS

        assert _DEFAULT_HPARAMS["xgb_n_estimators"] == _US_EXPECTED_XGB_N_ESTIMATORS

    def test_get_hparams_returns_copy(self) -> None:
        """_get_hparams returns a copy so mutations do not affect the constant."""
        from scripts.auto_ml_research import _MOEX_HPARAMS, _get_hparams

        hp = _get_hparams("ru_blue_chips")
        hp["xgb_max_depth"] = 999
        assert _MOEX_HPARAMS["xgb_max_depth"] == _MOEX_EXPECTED_XGB_MAX_DEPTH
