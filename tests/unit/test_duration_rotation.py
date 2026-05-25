"""Unit tests for BondDurationRotationStrategy and classify_regime."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.bond_duration_rotation import (
    BondDurationRotationStrategy,
    CBRRegime,
    classify_regime,
)

# ── Named constants (no magic numbers) ───────────────────────────────────────

# Key rates
KEY_RATE_16 = Decimal("16.00")
KEY_RATE_21 = Decimal("21.00")
# Below restrictive threshold (15%) — used to test gap tiebreaker in isolation
KEY_RATE_10 = Decimal("10.00")

# RUONIA gaps relative to key rate (for KEY_RATE_16 / KEY_RATE_21 — cut/hike tests)
RUONIA_DOVISH = Decimal("15.40")  # gap = 15.40 - 16.00 = -0.60 (< -0.50)
RUONIA_HAWKISH = Decimal("21.60")  # gap = 21.60 - 21.00 = +0.60 (> +0.50)
RUONIA_NEUTRAL = Decimal("16.10")  # gap = 16.10 - 16.00 = +0.10 (within +-0.50)
# RUONIA relative to KEY_RATE_10 for hold-neutral tests
RUONIA_NEUTRAL_LOW = Decimal("10.10")  # gap = +0.10 (within +-0.75)

# CPI values
CPI_NORMAL = Decimal("5.0")
CPI_HIGH = Decimal("8.5")  # > 8.0 => forces at least NEUTRAL

# Regime duration boundaries
DOVISH_DURATION_LOW = Decimal("4.0")
DOVISH_DURATION_HIGH = Decimal("5.0")
NEUTRAL_DURATION_LOW = Decimal("2.5")
NEUTRAL_DURATION_HIGH = Decimal("3.5")
HAWKISH_DURATION_LOW = Decimal(0)
HAWKISH_DURATION_HIGH = Decimal("1.5")

# Bond symbols
LONG_BOND = "SU26246RMFS7"  # ~4.5Y duration, dovish-appropriate
MEDIUM_BOND = "SU26241RMFS8"  # ~3.0Y duration, neutral-appropriate
SHORT_BOND = "SU26239RMFS2"  # ~2.0Y duration

# Candle constants
CANDLE_MARKET_ID = "moex"
CANDLE_TIMEFRAME = "1d"
CANDLE_VOLUME = 500_000
CANDLE_PRICE = Decimal("85.50")

# Confidence thresholds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MEETING_BOOST_DAYS = 5  # within 5 days of CBR meeting => higher confidence

# Strategy parameters
DEFAULT_SEGMENT = "ru_ofz_pd"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candles(
    symbol: str,
    count: int = 10,
    price: Decimal = CANDLE_PRICE,
) -> list[Candle]:
    """Create a list of candles for testing."""
    base = datetime(2025, 6, 1, 10, 0, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=CANDLE_MARKET_ID,
            timeframe=CANDLE_TIMEFRAME,
            timestamp=base + timedelta(days=i),
            open=price,
            high=price + Decimal("1.00"),
            low=price - Decimal("1.00"),
            close=price,
            volume=CANDLE_VOLUME,
        )
        for i in range(count)
    ]


def _make_strategy(
    bond_durations: dict[str, Decimal] | None = None,
    bond_maturities: dict[str, date] | None = None,
    coupon_rates: dict[str, Decimal] | None = None,
) -> BondDurationRotationStrategy:
    """Create a BondDurationRotationStrategy with reasonable defaults."""
    if bond_durations is None:
        bond_durations = {
            LONG_BOND: Decimal("4.50"),
            MEDIUM_BOND: Decimal("3.00"),
            SHORT_BOND: Decimal("2.00"),
            "SU26252RMFS5": Decimal("4.20"),
            "SU26244RMFS2": Decimal("4.80"),
            "SU26243RMFS4": Decimal("4.10"),
        }
    if bond_maturities is None:
        bond_maturities = {
            LONG_BOND: date(2029, 6, 15),
            MEDIUM_BOND: date(2027, 6, 15),
            SHORT_BOND: date(2026, 12, 15),
            "SU26252RMFS5": date(2029, 3, 15),
            "SU26244RMFS2": date(2029, 9, 15),
            "SU26243RMFS4": date(2028, 12, 15),
        }
    if coupon_rates is None:
        coupon_rates = {
            LONG_BOND: Decimal("7.65"),
            MEDIUM_BOND: Decimal("6.70"),
            SHORT_BOND: Decimal("7.95"),
            "SU26252RMFS5": Decimal("10.50"),
            "SU26244RMFS2": Decimal("6.10"),
            "SU26243RMFS4": Decimal("5.90"),
        }
    return BondDurationRotationStrategy(
        bond_durations=bond_durations,
        bond_maturities=bond_maturities,
        coupon_rates=coupon_rates,
    )


# ── classify_regime tests ────────────────────────────────────────────────────


class TestClassifyRegime:
    """Tests for the standalone classify_regime function."""

    def test_dovish_when_gap_negative_and_last_cut(self) -> None:
        """RUONIA gap < -0.50 AND last decision was cut => DOVISH."""
        result = classify_regime(
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_hawkish_when_gap_positive_and_last_hike(self) -> None:
        """RUONIA gap > +0.50 AND last decision was hike => HAWKISH."""
        result = classify_regime(
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_neutral_when_gap_within_bounds(self) -> None:
        """RUONIA gap within +-0.75 at non-restrictive rate => NEUTRAL."""
        result = classify_regime(
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=RUONIA_NEUTRAL_LOW,
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_neutral_when_gap_moderate_negative_and_hold(self) -> None:
        """Gap = -0.60 with hold at non-restrictive rate: within +-0.75 => NEUTRAL."""
        result = classify_regime(
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=Decimal("9.40"),  # gap = -0.60
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_neutral_when_gap_moderate_positive_and_hold(self) -> None:
        """Gap = +0.60 with hold at non-restrictive rate: within +-0.75 => NEUTRAL."""
        result = classify_regime(
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=Decimal("10.60"),  # gap = +0.60
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        assert result == CBRRegime.NEUTRAL

    def test_cpi_override_dovish_to_neutral(self) -> None:
        """CPI > 8% forces at least NEUTRAL (overrides DOVISH)."""
        result = classify_regime(
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy_latest_published=CPI_HIGH,
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.NEUTRAL

    def test_cpi_override_does_not_affect_hawkish(self) -> None:
        """CPI > 8% forces at least NEUTRAL, but HAWKISH > NEUTRAL so unchanged."""
        result = classify_regime(
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy_latest_published=CPI_HIGH,
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_cpi_exactly_at_8_does_not_trigger(self) -> None:
        """CPI == 8.0 exactly does not trigger the override (> 8.0 required)."""
        result = classify_regime(
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy_latest_published=Decimal("8.0"),
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_cut_decision_is_dovish_regardless_of_gap(self) -> None:
        """Decision-first: cut => DOVISH regardless of RUONIA gap."""
        result = classify_regime(
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=Decimal("15.50"),  # gap = -0.50
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert result == CBRRegime.DOVISH

    def test_hike_decision_is_hawkish_regardless_of_gap(self) -> None:
        """Decision-first: hike => HAWKISH regardless of RUONIA gap."""
        result = classify_regime(
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=Decimal("21.50"),  # gap = +0.50
            cpi_yoy_latest_published=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        assert result == CBRRegime.HAWKISH

    def test_regime_ordering(self) -> None:
        """CBRRegime IntEnum ordering: DOVISH < NEUTRAL < HAWKISH."""
        assert CBRRegime.DOVISH < CBRRegime.NEUTRAL < CBRRegime.HAWKISH
        assert int(CBRRegime.DOVISH) == 0
        assert int(CBRRegime.NEUTRAL) == 1
        assert int(CBRRegime.HAWKISH) == 2


# ── Strategy signal tests ────────────────────────────────────────────────────


class TestBondDurationRotationStrategy:
    """Tests for BondDurationRotationStrategy.generate_signal."""

    def test_buy_long_duration_in_dovish(self) -> None:
        """In DOVISH regime, BUY long-duration bonds in the regime bond list."""
        strategy = _make_strategy()
        candles = _make_candles(LONG_BOND)

        signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert MIN_CONFIDENCE <= signal.confidence <= MAX_CONFIDENCE
        assert signal.instrument_type == "bond"
        assert signal.strategy_name == "bond_duration_rotation"

    def test_sell_all_in_hawkish(self) -> None:
        """In HAWKISH regime, SELL all OFZ-PD bonds."""
        strategy = _make_strategy()
        candles = _make_candles(LONG_BOND)

        signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={LONG_BOND: Decimal(10)},
            bar_idx=9,
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert MIN_CONFIDENCE <= signal.confidence <= MAX_CONFIDENCE
        assert signal.instrument_type == "bond"

    def test_buy_medium_duration_in_neutral(self) -> None:
        """In NEUTRAL regime, BUY medium-duration bonds in the regime bond list."""
        strategy = _make_strategy()
        candles = _make_candles(MEDIUM_BOND)

        signal = strategy.generate_signal(
            symbol=MEDIUM_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=RUONIA_NEUTRAL_LOW,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.instrument_type == "bond"

    def test_sell_long_bond_in_neutral_when_held(self) -> None:
        """In NEUTRAL, sell a long-duration bond (not in neutral list) when held."""
        strategy = _make_strategy()
        candles = _make_candles(LONG_BOND)

        signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={LONG_BOND: Decimal(5)},
            bar_idx=9,
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=RUONIA_NEUTRAL_LOW,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        # Long bond is NOT in the neutral list => should sell
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_no_signal_when_already_at_target_duration(self) -> None:
        """No signal when bond is already in the regime list and already held."""
        strategy = _make_strategy()
        candles = _make_candles(MEDIUM_BOND)

        signal = strategy.generate_signal(
            symbol=MEDIUM_BOND,
            candles=candles,
            open_positions={MEDIUM_BOND: Decimal(10)},  # already held
            bar_idx=9,
            key_rate=KEY_RATE_10,
            ruonia_7d_avg=RUONIA_NEUTRAL_LOW,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hold",
        )
        # Already holding a bond that's appropriate for the regime => no action
        assert signal is None

    def test_no_regime_data_defaults_to_neutral(self) -> None:
        """When regime data is missing, default to NEUTRAL regime behavior."""
        strategy = _make_strategy()
        candles = _make_candles(MEDIUM_BOND)

        signal = strategy.generate_signal(
            symbol=MEDIUM_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            # No regime kwargs provided => all None => NEUTRAL
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY  # medium bond, neutral-appropriate
        assert signal.instrument_type == "bond"

    def test_hawkish_no_buy_any_bond(self) -> None:
        """In HAWKISH regime, do not BUY any OFZ-PD bond (empty regime list)."""
        strategy = _make_strategy()
        candles = _make_candles(MEDIUM_BOND)

        signal = strategy.generate_signal(
            symbol=MEDIUM_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        # Bond is not in hawkish list (empty) and not held => no signal
        assert signal is None

    def test_hawkish_sell_even_medium_bond(self) -> None:
        """In HAWKISH regime, SELL even medium-duration bonds when held."""
        strategy = _make_strategy()
        candles = _make_candles(MEDIUM_BOND)

        signal = strategy.generate_signal(
            symbol=MEDIUM_BOND,
            candles=candles,
            open_positions={MEDIUM_BOND: Decimal(10)},
            bar_idx=9,
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_signal_has_bond_instrument_type(self) -> None:
        """All signals must have instrument_type='bond'."""
        strategy = _make_strategy()
        candles = _make_candles(LONG_BOND)

        signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert signal is not None
        assert signal.instrument_type == "bond"

    def test_signal_features_contain_regime(self) -> None:
        """Signal features dict should contain regime information."""
        strategy = _make_strategy()
        candles = _make_candles(LONG_BOND)

        signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert signal is not None
        assert "regime" in signal.strategy_payload
        assert signal.strategy_payload["regime"] == float(CBRRegime.DOVISH)

    def test_unknown_bond_no_signal(self) -> None:
        """A bond not in the strategy's bond_durations dict gets no signal."""
        strategy = _make_strategy()
        candles = _make_candles("SU99999RMFS0")

        signal = strategy.generate_signal(
            symbol="SU99999RMFS0",
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert signal is None

    def test_confidence_in_valid_range(self) -> None:
        """Confidence must always be in [0.0, 1.0]."""
        strategy = _make_strategy()

        # Test BUY signal confidence
        candles = _make_candles(LONG_BOND)
        buy_signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        assert buy_signal is not None
        assert MIN_CONFIDENCE <= buy_signal.confidence <= MAX_CONFIDENCE

        # Test SELL signal confidence
        sell_candles = _make_candles(LONG_BOND)
        sell_signal = strategy.generate_signal(
            symbol=LONG_BOND,
            candles=sell_candles,
            open_positions={LONG_BOND: Decimal(5)},
            bar_idx=9,
            key_rate=KEY_RATE_21,
            ruonia_7d_avg=RUONIA_HAWKISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="hike",
        )
        assert sell_signal is not None
        assert MIN_CONFIDENCE <= sell_signal.confidence <= MAX_CONFIDENCE

    def test_dovish_does_not_buy_short_bonds(self) -> None:
        """In DOVISH regime, do not BUY bonds not in the dovish list."""
        strategy = _make_strategy()
        candles = _make_candles(SHORT_BOND)

        signal = strategy.generate_signal(
            symbol=SHORT_BOND,
            candles=candles,
            open_positions={},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        # SHORT_BOND (SU26239RMFS2) is NOT in dovish list, not held => no signal
        assert signal is None

    def test_dovish_sell_short_bond_if_held(self) -> None:
        """In DOVISH regime, sell short-duration bonds if held (rotate to longer)."""
        strategy = _make_strategy()
        candles = _make_candles(SHORT_BOND)

        signal = strategy.generate_signal(
            symbol=SHORT_BOND,
            candles=candles,
            open_positions={SHORT_BOND: Decimal(10)},
            bar_idx=9,
            key_rate=KEY_RATE_16,
            ruonia_7d_avg=RUONIA_DOVISH,
            cpi_yoy=CPI_NORMAL,
            last_cbr_decision="cut",
        )
        # SHORT_BOND not in dovish list, but held => SELL to rotate
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
