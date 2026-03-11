"""Unit tests for TransactionCosts model (6B.4)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.backtest.costs import (
    MOEX_BOND_COSTS,
    MOEX_BOND_EVENT_COSTS,
    MOEX_COSTS,
    OFF_THE_RUN_SPREAD_UPLIFT_BPS,
    US_COSTS,
    TransactionCosts,
    bond_total_cost,
)

# Constants
US_PRICE = Decimal(150)
US_QTY = Decimal(100)
MOEX_PRICE = Decimal(100)
MOEX_QTY = Decimal(100)
TINY_PRICE = Decimal(1)
TINY_QTY = Decimal(1)

# Bond constants
OFZ_FACE_VALUE = Decimal(1000)
OFZ_CLEAN_PRICE_PCT = Decimal("85.50")  # 85.50% of face value
OFZ_QTY = Decimal(100)
_BPS_DIVISOR = Decimal(10000)


class TestTransactionCosts:
    def test_us_costs_per_share_unchanged(self) -> None:
        """US_COSTS uses per-share commission model."""
        cost = US_COSTS.total_cost(US_PRICE, US_QTY)
        # commission = max(1.00, 0.005 * 100) = 1.00 (min applies)
        # spread = 150 * 5/10000 = 0.075
        # slippage = 150 * 3/10000 = 0.045
        # total = 1.00 + (0.075 + 0.045) * 100 = 1.00 + 12.00 = 13.00
        expected_commission = max(Decimal("1.00"), Decimal("0.005") * US_QTY)
        spread = US_PRICE * Decimal(5) / Decimal(10000)
        slippage = US_PRICE * Decimal(3) / Decimal(10000)
        expected = expected_commission + (spread + slippage) * US_QTY
        assert cost == expected

    def test_moex_costs_rate_based(self) -> None:
        """MOEX_COSTS uses commission_rate (percentage of trade value)."""
        cost = MOEX_COSTS.total_cost(MOEX_PRICE, MOEX_QTY)
        # commission = max(0.10, 100 * 100 * 0.0003) = max(0.10, 3.0) = 3.0
        expected_commission = max(Decimal("0.10"), MOEX_PRICE * MOEX_QTY * Decimal("0.0003"))
        spread = MOEX_PRICE * Decimal(10) / Decimal(10000)
        slippage = MOEX_PRICE * Decimal(7) / Decimal(10000)
        expected = expected_commission + (spread + slippage) * MOEX_QTY
        assert cost == expected
        # Verify commission is 3.0 (not the old 0.30)
        assert expected_commission == Decimal("3.0")

    def test_commission_rate_respects_min_commission(self) -> None:
        """For tiny trades, min_commission kicks in."""
        # commission = max(0.10, 1 * 1 * 0.0003) = max(0.10, 0.0003) = 0.10
        costs = TransactionCosts(
            commission_per_share=Decimal(0),
            commission_rate=Decimal("0.0003"),
            min_commission=Decimal("0.10"),
            spread_bps=Decimal(0),
            slippage_bps=Decimal(0),
        )
        cost = costs.total_cost(TINY_PRICE, TINY_QTY)
        assert cost == Decimal("0.10")


class TestMoexBondCosts:
    """Tests for MOEX OFZ bond cost presets and bond_total_cost helper."""

    def test_moex_bond_costs_typical_ofz_trade(self) -> None:
        """MOEX_BOND_COSTS computes correctly for a typical OFZ trade.

        100 bonds at 85.50% of 1000 RUB face value.
        price_rub = 85.50 / 100 * 1000 = 855.00 RUB per bond
        trade_value = 855.00 * 100 = 85500.00 RUB
        commission = max(0.01, 85500 * 0.0005) = max(0.01, 42.75) = 42.75
        spread = 855.00 * 5 / 10000 = 0.4275 per bond
        slippage = 855.00 * 3 / 10000 = 0.2565 per bond
        market_impact = (0.4275 + 0.2565) * 100 = 68.40
        total = 42.75 + 68.40 = 111.15
        """
        price_rub = OFZ_CLEAN_PRICE_PCT / Decimal(100) * OFZ_FACE_VALUE
        cost = MOEX_BOND_COSTS.total_cost(price_rub, OFZ_QTY)

        expected_commission = max(
            Decimal("0.01"), price_rub * OFZ_QTY * Decimal("0.0005")
        )
        spread = price_rub * Decimal(5) / _BPS_DIVISOR
        slippage = price_rub * Decimal(3) / _BPS_DIVISOR
        expected = expected_commission + (spread + slippage) * OFZ_QTY

        assert cost == expected
        # Verify commission is rate-based, not min
        assert expected_commission == Decimal("42.750")

    def test_moex_bond_event_costs_wider_than_normal(self) -> None:
        """MOEX_BOND_EVENT_COSTS has wider spread and slippage than MOEX_BOND_COSTS."""
        assert MOEX_BOND_EVENT_COSTS.spread_bps > MOEX_BOND_COSTS.spread_bps
        assert MOEX_BOND_EVENT_COSTS.slippage_bps > MOEX_BOND_COSTS.slippage_bps
        # Same commission rate
        assert MOEX_BOND_EVENT_COSTS.commission_rate == MOEX_BOND_COSTS.commission_rate

        # Event costs should be strictly higher for same trade
        price_rub = OFZ_CLEAN_PRICE_PCT / Decimal(100) * OFZ_FACE_VALUE
        normal_cost = MOEX_BOND_COSTS.total_cost(price_rub, OFZ_QTY)
        event_cost = MOEX_BOND_EVENT_COSTS.total_cost(price_rub, OFZ_QTY)
        assert event_cost > normal_cost

    def test_bond_total_cost_without_uplift(self) -> None:
        """bond_total_cost matches manual calculation for on-the-run bond."""
        cost = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
        )
        # Should equal MOEX_BOND_COSTS.total_cost(855.00, 100) with no uplift
        price_rub = OFZ_CLEAN_PRICE_PCT / Decimal(100) * OFZ_FACE_VALUE
        expected = MOEX_BOND_COSTS.total_cost(price_rub, OFZ_QTY)
        assert cost == expected

    def test_bond_total_cost_with_off_the_run_uplift(self) -> None:
        """bond_total_cost adds spread uplift for off-the-run tickers."""
        off_the_run_ticker = "SU26238RMFS4"
        cost_with_uplift = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
            ticker=off_the_run_ticker,
        )
        cost_without_uplift = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
        )

        # Uplift = 855.00 * 10 / 10000 * 100 = 85.50 RUB
        price_rub = OFZ_CLEAN_PRICE_PCT / Decimal(100) * OFZ_FACE_VALUE
        uplift_bps = OFF_THE_RUN_SPREAD_UPLIFT_BPS[off_the_run_ticker]
        expected_uplift = price_rub * uplift_bps / _BPS_DIVISOR * OFZ_QTY

        assert cost_with_uplift == cost_without_uplift + expected_uplift
        assert cost_with_uplift > cost_without_uplift

    def test_bond_total_cost_unlisted_ticker_no_uplift(self) -> None:
        """bond_total_cost does not add uplift for tickers not in the dict."""
        cost_with_unknown = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
            ticker="SU26240RMFS0",  # Not in OFF_THE_RUN_SPREAD_UPLIFT_BPS
        )
        cost_without_ticker = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
        )
        assert cost_with_unknown == cost_without_ticker

    def test_bond_total_cost_none_ticker_no_uplift(self) -> None:
        """bond_total_cost with ticker=None produces no uplift."""
        cost = bond_total_cost(
            MOEX_BOND_COSTS,
            clean_price_pct=OFZ_CLEAN_PRICE_PCT,
            face_value=OFZ_FACE_VALUE,
            quantity=OFZ_QTY,
            ticker=None,
        )
        price_rub = OFZ_CLEAN_PRICE_PCT / Decimal(100) * OFZ_FACE_VALUE
        expected = MOEX_BOND_COSTS.total_cost(price_rub, OFZ_QTY)
        assert cost == expected

    def test_us_costs_regression(self) -> None:
        """US_COSTS still produces expected values after bond cost additions."""
        cost = US_COSTS.total_cost(US_PRICE, US_QTY)
        expected_commission = max(Decimal("1.00"), Decimal("0.005") * US_QTY)
        spread = US_PRICE * Decimal(5) / _BPS_DIVISOR
        slippage = US_PRICE * Decimal(3) / _BPS_DIVISOR
        expected = expected_commission + (spread + slippage) * US_QTY
        assert cost == expected

    def test_moex_costs_regression(self) -> None:
        """MOEX_COSTS still produces expected values after bond cost additions."""
        cost = MOEX_COSTS.total_cost(MOEX_PRICE, MOEX_QTY)
        expected_commission = max(
            Decimal("0.10"), MOEX_PRICE * MOEX_QTY * Decimal("0.0003")
        )
        spread = MOEX_PRICE * Decimal(10) / _BPS_DIVISOR
        slippage = MOEX_PRICE * Decimal(7) / _BPS_DIVISOR
        expected = expected_commission + (spread + slippage) * MOEX_QTY
        assert cost == expected
