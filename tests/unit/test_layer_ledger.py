"""Unit tests for PortfolioLayer, LayerConfig, DEFAULT_LAYER_CONFIGS, and LayerLedger."""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.layer_ledger import LayerLedger
from finalayze.core.schemas import (
    DEFAULT_LAYER_CONFIGS,
    LayerConfig,
    PortfolioLayer,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

EXPECTED_LAYER_COUNT = 4
INITIAL_CASH = Decimal(10000)
CORE_CAPITAL_PCT = Decimal("0.45")
STRATEGIC_CAPITAL_PCT = Decimal("0.275")
TACTICAL_CAPITAL_PCT = Decimal("0.175")
SHORT_CAPITAL_PCT = Decimal("0.10")
CAPITAL_SUM = Decimal("1.0")
CORE_MAX_DD = Decimal("0.03")
CORE_MAX_POSITIONS = 4
STRATEGIC_MAX_POSITIONS = 5
TACTICAL_MAX_POSITIONS = 5
SHORT_MAX_POSITIONS = 6
BUY_QTY = Decimal(100)
SELL_QTY = Decimal(40)
REMAINING_QTY = Decimal(60)
DEBIT_AMOUNT = Decimal(3000)
CREDIT_AMOUNT = Decimal(500)
CASH_AFTER_DEBIT = Decimal(7000)
CASH_AFTER_CREDIT = Decimal(7500)
EXCESS_DEBIT = Decimal(20000)
EQUITY_HIGH = Decimal(12000)
EQUITY_MID = Decimal(11000)
EQUITY_NEW_HIGH = Decimal(13000)
STRATEGIC_YIELD_STOP = 50
TACTICAL_YIELD_STOP = 30


# ── PortfolioLayer enum ─────────────────────────────────────────────────


class TestPortfolioLayer:
    def test_values_exist(self) -> None:
        assert PortfolioLayer.CORE.value == "core"
        assert PortfolioLayer.STRATEGIC.value == "strategic"
        assert PortfolioLayer.TACTICAL.value == "tactical"
        assert PortfolioLayer.SHORT.value == "short"

    def test_is_str_enum(self) -> None:
        assert isinstance(PortfolioLayer.CORE, str)

    def test_member_count(self) -> None:
        assert len(PortfolioLayer) == EXPECTED_LAYER_COUNT


# ── LayerConfig ──────────────────────────────────────────────────────────


class TestLayerConfig:
    def test_frozen(self) -> None:
        cfg = LayerConfig(
            layer=PortfolioLayer.CORE,
            capital_pct=CORE_CAPITAL_PCT,
            max_drawdown_pct=CORE_MAX_DD,
            max_positions=CORE_MAX_POSITIONS,
            rebalance_interval="quarterly",
            allowed_instrument_types=("bond",),
        )
        assert cfg.layer == PortfolioLayer.CORE
        assert cfg.capital_pct == CORE_CAPITAL_PCT

    def test_defaults(self) -> None:
        cfg = LayerConfig(
            layer=PortfolioLayer.SHORT,
            capital_pct=SHORT_CAPITAL_PCT,
            max_drawdown_pct=Decimal("0.05"),
            max_positions=SHORT_MAX_POSITIONS,
            rebalance_interval="daily",
        )
        assert cfg.allowed_instrument_types == ("stock",)
        assert cfg.yield_stop_bps == 0


# ── DEFAULT_LAYER_CONFIGS ───────────────────────────────────────────────


class TestDefaultLayerConfigs:
    def test_has_all_four_layers(self) -> None:
        assert len(DEFAULT_LAYER_CONFIGS) == EXPECTED_LAYER_COUNT
        for layer in PortfolioLayer:
            assert layer in DEFAULT_LAYER_CONFIGS

    def test_capital_sums_to_one(self) -> None:
        total = sum(cfg.capital_pct for cfg in DEFAULT_LAYER_CONFIGS.values())
        assert total == CAPITAL_SUM

    def test_core_config(self) -> None:
        cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.CORE]
        assert cfg.capital_pct == CORE_CAPITAL_PCT
        assert cfg.max_positions == CORE_MAX_POSITIONS
        assert cfg.rebalance_interval == "quarterly"
        assert cfg.allowed_instrument_types == ("bond",)
        assert cfg.yield_stop_bps == 0

    def test_strategic_config(self) -> None:
        cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.STRATEGIC]
        assert cfg.capital_pct == STRATEGIC_CAPITAL_PCT
        assert cfg.max_positions == STRATEGIC_MAX_POSITIONS
        assert cfg.rebalance_interval == "monthly"
        assert cfg.yield_stop_bps == STRATEGIC_YIELD_STOP

    def test_tactical_config(self) -> None:
        cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.TACTICAL]
        assert cfg.capital_pct == TACTICAL_CAPITAL_PCT
        assert cfg.max_positions == TACTICAL_MAX_POSITIONS
        assert cfg.rebalance_interval == "weekly"
        assert "bond" in cfg.allowed_instrument_types
        assert "stock" in cfg.allowed_instrument_types
        assert cfg.yield_stop_bps == TACTICAL_YIELD_STOP

    def test_short_config(self) -> None:
        cfg = DEFAULT_LAYER_CONFIGS[PortfolioLayer.SHORT]
        assert cfg.capital_pct == SHORT_CAPITAL_PCT
        assert cfg.max_positions == SHORT_MAX_POSITIONS
        assert cfg.rebalance_interval == "daily"
        assert cfg.allowed_instrument_types == ("stock",)


# ── LayerLedger ──────────────────────────────────────────────────────────


class TestLayerLedgerInit:
    def test_init_with_cash(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        assert ledger.layer_id == "core"
        assert ledger.cash == INITIAL_CASH
        assert ledger.positions == {}
        assert ledger.peak_equity == INITIAL_CASH
        assert ledger.current_equity == INITIAL_CASH

    def test_init_empty_positions(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        assert ledger.is_empty


class TestLayerLedgerDrawdown:
    def test_zero_drawdown_at_start(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        assert ledger.drawdown_pct == Decimal(0)

    def test_drawdown_after_decline(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.update_equity(EQUITY_HIGH)
        ledger.update_equity(EQUITY_MID)
        # DD = (12000 - 11000) / 12000 = 1/12
        expected = (EQUITY_HIGH - EQUITY_MID) / EQUITY_HIGH
        assert ledger.drawdown_pct == expected

    def test_drawdown_zero_peak(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=Decimal(0))
        ledger.peak_equity = Decimal(0)
        assert ledger.drawdown_pct == Decimal(0)


class TestLayerLedgerPeakTracking:
    def test_peak_updates_on_new_high(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.update_equity(EQUITY_HIGH)
        assert ledger.peak_equity == EQUITY_HIGH
        ledger.update_equity(EQUITY_MID)
        assert ledger.peak_equity == EQUITY_HIGH  # peak unchanged
        ledger.update_equity(EQUITY_NEW_HIGH)
        assert ledger.peak_equity == EQUITY_NEW_HIGH


class TestLayerLedgerPositions:
    def test_add_position(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.add_position("SBER", BUY_QTY)
        assert ledger.positions["SBER"] == BUY_QTY
        assert not ledger.is_empty

    def test_add_to_existing_position(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.add_position("SBER", BUY_QTY)
        ledger.add_position("SBER", BUY_QTY)
        expected = BUY_QTY + BUY_QTY
        assert ledger.positions["SBER"] == expected

    def test_remove_partial_position(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.add_position("SBER", BUY_QTY)
        ledger.remove_position("SBER", SELL_QTY)
        assert ledger.positions["SBER"] == REMAINING_QTY

    def test_remove_full_position(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.add_position("SBER", BUY_QTY)
        ledger.remove_position("SBER", BUY_QTY)
        assert "SBER" not in ledger.positions
        assert ledger.is_empty

    def test_remove_over_quantity(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.add_position("SBER", BUY_QTY)
        ledger.remove_position("SBER", EXCESS_DEBIT)
        assert "SBER" not in ledger.positions

    def test_is_empty_with_zero_qty(self) -> None:
        ledger = LayerLedger(layer_id="tactical", cash=INITIAL_CASH)
        ledger.positions["SBER"] = Decimal(0)
        assert ledger.is_empty


class TestLayerLedgerCash:
    def test_debit_cash_success(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        result = ledger.debit_cash(DEBIT_AMOUNT)
        assert result is True
        assert ledger.cash == CASH_AFTER_DEBIT

    def test_debit_cash_insufficient(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        result = ledger.debit_cash(EXCESS_DEBIT)
        assert result is False
        assert ledger.cash == INITIAL_CASH  # unchanged

    def test_credit_cash(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.debit_cash(DEBIT_AMOUNT)
        ledger.credit_cash(CREDIT_AMOUNT)
        assert ledger.cash == CASH_AFTER_CREDIT
