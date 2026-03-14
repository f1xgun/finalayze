"""Unit tests for PortfolioLayer, LayerConfig, DEFAULT_LAYER_CONFIGS, and LayerLedger."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core.layer_ledger import LayerLedger, reconcile_with_broker
from finalayze.core.models import LayerLedgerModel
from finalayze.core.schemas import (
    DEFAULT_LAYER_CONFIGS,
    BondPositionRecord,
    LayerConfig,
    PortfolioLayer,
    PortfolioState,
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
        assert cfg.allowed_instrument_types == ("stock", "bond")


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


# ── BondPositionRecord in LayerLedger ─────────────────────────────────────

BOND_ENTRY_YTM = Decimal("12.50")
BOND_ENTRY_PRICE = Decimal("1030.00")
BOND_ENTRY_CLEAN = Decimal("98.50")
BOND_QTY = Decimal(10)
BOND_SELL_QTY = Decimal(4)
BOND_REMAINING = Decimal(6)
BOND_DATE = date(2026, 3, 14)


def _make_bond_record(
    symbol: str = "SU26244RMFS2",
    quantity: Decimal = BOND_QTY,
    layer_id: str = "core",
) -> BondPositionRecord:
    return BondPositionRecord(
        symbol=symbol,
        quantity=quantity,
        entry_ytm_pct=BOND_ENTRY_YTM,
        entry_date=BOND_DATE,
        entry_price=BOND_ENTRY_PRICE,
        entry_clean_pct=BOND_ENTRY_CLEAN,
        layer_id=layer_id,
    )


class TestLayerLedgerBondPositions:
    def test_bond_positions_initially_empty(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        assert ledger.bond_positions == {}

    def test_add_bond_position(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        record = _make_bond_record()
        ledger.add_bond_position(record)
        assert "SU26244RMFS2" in ledger.bond_positions
        assert ledger.bond_positions["SU26244RMFS2"].quantity == BOND_QTY

    def test_remove_bond_position_partial(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.add_bond_position(_make_bond_record())
        ledger.remove_bond_position("SU26244RMFS2", BOND_SELL_QTY)
        assert ledger.bond_positions["SU26244RMFS2"].quantity == BOND_REMAINING

    def test_remove_bond_position_full(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.add_bond_position(_make_bond_record())
        ledger.remove_bond_position("SU26244RMFS2", BOND_QTY)
        assert "SU26244RMFS2" not in ledger.bond_positions


# ── LayerLedgerModel ORM ──────────────────────────────────────────────────


class TestLayerLedgerModel:
    def test_has_correct_columns(self) -> None:
        """LayerLedgerModel has all required columns."""
        model = LayerLedgerModel(
            layer_id="core",
            symbol="SU26244RMFS2",
            quantity=Decimal(10),
            entry_ytm_pct=Decimal("12.50"),
            entry_price=Decimal("1030.00"),
            entry_clean_pct=Decimal("98.50"),
            entry_date=datetime(2026, 3, 14, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        assert model.layer_id == "core"
        assert model.symbol == "SU26244RMFS2"
        assert model.quantity == Decimal(10)
        assert model.entry_ytm_pct == Decimal("12.50")
        assert model.entry_price == Decimal("1030.00")
        assert model.entry_clean_pct == Decimal("98.50")


# ── ORM round-trip ────────────────────────────────────────────────────────


class TestLayerLedgerOrmRoundTrip:
    def test_to_orm_rows(self) -> None:
        ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
        ledger.add_bond_position(_make_bond_record())
        rows = ledger.to_orm_rows()
        assert len(rows) == 1
        assert isinstance(rows[0], LayerLedgerModel)
        assert rows[0].symbol == "SU26244RMFS2"
        assert rows[0].layer_id == "core"

    def test_from_orm_rows(self) -> None:
        row = LayerLedgerModel(
            layer_id="core",
            symbol="SU26244RMFS2",
            quantity=Decimal(10),
            entry_ytm_pct=Decimal("12.50"),
            entry_price=Decimal("1030.00"),
            entry_clean_pct=Decimal("98.50"),
            entry_date=datetime(2026, 3, 14, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledger = LayerLedger.from_orm_rows("core", Decimal(500000), [row])
        assert "SU26244RMFS2" in ledger.bond_positions
        assert ledger.bond_positions["SU26244RMFS2"].quantity == Decimal(10)
        assert ledger.layer_id == "core"
        assert ledger.cash == Decimal(500000)


# ── Reconciliation ────────────────────────────────────────────────────────

RECON_BROKER_QTY = Decimal(15)
RECON_LEDGER_QTY = Decimal(10)


def _make_mock_registry() -> MagicMock:
    """Mock InstrumentRegistry that maps FIGIs to symbols."""
    registry = MagicMock()

    def get_by_figi(figi: str) -> MagicMock:
        mapping = {
            "BBG000FIGI01": MagicMock(
                symbol="SU26244RMFS2",
                instrument_type="bond",
                figi="BBG000FIGI01",
            ),
            "BBG000FIGI02": MagicMock(
                symbol="SU26230RMFS2",
                instrument_type="bond",
                figi="BBG000FIGI02",
            ),
            "BBG000FIGI03": MagicMock(
                symbol="SBER",
                instrument_type="stock",
                figi="BBG000FIGI03",
            ),
        }
        return mapping.get(figi, MagicMock(instrument_type="unknown"))

    registry.get_by_figi = get_by_figi
    return registry


class TestReconcileWithBroker:
    def test_unknown_position_added_to_core(self) -> None:
        """Unknown bond in broker portfolio added to Core layer."""
        portfolio = PortfolioState(
            cash=Decimal(500000),
            positions={"BBG000FIGI01": RECON_BROKER_QTY},
            equity=Decimal(1000000),
            timestamp=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledgers = {
            "core": LayerLedger(layer_id="core", cash=Decimal(500000)),
        }
        registry = _make_mock_registry()
        alerts = reconcile_with_broker(portfolio, ledgers, registry)
        assert len(alerts) >= 1
        assert "SU26244RMFS2" in alerts[0]
        # Position should be added to core layer
        assert "SU26244RMFS2" in ledgers["core"].bond_positions

    def test_quantity_mismatch_trusts_broker(self) -> None:
        """When ledger qty != broker qty, trust broker."""
        portfolio = PortfolioState(
            cash=Decimal(500000),
            positions={"BBG000FIGI01": RECON_BROKER_QTY},
            equity=Decimal(1000000),
            timestamp=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledger = LayerLedger(layer_id="core", cash=Decimal(500000))
        ledger.add_bond_position(_make_bond_record(quantity=RECON_LEDGER_QTY))
        ledgers = {"core": ledger}
        registry = _make_mock_registry()

        alerts = reconcile_with_broker(portfolio, ledgers, registry)
        assert len(alerts) >= 1
        assert "mismatch" in alerts[0].lower() or "15" in alerts[0]
        # Trust broker: quantity should be updated to 15
        assert ledgers["core"].bond_positions["SU26244RMFS2"].quantity == RECON_BROKER_QTY

    def test_alerts_suitable_for_telegram(self) -> None:
        """Alert messages contain symbol and discrepancy detail."""
        portfolio = PortfolioState(
            cash=Decimal(500000),
            positions={"BBG000FIGI02": Decimal(5)},
            equity=Decimal(1000000),
            timestamp=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledgers = {"core": LayerLedger(layer_id="core", cash=Decimal(500000))}
        registry = _make_mock_registry()

        alerts = reconcile_with_broker(portfolio, ledgers, registry)
        assert any("SU26230RMFS2" in a for a in alerts)

    def test_stock_positions_ignored(self) -> None:
        """Stock positions are skipped during bond reconciliation."""
        portfolio = PortfolioState(
            cash=Decimal(500000),
            positions={"BBG000FIGI03": Decimal(100)},
            equity=Decimal(1000000),
            timestamp=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledgers = {"core": LayerLedger(layer_id="core", cash=Decimal(500000))}
        registry = _make_mock_registry()

        alerts = reconcile_with_broker(portfolio, ledgers, registry)
        assert len(alerts) == 0

    def test_telegram_alerter_called(self) -> None:
        """If alerter is provided, on_error is called for each discrepancy."""
        portfolio = PortfolioState(
            cash=Decimal(500000),
            positions={"BBG000FIGI01": Decimal(5)},
            equity=Decimal(1000000),
            timestamp=datetime(2026, 3, 14, tzinfo=timezone.utc),
        )
        ledgers = {"core": LayerLedger(layer_id="core", cash=Decimal(500000))}
        registry = _make_mock_registry()
        alerter = MagicMock()

        reconcile_with_broker(portfolio, ledgers, registry, alerter=alerter)
        alerter.on_error.assert_called()
