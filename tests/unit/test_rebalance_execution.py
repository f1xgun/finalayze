"""Phase 80 P80-01/02/03: pure execution-wiring helpers.

normalize_positions_to_symbols (FIGI/symbol -> symbol), resolve_leg_instruments (config ->
Instrument, fail-loud), to_rub_price (bond %-of-face -> RUB; ETF passthrough).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from finalayze.core.clock import SimulatedClock
from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.modes import ModeManager
from finalayze.core.schemas import AssetClass, DepositTranche
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter
from finalayze.execution.deposit_broker import DepositSimulatedBroker
from finalayze.markets.instruments import Instrument, build_default_registry
from finalayze.orchestration import rebalance_execution
from finalayze.orchestration.rebalance_execution import (
    format_rebalance_plan,
    normalize_positions_to_symbols,
    reconcile_leg_positions,
    resolve_leg_instruments,
    run_rebalance,
    to_rub_price,
)

_EQUITY_SYMBOL = "EQMX"
_OFZ_SYMBOL = "SU29024RMFS5"


def _registry() -> object:
    return build_default_registry()


class TestNormalizePositions:
    def test_figi_key_maps_to_symbol(self) -> None:
        """A FIGI-keyed position (TinkoffBroker) maps to its instrument symbol."""
        registry = _registry()
        figi = registry.get(_EQUITY_SYMBOL, "moex").figi
        assert figi is not None
        out = normalize_positions_to_symbols({figi: Decimal(100)}, registry)
        assert out == {_EQUITY_SYMBOL: Decimal(100)}

    def test_symbol_key_passthrough(self) -> None:
        """A symbol-keyed position (SimulatedBroker) passes through unchanged."""
        registry = _registry()
        out = normalize_positions_to_symbols({_OFZ_SYMBOL: Decimal(50)}, registry)
        assert out == {_OFZ_SYMBOL: Decimal(50)}

    def test_unknown_key_skipped(self) -> None:
        """A key that is neither a known FIGI nor a known MOEX symbol is skipped, not an error."""
        registry = _registry()
        out = normalize_positions_to_symbols({"NOT_A_REAL_KEY_XYZ": Decimal(5)}, registry)
        assert out == {}

    def test_mixed_keys(self) -> None:
        """Mixed FIGI + symbol + junk normalizes the recognized ones only."""
        registry = _registry()
        figi = registry.get(_EQUITY_SYMBOL, "moex").figi
        out = normalize_positions_to_symbols(
            {figi: Decimal(10), _OFZ_SYMBOL: Decimal(20), "JUNK": Decimal(1)}, registry
        )
        assert out == {_EQUITY_SYMBOL: Decimal(10), _OFZ_SYMBOL: Decimal(20)}


class TestResolveLegInstruments:
    def test_resolves_equity_and_ofz(self) -> None:
        """The default config tickers resolve to the EQMX ETF + SU29024 bond instruments."""
        legs = resolve_leg_instruments(_registry())
        assert legs[AssetClass.EQUITY].symbol == _EQUITY_SYMBOL
        assert legs[AssetClass.OFZ_PK].symbol == _OFZ_SYMBOL
        assert legs[AssetClass.EQUITY].figi is not None
        assert legs[AssetClass.OFZ_PK].figi is not None

    def test_unresolvable_symbol_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured equity ticker absent from the registry raises InstrumentNotFoundError."""
        monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "NOPE_NOT_LISTED")
        with pytest.raises(InstrumentNotFoundError):
            resolve_leg_instruments(_registry())


class TestToRubPrice:
    def test_bond_percent_of_face_converts_to_rub(self) -> None:
        """A bond quote (% of face) converts to RUB: 95.5% of 1000 face = 955 RUB."""
        bond = Instrument(
            symbol="SU29024RMFS5",
            market_id="moex",
            name="OFZ 29024",
            instrument_type="bond",
            face_value=Decimal(1000),
            floating_coupon=True,
        )
        assert to_rub_price(bond, Decimal("95.5")) == Decimal("955.0")

    def test_etf_price_passes_through(self) -> None:
        """An ETF/share quote is already RUB-per-unit and passes through unchanged."""
        etf = Instrument(symbol="EQMX", market_id="moex", name="x", instrument_type="etf")
        assert to_rub_price(etf, Decimal("123.45")) == Decimal("123.45")

    def test_bond_without_face_value_fails_loud(self) -> None:
        """A bond lacking face_value cannot be priced and raises ValueError."""
        bond = Instrument(
            symbol="X", market_id="moex", name="x", instrument_type="bond", face_value=None
        )
        with pytest.raises(ValueError, match="face_value"):
            to_rub_price(bond, Decimal("95.5"))


# --- run_rebalance orchestration (P80-04..07) -------------------------------------------------

_AS_OF = date(2026, 6, 23)  # easing regime (after the 2025-06-06 first cut)
_BUDGET = Decimal(1_000_000)
_CLOCK = SimulatedClock(datetime(2026, 6, 23, tzinfo=UTC))
# Tinkoff-style raw quotes: equity in RUB-per-share; bond as % of face (95.5% of 1000 = 955 RUB).
_RAW_PRICES = {_EQUITY_SYMBOL: Decimal(100), _OFZ_SYMBOL: Decimal("95.5")}


class _FakeBroker:
    """A fake MOEX broker: configurable positions + a FILLED submit (records orders)."""

    def __init__(self, positions: dict[str, Decimal] | None = None) -> None:
        self._positions = positions or {}
        self.submitted: list[OrderRequest] = []

    def get_positions(self) -> dict[str, Decimal]:
        return dict(self._positions)

    def submit_order(self, order: OrderRequest, fill_candle: object = None) -> OrderResult:
        self.submitted.append(order)
        return OrderResult(
            filled=True, symbol=order.symbol, side=order.side, quantity=order.quantity
        )


def _fetch_prices(_symbols: list[str]) -> dict[str, Decimal]:
    return dict(_RAW_PRICES)


def _patch_db(
    monkeypatch: pytest.MonkeyPatch,
    *,
    portfolio_id: object,
    active: object,
    deposit: object,
) -> AsyncMock:
    async def _get_active(_sf: object) -> object:
        return active

    async def _load_deposit(_pid: object, _date: object, _sf: object) -> object:
        return deposit

    persist = AsyncMock()  # Phase 82: stub the audit-persist so run_rebalance never hits a real DB.
    monkeypatch.setattr(rebalance_execution, "get_active_portfolio", _get_active)
    monkeypatch.setattr(rebalance_execution, "load_deposit_broker_from_db", _load_deposit)
    monkeypatch.setattr(rebalance_execution, "persist_rebalance_run", persist)
    return persist


async def test_run_rebalance_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end (mock DB): real weights + price conversion + plan + dry-run submit."""
    pid = uuid4()
    _patch_db(monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None)
    broker = _FakeBroker(positions={})  # first build, no holdings
    plan, outcomes = await run_rebalance(
        broker_router=BrokerRouter({"moex": broker}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),  # unused (DB calls patched)
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
    )
    assert plan.budget_rub == _BUDGET
    legs = {leg.asset_class: leg for leg in plan.auto_legs}
    # easing-regime balanced weights at 2026-06-23: deposit 0.25 / ofz 0.40 / equity 0.35.
    # equity: 0.35*1M / 100 = 3500 units.
    assert legs[AssetClass.EQUITY].order.quantity == Decimal(3500)
    # OFZ: 0.40*1M / 955 RUB = 418.8 -> floor 418. (Without the %->RUB conversion it would be
    # 400000/95.5 = 4188 -- so asserting 418 proves to_rub_price was applied. ANTI-HOLLOW.)
    assert legs[AssetClass.OFZ_PK].order.quantity == Decimal(418)
    # deposit: 0.25*1M = 250k, manual action only.
    assert plan.manual_actions[0].asset_class is AssetClass.DEPOSIT
    assert plan.manual_actions[0].target_notional == Decimal(250_000)
    assert len(outcomes) == 2
    assert all(o.status == "FILLED" for o in outcomes)


async def test_run_rebalance_no_active_portfolio_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No active portfolio -> a clear ValueError, never a silent no-op (P80-R5)."""
    _patch_db(monkeypatch, portfolio_id=uuid4(), active=None, deposit=None)
    with pytest.raises(ValueError, match="no active"):
        await run_rebalance(
            broker_router=BrokerRouter({"moex": _FakeBroker()}),
            mode_manager=ModeManager(),
            registry=build_default_registry(),
            session_factory=object(),
            clock=_CLOCK,
            fetch_last_prices=_fetch_prices,
        )


async def test_run_rebalance_plan_id_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same portfolio + as_of -> identical plan_id and identical leg client_order_ids (P80-R6)."""
    pid = uuid4()
    _patch_db(monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None)
    kwargs = {
        "mode_manager": ModeManager(),
        "registry": build_default_registry(),
        "session_factory": object(),
        "clock": _CLOCK,
        "fetch_last_prices": _fetch_prices,
    }
    plan_a, _ = await run_rebalance(broker_router=BrokerRouter({"moex": _FakeBroker()}), **kwargs)
    plan_b, _ = await run_rebalance(broker_router=BrokerRouter({"moex": _FakeBroker()}), **kwargs)
    assert plan_a.plan_id == plan_b.plan_id == f"{pid}:{_AS_OF.isoformat()}"
    ids_a = {leg.asset_class: leg.order.client_order_id for leg in plan_a.auto_legs}
    ids_b = {leg.asset_class: leg.order.client_order_id for leg in plan_b.auto_legs}
    assert ids_a == ids_b


async def test_run_rebalance_wires_deposit_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deposit ManualAction reflects the loaded deposit_value() (P80-R7)."""
    pid = uuid4()
    tranche = DepositTranche(
        principal=Decimal(100_000),
        term_months=12,
        annual_rate=Decimal("0.20"),
        open_date=date(2026, 1, 1),
        maturity_date=date(2027, 1, 1),
        broken=False,
    )
    deposit_broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[tranche])
    _patch_db(
        monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=deposit_broker
    )
    plan, _ = await run_rebalance(
        broker_router=BrokerRouter({"moex": _FakeBroker()}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
    )
    deposit_action = plan.manual_actions[0]
    assert deposit_action.current_notional == Decimal(100_000)  # == deposit_value()


async def test_run_rebalance_preview_places_no_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    """submit=False assembles the real plan but places NO orders (safe preview)."""
    pid = uuid4()
    _patch_db(monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None)
    broker = _FakeBroker(positions={})
    plan, outcomes = await run_rebalance(
        broker_router=BrokerRouter({"moex": broker}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
        submit=False,
    )
    assert outcomes == []  # nothing submitted
    assert broker.submitted == []  # the broker was never asked to place an order
    assert len(plan.auto_legs) == 2  # but the real plan was still assembled


def test_format_rebalance_plan_renders_legs_actions_and_preview() -> None:
    """The formatter shows AUTO legs, the deposit MANUAL action, and a preview marker."""
    from datetime import UTC, datetime
    from decimal import Decimal

    from finalayze.execution.broker_base import OrderRequest
    from finalayze.orchestration.rebalance_planner import (
        ManualAction,
        PlannedLeg,
        RebalancePlan,
    )

    leg = PlannedLeg(
        asset_class=AssetClass.EQUITY,
        market_id="moex",
        order=OrderRequest(
            symbol="EQMX", side="BUY", quantity=Decimal(3500), client_order_id="fnz-x"
        ),
        side="BUY",
        target_notional=Decimal(350_000),
        est_price=Decimal(100),
    )
    deposit = ManualAction(
        asset_class=AssetClass.DEPOSIT,
        description="DEPOSIT: place 250000 RUB on a bank deposit",
        target_notional=Decimal(250_000),
        current_notional=Decimal(0),
    )
    plan = RebalancePlan(
        plan_id="p1",
        created_at=datetime(2026, 6, 23, tzinfo=UTC),
        portfolio_id=uuid4(),
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode="DRY_RUN",
        auto_legs=(leg,),
        manual_actions=(deposit,),
    )
    rendered = format_rebalance_plan(plan, [])
    assert "EQMX" in rendered
    assert "BUY" in rendered
    assert "place 250000 RUB" in rendered
    assert "preview -- no orders submitted" in rendered


class TestReconcileLegPositions:
    """SMP-02 guard: flag a leg showing zero holdings against a non-empty broker book."""

    def test_flags_zero_leg_against_nonempty_book(self) -> None:
        """A leg missing from normalized positions while the book is non-empty is flagged."""
        legs = resolve_leg_instruments(build_default_registry())
        # equity present, OFZ absent (e.g. its live FIGI drifted and was dropped).
        flagged = reconcile_leg_positions(
            legs, {_EQUITY_SYMBOL: Decimal(100)}, raw_book_nonempty=True
        )
        assert flagged == [AssetClass.OFZ_PK]

    def test_empty_book_flags_nothing(self) -> None:
        """An empty broker book (a genuine first build) flags nothing -- no false positive."""
        legs = resolve_leg_instruments(build_default_registry())
        assert reconcile_leg_positions(legs, {}, raw_book_nonempty=False) == []

    def test_all_legs_present_flags_nothing(self) -> None:
        """Both legs present in normalized positions -> no flag."""
        legs = resolve_leg_instruments(build_default_registry())
        positions = {_EQUITY_SYMBOL: Decimal(100), _OFZ_SYMBOL: Decimal(50)}
        assert reconcile_leg_positions(legs, positions, raw_book_nonempty=True) == []


async def test_run_rebalance_nkd_dirty_price_sizes_fewer_bonds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NKD (accrued coupon) raises the OFZ dirty price -> fewer bonds than clean-only (P82-R6)."""
    pid = uuid4()
    _patch_db(monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None)
    broker = _FakeBroker(positions={})
    # OFZ clean = 95.5% of 1000 = 955; dirty = 955 + 50 NKD = 1005. 0.40*1M / 1005 = 398 (vs 418).
    plan, _ = await run_rebalance(
        broker_router=BrokerRouter({"moex": broker}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
        nkd_by_symbol={_OFZ_SYMBOL: Decimal(50)},
    )
    legs = {leg.asset_class: leg for leg in plan.auto_legs}
    assert legs[AssetClass.OFZ_PK].order.quantity == Decimal(398)  # NKD reduced the bond qty
    assert legs[AssetClass.EQUITY].order.quantity == Decimal(3500)  # equity unaffected


async def test_run_rebalance_persists_audit_on_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real submit reconciles + persists the run (P82-R7)."""
    pid = uuid4()
    persist = _patch_db(
        monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None
    )
    plan, outcomes = await run_rebalance(
        broker_router=BrokerRouter({"moex": _FakeBroker()}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
    )
    persist.assert_awaited_once()
    args = persist.await_args.args  # (session_factory, plan, outcomes, reconciliation)
    assert args[1] is plan
    assert args[2] == outcomes
    assert args[3].plan_id == plan.plan_id  # the RebalanceReconciliation


async def test_run_rebalance_preview_does_not_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preview (submit=False) records nothing (P82-R7)."""
    pid = uuid4()
    persist = _patch_db(
        monkeypatch, portfolio_id=pid, active=(pid, "balanced", _BUDGET), deposit=None
    )
    await run_rebalance(
        broker_router=BrokerRouter({"moex": _FakeBroker()}),
        mode_manager=ModeManager(),
        registry=build_default_registry(),
        session_factory=object(),
        clock=_CLOCK,
        fetch_last_prices=_fetch_prices,
        submit=False,
    )
    persist.assert_not_awaited()
