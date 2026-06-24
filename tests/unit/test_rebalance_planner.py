"""Phase 79 P79-01: frozen rebalance-plan dataclasses + deposit-never-an-order invariant.

The plan record is immutable (an audit artifact) and structurally forbids a DEPOSIT auto leg:
the deposit is mark-only with no broker API, so it can only ever surface as a ManualAction
(L-01). Constructing a plan whose auto_legs contains DEPOSIT must raise, not silently allow it.
"""

from __future__ import annotations

import uuid
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import ClassVar
from uuid import uuid4

import pytest

from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import AssetClass, DepositTranche
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.deposit_broker import DepositSimulatedBroker
from finalayze.markets.instruments import Instrument
from finalayze.orchestration.rebalance_planner import (
    FundedEquityCash,
    LegOutcome,
    LegSizing,
    ManualAction,
    PlannedLeg,
    RebalancePlan,
    compute_funded_equity_cash,
    compute_funding_advisory,
    plan_rebalance,
    size_auto_leg,
)

_BUDGET = Decimal(1_000_000)

_CREATED = datetime(2026, 1, 1, tzinfo=UTC)


def _equity_leg() -> PlannedLeg:
    return PlannedLeg(
        asset_class=AssetClass.EQUITY,
        market_id="moex",
        order=OrderRequest(
            symbol="EQMX", side="BUY", quantity=Decimal(10), client_order_id="fnz-eq"
        ),
        side="BUY",
        target_notional=Decimal(1000),
        est_price=Decimal(100),
    )


def _deposit_action() -> ManualAction:
    return ManualAction(
        asset_class=AssetClass.DEPOSIT,
        description="place 300000 RUB on deposit",
        target_notional=Decimal(300_000),
        current_notional=Decimal(0),
        funding_advisory=None,
    )


def _make_plan(
    *,
    auto_legs: tuple[PlannedLeg, ...] = (),
    manual_actions: tuple[ManualAction, ...] = (),
) -> RebalancePlan:
    return RebalancePlan(
        plan_id="p1",
        created_at=_CREATED,
        portfolio_id=uuid4(),
        risk_profile="balanced",
        budget_rub=Decimal(1_000_000),
        mode="DRY_RUN",
        auto_legs=auto_legs,
        manual_actions=manual_actions,
    )


def test_plan_constructs_with_auto_leg_and_deposit_action() -> None:
    """A plan holds an EQUITY auto leg and a DEPOSIT manual action."""
    plan = _make_plan(auto_legs=(_equity_leg(),), manual_actions=(_deposit_action(),))
    assert len(plan.auto_legs) == 1
    assert plan.auto_legs[0].asset_class is AssetClass.EQUITY
    assert plan.manual_actions[0].asset_class is AssetClass.DEPOSIT


def test_plan_is_frozen() -> None:
    """RebalancePlan is an immutable audit record; reassigning a field raises."""
    plan = _make_plan(auto_legs=(_equity_leg(),))
    with pytest.raises(FrozenInstanceError):
        plan.budget_rub = Decimal(0)  # type: ignore[misc]


def test_planned_leg_is_frozen() -> None:
    """A PlannedLeg cannot be mutated after construction."""
    leg = _equity_leg()
    with pytest.raises(FrozenInstanceError):
        leg.side = "SELL"  # type: ignore[misc]


def test_manual_action_is_frozen() -> None:
    """A ManualAction cannot be mutated after construction."""
    action = _deposit_action()
    with pytest.raises(FrozenInstanceError):
        action.target_notional = Decimal(0)  # type: ignore[misc]


def test_deposit_cannot_be_an_auto_leg() -> None:
    """The structural invariant: DEPOSIT never produces an order / auto leg (L-01)."""
    deposit_leg = PlannedLeg(
        asset_class=AssetClass.DEPOSIT,
        market_id="moex",
        order=OrderRequest(
            symbol="DEP", side="BUY", quantity=Decimal(1), client_order_id="fnz-dep"
        ),
        side="BUY",
        target_notional=Decimal(1),
        est_price=None,
    )
    with pytest.raises(ValueError, match="DEPOSIT"):
        _make_plan(auto_legs=(deposit_leg,))


def test_leg_outcome_constructs() -> None:
    """LegOutcome wraps the per-leg submit result with a classification status."""
    outcome = LegOutcome(
        asset_class=AssetClass.EQUITY,
        requested_qty=Decimal(10),
        result=OrderResult(filled=True, quantity=Decimal(10)),
        status="FILLED",
    )
    assert outcome.status == "FILLED"
    assert outcome.result.filled is True


class TestSizeAutoLeg:
    """P79-03/04/05: pure signed delta sizing, no-churn band, lot-size flooring."""

    def test_first_build_is_buy(self) -> None:
        """current=0 -> a full BUY of the lot-floored target qty (P79-R3)."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "BUY"
        assert sizing.delta_qty == Decimal(5500)

    def test_overweight_is_sell_of_signed_delta(self) -> None:
        """Holding more than target -> SELL the signed delta, not the absolute target (P79-R3)."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(6000),  # target qty is 5500 -> sell 500
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "SELL"
        assert sizing.delta_qty == Decimal(500)

    def test_dust_below_band_is_suppressed(self) -> None:
        """|delta_notional| < 2% of budget -> no trade (P79-R4)."""
        sizing = size_auto_leg(
            target_notional=Decimal(510_000),  # vs current 500_000 -> delta 10_000 < 20_000 band
            est_price=Decimal(100),
            current_qty=Decimal(5000),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is None

    def test_above_band_emits(self) -> None:
        """|delta_notional| above the band -> a real leg (P79-R4)."""
        sizing = size_auto_leg(
            target_notional=Decimal(530_000),  # vs current 500_000 -> delta 30_000 > 20_000 band
            est_price=Decimal(100),
            current_qty=Decimal(5000),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert sizing is not None
        assert sizing.side == "BUY"
        assert sizing.delta_qty == Decimal(300)

    def test_lot_floor_rounds_down(self) -> None:
        """A 185-share target at lot_size 10 floors to 180 (matches the broker rule, P79-R5)."""
        sizing = size_auto_leg(
            target_notional=Decimal(18_500),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=10,
            budget_rub=Decimal(100_000),
        )
        assert sizing is not None
        assert sizing.delta_qty == Decimal(180)

    def test_below_one_lot_is_suppressed(self) -> None:
        """A target under one lot (7 shares, lot 10) emits no leg even above the band (P79-R5)."""
        sizing = size_auto_leg(
            target_notional=Decimal(700),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=10,
            budget_rub=Decimal(10_000),  # band 200 < delta 700, so band passes; lot floor kills it
        )
        assert sizing is None

    def test_zero_price_raises(self) -> None:
        """A non-positive price cannot size a leg (avoids div-by-zero on a money path)."""
        with pytest.raises(ValueError, match="price"):
            size_auto_leg(
                target_notional=Decimal(1000),
                est_price=Decimal(0),
                current_qty=Decimal(0),
                lot_size=1,
                budget_rub=_BUDGET,
            )

    def test_returns_legsizing_type(self) -> None:
        """The sizing result is the immutable LegSizing value type."""
        sizing = size_auto_leg(
            target_notional=Decimal(550_000),
            est_price=Decimal(100),
            current_qty=Decimal(0),
            lot_size=1,
            budget_rub=_BUDGET,
        )
        assert isinstance(sizing, LegSizing)


def _equity_instrument() -> Instrument:
    return Instrument(
        symbol="EQMX",
        market_id="moex",
        name="VIM MOEX-Index ETF",
        instrument_type="etf",
        figi="TCS00A101EJ5",
        lot_size=1,
        currency="RUB",
    )


def _ofz_instrument() -> Instrument:
    return Instrument(
        symbol="SU29024RMFS5",
        market_id="moex",
        name="OFZ 29024",
        instrument_type="bond",
        figi="BBG01GJ1FRZ6",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        floating_coupon=True,
    )


class TestPlanRebalance:
    """P79-06/07: compose plan_rebalance -- deterministic ids + deposit manual action/advisory."""

    _PID = uuid4()
    _PLAN_ID = "plan-2026Q2"
    _AS_OF = date(2026, 6, 23)
    _CREATED = datetime(2026, 6, 23, tzinfo=UTC)
    _BUDGET = Decimal(1_000_000)
    _WEIGHTS: ClassVar[dict[AssetClass, Decimal]] = {
        AssetClass.DEPOSIT: Decimal("0.45"),
        AssetClass.OFZ_PK: Decimal("0.25"),
        AssetClass.EQUITY: Decimal("0.30"),
    }
    _PRICES: ClassVar[dict[str, Decimal]] = {"EQMX": Decimal(100), "SU29024RMFS5": Decimal(1000)}

    def _plan(
        self,
        *,
        current_positions: dict[str, Decimal] | None = None,
        deposit_current: Decimal = Decimal(0),
        deposit_broker: DepositSimulatedBroker | None = None,
    ) -> RebalancePlan:
        return plan_rebalance(
            active_portfolio=(self._PID, "balanced", self._BUDGET),
            target_weights=self._WEIGHTS,
            current_positions=current_positions or {},
            last_prices=self._PRICES,
            leg_instruments={
                AssetClass.EQUITY: _equity_instrument(),
                AssetClass.OFZ_PK: _ofz_instrument(),
            },
            deposit_current_notional=deposit_current,
            plan_id=self._PLAN_ID,
            created_at=self._CREATED,
            deposit_broker=deposit_broker,
            as_of=self._AS_OF,
        )

    def test_first_build_two_auto_legs_and_one_deposit_action(self) -> None:
        """A first build BUYs both AUTO legs and emits one DEPOSIT manual action."""
        plan = self._plan()
        assert {leg.asset_class for leg in plan.auto_legs} == {
            AssetClass.EQUITY,
            AssetClass.OFZ_PK,
        }
        assert all(leg.side == "BUY" for leg in plan.auto_legs)
        assert len(plan.manual_actions) == 1
        assert plan.manual_actions[0].asset_class is AssetClass.DEPOSIT

    def test_equity_leg_sizing_and_symbol(self) -> None:
        """0.30 * 1_000_000 / 100 = 3000 units of EQMX, routed to moex."""
        plan = self._plan()
        eq = next(leg for leg in plan.auto_legs if leg.asset_class is AssetClass.EQUITY)
        assert eq.order.symbol == "EQMX"
        assert eq.order.quantity == Decimal(3000)
        assert eq.market_id == "moex"

    def test_deterministic_client_order_ids_byte_stable(self) -> None:
        """Same plan_id + inputs -> identical client_order_ids (replay-safe, P79-R6)."""
        ids_a = {leg.asset_class: leg.order.client_order_id for leg in self._plan().auto_legs}
        ids_b = {leg.asset_class: leg.order.client_order_id for leg in self._plan().auto_legs}
        assert ids_a == ids_b
        # Each id MUST be a valid UUID -- Tinkoff post_order rejects a non-UUID order_id
        # ("order_id should be empty or uuid", found by the sandbox cert).
        assert all(str(uuid.UUID(cid)) == cid for cid in ids_a.values())
        # distinct legs get distinct ids (no accidental collision)
        assert len(set(ids_a.values())) == len(ids_a)

    def test_deposit_never_produces_an_order(self) -> None:
        """No AUTO leg is the deposit; deposit is manual-only (P79-R7)."""
        plan = self._plan()
        assert all(leg.asset_class is not AssetClass.DEPOSIT for leg in plan.auto_legs)

    def test_positive_deposit_delta_has_no_advisory(self) -> None:
        """Deposit underweight (delta > 0) -> a 'place' action, no funding advisory."""
        plan = self._plan(deposit_current=Decimal(0))  # target 450k, current 0 -> +450k
        action = plan.manual_actions[0]
        assert action.funding_advisory is None
        assert "place" in action.description.lower()

    def test_negative_deposit_delta_advisory_no_broker_mutation(self) -> None:
        """Deposit overweight (delta < 0) -> READ-ONLY advisory, broker UNTOUCHED (P79-R8)."""
        tranche = DepositTranche(
            principal=Decimal(1_000_000),
            term_months=12,
            annual_rate=Decimal("0.20"),
            open_date=date(2026, 1, 1),
            maturity_date=date(2027, 1, 1),  # > as_of -> locked, forces a last-resort break
            broken=False,
        )
        broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[tranche])
        before = broker.deposit_value()

        # deposit overweight: current 700k > target 450k -> delta -250k -> withdraw 250k
        plan = self._plan(deposit_current=Decimal(700_000), deposit_broker=broker)
        advisory = plan.manual_actions[0].funding_advisory

        assert advisory is not None
        assert advisory.broke_tranche is True  # the shadow exercised the real break path
        assert advisory.total >= Decimal(250_000)  # the need was sourced
        # the REAL broker is untouched: no tranche broken/removed, mark unchanged.
        assert broker.deposit_value() == before == Decimal(1_000_000)
        assert "withdraw" in plan.manual_actions[0].description.lower()

    def test_compute_funding_advisory_is_non_mutating(self) -> None:
        """compute_funding_advisory runs the strict order on a shadow, never the real broker."""
        tranche = DepositTranche(
            principal=Decimal(1_000_000),
            term_months=12,
            annual_rate=Decimal("0.20"),
            open_date=date(2026, 1, 1),
            maturity_date=date(2027, 1, 1),
            broken=False,
        )
        broker = DepositSimulatedBroker(initial_cash=Decimal(0), tranches=[tranche])
        advisory = compute_funding_advisory(broker, Decimal(300_000), self._AS_OF)
        assert advisory.total >= Decimal(300_000)
        assert broker.deposit_value() == Decimal(1_000_000)  # unchanged


class TestPlanRebalanceGuards:
    """P79-08/09: FIGI fail-loud whole-plan abort + notional sanity guard."""

    _GOOD_WEIGHTS: ClassVar[dict[AssetClass, Decimal]] = {
        AssetClass.DEPOSIT: Decimal("0.45"),
        AssetClass.OFZ_PK: Decimal("0.25"),
        AssetClass.EQUITY: Decimal("0.30"),
    }
    _PRICES: ClassVar[dict[str, Decimal]] = {"EQMX": Decimal(100), "SU29024RMFS5": Decimal(1000)}

    def _call(
        self,
        *,
        instruments: dict[AssetClass, Instrument],
        weights: dict[AssetClass, Decimal],
    ) -> RebalancePlan:
        return plan_rebalance(
            active_portfolio=(uuid4(), "balanced", Decimal(1_000_000)),
            target_weights=weights,
            current_positions={},
            last_prices=self._PRICES,
            leg_instruments=instruments,
            deposit_current_notional=Decimal(0),
            plan_id="p",
            created_at=datetime(2026, 6, 23, tzinfo=UTC),
        )

    def test_missing_figi_aborts_whole_plan(self) -> None:
        """An AUTO leg whose instrument has no FIGI aborts the WHOLE plan (P79-R13)."""
        no_figi = Instrument(
            symbol="EQMX", market_id="moex", name="x", instrument_type="etf", figi=None, lot_size=1
        )
        with pytest.raises(InstrumentNotFoundError):
            self._call(
                instruments={AssetClass.EQUITY: no_figi, AssetClass.OFZ_PK: _ofz_instrument()},
                weights=self._GOOD_WEIGHTS,
            )

    def test_missing_leg_instrument_aborts(self) -> None:
        """A missing AUTO-leg instrument aborts the plan (no half-rebalance)."""
        with pytest.raises(InstrumentNotFoundError):
            self._call(
                instruments={AssetClass.EQUITY: _equity_instrument()},  # OFZ_PK missing
                weights=self._GOOD_WEIGHTS,
            )

    def test_weights_not_summing_to_budget_rejected(self) -> None:
        """A weight vector that does not sum to 1 (targets != budget) is rejected (P79-R12)."""
        over = {
            AssetClass.DEPOSIT: Decimal("0.45"),
            AssetClass.OFZ_PK: Decimal("0.25"),
            AssetClass.EQUITY: Decimal("0.40"),  # sums to 1.10
        }
        with pytest.raises(ValueError, match="sum"):
            self._call(
                instruments={
                    AssetClass.EQUITY: _equity_instrument(),
                    AssetClass.OFZ_PK: _ofz_instrument(),
                },
                weights=over,
            )

    def test_negative_weight_rejected(self) -> None:
        """A negative leg weight (negative target notional) is rejected (P79-R12)."""
        negative = {
            AssetClass.DEPOSIT: Decimal("-0.10"),
            AssetClass.OFZ_PK: Decimal("0.40"),
            AssetClass.EQUITY: Decimal("0.70"),  # sums to 1.00 but deposit is negative
        }
        with pytest.raises(ValueError, match="negative"):
            self._call(
                instruments={
                    AssetClass.EQUITY: _equity_instrument(),
                    AssetClass.OFZ_PK: _ofz_instrument(),
                },
                weights=negative,
            )

    def test_good_weights_and_figis_pass(self) -> None:
        """A well-formed weight vector with resolved FIGIs builds both AUTO legs."""
        plan = self._call(
            instruments={
                AssetClass.EQUITY: _equity_instrument(),
                AssetClass.OFZ_PK: _ofz_instrument(),
            },
            weights=self._GOOD_WEIGHTS,
        )
        assert len(plan.auto_legs) == 2

    def test_missing_weight_leg_raises_clear_value_error(self) -> None:
        """A weight vector missing a leg raises a clear ValueError, not a KeyError (WR-02)."""
        incomplete = {AssetClass.DEPOSIT: Decimal("0.5"), AssetClass.EQUITY: Decimal("0.5")}
        with pytest.raises(ValueError, match="missing legs"):
            self._call(
                instruments={
                    AssetClass.EQUITY: _equity_instrument(),
                    AssetClass.OFZ_PK: _ofz_instrument(),
                },
                weights=incomplete,
            )

    def test_missing_est_price_raises_clear_value_error(self) -> None:
        """A leg symbol absent from last_prices fails loud, not a bare KeyError (INFO-01)."""
        with pytest.raises(ValueError, match="est_price"):
            plan_rebalance(
                active_portfolio=(uuid4(), "balanced", Decimal(1_000_000)),
                target_weights=self._GOOD_WEIGHTS,
                current_positions={},
                last_prices={"EQMX": Decimal(100)},  # SU29024RMFS5 price missing
                leg_instruments={
                    AssetClass.EQUITY: _equity_instrument(),
                    AssetClass.OFZ_PK: _ofz_instrument(),
                },
                deposit_current_notional=Decimal(0),
                plan_id="p",
                created_at=datetime(2026, 6, 23, tzinfo=UTC),
            )


# ── Phase 86: fully-funded synthetic equity (margin + reserve + deposit-as-plug) ──────────────

# IMOEXF future facts: lot 1, point_value 10 -> contract notional = 2275 pts * 10 = 22,750;
# initial margin per contract ~= 2,342 (~10%).
_IMOEXF_CONTRACT_NOTIONAL = Decimal(22_750)
_IMOEXF_MARGIN = Decimal(2_342)


def _future_equity_instrument() -> Instrument:
    return Instrument(
        symbol="IMOEXF",
        market_id="moex",
        name="MOEX Index future",
        instrument_type="future",
        figi="FUTIMOEXF000",
        lot_size=1,
        currency="RUB",
    )


class TestComputeFundedEquityCash:
    """P86: the pure margin + drawdown-reserve sizing for a fully-funded equity FUTURE."""

    def test_exact_split_for_a_15_contract_target(self) -> None:
        """350k target / 22,750 -> 15 contracts; margin/reserve/equity_cash are exact Decimals."""
        funded = compute_funded_equity_cash(
            target_notional=Decimal(350_000),
            contract_notional=_IMOEXF_CONTRACT_NOTIONAL,
            lot_size=1,
            margin_per_contract=_IMOEXF_MARGIN,
            drawdown_survival_pct=Decimal("0.45"),
            im_hike_mult=Decimal("2.5"),
        )
        assert isinstance(funded, FundedEquityCash)
        assert funded.target_contracts == Decimal(15)
        assert funded.exposure == Decimal(341_250)  # 15 * 22,750
        assert funded.margin_cash == Decimal(35_130)  # 15 * 2,342
        # reserve = 341,250 * 0.45 + 35,130 * (2.5 - 1) = 153,562.5 + 52,695 = 206,257.5
        assert funded.reserve_cash == Decimal("206257.5")
        assert funded.equity_cash == Decimal("241387.5")  # 35,130 + 206,257.5

    def test_reserve_survives_target_drawdown_even_after_an_im_hike(self) -> None:
        """Force-liq drawdown == target_dd even when the IM is hiked im_hike_mult x (refuter 1)."""
        dd, hike = Decimal("0.45"), Decimal("2.5")
        funded = compute_funded_equity_cash(
            target_notional=Decimal(350_000),
            contract_notional=_IMOEXF_CONTRACT_NOTIONAL,
            lot_size=1,
            margin_per_contract=_IMOEXF_MARGIN,
            drawdown_survival_pct=dd,
            im_hike_mult=hike,
        )
        # Force-liquidation fires when posted_cash - exposure*P < hiked_IM. The buffer for variation
        # loss after the hike = posted - hike*margin; the survivable drawdown is buffer / exposure.
        hiked_im = hike * funded.margin_cash
        survivable = (funded.equity_cash - hiked_im) / funded.exposure
        assert survivable == dd  # exactly the target, not less

    @pytest.mark.parametrize(
        "bad_margin", [Decimal(0), Decimal(-1), Decimal("inf"), Decimal("nan")]
    )
    def test_fails_closed_on_bad_margin(self, bad_margin: Decimal) -> None:
        """A zero / negative / non-finite margin per contract fails closed (real IM is never 0)."""
        with pytest.raises(ValueError, match="margin_per_contract"):
            compute_funded_equity_cash(
                target_notional=Decimal(350_000),
                contract_notional=_IMOEXF_CONTRACT_NOTIONAL,
                lot_size=1,
                margin_per_contract=bad_margin,
            )

    def test_fails_closed_on_nonpositive_contract_notional(self) -> None:
        """A zero contract notional fails closed (cannot divide to size contracts)."""
        with pytest.raises(ValueError, match="contract_notional"):
            compute_funded_equity_cash(
                target_notional=Decimal(350_000),
                contract_notional=Decimal(0),
                lot_size=1,
                margin_per_contract=_IMOEXF_MARGIN,
            )

    def test_lot_flooring_makes_exposure_below_the_target(self) -> None:
        """15.38 contracts floor to 15 -> exposure (341,250) < target notional (350,000)."""
        funded = compute_funded_equity_cash(
            target_notional=Decimal(350_000),
            contract_notional=_IMOEXF_CONTRACT_NOTIONAL,
            lot_size=1,
            margin_per_contract=_IMOEXF_MARGIN,
        )
        assert funded.target_contracts == Decimal(15)
        assert funded.exposure < Decimal(350_000)


class TestPlanRebalanceFundedEquity:
    """P86: plan_rebalance with a leveraged equity FUTURE -> deposit-as-plug, idle == 0."""

    _PID = uuid4()
    _AS_OF = date(2026, 6, 23)
    _CREATED = datetime(2026, 6, 23, tzinfo=UTC)
    _BUDGET = Decimal(1_000_000)
    _WEIGHTS: ClassVar[dict[AssetClass, Decimal]] = {
        AssetClass.DEPOSIT: Decimal("0.25"),
        AssetClass.OFZ_PK: Decimal("0.40"),
        AssetClass.EQUITY: Decimal("0.35"),
    }
    _PRICES: ClassVar[dict[str, Decimal]] = {
        "IMOEXF": _IMOEXF_CONTRACT_NOTIONAL,
        "SU29024RMFS5": Decimal(1000),
    }
    _MARGINS: ClassVar[dict[str, Decimal]] = {"IMOEXF": _IMOEXF_MARGIN}
    # Expected (see TestComputeFundedEquityCash): equity_cash 241,387.5; ofz 0.40*1M = 400,000.
    _EQUITY_CASH = Decimal("241387.5")
    _OFZ_CASH = Decimal(400_000)
    _DEPOSIT_REALIZED = Decimal("358612.5")  # 1,000,000 - 241,387.5 - 400,000

    def _plan(
        self,
        *,
        current_positions: dict[str, Decimal] | None = None,
        deposit_current: Decimal = Decimal(0),
        margin_by_symbol: dict[str, Decimal] | None = "default",  # type: ignore[assignment]
        drawdown: Decimal = Decimal("0.45"),
        im_hike: Decimal = Decimal("2.5"),
    ) -> RebalancePlan:
        margins = self._MARGINS if margin_by_symbol == "default" else margin_by_symbol
        return plan_rebalance(
            active_portfolio=(self._PID, "balanced", self._BUDGET),
            target_weights=self._WEIGHTS,
            current_positions=current_positions or {},
            last_prices=self._PRICES,
            leg_instruments={
                AssetClass.EQUITY: _future_equity_instrument(),
                AssetClass.OFZ_PK: _ofz_instrument(),
            },
            deposit_current_notional=deposit_current,
            plan_id="p86",
            created_at=self._CREATED,
            as_of=self._AS_OF,
            margin_by_symbol=margins,  # type: ignore[arg-type]
            equity_drawdown_survival_pct=drawdown,
            equity_im_hike_mult=im_hike,
        )

    def test_deposit_is_the_plug_and_idle_is_zero(self) -> None:
        """Greenfield: equity_cash + ofz_cash + deposit_realized == budget EXACTLY (idle 0)."""
        plan = self._plan()
        deposit = plan.manual_actions[0]
        assert deposit.target_notional == self._DEPOSIT_REALIZED
        eq = next(leg for leg in plan.auto_legs if leg.asset_class is AssetClass.EQUITY)
        assert eq.margin_cash == Decimal(35_130)
        assert eq.reserve_cash == Decimal("206257.5")
        equity_cash = (eq.margin_cash or Decimal(0)) + (eq.reserve_cash or Decimal(0))
        assert equity_cash + self._OFZ_CASH + deposit.target_notional == self._BUDGET

    def test_equity_order_qty_is_the_floored_contracts(self) -> None:
        """The ORDER is the lot-floored exposure (15 contracts), unchanged from Phase 85 sizing."""
        plan = self._plan()
        eq = next(leg for leg in plan.auto_legs if leg.asset_class is AssetClass.EQUITY)
        assert eq.order.symbol == "IMOEXF"
        assert eq.order.quantity == Decimal(15)
        # target_notional stays the EXPOSURE allocation for audit honesty (not the cash split).
        assert eq.target_notional == Decimal(350_000)

    def test_top_up_does_not_abort_and_reserves_on_held_target(self) -> None:
        """current 14 -> target 15: trades 1 contract, but reserves on the full 15 (refuter 4)."""
        plan = self._plan(current_positions={"IMOEXF": Decimal(14)})
        eq = next(leg for leg in plan.auto_legs if leg.asset_class is AssetClass.EQUITY)
        assert eq.order.quantity == Decimal(1)  # the traded delta is unchanged
        # the cash split / deposit plug are on the HELD target (15), identical to greenfield
        assert plan.manual_actions[0].target_notional == self._DEPOSIT_REALIZED

    def test_within_band_equity_still_plugs_the_deposit(self) -> None:
        """current 15 == target 15 -> no equity order, but the deposit reflects the held reserve."""
        plan = self._plan(current_positions={"IMOEXF": Decimal(15)})
        assert all(leg.asset_class is not AssetClass.EQUITY for leg in plan.auto_legs)
        assert plan.manual_actions[0].target_notional == self._DEPOSIT_REALIZED

    def test_non_greenfield_deposit_delta_is_the_flow_not_the_stock(self) -> None:
        """deposit_current != 0: the action's delta is realized - current (the cash to move)."""
        plan = self._plan(deposit_current=Decimal(100_000))
        action = plan.manual_actions[0]
        assert action.target_notional == self._DEPOSIT_REALIZED  # stock
        assert action.current_notional == Decimal(100_000)
        # the operator moves the DELTA: 358,612.5 - 100,000 = 258,612.5
        assert "258612.5" in action.description

    def test_future_leg_without_margin_aborts_whole_plan(self) -> None:
        """A FUTURE equity leg with no injected margin aborts the plan (mirrors point_value)."""
        with pytest.raises(ValueError, match="no margin for the future"):
            self._plan(margin_by_symbol={})

    def test_oversized_reserve_overflows_budget_and_hard_stops(self) -> None:
        """A reserve so large that equity+ofz cash exceeds the budget hard-stops (no 1.0x fund)."""
        with pytest.raises(ValueError, match="cannot fund"):
            self._plan(drawdown=Decimal(3))  # absurd survival -> equity_cash > budget

    def test_etf_equity_is_backward_compatible_deposit_equals_weight(self) -> None:
        """A non-future (ETF) equity leg with clean lots -> deposit_realized == budget*weight."""
        plan = plan_rebalance(
            active_portfolio=(self._PID, "balanced", self._BUDGET),
            target_weights=self._WEIGHTS,
            current_positions={},
            last_prices={"EQMX": Decimal(100), "SU29024RMFS5": Decimal(1000)},
            leg_instruments={
                AssetClass.EQUITY: _equity_instrument(),  # ETF @ 100, lot 1 (clean division)
                AssetClass.OFZ_PK: _ofz_instrument(),
            },
            deposit_current_notional=Decimal(0),
            plan_id="p86-etf",
            created_at=self._CREATED,
        )
        # 0.35*1M/100 = 3500 units * 100 = 350,000; ofz 400,000 -> deposit = 250,000 = 0.25*budget
        assert plan.manual_actions[0].target_notional == Decimal(250_000)

    def test_cash_legs_with_non_dividing_prices_plug_residual_into_deposit(self) -> None:
        """Non-clean lot prices leave a residual the deposit plug absorbs (idle 0, WR-02)."""
        # ETF equity @ 137 (lot 1): 350,000/137 -> floor 2554 -> 2554*137 = 349,898 (residual 102).
        # OFZ @ 997 (lot 1): 400,000/997 -> floor 401 -> 401*997 = 399,797 (residual 203).
        plan = plan_rebalance(
            active_portfolio=(self._PID, "balanced", self._BUDGET),
            target_weights=self._WEIGHTS,
            current_positions={},
            last_prices={"EQMX": Decimal(137), "SU29024RMFS5": Decimal(997)},
            leg_instruments={
                AssetClass.EQUITY: _equity_instrument(),  # ETF, fully funded (no margin)
                AssetClass.OFZ_PK: _ofz_instrument(),
            },
            deposit_current_notional=Decimal(0),
            plan_id="p86-residual",
            created_at=self._CREATED,
        )
        equity_cash = Decimal(349_898)
        ofz_cash = Decimal(399_797)
        deposit_realized = plan.manual_actions[0].target_notional
        # the deposit absorbs BOTH lot-flooring residuals (102 + 203 = 305) over the strategic 250k
        assert deposit_realized == Decimal(250_305)
        # the cash identity closes EXACTLY despite the non-clean lots (idle == 0 by construction)
        assert equity_cash + ofz_cash + deposit_realized == self._BUDGET
