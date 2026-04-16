"""Bond auto-discovery service (Layer 2).

Fetches all MOEX bonds from T-Invest API, applies a 6-step filter chain,
classifies into ru_ofz/ru_corporate segments, registers qualifying bonds
in InstrumentRegistry, and emits CouponEvents on ex-coupon dates.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.core.schemas import BondInfo, CouponEvent
from finalayze.markets.instruments import Instrument

if TYPE_CHECKING:
    from finalayze.core.events import EventBus
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
    from finalayze.markets.instruments import InstrumentRegistry

_log = structlog.get_logger()

# Minimum maturity horizon (bonds maturing within this window are excluded)
_MIN_MATURITY_DAYS = 90  # ~3 months

# Maximum risk level (T-Invest risk classification: 1=low, 2=moderate, 3=high)
_MAX_RISK_LEVEL = 2

# OFZ class codes on MOEX
_OFZ_CLASS_CODES = frozenset({"TQOB", "TQOD"})


@dataclass
class DiscoveryResult:
    """Result of bond discovery with segment classification."""

    ofz: list[BondInfo] = field(default_factory=list)
    corporate: list[BondInfo] = field(default_factory=list)
    total_count: int = 0
    filtered_count: int = 0


class BondDiscoveryService:
    """Discovers and filters MOEX bonds from T-Invest API.

    Filter chain (order matters -- free metadata filters first):
    1. maturity_date > today + 3 months
    2. risk_level <= 2
    3. currency == "rub"
    4. api_trade_available_flag == True
    5. perpetual_flag == False
    6. liquidity_flag == True (T-Invest liquidity classification)
    """

    def __init__(
        self,
        fetcher: TinkoffFetcher,
        registry: InstrumentRegistry,
        event_bus: EventBus | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._registry = registry
        self._event_bus = event_bus

    async def discover(self, *, today: date | None = None) -> DiscoveryResult:
        """Fetch all MOEX bonds, apply filters, classify into segments.

        Args:
            today: Override for current date (useful for testing).

        Returns:
            DiscoveryResult with ofz/corporate lists and counts.
        """
        if today is None:
            today = datetime.now(tz=UTC).date()

        all_bonds = self._fetcher.fetch_all_bonds()
        total = len(all_bonds)

        min_maturity = today + timedelta(days=_MIN_MATURITY_DAYS)
        passed: list[dict[str, Any]] = []

        for bond_dict in all_bonds:
            # Filter 1: maturity date
            mat = bond_dict.get("maturity_date")
            if mat is None or mat <= min_maturity:
                continue

            # Filter 2: risk level
            risk = bond_dict.get("risk_level", 0)
            if risk > _MAX_RISK_LEVEL:
                continue

            # Filter 3: currency
            curr = str(bond_dict.get("currency", "")).lower()
            if curr != "rub":
                continue

            # Filter 4: API trade availability
            if not bond_dict.get("api_trade_available_flag", False):
                continue

            # Filter 5: not perpetual
            if bond_dict.get("perpetual_flag", False):
                continue

            # Filter 6: liquidity flag
            if not bond_dict.get("liquidity_flag", False):
                continue

            passed.append(bond_dict)

        # Classify into segments and convert to BondInfo
        ofz_list: list[BondInfo] = []
        corporate_list: list[BondInfo] = []

        for bond_dict in passed:
            bond_info = _bond_proto_to_info(bond_dict)

            # Fetch amortization schedule for amortizing bonds
            if bond_dict.get("amortization_flag", False):
                self._fetcher.fetch_amortization_schedule(bond_dict["figi"])

            if _is_ofz(bond_dict):
                ofz_list.append(bond_info)
            else:
                corporate_list.append(bond_info)

        _log.info(
            "bond_discovery_complete",
            total=total,
            filtered=len(passed),
            ofz=len(ofz_list),
            corporate=len(corporate_list),
        )

        return DiscoveryResult(
            ofz=ofz_list,
            corporate=corporate_list,
            total_count=total,
            filtered_count=len(passed),
        )

    async def populate_candle_cache(
        self,
        bonds: list[BondInfo],
        db_session_factory: Any,
        lookback_days: int = 365,
    ) -> int:
        """Fetch and cache daily bond candles for discovered bonds.

        Per CONTEXT.md: cache bond candles in TimescaleDB with daily append.
        Bond candle prices are in percentage of face value (MOEX convention).

        Args:
            bonds: List of discovered BondInfo objects.
            db_session_factory: Async session factory for DB writes.
            lookback_days: Default lookback for initial fetch.

        Returns:
            Total count of new candles cached.
        """
        from finalayze.core.models import BondCandleModel  # noqa: PLC0415

        total_cached = 0

        for bond_info in bonds:
            try:
                # Check latest cached date
                async with db_session_factory() as session:
                    from sqlalchemy import func, select  # noqa: PLC0415

                    stmt = select(func.max(BondCandleModel.date)).where(
                        BondCandleModel.bond_figi == bond_info.figi
                    )
                    result = await session.execute(stmt)
                    latest_date = result.scalar()

                # Determine fetch range
                today = datetime.now(tz=UTC).date()
                if latest_date is not None:
                    if hasattr(latest_date, "date"):
                        from_date = latest_date.date() + timedelta(days=1)
                    else:
                        from_date = latest_date + timedelta(days=1)
                else:
                    from_date = today - timedelta(days=lookback_days)

                candle_dicts = self._fetcher.fetch_bond_candles(bond_info.figi, from_date, today)

                if not candle_dicts:
                    continue

                # Convert to ORM models
                models = []
                for cd in candle_dicts:
                    candle_dt = cd["date"]
                    if isinstance(candle_dt, date) and not isinstance(candle_dt, datetime):
                        candle_dt = datetime.combine(candle_dt, datetime.min.time(), tzinfo=UTC)
                    models.append(
                        BondCandleModel(
                            bond_figi=bond_info.figi,
                            date=candle_dt,
                            open=cd["open"],
                            high=cd["high"],
                            low=cd["low"],
                            close=cd["close"],
                            volume=cd["volume"],
                        )
                    )

                async with db_session_factory() as session:
                    session.add_all(models)
                    await session.commit()

                total_cached += len(models)
                _log.info(
                    "bond_candles_cached",
                    figi=bond_info.figi,
                    count=len(models),
                )
            except Exception:
                _log.warning(
                    "bond_candle_cache_failed",
                    figi=bond_info.figi,
                    exc_info=True,
                )
                continue

        return total_cached

    async def check_and_emit_coupon_events(
        self,
        discovered_bonds: list[BondInfo],
        coupon_schedules: dict[str, list[dict[str, Any]]],
        today: date | None = None,
    ) -> int:
        """Emit CouponEvent for bonds with record_date == today.

        Args:
            discovered_bonds: List of discovered BondInfo objects.
            coupon_schedules: Dict mapping FIGI to list of coupon schedule dicts.
            today: Override for current date.

        Returns:
            Count of emitted coupon events.
        """
        if self._event_bus is None:
            return 0

        if today is None:
            today = datetime.now(tz=UTC).date()

        emitted = 0
        bond_map = {b.figi: b for b in discovered_bonds}

        for figi, schedules in coupon_schedules.items():
            bond_info = bond_map.get(figi)
            if bond_info is None:
                continue

            for sched in schedules:
                if sched.get("record_date") == today:
                    event = CouponEvent(
                        bond_figi=figi,
                        bond_ticker=bond_info.ticker,
                        coupon_date=sched["coupon_date"],
                        record_date=sched["record_date"],
                        amount_per_bond=sched["amount_per_bond"],
                        coupon_number=sched["coupon_number"],
                        is_floating=sched.get("is_floating", False),
                    )
                    from finalayze.core.events import EventBus  # noqa: PLC0415

                    await self._event_bus.publish(EventBus.STREAM_COUPONS, event)
                    _log.info(
                        "coupon_event_emitted",
                        bond_ticker=bond_info.ticker,
                        coupon_date=str(sched["coupon_date"]),
                    )
                    emitted += 1

        return emitted


def register_discovered_bonds(result: DiscoveryResult, registry: InstrumentRegistry) -> int:
    """Register discovered bonds in InstrumentRegistry.

    Args:
        result: DiscoveryResult from discover().
        registry: InstrumentRegistry to register bonds in.

    Returns:
        Count of registered bonds.
    """
    count = 0

    for bond_info in result.ofz:
        instrument = _bond_info_to_instrument(bond_info, segment_id="ru_ofz")
        registry.register(instrument)
        _log.info("bond_discovered", ticker=bond_info.ticker, segment="ru_ofz")
        count += 1

    for bond_info in result.corporate:
        instrument = _bond_info_to_instrument(bond_info, segment_id="ru_corporate")
        registry.register(instrument)
        _log.info("bond_discovered", ticker=bond_info.ticker, segment="ru_corporate")
        count += 1

    _log.info("bond_filtered_summary", registered=count)
    return count


def _is_ofz(bond_dict: dict[str, Any]) -> bool:
    """Determine if a bond is OFZ (government) based on class code or sector."""
    class_code = str(bond_dict.get("class_code", "")).upper()
    sector = str(bond_dict.get("sector", "")).lower()
    return class_code in _OFZ_CLASS_CODES or "government" in sector


def _bond_proto_to_info(bond_dict: dict[str, Any]) -> BondInfo:
    """Convert a T-Invest bond dict to BondInfo schema."""
    floating = bond_dict.get("floating_coupon_flag", False)
    amortizing = bond_dict.get("amortization_flag", False)

    # Determine bond type
    if floating:
        bond_type = "floating"
    elif amortizing:
        bond_type = "amortizing"
    else:
        bond_type = "fixed"

    # Day count convention from class code
    class_code = str(bond_dict.get("class_code", "TQOB")).upper()
    day_count = "actual/365"

    initial_nom = bond_dict.get("initial_nominal")
    nominal = bond_dict.get("nominal", Decimal(1000))

    return BondInfo(
        figi=bond_dict["figi"],
        ticker=bond_dict["ticker"],
        isin=bond_dict.get("isin", ""),
        name=bond_dict["name"],
        face_value=nominal,
        coupon_rate=Decimal(0),  # populated from coupon data when available
        coupon_frequency=bond_dict.get("coupon_quantity_per_year", 2),
        maturity_date=bond_dict["maturity_date"],
        floating_coupon=floating,
        class_code=class_code,
        currency=str(bond_dict.get("currency", "RUB")).upper(),
        amortization_flag=amortizing,
        inflation_linked=False,
        initial_nominal=initial_nom,
        day_count_convention=day_count,
        bond_type=bond_type,
    )


def _bond_info_to_instrument(bond_info: BondInfo, *, segment_id: str) -> Instrument:
    """Convert BondInfo to Instrument for registry registration."""
    return Instrument(
        symbol=bond_info.ticker,
        market_id="moex",
        name=bond_info.name,
        instrument_type="bond",
        figi=bond_info.figi,
        lot_size=1,
        currency=bond_info.currency,
        is_active=True,
        segment_id=segment_id,
        face_value=bond_info.face_value,
        coupon_rate=bond_info.coupon_rate,
        coupon_frequency=bond_info.coupon_frequency,
        maturity_date=bond_info.maturity_date,
        floating_coupon=bond_info.floating_coupon,
    )
