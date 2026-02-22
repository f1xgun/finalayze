"""Market definitions and registry for supported exchanges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from finalayze.core.exceptions import MarketNotFoundError

# ── Weekend day threshold ───────────────────────────────────────────────
_SATURDAY = 5


@dataclass(frozen=True)
class MarketDefinition:
    """Immutable definition of a supported market/exchange."""

    id: str
    name: str
    currency: str
    timezone: str
    open_time: time
    close_time: time


# ── Pre-built market definitions ────────────────────────────────────────

US_MARKET = MarketDefinition(
    id="us",
    name="US Stock Market",
    currency="USD",
    timezone="America/New_York",
    open_time=time(9, 30),
    close_time=time(16, 0),
)

MOEX_MARKET = MarketDefinition(
    id="moex",
    name="Moscow Exchange",
    currency="RUB",
    timezone="Europe/Moscow",
    open_time=time(10, 0),
    close_time=time(18, 40),
)


class MarketRegistry:
    """Registry of available market definitions."""

    def __init__(self, markets: dict[str, MarketDefinition]) -> None:
        self._markets = dict(markets)

    def get_market(self, market_id: str) -> MarketDefinition:
        """Return market definition by id, or raise MarketNotFoundError."""
        try:
            return self._markets[market_id]
        except KeyError:
            raise MarketNotFoundError(f"Market '{market_id}' not found") from None

    def list_markets(self) -> list[MarketDefinition]:
        """Return all registered market definitions."""
        return list(self._markets.values())

    def is_market_open(self, market_id: str, *, at: datetime) -> bool:
        """Check if a market is open at the given UTC datetime.

        Converts the UTC datetime to the market's local timezone, then checks:
        1. Weekends (Saturday/Sunday) are always closed.
        2. The local time must be within [open_time, close_time).
        """
        market = self.get_market(market_id)
        tz = ZoneInfo(market.timezone)
        local_dt = at.astimezone(tz)

        # Weekends are always closed
        if local_dt.weekday() >= _SATURDAY:
            return False

        local_time = local_dt.time()
        return market.open_time <= local_time < market.close_time


def default_registry() -> MarketRegistry:
    """Create a MarketRegistry pre-loaded with US and MOEX markets."""
    return MarketRegistry(
        {
            US_MARKET.id: US_MARKET,
            MOEX_MARKET.id: MOEX_MARKET,
        }
    )
