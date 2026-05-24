"""Cross-market exposure calculation (Layer 4 — risk).

Extracted from signal_executor.process_instrument (Phase 2b). Sums invested
value across all known markets in a base currency and projects what the
exposure ratio would become if a prospective order were submitted.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from config.settings import Settings

    from finalayze.execution.broker_router import BrokerRouter

_ZERO = Decimal(0)
_DEFAULT_MAX_EXPOSURE = Decimal("0.80")
_BASE_CURRENCY = "USD"
_MARKET_CURRENCY: dict[str, str] = {"us": "USD", "moex": "RUB"}


class ExposureCalculator:
    """Compute the prospective cross-market exposure ratio for a new order.

    The ratio is ``(existing_invested + prospective_order) / total_equity``,
    all in the base currency. Callers compare it against the configured
    ``max_cross_market_exposure_pct`` to gate trades.

    Note on market sets: total_equity sums *all known markets* (so the
    denominator stays stable when only a subset of markets carries symbol
    limits), while total_invested only sums markets that contributed to
    ``symbol_limit_markets`` (matching the legacy inline behavior).
    """

    def __init__(
        self,
        *,
        broker_router: BrokerRouter,
        symbol_limit_markets: Iterable[str],
        settings: Settings,
        get_market_equity: Callable[[str], Decimal | None],
        equity_markets: Iterable[str] | None = None,
    ) -> None:
        """Initialise the calculator.

        Args:
            broker_router: Used to fetch per-market portfolio cash.
            symbol_limit_markets: Iterable of market ids with active symbol
                limits — the set of markets whose invested value contributes
                to the total. Caller passes
                ``pre_trade_checker._symbol_limits.keys()``.
            settings: Source of ``max_cross_market_exposure_pct``.
            get_market_equity: Callable resolving a market id to its current
                equity (or ``None`` when the market is unknown / unreachable).
            equity_markets: Iterable of market ids whose equities sum into the
                denominator. Defaults to the known markets in ``_MARKET_CURRENCY``
                — i.e., the same ``["us", "moex"]`` set the legacy inline code
                used. Pass an explicit list to override (e.g. in tests).
        """
        self._broker_router = broker_router
        self._symbol_limit_markets = list(symbol_limit_markets)
        self._settings = settings
        self._get_market_equity = get_market_equity
        self._equity_markets = (
            list(equity_markets) if equity_markets is not None else list(_MARKET_CURRENCY)
        )

    def compute(self, *, market_id: str, order_value: Decimal) -> tuple[Decimal, Decimal]:
        """Return ``(cross_exposure_pct, max_exposure_pct)``.

        Args:
            market_id: Market the prospective order targets — its currency
                determines how ``order_value`` is converted to base.
            order_value: Prospective order notional in the market's currency.
        """
        from finalayze.markets.currency import CurrencyConverter  # noqa: PLC0415

        fx = CurrencyConverter(base_currency=_BASE_CURRENCY)
        total_equity = self._total_equity_base(fx)
        total_invested = self._total_invested_base(fx)

        order_currency = _MARKET_CURRENCY.get(market_id, _BASE_CURRENCY)
        prospective_invested = total_invested + fx.to_base(order_value, order_currency)
        cross = prospective_invested / total_equity if total_equity > _ZERO else _ZERO

        try:
            raw = getattr(self._settings, "max_cross_market_exposure_pct", 0.80)
            max_pct = Decimal(str(float(raw)))
        except (TypeError, ValueError):
            max_pct = _DEFAULT_MAX_EXPOSURE
        return cross, max_pct

    def _total_equity_base(self, fx: object) -> Decimal:
        total = _ZERO
        for m_id in self._equity_markets:
            equity = self._get_market_equity(m_id)
            if equity is None:
                continue
            currency = _MARKET_CURRENCY.get(m_id, _BASE_CURRENCY)
            total += fx.to_base(equity, currency)  # type: ignore[attr-defined]
        return total

    def _total_invested_base(self, fx: object) -> Decimal:
        total = _ZERO
        for m_id in self._symbol_limit_markets:
            m_equity = self._get_market_equity(m_id)
            if m_equity is None:
                continue
            m_broker = self._broker_router.route(m_id)
            m_portfolio = m_broker.get_portfolio()
            m_invested = max(m_equity - m_portfolio.cash, _ZERO)
            currency = _MARKET_CURRENCY.get(m_id, _BASE_CURRENCY)
            total += fx.to_base(m_invested, currency)  # type: ignore[attr-defined]
        return total
