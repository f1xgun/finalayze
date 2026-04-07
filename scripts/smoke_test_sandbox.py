"""Phase 5 Smoke Test: validate all layers work in Tinkoff Sandbox.

Tests:
  1. Sandbox connectivity + account balance
  2. Bond order: buy OFZ-PD (26244) via sandbox
  3. Bond order: buy OFZ-PK (29009) via sandbox
  4. Portfolio state after bond purchases
  5. SandboxPortfolioTracker coupon processing
  6. Equity order: buy SBER via sandbox
  7. Sell orders: close positions
  8. DrawdownMonitor integration
  9. CBREventStrategy signal check (next meeting: 2026-03-20)
  10. Full portfolio tracker with shadow equity

Usage:
    uv run python scripts/smoke_test_sandbox.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from t_tech.invest import OrderDirection, OrderType
from t_tech.invest.sandbox.async_client import AsyncSandboxClient

from finalayze.markets.instruments import build_default_registry

_TOKEN = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
_SANDBOX_TARGET = "sandbox-invest-public-api.tbank.ru:443"
_LABEL_PASS = "✓ PASS"  # noqa: S105
_LABEL_FAIL = "✗ FAIL"

# OFZ FIGIs
_OFZ_26244_FIGI = "TCS00A1074G2"  # OFZ-PD 26244
_OFZ_29009_FIGI = "BBG007Z5F748"  # OFZ-PK 29009
_SBER_FIGI = "BBG004730N88"  # SBER


class SmokeTestRunner:
    """Runs Phase 5 smoke tests against Tinkoff Sandbox."""

    def __init__(self) -> None:
        self._token = _TOKEN
        self._account_id = ""
        self._results: list[tuple[str, bool, str]] = []
        self._registry = build_default_registry()

    def _record(self, name: str, passed: bool, detail: str = "") -> None:
        status = _LABEL_PASS if passed else _LABEL_FAIL
        self._results.append((name, passed, detail))
        print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

    async def _get_services(self) -> object:
        """Get sandbox services."""
        client = AsyncSandboxClient(self._token, target=_SANDBOX_TARGET)
        return await client.__aenter__()

    async def run_all(self) -> None:
        """Run all smoke tests sequentially."""
        print("=" * 72)
        print("  PHASE 5 SMOKE TEST: Tinkoff Sandbox")
        print("=" * 72)
        print()

        client = AsyncSandboxClient(self._token, target=_SANDBOX_TARGET)
        async with client as services:
            # Get account ID
            accounts = await services.users.get_accounts()
            if not accounts.accounts:
                new_acc = await services.sandbox.open_sandbox_account()
                self._account_id = new_acc.account_id
            else:
                self._account_id = accounts.accounts[0].id

            print(f"  Account: {self._account_id}")
            print()

            # Test 1: Connectivity + balance
            await self._test_connectivity(services)

            # Test 2: Buy OFZ-PD 26244
            await self._test_buy_ofz_pd(services)

            # Test 3: Buy OFZ-PK 29009
            await self._test_buy_ofz_pk(services)

            # Test 4: Portfolio state after bonds
            await self._test_portfolio_after_bonds(services)

            # Test 5: SandboxPortfolioTracker coupon
            await self._test_coupon_tracking(services)

            # Test 6: Buy equity (SBER)
            await self._test_buy_equity(services)

            # Test 7: Portfolio with all positions
            await self._test_full_portfolio(services)

            # Test 8: Sell all positions
            await self._test_sell_all(services)

            # Test 9: DrawdownMonitor integration
            self._test_drawdown_monitor()

            # Test 10: CBR event strategy check
            await self._test_cbr_event_signal(services)

        self._print_summary()

    async def _test_connectivity(self, services: object) -> None:
        """Test 1: Verify sandbox connectivity and balance."""
        try:
            portfolio = await services.operations.get_portfolio(account_id=self._account_id)
            total = portfolio.total_amount_portfolio
            balance = Decimal(total.units) + Decimal(total.nano) / Decimal(1_000_000_000)
            min_balance = 100_000
            passed = balance >= min_balance
            self._record(
                "Sandbox connectivity + balance",
                passed,
                f"{balance:,.0f} RUB",
            )
        except Exception as exc:
            self._record("Sandbox connectivity + balance", False, str(exc))

    async def _test_buy_ofz_pd(self, services: object) -> None:
        """Test 2: Buy OFZ-PD 26244 (1 lot = 1 bond)."""
        try:
            result = await services.orders.post_order(
                figi=_OFZ_26244_FIGI,
                quantity=1,
                direction=OrderDirection.ORDER_DIRECTION_BUY,
                order_type=OrderType.ORDER_TYPE_MARKET,
                account_id=self._account_id,
            )
            price = Decimal(result.executed_order_price.units) + Decimal(
                result.executed_order_price.nano
            ) / Decimal(1_000_000_000)
            passed = result.lots_executed >= 1
            self._record(
                "Buy OFZ-PD 26244 (1 bond)",
                passed,
                f"price={price}, lots_executed={result.lots_executed}",
            )
        except Exception as exc:
            self._record("Buy OFZ-PD 26244 (1 bond)", False, str(exc))

    async def _test_buy_ofz_pk(self, services: object) -> None:
        """Test 3: Buy OFZ-PK 29009 (1 lot = 1 bond)."""
        try:
            result = await services.orders.post_order(
                figi=_OFZ_29009_FIGI,
                quantity=1,
                direction=OrderDirection.ORDER_DIRECTION_BUY,
                order_type=OrderType.ORDER_TYPE_MARKET,
                account_id=self._account_id,
            )
            price = Decimal(result.executed_order_price.units) + Decimal(
                result.executed_order_price.nano
            ) / Decimal(1_000_000_000)
            passed = result.lots_executed >= 1
            self._record(
                "Buy OFZ-PK 29009 (1 bond)",
                passed,
                f"price={price}, lots_executed={result.lots_executed}",
            )
        except Exception as exc:
            self._record("Buy OFZ-PK 29009 (1 bond)", False, str(exc))

    async def _test_portfolio_after_bonds(self, services: object) -> None:
        """Test 4: Verify portfolio has bond positions."""
        try:
            portfolio = await services.operations.get_portfolio(account_id=self._account_id)
            bond_positions = [
                p for p in portfolio.positions if getattr(p, "instrument_type", "") == "bond"
            ]
            passed = len(bond_positions) >= 2  # noqa: PLR2004
            figis = [p.figi for p in bond_positions]
            self._record(
                "Portfolio has bond positions",
                passed,
                f"{len(bond_positions)} bonds: {figis}",
            )
        except Exception as exc:
            self._record("Portfolio has bond positions", False, str(exc))

    async def _test_coupon_tracking(self, services: object) -> None:
        """Test 5: ShadowLedger coupon/dividend processing.

        Tests the shadow accounting layer directly (avoids TinkoffBroker's
        asyncio.run() which cannot nest inside the running event loop).
        """
        try:
            from datetime import UTC, datetime  # noqa: PLC0415

            from finalayze.execution.sandbox_tracker import ShadowLedger  # noqa: PLC0415

            today = datetime.now(tz=UTC).date()
            ledger = ShadowLedger()

            # Get current sandbox positions to check bond holdings
            portfolio = await services.operations.get_portfolio(account_id=self._account_id)
            bond_positions = [
                p for p in portfolio.positions if getattr(p, "instrument_type", "") == "bond"
            ]

            # Simulate coupon payment for OFZ 26244
            bond_qty = Decimal(0)
            for pos in bond_positions:
                if pos.figi == _OFZ_26244_FIGI:
                    bond_qty = Decimal(pos.quantity.units)
                    break

            coupon_per_bond = Decimal("56.19")  # ~11.25% / 2 semiannual
            if bond_qty > 0:
                gross = coupon_per_bond * bond_qty
                adj = ledger.add_coupon("SU26244RMFS2", today, gross)
                assert adj is not None
                assert adj.net_amount == gross - gross * Decimal("0.13")

                # Idempotent: second call returns None
                adj2 = ledger.add_coupon("SU26244RMFS2", today, gross)
                assert adj2 is None

            # Always test ledger math even without bond position
            test_gross = Decimal("100.00")
            adj3 = ledger.add_coupon("TEST_BOND", today, test_gross)
            assert adj3 is not None
            expected_tax = test_gross * Decimal("0.13")
            assert adj3.tax == expected_tax
            assert adj3.net_amount == test_gross - expected_tax
            assert ledger.total_adjustment > 0

            detail = (
                f"bond_qty={bond_qty}, ledger_adj={ledger.total_adjustment}, "
                f"{len(ledger.adjustments)} entries"
            )
            self._record("ShadowLedger coupon processing", True, detail)
        except Exception as exc:
            self._record("ShadowLedger coupon processing", False, str(exc)[:200])

    async def _test_buy_equity(self, services: object) -> None:
        """Test 6: Buy SBER (10 lots = 100 shares)."""
        try:
            result = await services.orders.post_order(
                figi=_SBER_FIGI,
                quantity=10,  # 10 lots x 10 shares/lot = 100 shares
                direction=OrderDirection.ORDER_DIRECTION_BUY,
                order_type=OrderType.ORDER_TYPE_MARKET,
                account_id=self._account_id,
            )
            price = Decimal(result.executed_order_price.units) + Decimal(
                result.executed_order_price.nano
            ) / Decimal(1_000_000_000)
            passed = result.lots_executed >= 1
            self._record(
                "Buy SBER (10 lots)",
                passed,
                f"price={price}, lots={result.lots_executed}",
            )
        except Exception as exc:
            self._record("Buy SBER (10 lots)", False, str(exc))

    async def _test_full_portfolio(self, services: object) -> None:
        """Test 7: Verify portfolio has bonds + equity."""
        try:
            portfolio = await services.operations.get_portfolio(account_id=self._account_id)
            positions = portfolio.positions
            types = {getattr(p, "instrument_type", "unknown") for p in positions}
            total = portfolio.total_amount_portfolio
            equity = Decimal(total.units) + Decimal(total.nano) / Decimal(1_000_000_000)

            has_bonds = "bond" in types
            has_shares = "share" in types
            passed = has_bonds and has_shares
            self._record(
                "Portfolio: bonds + equity",
                passed,
                f"{len(positions)} positions, types={types}, equity={equity:,.0f} RUB",
            )
        except Exception as exc:
            self._record("Portfolio: bonds + equity", False, str(exc))

    async def _test_sell_all(self, services: object) -> None:
        """Test 8: Sell all positions to clean up."""
        try:
            portfolio = await services.operations.get_portfolio(account_id=self._account_id)
            sold_count = 0
            for pos in portfolio.positions:
                qty_units = pos.quantity.units
                if qty_units <= 0:
                    continue
                try:
                    await services.orders.post_order(
                        figi=pos.figi,
                        quantity=qty_units,
                        direction=OrderDirection.ORDER_DIRECTION_SELL,
                        order_type=OrderType.ORDER_TYPE_MARKET,
                        account_id=self._account_id,
                    )
                    sold_count += 1
                except Exception:  # noqa: S110
                    pass

            passed = sold_count > 0
            self._record(
                "Sell all positions",
                passed,
                f"sold {sold_count} positions",
            )
        except Exception as exc:
            self._record("Sell all positions", False, str(exc))

    def _test_drawdown_monitor(self) -> None:
        """Test 9: DrawdownMonitor with synthetic layer equities."""
        try:
            from finalayze.risk.drawdown_monitor import LayeredDrawdownMonitor  # noqa: PLC0415

            monitor = LayeredDrawdownMonitor()

            # Simulate initial equities (1.5M total)
            equities = {
                "core": Decimal(675000),
                "strategic": Decimal(412500),
                "tactical": Decimal(262500),
                "short": Decimal(150000),
            }
            status = monitor.update(equities)
            initial_ok = not status.portfolio_breach

            # Simulate stress: 12% portfolio drop
            stress_equities = {
                "core": Decimal(650000),
                "strategic": Decimal(330000),  # -20%
                "tactical": Decimal(210000),  # -20%
                "short": Decimal(120000),  # -20%
            }
            status2 = monitor.update(stress_equities)
            breach_ok = status2.portfolio_breach
            core_preserved = status2.sizing_multipliers.get("core", Decimal(0)) > 0
            non_core_zero = all(
                status2.sizing_multipliers.get(layer, Decimal(1)) == 0
                for layer in ["strategic", "tactical", "short"]
            )

            passed = initial_ok and breach_ok and core_preserved and non_core_zero
            self._record(
                "DrawdownMonitor cascade",
                passed,
                f"breach={breach_ok}, core_preserved={core_preserved}",
            )
        except Exception as exc:
            self._record("DrawdownMonitor cascade", False, str(exc))

    async def _test_cbr_event_signal(self, services: object) -> None:  # noqa: ARG002
        """Test 10: CBR event strategy signal check."""
        try:
            from datetime import UTC, datetime  # noqa: PLC0415

            from finalayze.data.fetchers.cbr import (  # noqa: PLC0415
                days_to_next_cbr,
                get_next_cbr_meeting,
            )

            today = datetime.now(tz=UTC).date()
            next_meeting = get_next_cbr_meeting(today)
            days_left = days_to_next_cbr(today)

            # Check if CBR event strategy would be active
            signal_window = 3 <= days_left <= 5 if days_left is not None else False  # noqa: PLR2004

            detail = (
                f"next_meeting={next_meeting}, days={days_left}, in_signal_window={signal_window}"
            )
            # This test passes regardless — it's informational
            self._record("CBR event strategy check", True, detail)
        except Exception as exc:
            self._record("CBR event strategy check", False, str(exc))

    def _print_summary(self) -> None:
        """Print test summary."""
        print()
        print("=" * 72)
        total = len(self._results)
        passed = sum(1 for _, p, _ in self._results if p)
        failed = total - passed
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        print("=" * 72)

        if failed > 0:
            print("\n  Failed tests:")
            for name, p, detail in self._results:
                if not p:
                    print(f"    {_LABEL_FAIL}  {name}: {detail}")
        print()


def main() -> None:
    """Run smoke tests."""
    if not _TOKEN:
        print("ERROR: FINALAYZE_TINKOFF_TOKEN not set in .env")
        sys.exit(1)

    runner = SmokeTestRunner()
    asyncio.run(runner.run_all())


if __name__ == "__main__":
    main()
