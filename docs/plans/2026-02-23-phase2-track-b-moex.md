# Phase 2 Track B — MOEX / Tinkoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add MOEX market support via t-tech-investments SDK, implement Alpaca and Tinkoff live brokers, and wire a broker router that dispatches orders by market ID.

**Architecture:** `TinkoffFetcher` fetches MOEX candles using `AsyncClient.get_all_candles()` wrapped in `asyncio.run()` (sync interface, consistent with `YFinanceFetcher`). `AlpacaBroker` and `TinkoffBroker` both implement `BrokerBase` with `fill_candle: Candle | None = None` (None for live brokers, required for simulated). `BrokerRouter` maps `market_id` → `BrokerBase` and raises `BrokerError` on unknown markets.

**Tech Stack:** `t-tech-investments` (custom tbank PyPI index), `alpaca-py` (already in pyproject.toml), `asyncio.run()` for wrapping async SDK calls in sync interface.

**Worktree:** `.worktrees/phase2-moex` on branch `feature/phase2-moex`

**Merge order:** Track A (`feature/phase2-intelligence`) merges first. Track B is then rebased on updated main before opening its PR.

---

## Project Conventions (read before writing any code)

- Every file starts with `"""Docstring."""\n\nfrom __future__ import annotations`
- Use `StrEnum` not `str, Enum` (ruff UP042)
- Exception names end in `Error` (ruff N818)
- No magic numbers — define named constants
- `from __future__ import annotations` means type hints are strings — safe to use `X | Y` everywhere
- Run quality checks: `source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header`
- The project uses `uv run` for all Python commands
- Tests live in `tests/unit/` — mirror source structure

---

## Task 1: Add `t-tech-investments` + tbank PyPI index + BrokerBase refactor

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/finalayze/execution/broker_base.py`
- Modify: `src/finalayze/execution/simulated_broker.py`
- Test: `tests/unit/execution/test_simulated_broker.py` (existing — run to confirm nothing broke)

### Step 1: Add custom PyPI index and `t-tech-investments` to `pyproject.toml`

After the existing `[tool.uv]` section (which has `environments`), add a new index block:

```toml
[[tool.uv.index]]
name = "tbank"
url = "https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
explicit = true
```

In the `[project]` `dependencies` list, add after `"alpaca-py>=0.33.0"`:

```toml
    "t-tech-investments>=0.2.0; sys_platform != 'win32'",
```

> The `explicit = true` flag means uv only uses this index for packages that explicitly request it (via `tool.uv.sources`). We need to also add a source entry so uv knows to look at tbank index for `t-tech-investments`.

After the `[[tool.uv.index]]` block, add:

```toml
[tool.uv.sources]
t-tech-investments = { index = "tbank" }
```

In `[[tool.mypy.overrides]]` `module` list, add:

```toml
    "t_tech.*",
    "openai.*",
```

> `openai.*` may already be present if Track A merged first. Only add if not already there.

Remove the comment block (lines ~61-63) about tinkoff-investments being quarantined.

### Step 2: Run `uv sync`

```bash
source ~/.zshrc && uv sync --extra dev
```

Expected: resolves and installs `t-tech-investments` from tbank index. If it fails due to version not found, try `"t-tech-investments"` without version constraint.

### Step 3: Write the test first — refactored SimulatedBroker must raise on None candle

In `tests/unit/execution/test_simulated_broker.py`, add a test (before implementing):

```python
def test_submit_order_raises_if_no_candle(broker: SimulatedBroker) -> None:
    """SimulatedBroker must reject orders when no candle is provided."""
    order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("1"))
    with pytest.raises(ValueError, match="fill_candle"):
        broker.submit_order(order, fill_candle=None)
```

### Step 4: Run the test — expect failure

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_simulated_broker.py::test_submit_order_raises_if_no_candle -v
```

Expected: `FAILED` — `submit_order` currently requires `fill_candle: Candle` (not optional).

### Step 5: Refactor `BrokerBase.submit_order` to make `fill_candle` optional

In `src/finalayze/execution/broker_base.py`, change the abstract method signature:

```python
@abstractmethod
def submit_order(self, order: OrderRequest, fill_candle: Candle | None = None) -> OrderResult:
    """Submit an order for execution.

    Args:
        order: The order to execute.
        fill_candle: For simulated brokers — fill price is taken from candle open.
                     Live brokers ignore this parameter (pass None).
    """
    ...
```

### Step 6: Update `SimulatedBroker.submit_order` to validate `fill_candle`

In `src/finalayze/execution/simulated_broker.py`, change the method signature and add a guard:

```python
def submit_order(self, order: OrderRequest, fill_candle: Candle | None = None) -> OrderResult:
    """Fill an order at the candle's open price.

    BUY: deducts cash, adds to position.
    SELL: adds proceeds to cash, reduces/removes position.
    """
    if fill_candle is None:
        msg = "SimulatedBroker requires fill_candle to determine the fill price"
        raise ValueError(msg)
    fill_price = fill_candle.open
    # ... rest of method unchanged ...
```

### Step 7: Run the new test — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_simulated_broker.py -v
```

Expected: ALL tests pass, including the new `test_submit_order_raises_if_no_candle`.

### Step 8: Run full quality suite

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero errors, all tests green.

### Step 9: Commit

```bash
git add pyproject.toml uv.lock src/finalayze/execution/broker_base.py src/finalayze/execution/simulated_broker.py tests/unit/execution/test_simulated_broker.py
git commit -m "feat(broker): add t-tech-investments dep + make fill_candle optional in BrokerBase"
```

---

## Task 2: MOEX instruments — add 8 Russian stocks to InstrumentRegistry

**Files:**
- Modify: `src/finalayze/markets/instruments.py`
- Test: `tests/unit/markets/test_instruments.py` (existing)

### Step 1: Write the failing test

In `tests/unit/markets/test_instruments.py`, add:

```python
def test_default_registry_includes_moex_instruments() -> None:
    """Default registry must include all 8 MOEX instruments."""
    registry = build_default_registry()
    moex_instruments = registry.list_by_market("moex")
    assert len(moex_instruments) == 8  # noqa: PLR2004


EXPECTED_MOEX_SYMBOLS = {"SBER", "GAZP", "LKOH", "GMKN", "YNDX", "NVTK", "ROSN", "VTBR"}


def test_moex_instruments_have_figi() -> None:
    """All MOEX instruments must have a non-empty FIGI identifier."""
    registry = build_default_registry()
    for inst in registry.list_by_market("moex"):
        assert inst.figi is not None and inst.figi != "", (
            f"{inst.symbol} missing FIGI"
        )


def test_moex_instruments_symbols() -> None:
    """Default registry must contain exactly the expected MOEX symbols."""
    registry = build_default_registry()
    symbols = {i.symbol for i in registry.list_by_market("moex")}
    assert symbols == EXPECTED_MOEX_SYMBOLS


def test_moex_instruments_currency_is_rub() -> None:
    """All MOEX instruments must be denominated in RUB."""
    registry = build_default_registry()
    for inst in registry.list_by_market("moex"):
        assert inst.currency == "RUB", f"{inst.symbol} currency is {inst.currency!r}, expected 'RUB'"
```

### Step 2: Run to confirm failure

```bash
source ~/.zshrc && uv run pytest tests/unit/markets/test_instruments.py -v -k "moex"
```

Expected: `FAILED` — `list_by_market("moex")` returns empty list.

### Step 3: Add `DEFAULT_MOEX_INSTRUMENTS` and update `build_default_registry`

In `src/finalayze/markets/instruments.py`, after `DEFAULT_US_INSTRUMENTS`, add:

```python
# Default MOEX instruments for Phase 2
# FIGI identifiers from Tinkoff Invest API instrument catalogue.
DEFAULT_MOEX_INSTRUMENTS: list[Instrument] = [
    Instrument(
        symbol="SBER",
        market_id="moex",
        name="Sberbank",
        instrument_type="stock",
        figi="BBG004730N88",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="GAZP",
        market_id="moex",
        name="Gazprom",
        instrument_type="stock",
        figi="BBG004730RP0",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="LKOH",
        market_id="moex",
        name="Lukoil",
        instrument_type="stock",
        figi="BBG004731032",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="GMKN",
        market_id="moex",
        name="Norilsk Nickel",
        instrument_type="stock",
        figi="BBG004731489",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="YNDX",
        market_id="moex",
        name="Yandex",
        instrument_type="stock",
        figi="BBG006L8G4H1",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="NVTK",
        market_id="moex",
        name="Novatek",
        instrument_type="stock",
        figi="BBG00475KKY8",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="ROSN",
        market_id="moex",
        name="Rosneft",
        instrument_type="stock",
        figi="BBG004731354",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="VTBR",
        market_id="moex",
        name="VTB Bank",
        instrument_type="stock",
        figi="BBG004730ZJ9",
        lot_size=10,
        currency="RUB",
    ),
]
```

Update `build_default_registry()` to include MOEX instruments:

```python
def build_default_registry() -> InstrumentRegistry:
    """Build and return a registry pre-populated with default instruments."""
    registry = InstrumentRegistry()
    for instrument in DEFAULT_US_INSTRUMENTS:
        registry.register(instrument)
    for instrument in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(instrument)
    return registry
```

### Step 4: Run the new tests — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/markets/test_instruments.py -v
```

Expected: ALL tests pass.

### Step 5: Run full quality suite

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero errors.

### Step 6: Commit

```bash
git add src/finalayze/markets/instruments.py tests/unit/markets/test_instruments.py
git commit -m "feat(markets): add 8 MOEX instruments with FIGI identifiers"
```

---

## Task 3: TinkoffFetcher — MOEX candle fetcher using t-tech-investments

**Files:**
- Create: `src/finalayze/data/fetchers/tinkoff_data.py`
- Create: `tests/unit/data/fetchers/test_tinkoff_data.py`

### Step 1: Write the failing tests

Create `tests/unit/data/fetchers/test_tinkoff_data.py`:

```python
"""Unit tests for TinkoffFetcher."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

# ---------- helpers ----------


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_fake_candle(
    open_u: int, open_n: int,
    close_u: int, close_n: int,
    high_u: int, high_n: int,
    low_u: int, low_n: int,
    volume: int,
    time_seconds: int,
) -> MagicMock:
    """Build a fake Tinkoff HistoricCandle object."""
    candle = MagicMock()
    candle.open.units = open_u
    candle.open.nano = open_n
    candle.close.units = close_u
    candle.close.nano = close_n
    candle.high.units = high_u
    candle.high.nano = high_n
    candle.low.units = low_u
    candle.low.nano = low_n
    candle.volume = volume
    candle.time.seconds = time_seconds
    candle.time.nanos = 0
    return candle


# ---------- unit tests ----------


class TestTinkoffFetcherQuotationToDecimal:
    def test_whole_number(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        q = MagicMock()
        q.units = 270
        q.nano = 0
        assert fetcher._quotation_to_decimal(q) == Decimal("270")

    def test_fractional(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        q = MagicMock()
        q.units = 270
        q.nano = 500_000_000  # 0.5
        assert fetcher._quotation_to_decimal(q) == Decimal("270.5")

    def test_sub_nano(self) -> None:
        """nano=1 → 0.000000001, should round correctly."""
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        q = MagicMock()
        q.units = 1
        q.nano = 1
        result = fetcher._quotation_to_decimal(q)
        assert result > Decimal("1")


class TestTinkoffFetcherSymbolToFigi:
    def test_known_symbol(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        assert fetcher._symbol_to_figi("SBER") == "BBG004730N88"

    def test_unknown_symbol_raises(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        with pytest.raises(InstrumentNotFoundError):
            fetcher._symbol_to_figi("UNKNOWN")


class TestTinkoffFetchCandles:
    def test_fetch_returns_candles(self) -> None:
        fake_candle = _make_fake_candle(
            open_u=270, open_n=0,
            close_u=275, close_n=0,
            high_u=280, high_n=0,
            low_u=265, low_n=0,
            volume=1_000_000,
            time_seconds=1_700_000_000,
        )

        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            return_value=[fake_candle],
        ):
            fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            candles = fetcher.fetch_candles("SBER", start, end, timeframe="1d")

        assert len(candles) == 1
        c = candles[0]
        assert c.symbol == "SBER"
        assert c.market_id == "moex"
        assert c.source == "tinkoff"
        assert c.open == Decimal("270")
        assert c.close == Decimal("275")
        assert c.volume == 1_000_000

    def test_fetch_unknown_symbol_raises(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)
        with pytest.raises(InstrumentNotFoundError):
            fetcher.fetch_candles("UNKNOWN", start, end)

    def test_fetch_propagates_sdk_error(self) -> None:
        with patch(
            "finalayze.data.fetchers.tinkoff_data.asyncio.run",
            side_effect=RuntimeError("gRPC error"),
        ):
            fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 2, 1, tzinfo=UTC)
            with pytest.raises(DataFetchError, match="gRPC error"):
                fetcher.fetch_candles("SBER", start, end)

    def test_invalid_timeframe_raises(self) -> None:
        fetcher = TinkoffFetcher(token="fake", registry=_make_registry(), sandbox=True)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)
        with pytest.raises(DataFetchError, match="timeframe"):
            fetcher.fetch_candles("SBER", start, end, timeframe="5m")
```

### Step 2: Run to confirm failure

```bash
source ~/.zshrc && uv run pytest tests/unit/data/fetchers/test_tinkoff_data.py -v
```

Expected: `ERROR` — module `finalayze.data.fetchers.tinkoff_data` does not exist.

### Step 3: Implement `TinkoffFetcher`

Create `src/finalayze/data/fetchers/tinkoff_data.py`:

```python
"""Tinkoff Invest MOEX data fetcher (Layer 2).

Fetches OHLCV candles from MOEX via the t-tech-investments gRPC SDK.
Wraps async SDK calls in asyncio.run() to provide a sync interface
consistent with BaseFetcher.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from t_tech.invest import AsyncClient, CandleInterval
from t_tech.invest.sandbox import SandboxClient

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher

if TYPE_CHECKING:
    from finalayze.markets.instruments import InstrumentRegistry

_TIMEFRAME_MAP: dict[str, CandleInterval] = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

_MOEX_MARKET_ID = "moex"
_TINKOFF_SOURCE = "tinkoff"
_NANO_DIVISOR = Decimal("1_000_000_000")


class TinkoffFetcher(BaseFetcher):
    """Fetch MOEX candles from Tinkoff Invest gRPC API.

    Uses sandbox endpoint when sandbox=True (default for development).
    FIGI lookup is handled via InstrumentRegistry — raises InstrumentNotFoundError
    if the symbol is not registered.
    """

    def __init__(
        self,
        token: str,
        registry: InstrumentRegistry,
        *,
        sandbox: bool = True,
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles for a MOEX symbol."""
        if timeframe not in _TIMEFRAME_MAP:
            supported = ", ".join(sorted(_TIMEFRAME_MAP))
            msg = f"Unsupported timeframe '{timeframe}'. Supported: {supported}"
            raise DataFetchError(msg)

        figi = self._symbol_to_figi(symbol)
        interval = _TIMEFRAME_MAP[timeframe]

        try:
            raw_candles = asyncio.run(self._fetch_async(figi, start, end, interval))
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching {symbol}: {exc}"
            raise DataFetchError(msg) from exc

        return [self._map_candle(c, symbol) for c in raw_candles]

    async def _fetch_async(
        self,
        figi: str,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> list[object]:
        """Async call to Tinkoff SDK get_all_candles."""
        client_cls = SandboxClient if self._sandbox else AsyncClient
        async with client_cls(self._token) as client:
            candles = []
            async for candle in client.get_all_candles(
                figi=figi,
                from_=start,
                to=end,
                interval=interval,
            ):
                candles.append(candle)
            return candles

    def _symbol_to_figi(self, symbol: str) -> str:
        """Look up FIGI for a MOEX symbol via the instrument registry."""
        instrument = self._registry.get(symbol, _MOEX_MARKET_ID)
        if instrument.figi is None:
            msg = f"Instrument '{symbol}' has no FIGI assigned"
            raise InstrumentNotFoundError(msg)
        return instrument.figi

    def _quotation_to_decimal(self, q: object) -> Decimal:
        """Convert Tinkoff Quotation(units, nano) to Decimal.

        Quotation.units: integer part
        Quotation.nano: fractional part in billionths (1/1_000_000_000)
        """
        from t_tech.invest.schemas import Quotation  # type: ignore[import]

        assert isinstance(q, Quotation)  # noqa: S101
        return Decimal(q.units) + Decimal(q.nano) / _NANO_DIVISOR

    def _map_candle(self, raw: object, symbol: str) -> Candle:
        """Map a Tinkoff HistoricCandle to our Candle schema."""
        from datetime import UTC

        from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import]

        ts: Timestamp = raw.time  # type: ignore[attr-defined]
        timestamp = datetime.fromtimestamp(
            ts.seconds + ts.nanos / 1e9, tz=UTC
        )

        return Candle(
            symbol=symbol,
            market_id=_MOEX_MARKET_ID,
            timeframe="1d",
            timestamp=timestamp,
            open=self._quotation_to_decimal(raw.open),  # type: ignore[attr-defined]
            high=self._quotation_to_decimal(raw.high),  # type: ignore[attr-defined]
            low=self._quotation_to_decimal(raw.low),  # type: ignore[attr-defined]
            close=self._quotation_to_decimal(raw.close),  # type: ignore[attr-defined]
            volume=int(raw.volume),  # type: ignore[attr-defined]
            source=_TINKOFF_SOURCE,
        )
```

> **Implementation note:** The `_quotation_to_decimal` and `_map_candle` methods use `object` type for the raw Tinkoff types because mypy cannot resolve `t_tech.*` stubs. The `# type: ignore[attr-defined]` comments suppress attribute access warnings. This is standard practice for third-party gRPC clients without stub files.

> **Test note:** The tests mock `asyncio.run` at the module level to avoid actual gRPC calls. The `_fetch_async` and `_map_candle` internal methods are tested indirectly through `fetch_candles`.

### Step 4: Run the tests — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/data/fetchers/test_tinkoff_data.py -v
```

Expected: ALL tests pass.

> If `t_tech` is not yet installed (uv sync was skipped), tests that import the module will fail with `ModuleNotFoundError`. Run `uv sync --extra dev` first.

### Step 5: Run full quality suite

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero errors.

### Step 6: Commit

```bash
git add src/finalayze/data/fetchers/tinkoff_data.py tests/unit/data/fetchers/test_tinkoff_data.py
git commit -m "feat(data): add TinkoffFetcher for MOEX candle data via gRPC"
```

---

## Task 4: AlpacaBroker — US paper/live broker via alpaca-py

**Files:**
- Create: `src/finalayze/execution/alpaca_broker.py`
- Create: `tests/unit/execution/test_alpaca_broker.py`

### Step 1: Write the failing tests

Create `tests/unit/execution/test_alpaca_broker.py`:

```python
"""Unit tests for AlpacaBroker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InsufficientFundsError
from finalayze.execution.alpaca_broker import AlpacaBroker
from finalayze.execution.broker_base import OrderRequest


# ---------- helpers ----------


def _make_broker(paper: bool = True) -> AlpacaBroker:
    return AlpacaBroker(api_key="fake_key", secret_key="fake_secret", paper=paper)


def _mock_trading_client() -> MagicMock:
    client = MagicMock()
    # Default: get_account returns healthy account with enough buying power
    account = MagicMock()
    account.buying_power = "50000.00"
    account.cash = "50000.00"
    account.portfolio_value = "50000.00"
    account.status = "ACTIVE"
    client.get_account.return_value = account
    return client


def _mock_position(symbol: str, qty: str = "10", market_value: str = "1500.00") -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.market_value = market_value
    return pos


# ---------- tests ----------


class TestAlpacaBrokerSubmitOrder:
    def test_buy_order_success(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = "150.00"
        mock_order.filled_qty = "10"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("10"))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.symbol == "AAPL"
        assert result.side == "BUY"
        assert result.fill_price == Decimal("150.00")

    def test_sell_order_success(self) -> None:
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = "155.00"
        mock_order.filled_qty = "5"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal("5"))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.side == "SELL"
        assert result.fill_price == Decimal("155.00")

    def test_insufficient_funds_raises(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.submit_order.side_effect = Exception("insufficient buying power")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("1000"))
            with pytest.raises(InsufficientFundsError):
                broker.submit_order(order)

    def test_api_error_raises_broker_error(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.submit_order.side_effect = Exception("connection timeout")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("1"))
            with pytest.raises(BrokerError):
                broker.submit_order(order)

    def test_fill_candle_ignored(self) -> None:
        """AlpacaBroker must accept fill_candle=None (live broker doesn't need it)."""
        mock_client = _mock_trading_client()
        mock_order = MagicMock()
        mock_order.filled_avg_price = "150.00"
        mock_order.filled_qty = "1"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("1"))
            # fill_candle=None must not raise
            result = broker.submit_order(order, fill_candle=None)
        assert result.filled is True


class TestAlpacaBrokerGetPortfolio:
    def test_portfolio_returns_state(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = [
            _mock_position("AAPL", qty="10", market_value="1500.00"),
        ]

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.cash == Decimal("50000.00")
        assert "AAPL" in portfolio.positions

    def test_portfolio_api_error_raises(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_account.side_effect = Exception("API unavailable")

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            with pytest.raises(BrokerError):
                broker.get_portfolio()


class TestAlpacaBrokerHasPosition:
    def test_has_position_true(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = [
            _mock_position("AAPL", qty="10"),
        ]

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            assert broker.has_position("AAPL") is True

    def test_has_position_false(self) -> None:
        mock_client = _mock_trading_client()
        mock_client.get_all_positions.return_value = []

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = _make_broker()
            assert broker.has_position("MSFT") is False
```

### Step 2: Run to confirm failure

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_alpaca_broker.py -v
```

Expected: `ERROR` — module `finalayze.execution.alpaca_broker` does not exist.

### Step 3: Implement `AlpacaBroker`

Create `src/finalayze/execution/alpaca_broker.py`:

```python
"""Alpaca broker for US paper/live trading (Layer 5).

Uses alpaca-py SDK for order submission and portfolio management.
Paper trading uses Alpaca's paper endpoint; live uses production.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from finalayze.core.exceptions import BrokerError, InsufficientFundsError
from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_INSUFFICIENT_FUNDS_KEYWORDS = ("insufficient", "buying power", "not enough")


class AlpacaBroker(BrokerBase):
    """Alpaca paper/live broker for US market trading.

    Submits market orders via alpaca-py TradingClient.
    Raises BrokerError on API failures and InsufficientFundsError
    when buying power is too low.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
    ) -> None:
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

    def submit_order(
        self,
        order: OrderRequest,
        fill_candle: Candle | None = None,  # ignored for live broker
    ) -> OrderResult:
        """Submit a market order to Alpaca. fill_candle is not used."""
        side = OrderSide.BUY if order.side == "BUY" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=float(order.quantity),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            result = self._client.submit_order(order_data=request)
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(kw in exc_str for kw in _INSUFFICIENT_FUNDS_KEYWORDS):
                msg = f"Insufficient funds for {order.side} {order.quantity} {order.symbol}"
                raise InsufficientFundsError(msg) from exc
            msg = f"Alpaca order failed: {exc}"
            raise BrokerError(msg) from exc

        fill_price = (
            Decimal(str(result.filled_avg_price))
            if result.filled_avg_price
            else None
        )
        return OrderResult(
            filled=fill_price is not None,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(str(result.filled_qty or order.quantity)),
        )

    def get_portfolio(self) -> PortfolioState:
        """Return current portfolio state from Alpaca."""
        try:
            account = self._client.get_account()
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca portfolio fetch failed: {exc}"
            raise BrokerError(msg) from exc

        cash = Decimal(str(account.cash))
        pos_map: dict[str, Decimal] = {}
        position_value = Decimal(0)
        for pos in positions:
            qty = Decimal(str(pos.qty))
            pos_map[pos.symbol] = qty
            position_value += Decimal(str(pos.market_value))

        return PortfolioState(
            cash=cash,
            positions=pos_map,
            equity=cash + position_value,
            timestamp=datetime.now(tz=UTC),
        )

    def has_position(self, symbol: str) -> bool:
        """Return True if Alpaca account holds a non-zero position in symbol."""
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca positions fetch failed: {exc}"
            raise BrokerError(msg) from exc
        return any(p.symbol == symbol and Decimal(str(p.qty)) > 0 for p in positions)

    def get_positions(self) -> dict[str, Decimal]:
        """Return a copy of current Alpaca positions keyed by symbol."""
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            msg = f"Alpaca positions fetch failed: {exc}"
            raise BrokerError(msg) from exc
        return {p.symbol: Decimal(str(p.qty)) for p in positions}
```

### Step 4: Run the tests — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_alpaca_broker.py -v
```

Expected: ALL tests pass.

### Step 5: Run full quality suite

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero errors.

### Step 6: Commit

```bash
git add src/finalayze/execution/alpaca_broker.py tests/unit/execution/test_alpaca_broker.py
git commit -m "feat(execution): add AlpacaBroker for US paper/live trading"
```

---

## Task 5: TinkoffBroker — MOEX sandbox/live broker via t-tech-investments

**Files:**
- Create: `src/finalayze/execution/tinkoff_broker.py`
- Create: `tests/unit/execution/test_tinkoff_broker.py`

### Step 1: Write the failing tests

Create `tests/unit/execution/test_tinkoff_broker.py`:

```python
"""Unit tests for TinkoffBroker."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.tinkoff_broker import TinkoffBroker
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_broker(sandbox: bool = True) -> TinkoffBroker:
    return TinkoffBroker(token="fake_token", registry=_make_registry(), sandbox=sandbox)


def _fake_money(units: int, nano: int = 0) -> MagicMock:
    m = MagicMock()
    m.units = units
    m.nano = nano
    return m


class TestTinkoffBrokerSubmitOrder:
    def test_buy_order_success(self) -> None:
        mock_result = MagicMock()
        mock_result.order_id = "ord-123"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0
        mock_result.lots_executed = 1

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_result,
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal("10"))
            result = broker.submit_order(order)

        assert result.filled is True
        assert result.symbol == "SBER"
        assert result.side == "BUY"
        assert result.fill_price == Decimal("270")

    def test_unknown_symbol_raises(self) -> None:
        broker = _make_broker()
        order = OrderRequest(symbol="UNKNOWN", side="BUY", quantity=Decimal("10"))
        with pytest.raises(InstrumentNotFoundError):
            broker.submit_order(order)

    def test_api_error_raises_broker_error(self) -> None:
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=RuntimeError("gRPC unavailable"),
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal("10"))
            with pytest.raises(BrokerError, match="gRPC unavailable"):
                broker.submit_order(order)

    def test_fill_candle_ignored(self) -> None:
        """TinkoffBroker ignores fill_candle (live broker)."""
        mock_result = MagicMock()
        mock_result.order_id = "ord-456"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0
        mock_result.lots_executed = 1

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_result,
        ):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal("10"))
            result = broker.submit_order(order, fill_candle=None)
        assert result.filled is True

    def test_lot_size_rounding(self) -> None:
        """Quantity must be rounded down to nearest lot_size multiple.

        SBER has lot_size=10. Requesting qty=15 → actual qty=10.
        """
        submitted_quantities: list[int] = []

        def capture_run(coro: object) -> MagicMock:
            mock_result = MagicMock()
            mock_result.order_id = "ord-789"
            mock_result.executed_order_price.units = 270
            mock_result.executed_order_price.nano = 0
            mock_result.lots_executed = 1
            return mock_result

        with patch("finalayze.execution.tinkoff_broker.asyncio.run", side_effect=capture_run):
            broker = _make_broker()
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal("15"))
            result = broker.submit_order(order)

        # filled=True but actual quantity rounded to 10
        assert result.filled is True
        assert result.quantity == Decimal("10")


class TestTinkoffBrokerGetPortfolio:
    def test_portfolio_returned(self) -> None:
        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = 1_000_000
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_portfolio.positions = []

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.equity == Decimal("1000000")

    def test_portfolio_api_error_raises(self) -> None:
        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            side_effect=RuntimeError("gRPC timeout"),
        ):
            broker = _make_broker()
            with pytest.raises(BrokerError):
                broker.get_portfolio()
```

### Step 2: Run to confirm failure

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_tinkoff_broker.py -v
```

Expected: `ERROR` — module `finalayze.execution.tinkoff_broker` does not exist.

### Step 3: Implement `TinkoffBroker`

Create `src/finalayze/execution/tinkoff_broker.py`:

```python
"""Tinkoff Invest broker for MOEX sandbox/live trading (Layer 5).

Uses t-tech-investments gRPC SDK wrapped in asyncio.run() to provide
a sync interface consistent with BrokerBase.

Lot-size aware: MOEX shares trade in lots. Quantity is always rounded
down to the nearest multiple of the instrument's lot_size.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from t_tech.invest import AsyncClient, OrderDirection, OrderType
from t_tech.invest.sandbox import SandboxClient

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.markets.instruments import InstrumentRegistry

_MOEX_MARKET_ID = "moex"
_NANO_DIVISOR = Decimal("1_000_000_000")


class TinkoffBroker(BrokerBase):
    """Tinkoff Invest broker for MOEX market.

    Uses sandbox endpoint when sandbox=True (for development/testing).
    Lot-size aware: quantities are rounded down to the nearest lot multiple.
    Raises InstrumentNotFoundError for unknown symbols, BrokerError for API failures.
    """

    def __init__(
        self,
        token: str,
        registry: InstrumentRegistry,
        *,
        sandbox: bool = True,
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox

    def submit_order(
        self,
        order: OrderRequest,
        fill_candle: Candle | None = None,  # ignored for live broker
    ) -> OrderResult:
        """Submit a market order to Tinkoff Invest. fill_candle is not used."""
        instrument = self._registry.get(order.symbol, _MOEX_MARKET_ID)
        if instrument.figi is None:
            msg = f"Instrument '{order.symbol}' has no FIGI assigned"
            raise InstrumentNotFoundError(msg)

        # Round quantity down to nearest lot multiple
        lot_size = instrument.lot_size
        actual_qty = int(math.floor(float(order.quantity) / lot_size)) * lot_size

        if actual_qty <= 0:
            return OrderResult(
                filled=False,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal(0),
                reason=f"Quantity {order.quantity} is less than lot size {lot_size}",
            )

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if order.side == "BUY"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

        try:
            result = asyncio.run(
                self._post_order_async(instrument.figi, actual_qty, direction)
            )
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff order failed for {order.symbol}: {exc}"
            raise BrokerError(msg) from exc

        fill_price = self._quotation_to_decimal(result.executed_order_price)
        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(actual_qty),
        )

    async def _post_order_async(
        self,
        figi: str,
        quantity: int,
        direction: OrderDirection,
    ) -> object:
        """Async call to Tinkoff SDK post_order."""
        client_cls = SandboxClient if self._sandbox else AsyncClient
        async with client_cls(self._token) as client:
            return await client.orders.post_order(
                figi=figi,
                quantity=quantity,
                direction=direction,
                order_type=OrderType.ORDER_TYPE_MARKET,
                account_id="",  # Uses default account
            )

    def get_portfolio(self) -> PortfolioState:
        """Return current MOEX portfolio state from Tinkoff."""
        try:
            portfolio = asyncio.run(self._get_portfolio_async())
        except Exception as exc:
            msg = f"Tinkoff portfolio fetch failed: {exc}"
            raise BrokerError(msg) from exc

        total = self._quotation_to_decimal(portfolio.total_amount_portfolio)
        pos_map: dict[str, Decimal] = {}
        for pos in portfolio.positions:
            qty = self._quotation_to_decimal(pos.quantity)
            pos_map[pos.figi] = qty  # Tinkoff positions are FIGI-keyed

        return PortfolioState(
            cash=total,  # Tinkoff total_amount_portfolio ≈ equity
            positions=pos_map,
            equity=total,
            timestamp=datetime.now(tz=UTC),
        )

    async def _get_portfolio_async(self) -> object:
        """Async call to Tinkoff SDK get_portfolio."""
        client_cls = SandboxClient if self._sandbox else AsyncClient
        async with client_cls(self._token) as client:
            return await client.operations.get_portfolio(account_id="")

    def has_position(self, symbol: str) -> bool:
        """Return True if Tinkoff account holds a non-zero position in symbol."""
        instrument = self._registry.get(symbol, _MOEX_MARKET_ID)
        figi = instrument.figi
        portfolio = self.get_portfolio()
        held = portfolio.positions.get(figi or "", Decimal(0))
        return held > 0

    def get_positions(self) -> dict[str, Decimal]:
        """Return current Tinkoff positions (FIGI-keyed) as Decimal quantities."""
        return dict(self.get_portfolio().positions)

    @staticmethod
    def _quotation_to_decimal(q: object) -> Decimal:
        """Convert Tinkoff Quotation(units, nano) to Decimal."""
        units = getattr(q, "units", 0)
        nano = getattr(q, "nano", 0)
        return Decimal(units) + Decimal(nano) / _NANO_DIVISOR
```

### Step 4: Run the tests — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_tinkoff_broker.py -v
```

Expected: ALL tests pass.

### Step 5: Run full quality suite

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero errors.

### Step 6: Commit

```bash
git add src/finalayze/execution/tinkoff_broker.py tests/unit/execution/test_tinkoff_broker.py
git commit -m "feat(execution): add TinkoffBroker for MOEX sandbox/live trading"
```

---

## Task 6: BrokerRouter — route orders by market_id

**Files:**
- Create: `src/finalayze/execution/broker_router.py`
- Create: `tests/unit/execution/test_broker_router.py`

### Step 1: Write the failing tests

Create `tests/unit/execution/test_broker_router.py`:

```python
"""Unit tests for BrokerRouter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from finalayze.core.exceptions import BrokerError
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter


def _make_mock_broker(market_id: str) -> MagicMock:
    broker = MagicMock()
    broker.market_id = market_id
    return broker


def _make_router() -> tuple[BrokerRouter, MagicMock, MagicMock]:
    us_broker = _make_mock_broker("us")
    moex_broker = _make_mock_broker("moex")
    router = BrokerRouter({"us": us_broker, "moex": moex_broker})
    return router, us_broker, moex_broker


# ---------- tests ----------


class TestBrokerRouterRoute:
    def test_routes_us_order_to_alpaca(self) -> None:
        router, us_broker, _ = _make_router()
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("10"))
        routed = router.route(order, market_id="us")
        assert routed is us_broker

    def test_routes_moex_order_to_tinkoff(self) -> None:
        router, _, moex_broker = _make_router()
        order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal("10"))
        routed = router.route(order, market_id="moex")
        assert routed is moex_broker

    def test_unknown_market_raises_broker_error(self) -> None:
        router, _, _ = _make_router()
        order = OrderRequest(symbol="XYZ", side="BUY", quantity=Decimal("1"))
        with pytest.raises(BrokerError, match="No broker registered for market"):
            router.route(order, market_id="london")


class TestBrokerRouterSubmit:
    def test_submit_delegates_to_correct_broker(self) -> None:
        router, us_broker, _ = _make_router()
        expected_result = OrderResult(filled=True, symbol="AAPL", side="BUY")
        us_broker.submit_order.return_value = expected_result

        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("5"))
        result = router.submit(order, market_id="us")

        us_broker.submit_order.assert_called_once_with(order, fill_candle=None)
        assert result is expected_result

    def test_submit_moex_delegates_to_tinkoff(self) -> None:
        router, _, moex_broker = _make_router()
        expected_result = OrderResult(filled=True, symbol="SBER", side="SELL")
        moex_broker.submit_order.return_value = expected_result

        order = OrderRequest(symbol="SBER", side="SELL", quantity=Decimal("10"))
        result = router.submit(order, market_id="moex")

        moex_broker.submit_order.assert_called_once_with(order, fill_candle=None)
        assert result is expected_result


class TestBrokerRouterRegistration:
    def test_empty_router_raises_on_route(self) -> None:
        router = BrokerRouter({})
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("1"))
        with pytest.raises(BrokerError, match="No broker registered"):
            router.route(order, market_id="us")

    def test_registered_markets(self) -> None:
        router, _, _ = _make_router()
        assert set(router.registered_markets) == {"us", "moex"}
```

### Step 2: Run to confirm failure

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_broker_router.py -v
```

Expected: `ERROR` — module `finalayze.execution.broker_router` does not exist.

### Step 3: Implement `BrokerRouter`

Create `src/finalayze/execution/broker_router.py`:

```python
"""Broker router — dispatches orders to the correct broker by market ID (Layer 5).

Routes orders based on the order's market_id:
  - "us"   → AlpacaBroker (or any BrokerBase registered for "us")
  - "moex" → TinkoffBroker (or any BrokerBase registered for "moex")

Raises BrokerError if no broker is registered for the requested market_id.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from finalayze.core.exceptions import BrokerError
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult


class BrokerRouter:
    """Routes orders to the appropriate broker based on market_id.

    Example:
        router = BrokerRouter({
            "us": alpaca_broker,
            "moex": tinkoff_broker,
        })
        broker = router.route(order, market_id="us")
        result = router.submit(order, market_id="moex")
    """

    def __init__(self, brokers: dict[str, BrokerBase]) -> None:
        self._brokers = dict(brokers)

    def route(self, order: OrderRequest, market_id: str) -> BrokerBase:
        """Return the broker registered for market_id.

        Raises:
            BrokerError: If no broker is registered for the given market_id.
        """
        broker = self._brokers.get(market_id)
        if broker is None:
            registered = ", ".join(sorted(self._brokers)) or "(none)"
            msg = (
                f"No broker registered for market '{market_id}'. "
                f"Registered markets: {registered}"
            )
            raise BrokerError(msg)
        return broker

    def submit(
        self,
        order: OrderRequest,
        market_id: str,
        fill_candle: object = None,
    ) -> OrderResult:
        """Route and submit an order in one step.

        Args:
            order: The order to submit.
            market_id: The market this order belongs to.
            fill_candle: Optional candle for simulated brokers (None for live).

        Returns:
            OrderResult from the routed broker.
        """
        broker = self.route(order, market_id)
        return broker.submit_order(order, fill_candle=fill_candle)  # type: ignore[arg-type]

    @property
    def registered_markets(self) -> list[str]:
        """Return a sorted list of registered market IDs."""
        return sorted(self._brokers)
```

### Step 4: Run the tests — expect pass

```bash
source ~/.zshrc && uv run pytest tests/unit/execution/test_broker_router.py -v
```

Expected: ALL tests pass.

### Step 5: Run the complete test suite one final time

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest -q --no-header
```

Expected: zero lint/type errors, all tests pass.

### Step 6: Commit

```bash
git add src/finalayze/execution/broker_router.py tests/unit/execution/test_broker_router.py
git commit -m "feat(execution): add BrokerRouter to dispatch orders by market_id"
```

---

## Final verification before opening PR

Run the full quality suite from the repo root:

```bash
source ~/.zshrc && uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run pytest --cov=src/finalayze --cov-report=term-missing -q --no-header
```

Expected:
- ruff: 0 errors, 0 warnings
- mypy: Success (no issues)
- pytest: all tests pass, coverage ≥ 80% on new modules

Then invoke `superpowers:finishing-a-development-branch` to create the PR.
