"""Synchronous HTTP client for Streamlit dashboard.

Usage:
    client = ApiClient(base_url=st.secrets["api_url"], api_key=st.secrets["api_key"])
    data = client.get("/api/v1/portfolio").json()
"""

from __future__ import annotations

import httpx


class ApiClient:
    """Thin httpx wrapper that injects X-API-Key on every request."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._timeout = timeout

    def get(
        self,
        path: str,
        raise_on_error: bool = False,
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        resp = httpx.get(
            url,
            headers=self._headers,
            timeout=self._timeout,
            params=params,
        )
        if raise_on_error:
            resp.raise_for_status()
        return resp

    def post(
        self,
        path: str,
        raise_on_error: bool = False,
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url}{path}"
        resp = httpx.post(
            url,
            headers=self._headers,
            timeout=self._timeout,
            json=json,
        )
        if raise_on_error:
            resp.raise_for_status()
        return resp

    def list_alerts(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        alert_type: list[str] | None = None,
        symbol: str | None = None,
        priority: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, object]:
        """Fetch paginated alerts from /api/v1/alerts (Phase 57-04 ALRT-03).

        Builds a params dict that omits any unset filter so the FastAPI
        endpoint receives only the filters the caller actually supplied
        (matches the priority='All' sentinel handling on the page).

        Returns the parsed JSON envelope ``{alerts, total, page, page_size}``;
        on a non-dict response (network failure surfaced as text), returns
        an empty envelope so the dashboard never crashes.
        """
        params: dict[str, object] = {"page": page, "page_size": page_size}
        if alert_type:
            params["alert_type"] = alert_type
        if symbol:
            params["symbol"] = symbol
        if priority:
            params["priority"] = priority
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        result = self.get("/api/v1/alerts", params=params).json()
        if isinstance(result, dict):
            return result
        return {"alerts": [], "total": 0, "page": page, "page_size": page_size}

    def saa_target_allocation(self) -> dict[str, object]:
        """Fetch the SAA target allocation from /api/v1/saa/target-allocation (Phase 81).

        Returns the parsed JSON object on success; on a 404 (no active SAA portfolio) or any
        non-2xx / non-dict response, returns an empty dict so the dashboard renders a friendly
        empty state rather than crashing.
        """
        resp = self.get("/api/v1/saa/target-allocation")
        result = resp.json()
        if resp.is_success and isinstance(result, dict):
            return result
        return {}


# ── Convenience functions used by page modules ─────────────────────────────────


def get_health(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/health").json()


def get_health_feeds(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/health/feeds").json()


def get_system_status(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/system/status").json()


def get_system_errors(base_url: str, api_key: str) -> list[dict[str, object]]:
    result = ApiClient(base_url, api_key).get("/api/v1/system/errors").json()
    if isinstance(result, list):
        return result  # type: ignore[return-value]
    return []


def get_mode(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/mode").json()


def set_mode(
    base_url: str,
    api_key: str,
    mode: str,
    confirm_token: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"mode": mode}
    if confirm_token:
        payload["confirm_token"] = confirm_token
    return ApiClient(base_url, api_key).post("/api/v1/mode", json=payload).json()


def get_portfolio(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/portfolio").json()


def get_positions(base_url: str, api_key: str) -> list[dict[str, object]]:
    result = ApiClient(base_url, api_key).get("/api/v1/portfolio/positions").json()
    if isinstance(result, dict):
        positions = result.get("positions", [])
        if isinstance(positions, list):
            return positions  # type: ignore[return-value]
    return []


def get_portfolio_history(base_url: str, api_key: str) -> list[dict[str, object]]:
    result = ApiClient(base_url, api_key).get("/api/v1/portfolio/history").json()
    if isinstance(result, dict):
        snapshots = result.get("snapshots", [])
        if isinstance(snapshots, list):
            return snapshots  # type: ignore[return-value]
    return []


def get_portfolio_performance(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/portfolio/performance").json()


def get_trades(
    base_url: str,
    api_key: str,
    **kwargs: object,
) -> list[dict[str, object]]:
    params = {k: v for k, v in kwargs.items() if v is not None}
    result = ApiClient(base_url, api_key).get("/api/v1/trades", params=params).json()
    if isinstance(result, dict):
        trades = result.get("trades", [])
        if isinstance(trades, list):
            return trades  # type: ignore[return-value]
    return []


def get_signals(
    base_url: str,
    api_key: str,
    **kwargs: object,
) -> list[dict[str, object]]:
    params = {k: v for k, v in kwargs.items() if v is not None}
    result = ApiClient(base_url, api_key).get("/api/v1/signals", params=params).json()
    if isinstance(result, dict):
        signals = result.get("signals", [])
        if isinstance(signals, list):
            return signals  # type: ignore[return-value]
    return []


def get_strategies_performance(base_url: str, api_key: str) -> list[dict[str, object]]:
    result = ApiClient(base_url, api_key).get("/api/v1/strategies/performance").json()
    if isinstance(result, dict):
        strategies = result.get("strategies", [])
        if isinstance(strategies, list):
            return strategies  # type: ignore[return-value]
    return []


def get_risk_status(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/risk/status").json()


def get_risk_exposure(base_url: str, api_key: str) -> list[dict[str, object]]:
    result = ApiClient(base_url, api_key).get("/api/v1/risk/exposure").json()
    if isinstance(result, dict):
        segments = result.get("segments", [])
        if isinstance(segments, list):
            return segments  # type: ignore[return-value]
    return []


def post_risk_override(
    base_url: str,
    api_key: str,
    market_id: str,
    level: int,
) -> dict[str, object]:
    payload: dict[str, object] = {"market_id": market_id, "level": level}
    return ApiClient(base_url, api_key).post("/api/v1/risk/override", json=payload).json()


def get_ml_status(base_url: str, api_key: str) -> dict[str, object]:
    return ApiClient(base_url, api_key).get("/api/v1/ml/status").json()


def get_sandbox_metrics(
    base_url: str,
    api_key: str,
    days: int = 7,
    market_id: str = "moex",
) -> list[dict[str, object]]:
    """Fetch sandbox metric rows for the given date range and market."""
    params: dict[str, object] = {"days": days, "market_id": market_id}
    result = ApiClient(base_url, api_key).get("/api/v1/sandbox/metrics", params=params).json()
    if isinstance(result, list):
        return result  # type: ignore[return-value]
    return []


def get_sandbox_gonogo(base_url: str, api_key: str) -> dict[str, object]:
    """Fetch sandbox go/no-go gate evaluation report."""
    return ApiClient(base_url, api_key).get("/api/v1/sandbox/gonogo").json()
