"""Unit tests for the Streamlit /alerts dashboard page (Phase 57-04 Task 2).

Validates:
- ApiClient.list_alerts(...) helper exists with typed kwargs (D-16/D-18).
- render(api) calls /api/v1/alerts via the api_client helper.
- Filter inputs (alert_type, symbol, priority) flow through into request params.
- Anomaly threading: anomaly_llm rows are indented under their parent
  anomaly_raw row using the literal U+21B3 arrow glyph (D-15/D-17).
- Pagination caption appears (D-18).
- Empty state shows st.info "No alerts" (D-15).
- /alerts page is registered in dashboard/app.py navigation.

Test pattern follows Phase 56-05 lesson: ``inspect.getsource(render) +
inspect.getsource(module)`` so module-level constants are visible to
source-string assertions.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------- Helpers --------------------------------------------------------------


def _render_source() -> str:
    """Return render() body + module-level globals (Phase 56-05 lesson)."""
    from finalayze.dashboard.pages import alerts as page_mod

    return inspect.getsource(page_mod.render) + inspect.getsource(page_mod)


def _api_with_response(payload: dict) -> MagicMock:
    """Build a mock ApiClient whose .get(...).json() returns ``payload``."""
    api = MagicMock()
    api.get.return_value = SimpleNamespace(json=lambda: payload)
    return api


# ---------- Tests for the page ---------------------------------------------------


def test_render_calls_api_list_alerts() -> None:
    """render(api) must hit /api/v1/alerts at least once."""
    import streamlit as st  # noqa: PLC0415

    from finalayze.dashboard.pages import alerts as page_mod  # noqa: PLC0415

    api = _api_with_response({"alerts": [], "total": 0, "page": 1, "page_size": 50})

    # Patch the streamlit widgets so render() doesn't crash outside of an app.
    with (
        patch.object(st, "title"),
        patch.object(st, "selectbox", return_value="All"),
        patch.object(st, "multiselect", return_value=[]),
        patch.object(st, "text_input", return_value=""),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch.object(st, "caption"),
        patch.object(st, "info"),
        patch.object(st, "dataframe"),
        patch.object(st, "subheader"),
        patch.object(st, "number_input", return_value=1),
    ):
        page_mod.render(api)

    # Verify at least one /api/v1/alerts call happened.
    paths_called = [c.args[0] for c in api.get.call_args_list]
    assert any("/api/v1/alerts" in p for p in paths_called), (
        f"Expected /api/v1/alerts call; got paths={paths_called}"
    )


def test_render_passes_filter_params() -> None:
    """alert_type / symbol / priority widget values must flow into params=."""
    import streamlit as st  # noqa: PLC0415

    from finalayze.dashboard.pages import alerts as page_mod  # noqa: PLC0415

    api = _api_with_response({"alerts": [], "total": 0, "page": 1, "page_size": 50})

    with (
        patch.object(st, "title"),
        patch.object(st, "selectbox", return_value="INFO"),
        patch.object(st, "multiselect", return_value=["stop_loss"]),
        patch.object(st, "text_input", return_value="SBER"),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch.object(st, "caption"),
        patch.object(st, "info"),
        patch.object(st, "dataframe"),
        patch.object(st, "subheader"),
        patch.object(st, "number_input", return_value=1),
    ):
        page_mod.render(api)

    # Inspect the kwargs of the first /api/v1/alerts call.
    alert_calls = [
        c for c in api.get.call_args_list if "/api/v1/alerts" in c.args[0]
    ]
    assert alert_calls, "Expected at least one /api/v1/alerts call"
    params = alert_calls[0].kwargs.get("params", {})
    assert params.get("alert_type") == ["stop_loss"], (
        f"alert_type should be ['stop_loss'], got {params.get('alert_type')!r}"
    )
    assert params.get("symbol") == "SBER", (
        f"symbol should be 'SBER', got {params.get('symbol')!r}"
    )
    assert params.get("priority") == "INFO", (
        f"priority should be 'INFO', got {params.get('priority')!r}"
    )


def test_render_anomaly_threading() -> None:
    """anomaly_llm rows must be indented below their anomaly_raw parent."""
    import streamlit as st  # noqa: PLC0415

    from finalayze.dashboard.pages import alerts as page_mod  # noqa: PLC0415

    parent_id = "p1"
    payload = {
        "alerts": [
            {
                "id": parent_id,
                "timestamp": "2026-04-20T10:00:00+00:00",
                "alert_type": "anomaly_raw",
                "priority": "CRITICAL",
                "symbol": "SBER",
                "market_id": "moex",
                "message": "RAW anomaly",
                "parent_id": None,
                "delivery_status": "sent",
            },
            {
                "id": "c1",
                "timestamp": "2026-04-20T10:00:05+00:00",
                "alert_type": "anomaly_llm",
                "priority": "INFO",
                "symbol": "SBER",
                "market_id": "moex",
                "message": "LLM follow-up",
                "parent_id": parent_id,
                "delivery_status": "sent",
            },
        ],
        "total": 2,
        "page": 1,
        "page_size": 50,
    }
    api = _api_with_response(payload)

    captured_dataframes: list[object] = []

    def _capture_df(df: object, *_args: object, **_kwargs: object) -> None:
        captured_dataframes.append(df)

    with (
        patch.object(st, "title"),
        patch.object(st, "selectbox", return_value="All"),
        patch.object(st, "multiselect", return_value=[]),
        patch.object(st, "text_input", return_value=""),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch.object(st, "caption"),
        patch.object(st, "info"),
        patch.object(st, "dataframe", side_effect=_capture_df),
        patch.object(st, "subheader"),
        patch.object(st, "number_input", return_value=1),
    ):
        page_mod.render(api)

    # The page should render a dataframe whose rows include both alerts.
    assert captured_dataframes, "Expected at least one st.dataframe call"
    rendered = captured_dataframes[0]
    # Convert to list-of-dicts in a duck-typed way.
    if hasattr(rendered, "to_dict"):
        rows = rendered.to_dict("records")  # type: ignore[attr-defined]
    elif isinstance(rendered, list):
        rows = rendered
    else:
        # Fall back to inspecting str() — must contain both messages.
        assert "RAW anomaly" in str(rendered)
        assert "LLM follow-up" in str(rendered)
        return

    assert len(rows) == 2, f"Expected 2 rendered rows, got {len(rows)}"
    # Parent (anomaly_raw) row first, child (anomaly_llm) second.
    assert "RAW anomaly" in str(rows[0]), f"First row should be the parent: {rows[0]}"
    # Child message must be indented with the U+21B3 arrow glyph.
    child_msg = str(rows[1])
    assert "\u21b3" in child_msg or "↳" in child_msg, (
        f"Child anomaly_llm row must be indented with ↳ (U+21B3), got: {child_msg!r}"
    )


def test_render_includes_pagination_caption() -> None:
    """When total > page_size, page indicator (e.g., 'Page 2 of 3') is shown."""
    import streamlit as st  # noqa: PLC0415

    from finalayze.dashboard.pages import alerts as page_mod  # noqa: PLC0415

    payload = {"alerts": [], "total": 120, "page": 2, "page_size": 50}
    api = _api_with_response(payload)

    captions: list[str] = []

    def _capture_caption(text: str, *_args: object, **_kwargs: object) -> None:
        captions.append(text)

    with (
        patch.object(st, "title"),
        patch.object(st, "selectbox", return_value="All"),
        patch.object(st, "multiselect", return_value=[]),
        patch.object(st, "text_input", return_value=""),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch.object(st, "caption", side_effect=_capture_caption),
        patch.object(st, "info"),
        patch.object(st, "dataframe"),
        patch.object(st, "subheader"),
        patch.object(st, "number_input", return_value=2),
    ):
        page_mod.render(api)

    # Some caption must mention the pagination — Page X of Y or similar.
    joined = " | ".join(captions).lower()
    assert "page 2" in joined and "3" in joined, (
        f"Expected pagination caption mentioning page 2 of 3, got captions={captions}"
    )


def test_render_shows_info_when_empty() -> None:
    """Empty payload renders st.info containing 'No alerts'."""
    import streamlit as st  # noqa: PLC0415

    from finalayze.dashboard.pages import alerts as page_mod  # noqa: PLC0415

    api = _api_with_response({"alerts": [], "total": 0, "page": 1, "page_size": 50})

    info_messages: list[str] = []

    def _capture_info(text: str, *_args: object, **_kwargs: object) -> None:
        info_messages.append(text)

    with (
        patch.object(st, "title"),
        patch.object(st, "selectbox", return_value="All"),
        patch.object(st, "multiselect", return_value=[]),
        patch.object(st, "text_input", return_value=""),
        patch.object(st, "columns", return_value=[MagicMock(), MagicMock(), MagicMock()]),
        patch.object(st, "caption"),
        patch.object(st, "info", side_effect=_capture_info),
        patch.object(st, "dataframe"),
        patch.object(st, "subheader"),
        patch.object(st, "number_input", return_value=1),
    ):
        page_mod.render(api)

    joined = " | ".join(info_messages)
    assert "No alerts" in joined, f"Expected 'No alerts' in info, got {info_messages}"


def test_alerts_page_registered_in_app() -> None:
    """app.py navigation must include pages/alerts.py."""
    src = Path("src/finalayze/dashboard/app.py").read_text()
    assert "pages/alerts.py" in src, (
        "Expected dashboard/app.py to register the /alerts page via st.Page('pages/alerts.py')"
    )


def test_api_client_list_alerts_helper() -> None:
    """ApiClient.list_alerts builds the right path + params dict."""
    from finalayze.dashboard.api_client import ApiClient  # noqa: PLC0415

    captured: dict[str, object] = {}

    class _StubClient(ApiClient):
        def get(  # type: ignore[override]
            self,
            path: str,
            raise_on_error: bool = False,
            params: dict[str, object] | None = None,
        ) -> object:
            captured["path"] = path
            captured["params"] = params

            class _Resp:
                @staticmethod
                def json() -> dict[str, object]:
                    return {"alerts": [], "total": 0, "page": 1, "page_size": 50}

            return _Resp()

    client = _StubClient(base_url="http://x", api_key="k")
    result = client.list_alerts(page=1, alert_type=["signal"], symbol="SBER")

    assert captured["path"] == "/api/v1/alerts", (
        f"Expected path /api/v1/alerts, got {captured['path']!r}"
    )
    params = captured["params"]
    assert isinstance(params, dict)
    assert params.get("alert_type") == ["signal"], (
        f"alert_type should be ['signal'], got {params.get('alert_type')!r}"
    )
    assert params.get("symbol") == "SBER", (
        f"symbol should be 'SBER', got {params.get('symbol')!r}"
    )
    assert isinstance(result, dict)
    assert "alerts" in result


# ---------- Source-presence guards (Phase 56-05 lesson) --------------------------


def test_render_source_uses_arrow_glyph() -> None:
    """The render module must contain the literal ↳ glyph (U+21B3)."""
    src = _render_source()
    assert "\u21b3" in src or "↳" in src, (
        "Expected the literal U+21B3 arrow glyph in alerts.py for child indenting"
    )


def test_render_source_calls_list_alerts_path() -> None:
    """The render module source must reference /api/v1/alerts."""
    src = _render_source()
    assert "/api/v1/alerts" in src, (
        "Expected /api/v1/alerts in alerts.py source for the api.get path"
    )
