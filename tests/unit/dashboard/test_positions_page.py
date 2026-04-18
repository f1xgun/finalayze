from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_positions_page_importable() -> None:
    import finalayze.dashboard.pages.positions  # noqa: F401


def test_positions_page_has_render_function() -> None:
    from finalayze.dashboard.pages import positions as page

    assert callable(page.render)
    import inspect

    sig = inspect.signature(page.render)
    assert "api" in sig.parameters


@pytest.mark.parametrize(
    ("distance_atr", "expected"),
    [
        (2.0, "green"),
        (1.51, "green"),
        (1.5, "yellow"),
        (1.0, "yellow"),
        (0.5, "yellow"),
        (0.49, "red"),
        (0.0, "red"),
        (-0.1, "red"),
        (None, "gray"),
    ],
)
def test_bucket_color_follows_d10_thresholds(distance_atr: float | None, expected: str) -> None:
    from finalayze.dashboard.pages.positions import _bucket_color

    assert _bucket_color(distance_atr) == expected


def test_render_history_chart_handles_empty_events(mocker: object) -> None:
    info = mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    from finalayze.dashboard.pages.positions import _render_history_chart

    _render_history_chart([], "SBER")
    info.assert_called_once()


def test_render_history_chart_builds_figure_with_traces(mocker: object) -> None:
    patched = mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]
    from finalayze.dashboard.pages.positions import _render_history_chart

    events = [
        {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "event_type": "entry",
            "current_stop": 95.0,
            "entry_price": 100.0,
            "highest_price": 100.0,
            "current_price": 100.0,
            "atr_value": 2.5,
            "trail_activated": False,
        },
        {
            "timestamp": "2026-04-18T10:15:00+00:00",
            "event_type": "snapshot",
            "current_stop": 96.0,
            "entry_price": 100.0,
            "highest_price": 102.0,
            "current_price": 103.0,
            "atr_value": 2.5,
            "trail_activated": True,
        },
    ]
    _render_history_chart(events, "SBER")
    patched.assert_called_once()
    fig = patched.call_args.args[0]
    # Must pass use_container_width=True (sandbox.py convention)
    assert patched.call_args.kwargs.get("use_container_width") is True
    # Must have >=3 traces: Price + Trailing stop + High-water
    assert len(fig.data) >= 3


def test_render_history_chart_handles_activation_event(mocker: object) -> None:
    """Regression (UAT 2026-04-18): fig.add_vline(x=Timestamp) raised
    ``TypeError: Addition/subtraction of integers and Timestamp...`` under
    pandas 2.x because Plotly does arithmetic on the x-coordinate when
    computing annotation placement. Fix: pass ISO string to add_vline.

    An event with event_type='activation' must render without raising.
    """
    patched = mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]
    from finalayze.dashboard.pages.positions import _render_history_chart

    events = [
        {
            "timestamp": "2026-04-18T10:00:00+00:00",
            "event_type": "entry",
            "current_stop": 95.0,
            "entry_price": 100.0,
            "highest_price": 100.0,
            "current_price": 100.0,
            "atr_value": 2.5,
            "trail_activated": False,
        },
        {
            "timestamp": "2026-04-18T10:30:00+00:00",
            "event_type": "activation",
            "current_stop": 98.0,
            "entry_price": 100.0,
            "highest_price": 105.0,
            "current_price": 105.0,
            "atr_value": 2.5,
            "trail_activated": True,
        },
    ]
    _render_history_chart(events, "SBER")
    patched.assert_called_once()
    fig = patched.call_args.args[0]
    # The activation event becomes a vertical line shape in fig.layout.shapes
    # and an annotation in fig.layout.annotations. Both must exist.
    shapes = fig.layout.shapes or ()
    annotations = fig.layout.annotations or ()
    assert len(shapes) >= 1, "add_vline should append a shape"
    assert any(ann.text == "Trail activated" for ann in annotations), (
        "annotation text 'Trail activated' missing"
    )


def test_render_heatmap_assigns_colors_per_position(mocker: object) -> None:
    """I-05: _render_heatmap must emit the correct color per ATR bucket.

    Verifies D-10 mapping end-to-end through the HTML-emission path --
    not just the pure _bucket_color helper. Mocks st.columns / st.markdown
    and inspects the HTML strings passed to markdown.
    """
    # st.columns() returns a list of context managers; each one needs
    # __enter__/__exit__ so `with cols[i]:` works.
    col_ctx = MagicMock()
    col_ctx.__enter__ = MagicMock(return_value=col_ctx)
    col_ctx.__exit__ = MagicMock(return_value=False)
    mocker.patch("streamlit.columns", return_value=[col_ctx, col_ctx, col_ctx])  # type: ignore[attr-defined]
    md = mocker.patch("streamlit.markdown")  # type: ignore[attr-defined]

    from finalayze.dashboard.pages.positions import _render_heatmap

    positions = [
        {"symbol": "RED_SYM", "distance_atr": 0.3},
        {"symbol": "YEL_SYM", "distance_atr": 1.0},
        {"symbol": "GRN_SYM", "distance_atr": 2.0},
    ]
    _render_heatmap(positions)

    # Collect all HTML strings passed to st.markdown in order
    html_calls = [c.args[0] if c.args else c.kwargs.get("body", "") for c in md.call_args_list]
    assert len(html_calls) == 3, f"expected 3 markdown calls, got {len(html_calls)}"
    # Bucket assignment per D-10
    assert "background:red" in html_calls[0] and "RED_SYM" in html_calls[0]
    assert "background:yellow" in html_calls[1] and "YEL_SYM" in html_calls[1]
    assert "background:green" in html_calls[2] and "GRN_SYM" in html_calls[2]
