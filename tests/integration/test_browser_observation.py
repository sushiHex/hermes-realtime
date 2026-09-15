"""Shape enforcement for the browser self-acceptance observation line."""

from __future__ import annotations

import json

from tests.integration.test_browser_self_acceptance import _browser_observation

_PREFIX = "[browser-acceptance] "
_UNEXPECTED = "[unexpected-marker-shape]"


def _payload(line: str) -> dict[str, object]:
    assert line.startswith(_PREFIX)
    assert "\n" not in line
    decoded = json.loads(line[len(_PREFIX) :])
    assert type(decoded) is dict
    return decoded


def test_browser_observation_emits_one_bounded_content_free_line() -> None:
    line = _browser_observation(
        ("bootstrap_complete: 812.4 ms", "typed_input_ready: 1204.0 ms"),
        console_errors=0,
        typed_input_enabled=True,
    )

    payload = _payload(line)
    assert set(payload) == {
        "console_errors",
        "markers",
        "markers_dropped",
        "typed_input_enabled",
        "version",
    }
    assert payload["version"] == 1
    assert payload["markers"] == ["bootstrap_complete: 812.4 ms", "typed_input_ready: 1204.0 ms"]
    assert payload["markers_dropped"] == 0
    assert payload["console_errors"] == 0
    assert payload["typed_input_enabled"] is True


def test_browser_observation_refuses_unexpected_marker_shapes() -> None:
    secrets = (
        "navigated https://example.invalid/launch#unguessable: 3.0 ms",
        "authorization Bearer abc123deadbeef: 4.0 ms",
        "heard browser deterministic typed turn: 5.0 ms",
        "room browser-acceptance-0a1b2c3d4e: 6.0 ms",
    )

    line = _browser_observation(
        (*secrets, "bootstrap_complete: 7.0 ms"),
        console_errors=2,
        typed_input_enabled=False,
    )

    payload = _payload(line)
    assert payload["markers"] == [
        _UNEXPECTED,
        _UNEXPECTED,
        _UNEXPECTED,
        _UNEXPECTED,
        "bootstrap_complete: 7.0 ms",
    ]
    for leaked in (
        "example.invalid",
        "unguessable",
        "abc123deadbeef",
        "deterministic typed turn",
        "browser-acceptance-0a1b2c3d4e",
    ):
        assert leaked not in line
    assert payload["console_errors"] == 2
    assert payload["typed_input_enabled"] is False


def test_browser_observation_bounds_and_counts_dropped_markers() -> None:
    markers = tuple(f"marker_{index}: {index}.0 ms" for index in range(200))

    payload = _payload(
        _browser_observation(markers, console_errors=0, typed_input_enabled=True)
    )

    retained = payload["markers"]
    assert type(retained) is list
    assert len(retained) == 128
    assert retained[0] == "marker_72: 72.0 ms"
    assert retained[-1] == "marker_199: 199.0 ms"
    assert retained == list(markers[72:])
    assert payload["markers_dropped"] == 72


def test_browser_observation_reports_an_unreadable_page() -> None:
    line = _browser_observation(None, console_errors=0, typed_input_enabled=None)

    assert "\n" not in line
    assert '"markers":null' in line
    payload = _payload(line)
    assert payload["markers"] is None
    assert payload["typed_input_enabled"] is None
    assert payload["markers_dropped"] == 0
