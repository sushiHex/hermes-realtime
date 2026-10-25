"""Scoped interpreter binding and bounded transport refusal diagnostics."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/lookup_backend_gateway.py"


def interpreter_binding(actual, selected, install):
    tree = ast.parse(SOURCE.read_text())
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "interpreterExact"
    ]
    assert len(values) == 1
    expression = ast.Expression(values[0])
    return eval(
        compile(expression, str(SOURCE), "eval"),
        {
            "Path": Path,
            "sys": SimpleNamespace(executable=str(actual)),
            "manifest": {"hermes_python": str(selected)},
            "INSTALL": install,
        },
    )


def test_gateway_binds_selected_external_interpreter(tmp_path):
    selected = tmp_path / "separate-env/Scripts/python.exe"
    assert interpreter_binding(selected, selected, tmp_path / "hermes")


def test_default_clone_path_does_not_replace_selected_interpreter(tmp_path):
    install = tmp_path / "hermes"
    assert not interpreter_binding(
        install / "venv/Scripts/python.exe", tmp_path / "separate-env/Scripts/python.exe", install
    )


def test_matching_clone_interpreter_remains_supported(tmp_path):
    install = tmp_path / "hermes"
    selected = install / "venv/Scripts/python.exe"
    assert interpreter_binding(selected, selected, install)


def gateway():
    spec = importlib.util.spec_from_file_location("lookup_gateway_boundary", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def witness():
    return {
        "searchRequests": 0,
        "extractRequests": 0,
        "transportRefused": False,
        "transportResponses": [],
    }


def marker(output):
    lines = output.splitlines()
    assert len(lines) == 1 and len(lines[0].encode()) < 1024
    prefix = "[lookup-transport-refusal] "
    assert lines[0].startswith(prefix)
    return json.loads(lines[0][len(prefix) :])


def test_transport_refusal_emits_one_private_safe_marker(capsys):
    module, state = gateway(), witness()
    request = SimpleNamespace(url="https://private.invalid/secret", method="GET", body=None)
    send = module.bounded_exa_send(
        lambda *_a, **_k: pytest.fail("must not forward"),
        state,
        set(),
        lambda: None,
        SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="metadata_request_refused"):
        send(None, request)
    output = capsys.readouterr().out
    assert "private.invalid" not in output and "secret" not in output
    observed = marker(output)
    assert observed["category"] == "metadata_request_refused"
    assert observed["denialCount"] == 1 and observed["forwarded"] is False
    assert state["transportRefused"] is True


def test_transport_refusal_marker_survives_private_save_failure(capsys):
    module, state = gateway(), witness()

    def save():
        raise OSError("private-path-must-not-print")

    send = module.bounded_exa_send(
        lambda *_a, **_k: pytest.fail("must not forward"), state, set(), save, SimpleNamespace()
    )
    with pytest.raises(OSError):
        send(None, SimpleNamespace(url="https://private.invalid/", method="GET", body=None))
    output = capsys.readouterr().out
    assert "private-path" not in output
    assert marker(output)["category"] == "metadata_request_refused"


def test_success_does_not_emit_refusal_marker(capsys):
    module, state = gateway(), witness()
    response = SimpleNamespace(
        status_code=200, raw=SimpleNamespace(read=lambda *_a, **_k: b"{}"), close=lambda: None
    )
    session = SimpleNamespace(
        get_adapter=lambda _: SimpleNamespace(max_retries=SimpleNamespace(total=0))
    )
    url = "http://127.0.0.1:12345/v1/models"
    send = module.bounded_exa_send(
        lambda *_a, **_k: response, state, set(), lambda: None, SimpleNamespace(), frozenset({url})
    )
    assert send(session, SimpleNamespace(url=url, method="GET", body=None)) is response
    assert capsys.readouterr().out == ""
    assert state["transportRefused"] is False


def test_overflow_refusal_has_fixed_marker_and_preserves_private_cap(capsys):
    module, state = gateway(), witness()
    state.update(transportDenials=[{} for _ in range(8)], transportDenialCount=8)
    send = module.bounded_exa_send(
        lambda *_a, **_k: pytest.fail("must not forward"),
        state,
        set(),
        lambda: None,
        SimpleNamespace(),
    )
    with pytest.raises(RuntimeError):
        send(None, SimpleNamespace(url="https://private.invalid/", method="GET", body=None))
    observed = marker(capsys.readouterr().out)
    assert observed["category"] == "denial_overflow" and observed["denialCount"] == 9
    assert len(state["transportDenials"]) == 8 and state["transportDenialsTruncated"]
