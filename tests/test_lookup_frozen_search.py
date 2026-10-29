"""Explicit, source-bound replay without a provider or network."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location(
        "frozen_search_test", ROOT / "scripts/lookup_frozen_search.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(tmp_path, value=None):
    value = value or {"success": True, "data": {"web": [{
        "url": "https://example.invalid/source", "title": "Title", "description": "Dated passage.",
        "position": 1,
    }]}}
    raw = json.dumps(value).encode()
    path = tmp_path / "search.private.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), value


def test_default_has_no_fixture_or_file_access():
    assert helper().load_frozen_search(None, None, None) is None


@pytest.mark.parametrize("path,digest", [(None, "a" * 64), ("x", None), ("x", "secret")])
def test_requires_explicit_complete_opt_in(path, digest, capsys):
    with pytest.raises(ValueError, match="^frozen_search_config$"):
        helper().load_frozen_search(path, digest, "question")
    assert "secret" not in capsys.readouterr().out


def test_hash_binding_precedes_json_decode(tmp_path, capsys):
    path, _, _ = snapshot(tmp_path)
    with pytest.raises(ValueError, match="^frozen_search_digest$"):
        helper().load_frozen_search(path, "a" * 64, "question")
    assert '"category":"frozen_search_digest"' in capsys.readouterr().out


def test_retains_exact_rows_and_returns_independent_copy(tmp_path):
    path, digest, value = snapshot(tmp_path)
    replay = helper().load_frozen_search(path, digest, "question")
    result = replay.search("question", 5)
    assert result == value
    result["data"]["web"][0]["description"] = "changed"
    assert replay.result == value
    assert replay.summary() == {"mode": "frozen", "sha256": digest, "searches": 1,
                                "extraction_refused": False, "refused": False}


def test_query_scope_refuses_other_objective(tmp_path):
    path, digest, _ = snapshot(tmp_path)
    replay = helper().load_frozen_search(path, digest, "question")
    with pytest.raises(ValueError, match="^frozen_search_scope$"):
        replay.search("different", 5)
    assert replay.summary()["refused"] is True


def test_fixture_is_single_use_and_cannot_drop_rows(tmp_path):
    path, digest, _ = snapshot(tmp_path)
    replay = helper().load_frozen_search(path, digest, "question")
    with pytest.raises(ValueError, match="^frozen_search_scope$"):
        replay.search("question", 0)
    replay = helper().load_frozen_search(path, digest, "question")
    replay.search("question", 5)
    with pytest.raises(ValueError, match="^frozen_search_spent$"):
        replay.search("question", 5)


def test_extraction_is_explicit_fixture_capability_refusal(tmp_path):
    path, digest, _ = snapshot(tmp_path)
    replay = helper().load_frozen_search(path, digest, "question")
    with pytest.raises(ValueError, match="^frozen_extract_unavailable$"):
        replay.extract()
    assert replay.summary()["extraction_refused"] is True


@pytest.mark.parametrize("raw", [
    b"x" * 65537, b'{"success":true,"success":true,"data":{"web":[]}}'],
                         ids=["oversize", "duplicate-key"])
def test_input_bound_and_duplicate_keys_refuse(tmp_path, raw):
    path = tmp_path / "bad.private.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        helper().load_frozen_search(path, hashlib.sha256(raw).hexdigest(), "question")


@pytest.mark.parametrize("change", ["positions", "extra", "url", "row_type", "credentials"])
def test_fixture_schema_is_closed_and_ordered(tmp_path, change, capsys):
    path, _, value = snapshot(tmp_path)
    row = value["data"]["web"][0]
    if change == "positions":
        row["position"] = 2
    elif change == "extra":
        value["extra"] = "private"
    elif change == "url":
        row["url"] = "https://[invalid"
    elif change == "credentials":
        row["url"] = "https://user:PRIVATE@example.invalid/source"
    else:
        row["description"] = ["private"]
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="^frozen_search_schema$"):
        helper().load_frozen_search(path, hashlib.sha256(raw).hexdigest(), "question")
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_duplicate_key_refuses_an_otherwise_valid_fixture(tmp_path):
    path, _, _ = snapshot(tmp_path)
    raw = path.read_bytes().replace(b'"success": true', b'"success":true,"success":true')
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="^frozen_search_schema$"):
        helper().load_frozen_search(path, hashlib.sha256(raw).hexdigest(), "question")


def test_oversize_refuses_an_otherwise_valid_fixture(tmp_path):
    path, _, _ = snapshot(tmp_path)
    raw = path.read_bytes()
    raw += b" " * (65537 - len(raw))
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="^frozen_search_bound$"):
        helper().load_frozen_search(path, hashlib.sha256(raw).hexdigest(), "question")


def test_frozen_transport_never_forwards_web_even_outside_provider_scope():
    spec = importlib.util.spec_from_file_location(
        "frozen_gateway", ROOT / "scripts/lookup_backend_gateway.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    witness = {"searchRequests": 0, "extractRequests": 0, "transportResponses": [],
               "transportRefused": False}
    send = module.bounded_exa_send(lambda *_a, **_k: pytest.fail("fresh web request"),
                                   witness, set(), lambda: None, SimpleNamespace(
                                       search={"query": "q", "limit": 1, "spent": False}),
                                   frozen_replay=True)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "web_search_exa", "arguments": {"query": "q", "numResults": 1}}}
    request = SimpleNamespace(url="https://mcp.exa.ai/mcp", method="POST", body=json.dumps(body))
    with pytest.raises(RuntimeError):
        send(None, request, timeout=30)
    assert witness["transportRefused"] is True
