"""Offline observation at the owned terminal and formatter boundaries; no network/model calls."""

import asyncio
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import lookup_model_capture as capture


def config(port=11441):
    base = f"http://127.0.0.1:{port}/v1"
    return {"model": {"provider": "custom", "base_url": base, "default": "fixture"},
            "plugins": {"entries": {"ordinary-lookup-experimental": {"settings": {
                "presentation": {"provider": "custom", "base_url": base, "model": "fixture"},
            }}}}}


def recorder(tmp_path, mode="metadata"):
    return capture.ModelResponseCapture(tmp_path, mode, {"model_port": 11441}, config())


def response(request, text="Public fixture — October 21, 2026."):
    return httpx.Response(200, request=request, json={"choices": [{
        "finish_reason": "stop", "message": {"role": "assistant", "content": text},
    }]})


def read(tmp_path):
    path = tmp_path / "model-response-capture.private.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_metadata_hashes_assistant_content_without_prompts_headers_or_raw_text(tmp_path):
    observer = recorder(tmp_path)
    text = "Public fixture — October 21, 2026."
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions",
                        headers={"authorization": "SYNTHETIC_PRIVATE_TOKEN"},
                        json={"messages": [{"content": "SYNTHETIC_PRIVATE_PROMPT"}],
                              "response_format": {"type": "json_schema"}})
    observer.observe(req, response(req, text))
    data = read(tmp_path)
    event = data["events"][0]
    assert event["content_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert event["content_chars"] == len(text)
    assert event["content_bytes"] == len(text.encode("utf-8"))
    assert event["finish_reason"] == "stop" and event["response_format"] is True
    assert data["available"] is True and "content" not in event
    assert not any(term in json.dumps(data) for term in (
        text, "SYNTHETIC_PRIVATE", "authorization", "messages", "11441",
    ))


@pytest.mark.parametrize("mode", ["off", "metadata", "controlled-public"])
def test_raw_content_requires_controlled_public_mode(tmp_path, mode):
    observer = recorder(tmp_path, mode)
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    observer.observe(req, response(req, "Public fixture"))
    if mode == "off":
        assert not (tmp_path / "model-response-capture.private.json").exists()
    else:
        event = read(tmp_path)["events"][0]
        assert (event.get("content") == "Public fixture") is (mode == "controlled-public")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434/v1/chat/completions", "http://127.0.0.1:11442/v1/chat/completions",
    "http://localhost:11441/v1/chat/completions", "https://127.0.0.1:11441/v1/chat/completions",
    "http://127.0.0.1:11441/v1/chat/completions?extra=1",
    "http://127.0.0.1:11441/api/chat", "http://127.0.0.1:11441/v1/models",
])
def test_other_endpoints_are_never_observed(tmp_path, url):
    observer = recorder(tmp_path, "controlled-public")
    req = httpx.Request("POST", url, json={})
    observer.observe(req, response(req, "UNRELATED_PRIVATE_CONTENT"))
    data = read(tmp_path)
    assert data["events"] == [] and data["observed"] == 0 and data["available"] is True
    assert "UNRELATED_PRIVATE" not in json.dumps(data)


@pytest.mark.parametrize("mode", [True, None, "raw", "CONTROLLED-PUBLIC"])
def test_invalid_mode_refuses_before_config_access(mode):
    with pytest.raises(ValueError, match="capture_mode"):
        capture.capture_endpoint(mode, {}, {})


@pytest.mark.parametrize("port", [True, 11434, 0, 65536, "11441"])
def test_capture_rejects_nonowned_or_invalid_model_port(port):
    with pytest.raises(ValueError, match="capture_route"):
        capture.capture_endpoint("metadata", {"model_port": port}, config())


def test_native_and_presentation_route_must_match_manifest():
    changed = config()
    changed["plugins"]["entries"]["ordinary-lookup-experimental"]["settings"][
        "presentation"]["base_url"] = "http://127.0.0.1:11442/v1"
    with pytest.raises(ValueError, match="capture_route"):
        capture.capture_endpoint("metadata", {"model_port": 11441}, changed)
    assert capture.capture_endpoint("off", {}, {}) is None


def test_capture_refuses_configured_port_that_differs_from_owned_port():
    with pytest.raises(ValueError, match="capture_route"):
        capture.capture_endpoint("controlled-public", {"model_port": 11442}, config(11441))


def test_capture_refuses_personal_port_even_when_both_configured_routes_match():
    with pytest.raises(ValueError, match="capture_route"):
        capture.capture_endpoint("controlled-public", {"model_port": 11434}, config(11434))


def test_raw_content_bound_keeps_metadata_but_marks_capture_unavailable(tmp_path):
    observer = recorder(tmp_path, "controlled-public")
    text = "x" * (capture.MAX_CONTENT_BYTES + 1)
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    observer.observe(req, response(req, text))
    data = read(tmp_path)
    assert data["available"] is False and data["unavailable_reason"] == "content_bound"
    assert "content" not in data["events"][0]
    assert data["events"][0]["content_bytes"] == len(text)


def test_response_and_event_bounds_are_not_silent_truncation(tmp_path):
    observer = recorder(tmp_path)
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    observer.observe(req, httpx.Response(200, request=req, content=b"x" * 65537))
    assert read(tmp_path)["unavailable_reason"] == "response_bound"
    for _ in range(capture.MAX_EVENTS + 1):
        observer.observe(req, response(req))
    data = read(tmp_path)
    assert data["available"] is False and len(data["events"]) == capture.MAX_EVENTS
    assert data["truncated"] is True
    path = tmp_path / "model-response-capture.private.json"
    assert path.stat().st_size <= capture.MAX_FILE_BYTES


def test_unread_and_streaming_responses_are_not_consumed(tmp_path):
    observer = recorder(tmp_path)
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    unread = httpx.Response(200, request=req, stream=httpx.ByteStream(b"unread"))
    observer.observe(req, unread)
    assert unread.is_stream_consumed is False
    assert read(tmp_path)["unavailable_reason"] == "unread_response"
    observer.observe(req, response(req), streaming=True)
    assert read(tmp_path)["unavailable_reason"] == "streaming_response"


def test_redirected_response_is_not_captured(tmp_path):
    observer = recorder(tmp_path, "controlled-public")
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    other = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions", json={})
    observer.observe(req, response(other, "UNRELATED_PRIVATE_CONTENT"))
    data = read(tmp_path)
    assert data["available"] is False and data["events"] == []
    assert data["unavailable_reason"] == "redirected_response"


def test_async_wrapper_preserves_arguments_and_leaves_sync_unmodified(tmp_path, monkeypatch):
    observer = recorder(tmp_path, "controlled-public")
    seen = []
    request = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    returned = response(request)

    def sync(client, req, *args, **kwargs):
        seen.append((client, req, args, kwargs))
        return returned

    async def asynchronous(client, req, *args, **kwargs):
        return sync(client, req, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", sync)
    monkeypatch.setattr(httpx.AsyncClient, "send", asynchronous)
    with observer.installed(SimpleNamespace(run_llm_execution_middleware=None)):
        client = object()
        assert httpx.Client.send(client, request, "positional", follow_redirects=False) is returned
        assert asyncio.run(httpx.AsyncClient.send(
            client, request, "positional", follow_redirects=False)) is returned
    assert seen == [(client, request, ("positional",), {"follow_redirects": False})] * 2
    assert httpx.Client.send is sync and httpx.AsyncClient.send is asynchronous
    assert len(read(tmp_path)["events"]) == 1
    assert read(tmp_path)["events"][0]["stage"] == "presentation"


def test_observation_failure_does_not_change_send_result(tmp_path, monkeypatch, capsys):
    observer = recorder(tmp_path)
    request = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    returned = response(request)
    monkeypatch.setattr(observer, "_write", lambda: (_ for _ in ()).throw(OSError("PRIVATE_ERROR")))
    async def send(*a, **k):
        return returned

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    with observer.installed(SimpleNamespace(run_llm_execution_middleware=None)):
        assert asyncio.run(httpx.AsyncClient.send(object(), request)) is returned
    assert observer.status["available"] is False
    assert observer.status["unavailable_reason"] == "write_failed"
    marker = capsys.readouterr().out
    assert "[lookup-model-capture] " in marker and "write_failed" in marker
    assert "PRIVATE_ERROR" not in marker and str(tmp_path) not in marker
    assert all(len(line) <= 256 for line in marker.splitlines())


def test_total_artifact_bound_and_concurrent_atomic_writes(tmp_path):
    observer = recorder(tmp_path, "controlled-public")
    req = httpx.Request("POST", "http://127.0.0.1:11441/v1/chat/completions", json={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: observer.observe(req, response(req, '"' * 16384)), range(8)))
    data = read(tmp_path)
    assert len(data["events"]) == 8 and data["observed"] == 8
    assert [event["index"] for event in data["events"]] == list(range(1, 9))
    assert data["available"] is False and data["unavailable_reason"] == "artifact_bound"
    assert (tmp_path / "model-response-capture.private.json").stat().st_size <= 150000
    assert not list(tmp_path.glob("*.tmp"))


def test_original_send_exception_is_preserved(tmp_path, monkeypatch):
    observer = recorder(tmp_path)
    error = RuntimeError("synthetic transport exception")

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(httpx.AsyncClient, "send", fail)
    with (observer.installed(SimpleNamespace(run_llm_execution_middleware=None)),
          pytest.raises(RuntimeError) as caught):
        asyncio.run(httpx.AsyncClient.send(object(), object()))
    assert caught.value is error


def test_isolated_gateway_can_load_its_adjacent_capture_helper():
    root = Path(__file__).resolve().parents[1]
    code = f"""
import importlib.util
from pathlib import Path
path = Path({str(root / 'scripts/lookup_backend_gateway.py')!r})
spec = importlib.util.spec_from_file_location('gateway', path)
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)
capture = gateway._model_capture_module()
assert Path(capture.__file__).resolve() == path.with_name('lookup_model_capture.py')
assert capture.validate_mode('metadata') == 'metadata'
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_gateway_rejects_invalid_mode_before_reading_manifest():
    script = Path(__file__).resolve().parents[1] / "scripts/lookup_backend_gateway.py"
    completed = subprocess.run([
        sys.executable, "-I", "-B", str(script), "missing-manifest",
        "--model-response-capture", "raw",
    ], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2 and "invalid choice" in completed.stderr
    assert "FileNotFoundError" not in completed.stderr



def native_context():
    return {"provider": "custom", "model": "fixture",
            "base_url": "http://127.0.0.1:11441/v1", "api_mode": "chat_completions"}


def test_native_terminal_captured_before_presenter_mutates_and_called_once(tmp_path):
    observer = recorder(tmp_path, "controlled-public")
    original_text = "Synthetic native preimage"
    result = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=original_text), finish_reason="stop")])
    seen = []
    request = {"response_format": {"type": "json_schema"}}

    def terminal(payload):
        seen.append(payload)
        return result

    def middleware(payload, next_call, **context):
        assert context == native_context()
        returned = next_call(payload)
        assert returned is result
        assert read(tmp_path)["events"][0]["content"] == original_text
        returned.choices[0].message.content = "Synthetic formatted result"
        return returned

    module = SimpleNamespace(run_llm_execution_middleware=middleware)
    with observer.installed(module):
        assert module.run_llm_execution_middleware(
            request, terminal, **native_context()) is result
    assert module.run_llm_execution_middleware is middleware
    assert seen == [request]
    event = read(tmp_path)["events"][0]
    assert event["stage"] == "native" and event["response_format"] is True
    assert event["content_sha256"] == hashlib.sha256(original_text.encode()).hexdigest()


@pytest.mark.parametrize("changed", [{"provider": "other"}, {"model": "other"},
                                    {"base_url": "http://127.0.0.1:11434/v1"}])
def test_native_context_scope_excludes_mismatch(tmp_path, changed):
    observer = recorder(tmp_path, "controlled-public")
    result = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="UNRELATED_PRIVATE"), finish_reason="stop")])
    module = SimpleNamespace(run_llm_execution_middleware=lambda req, call, **ctx: call(req))
    with observer.installed(module):
        assert module.run_llm_execution_middleware(
            {}, lambda req: result, **(native_context() | changed)) is result
    assert read(tmp_path)["observed"] == 0
    assert "UNRELATED_PRIVATE" not in json.dumps(read(tmp_path))


def test_native_shortcircuit_and_terminal_exception_preserve_outcome(tmp_path):
    observer = recorder(tmp_path)
    sentinel = object()
    module = SimpleNamespace(run_llm_execution_middleware=lambda *a, **k: sentinel)
    with observer.installed(module):
        assert module.run_llm_execution_middleware(
            {}, lambda req: pytest.fail("terminal called"), **native_context()) is sentinel
    assert read(tmp_path)["observed"] == 0
    error = RuntimeError("synthetic native failure")

    def fail(req):
        raise error

    module.run_llm_execution_middleware = lambda req, call, **ctx: call(req)
    with observer.installed(module), pytest.raises(RuntimeError) as caught:
        module.run_llm_execution_middleware({}, fail, **native_context())
    assert caught.value is error and read(tmp_path)["observed"] == 0



def test_native_observation_failure_and_bounds_preserve_result(tmp_path, monkeypatch):
    observer = recorder(tmp_path, "controlled-public")
    result = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="x" * (capture.MAX_CONTENT_BYTES + 1)),
        finish_reason="stop")])
    module = SimpleNamespace(run_llm_execution_middleware=lambda req, call, **ctx: call(req))
    with observer.installed(module):
        assert module.run_llm_execution_middleware({}, lambda req: result,
                                                   **native_context()) is result
    data = read(tmp_path)
    assert data["unavailable_reason"] == "content_bound"
    assert "content" not in data["events"][0]
    monkeypatch.setattr(observer, "observe_native",
                        lambda *a: (_ for _ in ()).throw(ValueError("PRIVATE_ERROR")))
    with observer.installed(module):
        assert module.run_llm_execution_middleware({}, lambda req: result,
                                                   **native_context()) is result
    assert read(tmp_path)["unavailable_reason"] == "observation_failed"
