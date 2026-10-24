"""Model-free tests of the selected lookup's one tool-free presentation phase."""

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from aiohttp import web

SOURCE = Path(__file__).parent / "fixtures/plugin_generation_candidate/policy.py"
CONFIG = {
    "provider": "custom",
    "base_url": "https://model.example.invalid/v1",
    "model": "test-model",
    "api_key_env": "LOOKUP_TEST_KEY",
}


def completion(content, *, reason="stop", calls=None):
    return SimpleNamespace(
        id="original-id", model="test-model", created=42,
        choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(
            content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


@pytest.fixture
def setup(monkeypatch):
    for name in ("tools", "tools.approval", "tools.registry"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["tools.approval"].get_current_session_key = lambda **_: "selected"
    sys.modules["tools.registry"].registry = SimpleNamespace(get_entry=lambda _: None)
    spec = importlib.util.spec_from_file_location("lookup_presentation_policy", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {"requests": [], "closed": 0, "sdk": [], "http": []}

    class Http:
        def __init__(self, **kw):
            captured["http"].append(kw)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            if captured.get("exit_callback"):
                await captured["exit_callback"](type(self).__name__)
            captured["closed"] += 1

    class Client(Http):
        def __init__(self, **kw):
            captured["sdk"].append(kw)
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kw):
            captured["requests"].append(kw)
            for hook in captured["http"][-1]["event_hooks"]["request"]:
                await hook(SimpleNamespace(content=json.dumps(kw).encode()))
            if captured.get("callback"):
                await captured["callback"]()
            return captured.get(
                "response", completion('{"voice":"Short.","detail":"Full source result."}')
            )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=Http))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=Client))
    monkeypatch.setenv("LOOKUP_TEST_KEY", "local")
    formatter = module.Presentation(CONFIG)
    refused = object()
    g = module.Generation(lambda: refused, presentation=formatter)
    g.handles = [SimpleNamespace(active=True)]
    row = {"closed": False, "cancelled": False, "deadline": time.monotonic() + 1,
           "phase": "search", "settled": 1, "reserved": 1, "urls": set()}
    g.budgets["selected"] = row
    g.members = MappingProxyType({"selected": (module.VERSION, "identity", row)})
    g._membership_authority = g.members
    original = completion("Full source result.")
    request = {"model": "test-model", "messages": [{"role": "user", "content": "task"}],
               "tools": ["forbidden"], "tool_choice": "auto", "parallel_tool_calls": True,
               "functions": ["forbidden"], "function_call": "auto"}

    def run(**changes):
        kw = dict(api_mode="chat_completions", api_request_id="task-request",
                  provider=CONFIG["provider"], base_url=CONFIG["base_url"], model=CONFIG["model"],
                  request=request, next_call=lambda _: original)
        kw.update(changes)
        return g.model_gate(**kw)

    return SimpleNamespace(module=module, g=g, row=row, original=original, request=request,
                           run=run, captured=captured, refused=refused)


def test_presentation_is_one_fresh_tool_free_call_and_preserves_original(setup):
    s = setup
    result = s.run()
    assert result is not s.refused and result.id == s.original.id
    assert result.choices[0].message.content == (
        "<voice>Short.</voice><detail>Full source result.</detail>"
    )
    assert s.original.choices[0].message.content == "Full source result."
    assert result.usage is s.original.usage
    assert s.row["original_response"] is s.original
    assert len(s.captured["requests"]) == 1
    body = s.captured["requests"][0]
    assert set(body) == {"model", "messages", "response_format", "max_tokens", "stream"}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][-1]["content"] == s.original.choices[0].message.content
    assert s.captured["sdk"][0]["max_retries"] == 0
    assert s.captured["http"][0]["follow_redirects"] is False
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["usage"]["total_tokens"] == 7
    assert s.row["presentation_evidence"]["wire"]["tools_absent"] is True
    assert s.row["presentation_evidence"]["wire_requests"] == 1
    assert s.run() is s.refused and len(s.captured["requests"]) == 1


@pytest.mark.parametrize("field", ["provider", "base_url", "model"])
def test_presentation_refuses_foreign_route_before_sdk(setup, field):
    s = setup
    assert s.run(**{field: "foreign"}) is s.refused
    assert s.captured["requests"] == []


def test_presentation_refuses_request_model_mismatch(setup):
    s = setup
    s.request["model"] = "foreign"
    assert s.run() is s.refused and not s.captured["requests"]


def test_presentation_refuses_expired_admission_deadline(setup):
    s = setup
    s.row["deadline"] = time.monotonic() - 1
    assert s.run() is s.refused and not s.captured["requests"]


@pytest.mark.asyncio
async def test_selected_admission_has_one_120_second_deadline_without_replay_reset(setup):
    s = setup
    app = web.Application()
    s.g.wire(app, None)

    async def body():
        return {"input": "bounded synthetic task"}

    request = SimpleNamespace(method="POST", path="/v1/runs",
                              headers={"X-Lookup-Policy": s.module.VERSION},
                              match_info={}, json=body)

    async def native(_):
        return web.json_response({"run_id": "new-selected"}, status=202)

    before = time.monotonic()
    assert (await app.middlewares[-1](request, native)).status == 202
    deadline = s.g.budgets["new-selected"]["deadline"]
    assert before + 120 <= deadline <= time.monotonic() + 120
    assert (await app.middlewares[-1](request, native)).status == 202
    assert s.g.budgets["new-selected"]["deadline"] == deadline


def test_presentation_requires_explicit_credential_without_ambient_fallback(setup, monkeypatch):
    monkeypatch.delenv("LOOKUP_TEST_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-must-not-be-used")
    assert setup.run() is setup.refused and not setup.captured["sdk"]


@pytest.mark.parametrize("change", [{"extra": "unexpected"}, {"base_url": "https://user:pass@host/v1"},
                                   {"api_key_env": "invalid-env-name"}])
def test_presentation_config_is_explicit_and_strict(setup, change):
    with pytest.raises(ValueError, match="presentation-config"):
        setup.module.Presentation(CONFIG | change)


@pytest.mark.parametrize("source", ["x" * 4097, "", "   "], ids=["oversize", "empty", "whitespace"])
def test_presentation_refuses_unbounded_or_empty_source(setup, source):
    setup.original.choices[0].message.content = source
    assert setup.run() is setup.refused and not setup.captured["requests"]


@pytest.mark.parametrize("content", [
    '{"voice":"v","detail":"d","extra":"x"}',
    '{"voice":"v","voice":"v","detail":"d"}',
    '{"voice":5,"detail":"d"}',
    '{"voice":"' + "v" * 221 + '","detail":"d"}',
    '{"voice":"v","detail":"' + "d" * 1801 + '"}',
    '{"voice":"v</voice>","detail":"d"}',
    '{"voice":"v","detail":""}',
], ids=["extra-key", "duplicate-key", "wrong-type", "voice-bound", "detail-bound",
        "delimiter", "empty-detail"])
def test_presentation_refuses_invalid_content_without_repair(setup, content):
    s = setup
    s.captured["response"] = completion(content)
    assert s.run() is s.refused
    assert len(s.captured["requests"]) == 1 and "selected" in s.g.refused
    assert s.row["original_response"] is s.original


def test_presentation_refuses_tool_or_truncated_response(setup):
    s = setup
    s.captured["response"] = completion('{"voice":"v","detail":"d"}', reason="length")
    assert s.run() is s.refused


def test_presentation_refuses_explicit_native_source_refusal(setup):
    setup.original.choices[0].message.refusal = "synthetic-refusal"
    assert setup.run() is setup.refused and not setup.captured["requests"]


def test_presentation_refuses_explicit_formatter_refusal(setup):
    response = completion('{"voice":"v","detail":"d"}')
    response.choices[0].message.refusal = "synthetic-refusal"
    setup.captured["response"] = response
    assert setup.run() is setup.refused


def test_presentation_refuses_response_tool_calls(setup):
    s = setup
    s.captured["response"] = completion('{"voice":"v","detail":"d"}', calls=["not-allowed"])
    assert s.run() is s.refused


def test_presentation_refuses_extra_choices(setup):
    s = setup
    response = completion('{"voice":"v","detail":"d"}')
    response.choices *= 2
    s.captured["response"] = response
    assert s.run() is s.refused


@pytest.mark.parametrize("voice", ["line\nbreak", "control\x01", "**bold**", "# headline"],
                         ids=["newline", "control", "emphasis", "markup"])
def test_presentation_uses_existing_terminal_contract(setup, voice):
    setup.captured["response"] = completion(json.dumps({"voice": voice, "detail": "detail"}))
    assert setup.run() is setup.refused
    assert setup.row["presentation_evidence"]["failure_kind"] == "ValueError"


def test_presentation_never_falls_back_when_runtime_validator_unavailable(setup, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_realtime.integration.api", None)
    assert setup.run() is setup.refused
    assert setup.row["presentation_evidence"]["failure_kind"] == "ModuleNotFoundError"


@pytest.mark.parametrize("operation", ["stop", "unload"])
def test_native_stop_or_unload_cancels_owned_presentation(setup, operation):
    s = setup

    async def stop():
        if operation == "unload":
            s.g.close()
        else:
            app = web.Application()
            s.g.wire(app, None)
            request = SimpleNamespace(method="POST", path="/v1/runs/selected/stop",
                                      headers={}, match_info={})

            async def native(_):
                assert not s.row["cancelled"]
                return web.json_response({}, status=200)

            response = await app.middlewares[-1](request, native)
            assert response.status == 200
        await asyncio.sleep(0)

    s.captured["callback"] = stop
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["failure_kind"] == "CancelledError"


def test_presentation_never_publishes_after_deadline(setup):
    s = setup

    async def late():
        s.row["deadline"] = time.monotonic() - 1

    s.captured["callback"] = late
    assert s.run() is s.refused


def test_presentation_never_publishes_cancelled_result(setup):
    s = setup

    async def cancelled_after_http():
        s.row["cancelled"] = True  # Cancellation can race with completed HTTP/loop cleanup.

    s.captured["callback"] = cancelled_after_http
    assert s.run() is s.refused


def test_presentation_replaced_row_cannot_publish(setup):
    s = setup

    async def replace():
        replacement = dict(s.row)
        s.g.budgets["selected"] = replacement
        s.g.members = MappingProxyType({"selected": (s.module.VERSION, "identity", replacement)})
        s.g._membership_authority = s.g.members

    s.captured["callback"] = replace
    assert s.run() is s.refused


def test_presentation_records_only_bounded_categories_counts_hashes(setup, capsys):
    assert setup.run() is not setup.refused
    output = capsys.readouterr().out
    assert "Full source result." not in output and '"local"' not in output
    evidence = setup.row["presentation_evidence"]
    assert len(evidence["source_sha256"]) == 64
    assert set(evidence) == {"category", "stage", "calls", "wire_requests", "source_length",
                             "source_sha256", "wire", "usage", "local_http_settled"}


def test_presentation_cancellation_settles_client_and_never_publishes(setup):
    s = setup

    async def cancel():
        with s.g.lock:
            s.row["closed"] = s.row["cancelled"] = True
            s.g.cancel_presentation(s.row)
        await asyncio.sleep(0)

    s.captured["callback"] = cancel
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["failure_kind"] == "CancelledError"


def test_presentation_deadline_cancels_pending_request_and_settles(setup):
    s = setup
    s.row["deadline"] = time.monotonic() + .02

    async def stall():
        await asyncio.sleep(.08)  # A removed deadline must fail promptly, not hang a mutant.

    s.captured["callback"] = stall
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["failure_kind"] == "TimeoutError"


@pytest.mark.asyncio
async def test_presentation_refuses_existing_event_loop_before_sdk(setup):
    assert setup.run() is setup.refused
    assert not setup.captured["requests"]


@pytest.mark.parametrize("stop_before_cleanup", [True, False],
                         ids=["stop-then-unload", "stop-during-close"])
def test_presentation_cancellation_cannot_interrupt_suspended_cleanup(setup, stop_before_cleanup):
    s = setup

    async def stop():
        app = web.Application()
        s.g.wire(app, None)
        request = SimpleNamespace(method="POST", path="/v1/runs/selected/stop",
                                  headers={}, match_info={})

        async def native(_):
            return web.json_response({}, status=200)

        assert (await app.middlewares[-1](request, native)).status == 200

    async def provider():
        if stop_before_cleanup:
            await stop()
            await asyncio.sleep(0)

    async def during_exit(kind):
        assert s.row["presentation_active"] is not None
        if kind == "Client":
            if stop_before_cleanup:
                s.g.close()  # A second authorized cancellation during suspended SDK cleanup.
            else:
                await stop()  # First stop can also arrive after provider completion.
            await asyncio.sleep(.01)

    s.captured.update(callback=provider, exit_callback=during_exit)
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["local_http_settled"] is True


def test_presentation_retains_ownership_when_client_cleanup_fails(setup):
    async def fail(_):
        raise RuntimeError("synthetic cleanup failure")

    setup.captured["exit_callback"] = fail
    assert setup.run() is setup.refused
    assert setup.row["presentation_active"] is not None
    assert setup.row["presentation_evidence"]["local_http_settled"] is False


def test_presentation_requests_cancellation_once_per_active_handle(setup):
    s = setup

    async def twice():
        with s.g.lock:
            s.row["cancelled"] = True
            s.g.cancel_presentation(s.row)
            s.g.cancel_presentation(s.row)
        await asyncio.sleep(0)

    async def count(_):
        assert asyncio.current_task().cancelling() == 1

    s.captured.update(callback=twice, exit_callback=count)
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None


def test_presentation_own_deadline_cannot_cancel_cleanup(setup):
    s = setup
    s.row["deadline"] = time.monotonic() + .02

    async def slow_exit(kind):
        if kind == "Client":
            await asyncio.sleep(.04)

    s.captured["exit_callback"] = slow_exit
    assert s.run() is s.refused  # Expired results still never publish.
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["local_http_settled"] is True


def test_presentation_unload_cancels_from_authority_when_budget_container_is_corrupt(setup):
    s = setup

    async def corrupt_and_unload():
        s.g.budgets = []
        s.g.close()
        await asyncio.sleep(0)

    s.captured["callback"] = corrupt_and_unload
    assert s.run() is s.refused
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["failure_kind"] == "CancelledError"
