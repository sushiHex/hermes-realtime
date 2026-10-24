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
    assert set(body) == {
        "model", "messages", "response_format", "max_tokens", "stream", "reasoning_effort"
    }
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][-1]["content"] == s.original.choices[0].message.content
    assert s.captured["sdk"][0]["max_retries"] == 0
    assert s.captured["http"][0]["follow_redirects"] is False
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["usage"]["total_tokens"] == 7
    assert s.row["presentation_evidence"]["wire"]["tools_absent"] is True
    assert s.row["presentation_evidence"]["wire"]["reasoning_effort_none"] is True
    assert s.row["presentation_evidence"]["wire_requests"] == 1
    assert s.run() is s.refused and len(s.captured["requests"]) == 1


def test_formatter_reasoning_control_does_not_modify_native_task_request(setup):
    s = setup
    s.request["reasoning_effort"] = "high"
    seen = []

    def native(request):
        seen.append(dict(request))
        return s.original

    assert s.run(next_call=native) is not s.refused
    assert seen == [s.request] and s.request["reasoning_effort"] == "high"
    assert s.captured["requests"][0]["reasoning_effort"] == "none"


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


@pytest.mark.parametrize("source,reason,refusal,kind,length,finish", [
    (None, "stop", False, "none", 0, "stop"),
    (42, "stop", False, "other", 0, "stop"),
    ("", "stop", False, "string", 0, "stop"),
    ("x" * 4097, "stop", False, "string", 4097, "stop"),
    ("synthetic", "length", False, "string", 9, "length"),
    ("synthetic", "untrusted-finish-text", False, "string", 9, "other"),
    ("synthetic", None, False, "string", 9, "missing"),
    ("synthetic", "stop", True, "string", 9, "stop"),
], ids=["none", "other", "empty", "oversize", "truncated", "unknown-finish",
        "missing-finish", "explicit-refusal"])
def test_source_refusal_records_only_fixed_categories_and_counts(
    setup, capsys, source, reason, refusal, kind, length, finish,
):
    s = setup
    s.original.choices[0].message.content = source
    s.original.choices[0].message.refusal = refusal
    s.original.choices[0].finish_reason = reason
    assert s.run() is s.refused and not s.captured["requests"]
    evidence = s.row["presentation_evidence"]
    assert evidence == {
        "category": "refused", "stage": "source", "calls": 0, "wire_requests": 0,
        "source_kind": kind, "source_length": length, "source_finish_reason": finish,
        "source_refusal": refusal, "failure_kind": "ValueError",
        "source_choices_count": 1, "source_tool_calls": False, "source_parsed_present": False,
        "source_reasoning_content_kind": "none", "source_reasoning_content_length": 0,
        "source_reasoning_kind": "none", "source_reasoning_length": 0,
    }
    output = capsys.readouterr().out
    assert "synthetic" not in output and "untrusted-finish-text" not in output


def test_presentation_refuses_explicit_formatter_refusal(setup):
    response = completion('{"voice":"v","detail":"d"}')
    response.choices[0].message.refusal = "synthetic-refusal"
    setup.captured["response"] = response
    assert setup.run() is setup.refused


@pytest.mark.parametrize("case,count,kind,length,finish,tools,refusal,parsed", [
    ("zero-choices", 0, "none", 0, "missing", False, False, False),
    ("extra-choices", 2, "string", 26, "stop", False, False, False),
    ("none-content", 1, "none", 0, "stop", False, False, False),
    ("other-content", 1, "other", 0, "stop", False, False, False),
    ("oversize", 1, "string", 8193, "stop", False, False, False),
    ("length-stop", 1, "string", 26, "length", False, False, False),
    ("unknown-finish", 1, "string", 26, "other", False, False, False),
    ("missing-finish", 1, "string", 26, "missing", False, False, False),
    ("tool-call", 1, "string", 26, "stop", True, False, False),
    ("refusal", 1, "string", 26, "stop", False, True, False),
    ("parsed-only", 1, "none", 0, "stop", False, False, True),
])
def test_formatter_envelope_refusal_retains_shape_and_usage_without_content(
    setup, capsys, case, count, kind, length, finish, tools, refusal, parsed,
):
    s = setup
    response = completion('{"voice":"v","detail":"d"}')
    message = response.choices[0].message
    if case == "zero-choices":
        response.choices = []
    elif case == "extra-choices":
        response.choices *= 2
    elif case in {"none-content", "parsed-only"}:
        message.content = None
    elif case == "other-content":
        message.content = {"private": "SYNTHETIC_SECRET"}
    elif case == "oversize":
        message.content = "x" * 8193
    elif case == "length-stop":
        response.choices[0].finish_reason = "length"
    elif case == "unknown-finish":
        response.choices[0].finish_reason = "SYNTHETIC_SECRET"
    elif case == "missing-finish":
        response.choices[0].finish_reason = None
    elif case == "tool-call":
        message.tool_calls = ["SYNTHETIC_SECRET"]
    elif case == "refusal":
        message.refusal = "SYNTHETIC_SECRET"
    if parsed:
        message.parsed = {"private": "SYNTHETIC_SECRET"}
    response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=2)
    s.captured["response"] = response
    assert s.run() is s.refused
    evidence = s.row["presentation_evidence"]
    assert evidence["stage"] == "response-envelope"
    assert evidence["response_choices_count"] == count
    assert evidence["response_kind"] == kind and evidence["response_length"] == length
    assert evidence["response_finish_reason"] == finish
    assert evidence["response_tool_calls"] is tools
    assert evidence["response_refusal"] is refusal
    assert evidence["response_parsed_present"] is parsed
    assert evidence["usage"] == {"prompt_tokens": 3, "completion_tokens": 4,
                                  "total_tokens": 7, "reasoning_tokens": 2}
    output = capsys.readouterr().out
    assert "SYNTHETIC_SECRET" not in output and "Full source result." not in output


def test_native_envelope_refusal_retains_shared_shape_without_formatter(setup):
    setup.original.choices = []
    assert setup.run() is setup.refused and not setup.captured["requests"]
    evidence = setup.row["presentation_evidence"]
    assert evidence["stage"] == "native-response"
    assert evidence["source_choices_count"] == 0 and evidence["source_kind"] == "none"


@pytest.mark.parametrize("case,rejection", [
    ("oversize-batch", "batch-envelope"),
    ("missing-call-id", "call-envelope"),
    ("empty-query", "search-arguments"),
    ("unreturned-url", "url-membership"),
    ("second-search", "batch-budget"),
    ("malformed-json", "call-json"),
    ("guard-unavailable", "generation-unavailable"),
    ("budget-unavailable", "budget-unavailable"),
    ("unlisted-value-error", "native-policy-error"),
])
def test_native_rejection_category_is_fixed_and_does_not_emit_arguments(
    setup, capsys, case, rejection,
):
    s = setup
    name = "web_search"
    args = {"query": "SYNTHETIC_PRIVATE_QUERY"}
    call_id = "synthetic-call"
    if case == "missing-call-id":
        call_id = ""
    elif case == "empty-query":
        args = {"query": ""}
    elif case == "unreturned-url":
        name, args = "web_extract", {"urls": ["https://example.invalid/SYNTHETIC_PRIVATE_URL"]}
    elif case == "unlisted-value-error":
        args["extra"] = float("nan")
    arguments = "SYNTHETIC_PRIVATE_INVALID_JSON" if case == "malformed-json" else json.dumps(args)
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    s.original.choices[0].message.content = None
    s.original.choices[0].message.tool_calls = [call] * (6 if case == "oversize-batch" else 1)
    s.original.choices[0].finish_reason = "tool_calls"

    def native(_):
        if case == "guard-unavailable":
            s.g.handles = []
        elif case == "budget-unavailable":
            s.g.budgets["selected"] = {}
        return s.original

    assert s.run(next_call=native) is s.refused
    evidence = s.row["presentation_evidence"]
    assert evidence["stage"] == "native-response" and evidence["rejection"] == rejection
    assert evidence["calls"] == 0 and not s.captured["requests"]
    output = capsys.readouterr().out
    assert "SYNTHETIC_PRIVATE" not in output and "Out of range float values" not in output


def test_unknown_native_guard_exception_does_not_emit_its_secret_message(setup, capsys):
    s = setup

    def native(_):
        def unknown():
            raise ValueError("SYNTHETIC_PRIVATE_EXCEPTION")

        s.g.scope = unknown
        return s.original

    assert s.run(next_call=native) is s.refused
    assert s.row["presentation_evidence"]["rejection"] == "native-policy-error"
    output = capsys.readouterr().out
    assert "[lookup-presentation]" in output and "SYNTHETIC_PRIVATE_EXCEPTION" not in output


def test_response_reasoning_facts_and_usage_are_bounded_observations(setup, capsys):
    s = setup
    response = completion(None)
    response.choices[0].message.reasoning_content = "SYNTHETIC_SECRET"
    response.choices[0].message.reasoning = {"private": "SYNTHETIC_SECRET"}
    response.usage = SimpleNamespace(prompt_tokens=True, completion_tokens=-1,
                                    total_tokens=1000000001,
                                    completion_tokens_details=SimpleNamespace(reasoning_tokens="secret"))
    s.captured["response"] = response
    assert s.run() is s.refused
    evidence = s.row["presentation_evidence"]
    assert evidence["response_reasoning_content_kind"] == "string"
    assert evidence["response_reasoning_content_length"] == len("SYNTHETIC_SECRET")
    assert evidence["response_reasoning_kind"] == "other"
    assert evidence["response_reasoning_length"] == 0 and evidence["usage"] == {}
    assert "SYNTHETIC_SECRET" not in capsys.readouterr().out


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
    shape_keys = {prefix + "_" + field for prefix in ("source", "response") for field in (
        "choices_count", "kind", "length", "finish_reason", "tool_calls", "refusal",
        "parsed_present", "reasoning_content_kind", "reasoning_content_length",
        "reasoning_kind", "reasoning_length")}
    assert set(evidence) == shape_keys | {
        "category", "stage", "calls", "wire_requests", "source_sha256", "wire", "usage",
        "local_http_settled"}


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
