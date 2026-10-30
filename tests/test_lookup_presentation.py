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
CODEX_CONFIG = CONFIG | {"api_mode": "codex_responses"}


def completion(content, *, reason="stop", calls=None):
    return SimpleNamespace(
        id="original-id", model="test-model", created=42,
        choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(
            content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


def retrieve(generation, result, before_result=None):
    """Admit one search through the real model/tool/effect ticket chain."""
    args = {"query": "synthetic lookup"}
    call = SimpleNamespace(id="retrieval-call", function=SimpleNamespace(
        name="web_search", arguments=json.dumps(args)))
    response = completion(None, reason="tool_calls", calls=[call])
    response = generation.model_gate(
        api_mode="chat_completions", api_request_id="retrieval-request",
        request={"tools": [{"type": "function", "function": {"name": "web_search"}}]},
        next_call=lambda _: response)
    call = response.choices[0].message.tool_calls[0]

    def original(_args):
        if before_result:
            before_result()
        return json.dumps(result)

    return generation.tool_gate(
        tool_name="web_search", tool_call_id=call.id, api_request_id="retrieval-request", args=args,
        next_call=generation.wrap_search(original),
    )


@pytest.fixture
def setup(monkeypatch, request):
    retrieval = getattr(request, "param", {
        "success": True, "data": {"web": [{"url": "https://example.invalid/source"}]},
    })
    session = {"run": "selected"}
    for name in ("tools", "tools.approval", "tools.registry", "agent", "agent.codex_headers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["tools.approval"].get_current_session_key = lambda **_: session["run"]
    sys.modules["tools.registry"].registry = SimpleNamespace(get_entry=lambda _: None)
    sys.modules["agent.codex_headers"].codex_cloudflare_headers = (
        lambda token, *, base_url: {"X-Codex-Test": "explicit-token-bound"}
    )
    spec = importlib.util.spec_from_file_location("lookup_presentation_policy", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {"requests": [], "responses_requests": [], "closed": 0,
                "stream_closed": 0, "sdk": [], "http": []}

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
            self.responses = SimpleNamespace(create=self.create_responses)

        async def create(self, **kw):
            captured["requests"].append(kw)
            for hook in captured["http"][-1]["event_hooks"]["request"]:
                await hook(SimpleNamespace(content=json.dumps(kw).encode()))
            if captured.get("callback"):
                await captured["callback"]()
            return captured.get(
                "response", completion(
                    '{"voice":"Short.","detail":"Full source result.","source_ids":[0]}')
            )

        async def create_responses(self, **kw):
            if captured.get("wire_mutation"):
                captured["wire_mutation"](kw)
            captured["responses_requests"].append(kw)
            for hook in captured["http"][-1]["event_hooks"]["request"]:
                await hook(SimpleNamespace(content=json.dumps(kw).encode()))
            if captured.get("callback"):
                await captured["callback"]()
            return ResponseStream(captured.get("stream_events", codex_events()))

    class ResponseStream:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            async def iterate():
                for event in self.events:
                    yield event
            return iterate()

        async def close(self):
            captured["stream_closed"] += 1
            if captured.get("stream_close_callback"):
                await captured["stream_close_callback"]()

    def codex_message(text, *, phase="final_answer", item_id="msg_native"):
        return SimpleNamespace(
            type="message", id=item_id, role="assistant", status="completed", phase=phase,
            content=[SimpleNamespace(type="output_text", text=text)],
        )

    def codex_response(output, *, status="completed", usage=None,
                       incomplete_details=None, output_text=None):
        return SimpleNamespace(
            id="resp_native", object="response", status=status,
            output=output, output_text=output_text, usage=usage,
            incomplete_details=incomplete_details,
        )

    def codex_events(response=None, *, deltas=None):
        response = response or codex_response([
            codex_message(
                '{"voice":"Short.","detail":"Full source result.","source_ids":[0]}')
        ], usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7))
        events = [SimpleNamespace(type="response.created")]
        events.extend(SimpleNamespace(type="response.output_text.delta", delta=delta)
                      for delta in (deltas or ['{"voice":"Short."}']))
        events.append(SimpleNamespace(type="response.completed", response=response))
        return events

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=Http))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=Client))
    monkeypatch.setenv("LOOKUP_TEST_KEY", "local")
    formatter = module.Presentation(CONFIG)
    refused = object()
    g = module.Generation(lambda: refused, presentation=formatter)
    g.handles = [SimpleNamespace(active=True)]
    row = {"closed": False, "cancelled": False, "deadline": time.monotonic() + 1,
           "phase": "new", "settled": 0, "reserved": 0, "urls": set(),
           "objective": "synthetic lookup"}
    g.budgets["selected"] = row
    g.members = MappingProxyType({"selected": (module.VERSION, "identity", row)})
    g._membership_authority = g.members
    if retrieval is not None:
        retrieve(g, retrieval)
    original = completion("Full source result.")
    request = {"model": "test-model", "messages": [{"role": "user", "content": "task"}],
               "tools": ["forbidden"], "tool_choice": "auto", "parallel_tool_calls": True,
               "functions": ["forbidden"], "function_call": "auto"}

    def run(**changes):
        kw = dict(api_mode="chat_completions", api_request_id="task-request",
                  provider=CONFIG["provider"], base_url=CONFIG["base_url"], model=CONFIG["model"],
                  request=request, next_call=lambda _: run.original)
        kw.update(changes)
        return g.model_gate(**kw)
    run.original = original

    return SimpleNamespace(module=module, g=g, row=row, original=original, request=request,
                           run=run, captured=captured, refused=refused, session=session,
                           codex_response=codex_response, codex_message=codex_message,
                           codex_events=codex_events)


def test_presentation_is_one_fresh_tool_free_call_and_preserves_original(setup):
    s = setup
    result = s.run()
    assert result is not s.refused and result.id == s.original.id
    assert result.choices[0].message.content == (
        "<voice>Short.</voice><detail>Full source result.\n\nSources:\n"
        "https://example.invalid/source</detail>"
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
    payload = json.loads(body["messages"][-1]["content"])
    assert payload["answer"] == s.original.choices[0].message.content
    assert payload["sources"] == [{"id": 0, "url": "https://example.invalid/source"}]
    assert payload["objective"] == "synthetic lookup"
    assert payload["evidence"] == [{"source_id": 0, "kind": "search", "text": "",
                                    "original_bytes": 0, "truncated": False}]
    assert payload["missing_source_ids"] == [0]
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["source_ids"]["minItems"] == 0
    assert s.captured["sdk"][0]["max_retries"] == 0
    assert s.captured["http"][0]["follow_redirects"] is False
    assert s.captured["closed"] == 2 and s.row["presentation_active"] is None
    assert s.row["presentation_evidence"]["usage"]["total_tokens"] == 7
    assert s.row["presentation_evidence"]["wire"]["tools_absent"] is True
    assert s.row["presentation_evidence"]["wire"]["reasoning_effort_none"] is True
    assert s.row["presentation_evidence"]["wire_requests"] == 1
    assert s.run() is s.refused and len(s.captured["requests"]) == 1


@pytest.mark.parametrize("mutate", [
    lambda body: body["input"][0].update(content=body["input"][0]["content"] + " "),
    lambda body: body.update(tools=[]),
    lambda body: body["text"]["format"].update(strict=False),
    lambda body: body.update(store=True),
    lambda body: body.update(stream=False),
    lambda body: body["reasoning"].update(effort="high"),
    lambda body: body.update(model="other-model"),
    lambda body: body.update(instructions="changed"),
    lambda body: body.update(extra="unexpected"),
], ids=["input", "tools", "schema", "store", "stream", "reasoning", "model",
        "instructions", "extra-field"])
def test_codex_wire_contract_mutations_refuse_before_accepting_response(setup, mutate):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    s.run.original = s.codex_response([s.codex_message("Grounded source answer.")])
    s.captured["wire_mutation"] = mutate
    result = s.run(api_mode="codex_responses")
    assert result.status == "completed"
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert s.row["presentation_evidence"]["category"] == "refused"
    assert s.row["presentation_evidence"]["wire_requests"] == 0
    assert s.row["presentation_evidence"]["local_http_settled"] is True


def test_presentation_route_mode_is_explicit_and_bound(setup):
    s = setup
    formatter = s.module.Presentation(CODEX_CONFIG)
    formatter.validate_route({
        "provider": CODEX_CONFIG["provider"],
        "base_url": CODEX_CONFIG["base_url"],
        "model": CODEX_CONFIG["model"],
        "api_mode": "codex_responses",
        "request": {"model": CODEX_CONFIG["model"]},
    })
    with pytest.raises(ValueError, match="presentation-route"):
        formatter.validate_route({
            "provider": CODEX_CONFIG["provider"],
            "base_url": CODEX_CONFIG["base_url"],
            "model": CODEX_CONFIG["model"],
            "api_mode": "chat_completions",
            "request": {"model": CODEX_CONFIG["model"]},
        })
    with pytest.raises(ValueError, match="presentation-config"):
        s.module.Presentation(CONFIG | {"api_mode": "codex_app_server"})


def test_codex_responses_function_call_uses_call_id_for_ticket_identity(setup):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    source_url = "https://example.invalid/source"
    s.row.update(phase="search", reserved=1, settled=1, urls={source_url})
    args = {"urls": [source_url]}
    raw = SimpleNamespace(
        id="resp_native",
        status="completed",
        output=[SimpleNamespace(
            type="function_call", id="fc_item_id", call_id="call_ticket_id",
            name="web_extract", arguments=json.dumps(args), status="completed",
        )],
        usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
    )
    s.run.original = raw
    request = {
        "model": CODEX_CONFIG["model"],
        "tools": [{"type": "function", "name": "web_extract",
                   "parameters": {"type": "object"}}],
    }
    result = s.run(api_mode="codex_responses", request=request)
    assert result is raw
    assert s.row["plan"] == {"call_ticket_id": ("web_extract", s.module.frozen(args))}
    result = s.g.tool_gate(
        tool_name="web_extract", tool_call_id="call_ticket_id",
        api_request_id="task-request", args=args, next_call=lambda _: "admitted",
    )
    assert result == "admitted"
    assert s.row["spent"] == {"call_ticket_id"}


def test_codex_host_search_returns_native_responses_envelope(setup):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    s.row.update(phase="new", reserved=0, settled=0, urls=set())
    request = {"tools": [{
        "type": "function", "name": "web_search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }]}
    result = s.run(api_mode="codex_responses", request=request,
                   next_call=lambda _: pytest.fail("host retrieval must not call the model"))
    assert result.status == "completed" and result.id.startswith("host-response-")
    assert len(result.output) == 1 and result.output[0].type == "function_call"
    assert result.output[0].name == "web_search"
    assert result.output[0].call_id.startswith("host-search-")
    assert json.loads(result.output[0].arguments) == {"query": "synthetic lookup"}


def test_codex_mode_refusal_is_a_completed_native_message(setup):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    result = s.run(api_mode="codex_responses", request={"tools": []})
    assert result.status == "completed"
    assert result.output[0].status == "completed"
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert not s.captured["responses_requests"]


def test_codex_formatter_stream_uses_final_answer_and_preserves_raw_response(setup):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    original_final = s.codex_message(
        '{"voice":"Short.","detail":"Full source result.","source_ids":[0]}')
    reasoning = SimpleNamespace(type="reasoning", id="rs_keep", status="completed", summary=[])
    commentary = s.codex_message("Progress text must not be published.", phase="commentary",
                                 item_id="msg_commentary")
    raw = s.codex_response(
        [reasoning, commentary, original_final],
        usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
    )
    s.run.original = raw
    s.captured["stream_events"] = s.codex_events(
        s.codex_response([
            s.codex_message("{\"voice\":\"Short.\",\"detail\":\"Full source result.\","
                            "\"source_ids\":[0]}")
        ], usage=SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7)),
        deltas=['{"voice":', '"Short."}'],
    )
    result = s.run(api_mode="codex_responses")
    assert result is not s.refused and result.id == raw.id
    assert result.usage is raw.usage and result.output[0] is reasoning
    assert result.output[1] is commentary
    final_text = result.output[2].content[0].text
    assert final_text == (
        "<voice>Short.</voice><detail>Full source result.\n\nSources:\n"
        "https://example.invalid/source</detail>"
    )
    assert original_final.content[0].text.startswith('{"voice"')
    assert len(s.captured["responses_requests"]) == 1
    body = s.captured["responses_requests"][0]
    assert set(body) == {"model", "instructions", "input", "store", "stream", "text", "reasoning"}
    assert body["model"] == CODEX_CONFIG["model"]
    assert body["store"] is False and body["stream"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"] == {
        "type": "json_schema", "name": "voice_detail", "strict": True,
        "schema": s.module.PRESENTATION_SCHEMA,
    }
    assert "tools" not in body and "tool_choice" not in body
    assert s.captured["stream_closed"] == 1
    assert s.row["presentation_evidence"]["usage"] == {
        "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
    }
    assert s.row["presentation_evidence"]["wire"]["tools_absent"] is True
    assert s.row["presentation_evidence"]["wire"]["input_matches"] is True
    assert s.row["presentation_evidence"]["wire"]["instructions_match"] is True


def test_codex_stream_close_is_cancellation_boundary(setup):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    s.run.original = s.codex_response([s.codex_message("Grounded source answer.")])

    async def cancel_during_close():
        s.g.cancel_presentation(s.row)
        await asyncio.sleep(0)

    s.captured["stream_close_callback"] = cancel_during_close
    result = s.run(api_mode="codex_responses")
    assert result.status == "completed"
    assert s.captured["stream_closed"] == 1
    assert s.row["presentation_evidence"]["local_http_settled"] is True
    assert s.row["presentation_active"] is None


@pytest.mark.parametrize("events", [
    [],
    [SimpleNamespace(type="response.incomplete", response=SimpleNamespace(status="incomplete"))],
    [SimpleNamespace(type="response.unknown_future_event")],
    [SimpleNamespace(type="response.completed", response=None)],
    [SimpleNamespace(type="response.completed", response=SimpleNamespace(
        id="not-completed", status="incomplete", output=[], usage=None))],
    [SimpleNamespace(type="response.completed", response=SimpleNamespace(
        id="null-output", status="completed", output=None, usage=None))],
    [SimpleNamespace(type="response.completed", response=SimpleNamespace(
        id="refusal", status="completed", output=[SimpleNamespace(
            type="message", role="assistant", status="completed", phase="final_answer",
            content=[SimpleNamespace(type="refusal", refusal="denied")])], usage=None))],
], ids=["missing-terminal", "incomplete-event", "unknown-event", "terminal-null",
        "native-not-completed", "output-null", "refusal"])
def test_codex_formatter_fails_closed_on_noncompleted_or_unusable_terminal(setup, events):
    s = setup
    s.g.presentation = s.module.Presentation(CODEX_CONFIG)
    s.run.original = s.codex_response([
        s.codex_message("Grounded source answer.")
    ])
    s.captured["stream_events"] = events
    result = s.run(api_mode="codex_responses")
    assert result.status == "completed"
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert s.row["presentation_evidence"]["category"] == "refused"
    assert s.captured["stream_closed"] == 1


@pytest.mark.parametrize("invalid_commentary", [
    lambda s: [SimpleNamespace(type="refusal", refusal="do not log this"),
               SimpleNamespace(type="output_text", text="commentary")],
    lambda s: [SimpleNamespace(type="future_content_part", text="do not log this")],
], ids=["refusal-part", "unknown-content-part"])
def test_codex_commentary_content_is_validated_before_exclusion(setup, invalid_commentary):
    s = setup
    commentary = s.codex_message("untrusted", phase="commentary",
                                 item_id="msg_commentary_invalid")
    commentary.content = invalid_commentary(s)
    s.run.original = s.codex_response([
        commentary,
        s.codex_message(
            '{"voice":"Short.","detail":"Full source result.","source_ids":[0]}'),
    ])
    result = s.run(api_mode="codex_responses")
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert not s.captured["responses_requests"]


@pytest.mark.parametrize("response", [
    lambda s: s.codex_response([
        SimpleNamespace(type="reasoning", status="in_progress", id="rs_partial"),
        s.codex_message('{"voice":"Short.","detail":"Full source result.","source_ids":[0]}'),
    ]),
    lambda s: s.codex_response([
        SimpleNamespace(type="reasoning", status="incomplete", id="rs_partial"),
        s.codex_message('{"voice":"Short.","detail":"Full source result.","source_ids":[0]}'),
    ]),
], ids=["reasoning-in-progress", "reasoning-incomplete"])
def test_codex_refuses_noncompleted_reasoning_items(setup, response):
    s = setup
    s.run.original = response(s)
    result = s.run(api_mode="codex_responses")
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert not s.captured["responses_requests"]


def test_codex_commentary_with_unphased_final_message_is_ambiguous(setup):
    s = setup
    s.run.original = s.codex_response([
        s.codex_message("intermediate", phase="commentary"),
        s.codex_message('{"voice":"Short.","detail":"Full source result.","source_ids":[0]}',
                        phase=None),
    ])
    result = s.run(api_mode="codex_responses")
    assert result.output[0].content[0].text == s.module.REFUSAL
    assert not s.captured["responses_requests"]


def test_codex_diagnostic_whitelists_unrecognized_output_types(setup, capsys):
    s = setup
    sentinel = "PRIVATE_SENTINEL_" + "x" * 512
    raw = s.codex_response([SimpleNamespace(type=sentinel)])
    facts = s.module.response_facts(raw, "source", "codex_responses")
    assert facts["source_output_item_types"] == ["other"]
    s.run.original = raw
    s.run(api_mode="codex_responses")
    assert sentinel not in capsys.readouterr().out


@pytest.mark.parametrize("setup", [None], indirect=True)
def test_initial_lookup_is_a_host_request_without_provider_call(setup):
    s = setup
    objective = "  exact approved lookup\n"
    s.row["objective"] = objective
    calls = []
    request = {"tools": [{"type": "function", "function": {"name": "web_search"}}]}
    result = s.run(request=request, next_call=lambda _: calls.append("provider") or s.original)
    assert calls == []
    assert result.policy_origin == "host-initial-retrieval" and result.usage is None
    assert result.model == "host-policy/ordinary-lookup/v1"
    call = result.choices[0].message.tool_calls[0]
    assert call.id.startswith("host-search-") and call.function.name == "web_search"
    assert json.loads(call.function.arguments) == {"query": objective}
    assert s.row["phase"] == "search" and s.row["reserved"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", [None, 7, "", " ", "x" * 1025],
                         ids=["missing", "type", "empty", "blank", "bound"])
async def test_invalid_host_query_is_refused_before_native_admission(setup, objective):
    s = setup
    app = web.Application()
    s.g.wire(app, None)
    async def body():
        return {"input": objective}
    request = SimpleNamespace(method="POST", path="/v1/runs",
        headers={"X-Lookup-Policy": s.module.VERSION}, match_info={}, json=body)
    calls = []
    async def native(_):
        calls.append("admission")
        return web.json_response({"run_id": "new"}, status=202)
    assert (await app.middlewares[-1](request, native)).status == 400
    assert calls == []


@pytest.mark.asyncio
async def test_admission_freezes_exact_bounded_objective_before_native_handler(setup):
    s = setup
    objective = " " + "q" * 1022 + "\n"
    data = {"input": objective}
    app = web.Application()
    s.g.wire(app, None)
    async def body():
        return data
    request = SimpleNamespace(method="POST", path="/v1/runs",
        headers={"X-Lookup-Policy": s.module.VERSION}, match_info={}, json=body)
    async def native(_):
        data["input"] = "native handler changed its parsed object"
        return web.json_response({"run_id": "new"}, status=202)
    assert (await app.middlewares[-1](request, native)).status == 202
    assert s.g.budgets["new"]["objective"] == objective


@pytest.mark.parametrize("setup", [None], indirect=True)
@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "other"}}],
    [{"type": "other", "function": {"name": "web_search"}}],
    [{"type": "function", "function": {"name": "web_search"}}] * 2],
    ids=["missing", "empty", "different", "kind", "duplicate"])
def test_host_search_requires_one_advertised_search_before_any_provider(setup, tools):
    calls = []
    assert setup.run(request={"tools": tools},
        next_call=lambda _: calls.append("provider") or setup.original) is setup.refused
    assert calls == [] and setup.row["reserved"] == 0


@pytest.mark.parametrize("setup", [None], indirect=True)
@pytest.mark.parametrize("fault", ["cancelled", "deadline", "factory"])
def test_host_preparation_failure_never_falls_through_to_provider(setup, monkeypatch, fault):
    s = setup
    if fault == "cancelled":
        s.row["cancelled"] = True
    elif fault == "deadline":
        s.row["deadline"] = time.monotonic() - 1
    else:
        def failed(_):
            raise ValueError("synthetic construction failure")
        monkeypatch.setattr(s.module, "host_search_request", failed)
    calls = []
    assert s.run(request={"tools": [{"type": "function", "function": {"name": "web_search"}}]},
        next_call=lambda _: calls.append("provider") or s.original) is s.refused
    assert calls == []


@pytest.mark.parametrize("setup", [None], indirect=True)
@pytest.mark.parametrize("kind", ["ordinary", "budget-only"])
def test_initial_host_request_does_not_change_unselected_or_budget_only_runs(setup, kind):
    s = setup
    if kind == "ordinary":
        s.session["run"] = "ordinary"
    else:
        s.g.presentation = None
    calls = []
    assert s.run(next_call=lambda _: calls.append("provider") or s.original) is s.original
    assert calls == ["provider"]


@pytest.mark.parametrize("setup", [{"success": False, "data": {"web": []}}], indirect=True)
def test_failed_initial_search_never_retries_or_calls_provider(setup):
    calls = []
    result = setup.run(next_call=lambda _: calls.append("provider") or setup.original)
    assert result is setup.refused
    assert calls == [] and not setup.captured["requests"]


def test_successful_initial_search_returns_to_one_native_provider_call(setup):
    calls = []
    request = {"model": "test-model", "tools": [
        {"type": "function", "function": {"name": "web_search"}}]}
    result = setup.run(request=request,
        next_call=lambda _: calls.append("provider") or setup.original)
    assert result is not setup.refused and calls == ["provider"]


@pytest.mark.parametrize("setup", [
    None,
    {"success": False, "data": {"web": [{"url": "https://example.invalid/source"}]}},
    {"success": True, "error": "failed", "data": {"web": [{"url": "https://example.invalid/source"}]}},
    {"success": True, "data": {"web": []}},
    {"success": True, "data": {"web": [{"url": ""}]}},
    {"success": True, "data": {"web": [{"url": "  "}]}},
], indirect=True, ids=["no-retrieval", "failed", "error", "empty", "blank", "whitespace"])
def test_presentation_requires_successful_same_run_source(setup):
    assert setup.run() is setup.refused
    assert not setup.row["urls"] and not setup.captured["requests"]
    if setup.row["phase"] == "new":
        assert setup.row["closed"]  # Missing advertised search fails before any provider call.
    else:
        assert setup.row["presentation_evidence"]["stage"] == "retrieval"


def test_successful_admitted_search_without_extraction_reaches_presentation(setup):
    assert setup.row["urls"] == {"https://example.invalid/source"}
    assert setup.row["phase"] == "search" and setup.row["settled"] == 1
    assert setup.row["spent"] == setup.row["admitted"] and len(setup.row["spent"]) == 1
    assert setup.run() is not setup.refused and len(setup.captured["requests"]) == 1


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/b", "title": "Published date", "description":
     "Published October 21. Deployment in early 2027, once utilities are ready."},
    {"url": "https://example.invalid/a", "description": "Bolt with Lucid."}]}}], indirect=True)
def test_search_passages_are_immutable_and_bound_to_sorted_source_ids(setup):
    s = setup
    snapshots = s.row["passages"]
    with pytest.raises(TypeError):
        snapshots["https://example.invalid/a"] = ()
    assert s.run() is not s.refused
    payload = json.loads(s.captured["requests"][0]["messages"][-1]["content"])
    assert payload["objective"] == s.row["objective"]
    assert payload["missing_source_ids"] == []
    assert payload["evidence"][0] == {
        "source_id": 0, "kind": "search", "text": "Bolt with Lucid.",
        "original_bytes": len(b"Bolt with Lucid."), "truncated": False}
    assert payload["evidence"][1]["source_id"] == 1
    assert "Published October 21" in payload["evidence"][1]["text"]
    assert "once utilities are ready" in payload["evidence"][1]["text"]
    assert s.row["passages"] is snapshots


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/a", "description": "dated context. " + "界" * 20000},
    {"url": "https://example.invalid/b", "description": "qualified context. " + "b" * 20000}]}}],
    indirect=True)
def test_evidence_pack_bounds_serialized_bytes_and_exposes_exact_prefix_omissions(setup):
    s = setup
    assert s.run() is not s.refused
    packed = s.captured["requests"][0]["messages"][-1]["content"]
    assert len(packed.encode("utf-8")) <= 16384
    payload = json.loads(packed)
    assert len(payload["evidence"]) == 2
    for item in payload["evidence"]:
        original = s.row["passages"][payload["sources"][item["source_id"]]["url"]][0][1]
        assert original.startswith(item["text"]) and item["text"]
        assert item["truncated"] is True
        assert item["original_bytes"] == len(original.encode("utf-8"))


def test_passage_map_replacement_during_presentation_is_refused(setup):
    s = setup
    async def replace():
        s.row["passages"] = MappingProxyType(dict(s.row["passages"]))
    s.captured["callback"] = replace
    assert s.run() is s.refused
    assert s.row["presentation_evidence"]["stage"] == "catalog-publication"


def test_passage_catalog_mismatch_is_refused_before_formatter(setup):
    setup.row["passages"] = MappingProxyType(dict(setup.row["passages"]) | {
        "https://example.invalid/other": (("search", "Stale source text"),)})
    assert setup.run() is setup.refused and not setup.captured["requests"]


def test_mutable_passage_snapshot_is_refused_before_formatter(setup):
    setup.row["passages"] = dict(setup.row["passages"])
    assert setup.run() is setup.refused and not setup.captured["requests"]


@pytest.mark.parametrize("setup", [None], indirect=True)
def test_invalid_passage_shape_refuses_without_retaining_private_values(setup, capsys):
    retrieve(setup.g, {"success": True, "data": {"web": [
        {"url": "https://example.invalid/a", "description": 7}]}})
    assert setup.run() is setup.refused and not setup.captured["requests"]
    assert "[lookup-evidence]" in capsys.readouterr().out


async def extract_passage(s, payload, before_result=None, urls=None):
    args = {"urls": ["https://example.invalid/source"] if urls is None else urls}
    s.row["request"] = "extract-request"
    s.row["spent"].add("extract-call")
    token = s.g.ticket.set(("selected", "extract-request", "extract-call", "web_extract",
                            s.module.frozen(args)))
    async def original(_):
        if before_result:
            before_result()
        return payload
    try:
        return await s.g.wrap_extract(original)(args)
    finally:
        s.g.ticket.reset(token)


@pytest.mark.asyncio
async def test_admitted_extraction_retains_same_url_content_as_immutable_snapshot(setup):
    s = setup
    before = s.row["passages"]
    result = json.dumps({"results": [{"url": "https://example.invalid/source",
        "title": "Observation", "content": "Published October 21; available early 2027."}]})
    assert await extract_passage(s, result) == result
    assert s.row["passages"] is not before
    assert before["https://example.invalid/source"] == (("search", ""),)
    assert await asyncio.to_thread(s.run) is not s.refused
    payload = json.loads(s.captured["requests"][0]["messages"][-1]["content"])
    assert payload["missing_source_ids"] == []
    assert payload["evidence"][1]["kind"] == "extract"
    assert payload["evidence"][1]["text"] == (
        "Observation\nPublished October 21; available early 2027.")


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", ["request", "catalog"])
async def test_extraction_cannot_attach_another_admitted_catalog_sources_text(
    setup, capsys, binding,
):
    s = setup
    # Catalog membership alone is insufficient: only this extraction's request may bind text.
    if binding == "request":
        s.row["urls"].add("https://example.invalid/other")
    before = s.row["passages"]
    result = json.dumps({"results": [{"url": "https://example.invalid/other",
                                     "content": "SYNTHETIC_PRIVATE_TEXT"}]})
    urls = ["https://example.invalid/other"] if binding == "catalog" else None
    assert await extract_passage(s, result, urls=urls) == result
    assert not s.row["closed"]
    assert set(s.row["passages"]) == set(before)
    assert all("SYNTHETIC_PRIVATE_TEXT" not in text
               for values in s.row["passages"].values() for _, text in values)
    if binding == "request":
        assert ("extract-unavailable", "") in s.row["passages"]["https://example.invalid/source"]
    output = capsys.readouterr().out
    assert '"category":"unbound"' in output and "SYNTHETIC_PRIVATE_TEXT" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/caf\u00e9", "description": "Dated search context."}]}}],
    indirect=True)
async def test_canonicalized_extraction_echo_is_an_explicit_gap_without_refusing_run(setup):
    s = setup
    requested = "https://example.invalid/caf\u00e9"
    echoed = "https://example.invalid/caf%C3%A9"
    result = json.dumps({"results": [{"url": echoed, "content": "Unbound extraction text"}]})
    assert await extract_passage(s, result, urls=[requested]) == result
    assert not s.row["closed"] and s.row["urls"] == {requested}
    assert s.row["passages"][requested] == (
        ("search", "Dated search context."), ("extract-unavailable", ""))
    assert await asyncio.to_thread(s.run) is not s.refused
    payload = json.loads(s.captured["requests"][0]["messages"][-1]["content"])
    assert payload["evidence"][1]["kind"] == "extract-unavailable"
    assert payload["evidence"][1]["text"] == ""
    assert "Unbound extraction text" not in json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["closed", "replaced"])
async def test_late_extraction_never_updates_retained_passages(setup, kind):
    s = setup
    before = s.row["passages"]
    def late():
        if kind == "closed":
            s.row["closed"] = s.row["cancelled"] = True
        else:
            replacement = dict(s.row)
            s.g.budgets["selected"] = replacement
            s.g.members = MappingProxyType({
                "selected": (s.module.VERSION, "identity", replacement)})
            s.g._membership_authority = s.g.members
    result = json.dumps({"results": [{"url": "https://example.invalid/source", "content": "Late"}]})
    assert await extract_passage(s, result, late) == s.module.REFUSAL
    assert s.row["passages"] is before


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["x" * 262145, "界" * 100000], ids=["characters", "utf8"])
async def test_extraction_snapshot_ingress_is_bounded_before_retention(
    setup, capsys, monkeypatch, text,
):
    before = setup.row["passages"]
    result = json.dumps({"results": [{"url": "https://example.invalid/source", "content": text}]},
                        ensure_ascii=False)
    parsed = []
    original_loads = setup.module.json.loads
    def loads(value, *args, **kwargs):
        parsed.append(True)
        return original_loads(value, *args, **kwargs)
    monkeypatch.setattr(setup.module.json, "loads", loads)
    assert await extract_passage(setup, result) == setup.module.REFUSAL
    assert setup.row["passages"] is before
    assert parsed == []
    assert '"category":"bound"' in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/source", "description": "s" * 180000}]}}], indirect=True)
async def test_combined_snapshot_bytes_are_bounded_atomically(setup, capsys):
    before = setup.row["passages"]
    result = json.dumps({"results": [{"url": "https://example.invalid/source",
                                     "content": "e" * 100000}]})
    assert await extract_passage(setup, result) == setup.module.REFUSAL
    assert setup.row["passages"] is before
    assert '"category":"bound"' in capsys.readouterr().out


def test_attribution_renders_selected_admitted_url_and_preserves_answer(setup):
    s = setup
    s.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": [0]}))
    result = s.run()
    assert result is not s.refused
    assert result.choices[0].message.content == (
        "<voice>Short.</voice><detail>Supported answer.\n\nSources:\n"
        "https://example.invalid/source</detail>"
    )
    assert s.original.choices[0].message.content == "Full source result."


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/z"}, {"url": "https://example.invalid/a"}]}}], indirect=True)
def test_attribution_selects_subset_from_whole_frozen_catalog_without_url_rewrite(setup):
    s = setup
    s.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": [1]}))
    result = s.run()
    assert result is not s.refused
    content = result.choices[0].message.content
    assert content.endswith("\n\nSources:\nhttps://example.invalid/z</detail>")
    assert "https://example.invalid/a" not in content
    payload = json.loads(s.captured["requests"][0]["messages"][-1]["content"])
    assert payload["sources"] == [
        {"id": 0, "url": "https://example.invalid/a"},
        {"id": 1, "url": "https://example.invalid/z"}]


@pytest.mark.parametrize("ids", [[], [False], [-1], [1], [0, 0], "0", None],
                         ids=["empty", "boolean", "negative", "out-of-range", "duplicate",
                              "string", "null"])
def test_attribution_refuses_invalid_selected_ids(setup, ids):
    setup.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": ids}))
    assert setup.run() is setup.refused
    expected = "source-selection" if type(ids) is not list or not ids else (
        "source-id-type" if any(type(index) is not int for index in ids) else
        "source-id-range" if ids == [-1] or ids == [1] else "source-id-unique")
    assert setup.row["presentation_evidence"]["stage"] == expected


@pytest.mark.parametrize("field", ["voice", "detail"])
@pytest.mark.parametrize("marker", ["HtTp://", "HTTPS://", "WwW."], ids=["http", "https", "www"])
def test_attribution_refuses_model_inline_web_links_without_repair(setup, capsys, field, marker):
    value = {"voice": "Short.", "detail": "Supported answer.", "source_ids": [0]}
    value[field] = marker + "invented.example.invalid/PRIVATE_UNADMITTED"
    setup.captured["response"] = completion(json.dumps(value))
    assert setup.run() is setup.refused
    assert setup.row["presentation_evidence"]["stage"] == "inline-links"
    assert "PRIVATE_UNADMITTED" not in capsys.readouterr().out


def test_attribution_does_not_join_fields_into_an_artificial_web_link(setup):
    setup.captured["response"] = completion(json.dumps({
        "voice": "htt", "detail": "p://not-a-web-link", "source_ids": [0]}))
    assert setup.run() is not setup.refused


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/" + str(i) + "x" * 6000} for i in range(3)]}}], indirect=True)
def test_attribution_refuses_whole_catalog_over_input_bound_without_call(setup):
    assert setup.run() is setup.refused and not setup.captured["requests"]
    assert setup.row["presentation_evidence"]["stage"] == "catalog"


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/" + "x" * 1790}]}}], indirect=True)
def test_attribution_refuses_rendered_detail_over_1800_without_clipping(setup):
    setup.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": [0]}))
    assert setup.run() is setup.refused
    assert setup.row["presentation_evidence"]["stage"] == "terminal-contract"


@pytest.mark.parametrize("setup", [{"success": True, "data": {"web": [
    {"url": "https://example.invalid/</detail>"}]}}], indirect=True)
def test_attribution_rendered_urls_still_pass_reserved_markup_validation(setup):
    setup.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": [0]}))
    assert setup.run() is setup.refused
    assert setup.row["presentation_evidence"]["stage"] == "terminal-contract"


@pytest.mark.parametrize("change", ["add", "remove"])
def test_attribution_refuses_catalog_drift_before_publication(setup, change):
    s = setup
    s.captured["response"] = completion(json.dumps({
        "voice": "Short.", "detail": "Supported answer.", "source_ids": [0]}))

    async def drift():
        if change == "add":
            s.row["urls"].add("https://example.invalid/unrelated")
        else:
            s.row["urls"].clear()

    s.captured["callback"] = drift
    assert s.run() is s.refused
    assert s.row["presentation_evidence"]["stage"] == "catalog-publication"


@pytest.mark.parametrize("setup", [None], indirect=True)
def test_other_run_retrieval_cannot_satisfy_presentation(setup):
    s = setup
    other = dict(s.row, urls=set())
    s.g.budgets["other"] = other
    s.g.members = MappingProxyType(
        dict(s.g.members) | {"other": (s.module.VERSION, "other", other)}
    )
    s.g._membership_authority = s.g.members
    s.session["run"] = "other"
    retrieve(s.g, {"success": True, "data": {"web": [{"url": "https://example.invalid/other"}]}})
    s.session["run"] = "selected"
    assert other["urls"] and not s.row["urls"]
    assert s.run() is s.refused and not s.captured["requests"]


@pytest.mark.parametrize("setup", [None], indirect=True)
@pytest.mark.parametrize("stale", ["cancelled", "replaced"])
def test_stale_search_result_never_adds_source_evidence(setup, stale):
    s = setup

    def before_result():
        if stale == "cancelled":
            s.row["closed"] = s.row["cancelled"] = True
        else:
            s.g.budgets["selected"] = {}

    retrieve(s.g, {"success": True, "data": {"web": [{"url": "https://example.invalid/stale"}]}},
             before_result)
    assert not s.row["urls"]
    assert s.run() is s.refused and not s.captured["requests"]


@pytest.mark.parametrize("setup", [None], indirect=True)
def test_budget_only_completion_does_not_claim_or_require_retrieval(setup):
    setup.g.presentation = None
    assert setup.run() is setup.original and not setup.captured["requests"]


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
    '{"voice":"v","detail":"d","source_ids":[0],"extra":"x"}',
    '{"voice":"v","voice":"v","detail":"d","source_ids":[0]}',
    '{"voice":5,"detail":"d","source_ids":[0]}',
    '{"voice":"' + "v" * 221 + '","detail":"d","source_ids":[0]}',
    '{"voice":"v","detail":"' + "d" * 1801 + '","source_ids":[0]}',
    '{"voice":"v</voice>","detail":"d","source_ids":[0]}',
    '{"voice":"v","detail":"","source_ids":[0]}',
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
    s.captured["response"] = completion(
        '{"voice":"v","detail":"d","source_ids":[0]}', reason="length")
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
    response = completion('{"voice":"v","detail":"d","source_ids":[0]}')
    response.choices[0].message.refusal = "synthetic-refusal"
    setup.captured["response"] = response
    assert setup.run() is setup.refused


@pytest.mark.parametrize("case,count,kind,length,finish,tools,refusal,parsed", [
    ("zero-choices", 0, "none", 0, "missing", False, False, False),
    ("extra-choices", 2, "string", 43, "stop", False, False, False),
    ("none-content", 1, "none", 0, "stop", False, False, False),
    ("other-content", 1, "other", 0, "stop", False, False, False),
    ("oversize", 1, "string", 8193, "stop", False, False, False),
    ("length-stop", 1, "string", 43, "length", False, False, False),
    ("unknown-finish", 1, "string", 43, "other", False, False, False),
    ("missing-finish", 1, "string", 43, "missing", False, False, False),
    ("tool-call", 1, "string", 43, "stop", True, False, False),
    ("refusal", 1, "string", 43, "stop", False, True, False),
    ("parsed-only", 1, "none", 0, "stop", False, False, True),
])
def test_formatter_envelope_refusal_retains_shape_and_usage_without_content(
    setup, capsys, case, count, kind, length, finish, tools, refusal, parsed,
):
    s = setup
    response = completion('{"voice":"v","detail":"d","source_ids":[0]}')
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
    s.captured["response"] = completion(
        '{"voice":"v","detail":"d","source_ids":[0]}', calls=["not-allowed"])
    assert s.run() is s.refused


def test_presentation_refuses_extra_choices(setup):
    s = setup
    response = completion('{"voice":"v","detail":"d","source_ids":[0]}')
    response.choices *= 2
    s.captured["response"] = response
    assert s.run() is s.refused


@pytest.mark.parametrize("voice", ["line\nbreak", "control\x01", "**bold**", "# headline"],
                         ids=["newline", "control", "emphasis", "markup"])
def test_presentation_uses_existing_terminal_contract(setup, voice):
    setup.captured["response"] = completion(json.dumps({
        "voice": voice, "detail": "detail", "source_ids": [0]}))
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
        "local_http_settled", "input_sha256", "catalog_sha256", "catalog_count",
        "selected_source_count", "rendered_detail_length"}


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
