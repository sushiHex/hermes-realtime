"""Status instructions follow actual advertised capability, never a conditional denial."""

import json

import pytest
from test_codex_app_server import FakeCodexTransport
from test_codex_status_after_lookup import Controller, Lookup
from test_ollama_work_tools import Response, decision_frame

from hermes_realtime import host_launcher as host
from hermes_realtime.conversation.context import (
    ConversationContextSnapshot,
    ConversationContextStore,
    ConversationMessage,
    RecentTaskResult,
)
from hermes_realtime.conversation.output_style import communication_policy
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from scripts.evaluate_natural_work_scenario import ReadinessDouble
from tests.support.ollama_metadata import supported_ollama_metadata

AVAILABLE = "Use task_status only for pending, active, or cancellation state."
UNAVAILABLE = "Current task status cannot be checked here; no host status route is supplied."


@pytest.fixture(autouse=True)
def synthetic_codex_executable(monkeypatch):
    # Keep executable discovery hermetic while retaining the real host factory.
    monkeypatch.setattr(
        "hermes_realtime.providers.codex_app_server._resolve_codex_executable",
        lambda _explicit: "synthetic-codex",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["codex", "ollama"])
@pytest.mark.parametrize("mode", ["bound", "unbound", "private", "nonuser", "legacy",
                                  "preflight", "lookup", "completed"])
async def test_sent_status_policy_matches_actual_advertisement(provider, mode):
    # Mutations: render the opposite capability, or leave both branches in the policy.
    sent = []

    def opened(req, timeout):
        body = json.loads(req.data)
        sent.append(body)
        frame = decision_frame({"kind": "reply", "text": "Four."}) if "format" in body else {
            "message": {"content": "Four."}, "done": True, "done_reason": "stop",
        }
        return Response([frame])

    transport = FakeCodexTransport()
    inference = host._build_streaming_inference(
        inference_provider=provider, ollama_base_url="http://127.0.0.1:11434", ollama_model="test",
        codex_model="gpt-5.6-terra", codex_effort="low", codex_executable=None,
        current_fact_lookup=Lookup() if provider == "codex" and mode == "lookup" else None,
    )
    if provider == "codex":
        inference._transport_factory = lambda: transport
    else:
        inference._open_request = opened
        inference._model_client = supported_ollama_metadata()
    surface = ConversationWorkControlSurface(
        controller=Controller(), context=ConversationContextStore(), observer=lambda *args: None,
        reserve_observer_capacity=lambda: None,
    )
    try:
        if mode == "preflight":
            await host._run_full_host_preflight(
                task_session=ReadinessDouble(), inference=inference, synthesizer=ReadinessDouble(),
                surface=surface, natural_work_tools=True,
            )
        else:
            host._bind_natural_work_tools(
                inference=inference, surface=surface, enabled=mode != "unbound",
            )
            cls = ConversationContextSnapshot if mode == "legacy" else ConversationInferenceRequest
            snapshot = cls(revision=1, active_tasks=(), messages=(ConversationMessage(
                "assistant" if mode == "nonuser" else "user",
                "Read D:\\private\\secret.txt" if mode == "private" else
                "What were the findings?" if mode == "completed" else "What is running now?",
            ),), recent_task_results=(RecentTaskResult(
                "task_weather", "weather", "completed", "Sunny; observation time unavailable.",
            ),) if mode == "completed" else ())
            [part async for part in inference.stream(snapshot, turn_id="turn_policy")]
        if provider == "codex":
            thread = next(m for m in transport.sent if m.get("method") == "thread/start")
            dynamic_tools = thread["params"]["dynamicTools"]
            assert type(dynamic_tools) is list
            if dynamic_tools:
                assert len(dynamic_tools) == 1
                namespace = dynamic_tools[0]
                assert type(namespace) is dict and namespace.get("type") == "namespace"
                assert namespace.get("name") == "hermes"
                functions = namespace.get("tools")
                assert type(functions) is list and len(functions) <= 16
                assert all(
                    type(tool) is dict and tool.get("type") == "function"
                    for tool in functions
                )
                advertised = {tool["name"] for tool in functions}
            else:
                advertised = set()
            policy = thread["params"]["baseInstructions"]
        else:
            schema = sent[0].get("format", {})
            advertised = {row["properties"]["kind"]["const"] for row in schema.get("oneOf", [])}
            policy = sent[0]["messages"][0]["content"]
        selected, absent = (
            (AVAILABLE, UNAVAILABLE) if "task_status" in advertised else (UNAVAILABLE, AVAILABLE)
        )
        assert selected in policy and absent not in policy
        assert "when supplied; otherwise" not in policy
        if mode == "completed":
            # Sent-prompt regression only: this does not prove model route selection.
            assert "Use task_status only for pending, active, or cancellation state" in policy
            assert (
                "Answer completed findings from available results; preserve missing details."
            ) in policy
            assert "Preserve material qualifications and uncertainty." in policy
            outbound = json.dumps(transport.sent if provider == "codex" else sent)
            assert "Sunny; observation time unavailable." in outbound
    finally:
        await inference.close()
        await surface.close()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_status_policy_rejects_non_boolean_capability(value):
    # Mutation: accept a truthy or nullable capability instead of an exact boolean.
    with pytest.raises(TypeError, match="task_status_available"):
        communication_policy(task_status_available=value)


def test_invalid_status_capability_records_bounded_refusal(capsys):
    # Mutation: the exact-boolean refusal drops its bounded evidence marker.
    with pytest.raises(TypeError):
        communication_policy(task_status_available=None)
    assert capsys.readouterr().out == (
        '[status-policy-refusal] {"count":1,"category":"invalid_availability"}\n'
    )
