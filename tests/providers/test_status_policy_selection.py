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
)
from hermes_realtime.conversation.output_style import communication_policy
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.work_tools import ConversationWorkControlSurface
from scripts.evaluate_natural_work_scenario import ReadinessDouble
from tests.support.ollama_metadata import supported_ollama_metadata

AVAILABLE = "For current task status, call the host task_status route."
UNAVAILABLE = "Current task status cannot be checked here; no host status route is supplied."


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["codex", "ollama"])
@pytest.mark.parametrize("mode", ["bound", "unbound", "private", "nonuser", "legacy",
                                  "preflight", "lookup"])
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
                "Read D:\\private\\secret.txt" if mode == "private" else "What is running now?",
            ),))
            [part async for part in inference.stream(snapshot, turn_id="turn_policy")]
        if provider == "codex":
            thread = next(m for m in transport.sent if m.get("method") == "thread/start")
            advertised = {tool["name"] for tool in thread["params"]["dynamicTools"]}
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
