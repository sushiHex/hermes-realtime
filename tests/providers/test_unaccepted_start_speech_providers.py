"""Both adapters reach the shared loop's fixed host sentence for an unaccepted start."""

import pytest
from test_codex_app_server import DynamicStartCodexTransport, FakeWorkToolHandler
from test_ollama_work_tools import adapter, start_frame

from hermes_realtime.conversation import TaskDispatchOutcome
from hermes_realtime.conversation.context import ConversationContextStore
from hermes_realtime.conversation.foreground import ForegroundTurnCoordinator
from hermes_realtime.conversation.streaming import StreamingSpeechLoop
from hermes_realtime.conversation.task_proposals import ConversationTaskProposals
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript
from tests.conversation.test_streaming import RecordingPlayback, SegmentSynthesizer
from tests.conversation.test_work_tools import _surface

SENTINEL = "SENTINEL reason from Hermes 41be"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "codex"])
@pytest.mark.parametrize("outcome", ["rejected", "unavailable"])
async def test_adapter_start_selection_gets_one_host_sentence(provider, outcome):
    # Mutation: gate the sentence on the provider, or leave the Ollama turn silent.
    if provider == "ollama":
        inference, model_handler, _sent, _ = adapter([start_frame()])
    else:
        transport = DynamicStartCodexTransport(
            arguments={"objective": "Inspect release"}, assistant_delta="I started it.",
        )
        model_handler = FakeWorkToolHandler()
        inference = CodexAppServerStreamingInference(
            model="test", effort="low", transport_factory=lambda: transport,
        )
        inference.bind_work_tools(model_handler)
    context = ConversationContextStore()
    surface, controller, _ = _surface(context=context, accepted=False)
    controller.dispatch_outcome = TaskDispatchOutcome(
        task_id="task_release_check", accepted=False, reason=SENTINEL,
    )
    if outcome == "unavailable":
        await surface.close()
    proposals = ConversationTaskProposals(
        surface=surface, observer=lambda *_: None, start_policy="automatic",
    )
    proposals.bind("participant", 1)
    synth = SegmentSynthesizer()
    loop = StreamingSpeechLoop(
        context=context, foreground=ForegroundTurnCoordinator(), inference=inference,
        synthesizer=synth, playback=RecordingPlayback(), ledger=DeliveredSpeechLedger(),
        task_proposals=proposals,
    )
    expected = (
        "No new background task is running for that request." if outcome == "rejected"
        else "Background task control is unavailable, so that task was not started."
    )
    try:
        await loop.respond("turn_auto", Transcript(text="Inspect release", final=True))
        assert synth.texts == [expected]
        heard = context.snapshot().messages[-1]
        assert (heard.role, heard.source, heard.text) == (
            "assistant", "host_task_status", expected,
        )
        assert len(controller.dispatches) == (1 if outcome == "rejected" else 0)
        assert not any(SENTINEL in m.text for m in context.snapshot().messages)
        if provider == "ollama":
            assert model_handler.calls == []
        else:
            assert model_handler.starts == []
    finally:
        await loop.close()
        await inference.close()
        await surface.close()
