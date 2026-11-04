"""Model-free API result delivery through authoritative task and host projections."""

from types import SimpleNamespace

import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskController,
    TaskTerminalOutcome,
)
from hermes_realtime.conversation.result_projection import _browser_safe_task_result
from hermes_realtime.conversation.updates import UpdatePolicyInput
from hermes_realtime.host_launcher import ProactiveCompletionUpdatePolicy
from hermes_realtime.result_limits import UNUSABLE_RESULT_NOTICE
from tests.integration.test_hermes_api_result import _RUN_ID, _session


@pytest.mark.asyncio
@pytest.mark.parametrize("character", ["x", "\U0001f30a"])
async def test_maximum_api_result_reaches_browser_without_losing_trailing_source(character):
    api, authority = _session()
    tail = "\nObserved at 12:53 local time. Source: https://example.invalid/report"
    detail = character * (1800 - len(tail)) + tail
    output = "<voice>" + "R" * 220 + "</voice><detail>" + detail + "</detail>"
    effects = SimpleNamespace(dispatches=0, cancellations=0)

    class Session:
        async def dispatch(self, request):
            effects.dispatches += 1
            assert request.task_id == authority.task_id
            return api._dispatch_ack(
                request, accepted=True, run_id=authority.protocol_run_id,
            )

        async def next_update(self):
            return await api.next_update()

        async def cancel(self, _request):
            effects.cancellations += 1
            raise AssertionError("completed result must not cause cancellation")

        async def close(self):
            await api.close()

    context = ConversationContextStore()
    controller = ConversationTaskController(
        context=context, session=Session(), session_id="session_api_1",
        id_factory=iter(("release_check", "dispatch_contract")).__next__,
    )
    controller.start()
    try:
        admission = await controller.dispatch(
            objective="Inspect the report.", utterance_id="utterance_1",
        )
        assert admission.accepted
        await api._consume_event(authority, {
            "event": "run.completed", "run_id": _RUN_ID, "output": output,
        })
        completion = await controller.next_completion()
        assert completion.summary == output and completion.status == "completed"
        assert context.snapshot().active_tasks == ()
        projection = BrowserEventProjection()
        await ProactiveCompletionUpdatePolicy(projection=projection).decide(UpdatePolicyInput(
            sequence=1, completion=completion, context=context.snapshot(),
        ))
        assert projection.events_after(0)[-1].data["text"] == detail
        assert effects.dispatches == 1 and effects.cancellations == 0
        assert authority.terminal and api._record_admitted == []
        assert len(context.snapshot().recent_task_results[0].text) <= 1024
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_full_plain_result_preserves_4096_scalars_in_browser():
    tail = " Observed at 12:53. https://example.invalid/source"
    detail = "\U0001f30a" * (4096 - len(tail)) + tail
    projection = BrowserEventProjection()
    completion = TaskTerminalOutcome(task_id="task_full_result", status="completed", summary=detail)
    await ProactiveCompletionUpdatePolicy(projection=projection).decide(UpdatePolicyInput(
        sequence=1, completion=completion, context=ConversationContextStore().snapshot(),
    ))
    assert projection.events_after(0)[-1].data["text"] == detail


@pytest.mark.parametrize("text", ["x" * 4097, "\ud800", "\udfff"])
def test_invalid_completed_result_is_rejected_before_task_settlement(text):
    with pytest.raises(ValueError):
        TaskTerminalOutcome(task_id="task_invalid_result", status="completed", summary=text)


@pytest.mark.parametrize("text,category", [("run_abcdefgh " * 315, "bound"), ("\ud800", "unicode")])
def test_display_projection_reports_unusable_content_without_cutting_it(text, category, capsys):
    import json

    assert _browser_safe_task_result(text, completed=True) == UNUSABLE_RESULT_NOTICE
    line = capsys.readouterr().out
    assert json.loads(line.removeprefix("[task-result-display] ")) == {
        "version": 1, "category": category,
    }
    assert "run_abcdefgh" not in line
