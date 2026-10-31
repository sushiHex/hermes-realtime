"""Pinned retry notifications through the actual adapter; no process or model calls."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping

import pytest
from test_codex_app_server import EofOnCloseCodexTransport, FakeWorkToolHandler, _snapshot

from hermes_realtime.providers import codex_app_server as provider


def advisory(**changes: object) -> dict[str, object]:
    return {"method": "error", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1", "willRetry": True,
        "error": {"message": "SYNTHETIC_PRIVATE_error", "codexErrorInfo": "serverOverloaded"},
        **changes,
    }}


def item(kind: str = "reasoning", method: str = "item/started") -> dict[str, object]:
    return {"method": method, "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1",
        "item": {"id": "item_synthetic", "type": kind},
    }}


def delta(text: str = "Four.") -> dict[str, object]:
    return {"method": "item/agentMessage/delta", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1",
        "itemId": "item_synthetic", "delta": text,
    }}


class RetryTransport(EofOnCloseCodexTransport):
    def __init__(self, messages=(), *, before_ack=(), complete=True, ack_turn=None):
        super().__init__()
        self.messages = messages
        self.before_ack = before_ack
        self.complete = complete
        self.ack_turn = ack_turn or {
            "id": "turn_server_1", "items": [], "status": "inProgress",
        }

    async def send(self, message: Mapping[str, object]) -> None:
        if message.get("method") != "turn/start":
            await super().send(message)
            return
        self.sent.append(dict(message))
        for event in self.before_ack:
            await self.incoming.put(copy.deepcopy(event))
        await self.incoming.put({"id": message["id"], "result": {"turn": self.ack_turn}})
        # Deliberately do not yield: the reader can drain all events before opening resolves.
        for event in self.messages:
            await self.incoming.put(copy.deepcopy(event))
        if self.complete:
            await self.incoming.put({"method": "turn/completed", "params": {
                "threadId": "thread_server_1", "turn": {
                    "id": "turn_server_1", "items": [], "status": "completed",
                },
            }})


def inference_for(transport, **kwargs):
    return provider.CodexAppServerStreamingInference(
        model="gpt-5.6-terra", effort="low", transport_factory=lambda: transport,
        request_timeout_seconds=kwargs.pop("request_timeout_seconds", 1), **kwargs,
    )


async def collect(inference):
    return [value async for value in inference.stream(_snapshot(), turn_id="synthetic_retry")]


def refusal_category(error):
    current = error
    for _ in range(8):
        if current is None:
            break
        if type(current) is RuntimeError and str(current).startswith(
            "Codex error notification refused: "
        ):
            return str(current).removeprefix("Codex error notification refused: ")
        current = current.__cause__
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("with_advisory", [False, True], ids=["healthy", "advisory"])
async def test_pre_activity_advisory_keeps_one_original_turn(with_advisory):
    transport = RetryTransport(([advisory()] if with_advisory else []) + [delta()])
    inference = inference_for(transport)
    try:
        assert await collect(inference) == ["Four."]
    finally:
        await inference.close()
    assert [event.get("method") for event in transport.sent].count("thread/start") == 1
    assert [event.get("method") for event in transport.sent].count("turn/start") == 1
    assert transport.closed
    assert inference._retry_boundaries == {}
    assert inference._turn_start_requests == {}


@pytest.mark.asyncio
async def test_advisory_before_later_item_activity_is_not_retroactively_refused():
    transport = RetryTransport([advisory(), item(), delta()])
    inference = inference_for(transport)
    try:
        assert await collect(inference) == ["Four."]
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_pre_activity_advisory_then_empty_completion_does_not_replay():
    transport = RetryTransport([advisory()])
    inference = inference_for(transport)
    try:
        assert await collect(inference) == []
    finally:
        await inference.close()
    assert sum(event.get("method") == "turn/start" for event in transport.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prior", [delta("unfinished"), item(), item("agentMessage", "item/completed")],
    ids=["partial_text", "reasoning", "completed_item"],
)
async def test_any_prior_item_activity_refuses_advisory(prior):
    transport = RetryTransport([prior, advisory(), delta()])
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "activity"
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_item_activity_before_ack_is_not_lost():
    transport = RetryTransport([advisory(), delta()], before_ack=[item()])
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "activity"
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_dynamic_request_counts_as_activity_before_async_routing():
    call = {"id": 91, "method": "item/tool/call", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1", "callId": "call_synthetic",
        "namespace": "hermes", "tool": "start_work", "arguments": {"objective": "Synthetic work"},
    }}
    started = item("dynamicToolCall")
    started["params"]["item"].update({
        "id": "call_synthetic", "tool": "start_work", "namespace": "hermes",
        "arguments": {"objective": "Synthetic work"}, "status": "inProgress",
    })
    started["params"]["startedAtMs"] = 1
    transport = RetryTransport([], complete=False)
    inference = inference_for(transport)
    inference.bind_work_tools(FakeWorkToolHandler())
    operation = asyncio.create_task(collect(inference))
    try:
        while not inference._active_turns:
            await asyncio.sleep(0)
        for message in (started, call, advisory()):
            await transport.incoming.put(message)
        with pytest.raises(RuntimeError) as failure:
            await operation
        assert refusal_category(failure.value) == "activity"
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change,category", [
    ({"turnId": "stale_turn"}, "identity"), ({"threadId": "stale_thread"}, "identity"),
    ({"willRetry": False}, "terminal"), ({"willRetry": 1}, "malformed"),
    ({"error": {"message": 1}}, "malformed"),
    ({"error": {"message": "synthetic", "codexErrorInfo": "unknown_code"}}, "malformed"),
])
async def test_noneligible_notifications_remain_fail_closed(change, category, capsys):
    transport = RetryTransport([advisory(**change), delta()])
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == category
    finally:
        await inference.close()
    output = capsys.readouterr().out
    assert "[codex-error-notification] " in output
    assert "SYNTHETIC_PRIVATE" not in output


@pytest.mark.asyncio
async def test_advisory_before_accepted_ack_is_refused():
    transport = RetryTransport([delta()], before_ack=[advisory()])
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "identity"
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [{"message": "synthetic"},
                                  {"message": "synthetic", "codexErrorInfo": None}])
async def test_schema_optional_null_error_info_is_eligible(error):
    transport = RetryTransport([advisory(error=error), delta()])
    inference = inference_for(transport)
    try:
        assert await collect(inference) == ["Four."]
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_advisory_limit_is_ten_without_a_new_turn():
    transport = RetryTransport([advisory() for _ in range(11)] + [delta()])
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "advisory_limit"
    finally:
        await inference.close()
    assert sum(event.get("method") == "turn/start" for event in transport.sent) == 1


@pytest.mark.asyncio
async def test_advisory_flood_does_not_restart_the_original_event_wait():
    transport = RetryTransport([advisory() for _ in range(10)], complete=False)
    inference = inference_for(transport, request_timeout_seconds=0.03)
    original = inference._next_event
    waits = 0

    async def observed_wait(*args, **kwargs):
        nonlocal waits
        waits += 1
        return await original(*args, **kwargs)

    inference._next_event = observed_wait
    try:
        with pytest.raises(TimeoutError):
            await collect(inference)
        assert waits == 1
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_cancellation_after_advisory_still_interrupts_and_closes():
    transport = RetryTransport([advisory()], complete=False)
    inference = inference_for(transport)
    operation = asyncio.create_task(collect(inference))
    try:
        while not inference._active_turns:
            await asyncio.sleep(0)
        await inference.cancel("synthetic_retry")
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    finally:
        await inference.close()
    assert sum(event.get("method") == "turn/interrupt" for event in transport.sent) == 1
    assert transport.closed


@pytest.mark.asyncio
async def test_advisory_after_terminal_is_not_tolerated_before_thread_cleanup():
    completed = {"method": "turn/completed", "params": {
        "threadId": "thread_server_1", "turn": {
            "id": "turn_server_1", "items": [], "status": "completed",
        },
    }}
    transport = RetryTransport([completed, advisory()], complete=False)
    inference = inference_for(transport)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "terminal"
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_invalid_turn_start_identity_does_not_leave_pending_request():
    inference = inference_for(RetryTransport())
    try:
        with pytest.raises(RuntimeError):
            await inference._request_locked("turn/start", {"threadId": None})
        assert inference._pending == {}
        assert inference._turn_start_requests == {}
    finally:
        await inference.close()


@pytest.mark.parametrize("info", [
    None, "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "serverOverloaded", "cyberPolicy", "internalServerError", "unauthorized", "badRequest",
    "threadRollbackFailed", "sandboxError", "other",
    {"httpConnectionFailed": {}}, {"responseStreamConnectionFailed": {"httpStatusCode": None}},
    {"responseStreamDisconnected": {"httpStatusCode": 0}},
    {"responseTooManyFailedAttempts": {"httpStatusCode": 65535}},
    {"activeTurnNotSteerable": {"turnKind": "review"}},
    {"activeTurnNotSteerable": {"turnKind": "compact"}},
])
def test_pinned_error_info_schema_accepts_only_defined_shapes(info):
    assert provider.CodexAppServerStreamingInference._valid_error_info(info) is True


@pytest.mark.parametrize("info", [
    True, "synthetic_unknown", {}, {"synthetic_unknown": {}},
    {"httpConnectionFailed": {}, "responseStreamDisconnected": {}},
    {"httpConnectionFailed": None}, {"httpConnectionFailed": {"httpStatusCode": True}},
    {"httpConnectionFailed": {"httpStatusCode": -1}},
    {"httpConnectionFailed": {"httpStatusCode": 65536}},
    {"httpConnectionFailed": {"httpStatusCode": 429.0}},
    {"activeTurnNotSteerable": {}}, {"activeTurnNotSteerable": {"turnKind": "unknown"}},
    {"activeTurnNotSteerable": {"turnKind": []}},
])
def test_pinned_error_info_schema_refuses_malformed_or_unknown_shapes(info):
    assert provider.CodexAppServerStreamingInference._valid_error_info(info) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"threadId": None}, {"threadId": ""}, {"turnId": None}, {"turnId": ""},
    {"threadId": "x" * 257}, {"turnId": "x" * 257}, {"error": None},
    {"error": {"message": "synthetic", "additionalDetails": []}},
])
async def test_error_notification_shape_is_validated_before_eligibility(changes):
    inference = inference_for(RetryTransport([advisory(**changes)]))
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "malformed"
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_unknown_item_notification_is_not_newly_tolerated():
    unknown = {"method": "item/reasoning/textDelta", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1", "delta": "synthetic",
    }}
    inference = inference_for(RetryTransport([unknown, advisory(), delta()]))
    try:
        with pytest.raises(RuntimeError):
            await collect(inference)
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("turn,category", [
    ({"id": "turn_server_1", "items": [{"type": "reasoning"}], "status": "inProgress"},
     "activity"),
    ({"id": "turn_server_1", "items": [], "status": "completed"}, "identity"),
    ({"id": "turn_server_1", "items": [], "status": "unknown"}, "identity"),
    ({"id": "turn_server_1", "status": "inProgress"}, "identity"),
])
async def test_ack_must_establish_live_turn_without_prior_item_activity(turn, category):
    inference = inference_for(RetryTransport([advisory(), delta()], ack_turn=turn))
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == category
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_completion_before_ack_is_not_lost():
    completed = {"method": "turn/completed", "params": {
        "threadId": "thread_server_1", "turn": {
            "id": "turn_server_1", "items": [], "status": "completed",
        },
    }}
    inference = inference_for(RetryTransport([advisory()], before_ack=[completed]))
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "terminal"
    finally:
        await inference.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [None, {}, {"willRetry": True}])
async def test_malformed_error_notification_is_not_silently_skipped(params):
    inference = inference_for(RetryTransport([{"method": "error", "params": params}, delta()]))
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "malformed"
    finally:
        await inference.close()


@pytest.mark.asyncio
async def test_item_request_marks_activity_before_admission_and_async_routing(monkeypatch):
    request = {"id": 91, "method": "item/tool/call", "params": {
        "threadId": "thread_server_1", "turnId": "turn_server_1", "callId": "synthetic",
        "tool": "start_work", "arguments": {"objective": "Synthetic work"},
    }}
    inference = inference_for(RetryTransport([request, advisory(), delta()]))
    observed = []

    def admitted(message):
        observed.append(inference._retry_boundaries["thread_server_1"].item_activity)

    async def routed(message):
        pass

    # Isolate the new reader ordering gate from the existing lifecycle-admission guards.
    monkeypatch.setattr(inference, "_admit_server_request", admitted)
    monkeypatch.setattr(inference, "_route_server_request", routed)
    try:
        with pytest.raises(RuntimeError) as failure:
            await collect(inference)
        assert refusal_category(failure.value) == "activity"
        assert observed == [True]
    finally:
        await inference.close()
