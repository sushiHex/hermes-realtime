from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import pytest
from aiohttp import web

from hermes_realtime.integration import (
    HermesApiConfig,
    HermesApiTaskSession,
    HermesRestartSettlement,
)
from hermes_realtime.integration import api as api_module
from hermes_realtime.integration import run_record as run_record_module
from hermes_realtime.integration.api import _RunAuthority
from hermes_realtime.integration.run_record import write_run_record
from hermes_realtime.protocol import (
    CancelScope,
    ControlCancelEvent,
    ControlCancelPayload,
    WorkDispatchRequestedEvent,
    WorkDispatchRequestedPayload,
    WorkTerminalStatus,
)


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _RunState(TypedDict):
    status: str
    stop_calls: int


def _dispatch_request(
    *, sequence: int = 1, task_id: str = "task_release_check"
) -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        event_id=f"event_dispatch_{sequence}",
        session_id="session_api_1",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        type="work.dispatch.requested",
        task_id=task_id,
        utterance_id=f"utterance_{sequence}",
        payload=WorkDispatchRequestedPayload(objective="Inspect the release evidence"),
    )


def _cancel_request(*, sequence: int = 3) -> ControlCancelEvent:
    return ControlCancelEvent(
        event_id="event_cancel_1",
        session_id="session_api_1",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        type="control.cancel",
        task_id="task_release_check",
        payload=ControlCancelPayload(scope=CancelScope.TASK, reason="user requested stop"),
    )


def _capabilities(*, approval_response: bool = True) -> dict[str, object]:
    return {
        "object": "hermes.api_server.capabilities",
        "platform": "hermes-agent",
        "model": "test-model",
        "auth": {"type": "bearer", "required": True},
        "runtime": {
            "mode": "server_agent",
            "tool_execution": "server",
            "split_runtime": False,
            "description": "test runtime",
        },
        "features": {
            "run_submission": True,
            "run_status": True,
            "run_events_sse": True,
            "run_stop": True,
            "run_approval_response": approval_response,
            "approval_events": True,
        },
        "endpoints": {
            "runs": {"method": "POST", "path": "/v1/runs"},
            "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
            "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
            "run_approval": {
                "method": "POST",
                "path": "/v1/runs/{run_id}/approval",
            },
            "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
        },
    }


async def _capabilities_response(_request: web.Request) -> web.Response:
    return web.json_response(_capabilities())


async def _started_run_response(_request: web.Request) -> web.Response:
    return web.json_response({"run_id": "run_0123456789abcdef", "status": "started"}, status=202)


def test_api_config_is_loopback_only_and_redacts_bearer() -> None:
    bearer = "local-test-bearer-value-32-characters"
    config = HermesApiConfig(
        base_url="http://127.0.0.1:8642",
        bearer=bearer,
    )

    assert bearer not in repr(config)
    assert bearer not in str(config)
    with pytest.raises(ValueError, match="loopback"):
        HermesApiConfig(base_url="https://hermes.example.com", bearer=bearer)
    with pytest.raises(ValueError, match="strong bearer"):
        HermesApiConfig(base_url="http://127.0.0.1:8642", bearer="too-short")


@pytest.mark.parametrize(
    "terminal_text",
    [
        "run_0123456789abcdef",
        "Processed run_0123456789abcdef successfully.",
        "Xrun_0123456789abcdefY",
        "Xrun_0123456789abcdegY",
        "prefix\nrun_0123456789abcdef\nsuffix",
        "run_0123456789abcdef and run_fedcba9876543210",
    ],
)
def test_terminal_text_rejects_private_run_tokens_anywhere(
    terminal_text: str,
) -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8642",
            bearer="private-token-test-value-32-characters",
        ),
        session_id="session_private_token_test",
    )

    with pytest.raises(RuntimeError, match="private authority"):
        session._bounded_terminal_text(terminal_text, "output")


@pytest.mark.asyncio
async def test_api_session_requires_exact_capabilities_before_dispatch() -> None:
    port = _available_port()

    async def capabilities(_request: web.Request) -> web.Response:
        return web.json_response(_capabilities(approval_response=False))

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="capability-test-bearer-value-32-chars",
        ),
        session_id="session_api_1",
    )

    try:
        with pytest.raises(RuntimeError, match="approval"):
            await session.start()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_rejects_split_or_unauthenticated_runtime() -> None:
    port = _available_port()

    async def capabilities(_request: web.Request) -> web.Response:
        payload = _capabilities()
        payload["auth"] = {"type": "bearer", "required": False}
        payload["runtime"] = {
            "mode": "server_agent",
            "tool_execution": "server",
            "split_runtime": True,
        }
        return web.json_response(payload)

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="runtime-test-bearer-value-32-characters",
        ),
        session_id="session_api_1",
    )

    try:
        with pytest.raises(RuntimeError, match="authenticated server-side runtime"):
            await session.start()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_posts_independent_dispatches_concurrently() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="concurrent-post-test-bearer-value-32-characters",
        ),
        session_id="session_concurrent_posts",
        private_id_factory=iter(("first", "second")).__next__,
    )
    first_post_entered = asyncio.Event()
    release_first_post = asyncio.Event()
    post_count = 0

    async def request_json(
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        nonlocal post_count
        del body
        assert method == "POST"
        assert path == "/v1/runs"
        post_count += 1
        ordinal = post_count
        if ordinal == 1:
            first_post_entered.set()
            await release_first_post.wait()
        return 202, {"run_id": f"run_{ordinal:016x}", "status": "started"}

    def request(task_id: str, objective: str) -> WorkDispatchRequestedEvent:
        return WorkDispatchRequestedEvent(
            event_id=f"evt_{task_id}",
            session_id="session_concurrent_posts",
            sequence=1,
            timestamp=datetime.now(UTC),
            type="work.dispatch.requested",
            utterance_id=f"utt_{task_id}",
            task_id=task_id,
            payload=WorkDispatchRequestedPayload(objective=objective),
        )

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    first = asyncio.create_task(session.dispatch(request("task_first", "first")))
    await asyncio.wait_for(first_post_entered.wait(), timeout=1)
    second = asyncio.create_task(session.dispatch(request("task_second", "second")))
    try:
        second_ack = await asyncio.wait_for(second, timeout=0.2)
        assert second_ack.payload.accepted is True
        assert post_count == 2
        release_first_post.set()
        first_ack = await asyncio.wait_for(first, timeout=1)
        assert first_ack.payload.accepted is True
    finally:
        release_first_post.set()
        await asyncio.gather(first, second, return_exceptions=True)
        for authority in session._runs_by_task.values():
            authority.terminal = True
        await session.close()


@pytest.mark.asyncio
async def test_api_session_requests_a_bounded_priority_background_run() -> None:
    captured: list[dict[str, object]] = []
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="priority-background-test-bearer-value-32-characters",
            run_provider="openai-codex",
            run_model="gpt-5.6-terra",
            run_reasoning_effort="low",
            run_service_tier="priority",
        ),
        session_id="session_priority_background",
        private_id_factory=lambda: "priority_private",
    )

    async def request_json(
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        assert method == "POST"
        assert path == "/v1/runs"
        assert body is not None
        captured.append(body)
        return 202, {"run_id": "run_0123456789abcdef", "status": "started"}

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    request = WorkDispatchRequestedEvent(
        event_id="evt_priority_background",
        session_id="session_priority_background",
        sequence=1,
        timestamp=datetime.now(UTC),
        type="work.dispatch.requested",
        utterance_id="utt_priority_background",
        task_id="task_priority_background",
        payload=WorkDispatchRequestedPayload(objective="Find today's major news."),
    )

    try:
        acknowledgement = await session.dispatch(request)
        assert acknowledgement.payload.accepted is True
        assert captured == [
            {
                "input": "Find today's major news.",
                "instructions": captured[0]["instructions"],
                "provider": "openai-codex",
                "model": "gpt-5.6-terra",
                "model_options": {
                    "reasoning_effort": "low",
                    "service_tier": "priority",
                },
            }
        ]
        instructions = captured[0]["instructions"]
        assert isinstance(instructions, str)
        assert "one parallel batch" in instructions
        assert "no more than five web_search calls" in instructions
        assert "Local, runtime, file, process, build, and verification objectives" in instructions
        assert (
            "Never interpret an unavailable tool, failed command, or empty output"
            in instructions
        )
        assert "at most 1800 characters" in instructions
    finally:
        for authority in session._runs_by_task.values():
            authority.terminal = True
        await session.close()


@pytest.mark.asyncio
async def test_api_session_runs_two_tasks_concurrently_and_settles_independently() -> None:
    port = _available_port()
    run_ids = iter(("run_1111111111111111", "run_2222222222222222"))
    completion_gates = {
        "run_1111111111111111": asyncio.Event(),
        "run_2222222222222222": asyncio.Event(),
    }

    async def capabilities(_request: web.Request) -> web.Response:
        return web.json_response(_capabilities())

    async def create_run(request: web.Request) -> web.Response:
        body = await request.json()
        assert type(body) is dict
        run_id = next(run_ids)
        return web.json_response({"run_id": run_id, "status": "started"}, status=202)

    async def run_events(request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await completion_gates[run_id].wait()
        payload = {
            "event": "run.completed",
            "run_id": run_id,
            "output": "Completed independently.",
        }
        await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write_eof()
        return response

    async def stop_run(request: web.Request) -> web.Response:
        return web.json_response({"run_id": request.match_info["run_id"], "status": "stopping"})

    async def run_status(request: web.Request) -> web.Response:
        return web.json_response({"run_id": request.match_info["run_id"], "status": "running"})

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", stop_run)
    app.router.add_get("/v1/runs/{run_id}", run_status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    private_ids = iter(("multi_private_alpha", "multi_private_beta"))
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="multi-run-test-bearer-value-32-characters",
            request_timeout_seconds=1,
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: next(private_ids),
    )

    try:
        await session.start()
        first = await session.dispatch(_dispatch_request(sequence=1, task_id="task_alpha"))
        second = await session.dispatch(_dispatch_request(sequence=2, task_id="task_beta"))

        assert first.payload.accepted is True
        assert first.payload.run_id == "deleg_multi_private_alpha"
        assert second.payload.accepted is True
        assert second.payload.run_id == "deleg_multi_private_beta"

        completion_gates["run_2222222222222222"].set()
        second_terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert second_terminal.task_id == "task_beta"
        assert second_terminal.run_id == "deleg_multi_private_beta"

        completion_gates["run_1111111111111111"].set()
        first_terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert first_terminal.task_id == "task_alpha"
        assert first_terminal.run_id == "deleg_multi_private_alpha"
    finally:
        for gate in completion_gates.values():
            gate.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_accepts_eight_active_runs_and_rejects_ninth_without_post() -> None:
    port = _available_port()
    create_calls = 0
    states: dict[str, str] = {}

    async def create_run(_request: web.Request) -> web.Response:
        nonlocal create_calls
        create_calls += 1
        run_id = f"run_{create_calls:016x}"
        states[run_id] = "running"
        return web.json_response({"run_id": run_id, "status": "started"}, status=202)

    async def run_events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        while states[request.match_info["run_id"]] == "running":
            await asyncio.sleep(0.01)
        await response.write_eof()
        return response

    async def stop_run(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        states[run_id] = "cancelled"
        return web.json_response({"run_id": run_id, "status": "stopping"})

    async def run_status(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        return web.json_response({"run_id": run_id, "status": states[run_id]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", stop_run)
    app.router.add_get("/v1/runs/{run_id}", run_status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    private_ids = iter(f"capacity_private_{index}" for index in range(8))
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="capacity-test-bearer-value-32-characters",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: next(private_ids),
    )

    try:
        await session.start()
        acknowledgments = [
            await session.dispatch(
                _dispatch_request(sequence=index + 1, task_id=f"task_capacity_{index}")
            )
            for index in range(8)
        ]
        ninth = await session.dispatch(_dispatch_request(sequence=9, task_id="task_capacity_ninth"))

        assert all(ack.payload.accepted for ack in acknowledgments)
        assert ninth.payload.accepted is False
        assert ninth.payload.reason == "background task capacity exhausted"
        assert create_calls == 8
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_settles_nonstarted_accepted_run_with_usable_id() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="queued-response-test-bearer-value-32-characters",
        ),
        session_id="session_queued_response",
    )
    stopped: list[str] = []

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {"run_id": "run_1212121212121212", "status": "queued"}

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="lacks exact authority"):
        await session.dispatch(_dispatch_request())
    assert stopped == ["run_1212121212121212"]
    assert session._unpublished_api_run_ids == set()
    await session.close()


@pytest.mark.asyncio
async def test_api_session_accepts_exact_v021_first_admission_response() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="first-admission-test-bearer-value-32-characters",
        ),
        session_id="session_first_admission",
        private_id_factory=lambda: "first_admission_private",
    )
    stopped: list[str] = []

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {
            "run_id": "run_2323232323232323",
            "status": "started",
            "replayed": False,
        }

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    acknowledgment = await session.dispatch(_dispatch_request())
    assert acknowledgment.payload.accepted is True
    assert acknowledgment.payload.run_id == "deleg_first_admission_private"
    assert stopped == []
    authority = session._runs_by_task["task_release_check"]
    authority.terminal = True
    if authority.event_task is not None:
        authority.event_task.cancel()
        await asyncio.gather(authority.event_task, return_exceptions=True)
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replayed",
    [
        pytest.param(True, id="replayed-true"),
        pytest.param(0, id="integer-zero"),
        pytest.param(None, id="null"),
        pytest.param("false", id="string-false"),
    ],
)
async def test_api_session_rejects_nonfalse_replay_marker_and_settles_run(
    replayed: object,
) -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="replayed-response-test-bearer-value-32-characters",
        ),
        session_id="session_replayed_response",
    )
    stopped: list[str] = []

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {
            "run_id": "run_2424242424242424",
            "status": "started",
            "replayed": replayed,
        }

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="malformed"):
        await session.dispatch(_dispatch_request())
    assert stopped == ["run_2424242424242424"]
    assert session._unpublished_api_run_ids == set()
    assert session._reserved_api_run_ids == {"run_2424242424242424"}
    await session.close()


@pytest.mark.asyncio
async def test_api_session_cancelled_dispatch_settles_remote_acceptance() -> None:
    accepted = asyncio.Event()
    release = asyncio.Event()
    stopped: list[str] = []
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="cancelled-dispatch-test-bearer-value-32-characters",
        ),
        session_id="session_cancelled_dispatch",
        private_id_factory=lambda: "cancelled_dispatch_private",
    )

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        accepted.set()
        await release.wait()
        return 202, {"run_id": "run_3434343434343434", "status": "started"}

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    dispatch = asyncio.create_task(session.dispatch(_dispatch_request()))
    await accepted.wait()
    dispatch.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert stopped == ["run_3434343434343434"]
    assert session._runs_by_task == {}
    assert session._unpublished_api_run_ids == set()
    await session.close()


@pytest.mark.asyncio
async def test_api_session_settles_identifiable_run_from_malformed_accepted_response() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="malformed-response-test-bearer-value-32-characters",
        ),
        session_id="session_malformed_response",
    )
    stopped: list[str] = []

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {
            "run_id": "run_abcdabcdabcdabcd",
            "status": "started",
            "unexpected": True,
        }

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="malformed"):
        await session.dispatch(_dispatch_request())
    assert stopped == ["run_abcdabcdabcdabcd"]
    assert session._unpublished_api_run_ids == set()
    assert session._reserved_api_run_ids == {"run_abcdabcdabcdabcd"}
    await session.close()


@pytest.mark.asyncio
async def test_api_session_rejects_retired_api_run_id_reuse() -> None:
    private_ids = iter(("api_reuse_first", "api_reuse_second"))
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="api-reuse-test-bearer-value-32-characters",
        ),
        session_id="session_api_reuse",
        private_id_factory=lambda: next(private_ids),
    )

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {"run_id": "run_eeeeeeeeeeeeeeee", "status": "started"}

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    first = await session.dispatch(_dispatch_request(sequence=1, task_id="task_api_first"))
    assert first.payload.accepted is True
    authority = session._runs_by_task.pop("task_api_first")
    session._runs_by_api.pop(authority.api_run_id)
    authority.terminal = True
    if authority.event_task is not None:
        authority.event_task.cancel()
        await asyncio.gather(authority.event_task, return_exceptions=True)

    with pytest.raises(RuntimeError, match="reused a run authority"):
        await session.dispatch(_dispatch_request(sequence=2, task_id="task_api_second"))
    assert session._reserved_api_run_ids == {"run_eeeeeeeeeeeeeeee"}
    await session.close()


@pytest.mark.asyncio
async def test_api_session_cancelled_claim_is_settled_before_cancellation_returns() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="cancelled-claim-test-bearer-value-32-characters",
        ),
        session_id="session_cancelled_claim",
    )
    settled: list[str] = []

    async def stop_and_wait(api_run_id: str) -> None:
        settled.append(api_run_id)

    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    await session._state_lock.acquire()
    claim = asyncio.create_task(session._claim_unpublished("run_ffffffffffffffff"))
    await asyncio.sleep(0)
    claim.cancel()
    session._state_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await claim
    assert settled == ["run_ffffffffffffffff"]
    assert session._unpublished_api_run_ids == set()
    assert session._reserved_api_run_ids == {"run_ffffffffffffffff"}


@pytest.mark.asyncio
async def test_api_session_never_reuses_retired_private_authority_and_tracks_failed_cleanup() -> (
    None
):
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="retired-collision-test-bearer-value",
        ),
        session_id="session_retired_collision",
        private_id_factory=lambda: "same",
    )
    run_ids = iter(("run_1111111111111111", "run_2222222222222222"))
    cleanup_must_fail = True
    cleanup_calls: list[str] = []

    async def request_json(
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        del body
        assert (method, path) == ("POST", "/v1/runs")
        return 202, {"run_id": next(run_ids), "status": "started"}

    async def stop_and_wait(api_run_id: str) -> dict[str, object]:
        cleanup_calls.append(api_run_id)
        if cleanup_must_fail:
            raise RuntimeError("simulated settlement outage")
        return {"run_id": api_run_id, "status": "cancelled"}

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    first = await session.dispatch(_dispatch_request())
    assert first.payload.accepted is True
    first_authority = session._runs_by_task.pop("task_release_check")
    session._runs_by_api.pop(first_authority.api_run_id)
    first_authority.terminal = True

    try:
        with pytest.raises(RuntimeError, match="simulated settlement outage"):
            await session.dispatch(
                WorkDispatchRequestedEvent(
                    event_id="evt_retired_collision",
                    session_id="session_api_1",
                    sequence=2,
                    timestamp=datetime.now(UTC),
                    type="work.dispatch.requested",
                    utterance_id="utt_retired_collision",
                    task_id="task_retired_collision",
                    payload=WorkDispatchRequestedPayload(objective="second objective"),
                )
            )
        assert session._unpublished_api_run_ids == {"run_2222222222222222"}
        assert session._reserved_protocol_run_ids == {"deleg_same"}
    finally:
        cleanup_must_fail = False
        await session.close()

    assert cleanup_calls == ["run_2222222222222222", "run_2222222222222222"]


@pytest.mark.asyncio
async def test_api_session_rejects_private_protocol_id_collision_and_settles_new_run() -> None:
    port = _available_port()
    run_ids = iter(("run_aaaaaaaaaaaaaaaa", "run_bbbbbbbbbbbbbbbb"))
    states = {
        "run_aaaaaaaaaaaaaaaa": "running",
        "run_bbbbbbbbbbbbbbbb": "running",
    }
    stop_calls: list[str] = []
    release_first = asyncio.Event()

    async def create_run(_request: web.Request) -> web.Response:
        return web.json_response(
            {"run_id": next(run_ids), "status": "started"},
            status=202,
        )

    async def run_events(request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if run_id == "run_aaaaaaaaaaaaaaaa":
            await release_first.wait()
            payload = {
                "event": "run.completed",
                "run_id": run_id,
                "output": "First run remained authoritative.",
            }
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write_eof()
        return response

    async def stop_run(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        stop_calls.append(run_id)
        states[run_id] = "cancelled"
        return web.json_response({"run_id": run_id, "status": "stopping"})

    async def run_status(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        return web.json_response({"run_id": run_id, "status": states[run_id]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", stop_run)
    app.router.add_get("/v1/runs/{run_id}", run_status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="collision-test-bearer-value-32-characters",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "same_private_id",
    )

    try:
        await session.start()
        first = await session.dispatch(_dispatch_request(sequence=1, task_id="task_alpha"))
        assert first.payload.accepted is True

        with pytest.raises(RuntimeError, match="duplicate value"):
            await session.dispatch(_dispatch_request(sequence=2, task_id="task_beta"))

        assert stop_calls == ["run_bbbbbbbbbbbbbbbb"]
        release_first.set()
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert terminal.task_id == "task_alpha"
        assert terminal.run_id == "deleg_same_private_id"
    finally:
        release_first.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_concurrent_close_calls_share_one_operation() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="shared-close-test-bearer-value-32-characters",
        ),
        session_id="session_shared_close",
    )
    release = asyncio.Event()
    entered = asyncio.Event()
    settle_calls = 0
    authority = _RunAuthority(
        "task_shared_close",
        "run_7878787878787878",
        "deleg_shared_close",
    )

    async def settle(_authority: _RunAuthority) -> None:
        nonlocal settle_calls
        settle_calls += 1
        entered.set()
        await release.wait()
        _authority.terminal = True

    session._started = True
    session._runs_by_task[authority.task_id] = authority
    session._runs_by_api[authority.api_run_id] = authority
    session._settle_authority = settle  # type: ignore[method-assign]
    first = asyncio.create_task(session.close())
    second = asyncio.create_task(session.close())
    await asyncio.wait_for(entered.wait(), timeout=0.2)
    assert settle_calls == 1
    release.set()
    await asyncio.gather(first, second)
    assert settle_calls == 1


@pytest.mark.asyncio
async def test_api_session_close_settles_all_active_runs_concurrently() -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="parallel-close-test-bearer-value-32-characters",
        ),
        session_id="session_parallel_close",
    )
    release = asyncio.Event()
    entered: set[str] = set()
    all_entered = asyncio.Event()
    consumers_started: set[str] = set()
    all_consumers_started = asyncio.Event()
    authorities = (
        _RunAuthority("task_close_one", "run_1111111111111111", "deleg_close_one"),
        _RunAuthority("task_close_two", "run_2222222222222222", "deleg_close_two"),
    )

    async def settle(authority: _RunAuthority) -> None:
        assert authority.event_task is not None
        assert authority.event_task.done()
        entered.add(authority.api_run_id)
        if len(entered) == 2:
            all_entered.set()
        await release.wait()
        authority.terminal = True

    session._started = True
    session._settle_authority = settle  # type: ignore[method-assign]

    async def consume(api_run_id: str) -> None:
        consumers_started.add(api_run_id)
        if len(consumers_started) == 2:
            all_consumers_started.set()
        await asyncio.Event().wait()

    for authority in authorities:
        authority.event_task = asyncio.create_task(consume(authority.api_run_id))
        session._runs_by_task[authority.task_id] = authority
        session._runs_by_api[authority.api_run_id] = authority
    await asyncio.wait_for(all_consumers_started.wait(), timeout=0.2)
    close = asyncio.create_task(session.close())
    try:
        await asyncio.wait_for(all_entered.wait(), timeout=0.2)
        assert entered == {"run_1111111111111111", "run_2222222222222222"}
    finally:
        release.set()
        await close


@pytest.mark.asyncio
async def test_api_session_close_settles_run_accepted_before_local_publication() -> None:
    port = _available_port()
    remote_accepted = asyncio.Event()
    release_response = asyncio.Event()
    state = {"status": "running", "stop_calls": 0}

    async def create_run(_request: web.Request) -> web.Response:
        remote_accepted.set()
        await release_response.wait()
        return web.json_response(
            {"run_id": "run_cccccccccccccccc", "status": "started"},
            status=202,
        )

    async def stop_run(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        state["status"] = "cancelled"
        return web.json_response({"run_id": "run_cccccccccccccccc", "status": "stopping"})

    async def run_status(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": "run_cccccccccccccccc", "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_post("/v1/runs/{run_id}/stop", stop_run)
    app.router.add_get("/v1/runs/{run_id}", run_status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="close-race-test-bearer-value-32-characters",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "close_race_private",
    )

    dispatch: asyncio.Task[object] | None = None
    close: asyncio.Task[object] | None = None
    try:
        await session.start()
        dispatch = asyncio.create_task(session.dispatch(_dispatch_request()))
        await asyncio.wait_for(remote_accepted.wait(), timeout=1)
        await session._state_lock.acquire()

        close = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        release_response.set()
        await asyncio.sleep(0)
        session._state_lock.release()

        await asyncio.wait_for(close, timeout=1)
        with pytest.raises(RuntimeError, match="closed during dispatch"):
            await dispatch
        assert state == {"status": "cancelled", "stop_calls": 1}
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session.next_update(), timeout=0.05)
    finally:
        release_response.set()
        if session._state_lock.locked():
            session._state_lock.release()
        if dispatch is not None and not dispatch.done():
            dispatch.cancel()
            await asyncio.gather(dispatch, return_exceptions=True)
        if close is not None and not close.done():
            close.cancel()
            await asyncio.gather(close, return_exceptions=True)
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_approval_commit_survives_terminal_withdrawal_race() -> None:
    approval_events: list[dict[str, str | int | bool | None]] = []
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="approval-race-test-bearer-value-32-characters",
        ),
        session_id="session_approval_race",
        approval_observer=approval_events.append,
    )
    authority = _RunAuthority(
        "task_approval_race",
        "run_5656565656565656",
        "deleg_approval_race",
    )
    session._started = True
    session._runs_by_task[authority.task_id] = authority
    session._runs_by_api[authority.api_run_id] = authority
    await session._consume_approval(
        authority,
        {
            "command": "Inspect build output",
            "description": "Read the build log",
            "choices": ["deny", "once"],
        },
    )
    approval_id = str(approval_events[0]["approvalId"])

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        async with session._state_lock:
            session._pending_approvals.clear()
            authority.terminal = True
            session._runs_by_task.clear()
            session._runs_by_api.clear()
        return 200, {
            "choice": "once",
            "object": "hermes.run.approval_response",
            "resolved": 1,
            "run_id": authority.api_run_id,
        }

    session._request_json = request_json  # type: ignore[method-assign]
    await session.decide_approval(
        approval_id=approval_id,
        sequence=1,
        decision="approve",
    )
    assert session._approval_sequence == 1
    assert tuple(session._pending_approvals) == ()
    await session.close()


@pytest.mark.asyncio
async def test_api_session_resolves_concurrent_run_approvals_by_exact_id() -> None:
    port = _available_port()
    approval_events: list[dict[str, str | int | bool | None]] = []
    resolved_runs: list[str] = []

    async def capabilities(_request: web.Request) -> web.Response:
        return web.json_response(_capabilities())

    async def decide(request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        body = await request.json()
        resolved_runs.append(run_id)
        return web.json_response(
            {
                "object": "hermes.run.approval_response",
                "run_id": run_id,
                "choice": body["choice"],
                "resolved": 1,
            }
        )

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_post("/v1/runs/{run_id}/approval", decide)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="approval-concurrency-test-bearer-value",
        ),
        session_id="session_approval_concurrency",
        approval_observer=approval_events.append,
    )
    first = _RunAuthority(
        task_id="task_first_approval",
        api_run_id="run_1111111111111111",
        protocol_run_id="deleg_first_approval",
    )
    second = _RunAuthority(
        task_id="task_second_approval",
        api_run_id="run_2222222222222222",
        protocol_run_id="deleg_second_approval",
    )

    try:
        await session.start()
        session._runs_by_task.update({first.task_id: first, second.task_id: second})
        session._runs_by_api.update({first.api_run_id: first, second.api_run_id: second})
        payload = {
            "command": "touch bounded-fixture",
            "description": "create a harmless fixture",
            "choices": ["once", "deny"],
        }
        await session._consume_approval(first, payload)
        await session._consume_approval(second, payload)
        pending = [event for event in approval_events if event["state"] == "pending"]
        assert len(pending) == 2

        await session.decide_approval(
            approval_id=str(pending[1]["approvalId"]),
            sequence=1,
            decision="reject",
        )
        await session.decide_approval(
            approval_id=str(pending[0]["approvalId"]),
            sequence=2,
            decision="approve",
        )

        assert resolved_runs == [second.api_run_id, first.api_run_id]
        assert [event["state"] for event in approval_events] == [
            "pending",
            "pending",
            "resolved",
            "resolved",
        ]
    finally:
        first.terminal = True
        second.terminal = True
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_routes_authoritative_dispatch_approval_and_completion() -> None:
    port = _available_port()
    approval_seen = asyncio.Event()
    approval_resolved = asyncio.Event()
    requests: list[tuple[str, object]] = []
    authorization: list[str] = []

    async def capabilities(request: web.Request) -> web.Response:
        authorization.append(request.headers.get("Authorization", ""))
        return web.json_response(_capabilities())

    async def create_run(request: web.Request) -> web.Response:
        authorization.append(request.headers.get("Authorization", ""))
        body = await request.json()
        requests.append(("dispatch", body))
        return web.json_response(
            {"run_id": "run_0123456789abcdef", "status": "started"},
            status=202,
        )

    async def run_events(request: web.Request) -> web.StreamResponse:
        authorization.append(request.headers.get("Authorization", ""))
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store"},
        )
        await response.prepare(request)
        approval = {
            "event": "approval.request",
            "run_id": "run_0123456789abcdef",
            "timestamp": 1.0,
            "command": "chmod release-marker",
            "description": "change harmless fixture mode",
            "choices": ["once", "deny"],
        }
        import json

        await response.write(f"data: {json.dumps(approval)}\n\n".encode())
        approval_seen.set()
        await approval_resolved.wait()
        completed = {
            "event": "run.completed",
            "run_id": "run_0123456789abcdef",
            "timestamp": 2.0,
            "output": "Release evidence is internally consistent.",
            "usage": {"total_tokens": 12},
        }
        await response.write(f"data: {json.dumps(completed)}\n\n".encode())
        await response.write_eof()
        return response

    async def decide_approval(request: web.Request) -> web.Response:
        authorization.append(request.headers.get("Authorization", ""))
        body = await request.json()
        requests.append(("approval", body))
        approval_resolved.set()
        return web.json_response(
            {
                "object": "hermes.run.approval_response",
                "run_id": "run_0123456789abcdef",
                "choice": body["choice"],
                "resolved": 1,
            }
        )

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", decide_approval)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    approvals: list[dict[str, str | int | bool | None]] = []
    bearer = "routing-test-bearer-value-32-characters"
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer=bearer,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "api_private_1",
        approval_observer=approvals.append,
    )

    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        assert acknowledgment.payload.accepted is True
        assert acknowledgment.payload.run_id == "deleg_api_private_1"
        assert "run_0123456789abcdef" not in acknowledgment.model_dump_json()
        await asyncio.wait_for(approval_seen.wait(), timeout=1)
        while not approvals:
            await asyncio.sleep(0)
        assert approvals == [
            {
                "actionable": True,
                "approvalId": approvals[0]["approvalId"],
                "command": "chmod release-marker",
                "description": "change harmless fixture mode",
                "state": "pending",
                "taskId": "task_release_check",
            }
        ]
        approval_id = approvals[0]["approvalId"]
        assert isinstance(approval_id, str) and approval_id.startswith("approval_")

        with pytest.raises(RuntimeError, match="active authority"):
            await session.decide_approval(
                approval_id="approval_ffffffffffffffff",
                sequence=1,
                decision="approve",
            )
        await session.decide_approval(
            approval_id=approval_id,
            sequence=1,
            decision="approve",
        )
        assert approvals[-1] == {
            "actionable": False,
            "approvalId": approval_id,
            "state": "resolved",
            "taskId": "task_release_check",
        }
        completion = await asyncio.wait_for(session.next_update(), timeout=1)
        assert completion.task_id == "task_release_check"
        assert completion.run_id == "deleg_api_private_1"
        assert completion.payload.status is WorkTerminalStatus.COMPLETED
        assert completion.payload.summary == "Release evidence is internally consistent."
        assert requests[0][0] == "dispatch"
        dispatch_body = requests[0][1]
        assert isinstance(dispatch_body, dict)
        assert dispatch_body["input"] == "Inspect the release evidence"
        instructions = dispatch_body["instructions"]
        assert isinstance(instructions, str)
        assert "one parallel batch" in instructions
        assert "at most 1800 characters" in instructions
        assert requests[1:] == [("approval", {"choice": "once"})]
        assert authorization
        assert set(authorization) == {f"Bearer {bearer}"}
    finally:
        approval_resolved.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_event_stream_outlives_request_timeout_until_exact_terminal_evidence() -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    events_connected = asyncio.Event()
    release_terminal = asyncio.Event()
    state: _RunState = {"status": "running", "stop_calls": 0}

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        events_connected.set()
        await release_terminal.wait()
        terminal = (
            'data: {"event":"run.completed",'
            f'"run_id":"{api_run_id}",'
            '"output":"Exact delayed result."}\n\n'
        ).encode()
        try:
            await response.write(terminal)
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    async def status_handler(_request: web.Request) -> web.Response:
        payload: dict[str, object] = {
            "run_id": api_run_id,
            "status": state["status"],
        }
        if state["status"] == "completed":
            payload["output"] = "Exact delayed result."
        return web.json_response(payload)

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        state["status"] = "cancelled"
        return web.json_response({"run_id": api_run_id, "status": "stopping"})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="long-sse-test-bearer-value-32-characters",
            request_timeout_seconds=0.1,
            settlement_timeout_seconds=0.5,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "delayed_sse_private",
    )
    update: asyncio.Task[object] | None = None
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        assert acknowledgment.payload.accepted is True
        await asyncio.wait_for(events_connected.wait(), timeout=1)

        update = asyncio.create_task(session.next_update())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(update), timeout=0.15)
        assert update.done() is False
        assert state == {"status": "running", "stop_calls": 0}

        state["status"] = "completed"
        release_terminal.set()
        terminal = await asyncio.wait_for(update, timeout=1)
        assert terminal.task_id == "task_release_check"
        assert terminal.run_id == "deleg_delayed_sse_private"
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == "Exact delayed result."
        assert state["stop_calls"] == 0
    finally:
        release_terminal.set()
        if update is not None and not update.done():
            update.cancel()
            await asyncio.gather(update, return_exceptions=True)
        try:
            await session.close()
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_api_session_signals_exact_active_run_and_rejects_duplicate_stop() -> None:
    port = _available_port()
    stopped = asyncio.Event()
    stop_calls = 0

    async def capabilities(_request: web.Request) -> web.Response:
        return web.json_response(_capabilities())

    async def create_run(_request: web.Request) -> web.Response:
        return web.json_response(
            {"run_id": "run_abcdef0123456789", "status": "started"},
            status=202,
        )

    async def run_events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        await stopped.wait()
        import json

        await response.write(
            (
                "data: "
                + json.dumps(
                    {
                        "event": "run.cancelled",
                        "run_id": "run_abcdef0123456789",
                        "timestamp": 2.0,
                    }
                )
                + "\n\n"
            ).encode()
        )
        await response.write_eof()
        return response

    async def stop_run(_request: web.Request) -> web.Response:
        nonlocal stop_calls
        stop_calls += 1
        stopped.set()
        return web.json_response({"run_id": "run_abcdef0123456789", "status": "stopping"})

    app = web.Application()
    app.router.add_get("/v1/capabilities", capabilities)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", stop_run)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="stop-test-bearer-value-32-characters",
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "api_private_stop",
    )

    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        assert acknowledgment.payload.accepted
        cancellation = await session.cancel(_cancel_request())
        assert cancellation.payload.accepted is True
        assert cancellation.payload.signaled_run_ids == ["deleg_api_private_stop"]
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert terminal.payload.status is WorkTerminalStatus.INTERRUPTED

        duplicate = await session.cancel(_cancel_request(sequence=5))
        assert duplicate.payload.accepted is False
        assert duplicate.payload.reason == "task is not active"
        assert stop_calls == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.parametrize(
    "failure_mode",
    [
        "malformed",
        "eof",
        "approval_observer",
        "duplicate_approval",
        "disclosive_terminal",
        "disclosive_nonhex_terminal",
    ],
)
@pytest.mark.asyncio
async def test_event_stream_failure_stops_and_confirms_terminal_authority(
    failure_mode: str,
) -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    state: _RunState = {"status": "running", "stop_calls": 0}

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if failure_mode == "malformed":
            await response.write(b"data: {\n\n")
        elif failure_mode in {"approval_observer", "duplicate_approval"}:
            approval = (
                'data: {"event":"approval.request",'
                f'"run_id":"{api_run_id}",'
                '"command":"chmod marker","description":"test",'
                '"choices":["deny","once"]}\n\n'
            ).encode()
            await response.write(approval)
            if failure_mode == "duplicate_approval":
                await response.write(approval)
        elif failure_mode in {"disclosive_terminal", "disclosive_nonhex_terminal"}:
            token = (
                "Xrun_0123456789abcdefY"
                if failure_mode == "disclosive_terminal"
                else "Xrun_0123456789abcdegY"
            )
            await response.write(
                (
                    'data: {"event":"run.completed",'
                    f'"run_id":"{api_run_id}","output":"{token}"}}\n\n'
                ).encode()
            )
        await response.write_eof()
        return response

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        state["status"] = "cancelled"
        return web.json_response({"run_id": api_run_id, "status": "stopping"})

    async def status_handler(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": api_run_id, "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    def approval_observer(_event: object) -> None:
        if failure_mode == "approval_observer":
            raise RuntimeError("projection unavailable")

    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="event-failure-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        approval_observer=approval_observer,
    )
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        terminal = await asyncio.wait_for(session.next_update(), timeout=2)
        assert terminal.payload.status is WorkTerminalStatus.INTERRUPTED
        assert state["stop_calls"] == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_failed_close_retains_run_authority_for_retry() -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    state: _RunState = {"status": "running", "stop_calls": 0}
    release_events = asyncio.Event()

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await release_events.wait()
        return response

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        if state["stop_calls"] == 1:
            return web.json_response({"error": "temporary"}, status=500)
        state["status"] = "cancelled"
        release_events.set()
        return web.json_response({"run_id": api_run_id, "status": "stopping"})

    async def status_handler(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": api_run_id, "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="close-retry-test-bearer-value-32-characters",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
    )
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        with pytest.raises(BaseExceptionGroup, match="unresolved run authority"):
            await session.close()
        await session.close()
        assert state["stop_calls"] == 2
    finally:
        release_events.set()
        await runner.cleanup()


@pytest.mark.parametrize(
    ("terminal_body", "expected_status", "expected_summary", "expected_reason"),
    [
        (
            {"status": "completed", "output": "Finished before the stop."},
            WorkTerminalStatus.COMPLETED,
            "Finished before the stop.",
            None,
        ),
        (
            {"status": "cancelled"},
            WorkTerminalStatus.INTERRUPTED,
            None,
            "Hermes run was interrupted",
        ),
        (
            # Hermes rewrites a run whose gateway restarted mid-run as interrupted.
            {"status": "interrupted"},
            WorkTerminalStatus.INTERRUPTED,
            None,
            "Hermes run was interrupted",
        ),
    ],
)
@pytest.mark.asyncio
async def test_close_settles_run_that_finished_before_its_stop(
    terminal_body: dict[str, str],
    expected_status: WorkTerminalStatus,
    expected_summary: str | None,
    expected_reason: str | None,
) -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    state: _RunState = {"status": "running", "stop_calls": 0}
    release_events = asyncio.Event()

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await release_events.wait()
        return response

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        return web.json_response({"object": "hermes.run", "run_id": api_run_id, **terminal_body})

    async def status_handler(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": api_run_id, "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="terminal-stop-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "terminal_stop_private",
    )
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        await session.close()
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert terminal.type == "work.completed"
        assert terminal.task_id == "task_release_check"
        assert terminal.run_id == "deleg_terminal_stop_private"
        assert terminal.payload.status is expected_status
        assert terminal.payload.summary == expected_summary
        assert terminal.payload.reason == expected_reason
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session.next_update(), timeout=0.05)
        assert state["stop_calls"] == 1
    finally:
        release_events.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_event_stream_failure_settles_run_that_failed_before_its_stop() -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    state: _RunState = {"status": "running", "stop_calls": 0}

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write_eof()
        return response

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        return web.json_response(
            {
                "object": "hermes.run",
                "run_id": api_run_id,
                "status": "failed",
                "error": "Tool crashed before the stop.",
            }
        )

    async def status_handler(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": api_run_id, "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="terminal-stop-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "failed_stop_private",
    )
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        terminal = await asyncio.wait_for(session.next_update(), timeout=2)
        assert terminal.task_id == "task_release_check"
        assert terminal.run_id == "deleg_failed_stop_private"
        assert terminal.payload.status is WorkTerminalStatus.FAILED
        assert terminal.payload.reason == "Tool crashed before the stop."
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session.next_update(), timeout=0.05)
        assert state["stop_calls"] == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.parametrize(
    "stop_body",
    [
        {
            "object": "hermes.run",
            "run_id": "run_ffffffffffffffff",
            "status": "completed",
            "output": "Another run finished.",
        },
        {"object": "hermes.run", "run_id": "run_0123456789abcdef", "status": "running"},
        # A terminal status for this run is not evidence unless the body is a run status
        # object; a stop 200 has two shapes, and the discriminator is what tells them apart.
        {"run_id": "run_0123456789abcdef", "status": "completed", "output": "Unlabelled."},
        {
            "object": "hermes.run.steer",
            "run_id": "run_0123456789abcdef",
            "status": "completed",
            "output": "Misrouted.",
        },
    ],
)
@pytest.mark.asyncio
async def test_close_rejects_nonauthoritative_stop_status_body(
    stop_body: dict[str, str],
) -> None:
    port = _available_port()
    api_run_id = "run_0123456789abcdef"
    state: _RunState = {"status": "running", "stop_calls": 0}
    release_events = asyncio.Event()

    async def events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await release_events.wait()
        return response

    async def stop(_request: web.Request) -> web.Response:
        state["stop_calls"] += 1
        if state["stop_calls"] == 1:
            return web.json_response(stop_body)
        state["status"] = "cancelled"
        release_events.set()
        return web.json_response({"run_id": api_run_id, "status": "stopping"})

    async def status_handler(_request: web.Request) -> web.Response:
        return web.json_response({"run_id": api_run_id, "status": state["status"]})

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", _started_run_response)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", status_handler)
    app.router.add_post("/v1/runs/{run_id}/stop", stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="terminal-stop-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
    )
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        with pytest.raises(BaseExceptionGroup, match="unresolved run authority") as raised:
            await session.close()
        assert raised.group_contains(RuntimeError, match="stop response is not authoritative")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(session.next_update(), timeout=0.05)

        await session.close()
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        assert terminal.payload.status is WorkTerminalStatus.INTERRUPTED
        assert state["stop_calls"] == 2
    finally:
        release_events.set()
        await runner.cleanup()


_ABSENT = object()
_DURABLE_IDEMPOTENCY = {"supported": True, "durable": True, "retention_seconds": 86400}


class _IdempotentHermes:
    """A fake Hermes whose /v1/runs keeps idempotency records the way v0.21.0 does.

    ``first_attempt`` decides what happens to the first POST: ``answer`` responds,
    ``lose_response`` admits the run and then drops the connection before responding,
    ``never_arrive`` drops the connection before admitting, ``malformed`` answers with a
    body that is not JSON, ``legacy`` answers without the replay marker, and ``refuse``
    answers 429. ``lose_resend`` drops the resend's connection too.
    """

    run_id = "run_abcdefabcdefabcd"

    def __init__(
        self,
        first_attempt: str,
        *,
        idempotency: object = _DURABLE_IDEMPOTENCY,
        resend_status: int = 202,
        resend_text: str | None = None,
        replay_status: str = "running",
        run_status: str = "running",
        events_status: int = 200,
        hold_first_attempt: bool = False,
        lose_resend: bool = False,
        first_replayed: object = False,
        on_post: Callable[[], None] | None = None,
        release_events: asyncio.Event | None = None,
        run_known: bool = True,
        status_http: int = 200,
        answer_status: int = 202,
    ) -> None:
        self.answer_status = answer_status
        self.on_post = on_post
        self.release_events = release_events
        self.run_known = run_known
        self.status_http = status_http
        self.first_attempt = first_attempt
        self.lose_resend = lose_resend
        self.first_replayed = first_replayed
        self.idempotency = idempotency
        self.resend_status = resend_status
        self.resend_text = resend_text
        self.replay_status = replay_status
        self.run_status = run_status
        self.events_status = events_status
        self.hold_first_attempt = hold_first_attempt
        self.first_attempt_admitted = asyncio.Event()
        self.release_first_attempt = asyncio.Event()
        self.attempts: list[tuple[str | None, bytes]] = []
        self.records: dict[str, str] = {}
        self.runs_created = 0
        self.stop_calls = 0

    async def capabilities(self, _request: web.Request) -> web.Response:
        self.capability_calls = getattr(self, "capability_calls", 0) + 1
        payload = _capabilities()
        features = payload["features"]
        assert isinstance(features, dict)
        if self.idempotency is not _ABSENT:
            features["runs_idempotency"] = self.idempotency
        return web.json_response(payload)

    async def runs(self, request: web.Request) -> web.StreamResponse:
        if self.on_post is not None:
            self.on_post()
        raw = await request.read()
        key = request.headers.get("Idempotency-Key")
        self.attempts.append((key, raw))
        first = len(self.attempts) == 1
        if first and self.first_attempt == "never_arrive":
            return self._drop(request)
        if first and self.first_attempt == "malformed":
            return web.Response(status=202, text="not json", content_type="application/json")
        if first and self.first_attempt == "refuse":
            return web.json_response({"error": "busy"}, status=429)
        if not first and self.lose_resend:
            return self._drop(request)
        if not first and self.resend_status != 202:
            return web.json_response({"error": "unavailable"}, status=self.resend_status)
        if not first and self.resend_text is not None:
            return web.Response(
                status=202, text=self.resend_text, content_type="application/json"
            )
        # Hermes fingerprints the parsed body, not its bytes.
        fingerprint = json.dumps(json.loads(raw), sort_keys=True)
        if key is not None and key in self.records:
            if self.records[key] != fingerprint:
                return web.json_response({"error": "conflict"}, status=409)
            return web.json_response(
                {"run_id": self.run_id, "status": self.replay_status, "replayed": True},
                status=202,
                headers={"Idempotency-Replayed": "true"},
            )
        self.runs_created += 1
        if key is not None:
            self.records[key] = fingerprint
        if first and self.first_attempt == "lose_response":
            self.first_attempt_admitted.set()
            if self.hold_first_attempt:
                await self.release_first_attempt.wait()
            return self._drop(request)
        if first and self.first_attempt == "legacy":
            return web.json_response({"run_id": self.run_id, "status": "started"}, status=202)
        return web.json_response(
            {"run_id": self.run_id, "status": "started", "replayed": self.first_replayed},
            status=self.answer_status,
        )

    @staticmethod
    def _drop(request: web.Request) -> web.Response:
        transport = request.transport
        assert transport is not None
        transport.close()
        return web.Response(status=202)

    async def events(self, request: web.Request) -> web.StreamResponse:
        if self.events_status != 200 or request.match_info["run_id"] != self.run_id:
            return web.json_response({"error": "not found"}, status=self.events_status)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if self.release_events is not None:
            await self.release_events.wait()
        terminal = {"event": "run.completed", "run_id": self.run_id, "output": "Recovered."}
        await response.write(b"data: " + json.dumps(terminal).encode() + b"\n\n")
        return response

    @staticmethod
    def _not_found() -> web.Response:
        # Hermes v0.21.0 answers an unknown run with this OpenAI-style envelope.
        error = {"message": "Run not found", "type": "invalid_request_error"}
        return web.json_response(
            {"error": error | {"param": None, "code": "run_not_found"}}, status=404
        )

    async def status(self, _request: web.Request) -> web.Response:
        if not self.run_known:
            return self._not_found()
        if self.status_http != 200:
            return web.json_response({"error": "unavailable"}, status=self.status_http)
        return web.json_response({"run_id": self.run_id, "status": self.run_status})

    async def stop(self, _request: web.Request) -> web.Response:
        if not self.run_known:
            return self._not_found()
        self.stop_calls += 1
        self.run_status = "cancelled"
        return web.json_response({"run_id": self.run_id, "status": "stopping"})

    async def serve(self) -> tuple[web.AppRunner, int]:
        port = _available_port()
        app = web.Application()
        app.router.add_get("/v1/capabilities", self.capabilities)
        app.router.add_post("/v1/runs", self.runs)
        app.router.add_get("/v1/runs/{run_id}/events", self.events)
        app.router.add_get("/v1/runs/{run_id}", self.status)
        app.router.add_post("/v1/runs/{run_id}/stop", self.stop)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        return runner, port


def _recovery_markers(capsys: pytest.CaptureFixture[str]) -> list[object]:
    prefix = "[hermes-dispatch-recovery] "
    lines = capsys.readouterr().out.splitlines()
    return [json.loads(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]


def _recovery_session(port: int) -> HermesApiTaskSession:
    return HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="dispatch-recovery-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "recovery_private",
    )


@pytest.mark.asyncio
async def test_a_lost_dispatch_response_recovers_the_same_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes = _IdempotentHermes("lose_response")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)

        assert acknowledgment.payload.accepted is True
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == "Recovered."
        assert hermes.runs_created == 1
        assert hermes.stop_calls == 0
        (first_key, first_body), (second_key, second_body) = hermes.attempts
        assert first_key is not None and second_key == first_key
        assert second_body == first_body
        assert _recovery_markers(capsys) == []
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_dispatch_that_never_arrived_is_admitted_by_its_resend() -> None:
    hermes = _IdempotentHermes("never_arrive")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        await asyncio.wait_for(session.next_update(), timeout=1)

        assert acknowledgment.payload.accepted is True
        assert hermes.runs_created == 1
        assert len(hermes.attempts) == 2
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("resend_status", [500, 503, 429])
async def test_an_ambiguous_dispatch_never_reports_a_rejection(
    resend_status: int, capsys: pytest.CaptureFixture[str]
) -> None:
    # A refused resend says nothing about the first attempt, which Hermes admitted.
    hermes = _IdempotentHermes("lose_response", resend_status=resend_status)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            await session.dispatch(_dispatch_request())
        assert hermes.runs_created == 1
        assert _recovery_markers(capsys) == [
            {"cause": "refused", "status": resend_status, "version": 1}
        ]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "idempotency",
    [
        pytest.param(_ABSENT, id="not-advertised"),
        pytest.param(
            {"supported": True, "durable": False, "retention_seconds": 86400},
            id="not-durable",
        ),
        pytest.param(
            {"supported": False, "durable": True, "retention_seconds": 86400},
            id="not-supported",
        ),
    ],
)
async def test_without_durable_idempotency_a_dispatch_is_sent_once(idempotency: object) -> None:
    hermes = _IdempotentHermes("lose_response", idempotency=idempotency)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="request failed"):
            await session.dispatch(_dispatch_request())
        assert hermes.attempts == [(None, hermes.attempts[0][1])]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_an_abandoned_dispatch_still_recovers_and_stops_its_run() -> None:
    # Only the resend can reveal the run the lost attempt started, so it must still happen.
    hermes = _IdempotentHermes("lose_response", hold_first_attempt=True)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        dispatch = asyncio.create_task(session.dispatch(_dispatch_request()))
        await asyncio.wait_for(hermes.first_attempt_admitted.wait(), timeout=1)
        dispatch.cancel()
        hermes.release_first_attempt.set()
        with pytest.raises(asyncio.CancelledError):
            await dispatch

        assert len(hermes.attempts) == 2
        assert hermes.runs_created == 1
        assert hermes.stop_calls == 1
        assert session._unpublished_api_run_ids == set()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_replayed_interrupted_run_settles_as_interrupted() -> None:
    # After a Hermes restart the replay reports the lost run as interrupted, with no stream.
    hermes = _IdempotentHermes(
        "lose_response",
        replay_status="interrupted",
        run_status="interrupted",
        events_status=404,
    )
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)

        assert acknowledgment.payload.accepted is True
        assert terminal.payload.status is WorkTerminalStatus.INTERRUPTED
        assert terminal.payload.reason == "Hermes run was interrupted"
        assert hermes.stop_calls == 0
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_malformed_dispatch_response_is_never_resent() -> None:
    # A response arrived, so nothing was lost: only a missing response justifies a resend.
    hermes = _IdempotentHermes("malformed")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="malformed JSON"):
            await session.dispatch(_dispatch_request())
        assert len(hermes.attempts) == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_an_idempotent_dispatch_requires_the_replay_marker() -> None:
    hermes = _IdempotentHermes("legacy")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="malformed"):
            await session.dispatch(_dispatch_request())
        assert hermes.stop_calls == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_refused_first_attempt_is_a_rejection() -> None:
    # Nothing was lost, so a refusal is a truthful answer about this dispatch.
    hermes = _IdempotentHermes("refuse")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        assert acknowledgment.payload.accepted is False
        assert acknowledgment.payload.reason == "Hermes API rejected dispatch"
        assert len(hermes.attempts) == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_dispatch_whose_resend_is_also_lost_has_an_unknown_outcome(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes = _IdempotentHermes("lose_response", lose_resend=True)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            await session.dispatch(_dispatch_request())
        assert len(hermes.attempts) == 2
        assert hermes.runs_created == 1
        assert _recovery_markers(capsys) == [{"cause": "no_response", "status": None, "version": 1}]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_status", ["started", "unknown"])
async def test_a_replay_must_report_a_status_hermes_holds(replay_status: str) -> None:
    hermes = _IdempotentHermes("lose_response", replay_status=replay_status)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="lacks exact authority"):
            await session.dispatch(_dispatch_request())
        assert hermes.stop_calls == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("replayed", [None, "true", 0])
async def test_an_idempotent_dispatch_requires_an_exact_replay_marker(replayed: object) -> None:
    hermes = _IdempotentHermes("answer", first_replayed=replayed)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="malformed"):
            await session.dispatch(_dispatch_request())
        assert hermes.stop_calls == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_resend_older_than_the_retention_window_is_never_sent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Past retention Hermes may have pruned the key and would start the work again.
    hermes = _IdempotentHermes(
        "lose_response",
        idempotency={"supported": True, "durable": True, "retention_seconds": 1},
        hold_first_attempt=True,
    )
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        dispatch = asyncio.create_task(session.dispatch(_dispatch_request()))
        await asyncio.wait_for(hermes.first_attempt_admitted.wait(), timeout=1)
        await asyncio.sleep(1.1)
        hermes.release_first_attempt.set()
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            await dispatch
        assert len(hermes.attempts) == 1
        assert _recovery_markers(capsys) == [{"cause": "expired", "status": None, "version": 1}]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resend_text", "cause", "status"),
    [
        pytest.param("{}", "unidentified", 202, id="no-run-id"),
        pytest.param(
            '{"run_id": "not-a-run", "status": "running"}', "unidentified", 202, id="bad-id"
        ),
        # The body never parsed, so no status was ever returned to the adapter.
        pytest.param("not json", "error", None, id="not-json"),
    ],
)
async def test_an_unusable_resend_admission_leaves_the_outcome_unknown(
    resend_text: str, cause: str, status: int | None, capsys: pytest.CaptureFixture[str]
) -> None:
    # Recovery needs a run it can name; anything less cannot stop or track the lost run.
    hermes = _IdempotentHermes("lose_response", resend_text=resend_text)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            await session.dispatch(_dispatch_request())
        assert _recovery_markers(capsys) == [{"cause": cause, "status": status, "version": 1}]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [[], {}], ids=["list", "object"])
async def test_an_unhashable_admission_status_still_stops_the_run(status: object) -> None:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8765",
            bearer="unhashable-status-test-bearer-value-32-chars",
        ),
        session_id="session_unhashable_status",
    )
    stopped: list[str] = []

    async def request_json(
        _method: str,
        _path: str,
        *,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        del body
        return 202, {"run_id": "run_3434343434343434", "status": status}

    async def stop_and_wait(api_run_id: str) -> None:
        stopped.append(api_run_id)

    session._started = True
    session._request_json = request_json  # type: ignore[method-assign]
    session._stop_and_wait = stop_and_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="lacks exact authority"):
        await session.dispatch(_dispatch_request())
    assert stopped == ["run_3434343434343434"]
    await session.close()


@pytest.mark.asyncio
async def test_close_waits_for_a_dispatch_whose_caller_was_cancelled_twice() -> None:
    # The second cancellation releases the caller early; close must still await the work.
    hermes = _IdempotentHermes("lose_response", hold_first_attempt=True)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        dispatch = asyncio.create_task(session.dispatch(_dispatch_request()))
        await asyncio.wait_for(hermes.first_attempt_admitted.wait(), timeout=1)
        dispatch.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        hermes.release_first_attempt.set()
        await asyncio.wait_for(closing, timeout=5)

        assert len(hermes.attempts) == 2
        assert hermes.stop_calls == 1
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "idempotency",
    [
        pytest.param(None, id="null"),
        pytest.param("yes", id="not-an-object"),
        pytest.param({"supported": True, "durable": True}, id="retention-missing"),
        pytest.param(
            {"supported": True, "durable": True, "retention_seconds": 0}, id="retention-zero"
        ),
        pytest.param(
            {"supported": True, "durable": True, "retention_seconds": "86400"},
            id="retention-not-int",
        ),
        pytest.param({"supported": True}, id="durable-missing"),
        pytest.param({"supported": True, "durable": "true"}, id="durable-not-bool"),
        pytest.param({"durable": True}, id="supported-missing"),
        pytest.param({"supported": "true", "durable": True}, id="supported-not-bool"),
    ],
)
async def test_a_malformed_idempotency_capability_fails_start(idempotency: object) -> None:
    hermes = _IdempotentHermes("answer", idempotency=idempotency)
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        with pytest.raises(RuntimeError, match="idempotency capability is malformed"):
            await session.start()
    finally:
        await session.close()
        await runner.cleanup()


_RECORD_RUN_ID = _IdempotentHermes.run_id
_RECORD_KEY = "0123456789abcdef0123456789abcdef"
_RECORD_REQUEST: dict[str, object] = {
    "input": "Inspect the release evidence",
    "instructions": "Stay within the requested scope.",
}
_RESTART_PREFIX = "[hermes-restart-settlement] "


def _record_session(port: int, record: Path) -> HermesApiTaskSession:
    return HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer="dispatch-recovery-test-bearer-value-32-chars",
            settlement_timeout_seconds=1,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_api_1",
        private_id_factory=lambda: "recovery_private",
        run_record_path=record,
    )


def _read_record(record: Path) -> object:
    return json.loads(record.read_bytes())


def _write_record(
    record: Path,
    *,
    pending: list[dict[str, object]] | None = None,
    admitted: list[str] | None = None,
) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    document = {"admitted": admitted or [], "pending": pending or [], "version": 1}
    record.write_text(json.dumps(document), encoding="utf-8")


def _empty_record() -> dict[str, object]:
    return {"admitted": [], "pending": [], "version": 1}


def _restart_markers(capsys: pytest.CaptureFixture[str]) -> list[object]:
    lines = capsys.readouterr().out.splitlines()
    return [
        json.loads(line.removeprefix(_RESTART_PREFIX))
        for line in lines
        if line.startswith(_RESTART_PREFIX)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("idempotency", "keyed"),
    [
        pytest.param(_DURABLE_IDEMPOTENCY, True, id="keyed"),
        # Without durable idempotency no key is sent, so none is recorded.
        pytest.param(_ABSENT, False, id="keyless"),
    ],
)
async def test_a_dispatch_is_recorded_as_pending_before_its_post(
    idempotency: object, keyed: bool, tmp_path: Path
) -> None:
    record = tmp_path / "state" / "hermes-runs-v1.json"
    observed: list[object] = []
    hermes = _IdempotentHermes(
        "answer",
        idempotency=idempotency,
        on_post=lambda: observed.append(_read_record(record)),
    )
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        before = time.time()
        acknowledgment = await session.dispatch(_dispatch_request())

        assert acknowledgment.payload.accepted is True
        [(key, raw)] = hermes.attempts
        assert (key is not None) is keyed
        [snapshot] = observed
        assert type(snapshot) is dict
        [entry] = snapshot["pending"]
        assert snapshot == {"admitted": [], "pending": [entry], "version": 1}
        assert entry == {"key": key, "minted_at": entry["minted_at"], "request": json.loads(raw)}
        assert type(entry["minted_at"]) is float
        assert before <= entry["minted_at"] <= time.time()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_admission_promotes_the_pending_entry_to_its_run_id(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())

        assert acknowledgment.payload.accepted is True
        assert _read_record(record) == {
            "admitted": [_RECORD_RUN_ID],
            "pending": [],
            "version": 1,
        }
        # The acknowledgment drops the request: no transcript or private authority remains.
        assert b"Inspect the release evidence" not in record.read_bytes()
        assert b"deleg_" not in record.read_bytes()
    finally:
        release.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_terminal_settlement_removes_the_admitted_entry(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        terminal = await asyncio.wait_for(session.next_update(), timeout=1)
        await session.close()

        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert hermes.stop_calls == 0
        assert _read_record(record) == _empty_record()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_close_removes_the_entry_of_the_run_it_stops(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        await session.close()

        assert hermes.stop_calls == 1
        assert _read_record(record) == _empty_record()
    finally:
        release.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_rejected_dispatch_leaves_no_entry(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    observed: list[object] = []
    hermes = _IdempotentHermes("refuse", on_post=lambda: observed.append(_read_record(record)))
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())

        assert acknowledgment.payload.accepted is False
        [snapshot] = observed
        assert type(snapshot) is dict and len(snapshot["pending"]) == 1
        assert _read_record(record) == _empty_record()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_dispatch_with_an_unknown_outcome_stays_pending_for_restart(
    tmp_path: Path,
) -> None:
    # The lost attempt may have started a run; only a restart can still find and stop it.
    record = tmp_path / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("lose_response", lose_resend=True)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        with pytest.raises(RuntimeError, match="outcome is unknown"):
            await session.dispatch(_dispatch_request())
        await session.close()

        snapshot = _read_record(record)
        assert type(snapshot) is dict and snapshot["admitted"] == []
        [entry] = snapshot["pending"]
        assert entry["key"] == hermes.attempts[0][0]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_unknown_dispatches_hold_record_capacity_until_restart(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("lose_response", lose_resend=True)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        for index in range(8):
            with pytest.raises(RuntimeError, match="outcome is unknown"):
                await session.dispatch(_dispatch_request(task_id=f"task_unknown_{index}"))
        attempts = len(hermes.attempts)

        acknowledgment = await session.dispatch(_dispatch_request(task_id="task_over_capacity"))

        assert acknowledgment.payload.accepted is False
        assert acknowledgment.payload.reason == "background task capacity exhausted"
        assert len(hermes.attempts) == attempts
        snapshot = _read_record(record)
        assert type(snapshot) is dict and len(snapshot["pending"]) == 8
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_dispatch_whose_pending_entry_cannot_be_written_is_never_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        def refuse(_path: Path, _data: bytes) -> None:
            raise OSError("record volume is full")

        monkeypatch.setattr(run_record_module, "write_run_record", refuse)
        with pytest.raises(OSError, match="record volume is full"):
            await session.dispatch(_dispatch_request())
        assert hermes.attempts == []

        monkeypatch.undo()
        acknowledgment = await session.dispatch(_dispatch_request())
        assert acknowledgment.payload.accepted is True
        # The unsent dispatch left nothing behind in the record.
        assert _read_record(record) == {
            "admitted": [_RECORD_RUN_ID],
            "pending": [],
            "version": 1,
        }
    finally:
        release.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_run_whose_promotion_cannot_be_written_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A keyless pending entry cannot recover its run after a crash, so an unrecorded run stops.
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    writes: list[bytes] = []
    original = run_record_module.write_run_record

    def fail_promotion(path: Path, data: bytes) -> None:
        writes.append(data)
        if len(writes) == 2:
            raise OSError("record volume is full")
        original(path, data)

    monkeypatch.setattr(run_record_module, "write_run_record", fail_promotion)
    try:
        await session.start()
        with pytest.raises(OSError, match="record volume is full"):
            await session.dispatch(_dispatch_request())

        assert hermes.stop_calls == 1
        assert session._runs_by_task == {}
        assert _read_record(record) == _empty_record()
    finally:
        release.set()
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_restart_stops_an_admitted_run_once_and_empties_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(record, admitted=[_RECORD_RUN_ID])
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert hermes.stop_calls == 1
        assert hermes.attempts == []
        assert _read_record(record) == _empty_record()
        assert session.restart_settlement == HermesRestartSettlement(stopped=1, unknown=0)
        assert _restart_markers(capsys) == [
            {"stopped": 1, "unknown": 0, "unresolved": 0, "version": 1}
        ]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_restart_counts_a_run_hermes_no_longer_knows_as_ended(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(record, admitted=[_RECORD_RUN_ID])
    hermes = _IdempotentHermes("answer", run_known=False)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert hermes.stop_calls == 0
        assert _read_record(record) == _empty_record()
        assert session.restart_settlement == HermesRestartSettlement(stopped=1, unknown=0)
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_restart_treats_an_unexplained_not_found_as_unresolved(tmp_path: Path) -> None:
    # Only Hermes's own run_not_found answer shows the run is gone; any other 404 proves nothing.
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(record, admitted=[_RECORD_RUN_ID])
    hermes = _IdempotentHermes("answer", status_http=404)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        with pytest.raises(BaseExceptionGroup, match="restart settlement left runs unresolved"):
            await session.start()
        assert _read_record(record) == {"admitted": [_RECORD_RUN_ID], "pending": [], "version": 1}
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_dispatch_naming_a_retired_run_leaves_no_entry(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()
        await session.dispatch(_dispatch_request())
        await asyncio.wait_for(session.next_update(), timeout=1)
        with pytest.raises(RuntimeError, match="reused a run authority"):
            await session.dispatch(_dispatch_request(task_id="task_second"))

        assert _read_record(record) == _empty_record()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_restart_with_no_record_settles_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "missing" / "hermes-runs-v1.json"
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert session.restart_settlement == HermesRestartSettlement(stopped=0, unknown=0)
        assert not record.exists()
        assert _restart_markers(capsys) == []
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("arrived", [True, False], ids=["accepted-before-crash", "never-arrived"])
async def test_restart_replays_a_young_pending_entry_then_stops_its_run(
    arrived: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(
        record,
        pending=[{"key": _RECORD_KEY, "minted_at": time.time() - 60, "request": _RECORD_REQUEST}],
    )
    hermes = _IdempotentHermes("answer")
    if arrived:
        hermes.records[_RECORD_KEY] = json.dumps(_RECORD_REQUEST, sort_keys=True)
        hermes.runs_created = 1
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        [(key, raw)] = hermes.attempts
        assert key == _RECORD_KEY
        assert json.loads(raw) == _RECORD_REQUEST
        assert hermes.runs_created == 1
        assert hermes.stop_calls == 1
        assert _read_record(record) == _empty_record()
        assert session.restart_settlement == HermesRestartSettlement(stopped=1, unknown=0)
        assert _restart_markers(capsys) == [
            {"stopped": 1, "unknown": 0, "unresolved": 0, "version": 1}
        ]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_key", "age", "idempotency"),
    [
        pytest.param(None, 60.0, _DURABLE_IDEMPOTENCY, id="keyless"),
        # Past retention Hermes may have pruned the key and would start the work again.
        pytest.param(_RECORD_KEY, 86400.0 + 60, _DURABLE_IDEMPOTENCY, id="expired"),
        pytest.param(_RECORD_KEY, -3600.0, _DURABLE_IDEMPOTENCY, id="minted-in-the-future"),
        pytest.param(_RECORD_KEY, 60.0, _ABSENT, id="no-durable-idempotency"),
    ],
)
async def test_restart_never_replays_a_pending_entry_it_cannot_replay_safely(
    entry_key: str | None,
    age: float,
    idempotency: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(
        record,
        pending=[{"key": entry_key, "minted_at": time.time() - age, "request": _RECORD_REQUEST}],
    )
    hermes = _IdempotentHermes("answer", idempotency=idempotency)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert hermes.attempts == []
        assert hermes.stop_calls == 0
        assert _read_record(record) == _empty_record()
        assert session.restart_settlement == HermesRestartSettlement(stopped=0, unknown=1)
        assert _restart_markers(capsys) == [
            {"stopped": 0, "unknown": 1, "unresolved": 0, "version": 1}
        ]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replay", "answer_status", "run_id"),
    [
        pytest.param("refuse", 202, _RECORD_RUN_ID, id="refused"),
        pytest.param("never_arrive", 202, _RECORD_RUN_ID, id="unanswered"),
        pytest.param("malformed", 202, _RECORD_RUN_ID, id="malformed"),
        pytest.param("answer", 202, "run_short", id="inexact-run"),
        pytest.param("answer", 200, _RECORD_RUN_ID, id="not-an-admission"),
    ],
)
async def test_restart_counts_a_refused_or_unanswered_replay_as_unknown(
    replay: str, answer_status: int, run_id: str, tmp_path: Path
) -> None:
    # Only a 202 naming an exact run identifies what the lost attempt started; nothing is resent.
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(
        record,
        pending=[{"key": _RECORD_KEY, "minted_at": time.time() - 60, "request": _RECORD_REQUEST}],
    )
    hermes = _IdempotentHermes(replay, answer_status=answer_status)
    hermes.run_id = run_id
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert len(hermes.attempts) == 1
        assert hermes.stop_calls == 0
        assert _read_record(record) == _empty_record()
        assert session.restart_settlement == HermesRestartSettlement(stopped=0, unknown=1)
    finally:
        await session.close()
        await runner.cleanup()


_VALID_PENDING: dict[str, object] = {
    "key": _RECORD_KEY,
    "minted_at": 1.5,
    "request": _RECORD_REQUEST,
}


def _record_text(**fields: object) -> str:
    document: dict[str, object] = {"admitted": [], "pending": [], "version": 1}
    document.update(fields)
    return json.dumps(document)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param(_record_text(version=2), id="unknown-version"),
        pytest.param(_record_text(version=True), id="version-not-int"),
        pytest.param(_record_text(extra=1), id="extra-field"),
        pytest.param(json.dumps({"admitted": [], "version": 1}), id="missing-field"),
        pytest.param(_record_text(admitted=["deleg_recovery_private"]), id="private-id"),
        pytest.param(_record_text(admitted=["run_short"]), id="inexact-run-id"),
        pytest.param(_record_text(admitted=[_RECORD_RUN_ID] * 2), id="duplicate-run"),
        pytest.param(
            _record_text(admitted=[f"run_{index:016x}" for index in range(9)]), id="over-capacity"
        ),
        pytest.param(
            _record_text(
                admitted=[f"run_{index:016x}" for index in range(8)], pending=[_VALID_PENDING]
            ),
            id="over-capacity-combined",
        ),
        pytest.param(_record_text(pending=[_VALID_PENDING] * 2), id="duplicate-key"),
        pytest.param(_record_text(pending=[_VALID_PENDING | {"key": "ABC"}]), id="bad-key"),
        pytest.param(
            _record_text(pending=[_VALID_PENDING | {"minted_at": "1.5"}]), id="time-not-number"
        ),
        pytest.param(_record_text(pending=[_VALID_PENDING | {"minted_at": 2}]), id="time-int"),
        pytest.param(_record_text(pending=[_VALID_PENDING | {"minted_at": True}]), id="time-bool"),
        pytest.param(_record_text(pending=[_VALID_PENDING]).replace("1.5", "NaN"), id="time-nan"),
        pytest.param(_record_text(pending=[_VALID_PENDING | {"request": []}]), id="request-list"),
        pytest.param(
            _record_text(pending=[_VALID_PENDING | {"request": {"input": "Inspect"}}]),
            id="request-incomplete",
        ),
        pytest.param(
            _record_text(
                pending=[_VALID_PENDING | {"request": _RECORD_REQUEST | {"session_id": "voice"}}]
            ),
            id="request-extra-field",
        ),
        pytest.param(
            _record_text(pending=[_VALID_PENDING | {"objective": "Inspect"}]), id="entry-extra"
        ),
        pytest.param('{"admitted": [], "admitted": [], "pending": [], "version": 1}', id="dup"),
        pytest.param(_record_text() + " " * api_module._MAX_RUN_RECORD_BYTES, id="over-size"),
    ],
)
async def test_a_malformed_run_record_fails_start(
    text: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    record.write_text(text, encoding="utf-8")
    original = record.read_bytes()
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        with pytest.raises(RuntimeError, match="run record is malformed"):
            await session.start()

        assert hermes.attempts == []
        assert hermes.stop_calls == 0
        assert record.read_bytes() == original
        # The refusal leaves one bounded, content-free piece of evidence.
        assert _restart_markers(capsys) == [{"refusal": "malformed", "version": 1}]
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_failed_restart_settlement_keeps_its_entry_and_fails_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(
        record,
        admitted=[_RECORD_RUN_ID],
        pending=[{"key": None, "minted_at": time.time() - 60, "request": _RECORD_REQUEST}],
    )
    hermes = _IdempotentHermes("answer", status_http=500)
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    retry = _record_session(port, record)
    try:
        with pytest.raises(BaseExceptionGroup, match="restart settlement left runs unresolved"):
            await session.start()

        # The settled entry is gone; only the unsettled run is kept for the next start.
        assert _read_record(record) == {"admitted": [_RECORD_RUN_ID], "pending": [], "version": 1}
        assert _restart_markers(capsys) == [
            {"stopped": 0, "unknown": 1, "unresolved": 1, "version": 1}
        ]
        with pytest.raises(RuntimeError, match="not started"):
            await session.dispatch(_dispatch_request())

        hermes.status_http = 200
        await retry.start()
        assert hermes.stop_calls == 1
        assert _read_record(record) == _empty_record()
        assert retry.restart_settlement == HermesRestartSettlement(stopped=1, unknown=0)
    finally:
        await session.close()
        await retry.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_without_a_run_record_nothing_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_path: Path, _data: bytes) -> None:
        raise AssertionError("a session without a run record must never write one")

    monkeypatch.setattr(run_record_module, "write_run_record", forbidden)
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _recovery_session(port)
    try:
        await session.start()
        acknowledgment = await session.dispatch(_dispatch_request())
        await asyncio.wait_for(session.next_update(), timeout=1)
        await session.close()

        assert acknowledgment.payload.accepted is True
        assert session.restart_settlement is None
        assert list(tmp_path.iterdir()) == []
    finally:
        await session.close()
        await runner.cleanup()


_LOCK_PREFIX = "[hermes-run-record-lock] "


@pytest.mark.asyncio
async def test_a_second_host_cannot_take_a_live_hosts_run_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A second host must not settle, and so stop, the runs a live host still owns.
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    first = _record_session(port, record)
    second = _record_session(port, record)
    third = _record_session(port, record)
    try:
        await first.start()
        await first.dispatch(_dispatch_request())
        owned = record.read_bytes()
        capability_calls = hermes.capability_calls
        capsys.readouterr()

        with pytest.raises(RuntimeError, match="another host holds the Hermes run record"):
            await second.start()

        assert hermes.capability_calls == capability_calls
        assert hermes.stop_calls == 0
        assert record.read_bytes() == owned
        lines = capsys.readouterr().out.splitlines()
        assert [line for line in lines if line.startswith(_LOCK_PREFIX)] == [
            _LOCK_PREFIX + '{"cause":"held","version":1}'
        ]

        await first.close()
        await third.start()
        assert hermes.stop_calls == 1
    finally:
        release.set()
        await first.close()
        await second.close()
        await third.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_failed_start_releases_the_run_record(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    record.write_text("not json", encoding="utf-8")
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    failed = _record_session(port, record)
    retry = _record_session(port, record)
    try:
        with pytest.raises(RuntimeError, match="run record is malformed"):
            await failed.start()
        _write_record(record)

        # The failed session was never closed, yet the record is free again.
        await retry.start()
        assert retry.restart_settlement == HermesRestartSettlement(stopped=0, unknown=0)
    finally:
        await failed.close()
        await retry.close()
        await runner.cleanup()


@pytest.mark.parametrize("path", ["hermes-runs-v1.json", b"hermes-runs-v1.json", 1])
def test_run_record_path_must_be_an_exact_path(path: object) -> None:
    with pytest.raises(TypeError, match="run_record_path"):
        HermesApiTaskSession(
            config=HermesApiConfig(
                base_url="http://127.0.0.1:8765",
                bearer="run-record-path-test-bearer-value-32-chars",
            ),
            session_id="session_api_1",
            run_record_path=path,  # type: ignore[arg-type]
        )


def test_the_run_record_is_flushed_to_disk_before_it_replaces_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "state" / "hermes-runs-v1.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def spy_replace(source: str, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(run_record_module.os, "fsync", spy_fsync)
    monkeypatch.setattr(run_record_module.os, "replace", spy_replace)
    write_run_record(record, b'{"first":1}')

    assert events == ["fsync", "replace"]
    assert record.read_bytes() == b'{"first":1}'
    assert [path.name for path in record.parent.iterdir()] == [record.name]


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_an_interrupted_run_record_write_never_leaves_a_partial_file(
    failing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    write_run_record(record, b'{"first":1}')

    def interrupted(*_args: object) -> None:
        raise OSError("power lost")

    monkeypatch.setattr(run_record_module.os, failing, interrupted)
    with pytest.raises(OSError, match="power lost"):
        write_run_record(record, b'{"second":2}')

    assert record.read_bytes() == b'{"first":1}'
    assert [path.name for path in tmp_path.iterdir()] == [record.name]


def _orphan(path: Path) -> Path:
    """Leave exactly what a kill between mkstemp and os.replace leaves behind."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"orphaned plaintext")
    return Path(temporary)


def test_removing_orphans_takes_only_this_paths_interrupted_temporaries(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    write_run_record(record, b'{"kept":1}')
    orphans = [_orphan(record), _orphan(record)]
    kept = [
        tmp_path / ".hermes-runs-v1.json.bak.abcd1234.tmp",
        tmp_path / ".hermes-runs-v1.json.ABCD1234.tmp",
        tmp_path / ".hermes-runs-v1.json.abcd123.tmp",
        tmp_path / ".hermes-runs-v1.json.abcd1234.tmp.keep",
        tmp_path / "hermes-runs-v1.json.lock",
        tmp_path / ".other.json.abcd1234.tmp",
    ]
    for path in kept:
        path.write_bytes(b"unrelated")
    _orphan(tmp_path / "other.json")

    assert run_record_module.remove_orphaned_temporaries(record) == len(orphans)

    assert not any(orphan.exists() for orphan in orphans)
    assert all(path.exists() for path in kept)
    assert record.read_bytes() == b'{"kept":1}'
    assert len(list(tmp_path.glob(".other.json.*.tmp"))) == 2
    assert run_record_module.remove_orphaned_temporaries(tmp_path / "absent" / "x.json") == 0


@pytest.mark.asyncio
async def test_start_removes_orphaned_run_record_temporaries_after_taking_the_lock(
    tmp_path: Path,
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(record)
    orphan = _orphan(record)
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert not orphan.exists()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_scanner_held_run_record_orphan_never_fails_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    _write_record(record)
    held = _orphan(record)
    removable = _orphan(record)
    real_unlink = Path.unlink

    def scanner_held(self: Path, missing_ok: bool = False) -> None:
        if self == held:
            raise PermissionError(13, "sharing violation")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", scanner_held)
    hermes = _IdempotentHermes("answer")
    runner, port = await hermes.serve()
    session = _record_session(port, record)
    try:
        await session.start()

        assert session.restart_settlement == HermesRestartSettlement(stopped=0, unknown=0)
        assert held.exists()
        assert not removable.exists()
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_a_second_host_leaves_the_owners_temporaries_alone(tmp_path: Path) -> None:
    record = tmp_path / "hermes-runs-v1.json"
    release = asyncio.Event()
    hermes = _IdempotentHermes("answer", release_events=release)
    runner, port = await hermes.serve()
    first = _record_session(port, record)
    second = _record_session(port, record)
    try:
        await first.start()
        in_flight = _orphan(record)

        with pytest.raises(RuntimeError, match="another host holds the Hermes run record"):
            await second.start()

        assert in_flight.exists()
    finally:
        release.set()
        await first.close()
        await second.close()
        await runner.cleanup()
