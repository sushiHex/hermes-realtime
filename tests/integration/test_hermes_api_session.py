from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime
from typing import TypedDict

import pytest
from aiohttp import web

from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.integration.api import _RunAuthority
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
