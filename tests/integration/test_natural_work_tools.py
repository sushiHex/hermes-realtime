from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from itertools import count
from types import MethodType
from typing import Any, cast

import pytest
from aiohttp import web

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationTaskController,
    ConversationUpdateExecutor,
    ConversationWorkControlSurface,
    WorkCancelResult,
    WorkStartResult,
)
from hermes_realtime.conversation.worker import ConversationSessionWorker
from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.speech import AudioFrame, Transcript, VoiceActivity

_API_RUN_ID = "run_0123456789abcdef"
_SECOND_API_RUN_ID = "run_123456789abcdef0"
_PROTOCOL_RUN_ID = "deleg_cross_layer_private"
_BEARER = "cross-layer-test-bearer-value-32-characters"
_PRIVATE_DELEGATION_NAMESPACE = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_PRIVATE_RUN_NAMESPACE = re.compile(
    r"(?<![A-Za-z0-9_-])run_[A-Za-z0-9][A-Za-z0-9_-]{15,127}"
    r"(?![A-Za-z0-9_-])"
)


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _capabilities() -> dict[str, object]:
    return {
        "object": "hermes.api_server.capabilities",
        "platform": "hermes-agent",
        "model": "test-model",
        "auth": {"type": "bearer", "required": True},
        "runtime": {
            "mode": "server_agent",
            "tool_execution": "server",
            "split_runtime": False,
        },
        "features": {
            "run_submission": True,
            "run_status": True,
            "run_events_sse": True,
            "run_stop": True,
            "run_approval_response": True,
            "approval_events": True,
        },
        "endpoints": {
            "runs": {"method": "POST", "path": "/v1/runs"},
            "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
            "run_events": {
                "method": "GET",
                "path": "/v1/runs/{run_id}/events",
            },
            "run_approval": {
                "method": "POST",
                "path": "/v1/runs/{run_id}/approval",
            },
            "run_stop": {
                "method": "POST",
                "path": "/v1/runs/{run_id}/stop",
            },
        },
    }


@dataclass(slots=True)
class HermesApiStub:
    mode: str = "accepted"
    create_gate: asyncio.Event = field(default_factory=asyncio.Event)
    create_entered: asyncio.Event = field(default_factory=asyncio.Event)
    events_connected: asyncio.Event = field(default_factory=asyncio.Event)
    terminal_ready: asyncio.Event = field(default_factory=asyncio.Event)
    posts: list[dict[str, object]] = field(default_factory=list)
    stop_calls: int = 0
    stopped_run_ids: list[str] = field(default_factory=list)
    terminal_event: str = "run.cancelled"
    terminal_text: str = "Cross-layer work completed."
    status: str = "running"

    def __post_init__(self) -> None:
        if self.mode not in {
            "accepted",
            "disconnect",
            "rejected",
            "timeout",
        }:
            raise ValueError("unsupported Hermes API stub mode")
        if self.mode != "timeout":
            self.create_gate.set()

    async def capabilities(self, request: web.Request) -> web.Response:
        assert request.headers["Authorization"] == f"Bearer {_BEARER}"
        return web.json_response(_capabilities())

    async def create_run(self, request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"] == f"Bearer {_BEARER}"
        body = await request.json()
        assert type(body) is dict
        self.posts.append(cast(dict[str, object], body))
        run_id = _API_RUN_ID if len(self.posts) == 1 else _SECOND_API_RUN_ID
        self.create_entered.set()
        await self.create_gate.wait()
        if self.mode == "disconnect":
            transport = request.transport
            assert transport is not None
            transport.close()
            await asyncio.sleep(0)
            return web.Response(status=202)
        if self.mode == "rejected":
            return web.json_response(
                {
                    "error": {
                        "internal": {
                            "run": _API_RUN_ID,
                            "delegation": _PROTOCOL_RUN_ID,
                        }
                    }
                },
                status=503,
            )
        self.status = "running"
        return web.json_response(
            {"run_id": run_id, "status": "started"},
            status=202,
        )

    async def events(self, request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        assert run_id in {_API_RUN_ID, _SECOND_API_RUN_ID}
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        self.events_connected.set()
        await self.terminal_ready.wait()
        payload: dict[str, object] = {
            "event": self.terminal_event,
            "run_id": run_id,
        }
        if self.terminal_event == "run.completed":
            payload["output"] = self.terminal_text
        elif self.terminal_event == "run.failed":
            payload["error"] = self.terminal_text
        await response.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())
        await response.write_eof()
        return response

    async def stop(self, request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        assert run_id in {_API_RUN_ID, _SECOND_API_RUN_ID}
        self.stop_calls += 1
        self.stopped_run_ids.append(run_id)
        self.status = "stopping"
        return web.json_response({"run_id": run_id, "status": "stopping"})

    async def run_status(self, request: web.Request) -> web.Response:
        run_id = request.match_info["run_id"]
        assert run_id in {_API_RUN_ID, _SECOND_API_RUN_ID}
        payload: dict[str, object] = {
            "run_id": run_id,
            "status": self.status,
        }
        if self.status == "completed":
            payload["output"] = self.terminal_text
        return web.json_response(payload)

    def complete(self, *, status: str = "completed") -> None:
        if status == "completed":
            self.terminal_event = "run.completed"
        elif status == "failed":
            self.terminal_event = "run.failed"
        elif status == "cancelled":
            self.terminal_event = "run.cancelled"
        else:
            raise ValueError("unsupported terminal status")
        self.status = status
        self.terminal_ready.set()


class ToolCallingCodexTransport:
    def __init__(
        self,
        *,
        tool: str,
        arguments: dict[str, object],
        thread_id: str,
        turn_id: str,
        call_id: str,
        duplicate_request: bool = False,
        second_model_call: bool = False,
        hold_for_interrupt: bool = False,
        evidence_identity: tuple[str, str, str] | None = None,
        allow_unadvertised_tool: bool = False,
    ) -> None:
        self.tool = tool
        self.arguments = arguments
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.call_id = call_id
        self.duplicate_request = duplicate_request
        self.second_model_call = second_model_call
        self.hold_for_interrupt = hold_for_interrupt
        self.allow_unadvertised_tool = allow_unadvertised_tool
        (
            self.evidence_thread_id,
            self.evidence_turn_id,
            self.evidence_call_id,
        ) = evidence_identity or (thread_id, turn_id, call_id)
        self.second_call_id = f"{self.evidence_call_id}_second"
        self.sent: list[dict[str, object]] = []
        self.incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.tool_responses: list[dict[str, object]] = []
        self.advertised_tools: set[str] = set()
        self.turn_started = asyncio.Event()
        self.tool_requested = asyncio.Event()
        self.tool_responded = asyncio.Event()
        self.closed = False

    async def send(self, message: Mapping[str, object]) -> None:
        copied = dict(message)
        self.sent.append(copied)
        method = copied.get("method")
        request_id = copied.get("id")
        if method == "initialize":
            await self.incoming.put({"id": request_id, "result": {}})
            return
        if method == "model/list":
            await self.incoming.put(
                {
                    "id": request_id,
                    "result": {
                        "data": [
                            {
                                "defaultReasoningEffort": "low",
                                "description": "test model",
                                "displayName": "GPT-5.6 Terra",
                                "hidden": False,
                                "id": "gpt-5.6-terra",
                                "isDefault": True,
                                "model": "gpt-5.6-terra",
                                "supportedReasoningEfforts": [
                                    {
                                        "description": "Quick",
                                        "reasoningEffort": "low",
                                    }
                                ],
                            }
                        ],
                        "nextCursor": None,
                    },
                }
            )
            return
        if method == "thread/start":
            params = cast(dict[str, object], copied["params"])
            dynamic_tools = cast(list[dict[str, object]], params["dynamicTools"])
            self.advertised_tools = {cast(str, item["name"]) for item in dynamic_tools}
            await self.incoming.put(
                {
                    "id": request_id,
                    "result": {
                        "thread": {"id": self.thread_id},
                        "model": "gpt-5.6-terra",
                        "modelProvider": "openai",
                    },
                }
            )
            return
        if method == "turn/start":
            await self._start_tool_call(request_id)
            return
        if method == "turn/interrupt":
            await self.incoming.put({"id": request_id, "result": {}})
            await self._complete_turn("interrupted")
            return
        if request_id in {91, 92, 93} and "result" in copied:
            self.tool_responses.append(copied)
            if self.duplicate_request and len(self.tool_responses) == 1:
                await self._request_tool(92, call_id=self.evidence_call_id)
                return
            first_response_count = 2 if self.duplicate_request else 1
            if self.second_model_call and len(self.tool_responses) == first_response_count:
                await self._complete_tool(
                    copied,
                    call_id=self.evidence_call_id,
                )
                await self._start_model_call(self.second_call_id)
                await self._request_tool(93, call_id=self.second_call_id)
                return
            self.tool_responded.set()
            completed_call_id = self.second_call_id if request_id == 93 else self.evidence_call_id
            await self._complete_tool(copied, call_id=completed_call_id)
            if not self.hold_for_interrupt:
                await self._complete_with_assistant(self.tool_responses[0])

    async def _start_tool_call(self, request_id: object) -> None:
        if not self.allow_unadvertised_tool and self.tool not in self.advertised_tools:
            raise AssertionError(f"fake Codex attempted unadvertised dynamic tool {self.tool!r}")
        await self.incoming.put(
            {
                "id": request_id,
                "result": {
                    "turn": {
                        "id": self.turn_id,
                        "items": [],
                        "status": "inProgress",
                    }
                },
            }
        )
        self.turn_started.set()
        await self._start_model_call(self.evidence_call_id)
        await self._request_tool(91, call_id=self.evidence_call_id)

    async def _start_model_call(self, call_id: str) -> None:
        item = {
            "id": call_id,
            "type": "dynamicToolCall",
            "tool": self.tool,
            "namespace": None,
            "arguments": self.arguments,
            "status": "inProgress",
        }
        await self.incoming.put(
            {
                "method": "item/started",
                "params": {
                    "threadId": self.evidence_thread_id,
                    "turnId": self.evidence_turn_id,
                    "item": item,
                    "startedAtMs": 1,
                },
            }
        )

    async def _request_tool(self, request_id: int, *, call_id: str) -> None:
        await self.incoming.put(
            {
                "id": request_id,
                "method": "item/tool/call",
                "params": {
                    "threadId": self.evidence_thread_id,
                    "turnId": self.evidence_turn_id,
                    "callId": call_id,
                    "namespace": None,
                    "tool": self.tool,
                    "arguments": self.arguments,
                },
            }
        )
        self.tool_requested.set()

    async def _complete_tool(
        self,
        response: dict[str, object],
        *,
        call_id: str,
    ) -> None:
        result = cast(dict[str, object], response["result"])
        await self.incoming.put(
            {
                "method": "item/completed",
                "params": {
                    "threadId": self.evidence_thread_id,
                    "turnId": self.evidence_turn_id,
                    "completedAtMs": 2,
                    "item": {
                        "id": call_id,
                        "type": "dynamicToolCall",
                        "tool": self.tool,
                        "namespace": None,
                        "arguments": self.arguments,
                        "status": ("completed" if result["success"] is True else "failed"),
                        "contentItems": result["contentItems"],
                        "success": result["success"],
                    },
                },
            }
        )

    async def _complete_with_assistant(
        self,
        response: dict[str, object],
    ) -> None:
        payload = _tool_payload(response)
        if payload["state"] == "cancelling":
            delta = "I requested cancellation."
        elif payload["accepted"] is True:
            delta = "I started it."
        else:
            delta = "I couldn't start that work."
        await self.incoming.put(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": self.thread_id,
                    "turnId": self.turn_id,
                    "itemId": f"assistant_{self.call_id}",
                    "delta": delta,
                },
            }
        )
        await self._complete_turn("completed")

    async def _complete_turn(self, status: str) -> None:
        await self.incoming.put(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": self.thread_id,
                    "turn": {
                        "id": self.turn_id,
                        "items": [],
                        "status": status,
                    },
                },
            }
        )

    async def receive(self) -> Mapping[str, object]:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True


class _TypedInputVad:
    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        return VoiceActivity.SILENCE


class _TypedInputTranscriber:
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return ()

    async def finish_utterance(self) -> Transcript | None:
        return None

    async def cancel(self) -> None:
        return None


@dataclass(slots=True)
class NaturalWorkRuntime:
    stub: HermesApiStub
    runner: web.AppRunner
    context: ConversationContextStore
    controller: ConversationTaskController
    surface: ConversationWorkControlSurface
    browser_events: list[tuple[str, dict[str, str | int | bool | None]]]
    cancel_attempts: list[tuple[str, str | None]]
    runtime_name: str
    inferences: list[CodexAppServerStreamingInference] = field(default_factory=list)
    _transport_counter: Any = field(default_factory=lambda: count(1))

    def new_inference(
        self,
        *,
        tool: str = "start_work",
        arguments: dict[str, object] | None = None,
        duplicate_request: bool = False,
        second_model_call: bool = False,
        hold_for_interrupt: bool = False,
        handler: Any = None,
        evidence_identity: tuple[str, str, str] | None = None,
        allow_unadvertised_tool: bool = False,
    ) -> tuple[CodexAppServerStreamingInference, ToolCallingCodexTransport]:
        index = next(self._transport_counter)
        transport = ToolCallingCodexTransport(
            tool=tool,
            arguments=(
                {"objective": "Inspect the release evidence"} if arguments is None else arguments
            ),
            thread_id=f"thread_{self.runtime_name}_{index}",
            turn_id=f"turn_{self.runtime_name}_{index}",
            call_id=f"call_{self.runtime_name}_{index}",
            duplicate_request=duplicate_request,
            second_model_call=second_model_call,
            hold_for_interrupt=hold_for_interrupt,
            evidence_identity=evidence_identity,
            allow_unadvertised_tool=allow_unadvertised_tool,
        )
        inference = CodexAppServerStreamingInference(
            model="gpt-5.6-terra",
            effort="low",
            transport_factory=lambda: transport,
            request_timeout_seconds=1,
            work_tool_timeout_seconds=0.75,
        )
        inference.bind_work_tools(self.surface if handler is None else handler)
        self.inferences.append(inference)
        return inference, transport

    async def invoke(
        self,
        text: str,
        *,
        tool: str = "start_work",
        arguments: dict[str, object] | None = None,
        duplicate_request: bool = False,
        second_model_call: bool = False,
        evidence_identity: tuple[str, str, str] | None = None,
        allow_unadvertised_tool: bool = False,
    ) -> tuple[list[str], ToolCallingCodexTransport]:
        self.context.record_user_transcript(Transcript(text=text, final=True))
        inference, transport = self.new_inference(
            tool=tool,
            arguments=arguments,
            duplicate_request=duplicate_request,
            second_model_call=second_model_call,
            evidence_identity=evidence_identity,
            allow_unadvertised_tool=allow_unadvertised_tool,
        )
        segments = [
            segment
            async for segment in inference.stream(
                self.context.snapshot(),
                turn_id=f"foreground_{len(self.inferences)}",
            )
        ]
        _assert_model_visible_values_are_private_token_free(
            segments,
            transport.sent,
            transport.tool_responses,
            self.context.snapshot(),
            self.browser_events,
        )
        return segments, transport

    async def invoke_typed(
        self,
        text: str,
    ) -> tuple[list[str], ToolCallingCodexTransport]:
        inference, transport = self.new_inference()
        segments: list[str] = []
        executor = object.__new__(ConversationUpdateExecutor)
        response_reservation = object()

        def start(actions: ConversationUpdateExecutor) -> None:
            del actions

        async def respond(
            actions: ConversationUpdateExecutor,
            turn_id: str,
            transcript: Transcript,
            authority: object | None = None,
            *,
            reservation: object,
        ) -> None:
            del actions
            assert authority is None
            assert reservation is response_reservation
            self.context.record_user_transcript(transcript)
            segments.extend(
                [
                    segment
                    async for segment in inference.stream(
                        self.context.snapshot(),
                        turn_id=turn_id,
                    )
                ]
            )

        async def cancel_foreground(
            actions: ConversationUpdateExecutor,
            **_kwargs: object,
        ) -> None:
            del actions
            await inference.cancel("session_1_turn_1")

        async def close(actions: ConversationUpdateExecutor) -> None:
            del actions

        def reserve_response(actions: ConversationUpdateExecutor) -> object:
            del actions
            return response_reservation

        def release_response(
            actions: ConversationUpdateExecutor,
            reservation: object,
        ) -> None:
            del actions
            assert reservation is response_reservation

        executor.start = MethodType(start, executor)  # type: ignore[method-assign]
        executor.reserve_response = MethodType(  # type: ignore[method-assign]
            reserve_response,
            executor,
        )
        executor.release_response = MethodType(  # type: ignore[method-assign]
            release_response,
            executor,
        )
        executor.respond = MethodType(respond, executor)  # type: ignore[method-assign]
        executor.cancel_foreground = MethodType(  # type: ignore[method-assign]
            cancel_foreground,
            executor,
        )
        executor._cancel_foreground_for_cleanup = MethodType(  # type: ignore[method-assign]
            cancel_foreground,
            executor,
        )
        executor.close = MethodType(close, executor)  # type: ignore[method-assign]
        worker = ConversationSessionWorker(
            participant_identity="browser_user",
            session_generation=1,
            vad=_TypedInputVad(),
            stt=_TypedInputTranscriber(),
            actions=executor,
        )
        worker.start()
        try:
            await worker.submit_final_transcript(
                participant_identity="browser_user",
                session_generation=1,
                typed_sequence=1,
                text=text,
            )
            await worker.wait_for_responses()
        finally:
            await worker.close()
        _assert_model_visible_values_are_private_token_free(
            segments,
            transport.sent,
            transport.tool_responses,
            self.context.snapshot(),
            self.browser_events,
        )
        return segments, transport

    async def close(self) -> None:
        failures: list[BaseException] = []
        if self.context.snapshot().active_tasks:
            self.stub.complete(status="cancelled")
            while self.context.snapshot().active_tasks:
                try:
                    await asyncio.wait_for(
                        self.controller.next_completion(),
                        timeout=1,
                    )
                except BaseException as error:
                    failures.append(error)
                    break
        for inference in reversed(self.inferences):
            try:
                await inference.close()
            except BaseException as error:
                failures.append(error)
        for close in (
            self.surface.close,
            self.controller.close,
            self.runner.cleanup,
        ):
            try:
                await close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("natural work runtime cleanup failed", failures)


async def _runtime(
    *,
    mode: str = "accepted",
    request_timeout_seconds: float = 1,
    runtime_name: str = "cross",
    start_projection_gate: Any = None,
) -> NaturalWorkRuntime:
    port = _available_port()
    stub = HermesApiStub(mode=mode)
    app = web.Application()
    app.router.add_get("/v1/capabilities", stub.capabilities)
    app.router.add_post("/v1/runs", stub.create_run)
    app.router.add_get("/v1/runs/{run_id}/events", stub.events)
    app.router.add_get("/v1/runs/{run_id}", stub.run_status)
    app.router.add_post("/v1/runs/{run_id}/stop", stub.stop)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    context = ConversationContextStore(max_item_chars=1024)
    private_identifiers = count(1)
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}",
            bearer=_BEARER,
            request_timeout_seconds=request_timeout_seconds,
            settlement_timeout_seconds=0.2,
            settlement_poll_seconds=0.01,
        ),
        session_id="session_cross_layer",
        private_id_factory=lambda: (
            "cross_layer_private"
            if (identifier := next(private_identifiers)) == 1
            else f"cross_layer_private_{identifier}"
        ),
    )
    try:
        await session.start()
    except BaseException:
        try:
            await session.close()
        finally:
            await runner.cleanup()
        raise
    identifiers = count(1)
    controller = ConversationTaskController(
        context=context,
        session=session,
        session_id="session_cross_layer",
        id_factory=lambda: f"cross{next(identifiers)}",
    )
    cancel_attempts: list[tuple[str, str | None]] = []

    class RecordingController:
        async def dispatch(self, *, objective: str, utterance_id: str) -> Any:
            return await controller.dispatch(
                objective=objective,
                utterance_id=utterance_id,
            )

        async def request_cancel(
            self,
            task_id: str,
            *,
            reason: str | None = None,
        ) -> Any:
            cancel_attempts.append((task_id, reason))
            return await controller.request_cancel(task_id, reason=reason)

    browser_events: list[tuple[str, dict[str, str | int | bool | None]]] = []
    surface = ConversationWorkControlSurface(
        controller=RecordingController(),
        context=context,
        observer=lambda kind, data: browser_events.append((kind, dict(data))),
        reserve_observer_capacity=lambda: None,
        utterance_id_factory=lambda: "cross_layer_utterance",
        start_projection_gate=start_projection_gate,
    )
    return NaturalWorkRuntime(
        stub=stub,
        runner=runner,
        context=context,
        controller=controller,
        surface=surface,
        browser_events=browser_events,
        cancel_attempts=cancel_attempts,
        runtime_name=runtime_name,
    )


def _tool_payload(response: Mapping[str, object]) -> dict[str, object]:
    result = cast(dict[str, object], response["result"])
    items = cast(list[dict[str, object]], result["contentItems"])
    return cast(dict[str, object], json.loads(cast(str, items[0]["text"])))


def _assert_model_visible_values_are_private_token_free(*values: object) -> None:
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            assert _PRIVATE_RUN_NAMESPACE.search(value) is None
            assert _PRIVATE_DELEGATION_NAMESPACE.search(value) is None
            continue
        if isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            pending.extend(value)
            continue
        if hasattr(value, "__dataclass_fields__"):
            pending.extend(getattr(value, name) for name in value.__dataclass_fields__)


_EXPECTED_ACCEPTED_START_SPEECH = ["I started it."]


@pytest.mark.asyncio
async def test_typed_natural_start_crosses_codex_and_202_authority_once() -> None:
    runtime = await _runtime()
    try:
        spoken, transport = await runtime.invoke_typed(
            "Please inspect the release evidence and report back."
        )

        assert len(runtime.stub.posts) == 1
        dispatch_body = runtime.stub.posts[0]
        assert dispatch_body["input"] == "Inspect the release evidence"
        instructions = dispatch_body["instructions"]
        assert isinstance(instructions, str)
        assert "one parallel batch" in instructions
        assert "at most 1800 characters" in instructions
        assert spoken == _EXPECTED_ACCEPTED_START_SPEECH
        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": True,
            "state": "active",
            "task_id": "task_cross1",
        }
        assert runtime.context.snapshot().active_tasks == (
            runtime.context.snapshot().active_tasks[0],
        )
        assert runtime.context.snapshot().active_tasks[0].task_id == "task_cross1"
        assert runtime.browser_events == [
            (
                "task_state",
                {"status": "active", "taskId": "task_cross1"},
            )
        ]
        _assert_model_visible_values_are_private_token_free(
            spoken,
            transport.sent,
            transport.tool_responses,
            runtime.context.snapshot(),
            runtime.browser_events,
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_hermes_rejection_cannot_produce_active_state_or_started_claim() -> None:
    runtime = await _runtime(mode="rejected")
    try:
        spoken, transport = await runtime.invoke("Please inspect the release.")

        payload = _tool_payload(transport.tool_responses[0])
        assert payload == {
            "accepted": False,
            "state": "rejected",
            "task_id": "task_cross1",
            "reason": "Hermes API rejected dispatch",
        }
        assert spoken == ["I couldn't start that work."]
        assert runtime.context.snapshot().active_tasks == ()
        assert "started" not in " ".join(spoken).lower()
        _assert_model_visible_values_are_private_token_free(
            payload,
            spoken,
            transport.sent,
            runtime.browser_events,
            runtime.context.snapshot(),
        )
    finally:
        await runtime.close()


def test_private_namespace_scanner_does_not_reserve_ordinary_run_words() -> None:
    _assert_model_visible_values_are_private_token_free(
        "Please run_report and summarize the brunch_run_notes."
    )


@pytest.mark.asyncio
async def test_model_forged_delegation_objective_is_rejected_before_dispatch() -> None:
    runtime = await _runtime()
    inference, transport = runtime.new_inference(
        arguments={"objective": "Inspect deleg_forged_model_authority"},
    )
    runtime.context.record_user_transcript(
        Transcript(text="Inspect the release evidence.", final=True)
    )
    try:
        with pytest.raises(RuntimeError, match="Codex app-server reader failed"):
            await _collect(
                inference.stream(
                    runtime.context.snapshot(),
                    turn_id="forged_private_objective",
                )
            )

        assert runtime.stub.posts == []
        assert runtime.cancel_attempts == []
        assert runtime.context.snapshot().active_tasks == ()
        assert len(transport.tool_responses) == 1
        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work request rejected",
        }
        _assert_model_visible_values_are_private_token_free(
            transport.sent,
            transport.tool_responses,
            runtime.browser_events,
            runtime.context.snapshot(),
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "disconnect"])
async def test_uncertain_http_failure_blocks_later_work_admission(
    mode: str,
) -> None:
    runtime = await _runtime(mode=mode, request_timeout_seconds=0.05)
    try:
        first_spoken, first_transport = await runtime.invoke("Please inspect the release.")
        second_spoken, second_transport = await runtime.invoke("Try another background inspection.")

        assert first_spoken == ["I couldn't start that work."]
        assert second_spoken == ["I couldn't start that work."]
        assert _tool_payload(first_transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work control is unavailable",
        }
        assert _tool_payload(second_transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work control unavailable; restart required",
        }
        assert runtime.surface.health == "uncertain"
        assert len(runtime.stub.posts) == 1
        assert runtime.context.snapshot().active_tasks == ()
    finally:
        runtime.stub.create_gate.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_natural_cancel_stops_exact_run_once_and_waits_for_terminal() -> None:
    runtime = await _runtime()
    try:
        await runtime.invoke("Please inspect the release evidence.")
        await asyncio.wait_for(runtime.stub.events_connected.wait(), timeout=1)

        spoken, transport = await runtime.invoke(
            "Stop that background work.",
            tool="cancel_active_work",
            arguments={},
        )

        assert spoken == ["I requested cancellation."]
        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": True,
            "state": "cancelling",
            "task_id": "task_cross1",
        }
        assert runtime.stub.stop_calls == 1
        assert len(runtime.context.snapshot().active_tasks) == 1

        runtime.stub.complete(status="cancelled")
        terminal = await asyncio.wait_for(
            runtime.controller.next_completion(),
            timeout=1,
        )
        assert terminal.task_id == "task_cross1"
        assert terminal.status == "interrupted"
        assert runtime.context.snapshot().active_tasks == ()
        assert runtime.stub.stop_calls == 1

        with pytest.raises(RuntimeError, match="Codex app-server reader failed"):
            await runtime.invoke(
                "Continue that canceled task.",
                allow_unadvertised_tool=True,
            )
        assert len(runtime.stub.posts) == 1
        assert runtime.stub.stop_calls == 1
        assert runtime.context.snapshot().active_tasks == ()
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_stop_calls", "expected_start_state"),
    [
        ("accepted", 1, "cancelling"),
        ("rejected", 0, "rejected"),
    ],
)
async def test_cancel_while_dispatch_pending_settles_after_late_ack(
    mode: str,
    expected_stop_calls: int,
    expected_start_state: str,
) -> None:
    runtime = await _runtime(mode=mode)
    runtime.stub.create_gate.clear()
    try:
        start_task = asyncio.create_task(runtime.invoke("Please inspect the release evidence."))
        await asyncio.wait_for(runtime.stub.create_entered.wait(), timeout=1)
        cancel_task = asyncio.create_task(
            runtime.invoke(
                "Cancel that work.",
                tool="cancel_active_work",
                arguments={},
            )
        )

        async def wait_for_pending_cancel() -> None:
            while True:
                pending = tuple(runtime.surface._pending_starts.values())
                if len(pending) == 1 and pending[0].cancel_after_ack:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_pending_cancel(), timeout=1)
        assert runtime.stub.stop_calls == 0

        runtime.stub.create_gate.set()
        (
            (start_spoken, start_transport),
            (
                cancel_spoken,
                cancel_transport,
            ),
        ) = await asyncio.gather(start_task, cancel_task)

        start_payload = _tool_payload(start_transport.tool_responses[0])
        cancel_payload = _tool_payload(cancel_transport.tool_responses[0])
        assert start_payload["state"] == expected_start_state
        assert cancel_payload["state"] == expected_start_state
        assert runtime.stub.stop_calls == expected_stop_calls
        assert "started" not in " ".join(start_spoken + cancel_spoken).lower()
        if mode == "accepted":
            assert start_payload["accepted"] is True
            assert cancel_payload["accepted"] is True
            runtime.stub.complete(status="cancelled")
            await asyncio.wait_for(runtime.controller.next_completion(), timeout=1)
        else:
            assert start_payload["accepted"] is False
            assert cancel_payload["accepted"] is False
            assert runtime.context.snapshot().active_tasks == ()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_duplicate_model_call_and_jsonrpc_replay_issue_one_hermes_post() -> None:
    runtime = await _runtime()
    try:
        spoken, transport = await runtime.invoke(
            "Inspect the release evidence.",
            duplicate_request=True,
            second_model_call=True,
        )

        assert spoken == _EXPECTED_ACCEPTED_START_SPEECH
        assert len(transport.tool_responses) == 3
        assert transport.tool_responses[0]["id"] == 91
        assert transport.tool_responses[1]["id"] == 92
        assert transport.tool_responses[2]["id"] == 93
        assert transport.tool_responses[0]["result"] == transport.tool_responses[1]["result"]
        assert _tool_payload(transport.tool_responses[2]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "one knowledge search or work start is allowed per turn",
        }
        assert len(runtime.stub.posts) == 1
        assert len(runtime.context.snapshot().active_tasks) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_exact_cancel_of_second_task_preserves_first_task_authority() -> None:
    runtime = await _runtime()
    try:
        await runtime.invoke("Inspect the release evidence.")
        await runtime.invoke(
            "Audit the dependency licenses too.",
            arguments={"objective": "Audit the dependency licenses."},
        )

        spoken, transport = await runtime.invoke(
            "Cancel the dependency-license audit.",
            tool="cancel_active_work",
            arguments={"task_id": "task_cross3"},
        )

        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": True,
            "state": "cancelling",
            "task_id": "task_cross3",
        }
        assert spoken == ["I requested cancellation."]
        assert runtime.stub.stopped_run_ids == [_SECOND_API_RUN_ID]
        assert {task.task_id for task in runtime.context.snapshot().active_tasks} == {
            "task_cross1",
            "task_cross3",
        }
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_second_distinct_natural_start_is_accepted_while_one_task_is_active() -> None:
    runtime = await _runtime()
    try:
        await runtime.invoke("Inspect the release evidence.")
        spoken, transport = await runtime.invoke(
            "Audit the dependency licenses too.",
            arguments={"objective": "Audit the dependency licenses."},
        )

        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": True,
            "state": "active",
            "task_id": "task_cross3",
        }
        assert spoken == _EXPECTED_ACCEPTED_START_SPEECH
        assert len(runtime.stub.posts) == 2
        assert {task.task_id for task in runtime.context.snapshot().active_tasks} == {
            "task_cross1",
            "task_cross3",
        }
    finally:
        await runtime.close()


class _CompletionBeforeResponse:
    def __init__(
        self,
        surface: ConversationWorkControlSurface,
        controller: ConversationTaskController,
        stub: HermesApiStub,
    ) -> None:
        self._surface = surface
        self._controller = controller
        self._stub = stub
        self.terminal: Any = None

    @property
    def max_objective_chars(self) -> int:
        return self._surface.max_objective_chars

    @property
    def can_cancel_work(self) -> bool:
        return self._surface.can_cancel_work

    async def start_work(
        self,
        *,
        objective: str,
        invocation_id: str,
    ) -> WorkStartResult:
        result = await self._surface.start_work(
            objective=objective,
            invocation_id=invocation_id,
        )
        self._stub.complete(status="completed")
        self.terminal = await self._controller.next_completion()
        return result

    async def cancel_active_work(
        self,
        *,
        invocation_id: str,
    ) -> WorkCancelResult:
        return await self._surface.cancel_active_work(invocation_id=invocation_id)

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> WorkCancelResult:
        return await self._surface.cancel_work(task_id=task_id, invocation_id=invocation_id)


@pytest.mark.asyncio
async def test_completion_race_yields_one_terminal_and_no_stale_context_task() -> None:
    runtime = await _runtime()
    handler = _CompletionBeforeResponse(
        runtime.surface,
        runtime.controller,
        runtime.stub,
    )
    inference, transport = runtime.new_inference(handler=handler)
    runtime.context.record_user_transcript(
        Transcript(text="Inspect the release evidence.", final=True)
    )
    try:
        spoken = await _collect(
            inference.stream(
                runtime.context.snapshot(),
                turn_id="completion_before_tool_response",
            )
        )

        assert spoken == _EXPECTED_ACCEPTED_START_SPEECH
        assert _tool_payload(transport.tool_responses[0])["accepted"] is True
        assert handler.terminal.task_id == "task_cross1"
        assert handler.terminal.status == "completed"
        assert runtime.context.snapshot().active_tasks == ()
        _assert_model_visible_values_are_private_token_free(
            spoken,
            transport.sent,
            transport.tool_responses,
            handler.terminal,
            runtime.context.snapshot(),
            runtime.browser_events,
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                runtime.controller.next_completion(),
                timeout=0.02,
            )
    finally:
        await runtime.close()


class _StartProjectionGate:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, result: WorkStartResult) -> None:
        if result.accepted and result.state == "active":
            self.entered.set()
            await self.release.wait()


@pytest.mark.asyncio
async def test_completion_before_projection_rolls_back_without_stale_active_state() -> None:
    gate = _StartProjectionGate()
    runtime = await _runtime(start_projection_gate=gate)
    invocation = asyncio.create_task(runtime.invoke("Inspect the release evidence."))
    try:
        await asyncio.wait_for(gate.entered.wait(), timeout=1)
        await asyncio.wait_for(runtime.stub.events_connected.wait(), timeout=1)

        runtime.stub.complete(status="completed")
        terminal = await asyncio.wait_for(runtime.controller.next_completion(), timeout=1)
        assert terminal.task_id == "task_cross1"
        assert terminal.status == "completed"
        assert runtime.context.snapshot().active_tasks == ()

        gate.release.set()
        spoken, transport = await asyncio.wait_for(invocation, timeout=1)

        assert spoken == ["I couldn't start that work."]
        assert _tool_payload(transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work control is unavailable",
        }
        assert runtime.cancel_attempts == [("task_cross1", "task state projection failed")]
        assert runtime.stub.stop_calls == 0
        assert runtime.browser_events == []
        assert runtime.context.snapshot().active_tasks == ()
        assert runtime.surface.health == "open"
        _assert_model_visible_values_are_private_token_free(
            spoken,
            transport.sent,
            transport.tool_responses,
            terminal,
            runtime.browser_events,
            runtime.context.snapshot(),
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(runtime.controller.next_completion(), timeout=0.02)
    finally:
        gate.release.set()
        if not invocation.done():
            invocation.cancel()
        await asyncio.gather(invocation, return_exceptions=True)
        await runtime.close()


class _StartGate:
    def __init__(self, surface: ConversationWorkControlSurface) -> None:
        self._surface = surface
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def max_objective_chars(self) -> int:
        return self._surface.max_objective_chars

    @property
    def can_cancel_work(self) -> bool:
        return self._surface.can_cancel_work

    async def start_work(
        self,
        *,
        objective: str,
        invocation_id: str,
    ) -> WorkStartResult:
        self.started.set()
        await self.release.wait()
        return await self._surface.start_work(
            objective=objective,
            invocation_id=invocation_id,
        )

    async def cancel_active_work(
        self,
        *,
        invocation_id: str,
    ) -> WorkCancelResult:
        return await self._surface.cancel_active_work(invocation_id=invocation_id)

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> WorkCancelResult:
        return await self._surface.cancel_work(task_id=task_id, invocation_id=invocation_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_dispatch", "during_ack", "after_ack"])
async def test_foreground_barge_in_never_stops_background_run(phase: str) -> None:
    runtime = await _runtime()
    runtime.stub.create_gate.clear()
    gate = _StartGate(runtime.surface)
    inference, transport = runtime.new_inference(
        hold_for_interrupt=True,
        handler=gate,
    )
    runtime.context.record_user_transcript(
        Transcript(text="Inspect the release evidence.", final=True)
    )
    stream_task = asyncio.create_task(
        _collect(
            inference.stream(
                runtime.context.snapshot(),
                turn_id=f"barge_{phase}",
            )
        )
    )
    try:
        await asyncio.wait_for(gate.started.wait(), timeout=1)
        if phase == "before_dispatch":
            cancellation = asyncio.create_task(inference.cancel(f"barge_{phase}"))
            await asyncio.wait_for(transport.tool_responded.wait(), timeout=1)
            gate.release.set()
            await asyncio.wait_for(runtime.stub.create_entered.wait(), timeout=1)
            runtime.stub.create_gate.set()
        elif phase == "during_ack":
            gate.release.set()
            await asyncio.wait_for(runtime.stub.create_entered.wait(), timeout=1)
            cancellation = asyncio.create_task(inference.cancel(f"barge_{phase}"))
            runtime.stub.create_gate.set()
        else:
            gate.release.set()
            runtime.stub.create_gate.set()
            await asyncio.wait_for(transport.tool_responded.wait(), timeout=1)
            cancellation = asyncio.create_task(inference.cancel(f"barge_{phase}"))

        await asyncio.wait_for(cancellation, timeout=1)
        assert await asyncio.wait_for(stream_task, timeout=1) == []
        for _ in range(100):
            if runtime.context.snapshot().active_tasks:
                break
            await asyncio.sleep(0)
        assert len(runtime.stub.posts) == 1
        assert len(runtime.context.snapshot().active_tasks) == 1
        assert runtime.stub.stop_calls == 0
        _assert_model_visible_values_are_private_token_free(
            transport.sent,
            transport.tool_responses,
            runtime.context.snapshot(),
            runtime.browser_events,
        )
    finally:
        gate.release.set()
        runtime.stub.create_gate.set()
        await runtime.close()


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    return [segment async for segment in stream]


@pytest.mark.asyncio
async def test_stale_codex_evidence_cannot_route_through_replacement_runtime() -> None:
    old = await _runtime(runtime_name="old")
    fresh: NaturalWorkRuntime | None = None
    try:
        _old_start_spoken, old_start_transport = await old.invoke("Start the old runtime work.")
        _old_cancel_spoken, old_cancel_transport = await old.invoke(
            "Cancel the old runtime work.",
            tool="cancel_active_work",
            arguments={},
        )
        old.stub.complete(status="cancelled")
        old_terminal = await asyncio.wait_for(old.controller.next_completion(), timeout=1)
        assert old_terminal.status == "interrupted"
        old_start_evidence = (
            old_start_transport.thread_id,
            old_start_transport.turn_id,
            old_start_transport.call_id,
        )
        old_cancel_evidence = (
            old_cancel_transport.thread_id,
            old_cancel_transport.turn_id,
            old_cancel_transport.call_id,
        )
        await old.close()

        fresh = await _runtime(runtime_name="replacement")
        stale_start_spoken, stale_start_transport = await fresh.invoke(
            "Offer stale start evidence.",
            evidence_identity=old_start_evidence,
        )
        stale_cancel_spoken, stale_cancel_transport = await fresh.invoke(
            "Offer stale cancel evidence.",
            tool="cancel_active_work",
            arguments={},
            evidence_identity=old_cancel_evidence,
            allow_unadvertised_tool=True,
        )

        assert stale_start_spoken == ["I couldn't start that work."]
        assert _tool_payload(stale_start_transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work request rejected",
        }
        assert stale_cancel_spoken == ["I couldn't start that work."]
        assert _tool_payload(stale_cancel_transport.tool_responses[0]) == {
            "accepted": False,
            "state": "rejected",
            "reason": "work request rejected",
        }
        assert fresh.stub.posts == []
        assert fresh.stub.stop_calls == 0
        assert fresh.cancel_attempts == []
        assert fresh.context.snapshot().active_tasks == ()
        assert fresh.browser_events == []
        assert fresh.surface.health == "open"
        _assert_model_visible_values_are_private_token_free(
            stale_start_spoken,
            stale_start_transport.sent,
            stale_start_transport.tool_responses,
            stale_cancel_spoken,
            stale_cancel_transport.sent,
            stale_cancel_transport.tool_responses,
            fresh.context.snapshot(),
            fresh.browser_events,
        )
    finally:
        if fresh is not None:
            await fresh.close()
        await old.close()
