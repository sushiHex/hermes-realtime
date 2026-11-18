"""Test-only model lifetimes for a native-runtime rehearsal, never provider qualification.

Only exact preregistered objectives are admitted, once each. No task or witness is
fabricated. A real client disconnect ends its stream; a local release is for unit
tests/cleanup only and must not be used before a journey's cancellation witnesses.
Keep this endpoint inside the qualification controller's existing ownership scope.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping
from typing import Any

from aiohttp import web


def _chunk(text: str, finish: str | None) -> bytes:
    payload = {"id": "controlled", "object": "chat.completion.chunk",
               "created": 0, "model": "controlled",
               "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


class ControlledTaskModel:
    def __init__(self, objectives: Mapping[str, str]) -> None:
        if (not 1 <= len(objectives) <= 4
                or any(type(k) is not str or type(v) is not str or not v
                       for k, v in objectives.items())
                or len(set(objectives.values())) != len(objectives)):
            raise ValueError("controlled_objective_binding")
        self._objectives = dict(objectives)
        self._started = {label: asyncio.Event() for label in objectives}
        self._released = {label: asyncio.Event() for label in objectives}
        self._closed = {label: asyncio.Event() for label in objectives}
        self._handlers: set[asyncio.Task[Any]] = set()
        self._admitted: set[str] = set()
        self._refused = 0
        self._closing = False
        self._runner: web.AppRunner | None = None

    async def start(self) -> str:
        if self._runner is not None or self._closing:
            raise RuntimeError("controlled_model_already_started")
        app = web.Application(client_max_size=1_048_576)
        app.router.add_post("/v1/chat/completions", self._complete)
        app.router.add_get("/v1/models", self._models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        return f"http://127.0.0.1:{port}/v1"

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": "controlled", "object": "model", "context_length": 131072}]})

    def snapshot(self) -> dict[str, int]:
        closed = sum(self._closed[label].is_set() for label in self._admitted)
        return {"admitted": len(self._admitted), "open": len(self._admitted) - closed,
                "closed": closed, "refused": self._refused}

    async def wait_started(self, label: str) -> None:
        await self._started[label].wait()

    async def wait_closed(self, label: str) -> None:
        await self._closed[label].wait()

    def release(self, label: str) -> None:
        self._released[label].set()

    async def close(self) -> None:
        self._closing = True
        for handler in tuple(self._handlers):
            handler.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()

    def _label(self, body: object) -> str | None:
        if type(body) is not dict or body.get("stream") is not True:
            return None
        messages = body.get("messages")
        if type(messages) is not list or not messages:
            return None
        last = messages[-1]
        if type(last) is not dict or last.get("role") != "user":
            return None
        return next((label for label, objective in self._objectives.items()
                     if last.get("content") == objective), None)

    async def _complete(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except (ValueError, web.HTTPRequestEntityTooLarge):
            body = None
        label = self._label(body)
        if self._closing or label is None or label in self._admitted:
            self._refused = min(4096, self._refused + 1)
            status = 409 if label in self._admitted else 400
            try:
                return web.json_response({"error": "controlled_request_refused"}, status=status)
            finally:
                print('[controlled-task-refusal] ' + json.dumps(
                    {"category": "request_refused", "count": int(self._refused)}), flush=True)
        # No await between the one-shot decision and admission.
        self._admitted.add(label)
        handler = asyncio.current_task()
        assert handler is not None
        self._handlers.add(handler)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await response.prepare(request)
            self._started[label].set()
            while not self._released[label].is_set():
                # Data chunks let the native model consumer observe cancellation;
                # comments can be discarded inside the SDK. No completion timer.
                await response.write(_chunk(".", None))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._released[label].wait(), 0.1)
            await response.write(_chunk("Released.", "stop"))
            await response.write(b"data: [DONE]\n\n")
        except ConnectionResetError:
            pass
        finally:
            self._closed[label].set()
            self._handlers.discard(handler)
        return response
