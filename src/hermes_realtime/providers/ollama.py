"""Explicit loopback Ollama adapter for latency-critical foreground inference."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Protocol, cast
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from hermes_realtime.conversation.context import ConversationContextSnapshot
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.providers._text_segmentation import first_speakable_sentence_end

_MAX_MODEL_CHARS = 256
_MAX_SEGMENT_CHARS = 4096
_MAX_ACTIVE_STREAMS = 16


class _LineResponse(Protocol):
    def readline(self) -> bytes: ...

    def close(self) -> None: ...


_OpenRequest = Callable[[Request, float], _LineResponse]


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class OllamaStreamingInference:
    """Stream bounded speakable segments from one explicit loopback Ollama model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        open_request: _OpenRequest | None = None,
        request_timeout_seconds: float = 30.0,
        max_segment_chars: int = 1024,
        max_active_streams: int = 4,
    ) -> None:
        if type(base_url) is not str:
            raise TypeError("base_url must be an exact built-in string")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama base_url must be an exact loopback HTTP origin")
        if parsed.port is None:
            raise ValueError("Ollama base_url must include an explicit port")
        if type(model) is not str:
            raise TypeError("model must be an exact built-in string")
        if not model.strip() or len(model) > _MAX_MODEL_CHARS:
            raise ValueError("model must contain 1 to 256 characters")
        if type(request_timeout_seconds) not in (int, float):
            raise TypeError("request_timeout_seconds must be an exact number")
        if (
            not math.isfinite(request_timeout_seconds)
            or not 0 < request_timeout_seconds <= 300
        ):
            raise ValueError("request_timeout_seconds must be between 0 and 300")
        if type(max_segment_chars) is not int:
            raise TypeError("max_segment_chars must be an exact integer")
        if not 1 <= max_segment_chars <= _MAX_SEGMENT_CHARS:
            raise ValueError("max_segment_chars must be between 1 and 4096")
        if type(max_active_streams) is not int:
            raise TypeError("max_active_streams must be an exact integer")
        if not 1 <= max_active_streams <= _MAX_ACTIVE_STREAMS:
            raise ValueError("max_active_streams must be between 1 and 16")
        if open_request is not None and not callable(open_request):
            raise TypeError("open_request must be callable")

        origin = f"http://{parsed.hostname}"
        if ":" in parsed.hostname:
            origin = f"http://[{parsed.hostname}]"
        self._endpoint = f"{origin}:{parsed.port}/api/chat"
        self._model = model
        self._open_request = open_request or self._urlopen
        self._request_timeout = float(request_timeout_seconds)
        self._max_segment_chars = max_segment_chars
        self._max_active_streams = max_active_streams
        self._openings: dict[str, asyncio.Task[_LineResponse]] = {}
        self._opening_cleanups: dict[str, asyncio.Task[None]] = {}
        self._responses: dict[str, _LineResponse] = {}
        self._reads: dict[str, asyncio.Task[bytes]] = {}
        self._response_cleanups: dict[str, asyncio.Task[None]] = {}
        self._cancelled_turns: set[str] = set()
        self._state_lock = asyncio.Lock()
        self._close_operation: asyncio.Task[None] | None = None
        self._closed = False

    async def stream(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        turn_id: str,
    ) -> AsyncIterator[str]:
        snapshot = self._trusted_snapshot(snapshot)
        self._validate_turn_id(turn_id)
        request = Request(
            self._endpoint,
            data=json.dumps(
                {
                    "model": self._model,
                    "messages": self._messages(snapshot),
                    "stream": True,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opening: asyncio.Task[_LineResponse] | None = None
        response: _LineResponse | None = None
        read_operation: asyncio.Task[bytes] | None = None
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("Ollama inference adapter is closed")
            if turn_id in self._openings or turn_id in self._responses:
                raise RuntimeError("turn already owns an Ollama stream")
            if len(self._openings) + len(self._responses) >= self._max_active_streams:
                raise RuntimeError("Ollama stream capacity exhausted")
            opening = asyncio.create_task(
                asyncio.to_thread(
                    self._open_request,
                    request,
                    self._request_timeout,
                ),
                name=f"ollama-open:{turn_id}",
            )
            self._openings[turn_id] = opening
        try:
            opened_response = await asyncio.shield(opening)
            async with self._state_lock:
                if self._closed or turn_id in self._opening_cleanups:
                    self._ensure_opening_cleanup_locked(turn_id, opening)
                    raise RuntimeError("Ollama stream was cancelled while opening")
                if self._openings.get(turn_id) is not opening:
                    raise RuntimeError("Ollama opening ownership was lost")
                del self._openings[turn_id]
                self._responses[turn_id] = opened_response
                response = opened_response
                opening = None

            buffer = ""
            while True:
                async with self._state_lock:
                    if self._closed or turn_id in self._cancelled_turns:
                        raise RuntimeError("Ollama stream was cancelled while reading")
                    if self._responses.get(turn_id) is not response:
                        raise RuntimeError("Ollama response ownership was lost")
                    read_operation = asyncio.create_task(
                        asyncio.to_thread(response.readline),
                        name=f"ollama-read:{turn_id}",
                    )
                    self._reads[turn_id] = read_operation
                try:
                    raw_line = await asyncio.shield(read_operation)
                finally:
                    if read_operation.done():
                        async with self._state_lock:
                            if self._reads.get(turn_id) is read_operation:
                                del self._reads[turn_id]
                        read_operation = None
                if not raw_line:
                    break
                payload = self._payload(raw_line)
                content, done = self._content(payload)
                if content:
                    buffer += content
                    segments, buffer = self._extract_segments(buffer, final=False)
                    for segment in segments:
                        yield segment
                if done:
                    break
            segments, buffer = self._extract_segments(buffer, final=True)
            for segment in segments:
                yield segment
            if buffer:
                raise AssertionError("final segmentation retained text")
        finally:
            response_cleanup: asyncio.Task[None] | None = None
            async with self._state_lock:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    self._cancelled_turns.add(turn_id)
                if opening is not None:
                    self._ensure_opening_cleanup_locked(turn_id, opening)
                if response is not None:
                    response_cleanup = self._ensure_response_cleanup_locked(
                        turn_id, response, read_operation
                    )
                if opening is None and response is None:
                    self._cancelled_turns.discard(turn_id)
            if response_cleanup is not None:
                await asyncio.shield(response_cleanup)

    async def cancel(self, turn_id: str) -> None:
        self._validate_turn_id(turn_id)
        cleanups: list[asyncio.Task[None]] = []
        async with self._state_lock:
            response = self._responses.get(turn_id)
            read_operation = self._reads.get(turn_id)
            opening = self._openings.get(turn_id)
            opening_cleanup = self._opening_cleanups.get(turn_id)
            response_cleanup = self._response_cleanups.get(turn_id)
            if response is not None or opening is not None:
                self._cancelled_turns.add(turn_id)
            if opening is not None:
                opening_cleanup = self._ensure_opening_cleanup_locked(turn_id, opening)
            if response is not None:
                response_cleanup = self._ensure_response_cleanup_locked(
                    turn_id, response, read_operation
                )
            if opening_cleanup is not None:
                cleanups.append(opening_cleanup)
            if response_cleanup is not None:
                cleanups.append(response_cleanup)
        if cleanups:
            await asyncio.shield(asyncio.gather(*cleanups))

    async def close(self) -> None:
        operation = self._close_operation
        if operation is None or self._close_failed(operation):
            operation = asyncio.create_task(
                self._close_owned(),
                name="ollama-inference-close",
            )
            self._close_operation = operation
        await asyncio.shield(operation)

    @staticmethod
    def _close_failed(operation: asyncio.Task[None]) -> bool:
        if not operation.done():
            return False
        if operation.cancelled():
            return True
        return operation.exception() is not None

    async def _close_owned(self) -> None:
        async with self._state_lock:
            self._closed = True
            cleanups = [
                self._ensure_opening_cleanup_locked(turn_id, opening)
                for turn_id, opening in tuple(self._openings.items())
            ]
            cleanups.extend(
                self._ensure_response_cleanup_locked(
                    turn_id, response, self._reads.get(turn_id)
                )
                for turn_id, response in tuple(self._responses.items())
            )
        results = await asyncio.gather(*cleanups, return_exceptions=True)
        errors = [
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Ollama inference cleanup failed", errors)

    def _ensure_response_cleanup_locked(
        self,
        turn_id: str,
        response: _LineResponse,
        read_operation: asyncio.Task[bytes] | None,
    ) -> asyncio.Task[None]:
        cleanup = self._response_cleanups.get(turn_id)
        if cleanup is None:
            cleanup = asyncio.create_task(
                self._close_response(turn_id, response, read_operation),
                name=f"ollama-response-cleanup:{turn_id}",
            )
            self._response_cleanups[turn_id] = cleanup
        return cleanup

    async def _close_response(
        self,
        turn_id: str,
        response: _LineResponse,
        read_operation: asyncio.Task[bytes] | None,
    ) -> None:
        errors: list[BaseException] = []
        try:
            await asyncio.to_thread(response.close)
        except BaseException as error:
            errors.append(error)
        reader_error: BaseException | None = None
        if read_operation is not None:
            try:
                await asyncio.shield(read_operation)
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                reader_error = error
        async with self._state_lock:
            cancelled = turn_id in self._cancelled_turns
            if reader_error is not None and not cancelled:
                errors.append(reader_error)
            if (
                read_operation is not None
                and self._reads.get(turn_id) is read_operation
            ):
                del self._reads[turn_id]
            if not errors:
                if self._responses.get(turn_id) is response:
                    del self._responses[turn_id]
                current = asyncio.current_task()
                if self._response_cleanups.get(turn_id) is current:
                    del self._response_cleanups[turn_id]
                self._cancelled_turns.discard(turn_id)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Ollama response cleanup failed", errors)

    def _ensure_opening_cleanup_locked(
        self,
        turn_id: str,
        opening: asyncio.Task[_LineResponse],
    ) -> asyncio.Task[None]:
        cleanup = self._opening_cleanups.get(turn_id)
        if cleanup is None:
            cleanup = asyncio.create_task(
                self._close_opening(turn_id, opening),
                name=f"ollama-open-cleanup:{turn_id}",
            )
            self._opening_cleanups[turn_id] = cleanup
        return cleanup

    async def _close_opening(
        self,
        turn_id: str,
        opening: asyncio.Task[_LineResponse],
    ) -> None:
        response: _LineResponse | None = None
        with suppress(Exception):
            response = await asyncio.shield(opening)
        if response is not None:
            await asyncio.to_thread(response.close)
        async with self._state_lock:
            if self._openings.get(turn_id) is opening:
                del self._openings[turn_id]
            current = asyncio.current_task()
            if self._opening_cleanups.get(turn_id) is current:
                del self._opening_cleanups[turn_id]

    @staticmethod
    def _urlopen(request: Request, timeout: float) -> _LineResponse:
        opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        return cast(_LineResponse, opener.open(request, timeout=timeout))

    @staticmethod
    def _validate_turn_id(turn_id: object) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id.strip() or len(turn_id) > 128:
            raise ValueError("turn_id must contain 1 to 128 characters")

    @staticmethod
    def _trusted_snapshot(value: object) -> ConversationContextSnapshot:
        if type(value) is ConversationInferenceRequest:
            return ConversationInferenceRequest(
                revision=value.revision,
                messages=value.messages,
                active_tasks=value.active_tasks,
                terminal_task_count=value.terminal_task_count,
                updates=value.updates,
            )
        if type(value) is not ConversationContextSnapshot:
            raise TypeError(
                "snapshot must be an exact ConversationContextSnapshot or inference request"
            )
        return ConversationContextSnapshot(
            revision=value.revision,
            messages=value.messages,
            active_tasks=value.active_tasks,
            terminal_task_count=value.terminal_task_count,
        )

    @staticmethod
    def _messages(snapshot: ConversationContextSnapshot) -> list[dict[str, str]]:
        messages = [
            {"role": message.role, "content": message.text}
            for message in snapshot.messages
        ]
        if snapshot.active_tasks:
            tasks = "\n".join(
                f"- {task.task_id}: {task.objective}"
                for task in snapshot.active_tasks
            )
            messages.append(
                {
                    "role": "system",
                    "content": f"Active background work:\n{tasks}",
                }
            )
        if type(snapshot) is ConversationInferenceRequest and snapshot.updates:
            updates = "\n".join(
                f"- {update.status}: {update.text}" for update in snapshot.updates
            )
            messages.append(
                {
                    "role": "system",
                    "content": f"Authoritative background updates:\n{updates}",
                }
            )
        return messages

    @staticmethod
    def _payload(raw_line: object) -> dict[str, object]:
        if type(raw_line) is not bytes:
            raise TypeError("Ollama response line must be exact bytes")
        if len(raw_line) > 1_048_576:
            raise ValueError("Ollama response line exceeds supported size")
        parsed = json.loads(raw_line)
        if type(parsed) is not dict:
            raise TypeError("Ollama response line must contain an exact object")
        return cast(dict[str, object], parsed)

    @staticmethod
    def _content(payload: dict[str, object]) -> tuple[str, bool]:
        done = payload.get("done", False)
        if type(done) is not bool:
            raise TypeError("Ollama done marker must be an exact boolean")
        message = payload.get("message")
        if type(message) is not dict:
            raise TypeError("Ollama response must contain an exact message object")
        content = message.get("content", "")
        if type(content) is not str:
            raise TypeError("Ollama content must be an exact built-in string")
        return content, done

    def _extract_segments(self, buffer: str, *, final: bool) -> tuple[list[str], str]:
        segments: list[str] = []
        while buffer:
            sentence_end = first_speakable_sentence_end(buffer)
            if sentence_end is not None and sentence_end <= self._max_segment_chars:
                candidate = buffer[:sentence_end].strip()
                buffer = buffer[sentence_end:].lstrip()
            elif len(buffer) > self._max_segment_chars:
                split = buffer.rfind(" ", 0, self._max_segment_chars + 1)
                if split <= 0:
                    split = self._max_segment_chars
                candidate = buffer[:split].strip()
                buffer = buffer[split:].lstrip()
            elif final:
                candidate = buffer.strip()
                buffer = ""
            else:
                break
            if candidate:
                segments.append(candidate)
        return segments, buffer
