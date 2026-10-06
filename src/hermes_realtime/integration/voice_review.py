"""Durable review requests over already acknowledged voice archive ranges."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from collections.abc import Awaitable, Callable
from typing import Protocol

from hermes_realtime.integration.bridge import LocalHermesBridgeClient
from hermes_realtime.integration.voice_tail import ReviewRange, VoiceTailWriter
from hermes_realtime.protocol import (
    VOICE_REVIEW_CAPABILITY,
    VOICE_TRANSIENT_REFUSALS,
    VoiceReviewAckEvent,
    VoiceReviewEvent,
    VoiceReviewRefusedEvent,
)

_MARKER = "[voice-review-send] "


class VoiceReviewLink(Protocol):
    @property
    def capabilities(self) -> frozenset[str]: ...

    @property
    def review_interval(self) -> int | None: ...

    async def review(
        self, event: VoiceReviewEvent
    ) -> VoiceReviewAckEvent | VoiceReviewRefusedEvent: ...

    async def close(self) -> None: ...


def _marker(evidence: dict[str, str | int]) -> None:
    print(_MARKER + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


def review_idle_allowed(*, speech_active: bool, foreground_tasks: int) -> bool:
    """A quiet voice turn can end while independently owned named work continues."""
    if type(speech_active) is not bool or type(foreground_tasks) is not int:
        raise TypeError("review idle state must have exact types")
    if foreground_tasks < 0:
        raise ValueError("foreground task count must be nonnegative")
    return not speech_active and foreground_tasks == 0


def review_connector(
    *, host: str, port: int, token: str
) -> Callable[[], Awaitable[VoiceReviewLink]]:
    async def connect() -> VoiceReviewLink:
        return await LocalHermesBridgeClient.connect(
            host=host,
            port=port,
            token=token,
            participant_id="voice-review",
            capabilities=(VOICE_REVIEW_CAPABILITY,),
        )

    return connect


class VoiceReviewSender:
    """One background task; refused or uncertain coverage remains frozen in the tail."""

    def __init__(
        self,
        writer: VoiceTailWriter,
        connect: Callable[[], Awaitable[VoiceReviewLink]],
        *,
        idle_allowed: Callable[[], bool],
        idle_seconds: float = 300.0,
        reply_timeout: float = 30.0,
        close_timeout: float = 5.0,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        if type(writer) is not VoiceTailWriter:
            raise TypeError("writer must be an exact VoiceTailWriter")
        if not callable(connect) or not callable(idle_allowed):
            raise TypeError("connect and idle_allowed must be callable")
        if any(
            type(v) not in (int, float)
            for v in (
                idle_seconds,
                reply_timeout,
                close_timeout,
                initial_backoff_seconds,
                max_backoff_seconds,
            )
        ) or not all(
            math.isfinite(v)
            for v in (
                idle_seconds,
                reply_timeout,
                close_timeout,
                initial_backoff_seconds,
                max_backoff_seconds,
            )
        ):
            raise ValueError("review timing must be finite")
        if not 0 < idle_seconds <= 3600 or not 0 < reply_timeout <= 3600:
            raise ValueError("review idle and reply timeouts must be bounded")
        if not 0 < close_timeout <= 30:
            raise ValueError("review close timeout must be bounded")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= 300:
            raise ValueError("review backoff must be bounded")
        self._writer = writer
        self._connect = connect
        self._idle_allowed = idle_allowed
        self._idle_seconds = idle_seconds
        self._timeout = reply_timeout
        self._close_timeout = close_timeout
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._task: asyncio.Task[None] | None = None
        self._link: VoiceReviewLink | None = None
        self._last_refusal: str | None = None
        self._blocked = False

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("review sender starts at most once")
        self._task = asyncio.create_task(self._run(), name="voice-review-sender")

    async def close(self) -> None:
        self._writer.request_review_close()
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._close_timeout)
            except asyncio.CancelledError:
                owner = asyncio.current_task()
                if not task.done() or (owner is not None and owner.cancelling()):
                    raise
            except TimeoutError:
                raise RuntimeError("review sender did not stop before close timeout") from None
            if not task.done():
                raise RuntimeError("review sender is still running")
            self._task = None
        await self._drop()

    async def _drop(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(self._close_timeout):
                    await link.close()

    async def _run(self) -> None:
        backoff = self._initial_backoff
        while True:
            try:
                self._maybe_close_idle()
                if self._blocked:
                    await asyncio.sleep(min(self._idle_seconds, 1.0))
                    continue
                async with asyncio.timeout(self._timeout):
                    link = await self._connect()
                self._link = link
                if (
                    VOICE_REVIEW_CAPABILITY not in link.capabilities
                    or type(link.review_interval) is not int
                ):
                    raise RuntimeError("review capability or interval unavailable")
                interval = link.review_interval
                assert interval is not None
                while True:
                    # The timer measures the last actual store change, not a poll.
                    self._maybe_close_idle()
                    try:
                        async with asyncio.timeout(min(self._idle_seconds, 1.0)):
                            request = await self._writer.next_review(interval)
                    except TimeoutError:
                        continue
                    event = VoiceReviewEvent(
                        protocol_version="0.3",
                        type="voice_review",
                        conversation_id=request.conversation_id,
                        generation=request.generation,
                        seq_from=request.seq_from,
                        seq_through=request.seq_through,
                        memory=True,
                        skills=True,
                        closing=request.closing,
                    )
                    async with asyncio.timeout(self._timeout):
                        reply = await link.review(event)
                    if not _names(reply, request):
                        raise RuntimeError("review reply named another range")
                    if type(reply) is VoiceReviewAckEvent:
                        if not self._writer.acknowledge_review(request):
                            _marker({"refusal": "review_capacity", "version": 1})
                            await self._drop()
                            self._blocked = True
                            break
                        self._last_refusal = None
                        backoff = self._initial_backoff
                        continue
                    assert type(reply) is VoiceReviewRefusedEvent
                    if self._last_refusal != reply.category:
                        _marker(
                            {
                                "refusal": reply.category,
                                "closing": int(request.closing),
                                "version": 1,
                            }
                        )
                    self._last_refusal = reply.category
                    if reply.category != "busy" and reply.category not in VOICE_TRANSIENT_REFUSALS:
                        await self._drop()
                        self._blocked = True
                        break
                    raise RuntimeError("review refused")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._last_refusal is None:
                    _marker({"outcome": "retained", "cause": type(error).__name__, "version": 1})
                await self._drop()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    def _maybe_close_idle(self) -> None:
        if (
            self._writer.idle_for() >= self._idle_seconds
            and not self._writer.has_unsettled_rows
            and self._idle_allowed()
        ):
            self._writer.request_review_close()


def _names(reply: VoiceReviewAckEvent | VoiceReviewRefusedEvent, request: ReviewRange) -> bool:
    return type(reply) in (VoiceReviewAckEvent, VoiceReviewRefusedEvent) and (
        reply.conversation_id,
        reply.generation,
        reply.seq_from,
        reply.seq_through,
        reply.closing,
    ) == (
        request.conversation_id,
        request.generation,
        request.seq_from,
        request.seq_through,
        request.closing,
    )


__all__ = ["VoiceReviewLink", "VoiceReviewSender", "review_connector", "review_idle_allowed"]
