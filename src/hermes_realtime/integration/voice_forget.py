"""Off-turn retry of one durable voice delete intent until Hermes verifies completion."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Protocol

from hermes_realtime.integration.bridge import LocalHermesBridgeClient
from hermes_realtime.integration.voice_tail import VoiceTailWriter
from hermes_realtime.protocol import (
    VOICE_FORGET_CAPABILITY,
    VoiceForgetAckEvent,
    VoiceForgetEvent,
    VoiceForgetRefusedEvent,
)

_MARKER = "[voice-forget-send] "


class VoiceForgetLink(Protocol):
    @property
    def capabilities(self) -> frozenset[str]: ...

    async def forget(
        self, event: VoiceForgetEvent
    ) -> VoiceForgetAckEvent | VoiceForgetRefusedEvent: ...

    async def close(self) -> None: ...


def forget_connector(
    *, host: str, port: int, token: str
) -> Callable[[], Awaitable[VoiceForgetLink]]:
    async def connect() -> VoiceForgetLink:
        return await LocalHermesBridgeClient.connect(
            host=host,
            port=port,
            token=token,
            participant_id="voice-forget",
            capabilities=(VOICE_FORGET_CAPABILITY,),
        )

    return connect


def _marker(evidence: dict[str, str | int]) -> None:
    print(
        _MARKER + json.dumps(evidence | {"version": 1}, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


class VoiceForgetSender:
    """Resend every persisted delete, unchanged, until each has an exact complete ack.

    The link is held open while idle, so ``negotiated`` reports whether the companion
    offers ``voice_forget`` now, not whether it did at the last delete. Each round sends
    every pending delete, so one that stays pending never holds up a later one.
    """

    def __init__(
        self,
        writer: VoiceTailWriter,
        connect: Callable[[], Awaitable[VoiceForgetLink]],
        *,
        reply_timeout: float = 30.0,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if type(writer) is not VoiceTailWriter:
            raise TypeError("writer must be an exact VoiceTailWriter")
        if not callable(connect) or not callable(sleep):
            raise TypeError("connect and sleep must be callable")
        values = (reply_timeout, initial_backoff_seconds, max_backoff_seconds)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise TypeError("forget timing must be finite exact numbers")
        if not 0 < reply_timeout <= 3600:
            raise ValueError("forget reply timeout must be positive and bounded")
        if not 0 < initial_backoff_seconds <= max_backoff_seconds <= 300:
            raise ValueError("forget backoff must be positive and bounded")
        self._writer = writer
        self._connect = connect
        self._timeout = float(reply_timeout)
        self._initial_backoff = float(initial_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._link: VoiceForgetLink | None = None
        self._last_failure: str | None = None

    @property
    def negotiated(self) -> bool:
        """Whether a live link to the companion negotiated ``voice_forget``.

        Only a link that offers the capability is ever held, so holding one is the answer.
        """
        return self._link is not None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("forget sender starts at most once")
        self._task = asyncio.create_task(self._run(), name="voice-forget-sender")

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._drop()

    async def _drop(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            with suppress(Exception):
                await link.close()

    async def _run(self) -> None:
        backoff = self._initial_backoff
        while True:
            try:
                link = self._link
                if link is None:
                    async with asyncio.timeout(self._timeout):
                        link = await self._connect()
                    if VOICE_FORGET_CAPABILITY not in link.capabilities:
                        with suppress(Exception):
                            await link.close()
                        raise RuntimeError("forget capability unavailable")
                    self._link = link
                pending = await self._writer.next_deletes()
                settled = 0
                for binding in pending:
                    settled += await self._settle(link, binding)
                if settled == len(pending):
                    self._last_failure = None
                    backoff = self._initial_backoff
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                reason = type(error).__name__
                if self._last_failure != reason:
                    _marker({"outcome": "unknown", "cause": reason})
                self._last_failure = reason
                await self._drop()
            await self._sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def _settle(self, link: VoiceForgetLink, binding: tuple[str, int]) -> bool:
        """Send one delete; True once the companion verified it complete and it is settled."""

        conversation_id, generation = binding
        event = VoiceForgetEvent(
            protocol_version="0.3",
            type="voice_forget",
            conversation_id=conversation_id,
            generation=generation,
        )
        async with asyncio.timeout(self._timeout):
            reply = await link.forget(event)
        if (
            type(reply) not in (VoiceForgetAckEvent, VoiceForgetRefusedEvent)
            or (reply.conversation_id, reply.generation) != binding
        ):
            raise RuntimeError("forget reply named another binding")
        if type(reply) is VoiceForgetAckEvent and reply.state == "complete":
            if self._writer.acknowledge_forget(binding):
                return True
            raise RuntimeError("forget intent changed before acknowledgment")
        reason = reply.category if type(reply) is VoiceForgetRefusedEvent else "pending"
        if self._last_failure != reason:
            _marker({"outcome": reason})
        self._last_failure = reason
        return False


__all__ = ["VoiceForgetLink", "VoiceForgetSender", "forget_connector"]
