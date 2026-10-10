"""Restart recall through the real speech loop, Ollama adapter, context store and tail.

Only Ollama's HTTP is faked. One heard row per turn keeps an early user statement inside
the window that a restart restores; with one row per sentence, a few multi-sentence
replies would evict it before the crash.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import cast
from urllib.request import Request

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ConversationMessage,
    DurableConversation,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.integration.voice_tail import (
    ArchiveOutbox,
    ReviewProgress,
    VoiceTailWriter,
    parse_voice_tail,
    voice_tail_bytes,
)
from hermes_realtime.providers import OllamaStreamingInference
from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk, Transcript

_FACT = "My favorite bird is the kestrel."


class _LineResponse:
    def __init__(self, content: str) -> None:
        payload = {"message": {"content": content}, "done": True, "prompt_eval_count": 7}
        self._lines = deque([json.dumps(payload).encode("utf-8") + b"\n"])

    def readline(self, limit: int = -1, /) -> bytes:
        del limit
        return self._lines.popleft() if self._lines else b""

    def close(self) -> None:
        self._lines.clear()


class _FakeOllama:
    """Answer each request with the next scripted reply and keep every request body."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = deque(replies)
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: Request, timeout: float) -> _LineResponse:
        del timeout
        self.bodies.append(json.loads(cast(bytes, request.data)))
        return _LineResponse(self._replies.popleft())


class _Synthesizer:
    def __init__(self) -> None:
        self._count = 0

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self._count += 1
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"chunk_{self._count}",
            text=text,
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
        )

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        del turn_id


class _Playback:
    async def play(self, chunk: SpeechChunk, *, is_valid: Callable[[], bool]) -> None:
        del chunk
        if not is_valid():
            raise asyncio.CancelledError

    async def cancel(self, turn_id: str) -> None:
        del turn_id


def _loop(store: ConversationContextStore, ollama: _FakeOllama) -> StreamingSpeechLoop:
    return StreamingSpeechLoop(
        context=store,
        foreground=ForegroundTurnCoordinator(),
        inference=OllamaStreamingInference(
            base_url="http://127.0.0.1:11434", model="stand-in", open_request=ollama
        ),
        synthesizer=_Synthesizer(),
        playback=_Playback(),
        ledger=DeliveredSpeechLedger(),
    )


def _reply(topic: int) -> str:
    return " ".join(f"Topic {topic} has detail {index}." for index in range(1, 6))


@pytest.mark.asyncio
async def test_idle_speech_cancel_preserves_active_work_reference_in_next_prompt() -> None:
    store = ConversationContextStore()
    store.record_user_transcript(Transcript(text="Synthetic earlier topic.", final=True))
    store.record_task_accepted(task_id="task_1", run_id="deleg_1", objective="Synthetic objective.")
    ollama = _FakeOllama(["Synthetic answer."])
    loop = _loop(store, ollama)
    await loop.cancel()  # Nothing is playing; speech scope cannot clear named-task context.
    await loop.respond(
        "turn_after_cancel", Transcript(text="Synthetic current question.", final=True),
    )
    await loop.close()
    messages = cast(list[dict[str, str]], ollama.bodies[0]["messages"])
    sections = json.loads(messages[0]["content"].split("\n\nReference data:\n", 1)[1])
    assert sections["active_work"] == [{"task_id": "task_1", "objective": "Synthetic objective."}]
    assert OllamaStreamingInference.prompt_metadata(messages)["reference_kinds"] == ["active_work"]
    assert messages[-1] == {"role": "user", "content": "Synthetic current question."}


def _tail(path: Path) -> tuple[DurableConversation, ArchiveOutbox]:
    tail = parse_voice_tail(path.read_bytes(), max_messages=32, max_item_chars=1024)
    assert tail is not None and tail.archive is not None
    return tail.conversation, tail.archive


@pytest.mark.asyncio
async def test_a_restart_restores_an_early_user_statement_into_the_ollama_prompt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    replies = ["Noted. I will remember that.", *(_reply(topic) for topic in range(2, 9))]
    writer = VoiceTailWriter(path, max_outbox_rows=128)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        loop = _loop(store, _FakeOllama(replies))
        await loop.respond("turn_001", Transcript(text=_FACT, final=True))
        for topic in range(2, 9):
            await loop.respond(
                f"turn_{topic:03d}", Transcript(text=f"Tell me about topic {topic}.", final=True)
            )
        await loop.close()
    finally:
        # The last write lands; a crash after it leaves exactly this tail.
        await writer.close()

    conversation, archive = _tail(path)
    # One row per turn: eight user rows and eight whole replies, each with one seq.
    assert [message.role for message in conversation.messages] == ["user", "assistant"] * 8
    assert conversation.messages[1].text == replies[0]
    assert conversation.messages[-1].text == replies[-1]
    assert [(row.seq, row.role) for row in archive.rows] == [
        (seq, "user" if seq % 2 == 0 else "assistant") for seq in range(16)
    ]

    ollama = _FakeOllama(["The kestrel."])
    successor = VoiceTailWriter(path, max_outbox_rows=128)
    restored = ConversationContextStore(on_change=successor.update)
    await successor.open(restored)
    try:
        loop = _loop(restored, ollama)
        await loop.respond("turn_101", Transcript(text="What is my favorite bird?", final=True))
        await loop.close()
    finally:
        await successor.close()

    messages = cast(list[dict[str, str]], ollama.bodies[0]["messages"])
    assert {"role": "user", "content": _FACT} in messages
    assert messages[-1] == {"role": "user", "content": "What is my favorite bird?"}
    assert ollama.bodies[0]["options"] == {"num_ctx": 16_384}


@pytest.mark.asyncio
async def test_an_older_per_sentence_tail_restores_and_seq_continues_one_per_turn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice-tail.json"
    # An older build's tail: one row per sentence, all closed and already numbered 0-15.
    old = DurableConversation(
        messages=tuple(
            ConversationMessage("user" if index % 4 == 0 else "assistant", f"Old {index}.")
            for index in range(16)
        ),
        prior_work=False,
    )
    path.write_bytes(
        voice_tail_bytes(
            old,
            ArchiveOutbox(
                conversation_id="conv",
                generation=0,
                next_seq=16,
                settled=16,
                cursor=15,
                rows=(),
                frozen=0,
                gap=None,
                review=ReviewProgress(cursor=15),
            ),
        )
    )
    replies = [_reply(topic) for topic in range(1, 10)]
    writer = VoiceTailWriter(path, max_outbox_rows=128)
    store = ConversationContextStore(on_change=writer.update)
    await writer.open(store)
    try:
        loop = _loop(store, _FakeOllama(replies))
        await loop.respond("turn_001", Transcript(text="New question 1.", final=True))
        conversation = store.durable_view()
        assert conversation.messages[:16] == old.messages
        assert [message.text for message in conversation.messages[16:]] == [
            "New question 1.",
            replies[0],
        ]
        for turn in range(2, 10):
            await loop.respond(
                f"turn_{turn:03d}", Transcript(text=f"New question {turn}.", final=True)
            )
        await loop.close()
    finally:
        await writer.close()

    conversation, archive = _tail(path)
    # Nine new turns are 18 rows: with the 32-row cap, the two oldest rows aged out.
    assert conversation.messages[:14] == old.messages[2:]
    assert [(row.seq, row.role, row.text) for row in archive.rows] == [
        (16 + 2 * turn + offset, role, text)
        for turn in range(9)
        for offset, (role, text) in enumerate(
            (("user", f"New question {turn + 1}."), ("assistant", replies[turn]))
        )
    ]
