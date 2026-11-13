"""Measure whether Codex foreground turns can hit the provider prompt cache.

An operator launches this explicitly; it spends model quota (six short requests) and is never
run by tests or gates. It sends only synthetic text and prints only token counts:

  (a) the production adapter's current request shape, two consecutive turns;
  (b) one input sent twice on one persisted thread;
  (c) one input sent on two fresh ephemeral threads, identical but for the thread.

The second request of (b) and of (c) decides between the outcomes in ``VERDICT_MEANINGS``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hermes_realtime import __version__
from hermes_realtime.conversation.context import ConversationMessage
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    CodexJsonLineTransport,
    SubprocessCodexJsonLineTransport,
    _resolve_codex_executable,
)

QUOTA_FLAG = "--i-understand-this-spends-model-quota"
REQUEST_COUNT = 6
MINIMUM_CACHEABLE_INPUT_TOKENS = 1024
VERDICT_MEANINGS = {
    "no_cache_reported": "b=0, c=0: no cached input was reported within the pause",
    "affinity_survives_new_thread": "b>0, c>0: a new thread can reuse the cache",
    "affinity_per_thread": "b>0, c=0: only requests on one thread reuse the cache",
    "inconclusive": "a deciding count is missing, too small, or contradicts the table",
}

_STAGES = ("a1", "a2", "b1", "b2", "c1", "c2")
_MAX_COUNT = (1 << 53) - 1
_NONCE = re.compile(r"[0-9a-f]{16}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_USER_AGENT_VERSION = re.compile(
    r"[A-Za-z0-9_.-]{1,64}/(\d{1,4}\.\d{1,4}\.\d{1,4}(?:-[A-Za-z0-9.]{1,32})?)(?:\s|$)"
)
_RAW_BASE_INSTRUCTIONS = "You are a concise assistant. Never use tools."
_RAW_DEVELOPER_INSTRUCTIONS = "Reply in one short sentence."

TransportFactory = Callable[[str], Awaitable[CodexJsonLineTransport]]


def synthetic_text(case: str, nonce: str) -> str:
    """Return inert filler well above the cacheable minimum, unique from its first line.

    The per-run nonce keeps an earlier run's cache entries from answering this one, and the
    case label keeps the cases from answering each other.
    """

    if type(nonce) is not str or _NONCE.fullmatch(nonce) is None:
        raise ValueError("nonce must be 16 lowercase hexadecimal characters")
    lines = [f"Synthetic cache probe {nonce} case {case}."]
    lines.extend(
        f"Calibration line {index:03d} holds the neutral value {index:03d} for this exercise."
        for index in range(1, 221)
    )
    lines.append("Reply with the single word: ready.")
    return "\n".join(lines)


@dataclass(slots=True)
class _RequestRecord:
    case: str
    request: int
    model: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    usage_updates: int = 0


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_COUNT else None


class _Recorder:
    """Keep counts and two fixed-format labels from the wire; never text or identities."""

    def __init__(self) -> None:
        self.codex_version: str | None = None
        self.records = {stage: _RequestRecord(stage[0], int(stage[1])) for stage in _STAGES}
        self.stage: str | None = None
        self._pending: dict[int, str] = {}

    def sent(self, message: Mapping[str, object]) -> None:
        request_id, method = message.get("id"), message.get("method")
        if type(request_id) is int and type(method) is str:
            self._pending[request_id] = method

    def received(self, message: Mapping[str, object]) -> None:
        method = message.get("method")
        if method is None:
            request_id = message.get("id")
            requested = self._pending.pop(request_id, None) if type(request_id) is int else None
            result = message.get("result")
            if type(result) is not dict:
                return
            if requested == "initialize":
                agent = result.get("userAgent")
                match = _USER_AGENT_VERSION.match(agent) if type(agent) is str else None
                self.codex_version = None if match is None else match.group(1)
            elif requested == "thread/start" and self.stage is not None:
                model = result.get("model")
                if type(model) is str and _MODEL.fullmatch(model) is not None:
                    self.records[self.stage].model = model
            return
        if method != "thread/tokenUsage/updated" or self.stage is None:
            return
        params = message.get("params")
        usage = params.get("tokenUsage") if type(params) is dict else None
        last = usage.get("last") if type(usage) is dict else None
        if type(last) is not dict:
            return
        record = self.records[self.stage]
        record.usage_updates += 1
        record.input_tokens = _count(last.get("inputTokens"))
        record.cached_input_tokens = _count(last.get("cachedInputTokens"))
        record.cache_write_input_tokens = _count(last.get("cacheWriteInputTokens"))


class _ObservedTransport:
    def __init__(self, inner: CodexJsonLineTransport, recorder: _Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def send(self, message: Mapping[str, object]) -> None:
        self._recorder.sent(message)
        await self._inner.send(message)

    async def receive(self) -> Mapping[str, object]:
        message: Mapping[str, object] = await self._inner.receive()
        self._recorder.received(message)
        return message

    async def close(self) -> None:
        await self._inner.close()


class _RawClient:
    """The few requests the production adapter has no path for: thread reuse and persistence."""

    def __init__(self, transport: CodexJsonLineTransport, *, model: str, effort: str,
                 cwd: str) -> None:
        self._transport = transport
        self._model = model
        self._effort = effort
        self._cwd = cwd
        self._next_request_id = 1

    async def _send_request(self, method: str, params: Mapping[str, object]) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        await self._transport.send({"id": request_id, "method": method, "params": dict(params)})
        return request_id

    @staticmethod
    def _response(message: Mapping[str, object], request_id: int) -> dict[str, object]:
        result = message.get("result")
        if message.get("id") != request_id or "error" in message or type(result) is not dict:
            raise RuntimeError("Codex app-server rejected a probe request")
        return result

    async def _request(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        request_id = await self._send_request(method, params)
        while True:
            message = await self._transport.receive()
            if "method" not in message:
                return self._response(message, request_id)
            if "id" in message:
                raise RuntimeError("Codex app-server sent an unexpected request")

    async def initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "clientInfo": {"name": "hermes-realtime", "version": __version__},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self._transport.send({"method": "initialized"})

    async def start_thread(self, *, ephemeral: bool) -> str:
        result = await self._request(
            "thread/start",
            {
                "model": self._model,
                "modelProvider": "openai",
                "cwd": self._cwd,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": ephemeral,
                "allowProviderModelFallback": False,
                "baseInstructions": _RAW_BASE_INSTRUCTIONS,
                "developerInstructions": _RAW_DEVELOPER_INSTRUCTIONS,
                "dynamicTools": [],
                "experimentalRawEvents": False,
            },
        )
        thread = result.get("thread")
        thread_id = thread.get("id") if type(thread) is dict else None
        if type(thread_id) is not str or not thread_id or result.get("model") != self._model:
            raise RuntimeError("Codex app-server returned an unexpected thread")
        return thread_id

    async def turn(self, thread_id: str, text: str) -> None:
        request_id = await self._send_request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
                "effort": self._effort,
                "summary": "none",
            },
        )
        acknowledged = False
        while True:
            message = await self._transport.receive()
            if "method" not in message:
                self._response(message, request_id)
                acknowledged = True
                continue
            if "id" in message:
                raise RuntimeError("Codex app-server sent an unexpected request")
            if message["method"] != "turn/completed":
                continue
            params = message.get("params")
            turn = params.get("turn") if type(params) is dict else None
            if (
                not acknowledged
                or type(params) is not dict
                or params.get("threadId") != thread_id
                or type(turn) is not dict
                or turn.get("status") != "completed"
            ):
                raise RuntimeError("Codex probe turn did not complete")
            return


def _snapshot(revision: int, *texts: str) -> ConversationInferenceRequest:
    messages: list[ConversationMessage] = []
    for text in texts:
        if messages:
            messages.append(ConversationMessage(role="assistant", text="Ready."))
        messages.append(ConversationMessage(role="user", text=text))
    return ConversationInferenceRequest(
        revision=revision, messages=tuple(messages), active_tasks=()
    )


async def _run_cases(
    recorder: _Recorder,
    *,
    transport_factory: TransportFactory,
    model: str,
    effort: str,
    cwd: str,
    pause_seconds: float,
    sleep: Callable[[float], Awaitable[None]],
    nonce: str,
) -> None:
    async def observed() -> CodexJsonLineTransport:
        return _ObservedTransport(await transport_factory(cwd), recorder)

    first, follow_up = synthetic_text("a", nonce), synthetic_text("a-follow-up", nonce)
    inference = CodexAppServerStreamingInference(
        model=model,
        effort=effort,
        conversation_cwd=cwd,
        transport_factory=observed,
        request_timeout_seconds=60.0,
    )
    try:
        for stage, snapshot in (
            ("a1", _snapshot(1, first)),
            ("a2", _snapshot(2, first, follow_up)),
        ):
            recorder.stage = stage
            async for _ in inference.stream(snapshot, turn_id=f"turn_probe_{stage}"):
                pass
    finally:
        await inference.close()

    recorder.stage = "b1"
    transport = await observed()
    try:
        client = _RawClient(transport, model=model, effort=effort, cwd=cwd)
        await client.initialize()
        repeated = synthetic_text("b", nonce)
        persisted = await client.start_thread(ephemeral=False)
        await client.turn(persisted, repeated)
        # Both deciding requests wait equally, so a slow cache write cannot favour one case.
        await sleep(pause_seconds)
        recorder.stage = "b2"
        recorder.records["b2"].model = recorder.records["b1"].model
        await client.turn(persisted, repeated)

        identical = synthetic_text("c", nonce)
        recorder.stage = "c1"
        await client.turn(await client.start_thread(ephemeral=True), identical)
        await sleep(pause_seconds)
        recorder.stage = "c2"
        await client.turn(await client.start_thread(ephemeral=True), identical)
    finally:
        await transport.close()


def _verdict(persisted: _RequestRecord, fresh: _RequestRecord) -> str:
    counts = (
        persisted.input_tokens,
        fresh.input_tokens,
        persisted.cached_input_tokens,
        fresh.cached_input_tokens,
    )
    if any(count is None for count in counts):
        return "inconclusive"
    persisted_input, fresh_input, same_thread, new_thread = (count or 0 for count in counts)
    if min(persisted_input, fresh_input) <= MINIMUM_CACHEABLE_INPUT_TOKENS:
        return "inconclusive"
    if same_thread == 0 and new_thread == 0:
        return "no_cache_reported"
    if same_thread > 0 and new_thread > 0:
        return "affinity_survives_new_thread"
    if same_thread > 0:
        return "affinity_per_thread"
    return "inconclusive"


async def run_probe(
    *,
    transport_factory: TransportFactory,
    model: str,
    effort: str,
    pause_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, object]:
    recorder = _Recorder()
    failure: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(
        prefix="hermes-realtime-cache-probe-", ignore_cleanup_errors=True
    ) as workspace:
        try:
            async with asyncio.timeout(timeout_seconds):
                await _run_cases(
                    recorder,
                    transport_factory=transport_factory,
                    model=model,
                    effort=effort,
                    cwd=str(Path(workspace).resolve()),
                    pause_seconds=pause_seconds,
                    sleep=sleep,
                    nonce=secrets.token_hex(8),
                )
        except Exception as error:
            # The type name only: provider messages can carry text, paths or identities.
            failure = {"stage": recorder.stage or "launch", "error": type(error).__name__}
    verdict = (
        "inconclusive"
        if failure is not None
        else _verdict(recorder.records["b2"], recorder.records["c2"])
    )
    report: dict[str, object] = {
        "probe": "codex-prompt-cache/1",
        "status": "completed" if failure is None else "failed",
        "codexVersion": recorder.codex_version,
        "minimumCacheableInputTokens": MINIMUM_CACHEABLE_INPUT_TOKENS,
        "requests": [
            {
                "case": record.case,
                "request": record.request,
                "model": record.model,
                "inputTokens": record.input_tokens,
                "cachedInputTokens": record.cached_input_tokens,
                "cacheWriteInputTokens": record.cache_write_input_tokens,
                "usageUpdates": record.usage_updates,
            }
            for record in recorder.records.values()
        ],
        "verdict": verdict,
        "verdictMeaning": VERDICT_MEANINGS[verdict],
    }
    if failure is not None:
        report["failure"] = failure
    return report


def _bounded_seconds(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(text: str) -> float:
        value = float(text)
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum:g} and {maximum:g}")
        return value

    return parse


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(QUOTA_FLAG, dest="acknowledged", action="store_true",
                        help=f"required: the probe sends {REQUEST_COUNT} model requests")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--codex-executable", default=None)
    parser.add_argument("--pause-seconds", type=_bounded_seconds(0, 60), default=5.0,
                        help="wait before the second request of case (b) and of case (c)")
    parser.add_argument("--timeout-seconds", type=_bounded_seconds(1, 900), default=300.0,
                        help="hard limit for the whole probe")
    args = parser.parse_args(argv)
    if not args.acknowledged:
        print(
            f"Refusing to run: this probe sends {REQUEST_COUNT} model requests and spends "
            f"quota. Pass {QUOTA_FLAG} to proceed.",
            file=sys.stderr,
        )
        return 2
    executable = _resolve_codex_executable(args.codex_executable)

    async def transport_factory(cwd: str) -> CodexJsonLineTransport:
        return await SubprocessCodexJsonLineTransport.create(executable=executable, cwd=cwd)

    report = asyncio.run(
        run_probe(
            transport_factory=transport_factory,
            model=args.model,
            effort=args.effort,
            pause_seconds=args.pause_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
