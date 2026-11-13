from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from hermes_realtime.providers.current_facts import (
    contains_private_material,
    foreground_search_query,
)
from scripts import probe_codex_prompt_cache as probe

_FLAG = "--i-understand-this-spends-model-quota"
_REPLY = "Unmistakable synthetic reply."
_USER_AGENT = "codex_cli_rs/0.167.0 (PrivateOS 1.2; arch) private-terminal"
_COMPLETE = {
    "inputTokens": 3000,
    "cachedInputTokens": 0,
    "outputTokens": 5,
    "reasoningOutputTokens": 0,
    "totalTokens": 3005,
}


def _usage(**overrides: object) -> dict[str, object]:
    return {**_COMPLETE, **overrides}


class FakeAppServer:
    """One scripted app-server shared by every transport the probe opens."""

    def __init__(self, last: list[dict[str, object]], *, stall_turn: int | None = None,
                 failed_turn: int | None = None, close_seconds: float = 0.0,
                 launch_seconds: float = 0.0) -> None:
        self.last = last
        self.stall_turn = stall_turn
        self.failed_turn = failed_turn
        self.close_seconds = close_seconds
        self.launch_seconds = launch_seconds
        self.transports: list[FakeTransport] = []
        self.thread_starts: list[dict[str, Any]] = []
        self.turn_starts: list[dict[str, Any]] = []
        self.order: list[str] = []

    def transport(self) -> FakeTransport:
        created = FakeTransport(self)
        self.transports.append(created)
        return created


class FakeTransport:
    def __init__(self, server: FakeAppServer) -> None:
        self.server = server
        self.incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.closing = False
        self.closed = False

    async def send(self, message: Mapping[str, object]) -> None:
        server = self.server
        method = message.get("method")
        request_id = message.get("id")
        params: Any = message.get("params")
        if method == "initialize":
            await self.incoming.put({"id": request_id, "result": {"userAgent": _USER_AGENT}})
        elif method == "thread/start":
            server.thread_starts.append(dict(params))
            server.order.append("thread")
            thread_id = f"thread_private_{len(server.thread_starts)}"
            await self.incoming.put({
                "id": request_id,
                "result": {"thread": {"id": thread_id}, "model": params["model"],
                           "modelProvider": "openai"},
            })
        elif method == "turn/start":
            index = len(server.turn_starts)
            server.turn_starts.append(dict(params))
            server.order.append("turn")
            thread_id = params["threadId"]
            turn_id = f"turn_private_{index + 1}"
            await self.incoming.put({
                "id": request_id,
                "result": {"turn": {"id": turn_id, "items": [], "status": "inProgress"}},
            })
            if index == server.stall_turn:
                return
            await self.incoming.put({
                "method": "item/agentMessage/delta",
                "params": {"threadId": thread_id, "turnId": turn_id, "itemId": "item_1",
                           "delta": _REPLY},
            })
            await self.incoming.put({
                "method": "thread/tokenUsage/updated",
                "params": {"threadId": thread_id, "turnId": turn_id, "tokenUsage": {
                    "last": server.last[index], "modelContextWindow": 272_000,
                    "total": _COMPLETE,
                }},
            })
            status = "failed" if index == server.failed_turn else "completed"
            await self.incoming.put({
                "method": "turn/completed",
                "params": {"threadId": thread_id,
                           "turn": {"id": turn_id, "items": [], "status": status}},
            })

    async def receive(self) -> Mapping[str, object]:
        return await self.incoming.get()

    async def close(self) -> None:
        # Like the real transport: interrupted before the end, the process is still alive.
        self.closing = True
        await asyncio.sleep(self.server.close_seconds)
        self.closed = True


async def _run(server: FakeAppServer, *, timeout_seconds: float = 30.0,
               pauses: list[float] | None = None) -> dict[str, Any]:
    async def factory(cwd: str) -> FakeTransport:
        created = server.transport()
        await asyncio.sleep(server.launch_seconds)
        return created

    async def sleep(seconds: float) -> None:
        server.order.append("pause")
        if pauses is not None:
            pauses.append(seconds)

    return await probe.run_probe(
        transport_factory=factory, model="gpt-5.6-terra", effort="low",
        pause_seconds=5.0, timeout_seconds=timeout_seconds, sleep=sleep,
    )


def _six(b2: object, c2: object) -> list[dict[str, object]]:
    rows = [_usage() for _ in range(6)]
    rows[3] = _usage(cachedInputTokens=b2)
    rows[5] = _usage(cachedInputTokens=c2)
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("b2", "c2", "expected"),
    [
        (0, 0, "no_cache_reported"),
        (2048, 1900, "affinity_survives_new_thread"),
        (2048, 0, "affinity_per_thread"),
        (0, 1900, "inconclusive"),
    ],
)
async def test_verdict_follows_the_decision_table(b2: int, c2: int, expected: str) -> None:
    report = await _run(FakeAppServer(_six(b2, c2)))

    assert report["status"] == "completed"
    assert report["verdict"] == expected
    assert [(row["case"], row["request"]) for row in report["requests"]] == [
        ("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1), ("c", 2),
    ]
    assert report["requests"][3]["cachedInputTokens"] == b2
    assert report["requests"][5]["cachedInputTokens"] == c2
    assert report["codexVersion"] == "0.167.0"
    assert {row["model"] for row in report["requests"]} == {"gpt-5.6-terra"}


@pytest.mark.asyncio
async def test_absent_cache_write_is_null_and_present_is_exact() -> None:
    rows = _six(2048, 1900)
    rows[4] = _usage(cacheWriteInputTokens=2900)
    report = await _run(FakeAppServer(rows))

    writes = [row["cacheWriteInputTokens"] for row in report["requests"]]
    assert writes == [None, None, None, None, 2900, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, True, "2048", 2048.0, -1])
async def test_non_exact_deciding_value_is_null_and_inconclusive(bad: object) -> None:
    rows = _six(2048, 1900)
    if bad is None:
        del rows[5]["cachedInputTokens"]
    else:
        rows[5]["cachedInputTokens"] = bad
    report = await _run(FakeAppServer(rows))

    assert report["requests"][5]["cachedInputTokens"] is None
    assert report["verdict"] == "inconclusive"


@pytest.mark.asyncio
async def test_input_at_or_below_the_cacheable_minimum_is_inconclusive() -> None:
    rows = _six(0, 0)
    rows[3] = _usage(inputTokens=1024)
    report = await _run(FakeAppServer(rows))

    assert report["minimumCacheableInputTokens"] == 1024
    assert report["verdict"] == "inconclusive"


@pytest.mark.asyncio
async def test_request_shapes_isolate_one_variable_per_case() -> None:
    server = FakeAppServer(_six(0, 0))
    pauses: list[float] = []
    await _run(server, pauses=pauses)

    assert len(server.turn_starts) == 6
    assert len(server.transports) == 2 and all(item.closed for item in server.transports)
    a1, a2, b, c1, c2 = server.thread_starts
    # (a) goes through the production adapter: its instructions and its one-item prompt.
    assert a1["ephemeral"] is True and a2["ephemeral"] is True
    assert "realtime conversational assistant" in a1["baseInstructions"]
    assert all(start["dynamicTools"] == [] for start in server.thread_starts)
    texts = [turn["input"][0]["text"] for turn in server.turn_starts]
    assert texts[0] != texts[1] and "Host-local timestamp" in texts[0]
    # (b) repeats one input on one persisted thread.
    assert b["ephemeral"] is False
    assert server.turn_starts[2]["threadId"] == server.turn_starts[3]["threadId"]
    assert texts[2] == texts[3]
    # (c) repeats one input on two fresh ephemeral threads, identical but for the thread.
    assert c1 == c2 and c1["ephemeral"] is True
    assert server.turn_starts[4]["threadId"] != server.turn_starts[5]["threadId"]
    assert texts[4] == texts[5]
    assert {key: value for key, value in b.items() if key != "ephemeral"} == {
        key: value for key, value in c1.items() if key != "ephemeral"
    }
    # No case can read another case's cache entry: each input differs from its first line.
    assert len({text.splitlines()[0] for text in (texts[2], texts[4])}) == 2
    # Both deciding requests (b2, c2) wait the same time after the request that wrote.
    assert pauses == [5.0, 5.0]
    assert server.order == [
        "thread", "turn", "thread", "turn",  # a1, a2
        "thread", "turn", "pause", "turn",  # b1, wait, b2
        "thread", "turn", "pause", "thread", "turn",  # c1, wait, c2
    ]


def test_synthetic_text_is_long_inert_and_differs_per_run() -> None:
    text = probe.synthetic_text("b", "0123456789abcdef")

    assert text == probe.synthetic_text("b", "0123456789abcdef")
    assert text.startswith("Synthetic cache probe 0123456789abcdef case b.")
    assert len(text.split()) > 2000
    assert len(text) < 32_768
    assert foreground_search_query(text) is None
    assert not contains_private_material(text)
    with pytest.raises(ValueError):
        probe.synthetic_text("b", "not-a-nonce")


def _patch_launch(monkeypatch: pytest.MonkeyPatch, server: FakeAppServer) -> None:
    async def create(*, executable: str, cwd: str) -> FakeTransport:
        assert executable == "resolved-codex"
        return server.transport()

    monkeypatch.setattr(probe, "_resolve_codex_executable", lambda explicit: "resolved-codex")
    monkeypatch.setattr(probe.SubprocessCodexJsonLineTransport, "create", create)


def test_refuses_to_run_without_the_quota_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        probe, "_resolve_codex_executable", lambda explicit: pytest.fail("resolved Codex")
    )
    monkeypatch.setattr(
        probe.SubprocessCodexJsonLineTransport, "create",
        lambda **kwargs: pytest.fail("launched Codex"),
    )

    assert probe.main([]) == 2
    captured = capsys.readouterr()
    assert _evidence(captured.out) == {
        "probe": "codex-prompt-cache/1",
        "status": "refused",
        "refusal": "quota_acknowledgement_missing",
    }
    assert _FLAG in captured.err and "6 model requests" in captured.err


def _evidence(output: str) -> dict[str, Any]:
    """The one bounded, marker-prefixed JSON line every outcome must emit."""

    assert output.endswith("\n") and output.count("\n") == 1
    assert output.startswith("[codex-cache-probe] ") and len(output) < 4096
    evidence = json.loads(output.removeprefix("[codex-cache-probe] "))
    assert type(evidence) is dict
    return evidence


def test_launch_failure_emits_evidence_without_the_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(explicit: str | None) -> str:
        raise FileNotFoundError("C:/private/place/codex.exe")

    monkeypatch.setattr(probe, "_resolve_codex_executable", missing)

    assert probe.main([_FLAG]) == 1
    captured = capsys.readouterr()
    assert _evidence(captured.out) == {
        "probe": "codex-prompt-cache/1",
        "status": "failed",
        "failure": {"stage": "launch", "error": "FileNotFoundError"},
    }
    assert "private" not in captured.out + captured.err


def _strings(value: object) -> list[str]:
    if type(value) is dict:
        return [text for item in value.items() for part in item for text in _strings(part)]
    if type(value) is list:
        return [text for item in value for text in _strings(item)]
    return [value] if type(value) is str else []


def test_output_is_numbers_and_fixed_labels_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeAppServer(_six(2048, 0))
    _patch_launch(monkeypatch, server)

    assert probe.main([_FLAG, "--pause-seconds", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    report = _evidence(captured.out)
    assert report["verdict"] == "affinity_per_thread"

    assert set(report) == {
        "codexVersion", "minimumCacheableInputTokens", "probe", "requests", "status", "verdict",
        "verdictMeaning",
    }
    for row in report["requests"]:
        assert set(row) == {
            "cacheWriteInputTokens", "cachedInputTokens", "case", "inputTokens", "model",
            "request", "usageUpdates",
        }
        assert all(
            row[key] is None or type(row[key]) is int
            for key in ("cacheWriteInputTokens", "cachedInputTokens", "inputTokens",
                        "request", "usageUpdates")
        )
    allowed = {
        "0.167.0", "gpt-5.6-terra", "a", "b", "c", "codex-prompt-cache/1", "completed",
        "affinity_per_thread", probe.VERDICT_MEANINGS["affinity_per_thread"],
        *report, *report["requests"][0],
    }
    assert set(_strings(report)) <= allowed

    sent = json.dumps(server.thread_starts + server.turn_starts)
    nonce = server.turn_starts[2]["input"][0]["text"].split()[3]
    assert len(nonce) == 16 and nonce in sent
    for leak in (nonce, "Calibration", "Synthetic cache probe", _REPLY, "thread_private",
                 "turn_private", "PrivateOS", "private-terminal", server.thread_starts[0]["cwd"],
                 json.dumps(server.thread_starts[0]["cwd"])[1:-1]):
        assert leak not in captured.out


def test_failed_turn_reports_stage_and_error_type_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeAppServer(_six(2048, 1900), failed_turn=3)
    _patch_launch(monkeypatch, server)

    assert probe.main([_FLAG, "--pause-seconds", "0"]) == 1
    report = _evidence(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["failure"] == {"stage": "b2", "error": "RuntimeError"}
    assert report["verdict"] == "inconclusive"
    assert len(server.turn_starts) == 4
    assert all(item.closed for item in server.transports)


@pytest.mark.asyncio
async def test_hard_timeout_closes_every_transport() -> None:
    server = FakeAppServer(_six(2048, 1900), stall_turn=4)
    report = await _run(server, timeout_seconds=0.5)

    assert report["status"] == "failed"
    assert report["failure"] == {"stage": "c1", "error": "TimeoutError"}
    assert report["verdict"] == "inconclusive"
    assert report["requests"][3]["cachedInputTokens"] == 2048
    assert len(server.transports) == 2 and all(item.closed for item in server.transports)


@pytest.mark.asyncio
async def test_shutdown_is_not_cut_short_by_the_probe_deadline() -> None:
    # Closing takes longer than the whole deadline. The measurement finished in time, so the
    # deadline has nothing left to bound, and every process must still be closed.
    server = FakeAppServer(_six(2048, 1900), close_seconds=1.0)
    report = await _run(server, timeout_seconds=0.5)

    assert report["status"] == "completed"
    assert report["verdict"] == "affinity_survives_new_thread"
    assert len(server.transports) == 2 and all(item.closed for item in server.transports)


@pytest.mark.asyncio
async def test_process_launched_as_the_deadline_expires_is_still_closed() -> None:
    server = FakeAppServer(_six(2048, 1900), launch_seconds=1.0)
    report = await _run(server, timeout_seconds=0.3)

    assert report["status"] == "failed"
    assert report["failure"] == {"stage": "a1", "error": "TimeoutError"}
    assert len(server.transports) == 1 and server.transports[0].closed


@pytest.mark.asyncio
async def test_cancellation_during_shutdown_still_closes_every_transport() -> None:
    server = FakeAppServer(_six(2048, 1900), close_seconds=0.5)
    running = asyncio.create_task(_run(server))
    async with asyncio.timeout(10):
        while not any(item.closing for item in server.transports):
            await asyncio.sleep(0.01)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert len(server.transports) == 2 and all(item.closed for item in server.transports)


def test_arguments_are_bounded(capsys: pytest.CaptureFixture[str]) -> None:
    for arguments in (["--timeout-seconds", "0"], ["--timeout-seconds", "901"],
                      ["--pause-seconds", "-1"], ["--pause-seconds", "61"]):
        with pytest.raises(SystemExit):
            probe.main([_FLAG, *arguments])
    capsys.readouterr()
