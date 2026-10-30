"""Offline contract tests for one scoped MCP warm; no model or service is started."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from scripts import lookup_backend_mcp as mcp
from scripts.lookup_backend_model import StartupProfile

BASE = "http://127.0.0.1:11441"
MODEL = "phi4-mini:latest"


def reply(payload, *, error=False):
    return SimpleNamespace(isError=error, content=[SimpleNamespace(
        type="text", text=json.dumps(payload))])


def schema():
    types = {"model": "string", "keep_alive": "string", "by": "string",
             "force": "boolean", "strict_fit": "boolean", "num_ctx": "integer",
             "num_parallel": "integer", "timeout_seconds": "number"}
    return SimpleNamespace(name="warm", inputSchema={"type": "object",
        "properties": {key: {"type": value} for key, value in types.items()}})


def loaded(*, scope=BASE):
    return {"loaded": [], "observations": {"ollama": {
        "status": "available", "source": "ollama:/api/ps", "scope": scope}}}


def warm(*, scope=BASE, **changes):
    value = {"ok": True, "outcome": "succeeded", "model": MODEL,
             "resident": True, "keep_alive": "-1", "num_ctx": 64000,
             "num_parallel": 1, "strict_fit": True, "fit_margin_mb": 1024,
             "size_verified": True, "timeout_seconds": 300.0,
             "observations": {"ollama": {"status": "available",
                "source": "ollama:/api/ps", "scope": scope}}}
    value.update(changes)
    return value


class Session:
    def __init__(self, *, tools=None, preflight=None, result=None):
        self.tools = tools if tools is not None else [
            SimpleNamespace(name="list_loaded", inputSchema={"type": "object", "properties": {}}),
            schema()]
        self.preflight = loaded() if preflight is None else preflight
        self.result = warm() if result is None else result
        self.calls = []
        self.closed = False

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "list_loaded":
            return reply(self.preflight)
        result = dict(self.result)
        if result.get("timeout_seconds") == 300.0:
            result["timeout_seconds"] = arguments["timeout_seconds"]
        return reply(result)


def loader(session):
    @asynccontextmanager
    async def factory(base_url, deadline):
        assert base_url == BASE and deadline > time.monotonic()
        try:
            yield session
        finally:
            session.closed = True

    return mcp.make_strict_mcp_loader(factory)


async def invoke(session, *, seconds=360, profile=StartupProfile.LOOKUP_64K):
    return await loader(session)(base_url=BASE, model=MODEL,
        deadline=time.monotonic() + seconds, startup_profile=profile)


@pytest.mark.asyncio
async def test_one_warm_with_exact_isolated_settings_and_bounded_wait():
    session = Session()
    assert await invoke(session) is True
    assert session.closed
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]
    arguments = session.calls[1][1]
    assert {key: arguments[key] for key in ("model", "keep_alive", "force",
        "strict_fit", "num_ctx", "num_parallel")} == {
        "model": MODEL, "keep_alive": "-1", "force": False,
        "strict_fit": True, "num_ctx": 64000, "num_parallel": 1}
    assert arguments["by"] == "hermes-realtime"
    assert 0 < arguments["timeout_seconds"] <= 300


@pytest.mark.asyncio
async def test_lookup_16k_warm_uses_exact_context_and_strict_fit():
    session = Session(result=warm(num_ctx=16384))
    assert await invoke(session, profile=StartupProfile.LOOKUP_16K) is True
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]
    arguments = session.calls[1][1]
    assert arguments["num_ctx"] == 16384
    assert arguments["strict_fit"] is True and arguments["force"] is False
    assert arguments["num_parallel"] == 1 and arguments["keep_alive"] == "-1"


@pytest.mark.asyncio
async def test_deadline_reserve_refuses_before_warm():
    session = Session()
    with pytest.raises(ValueError, match="mcp-budget"):
        await invoke(session, seconds=60)
    assert all(name != "warm" for name, _ in session.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,clock,deadline", [
    ("initial", (200.1,), 200.1 + 60.0),
    ("after_preflight", (900.0, 1023.9), 1023.9 + 60.0),
])
async def test_non_dyadic_exact_reserve_refuses_without_warm(
        monkeypatch, phase, clock, deadline):
    readings = list(clock)

    def monotonic():
        return readings.pop(0) if len(readings) > 1 else readings[0]

    monkeypatch.setattr(mcp, "time", SimpleNamespace(monotonic=monotonic))

    @asynccontextmanager
    async def no_timer(_deadline):
        yield

    monkeypatch.setattr(mcp.asyncio, "timeout_at", no_timer)
    session = Session()
    entered = []

    @asynccontextmanager
    async def factory(_base_url, _deadline):
        entered.append(True)
        yield session

    load = mcp.make_strict_mcp_loader(factory)
    with pytest.raises(ValueError, match="mcp-budget"):
        await load(base_url=BASE, model=MODEL, deadline=deadline,
                   startup_profile=StartupProfile.LOOKUP_64K)
    assert [name for name, _ in session.calls] == (
        [] if phase == "initial" else ["list_loaded"])
    assert entered == ([] if phase == "initial" else [True])


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds,maximum", [(360.0, 300.0), (600.0, 540.0)])
async def test_non_dyadic_warm_timeout_preserves_reserve(
        monkeypatch, seconds, maximum):
    now = 1023.9  # Both deadlines cross a binary exponent boundary.
    monkeypatch.setattr(mcp, "time", SimpleNamespace(monotonic=lambda: now))

    @asynccontextmanager
    async def no_timer(_deadline):
        yield

    monkeypatch.setattr(mcp.asyncio, "timeout_at", no_timer)
    session = Session()

    @asynccontextmanager
    async def factory(_base_url, _deadline):
        yield session

    assert await mcp.make_strict_mcp_loader(factory)(
        base_url=BASE, model=MODEL, deadline=now + seconds,
        startup_profile=StartupProfile.LOOKUP_64K) is True
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]
    assert 0 < session.calls[1][1]["timeout_seconds"] <= maximum


@pytest.mark.asyncio
async def test_warm_capability_schema_is_required():
    entry = schema()
    entry.inputSchema["properties"].pop("strict_fit")
    session = Session(tools=[SimpleNamespace(name="list_loaded",
        inputSchema={"type": "object", "properties": {}}), entry])
    with pytest.raises(ValueError, match="mcp-capability"):
        await invoke(session)
    assert session.calls == []


@pytest.mark.asyncio
async def test_preflight_endpoint_scope_is_required_before_warm():
    session = Session(preflight=loaded(scope="http://127.0.0.1:11434"))
    with pytest.raises(ValueError, match="mcp-scope"):
        await invoke(session)
    assert [name for name, _ in session.calls] == ["list_loaded"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"outcome": "unknown"}, {"ok": False}, {"resident": None},
    {"model": "other:latest"}, {"num_ctx": 4096},
    {"num_parallel": 2}, {"strict_fit": False},
    {"fit_margin_mb": None}, {"size_verified": False},
    {"partial_fit": True}, {"partial_fit": None},
    {"observations": {"ollama": {"status": "available",
        "source": "ollama:/api/ps", "scope": "http://127.0.0.1:11434"}}},
])
async def test_incomplete_or_mismatched_warm_refuses(change):
    session = Session(result=warm(**change))
    with pytest.raises(ValueError, match="mcp-warm-ack"):
        await invoke(session)
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]


@pytest.mark.asyncio
async def test_native_profile_requests_exact_4k_context():
    session = Session(result=warm(num_ctx=4096))
    assert await invoke(session, profile=StartupProfile.NATIVE_TOOLS_4K) is True
    assert session.calls[1][1]["num_ctx"] == 4096


@pytest.mark.asyncio
async def test_unavailable_preflight_refuses_without_warm():
    session = Session(preflight={"loaded": None, "observations": {"ollama": {
        "status": "unavailable", "source": "ollama:/api/ps", "scope": BASE}}})
    with pytest.raises(ValueError, match="mcp-scope"):
        await invoke(session)
    assert [name for name, _ in session.calls] == ["list_loaded"]


@pytest.mark.asyncio
async def test_existing_residency_refuses_fresh_fit_claim():
    preflight = loaded()
    preflight["loaded"] = [{"name": MODEL, "context_length": 64000}]
    session = Session(preflight=preflight)
    with pytest.raises(ValueError, match="mcp-preflight"):
        await invoke(session)
    assert [name for name, _ in session.calls] == ["list_loaded"]


@pytest.mark.asyncio
async def test_warm_mcp_error_never_retries():
    session = Session()

    async def call_tool(name, arguments):
        session.calls.append((name, arguments))
        return reply(loaded() if name == "list_loaded" else warm(), error=name == "warm")

    session.call_tool = call_tool
    with pytest.raises(ValueError, match="mcp-result"):
        await invoke(session)
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]
    assert session.closed


@pytest.mark.asyncio
async def test_malformed_loopback_url_refuses_before_session():
    session = Session()
    with pytest.raises(ValueError, match="mcp-input"):
        await loader(session)(base_url=BASE + "/api", model=MODEL,
            deadline=time.monotonic() + 360,
            startup_profile=StartupProfile.LOOKUP_64K)
    assert session.calls == []


@pytest.mark.asyncio
async def test_personal_ollama_port_refuses_before_session():
    session = Session()
    with pytest.raises(ValueError, match="mcp-input"):
        await loader(session)(base_url="http://127.0.0.1:11434", model=MODEL,
            deadline=time.monotonic() + 360,
            startup_profile=StartupProfile.LOOKUP_64K)
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"fit_margin_mb": 1023}, {"coordination_warning": "ledger error"},
])
async def test_fit_margin_and_coordination_must_be_sound(change):
    session = Session(result=warm(**change))
    with pytest.raises(ValueError, match="mcp-warm-ack"):
        await invoke(session)


@pytest.mark.asyncio
async def test_loader_closure_is_one_shot():
    session = Session()
    load = loader(session)
    kwargs = {"base_url": BASE, "model": MODEL,
        "deadline": time.monotonic() + 360,
        "startup_profile": StartupProfile.LOOKUP_64K}
    assert await load(**kwargs) is True
    with pytest.raises(ValueError, match="mcp-reused"):
        await load(**kwargs)
    assert [name for name, _ in session.calls] == ["list_loaded", "warm"]


@pytest.mark.asyncio
@pytest.mark.parametrize("before_warm", [True, False])
async def test_refusal_records_closed_category_and_attempt_counts(capsys, before_warm):
    session = Session(preflight=loaded(scope="foreign") if before_warm else None,
                      result=warm(outcome="unknown"))
    with pytest.raises(ValueError):
        await invoke(session)
    line = capsys.readouterr().out.strip()
    record = json.loads(line.removeprefix("[lookup-mcp] "))
    assert record == {"phase": "preflight" if before_warm else "acknowledgment",
        "accepted": False, "failure_category": "mcp-scope" if before_warm else "mcp-warm-ack",
        "read_only_tool_calls": 1, "warm_tool_calls": 0 if before_warm else 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
async def test_deadline_and_cancellation_record_attempt_and_close_session(
    capsys, monkeypatch, cancel,
):
    session = Session()
    started = asyncio.Event()
    original = session.call_tool

    async def call_tool(name, arguments):
        if name == "warm":
            started.set()
            await asyncio.sleep(10)
        return await original(name, arguments)

    session.call_tool = call_tool
    monkeypatch.setattr(mcp, "_OVERHEAD_SECONDS", 0.001)
    task = asyncio.create_task(invoke(session, seconds=360 if cancel else .05))
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert session.closed
    record = json.loads(capsys.readouterr().out.strip().removeprefix("[lookup-mcp] "))
    assert record["warm_tool_calls"] == 1 and record["read_only_tool_calls"] == 1
    assert record["phase"] == "warm" and record["accepted"] is False
    assert record["failure_category"] == ("mcp-cancelled" if cancel else "mcp-timeout")


@pytest.mark.asyncio
async def test_foreign_error_details_are_not_published(capsys):
    session = Session()

    async def fail():
        raise ValueError("private-untrusted-detail")

    session.list_tools = fail
    with pytest.raises(ValueError):
        await invoke(session)
    output = capsys.readouterr().out
    assert "private-untrusted-detail" not in output
    record = json.loads(output.strip().removeprefix("[lookup-mcp] "))
    assert record["failure_category"] == "mcp-other"
    assert record["warm_tool_calls"] == record["read_only_tool_calls"] == 0
