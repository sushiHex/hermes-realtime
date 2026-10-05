from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import sysconfig
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from hermes_realtime.protocol import BRIDGE_PROTOCOL_VERSION

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real_hermes_api_gate.py"
sys.path.insert(0, str(_GATE_PATH.parent))
_GATE_SPEC = importlib.util.spec_from_file_location("real_hermes_api_gate", _GATE_PATH)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = _GATE
_GATE_SPEC.loader.exec_module(_GATE)

qualify = _GATE.qualify

_MARKER = "[real-hermes-gate] "
_KEY = "installed-gate-test-key-" + "k" * 24
_COMPANION_TOKEN = "installed-gate-companion-" + "t" * 24
_RUN_IDS = ("run_" + "1" * 16, "run_" + "2" * 16, "run_" + "3" * 16)
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


def _git(checkout: Path, *arguments: str) -> None:
    subprocess.run(("git", *arguments), cwd=checkout, check=True, capture_output=True)


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hermes_root: Path | None = None,
    direct_url: object = None,
    candidate_in_venv: bool = True,
) -> Path:
    """A home laid out as the installer leaves it, with this interpreter posing as its own."""

    home = tmp_path / "home"
    checkout = home / "hermes-agent"
    checkout.mkdir(parents=True)
    _git(checkout, "init", "-q")
    (checkout / "README.md").write_text("hermes\n", encoding="utf-8")
    _git(checkout, "add", "README.md")
    _git(checkout, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "hermes")
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__file__ = str((hermes_root or checkout) / "hermes_cli" / "__init__.py")
    hermes_cli.__version__ = "0.21.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    venv = str(checkout / "venv")
    site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))
    metadata = site / "hermes_realtime-9.8.7.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: hermes-realtime\nVersion: 9.8.7\n", encoding="utf-8"
    )
    if direct_url is not None:
        (metadata / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    if candidate_in_venv:
        monkeypatch.syspath_prepend(str(site))
    monkeypatch.delenv("HERMES_REALTIME_LIVEKIT_LOCAL", raising=False)
    return home


def _write_env(home: Path, *, key: str = _KEY, companion_port: int | None) -> None:
    lines = [f"API_SERVER_KEY={key}"]
    if companion_port is not None:
        lines += [
            f"HERMES_REALTIME_COMPANION_PORT={companion_port}",
            f"HERMES_REALTIME_COMPANION_TOKEN={_COMPANION_TOKEN}",
        ]
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _capabilities() -> dict[str, object]:
    return {
        "object": "hermes.api_server.capabilities",
        "platform": "hermes-agent",
        "model": "test-model",
        "auth": {"type": "bearer", "required": True},
        "runtime": {"mode": "server_agent", "tool_execution": "server", "split_runtime": False},
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
            "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
            "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
            "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
        },
    }


@dataclass
class _Hermes:
    """Hermes's /v1/runs as v0.21.0 serves it: the gate's three runs, in dispatch order.

    ``status_answer`` changes what ``GET /v1/runs/{id}`` returns for the completion run after
    it finished: ``exact``, ``lingering`` (still running), ``foreign`` (another run's status)
    or ``oversized``. ``approval_skipped`` finishes the approval run without asking, and
    ``stop_refused`` answers a stop with 409, as for a run that already finished.
    """

    status_answer: str = "exact"
    approval_skipped: bool = False
    stop_refused: bool = False
    runs: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    decided: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def _authorized(self, request: web.Request) -> None:
        assert request.headers["Authorization"] == f"Bearer {_KEY}"

    async def capabilities(self, request: web.Request) -> web.Response:
        self._authorized(request)
        return web.json_response(_capabilities())

    async def create(self, request: web.Request) -> web.Response:
        self._authorized(request)
        run_id = _RUN_IDS[len(self.runs)]
        self.runs.append(run_id)
        self.statuses[run_id] = "running"
        return web.json_response({"run_id": run_id, "status": "started"}, status=202)

    async def events(self, request: web.Request) -> web.StreamResponse:
        self._authorized(request)
        run_id = request.match_info["run_id"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if run_id == _RUN_IDS[0]:
            await self._finish(response, run_id, "completed", output="The gate passed.")
        elif run_id == _RUN_IDS[1]:
            if not self.approval_skipped:
                await self._send(
                    response,
                    {
                        "event": "approval.request",
                        "run_id": run_id,
                        "command": "chmod 777 probe",
                        "description": "world-writable permissions",
                        "choices": ["once", "session", "always", "deny"],
                    },
                )
                await self.decided.wait()
            await self._finish(response, run_id, "completed", output="Rejected; not retried.")
        else:
            await self.stopped.wait()
            if self.stop_refused:
                await self._finish(response, run_id, "completed", output="Finished first.")
            else:
                await self._finish(response, run_id, "cancelled")
        return response

    async def _finish(
        self, response: web.StreamResponse, run_id: str, status: str, output: str | None = None
    ) -> None:
        # Hermes sets the terminal status before the terminal event reaches any client.
        self.statuses[run_id] = status
        event: dict[str, object] = {"event": f"run.{status}", "run_id": run_id}
        if output is not None:
            self.outputs[run_id] = event["output"] = output
        await self._send(response, event)

    @staticmethod
    async def _send(response: web.StreamResponse, event: dict[str, object]) -> None:
        await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")

    async def approval(self, request: web.Request) -> web.Response:
        self._authorized(request)
        choice = (await request.json())["choice"]
        self.decided.set()
        return web.json_response(
            {
                "choice": choice,
                "object": "hermes.run.approval_response",
                "resolved": 1,
                "run_id": request.match_info["run_id"],
            }
        )

    async def stop(self, request: web.Request) -> web.Response:
        self._authorized(request)
        run_id = request.match_info["run_id"]
        if self.stop_refused:
            # The run finished on its own before the stop reached it.
            self.stopped.set()
            return web.json_response({"error": {"code": "run_not_active"}}, status=409)
        if self.statuses[run_id] in _TERMINAL:
            return web.json_response(self._status(run_id))
        self.statuses[run_id] = "stopping"
        self.stopped.set()
        return web.json_response({"run_id": run_id, "status": "stopping"})

    def _status(self, run_id: str) -> dict[str, object]:
        status: dict[str, object] = {
            "object": "hermes.run",
            "run_id": run_id,
            "status": self.statuses[run_id],
        }
        if run_id in self.outputs:
            status["output"] = self.outputs[run_id]
        return status

    async def status(self, request: web.Request) -> web.Response:
        self._authorized(request)
        run_id = request.match_info["run_id"]
        if run_id == _RUN_IDS[0] and self.status_answer == "lingering":
            return web.json_response(self._status(run_id) | {"status": "running"})
        if run_id == _RUN_IDS[0] and self.status_answer == "foreign":
            return web.json_response(self._status(_RUN_IDS[1]))
        if run_id == _RUN_IDS[0] and self.status_answer == "oversized":
            return web.json_response(self._status(run_id) | {"output": "x" * 70_000})
        return web.json_response(self._status(run_id))


async def _serve_hermes(hermes: _Hermes) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/v1/capabilities", hermes.capabilities)
    app.router.add_post("/v1/runs", hermes.create)
    app.router.add_get("/v1/runs/{run_id}", hermes.status)
    app.router.add_get("/v1/runs/{run_id}/events", hermes.events)
    app.router.add_post("/v1/runs/{run_id}/approval", hermes.approval)
    app.router.add_post("/v1/runs/{run_id}/stop", hermes.stop)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


@dataclass
class _Companion:
    """The plugin's companion bridge as far as the hello: ``welcome``, ``refuse`` or ``silent``."""

    mode: str = "welcome"
    offered: frozenset[str] = frozenset({"voice_archive", "voice_review"})
    hellos: int = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hello = json.loads(await reader.readline())
        self.hellos += 1
        if self.mode == "silent":
            await reader.read()
        elif self.mode == "refuse" or hello["token"] != _COMPANION_TOKEN:
            writer.write(b'{"ok": false}\n')
        else:
            negotiated = sorted(set(hello["capabilities"]) & self.offered)
            welcome: dict[str, object] = {
                "ok": True,
                "protocol_version": BRIDGE_PROTOCOL_VERSION,
                "capabilities": negotiated,
            }
            if "voice_review" in negotiated:
                welcome["review_interval"] = 8
            writer.write(json.dumps(welcome).encode() + b"\n")
        await writer.drain()
        writer.close()


@dataclass
class _Gateway:
    hermes: _Hermes
    companion: _Companion
    api_url: str
    companion_port: int


@pytest_asyncio.fixture
async def gateway() -> AsyncIterator[_Gateway]:
    hermes, companion = _Hermes(), _Companion()
    runner, api_url = await _serve_hermes(hermes)
    server = await asyncio.start_server(companion.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield _Gateway(hermes, companion, api_url, port)
    finally:
        # A refused gate leaves event streams waiting; release them so shutdown is prompt.
        hermes.decided.set()
        hermes.stopped.set()
        server.close()
        await runner.cleanup()


def _marker(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(_MARKER)]
    assert len(lines) == 1
    evidence = json.loads(lines[0].removeprefix(_MARKER))
    assert type(evidence) is dict
    return evidence


@pytest.mark.asyncio
async def test_the_gate_qualifies_the_installed_runtime_and_records_counts_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)

    record = await qualify(home, gateway.api_url)

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=home / "hermes-agent",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert record == {
        "gate": "passed",
        "hermes": {"version": "0.21.0", "commit": commit, "baseline": False},
        "candidate": {"version": "9.8.7", "install": "wheel"},
        "discovery": {"capabilities": ["voice_archive", "voice_review"]},
        "behaviors": {
            "completion": "completed",
            "approval": "completed",
            "cancellation": "interrupted",
        },
        "cleanup": {"runs": 3, "statuses": {"cancelled": 1, "completed": 2}},
    }
    assert gateway.companion.hellos == 1
    serialized = json.dumps(record) + capsys.readouterr().out
    for private in (_KEY, _COMPANION_TOKEN, *_RUN_IDS, str(tmp_path), "gate passed."):
        assert private not in serialized


@pytest.mark.asyncio
async def test_the_gate_records_an_editable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    home = _install(
        tmp_path,
        monkeypatch,
        direct_url={"url": "file:///source", "dir_info": {"editable": True}},
    )
    _write_env(home, companion_port=gateway.companion_port)

    record = await qualify(home, gateway.api_url)

    assert record["candidate"] == {"version": "9.8.7", "install": "editable"}


@pytest.mark.asyncio
async def test_the_gate_refuses_an_interpreter_whose_hermes_is_not_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _install(tmp_path, monkeypatch, hermes_root=tmp_path / "elsewhere")

    with pytest.raises(RuntimeError, match="Hermes is not the install's checkout"):
        await qualify(home, "http://127.0.0.1:9")

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "identity", "version": 1}


@pytest.mark.asyncio
async def test_the_gate_refuses_an_interpreter_whose_candidate_is_not_installed_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # This interpreter's own hermes-realtime, from the repository's environment.
    home = _install(tmp_path, monkeypatch, candidate_in_venv=False)

    with pytest.raises(RuntimeError, match="hermes-realtime is not installed in the install"):
        await qualify(home, "http://127.0.0.1:9")

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "identity", "version": 1}


@pytest.mark.asyncio
async def test_the_gate_refuses_a_weak_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, key="short", companion_port=gateway.companion_port)

    with pytest.raises(RuntimeError, match="strong API_SERVER_KEY"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "endpoint", "version": 1}
    assert gateway.hermes.runs == [] and gateway.companion.hellos == 0


@pytest.mark.asyncio
async def test_the_gate_refuses_an_env_that_names_no_companion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=None)

    with pytest.raises(RuntimeError, match="names no companion endpoint"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "endpoint", "version": 1}
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
async def test_the_gate_refuses_a_hermes_api_off_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)

    with pytest.raises(ValueError, match="loopback"):
        await qualify(home, "http://192.0.2.10:8642")

    assert _marker(capsys) == {"failure": "ValueError", "stage": "endpoint", "version": 1}
    assert gateway.companion.hellos == 0


@pytest.mark.asyncio
async def test_the_gate_refuses_a_failed_hello(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.companion.mode = "refuse"

    with pytest.raises(Exception, match="bridge authentication failed"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {
        "failure": "BridgeAuthenticationError",
        "stage": "discovery",
        "version": 1,
    }
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
async def test_the_gate_refuses_a_companion_missing_a_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.companion.offered = frozenset({"voice_archive"})

    with pytest.raises(RuntimeError, match="does not offer every capability"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "discovery", "version": 1}
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
async def test_the_gate_bounds_a_companion_that_never_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.companion.mode = "silent"
    monkeypatch.setattr(_GATE, "_HELLO_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(qualify(home, gateway.api_url), timeout=10)

    assert _marker(capsys) == {"failure": "TimeoutError", "stage": "discovery", "version": 1}


@pytest.mark.asyncio
async def test_the_gate_refuses_an_approval_probe_that_never_asks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.hermes.approval_skipped = True

    with pytest.raises(RuntimeError, match="completed without approval"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "approval", "version": 1}


@pytest.mark.asyncio
async def test_the_gate_refuses_a_cancellation_hermes_does_not_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.hermes.stop_refused = True

    with pytest.raises(RuntimeError, match="exact cancellation was rejected"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "cancellation", "version": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("status_answer", ["lingering", "foreign", "oversized"])
async def test_the_gate_refuses_a_run_it_cannot_read_back_as_terminal(
    status_answer: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    home = _install(tmp_path, monkeypatch)
    _write_env(home, companion_port=gateway.companion_port)
    gateway.hermes.status_answer = status_answer

    with pytest.raises(RuntimeError, match="not terminal|body bound"):
        await qualify(home, gateway.api_url)

    assert _marker(capsys) == {"failure": "RuntimeError", "stage": "cleanup", "version": 1}


def test_the_default_home_is_the_installers() -> None:
    assert _GATE.default_hermes_home({"HERMES_HOME": " /h "}) == Path("/h")
    if sys.platform == "win32":
        local = {"HERMES_HOME": "", "LOCALAPPDATA": "C:/Local"}
        assert _GATE.default_hermes_home(local) == Path("C:/Local/hermes")
    else:
        assert _GATE.default_hermes_home({}) == Path.home() / ".hermes"
