from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
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

import hermes_realtime
from hermes_realtime.companion.attestation import attest_runtime
from hermes_realtime.protocol import BRIDGE_PROTOCOL_VERSION

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real_hermes_api_gate.py"
sys.path.insert(0, str(_GATE_PATH.parent))
_GATE_SPEC = importlib.util.spec_from_file_location("real_hermes_api_gate", _GATE_PATH)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
sys.modules[_GATE_SPEC.name] = _GATE
_GATE_SPEC.loader.exec_module(_GATE)

qualify = _GATE.qualify
Refusal = _GATE.Refusal

_MARKER = "[real-hermes-gate] "
_KEY = "installed-gate-test-key-" + "k" * 24
_COMPANION_TOKEN = "installed-gate-companion-" + "t" * 24
_RUN_IDS = ("run_" + "1" * 16, "run_" + "2" * 16, "run_" + "3" * 16)
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
_PACKAGE = Path(hermes_realtime.__file__).resolve().parent
_EVERY_CAPABILITY = frozenset({"voice_archive", "voice_review", "runtime_attestation"})


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass
class _Install:
    home: Path
    checkout: Path
    site: Path


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hermes_root: Path | None = None,
    realtime_in_install: bool = True,
    detached: bool = True,
    record: bool = True,
    copy_package: bool = False,
) -> _Install:
    """A home laid out as the installer leaves it, with this interpreter posing as its own.

    Hermes is a git checkout at ``home/hermes-agent`` whose committed ``hermes_cli`` this
    process imports, and hermes-realtime is a wheel installed in that checkout's ``venv``, from
    which this process imports it. ``copy_package`` puts the real package files there, for a
    gate run in its own process.
    """

    home = tmp_path / "home"
    checkout = home / "hermes-agent"
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "__init__.py").write_text(
        '__version__ = "0.21.0"\n', encoding="utf-8"
    )
    _git(checkout, "init", "-q")
    _git(checkout, "add", "hermes_cli/__init__.py")
    _git(checkout, "commit", "-q", "-m", "hermes")
    if detached:  # As the installer leaves it.
        _git(checkout, "checkout", "-q", "--detach")
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__file__ = str((hermes_root or checkout) / "hermes_cli" / "__init__.py")
    hermes_cli.__version__ = "0.21.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    venv = str(checkout / "venv")
    site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))
    metadata = site / f"hermes_realtime-{hermes_realtime.__version__}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: hermes-realtime\nVersion: {hermes_realtime.__version__}\n",
        encoding="utf-8",
    )
    if record:
        (metadata / "RECORD").write_bytes(b"hermes_realtime/__init__.py,sha256=test,1\n")
    if copy_package:
        shutil.copytree(
            _PACKAGE, site / "hermes_realtime", ignore=shutil.ignore_patterns("__pycache__")
        )
    if realtime_in_install:
        module = site / "hermes_realtime" / "__init__.py"
        monkeypatch.setattr(hermes_realtime, "__file__", str(module))
    monkeypatch.delenv("HERMES_REALTIME_LIVEKIT_LOCAL", raising=False)
    return _Install(home, checkout, site)


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
    """Hermes's API as v0.21.0 serves it: the gate's three runs, in dispatch order.

    ``status_answers`` overrides what ``GET /v1/runs/{id}`` returns for a run after it
    finished: ``running`` or ``stopping`` (still live), ``foreign`` (another run's status),
    ``oversized``, or ``http_error`` (a 500 whose body looks terminal). ``terminals``
    overrides how a run ends. ``approval_skipped`` finishes the approval run without asking;
    ``finished_before_stop`` ends the cancellation run before its stop arrives, which Hermes
    answers with 200 and the run's full status. ``reject_runs`` refuses every dispatch with
    503. ``pids`` is what ``/health/detailed`` reports on successive reads, the last one
    repeating.
    """

    pids: list[int] = field(default_factory=lambda: [os.getpid()])
    status_answers: dict[str, str] = field(default_factory=dict)
    terminals: dict[str, str] = field(default_factory=dict)
    approval_skipped: bool = False
    finished_before_stop: bool = False
    health_status: int = 200
    reject_runs: bool = False
    runs: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    decided: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def _authorized(self, request: web.Request) -> None:
        assert request.headers["Authorization"] == f"Bearer {_KEY}"

    async def health(self, request: web.Request) -> web.Response:
        self._authorized(request)
        pid = self.pids[0] if len(self.pids) == 1 else self.pids.pop(0)
        return web.json_response({"status": "ok", "pid": pid}, status=self.health_status)

    async def capabilities(self, request: web.Request) -> web.Response:
        self._authorized(request)
        return web.json_response(_capabilities())

    async def create(self, request: web.Request) -> web.Response:
        self._authorized(request)
        if self.reject_runs:
            return web.json_response({"error": {"code": "unavailable"}}, status=503)
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
            await self._finish(response, run_id, "completed", "The gate passed.")
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
            await self._finish(response, run_id, "completed", "Rejected; not retried.")
        else:
            await self.stopped.wait()
            await self._finish(response, run_id, "cancelled", None)
        return response

    async def _finish(
        self, response: web.StreamResponse, run_id: str, status: str, output: str | None
    ) -> None:
        status = self.terminals.get(run_id, status)
        if status in {"completed", "failed"} and output is None:
            output = "Finished."
        # Hermes sets the terminal status before the terminal event reaches any client.
        self.statuses[run_id] = status
        event: dict[str, object] = {"event": f"run.{status}", "run_id": run_id}
        if status == "completed":
            self.outputs[run_id] = event["output"] = output
        elif status == "failed":
            event["error"] = output
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
        if self.finished_before_stop and run_id == _RUN_IDS[2]:
            self.terminals[run_id] = "completed"
            self.statuses[run_id] = "completed"
            self.outputs[run_id] = "Finished first."
            self.stopped.set()
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
        answer = self.status_answers.get(run_id) if self.statuses[run_id] in _TERMINAL else None
        if answer in {"running", "stopping"}:
            return web.json_response(self._status(run_id) | {"status": answer})
        if answer == "foreign":
            other = next(other for other in self.runs if other != run_id)
            return web.json_response(self._status(other))
        if answer == "oversized":
            return web.json_response(self._status(run_id) | {"output": "x" * 70_000})
        if answer == "http_error":
            return web.json_response(self._status(run_id), status=500)
        return web.json_response(self._status(run_id))


async def _serve_hermes(hermes: _Hermes) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/health/detailed", hermes.health)
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
    """The plugin's companion bridge as far as the hello: ``welcome``, ``refuse`` or ``silent``.

    Its welcome attests this process's own runtime, which in these tests is the install the
    gate inspects, changed by ``attested``.
    """

    mode: str = "welcome"
    offered: frozenset[str] = _EVERY_CAPABILITY
    attested: dict[str, object] = field(default_factory=dict)
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
            if "runtime_attestation" in negotiated:
                welcome["runtime"] = attest_runtime().model_dump(mode="json") | self.attested
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


async def _refused(home: Path, api_url: str) -> tuple[BaseException, dict[str, object]]:
    evidence: dict[str, object] = {"stage": "start", "version": 1}
    with pytest.raises(BaseException) as raised:
        await qualify(home, api_url, evidence)
    return raised.value, evidence


def _category(error: BaseException) -> str:
    assert type(error) is Refusal, repr(error)
    return str(error.category)


@pytest.mark.asyncio
async def test_the_gate_qualifies_the_installed_runtime_and_prints_nothing_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)

    record = await qualify(install.home, gateway.api_url, {"stage": "start", "version": 1})

    assert record == {
        "gate": "passed",
        "hermes": {"version": "0.21.0", "commit": _git(install.checkout, "rev-parse", "HEAD"),
                   "baseline": False},
        "candidate": {"version": hermes_realtime.__version__, "install": "wheel"},
        "discovery": {"capabilities": sorted(_EVERY_CAPABILITY)},
        "behaviors": {
            "completion": "completed",
            "approval": "completed",
            "cancellation": "interrupted",
        },
        "cleanup": {"runs": 3, "statuses": {"cancelled": 1, "completed": 2}},
    }
    assert gateway.companion.hellos == 1
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["elsewhere", "vendored"])
async def test_the_gate_refuses_an_interpreter_whose_hermes_is_not_the_install(
    where: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    # "vendored" is a complete install nested inside the checkout, but not the checkout itself.
    base = tmp_path / "elsewhere" if where == "elsewhere" else tmp_path / "home" / "hermes-agent"
    root = base / "vendor" / "hermes"
    install = _install(tmp_path, monkeypatch, hermes_root=root)
    _write_env(install.home, companion_port=gateway.companion_port)
    if where == "vendored":
        (root / ".git").mkdir(parents=True)
        head = _git(install.checkout, "rev-parse", "HEAD") + "\n"
        (root / ".git" / "HEAD").write_bytes(head.encode("ascii"))
        venv = str(root / "venv")
        site = Path(sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv}))
        site.parent.mkdir(parents=True)
        install.site.rename(site)
        module = site / "hermes_realtime" / "__init__.py"
        monkeypatch.setattr(hermes_realtime, "__file__", str(module))

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "not_install" and evidence["stage"] == "identity"
    assert gateway.companion.hellos == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "unnamed"),
    [
        ("hermes_version", "unknown"),
        ("hermes_commit", "unknown"),
        ("realtime_version", "unknown"),
        ("realtime_install", "elsewhere"),
        ("realtime_record", "unknown"),
    ],
)
async def test_the_gate_refuses_an_install_any_field_of_which_it_cannot_name(
    field_name: str,
    unnamed: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    attestation = attest_runtime().model_copy(update={field_name: unnamed})
    monkeypatch.setattr(_GATE, "attest_runtime", lambda: attestation)

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "unnamed" and evidence["stage"] == "identity"
    assert gateway.companion.hellos == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param({"realtime_in_install": False}, id="hermes-realtime-elsewhere"),
        pytest.param({"detached": False}, id="no-detached-commit"),
        pytest.param({"record": False}, id="no-wheel-record"),
    ],
)
async def test_the_gate_refuses_an_install_it_cannot_name_in_full(
    layout: dict[str, bool], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch, **layout)
    _write_env(install.home, companion_port=gateway.companion_port)

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "unnamed" and evidence["stage"] == "identity"


@pytest.mark.asyncio
async def test_the_gate_refuses_a_weak_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, key="short", companion_port=gateway.companion_port)

    error, evidence = await _refused(install.home, gateway.api_url)

    assert type(error) is RuntimeError and evidence["stage"] == "endpoint"
    assert gateway.hermes.runs == [] and gateway.companion.hellos == 0


@pytest.mark.asyncio
async def test_the_gate_refuses_an_env_that_names_no_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=None)

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "no_companion" and evidence["stage"] == "endpoint"
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
async def test_the_gate_refuses_a_hermes_api_off_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)

    error, evidence = await _refused(install.home, "http://192.0.2.10:8642")

    assert type(error) is ValueError and evidence["stage"] == "endpoint"
    assert gateway.companion.hellos == 0


@pytest.mark.asyncio
async def test_the_gate_refuses_a_failed_hello(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.companion.mode = "refuse"

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "hello_refused"
    assert evidence["stage"] == "discovery" and gateway.hermes.runs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", sorted(_EVERY_CAPABILITY))
async def test_the_gate_refuses_a_companion_missing_a_capability(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.companion.offered = _EVERY_CAPABILITY - {missing}

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "capability" and evidence["stage"] == "discovery"
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
async def test_the_gate_bounds_a_companion_that_never_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.companion.mode = "silent"
    monkeypatch.setattr(_GATE, "_HELLO_TIMEOUT_SECONDS", 0.2)
    evidence: dict[str, object] = {"stage": "start", "version": 1}

    gate = asyncio.create_task(qualify(install.home, gateway.api_url, evidence))
    done, _ = await asyncio.wait({gate}, timeout=10)
    gate.cancel()
    await asyncio.gather(gate, return_exceptions=True)

    assert gate in done, "the gate's own bound did not stop it"
    assert type(gate.exception()) is TimeoutError and evidence["stage"] == "discovery"


@pytest.mark.asyncio
async def test_the_gate_refuses_a_gateway_that_does_not_report_its_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.hermes.health_status = 503

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "health" and evidence["stage"] == "discovery"


@pytest.mark.asyncio
async def test_the_gate_refuses_a_companion_another_process_owns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.companion.attested = {"pid": os.getpid() + 1}

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "foreign_companion" and evidence["stage"] == "discovery"
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attested",
    [
        pytest.param({"realtime_version": "9.9.9"}, id="older-wheel"),
        pytest.param({"realtime_record": "f" * 64}, id="rebuilt-wheel-same-version"),
        pytest.param({"realtime_install": "elsewhere"}, id="other-install"),
        pytest.param({"hermes_version": "0.20.0"}, id="older-hermes"),
        pytest.param({"hermes_commit": "f" * 40}, id="other-commit"),
    ],
)
async def test_the_gate_refuses_a_gateway_still_running_what_it_loaded_before(
    attested: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.companion.attested = attested

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "restart_gateway" and evidence["stage"] == "discovery"
    assert gateway.hermes.runs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "stage"),
    [
        pytest.param({"terminals": {_RUN_IDS[0]: "failed"}}, "completion", id="failed-completion"),
        pytest.param({"approval_skipped": True}, "approval", id="approval-never-asked"),
        pytest.param({"terminals": {_RUN_IDS[1]: "failed"}}, "approval", id="failed-approval"),
        pytest.param(
            {"terminals": {_RUN_IDS[2]: "completed"}}, "cancellation", id="stop-completed"
        ),
        pytest.param({"finished_before_stop": True}, "cancellation", id="finished-before-stop"),
    ],
)
async def test_the_gate_refuses_a_behavior_that_ends_otherwise(
    change: dict[str, object],
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    for name, value in change.items():
        setattr(gateway.hermes, name, value)

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "behavior" and evidence["stage"] == stage
    assert evidence["left_running"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "answer"),
    [
        (0, "running"),
        (1, "stopping"),
        (2, "stopping"),
        (0, "foreign"),
        (0, "oversized"),
        (0, "http_error"),
    ],
)
async def test_the_gate_refuses_a_run_it_cannot_read_back_as_terminal(
    run: int,
    answer: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.hermes.status_answers = {_RUN_IDS[run]: answer}

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "left_running" and evidence["stage"] == "cleanup"
    assert evidence["left_running"] == 1


@pytest.mark.asyncio
async def test_a_refused_gate_still_counts_the_runs_it_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.hermes.approval_skipped = True
    gateway.hermes.status_answers = {_RUN_IDS[1]: "running"}

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "behavior" and evidence["stage"] == "approval"
    assert evidence["left_running"] == 1


@pytest.mark.asyncio
async def test_the_gate_refuses_a_gateway_that_restarted_during_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch)
    _write_env(install.home, companion_port=gateway.companion_port)
    gateway.hermes.pids = [os.getpid(), os.getpid() + 1]

    error, evidence = await _refused(install.home, gateway.api_url)

    assert _category(error) == "gateway_restarted" and evidence["stage"] == "end"


def test_the_default_home_is_the_installers() -> None:
    assert _GATE.default_hermes_home({"HERMES_HOME": " /h "}) == Path("/h")
    if sys.platform == "win32":
        local = {"HERMES_HOME": "", "LOCALAPPDATA": "C:/Local"}
        assert _GATE.default_hermes_home(local) == Path("C:/Local/hermes")
    else:
        assert _GATE.default_hermes_home({}) == Path.home() / ".hermes"


# --- the command: exactly one line on stdout, and no traceback ------------------------------


async def _command(
    *arguments: str, environment: dict[str, str]
) -> tuple[int, list[str], str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_GATE_PATH),
        *arguments,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
    assert process.returncode is not None
    return process.returncode, stdout.decode().splitlines(), stderr.decode()


def _environment(*paths: Path, **overrides: str) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONPATH", "HERMES_HOME", "HERMES_REALTIME_LIVEKIT_LOCAL"}
    }
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    return environment | overrides


def _marker_line(lines: list[str]) -> dict[str, object]:
    assert len(lines) == 1 and lines[0].startswith(_MARKER), lines
    evidence = json.loads(lines[0].removeprefix(_MARKER))
    assert type(evidence) is dict
    return evidence


@pytest.mark.asyncio
async def test_the_command_prints_one_record_line_on_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway: _Gateway
) -> None:
    install = _install(tmp_path, monkeypatch, copy_package=True)
    _write_env(install.home, companion_port=gateway.companion_port)

    code, lines, stderr = await _command(
        "--hermes-home",
        str(install.home),
        "--hermes-api-url",
        gateway.api_url,
        environment=_environment(install.checkout, install.site),
    )

    assert code == 0 and stderr == ""
    assert len(lines) == 1 and not lines[0].startswith(_MARKER)
    assert json.loads(lines[0])["gate"] == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "marker"),
    [
        pytest.param(
            {"offered": frozenset({"voice_archive", "voice_review"})},
            {"category": "capability", "stage": "discovery"},
            id="missing-capability",
        ),
        pytest.param(
            # The task session logs a warning for a refused dispatch: counted, not printed.
            {"reject_runs": True},
            {
                "category": "behavior",
                "stage": "completion",
                "left_running": 0,
                "log": {"hermes_realtime.integration.api": {"WARNING": 1}},
            },
            id="non-202-dispatch",
        ),
    ],
)
async def test_the_command_prints_one_marker_line_on_a_refusal(
    change: dict[str, object],
    marker: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway: _Gateway,
) -> None:
    install = _install(tmp_path, monkeypatch, copy_package=True)
    _write_env(install.home, companion_port=gateway.companion_port)
    for name, value in change.items():
        setattr(gateway.companion if name == "offered" else gateway.hermes, name, value)

    code, lines, stderr = await _command(
        "--hermes-home",
        str(install.home),
        "--hermes-api-url",
        gateway.api_url,
        environment=_environment(install.checkout, install.site),
    )

    assert code == 1 and stderr == ""
    assert _marker_line(lines) == {"failure": "Refusal", "version": 1} | marker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "overrides", "stage"),
    [
        pytest.param(("--unknown",), {}, "arguments", id="unknown-argument"),
        pytest.param(("--hermes-home",), {}, "arguments", id="missing-value"),
        pytest.param((), {"LOCALAPPDATA": ""}, "arguments", id="no-default-home"),
    ],
)
async def test_the_command_refuses_its_arguments_with_one_marker_line(
    arguments: tuple[str, ...], overrides: dict[str, str], stage: str
) -> None:
    environment = _environment(**overrides)
    if "LOCALAPPDATA" in overrides:
        if sys.platform != "win32":
            pytest.skip("the installer's default home needs LOCALAPPDATA only on Windows")
        del environment["LOCALAPPDATA"]

    code, lines, stderr = await _command(*arguments, environment=environment)

    assert code == 1 and stderr == ""
    assert _marker_line(lines)["stage"] == stage


def test_a_component_marker_is_folded_into_the_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install = _install(tmp_path, monkeypatch)
    (install.checkout / "hermes_cli" / "__init__.py").write_text(
        '__version__ = "0.21.0"  # edited\n', encoding="utf-8"
    )

    code = _GATE.main(["--hermes-home", str(install.home), "--hermes-api-url", "http://127.0.0.1:9"])

    assert code == 1
    assert _marker_line(capsys.readouterr().out.splitlines()) == {
        "category": "error",
        "components": {"hermes-identity": {"refusal": "modified", "version": 1}},
        "failure": "RuntimeError",
        "stage": "identity",
        "version": 1,
    }


def test_a_folded_component_marker_is_bounded_in_length_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def installed(home: Path) -> object:
        print('[long-marker] {"filler": "' + "x" * 600 + '"}')
        for index in range(12):
            print(f'[marker-{index:02d}] {{"index": {index}}}')
        raise Refusal("not_install")

    monkeypatch.setattr(_GATE, "_installed", installed)

    assert _GATE.main(["--hermes-home", str(tmp_path)]) == 1

    evidence = _marker_line(capsys.readouterr().out.splitlines())
    assert evidence["components"] == {
        f"marker-{index:02d}": {"index": index} for index in range(8)
    }


@pytest.mark.asyncio
async def test_the_command_refuses_an_interpreter_that_cannot_import_it(tmp_path: Path) -> None:
    shadow = tmp_path / "shadow" / "hermes_realtime"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text('raise ImportError("not this one")\n', encoding="utf-8")

    code, lines, stderr = await _command(environment=_environment(shadow.parent))

    assert code == 1 and stderr == ""
    assert _marker_line(lines) == {
        "category": "error",
        "failure": "ImportError",
        "stage": "import",
        "version": 1,
    }
