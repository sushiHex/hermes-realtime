"""Qualify the operator's installed Hermes from the outside, the way the full host meets it.

One command, run with the install's own interpreter while its gateway is running:

    <HERMES_HOME>/hermes-agent/venv/Scripts/python.exe scripts/real_hermes_api_gate.py

``--hermes-home`` defaults to ``HERMES_HOME``, else the installer's default home, and
``--hermes-api-url`` to the full host's default. Every observation is bound to the process
under test:

1. this interpreter's Hermes must be the installer's checkout, at a detached commit, and its
   hermes-realtime must come from that checkout's environment;
2. the API key and the companion endpoint come from the home's ``.env``, through the full
   host's own loaders;
3. the gateway names its process (authenticated ``/health/detailed``), and a real bridge
   hello with the companion must offer every capability and attest that same process,
   having loaded exactly the Hermes and hermes-realtime inspected here: anything else is a
   gateway still running what it loaded before, or a companion another process owns;
4. dispatch, approval and exact cancellation run through the host's task session, and each
   must end with its exact status;
5. after the session closes, every run Hermes admitted for it must read back as terminal;
6. the gateway must still be the same process at the end.

It prints exactly one line: the JSON record (versions, kinds, statuses and counts) on a pass,
or one ``[real-hermes-gate]`` marker with the stage, a category and the failure's type on a
refusal, including runs left running once any were admitted. Nothing goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote, urlsplit

try:
    import aiohttp
    from real_gate_support import available_port, installed_hermes_identity

    from hermes_realtime.companion.attestation import attest_runtime
    from hermes_realtime.companion.host import CompanionEndpoint
    from hermes_realtime.host_launcher import (
        DEFAULT_HERMES_API_URL,
        _load_api_bearer,
        _load_voice_companion,
        build_local_host_launcher,
    )
    from hermes_realtime.integration import (
        BridgeAuthenticationError,
        HermesApiConfig,
        HermesApiTaskSession,
        LocalHermesBridgeClient,
    )
    from hermes_realtime.protocol import (
        RUNTIME_ATTESTATION_CAPABILITY,
        VOICE_ARCHIVE_CAPABILITY,
        VOICE_REVIEW_CAPABILITY,
        CancelScope,
        ControlCancelEvent,
        ControlCancelPayload,
        RuntimeAttestation,
        WorkDispatchRequestedEvent,
        WorkDispatchRequestedPayload,
        WorkTerminalStatus,
    )
except ImportError as error:  # Not the install's interpreter: main() refuses with the marker.
    _IMPORT_FAILURE: ImportError | None = error
else:
    _IMPORT_FAILURE = None

_MARKER = "[real-hermes-gate] "
_HELLO_TIMEOUT_SECONDS = 30.0
_MAX_BODY_BYTES = 64 * 1024
# Every status Hermes reports for a run that holds no more work.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
# What a read-back counts a run as when it cannot read it as terminal.
_NOT_TERMINAL = "not_terminal"
# What an attestation says for a field it cannot name.
_UNNAMED = frozenset({"unknown", "elsewhere"})
# A component's own bounded marker, folded into the gate's single line.
_COMPONENT_MARKER = re.compile(r"\[([a-z][a-z0-9-]{0,47})\] (\{.*\})")
_MAX_COMPONENT_MARKERS = 8
_MAX_COMPONENT_MARKER_CHARS = 512
# Library log records at WARNING and above are counted by logger and level, never printed.
_MAX_LOGGERS = 16


class _LogCount(logging.Handler):
    """Counts records at WARNING and above by logger name and level, keeping no message."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.counts: dict[str, Counter[str]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        name = record.name[:64]
        if name not in self.counts and len(self.counts) >= _MAX_LOGGERS:
            name = "other"
        self.counts.setdefault(name, Counter())[record.levelname] += 1

    def evidence(self) -> dict[str, object]:
        if not self.counts:
            return {}
        return {"log": {name: dict(sorted(levels.items())) for name, levels in self.counts.items()}}


class Refusal(RuntimeError):
    """A guard rejected the install; ``category`` is the bounded reason."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class _Arguments(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise Refusal("arguments")


def default_hermes_home(environ: Mapping[str, str]) -> Path:
    """The home the installer uses: ``HERMES_HOME``, else the platform's default."""

    configured = environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path(environ["LOCALAPPDATA"]) / "hermes"
    return Path.home() / ".hermes"


def _installed(home: Path) -> tuple[RuntimeAttestation, dict[str, object]]:
    """What this interpreter runs, refused unless it is the install at ``home``."""

    import hermes_cli  # type: ignore[import-not-found]

    checkout = (home / "hermes-agent").resolve()
    # hermes_cli sits at the root of the installer's checkout.
    if Path(hermes_cli.__file__).resolve().parents[1] != checkout:
        raise Refusal("not_install")
    local = attest_runtime()
    # One rule: every field of what this interpreter runs must be named.
    if _UNNAMED & set(local.model_dump(exclude={"pid"}).values()):
        raise Refusal("unnamed")
    return local, {
        "hermes": installed_hermes_identity(local.hermes_version, checkout),
        "candidate": {"version": local.realtime_version, "install": local.realtime_install},
    }


async def _get_json(client: aiohttp.ClientSession, config: HermesApiConfig, path: str) -> Any:
    """One authenticated GET; None unless Hermes answered 200 with bounded JSON."""

    try:
        async with client.get(
            config.base_url + path,
            headers={"Authorization": f"Bearer {config.bearer}"},
            allow_redirects=False,
        ) as response:
            body = await response.content.read(_MAX_BODY_BYTES + 1)
            if response.status != 200 or len(body) > _MAX_BODY_BYTES:
                return None
        return json.loads(body)
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return None


async def _gateway_pid(client: aiohttp.ClientSession, config: HermesApiConfig) -> int:
    health = await _get_json(client, config, "/health/detailed")
    pid = health.get("pid") if type(health) is dict else None
    if type(pid) is not int or pid < 1:
        raise Refusal("health")
    return pid


async def _hello(companion: CompanionEndpoint) -> tuple[RuntimeAttestation, list[str]]:
    """Only a running companion answers; it must offer every capability the gate asks for."""

    wanted = (VOICE_ARCHIVE_CAPABILITY, VOICE_REVIEW_CAPABILITY, RUNTIME_ATTESTATION_CAPABILITY)
    try:
        async with asyncio.timeout(_HELLO_TIMEOUT_SECONDS):
            link = await LocalHermesBridgeClient.connect(
                host="127.0.0.1",
                port=companion.port,
                token=companion.token,
                participant_id="real-hermes-gate",
                capabilities=wanted,
            )
    except BridgeAuthenticationError as error:
        # A wrong token, or a plugin that predates runtime attestation and refuses the
        # capability: a gateway still running it needs a restart.
        raise Refusal("hello_refused") from error
    async with link:
        if link.capabilities != frozenset(wanted) or link.runtime is None:
            raise Refusal("capability")
        return link.runtime, sorted(link.capabilities)


async def _read_back(
    client: aiohttp.ClientSession, config: HermesApiConfig, run_ids: frozenset[str]
) -> Counter[str]:
    """Each run's status as Hermes reports it now; anything unreadable is not terminal."""

    statuses: Counter[str] = Counter()
    for run_id in sorted(run_ids):
        run = await _get_json(client, config, f"/v1/runs/{quote(run_id, safe='')}")
        terminal = (
            type(run) is dict
            and run.get("run_id") == run_id
            and run.get("status") in _TERMINAL_STATUSES
        )
        statuses[run["status"] if terminal else _NOT_TERMINAL] += 1
    return statuses


def _dispatch(
    *,
    event_id: str,
    sequence: int,
    task_id: str,
    utterance_id: str,
    objective: str,
) -> WorkDispatchRequestedEvent:
    return WorkDispatchRequestedEvent(
        event_id=event_id,
        session_id="session_real_api_gate",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        type="work.dispatch.requested",
        task_id=task_id,
        utterance_id=utterance_id,
        payload=WorkDispatchRequestedPayload(objective=objective),
    )


async def qualify(home: Path, api_url: str, evidence: dict[str, object]) -> dict[str, object]:
    """Qualify the install at ``home`` against its running gateway; the passing record.

    ``evidence`` tracks the stage, and the runs left running once any were admitted.
    """

    evidence["stage"] = "identity"
    local, record = _installed(home)
    evidence["stage"] = "endpoint"
    env_file = home / ".env"
    config = HermesApiConfig(
        base_url=api_url,
        bearer=_load_api_bearer(env_file),
        request_timeout_seconds=120,
    )
    companion = _load_voice_companion(env_file, os.environ)
    if companion is None:
        raise Refusal("no_companion")
    timeout = aiohttp.ClientTimeout(total=config.request_timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
        evidence["stage"] = "discovery"
        gateway = await _gateway_pid(client, config)
        attested, capabilities = await _hello(companion)
        if attested.pid != gateway:
            raise Refusal("foreign_companion")
        if attested.model_dump(exclude={"pid"}) != local.model_dump(exclude={"pid"}):
            raise Refusal("restart_gateway")
        record["discovery"] = {"capabilities": capabilities}
        witness = _ApprovalWitness()
        session = HermesApiTaskSession(
            config=config,
            session_id="session_real_api_gate",
            private_id_factory=iter(("completion_gate", "approval_gate", "cancel_gate")).__next__,
            approval_observer=witness,
        )
        try:
            record["behaviors"] = await _behaviors(session, witness, evidence)
        finally:
            try:
                await session.close()
            finally:
                statuses = await _read_back(client, config, session.admitted_run_ids)
                evidence["left_running"] = statuses[_NOT_TERMINAL]
        evidence["stage"] = "cleanup"
        if evidence["left_running"] != 0:
            raise Refusal("left_running")
        record["cleanup"] = {
            "runs": len(session.admitted_run_ids),
            "statuses": dict(sorted(statuses.items())),
        }
        if os.environ.get("HERMES_REALTIME_LIVEKIT_LOCAL") == "1":
            evidence["stage"] = "host"
            record["host"] = await _compose_host(config)
        evidence["stage"] = "end"
        if await _gateway_pid(client, config) != gateway:
            raise Refusal("gateway_restarted")
    return {"gate": "passed"} | record


class _ApprovalWitness:
    """Holds the approval events the session publishes, and wakes the gate on the first."""

    def __init__(self) -> None:
        self.events: list[dict[str, str | int | bool | None]] = []
        self.seen = asyncio.Event()

    def __call__(self, data: dict[str, str | int | bool | None]) -> None:
        self.events.append(data)
        self.seen.set()


async def _behaviors(
    session: HermesApiTaskSession, witness: _ApprovalWitness, evidence: dict[str, object]
) -> dict[str, str]:
    """Dispatch, approval and exact cancellation, each required to end exactly as expected."""

    evidence["stage"] = "completion"
    await session.start()
    completion_ack = await session.dispatch(
        _dispatch(
            event_id="dispatch_completion_gate",
            sequence=1,
            task_id="task_completion_gate",
            utterance_id="utterance_completion_gate",
            objective=(
                "Do not use tools. Return one concise sentence confirming that the "
                "Hermes realtime API completion gate passed."
            ),
        )
    )
    if not completion_ack.payload.accepted:
        raise Refusal("behavior")
    completion = await asyncio.wait_for(session.next_update(), timeout=300)
    if completion.payload.status is not WorkTerminalStatus.COMPLETED:
        raise Refusal("behavior")

    evidence["stage"] = "approval"
    probe = Path(tempfile.gettempdir()) / f"hermes-approval-probe-{uuid.uuid4().hex}"
    if probe.exists():
        raise Refusal("behavior")
    approval_ack = await session.dispatch(
        _dispatch(
            event_id="dispatch_approval_gate",
            sequence=10,
            task_id="task_approval_gate",
            utterance_id="utterance_approval_gate",
            objective=(
                "Use the terminal tool to run exactly this command and do not choose an "
                f"alternative: chmod 777 {probe.as_posix()}. If approval is rejected, "
                "report that rejection without retrying."
            ),
        )
    )
    if not approval_ack.payload.accepted:
        raise Refusal("behavior")
    terminal_task = asyncio.create_task(session.next_update())
    approval_task = asyncio.create_task(witness.seen.wait())
    try:
        done, _ = await asyncio.wait(
            {terminal_task, approval_task}, timeout=300, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            raise TimeoutError("the approval probe produced no terminal or approval event")
        if approval_task not in done:
            raise Refusal("behavior")  # It ended without asking.
        if len(witness.events) != 1 or witness.events[0].get("actionable") is not True:
            raise Refusal("behavior")
        approval_id = witness.events[0].get("approvalId")
        if type(approval_id) is not str:
            raise Refusal("behavior")
        await session.decide_approval(approval_id=approval_id, sequence=1, decision="reject")
        approval_terminal = await asyncio.wait_for(terminal_task, timeout=300)
    finally:
        for task in (terminal_task, approval_task):
            task.cancel()
        await asyncio.gather(terminal_task, approval_task, return_exceptions=True)
    if probe.exists() or approval_terminal.payload.status is not WorkTerminalStatus.COMPLETED:
        raise Refusal("behavior")

    evidence["stage"] = "cancellation"
    cancel_ack = await session.dispatch(
        _dispatch(
            event_id="dispatch_cancel_gate",
            sequence=20,
            task_id="task_cancel_gate",
            utterance_id="utterance_cancel_gate",
            objective="Without using tools, think carefully for several minutes before replying.",
        )
    )
    if not cancel_ack.payload.accepted:
        raise Refusal("behavior")
    cancellation = await session.cancel(
        ControlCancelEvent(
            event_id="cancel_exact_gate",
            session_id="session_real_api_gate",
            sequence=22,
            timestamp=datetime.now(UTC),
            type="control.cancel",
            task_id="task_cancel_gate",
            payload=ControlCancelPayload(
                scope=CancelScope.TASK,
                reason="real boundary exact-stop gate",
            ),
        )
    )
    if not cancellation.payload.accepted:
        raise Refusal("behavior")
    interrupted = await asyncio.wait_for(session.next_update(), timeout=60)
    if interrupted.payload.status is not WorkTerminalStatus.INTERRUPTED:
        raise Refusal("behavior")
    return {
        "completion": completion.payload.status.value,
        "approval": approval_terminal.payload.status.value,
        "cancellation": interrupted.payload.status.value,
    }


async def _compose_host(config: HermesApiConfig) -> str:
    """Start and close the Desktop full-host composition against the same gateway."""

    launcher = build_local_host_launcher(
        hermes_api_bearer=config.bearer,
        hermes_api_url=config.base_url,
        livekit_url=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"),
        livekit_api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
        livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", "local-" + ("x" * 32)),
        room_name=f"real-host-gate-{uuid.uuid4().hex[:10]}",
        worker_identity=f"worker_{uuid.uuid4().hex[:16]}",
        browser_port=available_port(),
        inference_provider="codex",
        conversation_profile="natural_v1",
        natural_work_tools=True,
        knowledge_speculation=True,
        knowledge_recovery=True,
        knowledge_budget_seconds=0.35,
        stt_provider="moonshine",
        moonshine_model_tier="medium",
        tts_provider="kokoro",
        allow_unsandboxed_tasks=True,
    )
    try:
        launch_url = await launcher.start()
        parsed = urlsplit(launch_url)
        if parsed.hostname != "127.0.0.1" or not parsed.fragment.startswith("bootstrap="):
            raise Refusal("host")
    finally:
        await launcher.close()
    return "started_and_closed"


def _components(output: str) -> dict[str, object]:
    """The bounded markers components printed while the gate ran, by marker name."""

    markers: dict[str, object] = {}
    for line in output.splitlines():
        match = (
            _COMPONENT_MARKER.fullmatch(line) if len(line) <= _MAX_COMPONENT_MARKER_CHARS else None
        )
        if match is not None and len(markers) < _MAX_COMPONENT_MARKERS:
            with contextlib.suppress(ValueError):
                markers[match.group(1)] = json.loads(match.group(2))
    return {"components": markers} if markers else {}


def main(argv: list[str] | None = None) -> int:
    """Print exactly one line, the record or the marker; 0 on a pass."""

    evidence: dict[str, object] = {"stage": "import", "version": 1}
    output = io.StringIO()
    record: dict[str, object] | None = None
    # The only handler for the run: no library record reaches stderr, and each is counted.
    log = _LogCount()
    logging.getLogger().addHandler(log)
    logging.captureWarnings(True)
    try:
        if _IMPORT_FAILURE is not None:
            raise _IMPORT_FAILURE
        evidence["stage"] = "arguments"
        parser = _Arguments(description=__doc__.splitlines()[0], add_help=False)
        parser.add_argument("--hermes-home", type=Path, default=None)
        parser.add_argument("--hermes-api-url", default=DEFAULT_HERMES_API_URL)
        args = parser.parse_args(argv)
        home = args.hermes_home or default_hermes_home(os.environ)
        with contextlib.redirect_stdout(output):
            record = asyncio.run(qualify(home, args.hermes_api_url, evidence))
    except BaseException as error:
        evidence["category"] = error.category if type(error) is Refusal else "error"
        evidence["failure"] = type(error).__name__
    finally:
        logging.captureWarnings(False)
        logging.getLogger().removeHandler(log)
        observed = _components(output.getvalue()) | log.evidence()
        if record is not None:
            print(json.dumps(record | observed, sort_keys=True), flush=True)
        else:
            line = json.dumps(evidence | observed, separators=(",", ":"), sort_keys=True)
            print(_MARKER + line, flush=True)
    return 0 if record is not None else 1


if __name__ == "__main__":
    sys.exit(main())
