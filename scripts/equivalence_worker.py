"""Private archived-source child; no public CLI and no persistent raw trace."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

from scripts.deterministic_equivalence import ARMS_V1
from scripts.equivalence_process import _read_frame, _require, _write_frame
from scripts.qualify_evidence_slice_zero import canonical_json_bytes


def _server_environment(source: Mapping[str, str]) -> dict[str, str]:
    allowed = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "PATH"}
    environment = {name: value for name, value in source.items() if name.upper() in allowed}
    environment["LIVEKIT_KEYS"] = "dev" + "key: local" + "-" + "x" * 32 + "\n"
    return environment


def _observation(
    name: str, records: tuple[object, ...], metadata: tuple[object, ...], key: bytes
) -> dict[str, Any]:
    from hermes_realtime.production_observation import CloseStageObservationV1
    from tests.support import qualification as trace

    context_type = trace._CommittedConversationContextSnapshotQualificationObservationV1
    content_fields = {
        context_type: "committed_conversation_context_snapshot",
        trace._GeneratedTextQualificationObservationV1: "generated_text",
        trace._TransportConfirmedChunkQualificationObservationV1: "confirmed_text",
    }
    result: list[dict[str, Any]] = []
    for record in records:
        kind = cast(Any, record).kind.value
        if type(record) in content_fields:
            raw = getattr(record, content_fields[type(record)])
            _require(type(raw) is bytes and bool(raw), "production content observation is absent")
            value: Any = hmac.new(
                key, kind.encode("ascii") + b"\0" + raw, hashlib.sha256
            ).hexdigest()
        elif type(record) is trace._CancellationQualificationObservationV1:
            value = cast(Any, record).reason.value
        elif type(record) is trace._ForegroundCleanupQualificationObservationV1:
            value = cast(Any, record).succeeded
        elif type(record) is trace._HostReturnQualificationObservationV1:
            value = cast(Any, record).outcome.value
        else:
            raise ValueError("production trace contains an unknown observation type")
        result.append({"kind": kind, "value": value})
    return {
        "arm": name,
        "complete": True,
        "records": result,
        "close": [
            {"stage": cast(Any, item).stage.value, "result": cast(Any, item).result.value}
            for item in metadata
            if type(item) is CloseStageObservationV1
        ],
    }


def _verify_imports(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"hermes_realtime", "scripts", "tests"}:
            continue
        filename = getattr(module, "__file__", None)
        _require(
            type(filename) is str and Path(filename).resolve().is_relative_to(root),
            "qualification imported source outside the archived candidate",
        )


async def _run_arms(workspace: Path, livekit_url: str, emit: Any) -> None:
    # These helpers create the exact production launcher and drive its public
    # HTTP/LiveKit ingress. No pytest result is used as a machine assertion.
    from tests.integration.test_qualification_full_host_ingress import (
        _run_active_response_host_shutdown_arm,
        _run_non_mutation_arm,
    )

    key = os.urandom(32)
    source_root = Path(__file__).resolve().parent.parent
    for name in ARMS_V1:
        observed: list[dict[str, Any]] = []
        if name.startswith("shutdown_"):
            raw, metadata, _database = await _run_active_response_host_shutdown_arm(
                tmp_path=workspace,
                capture=name != "shutdown_disabled",
                consent=name == "shutdown_consented",
                livekit_url=livekit_url,
            )
            observed.append(_observation(name, raw, metadata, key))
        else:

            def observe(
                raw: tuple[object, ...],
                metadata: tuple[object, ...],
                *,
                arm_name: str = name,
                output: list[dict[str, Any]] = observed,
            ) -> None:
                output.append(_observation(arm_name, raw, metadata, key))

            await _run_non_mutation_arm(
                tmp_path=workspace,
                capture=name not in {"disabled", "perturbed"},
                consent=name in {"consented", "blocked", "faulted"},
                writer_fault=name == "faulted",
                writer_block=name == "blocked",
                livekit_url=livekit_url,
                typed_stimulus="deliberately different typed stimulus"
                if name == "perturbed"
                else "paired typed stimulus",
                observe=observe,
            )
        _verify_imports(source_root)
        _require(len(observed) == 1, "host observation is disconnected or duplicated")
        emit(name, observed[0])


def main() -> None:
    if os.name != "nt" or len(sys.argv) != 3 or sys.flags.optimize or not sys.flags.isolated:
        raise ValueError("equivalence child requires its exact isolated owned launch")
    import msvcrt

    request_handle, response_handle = (int(value) for value in sys.argv[1:])
    for handle in (request_handle, response_handle):
        os.set_handle_inheritable(handle, False)
    request_fd = msvcrt.open_osfhandle(request_handle, os.O_RDONLY | os.O_BINARY)
    response_fd = msvcrt.open_osfhandle(response_handle, os.O_WRONLY | os.O_BINARY)
    # Provider/library diagnostics must not persist conversation or path data.
    with open(os.devnull, "wb", buffering=0) as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
    config = _read_frame(request_fd, time.monotonic() + 10)
    _require(
        type(config) is dict
        and set(config)
        == {
            "version",
            "nonce",
            "livekit",
            "livekitSha256",
            "workspace",
            "sourceCommit",
            "sourceTree",
        },
        "child configuration is not closed",
    )
    _require(type(config["version"]) is int and config["version"] == 1, "unsupported child version")
    livekit = Path(config["livekit"])
    _require(
        hashlib.sha256(livekit.read_bytes()).hexdigest() == config["livekitSha256"],
        "LiveKit pin changed before launch",
    )
    workspace = Path(config["workspace"])
    sequence = 0

    def emit(stage: str, observation: dict[str, Any]) -> None:
        nonlocal sequence
        _write_frame(
            response_fd,
            {
                "version": 1,
                "nonce": config["nonce"],
                "sequence": sequence,
                "stage": stage,
                "observation": observation,
            },
        )
        ack = _read_frame(request_fd, time.monotonic() + 10)
        _require(
            ack == {"version": 1, "nonce": config["nonce"], "sequence": sequence},
            "child acknowledgment identity differs",
        )
        sequence += 1

    with ExitStack() as probes:
        ports = []
        for kind in (socket.SOCK_STREAM, socket.SOCK_STREAM, socket.SOCK_DGRAM):
            probe = probes.enter_context(socket.socket(socket.AF_INET, kind))
            probe.bind(("127.0.0.1", 0))
            ports.append(int(probe.getsockname()[1]))
    port, rtc_tcp, rtc_udp = ports
    server_config = canonical_json_bytes(
        {
            "port": port,
            "bind_addresses": ["127.0.0.1"],
            "development": True,
            "rtc": {
                "tcp_port": rtc_tcp,
                "udp_port": rtc_udp,
                "node_ip": "127.0.0.1",
                "use_external_ip": False,
                "enable_loopback_candidate": True,
                "ips": {"includes": ["127.0.0.0/8"]},
                "stun_servers": [],
            },
        }
    ).decode("utf-8")
    server = subprocess.Popen(
        (str(livekit), "--config-body", server_config),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_server_environment(os.environ),
        cwd=workspace,
        close_fds=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            _require(server.poll() is None, "owned LiveKit exited before readiness")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                _require(time.monotonic() < deadline, "owned LiveKit readiness timed out")
                time.sleep(0.05)
        emit("ready", {"port": port})
        asyncio.run(_run_arms(workspace, f"ws://127.0.0.1:{port}", emit))
        emit("done", {})
    except BaseException as error:
        failure_kinds: dict[type[BaseException], str] = {
            AssertionError: "assertion",
            TimeoutError: "timeout",
            RuntimeError: "runtime",
            ValueError: "value",
            OSError: "os",
        }
        failure_kind = failure_kinds.get(type(error), "other")
        source_line = 0
        traceback = error.__traceback__
        candidate = Path(__file__).resolve().parent.parent
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name != "_require" and Path(
                traceback.tb_frame.f_code.co_filename
            ).resolve().is_relative_to(candidate):
                source_line = traceback.tb_lineno
            traceback = traceback.tb_next
        _write_frame(
            response_fd,
            {
                "version": 1,
                "nonce": config["nonce"],
                "sequence": sequence,
                "stage": "failed",
                "observation": {"failure": failure_kind, "sourceLine": source_line},
            },
        )
        raise
    finally:
        server.terminate()
        server.wait(timeout=5)
        os.close(request_fd)
        os.close(response_fd)


if __name__ == "__main__":
    main()
