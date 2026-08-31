from __future__ import annotations

import hashlib
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_realtime.providers.kokoro_worker import (
    KokoroWorkerClient,
    _create_exclusive_listener,
    _is_close_request,
    _validate_cuda_profile_events,
    _worker_argument_parser,
    serve_kokoro_worker_connection,
)

_FAKE_CUDA_WORKER = r"""
import argparse
import hashlib
import json
import os
import socket
import struct


def recv_exact(sock, size):
    blocks = []
    remaining = size
    while remaining:
        block = sock.recv(remaining)
        if not block:
            raise RuntimeError("socket closed")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def recv_json(sock):
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    return json.loads(recv_exact(sock, size))


def send_json(sock, value, payload=b""):
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded + payload)


parser = argparse.ArgumentParser()
parser.add_argument("--connect-port", type=int, required=True)
parser.add_argument("--parent-pid", type=int, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--model-sha256", required=True)
parser.add_argument("--voices", required=True)
parser.add_argument("--voices-sha256", required=True)
parser.add_argument("--attestation-voice", required=True)
parser.add_argument("--language", required=True)
parser.add_argument("--max-pcm-bytes", type=int, required=True)
args = parser.parse_args()
with socket.create_connection(("127.0.0.1", args.connect_port), timeout=2.0) as sock:
    send_json(sock, {
        "version": 1,
        "type": "ready",
        "token": os.environ["HERMES_KOKORO_WORKER_TOKEN"],
        "provider": "CUDAExecutionProvider",
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "executionPolicy": "cuda-primary-profiled",
        "profiledCudaEvents": 3,
        "profiledCpuEvents": 1,
        "modelSha256": hashlib.sha256(open(args.model, "rb").read()).hexdigest(),
        "voicesSha256": hashlib.sha256(open(args.voices, "rb").read()).hexdigest(),
        "sampleRateHz": 24000,
    })
    while True:
        request = recv_json(sock)
        if request["type"] == "close":
            send_json(sock, {"version": 1, "type": "closed"})
            break
        pcm = (request["text"] + "|" + request["voice"]).encode("utf-8") * 2
        send_json(sock, {
            "version": 1,
            "type": "result",
            "requestId": request["requestId"],
            "pcmBytes": len(pcm),
            "sampleRateHz": 24000,
        }, pcm)
"""


def _write_fake_worker(path: Path, *, provider: str = "CUDAExecutionProvider") -> Path:
    path.write_text(
        _FAKE_CUDA_WORKER.replace("CUDAExecutionProvider", provider),
        encoding="utf-8",
    )
    return path


def _write_slow_worker(path: Path, signal_path: Path) -> Path:
    source = _FAKE_CUDA_WORKER.replace(
        'pcm = (request["text"] + "|" + request["voice"]).encode("utf-8") * 2',
        (
            f'Path(r"{signal_path}").write_text("received", encoding="utf-8")\n'
            "        time.sleep(5.0)\n"
            '        pcm = (request["text"] + "|" + request["voice"]).encode("utf-8") * 2'
        ),
    ).replace("import struct\n", "import struct\nimport time\nfrom pathlib import Path\n")
    path.write_text(source, encoding="utf-8")
    return path


def _write_slow_start_worker(path: Path, signal_path: Path) -> Path:
    source = _FAKE_CUDA_WORKER.replace(
        'with socket.create_connection(("127.0.0.1", args.connect_port), timeout=2.0) as sock:',
        (
            f'Path(r"{signal_path}").write_text("started", encoding="utf-8")\n'
            "time.sleep(5.0)\n"
            'with socket.create_connection(("127.0.0.1", args.connect_port), timeout=2.0) as sock:'
        ),
    ).replace("import struct\n", "import struct\nimport time\nfrom pathlib import Path\n")
    path.write_text(source, encoding="utf-8")
    return path


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    blocks: list[bytes] = []
    while size:
        block = connection.recv(size)
        assert block
        blocks.append(block)
        size -= len(block)
    return b"".join(blocks)


def _receive_json(connection: socket.socket) -> dict[str, object]:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    value = json.loads(_receive_exact(connection, size))
    assert type(value) is dict
    return value


def _send_json(connection: socket.socket, value: dict[str, object]) -> None:
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack("!I", len(encoded)) + encoded)


def test_kokoro_worker_server_attests_synthesizes_and_closes() -> None:
    server_socket, client_socket = socket.socketpair()
    calls: list[tuple[str, str, float]] = []

    def synthesize(text: str, voice: str, speed: float) -> bytes:
        calls.append((text, voice, speed))
        return b"\x01\x00\x02\x00"

    worker = threading.Thread(
        target=serve_kokoro_worker_connection,
        kwargs={
            "connection": server_socket,
            "token": "private-token",
            "model_sha256": "a" * 64,
            "voices_sha256": "b" * 64,
            "actual_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
            "profiled_cuda_events": 3,
            "profiled_cpu_events": 1,
            "max_pcm_bytes": 1024,
            "synthesize_pcm": synthesize,
        },
    )
    worker.start()
    try:
        assert _receive_json(client_socket) == {
            "version": 1,
            "type": "ready",
            "token": "private-token",
            "provider": "CUDAExecutionProvider",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "executionPolicy": "cuda-primary-profiled",
            "profiledCudaEvents": 3,
            "profiledCpuEvents": 1,
            "modelSha256": "a" * 64,
            "voicesSha256": "b" * 64,
            "sampleRateHz": 24000,
        }
        _send_json(
            client_socket,
            {
                "version": 1,
                "type": "synthesize",
                "requestId": "request_1",
                "text": "Hello",
                "voice": "bf_isabella",
                "speed": 1.0,
            },
        )
        assert _receive_json(client_socket) == {
            "version": 1,
            "type": "result",
            "requestId": "request_1",
            "pcmBytes": 4,
            "sampleRateHz": 24000,
        }
        assert _receive_exact(client_socket, 4) == b"\x01\x00\x02\x00"
        _send_json(client_socket, {"version": 1, "type": "close"})
        assert _receive_json(client_socket) == {"version": 1, "type": "closed"}
    finally:
        client_socket.close()
        worker.join(timeout=2.0)
        server_socket.close()

    assert not worker.is_alive()
    assert calls == [("Hello", "bf_isabella", 1.0)]


def test_kokoro_worker_client_round_trips_pcm_through_owned_process(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    client = KokoroWorkerClient(
        python_executable=Path(sys.executable),
        worker_script=_write_fake_worker(tmp_path / "fake_worker.py"),
        model_path=model_path,
        voices_path=voices_path,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_voices_sha256=hashlib.sha256(b"voices").hexdigest(),
        language="en-gb",
        max_pcm_bytes=1024,
        startup_timeout_seconds=2.0,
        request_timeout_seconds=2.0,
    )

    client.warm()
    assert client.actual_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert client.synthesize_pcm("Hello", "bf_isabella", 1.0) == b"Hello|bf_isabella" * 2
    client.close()

    assert client.closed
    with pytest.raises(RuntimeError, match="closed"):
        client.synthesize_pcm("again", "bf_isabella", 1.0)


def test_kokoro_worker_client_reports_child_startup_failure_promptly(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    worker_script = tmp_path / "failed_worker.py"
    worker_script.write_text(
        "raise SystemExit('CUDA provider missing from isolated worker')\n",
        encoding="utf-8",
    )
    client = KokoroWorkerClient(
        python_executable=Path(sys.executable),
        worker_script=worker_script,
        model_path=model_path,
        voices_path=voices_path,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_voices_sha256=hashlib.sha256(b"voices").hexdigest(),
        language="en-gb",
        startup_timeout_seconds=3.0,
        request_timeout_seconds=2.0,
        max_pcm_bytes=1024,
    )

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="CUDA provider missing"):
        client.warm()
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0


def test_kokoro_worker_client_rejects_cpu_provider_attestation(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    worker_script = _write_fake_worker(
        tmp_path / "cpu_worker.py",
        provider="CPUExecutionProvider",
    )
    client = KokoroWorkerClient(
        python_executable=Path(sys.executable),
        worker_script=worker_script,
        model_path=model_path,
        voices_path=voices_path,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_voices_sha256=hashlib.sha256(b"voices").hexdigest(),
        language="en-gb",
        startup_timeout_seconds=2.0,
        request_timeout_seconds=2.0,
        max_pcm_bytes=1024,
    )

    with pytest.raises(RuntimeError, match="failed startup attestation"):
        client.warm()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows socket ownership gate")
def test_cuda_listener_rejects_competing_windows_bind() -> None:
    listener = _create_exclusive_listener()
    attacker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        port = listener.getsockname()[1]
        attacker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            attacker.bind(("127.0.0.1", port))
    finally:
        attacker.close()
        listener.close()


def test_protocol_rejects_duplicate_json_keys() -> None:
    sender, receiver = socket.socketpair()
    try:
        payload = b'{"version":1,"version":2}'
        sender.sendall(struct.pack("!I", len(payload)) + payload)
        with pytest.raises(RuntimeError, match="duplicate JSON key"):
            KokoroWorkerClient._receive_json(receiver)
    finally:
        sender.close()
        receiver.close()


def test_protocol_frames_maximum_worst_case_text() -> None:
    sender, receiver = socket.socketpair()
    try:
        message = {"text": "\x00" * 4096}
        KokoroWorkerClient._send_json(sender, message)
        assert KokoroWorkerClient._receive_json(receiver) == message
    finally:
        sender.close()
        receiver.close()


def test_worker_argument_parser_uses_bounded_types_not_materialized_ranges() -> None:
    parser = _worker_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--connect-port", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-pcm-bytes", str(64 * 1024 * 1024 + 1)])
    help_text = parser.format_help()
    assert len(help_text) < 10_000


def test_close_interrupts_in_flight_worker_request(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    signal_path = tmp_path / "request-received"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    worker_script = _write_slow_worker(tmp_path / "slow_worker.py", signal_path)
    client = KokoroWorkerClient(
        python_executable=Path(sys.executable),
        worker_script=worker_script,
        model_path=model_path,
        voices_path=voices_path,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_voices_sha256=hashlib.sha256(b"voices").hexdigest(),
        language="en-gb",
        startup_timeout_seconds=2.0,
        request_timeout_seconds=10.0,
        max_pcm_bytes=1024,
    )
    client.warm()
    failures: list[BaseException] = []

    def synthesize() -> None:
        try:
            client.synthesize_pcm("Hello", "bf_isabella", 1.0)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=synthesize)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not signal_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert signal_path.exists()

    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started
    thread.join(timeout=2.0)

    assert elapsed < 2.0
    assert not thread.is_alive()
    assert failures


def test_cuda_profile_attestation_accepts_core_cuda_and_allowed_cpu_ops() -> None:
    events = [
        {"args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}},
        {"args": {"provider": "CUDAExecutionProvider", "op_name": "Gemm"}},
        {"args": {"provider": "CUDAExecutionProvider", "op_name": "LSTM"}},
        {"args": {"provider": "CPUExecutionProvider", "op_name": "Gather"}},
        {"args": {"provider": "CPUExecutionProvider", "op_name": "STFT"}},
    ]

    assert _validate_cuda_profile_events(events) == (3, 2)


@pytest.mark.parametrize(
    "events",
    [
        [{"args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}}],
        [
            {"args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}},
            {"args": {"provider": "CUDAExecutionProvider", "op_name": "Gemm"}},
            {"args": {"provider": "CPUExecutionProvider", "op_name": "LSTM"}},
        ],
    ],
)
def test_cuda_profile_attestation_rejects_missing_or_cpu_core_ops(
    events: list[dict[str, object]],
) -> None:
    with pytest.raises(RuntimeError, match="CUDA execution profile"):
        _validate_cuda_profile_events(events)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows parent-handle watchdog gate")
def test_parent_watchdog_exits_child_when_owned_parent_dies() -> None:
    parent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; "
                "from hermes_realtime.providers.kokoro_worker import _start_parent_watchdog; "
                "_start_parent_watchdog(int(sys.argv[1])); "
                "print('ready', flush=True); time.sleep(10)"
            ),
            str(parent.pid),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        parent.terminate()
        parent.wait(timeout=2.0)
        assert child.wait(timeout=2.0) == 70
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2.0)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2.0)


def test_close_interrupts_worker_startup_and_terminates_child(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    signal_path = tmp_path / "startup-began"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    client = KokoroWorkerClient(
        python_executable=Path(sys.executable),
        worker_script=_write_slow_start_worker(tmp_path / "slow_start.py", signal_path),
        model_path=model_path,
        voices_path=voices_path,
        expected_model_sha256=hashlib.sha256(b"model").hexdigest(),
        expected_voices_sha256=hashlib.sha256(b"voices").hexdigest(),
        language="en-gb",
        startup_timeout_seconds=10.0,
        request_timeout_seconds=2.0,
        max_pcm_bytes=1024,
    )
    failures: list[BaseException] = []

    def warm() -> None:
        try:
            client.warm()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=warm)
    thread.start()
    deadline = time.monotonic() + 2.0
    while not signal_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert signal_path.exists()
    process = client._process
    assert process is not None

    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started
    thread.join(timeout=2.0)

    assert elapsed < 2.0
    assert not thread.is_alive()
    assert process.poll() is not None
    assert failures


@pytest.mark.parametrize("version", [True, 1.0])
def test_close_request_requires_exact_integer_protocol_version(version: object) -> None:
    assert not _is_close_request({"version": version, "type": "close"})
    assert _is_close_request({"version": 1, "type": "close"})
