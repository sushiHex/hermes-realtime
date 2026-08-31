"""Bounded client for an isolated Kokoro CUDA worker process."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import math
import os
import secrets
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any, cast

_PROTOCOL_VERSION = 1
_MAX_HEADER_BYTES = 32 * 1024
_MAX_TEXT_CHARS = 4096
_MAX_ID_CHARS = 128
_MAX_PCM_BYTES = 64 * 1024 * 1024
_MODEL_SAMPLE_RATE_HZ = 24_000
_TOKEN_ENV = "HERMES_KOKORO_WORKER_TOKEN"
_WorkerSynthesizePcm = Callable[[str, str, float], bytes]
_REQUIRED_CUDA_PROFILE_OPS = frozenset({"Conv", "Gemm", "LSTM"})
_ALLOWED_CPU_PROFILE_OPS = frozenset(
    {
        "Atan",
        "Cast",
        "Concat",
        "Div",
        "Equal",
        "Gather",
        "Slice",
        "Sqrt",
        "Squeeze",
        "STFT",
        "Unsqueeze",
        "Where",
    }
)


class _DuplicateJsonKey(ValueError):
    pass


def _create_exclusive_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        return listener
    except BaseException:
        listener.close()
        raise


def _bounded_integer(name: str, minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"{name} must be from {minimum} through {maximum}")
        return result

    return parse


def _worker_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one owned Kokoro CUDA worker.")
    parser.add_argument(
        "--connect-port",
        type=_bounded_integer("connect port", 1, 65_535),
        required=True,
    )
    parser.add_argument(
        "--parent-pid",
        type=_bounded_integer("parent PID", 1, 2**32 - 1),
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--voices-sha256", required=True)
    parser.add_argument("--attestation-voice", required=True)
    parser.add_argument("--language", choices=("en-us", "en-gb"), required=True)
    parser.add_argument(
        "--max-pcm-bytes",
        type=_bounded_integer("maximum PCM bytes", 1, _MAX_PCM_BYTES),
        required=True,
    )
    return parser


def _validate_cuda_profile_events(events: object) -> tuple[int, int]:
    if type(events) is not list:
        raise RuntimeError("Kokoro CUDA execution profile is not a JSON list")
    cuda_ops: set[str] = set()
    cpu_ops: set[str] = set()
    cuda_events = 0
    cpu_events = 0
    for event in events:
        if type(event) is not dict:
            continue
        args = event.get("args")
        if type(args) is not dict:
            continue
        provider = args.get("provider")
        operation = args.get("op_name")
        if type(operation) is not str:
            continue
        if provider == "CUDAExecutionProvider":
            cuda_events += 1
            cuda_ops.add(operation)
        elif provider == "CPUExecutionProvider":
            cpu_events += 1
            cpu_ops.add(operation)
    if not cuda_ops >= _REQUIRED_CUDA_PROFILE_OPS:
        raise RuntimeError("Kokoro CUDA execution profile is missing required CUDA core ops")
    unsupported_cpu_ops = cpu_ops - _ALLOWED_CPU_PROFILE_OPS
    if unsupported_cpu_ops:
        raise RuntimeError(
            f"Kokoro CUDA execution profile assigned unsupported ops to CPU: "
            f"{sorted(unsupported_cpu_ops)!r}"
        )
    return cuda_events, cpu_events


class KokoroWorkerClient:
    """Own one sequential authenticated loopback worker connection."""

    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        model_path: Path,
        voices_path: Path,
        expected_model_sha256: str,
        expected_voices_sha256: str,
        language: str,
        max_pcm_bytes: int,
        attestation_voice: str = "bf_isabella",
        startup_timeout_seconds: float = 15.0,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        for name, path in (
            ("python_executable", python_executable),
            ("worker_script", worker_script),
            ("model_path", model_path),
            ("voices_path", voices_path),
        ):
            if not isinstance(path, Path):
                raise TypeError(f"{name} must be a Path")
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"{name} must be an existing absolute file")
        for name, digest in (
            ("expected_model_sha256", expected_model_sha256),
            ("expected_voices_sha256", expected_voices_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(language) is not str or language not in {"en-us", "en-gb"}:
            raise ValueError("language must be exactly 'en-us' or 'en-gb'")
        if (
            type(attestation_voice) is not str
            or not attestation_voice
            or len(attestation_voice) > _MAX_ID_CHARS
        ):
            raise ValueError("attestation_voice exceeds supported bounds")
        if type(max_pcm_bytes) is not int or not 1 <= max_pcm_bytes <= _MAX_PCM_BYTES:
            raise ValueError("max_pcm_bytes exceeds supported bounds")
        for name, value, ceiling in (
            ("startup_timeout_seconds", startup_timeout_seconds, 60.0),
            ("request_timeout_seconds", request_timeout_seconds, 300.0),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite exact number")
            if not 0 < value <= ceiling:
                raise ValueError(f"{name} exceeds supported bounds")

        self._python_executable = python_executable
        self._worker_script = worker_script
        self._model_path = model_path
        self._voices_path = voices_path
        self._expected_model_sha256 = expected_model_sha256
        self._expected_voices_sha256 = expected_voices_sha256
        self._language = language
        self._attestation_voice = attestation_voice
        self._max_pcm_bytes = max_pcm_bytes
        self._startup_timeout = float(startup_timeout_seconds)
        self._request_timeout = float(request_timeout_seconds)
        self._process: subprocess.Popen[bytes] | None = None
        self._listener: socket.socket | None = None
        self._socket: socket.socket | None = None
        self._stderr_file: IO[bytes] | None = None
        self._actual_providers: tuple[str, ...] = ()
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False

    @property
    def actual_providers(self) -> tuple[str, ...]:
        with self._state_lock:
            return self._actual_providers

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def warm(self) -> None:
        with self._operation_lock:
            self._ensure_started()

    def synthesize_pcm(self, text: str, voice: str, speed: float) -> bytes:
        self._validate_request(text, voice, speed)
        with self._operation_lock:
            self._ensure_started()
            with self._state_lock:
                connection = self._require_socket()
            request_id = secrets.token_hex(16)
            try:
                connection.settimeout(self._request_timeout)
                self._send_json(
                    connection,
                    {
                        "version": _PROTOCOL_VERSION,
                        "type": "synthesize",
                        "requestId": request_id,
                        "text": text,
                        "voice": voice,
                        "speed": speed,
                    },
                )
                response = self._receive_json(connection)
                self._validate_result(response, request_id)
                pcm_size = cast(int, response["pcmBytes"])
                pcm = self._receive_exact(connection, pcm_size)
                if len(pcm) % 2:
                    raise RuntimeError("Kokoro worker returned incomplete int16 PCM")
                return pcm
            except BaseException:
                self._fault(connection)
                raise

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            listener = self._listener
            connection = self._socket
            process = self._process
            stderr_file = self._stderr_file
            self._listener = None
            self._socket = None
            self._process = None
            self._stderr_file = None
            self._actual_providers = ()
        if listener is not None:
            listener.close()
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        self._settle_process(process)
        if stderr_file is not None:
            stderr_file.close()

    def _ensure_started(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Kokoro worker client is closed")
            started = self._socket is not None
        if not started:
            self._start()

    def _start(self) -> None:
        listener = _create_exclusive_listener()
        token = secrets.token_urlsafe(32)
        environment = os.environ.copy()
        environment[_TOKEN_ENV] = token
        port = cast(tuple[str, int], listener.getsockname())[1]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process: subprocess.Popen[bytes] | None = None
        connection: socket.socket | None = None
        stderr_file: IO[bytes] | None = None
        published = False
        try:
            # The file is owned until worker close/fault, beyond this startup scope.
            stderr_file = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
            process = subprocess.Popen(
                [
                    str(self._python_executable),
                    "-I",
                    str(self._worker_script),
                    "--connect-port",
                    str(port),
                    "--parent-pid",
                    str(os.getpid()),
                    "--model",
                    str(self._model_path),
                    "--model-sha256",
                    self._expected_model_sha256,
                    "--voices",
                    str(self._voices_path),
                    "--voices-sha256",
                    self._expected_voices_sha256,
                    "--attestation-voice",
                    self._attestation_voice,
                    "--language",
                    self._language,
                    "--max-pcm-bytes",
                    str(self._max_pcm_bytes),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                shell=False,
                env=environment,
                creationflags=creationflags,
            )
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Kokoro worker client is closed")
                self._listener = listener
                self._process = process
                self._stderr_file = stderr_file
                published = True
            deadline = time.monotonic() + self._startup_timeout
            while True:
                if process.poll() is not None:
                    diagnostics = self._read_diagnostics(stderr_file)
                    detail = f": {diagnostics}" if diagnostics else ""
                    raise RuntimeError(
                        f"Kokoro CUDA worker exited during startup with code "
                        f"{process.returncode}{detail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "Kokoro CUDA worker did not connect within the startup timeout"
                    )
                listener.settimeout(min(0.1, remaining))
                try:
                    connection, address = listener.accept()
                    break
                except TimeoutError:
                    continue
            if address[0] != "127.0.0.1":
                raise RuntimeError("Kokoro worker connected from a non-loopback address")
            connection.settimeout(self._startup_timeout)
            ready = self._receive_json(connection)
            providers = self._validate_ready(ready, token)
            with self._state_lock:
                if self._closed or self._process is not process:
                    raise RuntimeError("Kokoro worker client closed during startup")
                self._listener = None
                self._socket = connection
                self._actual_providers = providers
            process = None
            connection = None
            stderr_file = None
        except BaseException:
            if connection is not None:
                connection.close()
            if published:
                with self._state_lock:
                    if self._process is process:
                        self._listener = None
                        self._process = None
                        self._stderr_file = None
                        self._actual_providers = ()
                    else:
                        process = None
                        stderr_file = None
            self._settle_process(process)
            if stderr_file is not None:
                stderr_file.close()
            raise
        finally:
            listener.close()

    def _validate_ready(self, value: Mapping[str, object], token: str) -> tuple[str, ...]:
        required = {
            "modelSha256",
            "provider",
            "providers",
            "executionPolicy",
            "profiledCudaEvents",
            "profiledCpuEvents",
            "sampleRateHz",
            "token",
            "type",
            "version",
            "voicesSha256",
        }
        if set(value) != required:
            raise RuntimeError("Kokoro worker ready message has an invalid shape")
        providers = value.get("providers")
        profiled_cuda_events = value.get("profiledCudaEvents")
        profiled_cpu_events = value.get("profiledCpuEvents")
        if (
            type(value.get("version")) is not int
            or value.get("version") != _PROTOCOL_VERSION
            or type(value.get("type")) is not str
            or value.get("type") != "ready"
            or type(value.get("token")) is not str
            or not secrets.compare_digest(cast(str, value.get("token")), token)
            or type(value.get("provider")) is not str
            or value.get("provider") != "CUDAExecutionProvider"
            or type(providers) is not list
            or providers != ["CUDAExecutionProvider", "CPUExecutionProvider"]
            or type(value.get("executionPolicy")) is not str
            or value.get("executionPolicy") != "cuda-primary-profiled"
            or type(profiled_cuda_events) is not int
            or profiled_cuda_events < len(_REQUIRED_CUDA_PROFILE_OPS)
            or type(profiled_cpu_events) is not int
            or profiled_cpu_events < 0
            or type(value.get("modelSha256")) is not str
            or value.get("modelSha256") != self._expected_model_sha256
            or type(value.get("voicesSha256")) is not str
            or value.get("voicesSha256") != self._expected_voices_sha256
            or type(value.get("sampleRateHz")) is not int
            or value.get("sampleRateHz") != _MODEL_SAMPLE_RATE_HZ
        ):
            raise RuntimeError("Kokoro CUDA worker failed startup attestation")
        return tuple(cast(list[str], providers))

    def _validate_result(self, value: Mapping[str, object], request_id: str) -> None:
        if set(value) != {
            "pcmBytes",
            "requestId",
            "sampleRateHz",
            "type",
            "version",
        }:
            raise RuntimeError("Kokoro worker result has an invalid shape")
        pcm_bytes = value.get("pcmBytes")
        if (
            type(value.get("version")) is not int
            or value.get("version") != _PROTOCOL_VERSION
            or type(value.get("type")) is not str
            or value.get("type") != "result"
            or type(value.get("requestId")) is not str
            or value.get("requestId") != request_id
            or type(value.get("sampleRateHz")) is not int
            or value.get("sampleRateHz") != _MODEL_SAMPLE_RATE_HZ
            or type(pcm_bytes) is not int
            or not 1 <= pcm_bytes <= self._max_pcm_bytes // 2
        ):
            raise RuntimeError("Kokoro worker result failed validation")

    @staticmethod
    def _validate_request(text: str, voice: str, speed: float) -> None:
        if type(text) is not str or not text.strip() or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("text must contain 1 to 4096 characters")
        if type(voice) is not str or not voice or len(voice) > _MAX_ID_CHARS:
            raise ValueError("voice exceeds supported bounds")
        if type(speed) is not float or not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be an exact float from 0.5 through 2.0")

    @staticmethod
    def _send_json(connection: socket.socket, value: Mapping[str, object]) -> None:
        encoded = json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if not encoded or len(encoded) > _MAX_HEADER_BYTES:
            raise RuntimeError("Kokoro worker protocol header exceeds supported bounds")
        connection.sendall(struct.pack("!I", len(encoded)) + encoded)

    @classmethod
    def _receive_json(cls, connection: socket.socket) -> dict[str, object]:
        size = struct.unpack("!I", cls._receive_exact(connection, 4))[0]
        if not 1 <= size <= _MAX_HEADER_BYTES:
            raise RuntimeError("Kokoro worker protocol header exceeds supported bounds")
        try:
            value = json.loads(
                cls._receive_exact(connection, size),
                object_pairs_hook=cls._reject_duplicate_keys,
            )
        except _DuplicateJsonKey as error:
            raise RuntimeError(f"duplicate JSON key: {error}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Kokoro worker emitted malformed JSON") from error
        if type(value) is not dict:
            raise RuntimeError("Kokoro worker message must be an exact JSON object")
        return cast(dict[str, object], value)

    @staticmethod
    def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise _DuplicateJsonKey(key)
            value[key] = item
        return value

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        blocks: list[bytes] = []
        remaining = size
        while remaining:
            block = connection.recv(remaining)
            if not block:
                raise RuntimeError("Kokoro worker connection closed unexpectedly")
            blocks.append(block)
            remaining -= len(block)
        return b"".join(blocks)

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Kokoro worker is unavailable")
        return self._socket

    def _fault(self, expected_connection: socket.socket) -> None:
        with self._state_lock:
            if self._socket is not expected_connection:
                return
            connection = self._socket
            process = self._process
            stderr_file = self._stderr_file
            self._socket = None
            self._process = None
            self._stderr_file = None
            self._actual_providers = ()
        if connection is not None:
            connection.close()
        self._settle_process(process)
        if stderr_file is not None:
            stderr_file.close()

    @staticmethod
    def _settle_process(process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    @staticmethod
    def _read_diagnostics(stderr_file: IO[bytes]) -> str:
        stderr_file.flush()
        stderr_file.seek(0)
        return stderr_file.read(8192).decode("utf-8", errors="replace").strip()


def _is_close_request(request: Mapping[str, object]) -> bool:
    return (
        set(request) == {"type", "version"}
        and type(request.get("version")) is int
        and request.get("version") == _PROTOCOL_VERSION
        and type(request.get("type")) is str
        and request.get("type") == "close"
    )


def serve_kokoro_worker_connection(
    *,
    connection: socket.socket,
    token: str,
    model_sha256: str,
    voices_sha256: str,
    actual_providers: tuple[str, ...],
    profiled_cuda_events: int,
    profiled_cpu_events: int,
    max_pcm_bytes: int,
    synthesize_pcm: _WorkerSynthesizePcm,
) -> None:
    """Serve one authenticated sequential worker connection."""

    if (
        type(token) is not str
        or not token
        or type(model_sha256) is not str
        or type(voices_sha256) is not str
        or type(actual_providers) is not tuple
        or not actual_providers
        or any(type(provider) is not str for provider in actual_providers)
        or actual_providers[0] != "CUDAExecutionProvider"
        or type(profiled_cuda_events) is not int
        or profiled_cuda_events < len(_REQUIRED_CUDA_PROFILE_OPS)
        or type(profiled_cpu_events) is not int
        or profiled_cpu_events < 0
        or type(max_pcm_bytes) is not int
        or not 1 <= max_pcm_bytes <= _MAX_PCM_BYTES
        or not callable(synthesize_pcm)
    ):
        raise ValueError("Kokoro worker server configuration is invalid")
    KokoroWorkerClient._send_json(
        connection,
        {
            "version": _PROTOCOL_VERSION,
            "type": "ready",
            "token": token,
            "provider": "CUDAExecutionProvider",
            "providers": list(actual_providers),
            "executionPolicy": "cuda-primary-profiled",
            "profiledCudaEvents": profiled_cuda_events,
            "profiledCpuEvents": profiled_cpu_events,
            "modelSha256": model_sha256,
            "voicesSha256": voices_sha256,
            "sampleRateHz": _MODEL_SAMPLE_RATE_HZ,
        },
    )
    while True:
        request = KokoroWorkerClient._receive_json(connection)
        if _is_close_request(request):
            KokoroWorkerClient._send_json(
                connection,
                {"version": _PROTOCOL_VERSION, "type": "closed"},
            )
            return
        if set(request) != {"requestId", "speed", "text", "type", "version", "voice"}:
            raise RuntimeError("Kokoro worker request has an invalid shape")
        request_id = request.get("requestId")
        text = request.get("text")
        voice = request.get("voice")
        speed = request.get("speed")
        if (
            type(request.get("version")) is not int
            or request.get("version") != _PROTOCOL_VERSION
            or type(request.get("type")) is not str
            or request.get("type") != "synthesize"
            or type(request_id) is not str
            or not request_id
            or len(request_id) > _MAX_ID_CHARS
            or type(text) is not str
            or not text.strip()
            or len(text) > _MAX_TEXT_CHARS
            or type(voice) is not str
            or not voice
            or len(voice) > _MAX_ID_CHARS
            or type(speed) is not float
            or not 0.5 <= speed <= 2.0
        ):
            raise RuntimeError("Kokoro worker request failed validation")
        pcm = synthesize_pcm(text, voice, speed)
        if type(pcm) is not bytes or not pcm or len(pcm) > max_pcm_bytes // 2 or len(pcm) % 2:
            raise RuntimeError("Kokoro worker PCM failed validation")
        KokoroWorkerClient._send_json(
            connection,
            {
                "version": _PROTOCOL_VERSION,
                "type": "result",
                "requestId": request_id,
                "pcmBytes": len(pcm),
                "sampleRateHz": _MODEL_SAMPLE_RATE_HZ,
            },
        )
        connection.sendall(pcm)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _start_parent_watchdog(parent_pid: int) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        synchronize = 0x00100000
        infinite = 0xFFFFFFFF
        wait_object_0 = 0
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        open_process.restype = ctypes.c_void_p
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        wait_for_single_object.restype = ctypes.c_uint32
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        handle = open_process(synchronize, False, parent_pid)
        if not handle:
            raise RuntimeError("Kokoro worker could not own a parent-process handle")

        def watch_windows_parent() -> None:
            result = wait_for_single_object(handle, infinite)
            close_handle(handle)
            if result == wait_object_0:
                os._exit(70)
            os._exit(71)

        threading.Thread(target=watch_windows_parent, daemon=True).start()
        return

    def watch_posix_parent() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except OSError:
                os._exit(70)
            time.sleep(0.5)

    threading.Thread(target=watch_posix_parent, daemon=True).start()


def _run_cuda_worker(args: argparse.Namespace) -> None:
    token = os.environ.get(_TOKEN_ENV)
    if token is None or not token:
        raise RuntimeError("Kokoro worker authentication token is unavailable")
    _start_parent_watchdog(args.parent_pid)
    model_path = Path(args.model).resolve(strict=True)
    voices_path = Path(args.voices).resolve(strict=True)
    expected_model_sha256 = _require_sha256(args.model_sha256, "model SHA-256")
    expected_voices_sha256 = _require_sha256(args.voices_sha256, "voices SHA-256")
    model_sha256 = _sha256_file(model_path)
    voices_sha256 = _sha256_file(voices_path)
    if not secrets.compare_digest(model_sha256, expected_model_sha256):
        raise RuntimeError("Kokoro model failed pre-load SHA-256 attestation")
    if not secrets.compare_digest(voices_sha256, expected_voices_sha256):
        raise RuntimeError("Kokoro voices failed pre-load SHA-256 attestation")
    runtime = importlib.import_module("onnxruntime")
    preload_dlls = getattr(runtime, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls(directory="")
    module = importlib.import_module("kokoro_onnx")
    engine_type = getattr(module, "Kokoro", None)
    from_session = getattr(engine_type, "from_session", None)
    if not callable(from_session):
        raise RuntimeError("kokoro-onnx Kokoro.from_session is unavailable")
    session_options = runtime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(
        Path(tempfile.gettempdir()) / f"hermes-kokoro-profile-{os.getpid()}"
    )
    session = runtime.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    actual_providers = tuple(session.get_providers())
    if actual_providers != ("CUDAExecutionProvider", "CPUExecutionProvider"):
        raise RuntimeError(
            f"Kokoro CUDA worker selected {actual_providers!r}; refusing CPU fallback"
        )
    engine = cast(Any, from_session(session, str(voices_path)))
    numpy = importlib.import_module("numpy")
    attestation_samples, attestation_rate = engine.create(
        "CUDA provider attestation.",
        voice=args.attestation_voice,
        speed=1.0,
        lang=args.language,
    )
    if (
        type(attestation_rate) is not int
        or attestation_rate != _MODEL_SAMPLE_RATE_HZ
        or not numpy.asarray(attestation_samples).size
    ):
        raise RuntimeError("Kokoro CUDA worker attestation synthesis failed")
    profile_path = Path(session.end_profiling())
    try:
        profile_events = json.loads(profile_path.read_bytes())
        profiled_cuda_events, profiled_cpu_events = _validate_cuda_profile_events(profile_events)
    finally:
        profile_path.unlink(missing_ok=True)

    def synthesize_pcm(text: str, voice: str, speed: float) -> bytes:
        samples, sample_rate = engine.create(
            text,
            voice=voice,
            speed=speed,
            lang=args.language,
        )
        if type(sample_rate) is not int or sample_rate != _MODEL_SAMPLE_RATE_HZ:
            raise RuntimeError("Kokoro CUDA worker produced an unsupported sample rate")
        array = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
        if not array.size:
            raise RuntimeError("Kokoro CUDA worker produced no audio")
        clipped = numpy.clip(array, -1.0, 1.0)
        return bytes((clipped * 32767.0).astype("<i2", copy=False).tobytes())

    with socket.create_connection(("127.0.0.1", args.connect_port), timeout=15.0) as connection:
        connection.settimeout(None)
        serve_kokoro_worker_connection(
            connection=connection,
            token=token,
            model_sha256=model_sha256,
            voices_sha256=voices_sha256,
            actual_providers=actual_providers,
            profiled_cuda_events=profiled_cuda_events,
            profiled_cpu_events=profiled_cpu_events,
            max_pcm_bytes=args.max_pcm_bytes,
            synthesize_pcm=synthesize_pcm,
        )


def main() -> int:
    args = _worker_argument_parser().parse_args()
    _run_cuda_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["KokoroWorkerClient", "serve_kokoro_worker_connection"]
