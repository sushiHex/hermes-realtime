"""One source-bound vram-mcp worker scope; never a direct-launch fallback."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

from hermes_realtime.providers.kokoro_worker import KokoroWorkerClient

_RPC_SECONDS = 15.0
_WAIT_SECONDS = 20.0
_CLOSE_SECONDS = 45.0
_POLICIES = {"cuda-primary-profiled", "cuda-primary-profiled-kokoro-v1-benchmark"}
_CONFIG_FIELDS = {
    "version", "server_python", "server_python_sha256", "server_venv_python",
    "server_venv_python_sha256", "server_source", "server_commit", "server_tree",
    "profile", "profile_sha256",
}
_PROFILE_FIELDS = {
    "version", "python", "python_sha256", "venv_python", "venv_python_sha256",
    "worker", "worker_sha256", "model", "model_sha256", "voices", "voices_sha256",
    "gpu_index", "gpu_uuid", "reserve_gb", "headroom_mb", "voice", "language",
}
_RESULT_FIELDS = {
    "ok", "outcome", "state", "reason", "operation_id", "gpu_ready",
    "worker_exit_code", "worker_exit_category",
}
_ENV = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "USERPROFILE", "HOME",
        "LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"}
_T = TypeVar("_T")


class KokoroMcpError(RuntimeError):
    """Fixed refusal only: no SDK errors, paths, tokens, handles or payloads."""

    def __init__(self, category: str) -> None:
        try:
            super().__init__(category)
        finally:
            stream = sys.stderr
            if stream is not None:
                try:
                    print("[kokoro-mcp-refused] " + json.dumps(
                        {"version": 1, "category": category}, separators=(",", ":"),
                    ), file=stream, flush=True)
                except Exception:
                    if sys.stderr is stream:
                        sys.stderr = None


def _require(condition: bool, category: str) -> None:
    if not condition:
        raise KokoroMcpError(category)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, "configuration_invalid")
        result[key] = value
    return result


def _json_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(16385)
    _require(len(raw) <= 16384, "configuration_invalid")
    value = json.loads(raw, object_pairs_hook=_unique)
    _require(type(value) is dict, "configuration_invalid")
    return dict(value)


@dataclass(frozen=True)
class _Config:
    server_python: Path
    server_python_sha256: str
    server_venv_python: Path
    server_venv_python_sha256: str
    server_source: Path
    server_commit: str
    server_tree: str
    profile: Path
    profile_sha256: str


def _read_config(path: Path) -> tuple[_Config, dict[str, Any]]:
    try:
        raw = _json_file(path)
        _require(set(raw) == _CONFIG_FIELDS and type(raw["version"]) is int
                 and raw["version"] == 1, "configuration_invalid")
        for field in ("server_python", "server_venv_python", "server_source", "profile"):
            _require(type(raw[field]) is str, "configuration_invalid")
            value = Path(raw[field])
            _require(value.is_absolute() and value.exists() and not value.is_symlink(),
                     "configuration_invalid")
            raw[field] = value.resolve(strict=True)
        for field in ("server_python_sha256", "server_venv_python_sha256", "profile_sha256"):
            _require(type(raw[field]) is str and re.fullmatch(r"[0-9a-f]{64}", raw[field])
                     is not None, "configuration_invalid")
        for field in ("server_commit", "server_tree"):
            _require(type(raw[field]) is str and re.fullmatch(r"[0-9a-f]{40}", raw[field])
                     is not None, "configuration_invalid")
        del raw["version"]
        config = _Config(**raw)
        _require(_digest(config.profile) == config.profile_sha256, "profile_binding")
        profile = _json_file(config.profile)
        _require(set(profile) in (_PROFILE_FIELDS, _PROFILE_FIELDS | {"execution_policy"})
                 and type(profile["version"]) is int and profile["version"] == 1,
                 "profile_binding")
        policy = profile.get("execution_policy", "cuda-primary-profiled")
        _require(type(policy) is str and policy in _POLICIES, "profile_binding")
        profile["execution_policy"] = policy
        for field in ("venv_python", "worker", "model", "voices"):
            _require(type(profile[field]) is str and Path(profile[field]).is_absolute(),
                     "profile_binding")
        return config, profile
    except KokoroMcpError:
        raise
    except Exception:
        raise KokoroMcpError("configuration_invalid") from None


def _verify_source(config: _Config) -> None:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=config.server_source, text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    try:
        _require(git("remote", "get-url", "origin") == "https://github.com/sushiHex/vram-mcp.git"
                 and git("rev-parse", "HEAD") == config.server_commit
                 and git("rev-parse", "HEAD^{tree}") == config.server_tree, "source_binding")
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=config.server_source,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        tracked = set(git("ls-files").splitlines())
        suffixes = {".py", ".pyw", ".pyc", ".pyd", ".so"}
        expected_roots = {
            item.split("/", 1)[0] + "/" for item in tracked
            if "/" in item and item.split("/", 1)[0].isidentifier()
        } | {item for item in tracked if "/" not in item and Path(item).suffix in suffixes}
        actual_roots = {
            item.name + "/" for item in config.server_source.iterdir()
            if item.is_dir() and item.name.isidentifier()
        } | {item.name for item in config.server_source.iterdir()
             if item.is_file() and item.suffix in suffixes}
        package = config.server_source / "vram_mcp"
        expected_package = {item for item in tracked
                            if item.startswith("vram_mcp/") and Path(item).suffix in suffixes}
        actual_package = {item.relative_to(config.server_source).as_posix()
                          for item in package.rglob("*")
                          if item.is_file() and item.suffix in suffixes}
        _require(expected_roots == actual_roots and expected_package == actual_package
                 and not any(item.is_symlink() for item in package.rglob("*")), "source_binding")
    except KokoroMcpError:
        raise
    except Exception:
        raise KokoroMcpError("source_binding") from None


def _environment(config: _Config) -> dict[str, str]:
    profile = _json_file(config.profile)
    index = profile.get("gpu_index")
    _require(type(index) is int and index >= 0, "profile_binding")
    result = {key: value for key, value in os.environ.items() if key.upper() in _ENV}
    result.update(__PYVENV_LAUNCHER__=str(config.server_venv_python),
                  VRAM_MCP_KOKORO_PROFILE=str(config.profile), PYTHONDONTWRITEBYTECODE="1",
                  VRAM_MCP_GPU_INDEX=str(index))
    return result


def _verify_interpreter(config: _Config) -> None:
    try:
        _require(sys.platform == "win32", "unsupported_platform")
        _require(_digest(config.server_python) == config.server_python_sha256
                 and _digest(config.server_venv_python) == config.server_venv_python_sha256,
                 "interpreter_binding")
        code = ("import ctypes,json,sys;b=ctypes.create_unicode_buffer(32768);"
                "n=ctypes.windll.kernel32.GetModuleFileNameW(None,b,len(b));"
                "print(json.dumps([b.value if n else None,sys._base_executable,sys.executable]))")
        environment = _environment(config)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([str(config.server_python), "-I", "-B", "-S", "-c", code],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
                                env=environment,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _require(result.returncode == 0 and len(result.stdout) <= 4096, "interpreter_binding")
        values = json.loads(result.stdout)
        _require(type(values) is list and len(values) == 3
                 and all(type(value) is str for value in values), "interpreter_binding")
        _require(Path(values[0]).resolve() == config.server_python
                 and Path(values[1]).resolve() == config.server_python
                 and Path(values[2]).resolve() == config.server_venv_python, "interpreter_binding")
    except KokoroMcpError:
        raise
    except Exception:
        raise KokoroMcpError("interpreter_binding") from None


@asynccontextmanager
async def _open_session(config: _Config) -> AsyncIterator[Any]:
    # CPU installs do not import or require the optional SDK.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    bootstrap = (f"import sys;sys.path.insert(0,{str(config.server_source)!r});"
                 "from vram_mcp.server import main;main()")
    parameters = StdioServerParameters(command=str(config.server_python),
        args=["-I", "-B", "-c", bootstrap], env=_environment(config))
    with open(os.devnull, "w", encoding="utf-8") as errors:
        async with (
            stdio_client(parameters, errlog=errors) as (read, write),
            ClientSession(read, write,
                          read_timeout_seconds=timedelta(seconds=_RPC_SECONDS)) as session,
        ):
            await session.initialize()
            yield session


class KokoroMcpWorkerFactory:
    """Own one attempted dispatch and one same-task SDK scope for its lifetime."""

    def __init__(self, config: _Config, profile: dict[str, Any]) -> None:
        self._config, self._profile = config, profile
        self._lock, self._close_lock = threading.RLock(), threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._io_thread: int | None = None
        self._ready = threading.Event()
        self._finish: asyncio.Event | None = None
        self._session: Any = None
        self._requests: set[asyncio.Task[Any]] = set()
        self._load_future: concurrent.futures.Future[None] | None = None
        self._clients: list[KokoroWorkerClient] = []
        self._owner, self._request_id = "hermes-kokoro", secrets.token_urlsafe(32)
        self._operation: str | None = None
        self._launch_attempted = self._closing = self._closed = False
        self._known_refusal = self._stopped = False
        self._transport_closed = False
        self._exit_code: int | None = None
        self._cleanup_state = "not_started"

    @classmethod
    def from_config(cls, path: Path) -> KokoroMcpWorkerFactory:
        config, profile = _read_config(path)
        return cls(config, profile)

    @property
    def cleanup_state(self) -> str:
        with self._lock:
            return self._cleanup_state

    def create(self, *, model_path: Path, voices_path: Path, expected_model_sha256: str,
               expected_voices_sha256: str, language: str, max_pcm_bytes: int,
               attestation_voice: str) -> KokoroWorkerClient:
        with self._lock:
            _require(not self._closing and not self._closed and not self._clients, "closed")
            p = self._profile
            _require(model_path.resolve() == Path(p["model"]).resolve()
                     and voices_path.resolve() == Path(p["voices"]).resolve()
                     and expected_model_sha256 == p["model_sha256"]
                     and expected_voices_sha256 == p["voices_sha256"]
                     and language == p["language"] and attestation_voice == p["voice"]
                     and type(max_pcm_bytes) is int and 0 < max_pcm_bytes <= 16 * 1024 * 1024,
                     "client_binding")
            client = KokoroWorkerClient(
                python_executable=Path(p["venv_python"]), worker_script=Path(p["worker"]),
                model_path=model_path, voices_path=voices_path,
                expected_model_sha256=expected_model_sha256,
                expected_voices_sha256=expected_voices_sha256, language=language,
                max_pcm_bytes=max_pcm_bytes, attestation_voice=attestation_voice,
                execution_policy=p["execution_policy"], worker_launcher=self,
            )
            self._clients.append(client)
            return client

    def _not_io_thread(self) -> None:
        _require(threading.get_ident() != self._io_thread, "same_thread")

    def _start_transport(self) -> None:
        _verify_source(self._config)
        _verify_interpreter(self._config)
        _require(_digest(self._config.profile) == self._config.profile_sha256, "profile_binding")
        with self._lock:
            _require(not self._closing, "closed")
            self._thread = threading.Thread(target=self._run, name="kokoro-mcp", daemon=True)
            self._thread.start()
        _require(self._ready.wait(_WAIT_SECONDS) and self._session is not None,
                 "transport_unavailable")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop, self._io_thread = loop, threading.get_ident()
        try:
            loop.run_until_complete(self._own_scope())
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            self._ready.set()

    async def _own_scope(self) -> None:
        self._finish = asyncio.Event()
        try:
            if self._closing:
                self._transport_closed = True
                return
            async with _open_session(self._config) as session:
                self._session = session
                self._ready.set()
                try:
                    if not self._closing:
                        await self._finish.wait()
                finally:
                    pending = tuple(self._requests)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
        except BaseException:
            self._transport_closed = False
        else:
            self._transport_closed = True
        finally:
            self._session = None
            self._ready.set()

    async def _call(self, name: str, arguments: dict[str, object]) -> dict[str, Any]:
        _require(name in {"load_kokoro", "kokoro_status", "stop_kokoro"}, "transport_unavailable")
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        try:
            result = await asyncio.wait_for(self._session.call_tool(name, arguments), _RPC_SECONDS)
            value = result.structuredContent
            _require(result.isError is False and type(value) is dict
                     and set(value) == _RESULT_FIELDS and value.get("gpu_ready") is False,
                     "transport_unavailable")
            return dict(value)
        except KokoroMcpError:
            raise
        except Exception:
            raise KokoroMcpError("transport_unavailable") from None
        finally:
            self._requests.discard(task)

    def _submit(self, coroutine: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        if threading.get_ident() == self._io_thread:
            coroutine.close()
            raise KokoroMcpError("same_thread")
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            raise KokoroMcpError("transport_unavailable")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def _wait(self, future: concurrent.futures.Future[_T], timeout: float = _WAIT_SECONDS) -> _T:
        try:
            return future.result(timeout=timeout)
        except KokoroMcpError:
            raise
        except Exception:
            raise KokoroMcpError("transport_unavailable") from None

    async def _load(self, connect_port: int, token: str) -> None:
        _require(not self._closing, "closed")
        result = await self._call("load_kokoro", {
            "owner": self._owner, "request_id": self._request_id,
            "connect_port": connect_port, "auth_token": token,
        })
        operation = result["operation_id"]
        if (result["ok"] is False and result["outcome"] == "refused"
                and result["state"] == "refused" and operation is None):
            self._known_refusal = True
            raise KokoroMcpError("load_refused")
        if type(operation) is str and re.fullmatch(r"[A-Za-z0-9_-]{32,128}", operation):
            self._operation = operation
        _require(self._operation is not None and result["ok"] is True
                 and result["outcome"] == "succeeded" and result["state"] == "launch_accepted",
                 "load_unknown")

    def launch(self, *, connect_port: int, token: str) -> _OwnedWorker:
        self._not_io_thread()
        with self._lock:
            _require(not self._closing and not self._closed, "closed")
            _require(not self._launch_attempted, "launch_attempted")
            _require(type(connect_port) is int and 1024 <= connect_port <= 65535
                     and type(token) is str and re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token)
                     is not None, "client_binding")
            self._launch_attempted = True
        try:
            self._start_transport()
            future = self._submit(self._load(connect_port, token))
            self._load_future = future
            self._wait(future)
            _require(not self._closing, "closed")
            return _OwnedWorker(self)
        except BaseException:
            with contextlib.suppress(KokoroMcpError):
                self.close()
            raise

    async def _status(self) -> int | None:
        _require(self._operation is not None, "status_unknown")
        result = await self._call("kokoro_status", {"owner": self._owner,
                                                   "operation_id": self._operation})
        _require(result["operation_id"] == self._operation and result["ok"] is True
                 and result["outcome"] == "succeeded", "status_unknown")
        if result["state"] == "stopped":
            code = result["worker_exit_code"]
            _require(type(code) is int and -(2 ** 31) <= code <= 2 ** 32 - 1, "status_unknown")
            self._exit_code, self._stopped = cast(int, code), True
            return self._exit_code
        _require(result["ok"] is True and result["outcome"] == "succeeded"
                 and result["state"] == "launch_accepted", "status_unknown")
        return None

    async def _stop(self) -> None:
        _require(self._operation is not None, "cleanup_unknown")
        if not self._stopped:
            result = await self._call("stop_kokoro", {"owner": self._owner,
                                                     "operation_id": self._operation})
            _require(result["operation_id"] == self._operation and result["state"] == "stopped"
                     and result["ok"] is True and result["outcome"] == "succeeded",
                     "cleanup_unknown")
        _require(await self._status() is not None, "cleanup_unknown")

    async def _shutdown(self) -> None:
        try:
            pending = self._load_future
            if pending is not None and not pending.done():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(pending)),
                                           _RPC_SECONDS)
            if self._operation is not None:
                await self._stop()
                self._cleanup_state = "stopped"
            elif self._known_refusal:
                self._cleanup_state = "not_loaded"
            elif self._launch_attempted:
                self._cleanup_state = "cleanup_unknown"
        except BaseException:
            self._cleanup_state = "cleanup_unknown"
        finally:
            if self._finish is not None:
                self._finish.set()

    def close(self) -> None:
        self._not_io_thread()
        with self._close_lock:
            if self._closed:
                _require(self._cleanup_state != "cleanup_unknown", "cleanup_unknown")
                return
            with self._lock:
                self._closing = True
                clients = tuple(self._clients)
            clients_ok = True
            for client in clients:
                try:
                    client.close()
                except BaseException:
                    clients_ok = False
            thread = self._thread
            if thread is not None:
                try:
                    self._wait(self._submit(self._shutdown()), _CLOSE_SECONDS)
                except KokoroMcpError:
                    self._cleanup_state = "cleanup_unknown"
                thread.join(timeout=_CLOSE_SECONDS)
                if thread.is_alive() or not self._transport_closed:
                    self._cleanup_state = "cleanup_unknown"
            elif self._launch_attempted:
                self._cleanup_state = "cleanup_unknown"
            if not clients_ok:
                self._cleanup_state = "cleanup_unknown"
            self._closed = True
            _require(self._cleanup_state != "cleanup_unknown", "cleanup_unknown")


class _OwnedWorker:
    def __init__(self, factory: KokoroMcpWorkerFactory) -> None:
        self._factory = factory

    @property
    def returncode(self) -> int | None:
        return self._factory._exit_code

    def poll(self) -> int | None:
        return self._factory._wait(self._factory._submit(self._factory._status()))

    def terminate(self) -> None:
        self._factory._wait(self._factory._submit(self._factory._stop()))

    def wait(self, timeout: float | None = None) -> int:
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired("owned-kokoro-operation", timeout or 0.0)
        return code

    def kill(self) -> None:
        raise KokoroMcpError("unsupported_force")
