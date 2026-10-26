"""Verify artifacts and admit an owned, loaded Ollama before any gateway task.

This preflight binds bytes at verification time; it is not a file-seal, capacity or
useful-task qualification. The caller retains the owner through all consumers and settles it in
finally. Only the owned child receives the selected closed startup settings.
"""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, NoReturn

from scripts.qualification_file_seals import _relative_windows_member

MODEL_PROCESS_NAME = "lookup-model"
SUPPORTED_VERSION = "0.40.4"


class StartupProfile(Enum):
    """The two admitted Ollama startup settings; callers cannot supply raw values."""

    LOOKUP_64K = 1
    NATIVE_TOOLS_4K = 2

    @property
    def context_tokens(self) -> int:
        return 64000 if self is StartupProfile.LOOKUP_64K else 4096

    @property
    def go_template(self) -> str | None:
        return None if self is StartupProfile.LOOKUP_64K else "1"

    @property
    def profile_id(self) -> str:
        return "LOOKUP_64K" if self is StartupProfile.LOOKUP_64K else "NATIVE_TOOLS_4K"


_JSON_LIMIT = 1024 * 1024
_REFUSALS = frozenset({
    "model-deadline", "artifact-file-kind", "artifact-json-bound", "artifact-json-shape",
    "archive-digest", "runtime-file-bound", "archive-member-bound", "archive-member-spelling",
    "archive-member-kind", "archive-member-size", "runtime-members", "runtime-content",
    "model-reference", "model-manifest-digest", "model-manifest-shape", "model-blob-descriptor",
    "model-blob-size", "model-blob-digest", "model-config", "model-layer-layout",
    "artifact-context-override", "listener-owner", "metadata-redirect", "metadata-kind",
    "metadata-status", "metadata-bound", "metadata-shape", "model-port", "model-already-attempted",
    "owner-containment", "port-occupied", "model-exited", "metadata-version", "metadata-model",
    "metadata-template", "metadata-capabilities", "artifact-proof",
    "load-ack", "loaded-model", "loaded-context", "startup-profile", "readiness-status",
    "readiness-bound",
    "readiness-shape", "readiness-json", "model-readiness-loop",
})
_SELECTION = ("model", "model_store", "model_manifest_sha256", "ollama_runtime",
              "ollama_archive", "ollama_archive_sha256", "model_port")


@dataclass(frozen=True)
class VerifiedArtifacts:
    """Invocation-local, immutable result; serialized facts do not authorize reuse."""

    selection: tuple[Any, ...]
    facts: Mapping[str, Any]


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("model-deadline")
    return remaining


def _hash(stream: IO[bytes], deadline: float) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        _remaining(deadline)
        digest.update(chunk)
        size += len(chunk)
    _remaining(deadline)
    return digest.hexdigest(), size


def _file_hash(path: Path, deadline: float) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _hash(stream, deadline)


def _ordinary(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError("artifact-file-kind")


def _json_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(_JSON_LIMIT + 1)
    if len(raw) > _JSON_LIMIT:
        raise ValueError("artifact-json-bound")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("artifact-json-shape")
    return value


def _runtime(manifest: Mapping[str, Any], deadline: float) -> dict[str, Any]:
    archive_path = Path(manifest["ollama_archive"])
    runtime = Path(manifest["ollama_runtime"])
    _ordinary(archive_path)
    _ordinary(runtime, directory=True)
    archive_digest, archive_size = _file_hash(archive_path, deadline)
    if archive_digest != manifest["ollama_archive_sha256"]:
        raise ValueError("archive-digest")
    actual: dict[str, tuple[str, int]] = {}
    for base, directories, files in os.walk(runtime, followlinks=False):
        _remaining(deadline)
        for name in directories:
            _ordinary(Path(base) / name, directory=True)
        for name in files:
            path = Path(base) / name
            _ordinary(path)
            relative = path.relative_to(runtime).as_posix()
            _relative_windows_member(relative)
            if len(actual) >= 16384:
                raise ValueError("runtime-file-bound")
            actual[relative] = _file_hash(path, deadline)
    expected: dict[str, tuple[str, int]] = {}
    explicit: set[str] = set()
    spelling: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) > 32768:
            raise ValueError("archive-member-bound")
        for member in archive.infolist():
            _remaining(deadline)
            name = member.filename[:-1] if member.is_dir() else member.filename
            _relative_windows_member(name)
            parts = name.split("/")
            for index in range(1, len(parts) + 1):
                prefix = "/".join(parts[:index])
                if spelling.setdefault(prefix.casefold(), prefix) != prefix:
                    raise ValueError("archive-member-spelling")
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (name in explicit or member.flag_bits & 1
                    or kind not in ({0, stat.S_IFDIR} if member.is_dir() else {0, stat.S_IFREG})):
                raise ValueError("archive-member-kind")
            explicit.add(name)
            if not member.is_dir():
                with archive.open(member) as stream:
                    expected[name] = _hash(stream, deadline)
                if expected[name][1] != member.file_size:
                    raise ValueError("archive-member-size")
    if not expected or "ollama.exe" not in expected:
        raise ValueError("runtime-members")
    if actual != expected:
        raise ValueError("runtime-content")
    canonical = json.dumps(sorted((name, *value) for name, value in actual.items()),
                           separators=(",", ":")).encode()
    return {"archive_sha256": archive_digest, "archive_bytes": archive_size,
        "runtime_files": len(actual), "runtime_bytes": sum(value[1] for value in actual.values()),
        "runtime_closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "executable_sha256": actual["ollama.exe"][0]}


def _model(manifest: Mapping[str, Any], deadline: float) -> dict[str, Any]:
    name = manifest["model"]
    if type(name) is not str or re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{0,127}:[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None:
        raise ValueError("model-reference")
    model, tag = name.split(":")
    store = Path(manifest["model_store"])
    _ordinary(store, directory=True)
    path = store / "manifests/registry.ollama.ai/library" / model / tag
    for parent in (path.parent, *path.parents):
        if parent == store.parent:
            break
        _ordinary(parent, directory=True)
    _ordinary(path)
    identity, _ = _file_hash(path, deadline)
    if identity != manifest["model_manifest_sha256"]:
        raise ValueError("model-manifest-digest")
    selected = _json_file(path)
    if (type(selected.get("config")) is not dict or type(selected.get("layers")) is not list
            or not 1 <= len(selected["layers"]) <= 4096):
        raise ValueError("model-manifest-shape")
    descriptors = [selected["config"], *selected["layers"]]
    _ordinary(store / "blobs", directory=True)
    for descriptor in descriptors:
        if (type(descriptor) is not dict or type(descriptor.get("digest")) is not str
                or re.fullmatch(r"sha256:[0-9a-f]{64}", descriptor["digest"]) is None
                or type(descriptor.get("size")) is not int or descriptor["size"] < 0):
            raise ValueError("model-blob-descriptor")
        blob = store / "blobs" / descriptor["digest"].replace(":", "-")
        _ordinary(blob)
        actual, size = _file_hash(blob, deadline)
        if size != descriptor["size"]:
            raise ValueError("model-blob-size")
        if actual != descriptor["digest"][7:]:
            raise ValueError("model-blob-digest")
    config = _json_file(store / "blobs" / selected["config"]["digest"].replace(":", "-"))
    family = config.get("model_family")
    if (config.get("model_format") != "gguf" or type(family) is not str
            or re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", family) is None
            or config.get("renderer") is not None or config.get("parser") is not None
            or config.get("rootfs", {}).get("diff_ids")
            != [row["digest"] for row in selected["layers"]]):
        raise ValueError("model-config")
    templates = [row for row in selected["layers"]
                 if row.get("mediaType") in {"application/vnd.ollama.image.template",
                                            "application/vnd.ollama.image.prompt"}]
    params = [row for row in selected["layers"]
              if row.get("mediaType") == "application/vnd.ollama.image.params"]
    weights = [row for row in selected["layers"]
               if row.get("mediaType") == "application/vnd.ollama.image.model"]
    if (len(templates) != 1 or templates[0]["mediaType"] != "application/vnd.ollama.image.template"
            or len(params) > 1 or not weights or templates[0]["size"] > _JSON_LIMIT):
        raise ValueError("model-layer-layout")
    if params:
        options = _json_file(store / "blobs" / params[0]["digest"].replace(":", "-"))
        if "num_ctx" in options:
            raise ValueError("artifact-context-override")
    return {"model_manifest_sha256": identity,
        "model_config_sha256": selected["config"]["digest"][7:],
        "model_blob_count": len(descriptors), "model_family": family,
        "template_sha256": templates[0]["digest"][7:], "template_bytes": templates[0]["size"],
        "model_bytes": sum(row["size"] for row in weights)}


def verify_artifacts(manifest: Mapping[str, Any], deadline: float) -> VerifiedArtifacts:
    """Read only: exact archive/runtime equality and every selected model descriptor."""
    selected = dict(manifest)
    facts = {**_runtime(selected, deadline), **_model(selected, deadline)}
    return VerifiedArtifacts(tuple(selected[key] for key in _SELECTION), MappingProxyType(facts))


def _listeners(port: int, deadline: float) -> list[tuple[str, int]]:
    rows = []
    for protocol in ("tcp", "tcpv6"):
        result = subprocess.run(["netstat", "-ano", "-p", protocol], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=True, timeout=min(5.0, _remaining(deadline)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in result.stdout.splitlines():
            fields = line.split()
            if (len(fields) == 5 and fields[0] == "TCP" and fields[3] == "LISTENING"
                    and fields[1].rpartition(":")[2] == str(port)):
                rows.append((fields[1], int(fields[4])))
    return rows


def _image(pid: int) -> Path:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError("owned-image")
    try:
        size = ctypes.c_ulong(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(ctypes.c_void_p(handle), 0, buffer,
                                               ctypes.byref(size)):
            raise OSError("owned-image")
        return Path(buffer.value)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def _owned(process: Any, runtime: Path, port: int, deadline: float) -> None:
    rows = _listeners(port, deadline)
    if (process.poll() is not None or not rows
            or any(address != f"127.0.0.1:{port}" or pid != process.pid for address, pid in rows)
            or _image(process.pid).resolve() != (runtime / "ollama.exe").resolve()):
        raise RuntimeError("listener-owner")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> NoReturn:
        raise ValueError("metadata-redirect")


def _read(kind: str, manifest: Mapping[str, Any], process: Any,
          deadline: float) -> dict[str, Any]:
    if kind not in {"version", "tags", "show"}:
        raise ValueError("metadata-kind")
    runtime, port = Path(manifest["ollama_runtime"]), manifest["model_port"]
    _owned(process, runtime, port, deadline)
    body = json.dumps({"model": manifest["model"]}).encode() if kind == "show" else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/{kind}", data=body,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=min(8.0, _remaining(deadline))) as response:
        if response.status != 200:
            raise ValueError("metadata-status")
        raw = response.read(_JSON_LIMIT + 1)
    if len(raw) > _JSON_LIMIT:
        raise ValueError("metadata-bound")
    _owned(process, runtime, port, deadline)
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("metadata-shape")
    return value


async def _load_readiness(manifest: Mapping[str, Any], process: Any, deadline: float,
                          state: dict[str, Any], startup_profile: StartupProfile) -> None:
    """One empty load and one configured-context readback, under one remaining budget.

    Pinned Ollama's native chat handler awaits runner readiness before returning
    done_reason=load for empty messages. /api/ps reports configured per-sequence
    context, not independently measured KV capacity or future truncation behavior.
    """
    import httpx

    runtime, port = Path(manifest["ollama_runtime"]), manifest["model_port"]

    async def read(path: str, body: dict[str, Any] | None, phase: str) -> dict[str, Any]:
        _owned(process, runtime, port, deadline)
        state["stage"] = "model-" + phase
        if phase == "load":
            state["load_request_count"] += 1
        else:
            state["metadata_requests"].append("ps")
        state[phase + "_begin_ns"] = time.monotonic_ns()
        raw = bytearray()
        try:
            async with client.stream("POST" if body is not None else "GET",
                    f"http://127.0.0.1:{port}/api/{path}", json=body,
                    timeout=_remaining(deadline) if phase == "load"
                    else min(8.0, _remaining(deadline))) as response:
                state[phase + "_headers_ns"] = time.monotonic_ns()
                if response.status_code != 200:
                    raise ValueError("readiness-status")
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > _JSON_LIMIT:
                        state[phase + "_response_overflow"] = True
                        raise ValueError("readiness-bound")
                    raw.extend(chunk)
        finally:
            state[phase + "_response_bytes"] = len(raw)
            state[phase + "_settled_ns"] = time.monotonic_ns()
        _owned(process, runtime, port, deadline)
        _remaining(deadline)
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError):
            raise ValueError("readiness-json") from None
        if type(value) is not dict:
            raise ValueError("readiness-shape")
        return value

    # Cleanup is outside this timeout: one cancellation closes the scoped HTTP
    # response/client before the synchronous owner receives a refusal.
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False, trust_env=False) as client:
        async with asyncio.timeout(_remaining(deadline)):
            load = await read("chat", {"model": manifest["model"], "messages": [],
                "stream": False, "options": {"num_ctx": startup_profile.context_tokens},
                "keep_alive": "5m"}, "load")
            state["stage"] = "model-load-ack"
            message = load.get("message")
            if (load.get("done") is not True or load.get("done_reason") != "load"
                    or load.get("model") != manifest["model"] or type(message) is not dict
                    or message.get("role") != "assistant" or message.get("content") != ""
                    or message.get("thinking", "") != "" or message.get("tool_calls", []) != []):
                raise ValueError("load-ack")
            state["load_acknowledged"] = True
            snapshot = await read("ps", None, "readback")
            state["stage"] = "model-readback-identity"
            models = snapshot.get("models")
            if type(models) is not list or len(models) != 1 or type(models[0]) is not dict:
                raise ValueError("loaded-model")
            row = models[0]
            if (row.get("digest") != manifest["model_manifest_sha256"]
                    or row.get("name") != manifest["model"]
                    or row.get("model") != manifest["model"]):
                raise ValueError("loaded-model")
            state["stage"] = "model-readback-context"
            if (type(row.get("context_length")) is not int
                    or row["context_length"] != startup_profile.context_tokens):
                raise ValueError("loaded-context")
            state["configured_context_tokens"] = row["context_length"]


def start_model(manifest: Mapping[str, Any], processes: Any, containment: Any,
                evidence_dir: Path, deadline: float, *,
                verified: VerifiedArtifacts | None = None,
                startup_profile: StartupProfile = StartupProfile.LOOKUP_64K) -> dict[str, Any]:
    """Keep the ready owned server for the caller; never pull or generate task text."""
    selected = dict(manifest)
    profile_valid = type(startup_profile) is StartupProfile
    port = selected["model_port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("model-port")
    deadline = min(deadline, time.monotonic() + 60)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / "model-metadata.private.json"
    state: dict[str, Any] = {"version": 1, "category": "setup", "metadata_requests": [],
        "inference_request_count": 0,
        "context_default_tokens": startup_profile.context_tokens if profile_valid else None,
        "startup_profile": startup_profile.profile_id if profile_valid else "invalid",
        "allocation_observed": False, "port": port, "stage": "ownership",
        "load_request_count": 0, "load_acknowledged": False, "model_ready": False,
        "configured_context_basis": "configured-per-sequence"}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream)
    process = None
    try:
        if not profile_valid:
            raise ValueError("startup-profile")
        _remaining(deadline)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ValueError("model-readiness-loop")
        if MODEL_PROCESS_NAME in processes.roots:
            raise RuntimeError("model-already-attempted")
        if os.getpid() not in containment.members():
            raise RuntimeError("owner-containment")
        if _listeners(port, deadline):
            raise RuntimeError("port-occupied")
        state["stage"] = "artifacts"
        if verified is None:
            verified = verify_artifacts(selected, deadline)
        if (type(verified) is not VerifiedArtifacts
                or verified.selection != tuple(selected[key] for key in _SELECTION)):
            raise ValueError("artifact-proof")
        artifact = dict(verified.facts)
        state["artifact"] = artifact
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("OLLAMA_", "LLAMA_ARG_"))}
        env.update({"OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(selected["model_store"]),
            "OLLAMA_CONTEXT_LENGTH": str(startup_profile.context_tokens),
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NOPRUNE": "1", "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_DEBUG_LOG_REQUESTS": "0"})
        if startup_profile.go_template is not None:
            env["OLLAMA_GO_TEMPLATE"] = startup_profile.go_template
        if _listeners(port, deadline):
            raise RuntimeError("port-occupied")
        _remaining(deadline)
        state["stage"] = "start"
        process = processes.spawn(MODEL_PROCESS_NAME,
            [str(Path(selected["ollama_runtime"]) / "ollama.exe"), "serve"],
            env=env, cwd=evidence_dir, log=Path(os.devnull))
        while not _listeners(port, deadline):
            if process.poll() is not None:
                raise RuntimeError("model-exited")
            _remaining(deadline)
            time.sleep(min(0.1, _remaining(deadline)))
        def read(kind: str) -> dict[str, Any]:
            state["stage"] = "metadata-" + kind
            state["metadata_requests"].append(kind)
            return _read(kind, selected, process, deadline)

        if read("version") != {"version": SUPPORTED_VERSION}:
            raise ValueError("metadata-version")
        rows = [row for row in read("tags")["models"]
                if row.get("name") == selected["model"] or row.get("model") == selected["model"]]
        if len(rows) != 1 or rows[0].get("digest") != selected["model_manifest_sha256"]:
            raise ValueError("metadata-model")
        show = read("show")
        template = show.get("template")
        if (type(template) is not str or len(template.encode()) != artifact["template_bytes"]
                or hashlib.sha256(template.encode()).hexdigest() != artifact["template_sha256"]):
            raise ValueError("metadata-template")
        if (show.get("details", {}).get("format") != "gguf"
                or show.get("details", {}).get("family") != artifact["model_family"]
                or type(show.get("capabilities")) is not list
                or not {"completion", "tools"}.issubset(show["capabilities"])
                or any(show.get(field, "") != "" for field in ("remote_host", "remote_model"))):
            raise ValueError("metadata-capabilities")
        asyncio.run(_load_readiness(selected, process, deadline, state, startup_profile))
        _remaining(deadline)
        state.update(category="metadata_ready", stage="complete", runtime_version=SUPPORTED_VERSION,
                     model_ready=True)
        return state
    except BaseException as error:
        state["failure_category"] = (error.args[0] if type(error) in {
            ValueError, RuntimeError, TimeoutError} and len(error.args) == 1
            and type(error.args[0]) is str and error.args[0] in _REFUSALS else "other")
        state.update(category="refused", failure_kind=type(error).__name__
                     if type(error).__name__ in {"ValueError", "RuntimeError", "TimeoutError",
                                                "OSError", "HTTPError"} else "other")
        if process is not None:
            try:
                processes.kill(MODEL_PROCESS_NAME)
            except BaseException:
                state["stop_refused"] = True
        raise
    finally:
        state["metadata_request_count"] = len(state["metadata_requests"])
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        public = {key: state[key] for key in (
            "category", "stage", "inference_request_count", "metadata_request_count",
            "runtime_version", "context_default_tokens", "allocation_observed",
            "load_request_count", "load_acknowledged", "model_ready", "configured_context_tokens",
            "failure_kind", "failure_category") if key in state}
        print("[lookup-model] " + json.dumps(public, sort_keys=True), flush=True)
