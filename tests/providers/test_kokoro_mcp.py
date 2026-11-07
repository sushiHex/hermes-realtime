"""Owned SDK transport controls, with no model, GPU or speech execution."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
import threading
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def configured(tmp_path, monkeypatch):
    from hermes_realtime.providers import kokoro_mcp as module

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    profile = {"version": 1, "gpu_index": 0, "gpu_uuid": "GPU-synthetic-fixture",
               "reserve_gb": 2, "headroom_mb": 1024, "voice": "bf_isabella", "language": "en-gb"}
    for name in ("python", "venv_python", "worker", "model", "voices"):
        path = tmp_path / name
        path.write_bytes(("fixture-" + name).encode())
        profile[name], profile[name + "_sha256"] = str(path), digest(path)
    # Client construction validates paths but launches only through the injected launcher.
    profile["python"] = profile["venv_python"] = sys.executable
    profile["python_sha256"] = profile["venv_python_sha256"] = digest(Path(sys.executable))
    profile_path = tmp_path / "operator.json"
    profile_path.write_text(json.dumps(profile))
    shared_state = tmp_path / "shared-coordinator-state"
    shared_state.mkdir()
    config = {"version": 1, "server_python": sys.executable,
              "server_python_sha256": digest(Path(sys.executable)),
              "server_venv_python": sys.executable,
              "server_venv_python_sha256": digest(Path(sys.executable)),
              "server_source": str(tmp_path), "server_commit": "a" * 40,
              "server_tree": "b" * 40, "profile": str(profile_path),
              "profile_sha256": digest(profile_path), "server_state_dir": str(shared_state)}
    config_path = tmp_path / "managed.json"
    config_path.write_text(json.dumps(config))
    native_source_verifier = module._verify_source
    native_interpreter_verifier = module._verify_interpreter
    monkeypatch.setattr(module, "_verify_source", lambda config: None)
    monkeypatch.setattr(module, "_verify_interpreter", lambda config: None)
    records = []
    state = {"stopped": False, "load_gate": None, "response": "accepted"}
    state["native_transport"] = module._open_session
    state["native_interpreter_verifier"] = native_interpreter_verifier
    state["native_source_verifier"] = native_source_verifier
    state["config"] = config
    operation = "s" * 43

    def payload(status):
        return {"ok": True, "outcome": "succeeded", "state": status, "reason": None,
                "operation_id": operation, "gpu_ready": False,
                "worker_exit_code": 0 if status == "stopped" else None,
                "worker_exit_category": None}

    class Session:
        async def call_tool(self, name, arguments):
            records.append((name, dict(arguments)))
            if name == "load_kokoro":
                if state["load_gate"]:
                    await asyncio.sleep(state["load_gate"])
                if state["response"] == "unknown":
                    raise OSError("SYNTHETIC_PRIVATE_ERROR")
                if state["response"] == "refused":
                    value = payload("refused")
                    value.update(ok=False, outcome="refused", operation_id=None)
                else:
                    value = payload("launch_accepted")
            elif name == "stop_kokoro":
                state["stopped"] = True
                value = payload("stopped")
            else:
                value = payload("stopped" if state["stopped"] else "launch_accepted")
            if state.get("foreign") and name != "load_kokoro":
                value["operation_id"] = "f" * 43
            if state.get("failed_status") and name != "load_kokoro":
                value.update(ok=False, outcome="unknown", state="stopped", worker_exit_code=0)
            if state.get("failed_stop") and name == "stop_kokoro":
                value.update(ok=False, outcome="unknown")
            if state.get("foreign_stop") and name == "stop_kokoro":
                value["operation_id"] = "f" * 43
            if state.get("bad_exit") and name == "kokoro_status":
                value.update(state="stopped", worker_exit_code=state["bad_exit"])
            if state.get("gpu_ready"):
                value["gpu_ready"] = True
            return SimpleNamespace(isError=False, structuredContent=value)

    @asynccontextmanager
    async def transport(config):
        task = asyncio.current_task()
        records.append(("enter", task))
        try:
            yield Session()
        finally:
            records.append(("exit", asyncio.current_task()))
            assert asyncio.current_task() is task

    monkeypatch.setattr(module, "_open_session", transport)
    factory = module.KokoroMcpWorkerFactory.from_config(config_path)
    yield module, factory, state, records, profile, config_path
    with suppress(module.KokoroMcpError):
        factory.close()


def test_owned_load_stop_and_sdk_context_share_one_owner_task(configured):
    _, factory, _, records, _, _ = configured
    handle = factory.launch(connect_port=12345, token="x" * 43)
    assert handle.poll() is None
    factory.close()
    factory.close()
    calls = [(name, value) for name, value in records if name not in {"enter", "exit"}]
    assert [name for name, _ in calls].count("load_kokoro") == 1
    assert [name for name, _ in calls][-2:] == ["stop_kokoro", "kokoro_status"]
    entered = next(value for name, value in records if name == "enter")
    assert next(value for name, value in records if name == "exit") is entered
    operation_calls = [value for name, value in calls if name != "load_kokoro"]
    assert all(value["operation_id"] == "s" * 43 for value in operation_calls)
    assert len({value["owner"] for _, value in calls}) == 1


def test_unknown_ack_cannot_relaunch_or_claim_successful_cleanup(configured):
    module, factory, state, records, _, _ = configured
    state["response"] = "unknown"
    with pytest.raises(module.KokoroMcpError):
        factory.launch(connect_port=12345, token="x" * 43)
    with pytest.raises(module.KokoroMcpError):
        factory.launch(connect_port=12345, token="x" * 43)
    with pytest.raises(module.KokoroMcpError, match="cleanup_unknown"):
        factory.close()
    assert factory.cleanup_state == "cleanup_unknown"
    assert [name for name, _ in records].count("load_kokoro") == 1


def test_successful_launch_cannot_dispatch_a_second_worker(configured):
    module, factory, _, records, _, _ = configured
    factory.launch(connect_port=12345, token="x" * 43)
    with pytest.raises(module.KokoroMcpError, match="launch_attempted"):
        factory.launch(connect_port=12346, token="y" * 43)
    assert [name for name, _ in records].count("load_kokoro") == 1


@pytest.mark.parametrize("mode", ["failed_stop", "foreign_stop"])
def test_stop_reply_requires_its_own_success_and_identity(configured, mode):
    module, factory, state, _, _, _ = configured
    factory.launch(connect_port=12345, token="x" * 43)
    state[mode] = True
    with pytest.raises(module.KokoroMcpError, match="cleanup_unknown"):
        factory.close()
    assert factory.cleanup_state == "cleanup_unknown"


@pytest.mark.parametrize("code", [True, 2 ** 32, -(2 ** 31) - 1])
def test_exit_metadata_is_exact_integer_and_never_readiness(configured, code):
    module, factory, state, _, _, _ = configured
    handle = factory.launch(connect_port=12345, token="x" * 43)
    state["bad_exit"] = code
    with pytest.raises(module.KokoroMcpError, match="status_unknown"):
        handle.poll()


def test_launcher_result_never_substitutes_for_authenticated_readiness(configured):
    module, factory, state, _, _, _ = configured
    state["gpu_ready"] = True
    with pytest.raises(module.KokoroMcpError):
        factory.launch(connect_port=12345, token="x" * 43)
    assert factory.cleanup_state == "cleanup_unknown"


def test_profile_digest_change_refuses_before_server_open(configured):
    module, factory, _, records, _, _ = configured
    factory._config.profile.write_text(factory._config.profile.read_text() + "\n")
    with pytest.raises(module.KokoroMcpError, match="profile_binding"):
        factory.launch(connect_port=12345, token="x" * 43)
    assert not any(name in {"enter", "load_kokoro"} for name, _ in records)


def test_config_version_is_exact_integer_one(configured):
    module, _, _, _, _, path = configured
    config = json.loads(path.read_text())
    config["version"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(module.KokoroMcpError, match="configuration_invalid"):
        module.KokoroMcpWorkerFactory.from_config(path)


def test_io_thread_sync_calls_refuse_without_deadlock_or_orphan_coroutine(configured, monkeypatch):
    module, factory, _, _, _, _ = configured
    handle = factory.launch(connect_port=12345, token="x" * 43)
    original_wait = factory._wait
    monkeypatch.setattr(factory, "_wait", lambda future, timeout=0.1: original_wait(future, 0.1))
    async def check():
        with pytest.raises(module.KokoroMcpError, match="same_thread"):
            handle.poll()
        with pytest.raises(module.KokoroMcpError, match="same_thread"):
            factory.close()
    asyncio.run_coroutine_threadsafe(check(), factory._loop).result(timeout=1)


def test_foreign_status_cannot_establish_owned_settlement(configured):
    module, factory, state, _, _, _ = configured
    handle = factory.launch(connect_port=12345, token="x" * 43)
    state["foreign"] = True
    with pytest.raises(module.KokoroMcpError):
        handle.poll()
    with pytest.raises(module.KokoroMcpError, match="cleanup_unknown"):
        factory.close()
    assert factory.cleanup_state == "cleanup_unknown"


def test_close_during_launch_settles_late_accepted_handle(configured):
    module, factory, state, records, _, _ = configured
    state["load_gate"] = 0.1
    errors = []
    def launch():
        try:
            factory.launch(connect_port=12345, token="x" * 43)
        except module.KokoroMcpError as error:
            errors.append(error)
    thread = threading.Thread(target=launch)
    thread.start()
    for _ in range(1000):
        if any(name == "load_kokoro" for name, _ in records):
            break
        threading.Event().wait(0.001)
    factory.close()
    thread.join(timeout=2)
    assert not thread.is_alive() and errors
    assert state["stopped"] and factory.cleanup_state == "stopped"


def test_fixed_factory_asset_binding_and_no_force_fallback(configured):
    module, factory, _, _, profile, _ = configured
    with pytest.raises(module.KokoroMcpError):
        factory.create(model_path=Path(profile["model"]), voices_path=Path(profile["voices"]),
                       expected_model_sha256="0" * 64,
                       expected_voices_sha256=profile["voices_sha256"], language="en-gb",
                       max_pcm_bytes=1024, attestation_voice="bf_isabella")
    handle = factory.launch(connect_port=12345, token="x" * 43)
    with pytest.raises(module.KokoroMcpError, match="unsupported_force"):
        handle.kill()


def test_managed_startup_budget_matches_qualified_worker_configuration(configured):
    _, factory, _, _, profile, _ = configured
    client = factory.create(
        model_path=Path(profile["model"]), voices_path=Path(profile["voices"]),
        expected_model_sha256=profile["model_sha256"],
        expected_voices_sha256=profile["voices_sha256"], language="en-gb",
        max_pcm_bytes=1024, attestation_voice="bf_isabella",
    )
    assert client._startup_timeout == 60.0


def test_private_configuration_errors_never_echo_values(tmp_path, capsys):
    from hermes_realtime.providers import kokoro_mcp as module
    path = tmp_path / "config.json"
    path.write_text('{"SYNTHETIC_PRIVATE_TOKEN":"secret"}')
    with pytest.raises(module.KokoroMcpError) as caught:
        module.KokoroMcpWorkerFactory.from_config(path)
    assert "SYNTHETIC_PRIVATE" not in str(caught.value)
    assert "SYNTHETIC_PRIVATE" not in capsys.readouterr().err


def test_trusted_state_directory_survives_redirected_app_home(configured, monkeypatch):
    module, _, _, _, _, path = configured
    state_dir = path.parent / "coordinator-state"
    state_dir.mkdir()
    config = json.loads(path.read_text())
    config["server_state_dir"] = str(state_dir)
    path.write_text(json.dumps(config))
    factory = module.KokoroMcpWorkerFactory.from_config(path)
    monkeypatch.setenv("USERPROFILE", str(path.parent / "isolated-app-home"))
    monkeypatch.setenv("HOME", str(path.parent / "isolated-app-home"))
    try:
        assert module._environment(factory._config)["VRAM_MCP_STATE_DIR"] == str(
            state_dir.resolve())
    finally:
        factory.close()


def test_request_identity_always_matches_coordinator_label_schema(configured, monkeypatch):
    module, _, _, _, _, path = configured
    monkeypatch.setattr(module.secrets, "token_urlsafe", lambda size: "_" * 43)
    factory = module.KokoroMcpWorkerFactory.from_config(path)
    try:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", factory._request_id)
    finally:
        factory.close()


@pytest.mark.parametrize("invalid", ["file", "relative", "symlink", "missing"])
def test_trusted_state_directory_is_existing_absolute_directory_not_symlink(
    configured, monkeypatch, invalid,
):
    module, _, state, _, _, path = configured
    config = json.loads(path.read_text())
    directory = Path(config["server_state_dir"])
    if invalid == "file":
        file = path.parent / "not-a-directory"
        file.write_text("fixture")
        config["server_state_dir"] = str(file)
    elif invalid == "relative":
        monkeypatch.chdir(path.parent)
        config["server_state_dir"] = directory.name
    elif invalid == "symlink":
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda item: item == directory or original(item))
    else:
        config["server_state_dir"] = str(path.parent / "missing-directory")
    path.write_text(json.dumps(config))
    with pytest.raises(module.KokoroMcpError, match="configuration_invalid"):
        module.KokoroMcpWorkerFactory.from_config(path)


def test_broken_refusal_stream_preserves_error_and_detaches(monkeypatch):
    from hermes_realtime.providers import kokoro_mcp as module
    class Broken:
        def write(self, value):
            raise OSError("SYNTHETIC_PRIVATE_STREAM")
        def flush(self):
            raise OSError("SYNTHETIC_PRIVATE_STREAM")
    monkeypatch.setattr(module.sys, "stderr", Broken())
    error = module.KokoroMcpError("closed")
    assert str(error) == "closed" and module.sys.stderr is None


def test_failed_status_or_stop_cannot_establish_settlement(configured):
    module, factory, state, _, _, _ = configured
    handle = factory.launch(connect_port=12345, token="x" * 43)
    state["failed_status"] = True
    with pytest.raises(module.KokoroMcpError):
        handle.poll()
    with pytest.raises(module.KokoroMcpError, match="cleanup_unknown"):
        factory.close()
    assert factory.cleanup_state == "cleanup_unknown"


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows base/venv process custody")
def test_native_model_free_sdk_scope_settles_and_joins_thread(configured, monkeypatch):
    module, _, state, _, _, config_path = configured
    source = config_path.parent
    package = source / "vram_mcp"
    package.mkdir()
    (package / "__init__.py").write_text("")
    shared_state = Path(state["config"]["server_state_dir"])
    control_file = shared_state / "claims.json"
    control_bytes = json.dumps({"claims": [{"claim_id": "synthetic-control"}], "operations": []})
    control_file.write_text(control_bytes)
    isolated_home = source / "isolated-app-home"
    isolated_home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(isolated_home))
    monkeypatch.setenv("HOME", str(isolated_home))
    (package / "server.py").write_text('''
import json
import os
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict
from mcp.server.fastmcp import FastMCP
server = FastMCP("model-free-kokoro")
stopped = False
class Reply(TypedDict):
    ok: bool
    outcome: str
    state: str
    reason: str | None
    operation_id: str | None
    gpu_ready: bool
    worker_exit_code: int | None
    worker_exit_category: str | None
def reply(state: str) -> Reply:
    return dict(ok=True,outcome="succeeded",state=state,reason=None,
                operation_id="s"*43,gpu_ready=False,
                worker_exit_code=0 if state=="stopped" else None,worker_exit_category=None)
@server.tool()
def load_kokoro(owner: Any=None, request_id: Any=None, connect_port: Any=None,
                auth_token: Any=None) -> Reply:
    shared = json.loads((Path(os.environ["VRAM_MCP_STATE_DIR"]) / "claims.json").read_text())
    assert any(c["claim_id"] == "synthetic-control" for c in shared["claims"])
    return reply("launch_accepted")
@server.tool()
def kokoro_status(owner: Any=None, operation_id: Any=None) -> Reply:
    return reply("stopped" if stopped else "launch_accepted")
@server.tool()
def stop_kokoro(owner: Any=None, operation_id: Any=None) -> Reply:
    global stopped
    stopped = True
    return reply("stopped")
def main():
    server.run()
''')
    config = state["config"]
    config["server_python"] = sys._base_executable
    config["server_python_sha256"] = hashlib.sha256(
        Path(sys._base_executable).read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, "_open_session", state["native_transport"])
    if sys.platform == "win32":
        monkeypatch.setattr(module, "_verify_interpreter", state["native_interpreter_verifier"])
    factory = module.KokoroMcpWorkerFactory.from_config(config_path)
    try:
        handle = factory.launch(connect_port=12345, token="x" * 43)
        assert handle.poll() is None
    finally:
        factory.close()
    assert factory.cleanup_state == "stopped"
    assert factory._transport_closed and not factory._thread.is_alive()
    assert control_file.read_text() == control_bytes


@pytest.mark.parametrize("shadow", ["json.py", "vram_mcp/shadow.py", "mcp/__init__.py",
                                    "head", "tree", "origin", "dirty"])
def test_verified_source_rejects_ignored_importable_shadows(configured, tmp_path, shadow):
    module, factory, state, _, _, _ = configured
    source = tmp_path / "source"
    source.mkdir()
    package = source / "vram_mcp"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text("# tracked model-free source\n")
    (source / ".gitignore").write_text("json.py\nvram_mcp/shadow.py\nmcp/\n")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    git("init", "-q")
    git("config", "user.name", "Model-free fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("remote", "add", "origin", "https://github.com/sushiHex/vram-mcp.git")
    git("add", ".")
    git("commit", "-qm", "fixture")
    config = replace(factory._config, server_source=source, server_commit=git("rev-parse", "HEAD"),
                     server_tree=git("rev-parse", "HEAD^{tree}"))
    verify = state["native_source_verifier"]
    verify(config)
    if shadow == "head":
        config = replace(config, server_commit="f" * 40)
    elif shadow == "tree":
        config = replace(config, server_tree="f" * 40)
    elif shadow == "origin":
        git("remote", "set-url", "origin", "https://example.invalid/other.git")
    elif shadow == "dirty":
        (package / "server.py").write_text("# modified tracked code\n")
    else:
        malicious = source / shadow
        malicious.parent.mkdir(exist_ok=True)
        malicious.write_text("raise RuntimeError('SYNTHETIC_PRIVATE_SHADOW')\n")
    with pytest.raises(module.KokoroMcpError, match="source_binding"):
        verify(config)


def test_selected_gpu_index_is_derived_from_bound_profile(configured):
    module, factory, _, _, _, _ = configured
    profile = json.loads(factory._config.profile.read_text())
    profile["gpu_index"] = 3
    factory._config.profile.write_text(json.dumps(profile))
    config = replace(factory._config, profile_sha256=hashlib.sha256(
        factory._config.profile.read_bytes()).hexdigest())
    assert module._environment(config)["VRAM_MCP_GPU_INDEX"] == "3"


def test_close_before_owner_entry_cannot_open_scope_after_shutdown(configured, monkeypatch):
    module, factory, _, records, _, _ = configured
    entered = threading.Event()
    allow_entry = threading.Event()
    original = factory._run
    def delayed_owner():
        entered.set()
        allow_entry.wait(1)
        original()
    monkeypatch.setattr(factory, "_run", delayed_owner)
    monkeypatch.setattr(module, "_CLOSE_SECONDS", 0.3)
    errors = []
    def launch():
        try:
            factory.launch(connect_port=12345, token="x" * 43)
        except module.KokoroMcpError as error:
            errors.append(error)
    starter = threading.Thread(target=launch)
    starter.start()
    assert entered.wait(1)
    with suppress(module.KokoroMcpError):
        factory.close()
    allow_entry.set()
    starter.join(2)
    factory._thread.join(1)
    assert not starter.is_alive() and not factory._thread.is_alive()
    assert not any(name in {"enter", "load_kokoro"} for name, _ in records)


@pytest.mark.parametrize("bad", ["digest", "image"])
def test_server_interpreter_binds_actual_image_and_exact_binary_digest(
    configured, monkeypatch, bad,
):
    module, factory, state, _, _, _ = configured
    base = Path(sys._base_executable).resolve(strict=True)
    config = replace(factory._config, server_python=base,
                     server_python_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
    values = [str(base), str(base), str(config.server_venv_python)]
    if bad == "digest":
        config = replace(config, server_python_sha256="0" * 64)
    else:
        values[0] = str(base.parent / "unverified.exe")
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(values).encode()))
    with pytest.raises(module.KokoroMcpError, match="interpreter_binding"):
        state["native_interpreter_verifier"](config)
