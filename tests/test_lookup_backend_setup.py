"""Consequential selection controls for the explicit disposable setup command."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.setup_lookup_backend import validate_manifest


def manifest(tmp_path):
    return dict(
        version=1,
        candidate="a" * 40,
        tree="b" * 40,
        hermes_source=str(tmp_path / "hermes"),
        hermes_python=str(tmp_path / "hermes/venv/Scripts/python.exe"),
        runtime_target=str(tmp_path / "target"),
        wheel=str(tmp_path / "runtime.whl"),
        wheel_sha256="c" * 64,
        ollama_runtime=str(tmp_path / "ollama"),
        ollama_archive=str(tmp_path / "ollama.zip"),
        ollama_archive_sha256="d" * 64,
        model_store=str(tmp_path / "models"),
        model="local-model:tag",
        model_manifest_sha256="e" * 64,
        model_port=11440,
        api_port=28660,
        companion_port=28785,
        objective="Synthetic request.",
        presentation_key_env="LOOKUP_TEST_LOCAL_KEY",
    )


def test_exact_manifest_accepted(tmp_path):
    assert validate_manifest(manifest(tmp_path))["objective"] == "Synthetic request."


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"version": True},
        {"candidate": "main"},
        {"objective": " "},
        {"objective": "x" * 1025},
        {"api_port": True},
        {"api_port": 11440},
        {"model_port": 11434},
        {"model": ""},
        {"presentation_key_env": "PATH"},
        {"runtime_target": "relative-target"},
    ],
)
def test_invalid_selection_refused(tmp_path, change):
    value = deepcopy(manifest(tmp_path))
    value.update(change)
    with pytest.raises(ValueError):
        validate_manifest(value)


def prepared_stubs(monkeypatch, tmp_path):
    from scripts import lookup_backend_model as model
    from scripts import setup_lookup_backend as setup

    value = manifest(tmp_path)
    monkeypatch.setattr(
        setup, "os", SimpleNamespace(name="nt", environ={value["presentation_key_env"]: "local"})
    )
    monkeypatch.setattr(setup, "verify_runtime", lambda _: {})
    monkeypatch.setattr(model, "verify_artifacts", lambda *_: SimpleNamespace(facts={}))
    return setup, value


def test_private_protection_failure_precedes_secret_writes(monkeypatch, tmp_path):
    setup, value = prepared_stubs(monkeypatch, tmp_path)
    output = tmp_path / "output"

    def refused(_):
        raise OSError("synthetic permission failure")

    monkeypatch.setattr(setup, "protect_directory", refused)
    with pytest.raises(OSError):
        setup.prepare(value, output)
    assert output.is_dir() and list(output.iterdir()) == []


def test_output_cannot_contain_input_tree(monkeypatch, tmp_path):
    setup, value = prepared_stubs(monkeypatch, tmp_path)
    output = tmp_path / "output"
    value["runtime_target"] = str(output / "target")
    with pytest.raises(ValueError, match="output_scope"):
        setup.prepare(value, output)
    assert not output.exists()


def test_existing_profile_is_not_updated(monkeypatch, tmp_path):
    setup, value = prepared_stubs(monkeypatch, tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "existing"
    marker.write_bytes(b"untouched")
    with pytest.raises(ValueError, match="output_scope"):
        setup.prepare(value, output)
    assert marker.read_bytes() == b"untouched"


def test_root_reparse_is_checked_before_resolve(monkeypatch, tmp_path):
    from scripts.setup_lookup_backend import reject_links

    original = Path.lstat

    def observed(path):
        if path == tmp_path:
            return SimpleNamespace(st_file_attributes=0x400)
        return original(path)

    monkeypatch.setattr(Path, "lstat", observed)
    # is_symlink() also uses lstat; keep this control about Windows junctions.
    monkeypatch.setattr(Path, "is_symlink", lambda _: False)
    with pytest.raises(ValueError, match="path_link"):
        reject_links(tmp_path)


def test_cached_foreign_runtime_is_refused(monkeypatch, tmp_path):
    import sys

    from scripts.lookup_backend_task import select_runtime

    foreign = SimpleNamespace(__file__=str(tmp_path / "old/__init__.py"))
    monkeypatch.setitem(sys.modules, "hermes_realtime", foreign)
    with pytest.raises(ValueError, match="foreign_cached_namespace"):
        select_runtime({"packageRoot": str(tmp_path / "new")})
    assert sys.modules["hermes_realtime"] is foreign


def test_public_error_does_not_echo_valueerror(monkeypatch, tmp_path, capsys):
    from scripts import setup_lookup_backend as setup

    path = tmp_path / "input.json"
    path.write_text("{}")

    def refused(_):
        raise ValueError("private_secret_content")

    monkeypatch.setattr(setup, "validate_manifest", refused)
    assert setup.main(["--manifest", str(path), "--output", str(tmp_path / "output")]) == 1
    output = capsys.readouterr().out
    assert "private_secret_content" not in output
    assert "validation_refused" in output


def test_public_error_retains_fixed_model_guard(monkeypatch, tmp_path, capsys):
    import json

    from scripts import lookup_backend_model as model
    from scripts import setup_lookup_backend as setup

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest(tmp_path)))

    def refused(*_):
        raise ValueError("runtime-content")

    assert "runtime-content" in model._REFUSALS
    monkeypatch.setattr(setup, "prepare", refused)
    assert setup.main(["--manifest", str(path), "--output", str(tmp_path / "output")]) == 1
    assert '"failure_category": "runtime-content"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "rows", [[("127.0.0.1", 9), ("127.0.0.1", 99)], [("0.0.0.0", 9), ("127.0.0.1", 9)]]
)
def test_foreign_or_exposed_gateway_cannot_receive_bearer(rows):
    from scripts.lookup_backend_task import require_owned_listeners

    with pytest.raises(ValueError, match="gateway_ownership"):
        require_owned_listeners(rows, {9})


def test_owned_loopback_gateway_can_be_authenticated():
    from scripts.lookup_backend_task import require_owned_listeners

    require_owned_listeners([("127.0.0.1", 9), ("127.0.0.1", 9)], {9})


def test_readiness_redirect_cannot_forward_bearer(monkeypatch):
    from scripts import lookup_backend_task as task

    def opener(*handlers):
        class Redirect:
            def open(self, request, timeout):
                for handler in handlers:
                    if isinstance(handler, task.urllib.request.HTTPRedirectHandler):
                        handler.redirect_request(
                            request, None, 302, "redirect", {}, "https://unrelated.invalid/"
                        )
                raise AssertionError("redirect would forward the bearer")

        return Redirect()

    monkeypatch.setattr(task.urllib.request, "build_opener", opener)
    with pytest.raises(ValueError, match="readiness_redirect"):
        task.request_json("http://127.0.0.1:12345/health/detailed", "synthetic-bearer")


def test_untracked_python_refused_before_wheel_read(monkeypatch, tmp_path):
    from scripts import setup_lookup_backend as setup

    value = manifest(tmp_path)

    def git(path, *args):
        if Path(path) == setup.ROOT:
            if args == ("rev-parse", "HEAD"):
                return value["candidate"]
            if args == ("rev-parse", "HEAD^{tree}"):
                return value["tree"]
            return ""
        if args == ("rev-parse", "HEAD"):
            return setup.PIN
        if args[0] == "ls-files":
            return "unexpected.py"
        return ""

    monkeypatch.setattr(setup, "git", git)
    monkeypatch.setattr(
        setup, "digest", lambda _: pytest.fail("wheel read before source admission")
    )
    with pytest.raises(ValueError, match="hermes_untracked_python"):
        setup.verify_runtime(value)


def test_default_preparation_never_executes_task(monkeypatch, tmp_path, capsys):
    from scripts import lookup_backend_task as task
    from scripts import setup_lookup_backend as setup

    value = manifest(tmp_path)
    path = tmp_path / "manifest.json"
    import json

    path.write_text(json.dumps(value))
    monkeypatch.setattr(setup, "prepare", lambda *_: value)
    monkeypatch.setattr(task, "run_owned", lambda _: pytest.fail("execution without explicit flag"))
    assert setup.main(["--manifest", str(path), "--output", str(tmp_path / "output")]) == 0
    assert '"processes_started": 0' in capsys.readouterr().out


def test_private_directory_grants_inherit_to_gateway_descendants(monkeypatch, tmp_path):
    from scripts import qualification_execution_files as security
    from scripts import setup_lookup_backend as setup
    from scripts import windows_storage_oracle as windows

    applied = []
    monkeypatch.setattr(
        windows, "_api", lambda: (SimpleNamespace(CloseHandle=lambda _: True), None)
    )
    monkeypatch.setattr(security, "_open_security_control", lambda *_args, **_kwargs: (1, None))
    monkeypatch.setattr(security, "_security_sddl", lambda **_: "D:P(A;;FA;;;owner)")
    monkeypatch.setattr(security, "_set_security", lambda handle, sddl: applied.append(sddl))
    setup.protect_directory(tmp_path)
    assert applied == ["D:P(A;OICI;FA;;;owner)"]


@pytest.fixture
def owned_runner_before_gateway(monkeypatch, tmp_path):
    """Exercise the real runner through startup without processes or model calls."""
    import sys

    import scripts
    from scripts import lookup_backend_logs, lookup_backend_model
    from scripts import lookup_backend_task as task

    state = SimpleNamespace(model_calls=[], stops=0)

    class Processes:
        first_cleanup_refusal = False

        def stop_all(self, containment):
            state.stops += 1

        def survivors(self, containment):
            return {}

        def finish(self):
            return {}

    state.processes = Processes()
    owner = SimpleNamespace(Containment=object, Processes=lambda _: state.processes)
    monkeypatch.setattr(scripts, "rehearse_desktop_mvp", owner, raising=False)
    monkeypatch.setattr(lookup_backend_logs, "CappedProcesses", lambda process, _: process)
    monkeypatch.setattr(task, "os", SimpleNamespace(name="nt", _exit=lambda _: None))
    monkeypatch.setattr(
        task, "sys", SimpleNamespace(executable=sys.executable, path=list(sys.path)),
    )
    monkeypatch.setattr(task, "select_runtime", lambda _: None)
    monkeypatch.setattr(task, "listener_rows", lambda _: [])
    monkeypatch.setattr(
        task, "time", SimpleNamespace(monotonic=lambda: 1000, monotonic_ns=lambda: 1),
    )
    monkeypatch.setattr(task, "threading", SimpleNamespace(
        Timer=lambda *_args, **_kwargs: SimpleNamespace(start=lambda: None, cancel=lambda: None),
    ))

    def start_model(*args, **kwargs):
        state.model_calls.append((args, kwargs))
        raise ValueError("test_stopping_before_gateway")

    monkeypatch.setattr(lookup_backend_model, "start_model", start_model)
    state.manifest = {
        "hermes_python": sys.executable, "output": str(tmp_path), "candidate": "a" * 40,
        "api_port": 28660, "companion_port": 28785, "model_port": 11441,
        "_verified_artifacts": object(),
    }
    state.task = task
    return state


@pytest.mark.parametrize("timeout,seconds", [(None, 60), (180, 180)])
def test_owned_runner_forwards_loader_from_its_process_owner(
    owned_runner_before_gateway, timeout, seconds,
):
    state = owned_runner_before_gateway
    supplied = []

    async def loader(**_):
        return True

    def factory(processes, directory):
        supplied.append((processes, directory))
        return loader

    result = state.task.run_owned(
        state.manifest, mcp_loader_factory=factory, startup_timeout_seconds=timeout,
    )
    assert supplied == [(state.processes, Path(state.manifest["output"]) / "mcp-load")]
    args, kwargs = state.model_calls[0]
    assert args[1] is state.processes and args[4] == 1000 + seconds
    assert kwargs["mcp_loader"] is loader
    assert kwargs["startup_timeout_seconds"] is timeout
    assert kwargs["verified"] is state.manifest["_verified_artifacts"]
    assert result["category"] == "refused" and state.stops == 1


@pytest.mark.parametrize("loader", [None, object()])
def test_owned_runner_rejects_missing_factory_loader_before_native_fallback(
    owned_runner_before_gateway, loader, capsys,
):
    state = owned_runner_before_gateway
    state.manifest["objective"] = "SYNTHETIC_PRIVATE_OBJECTIVE"
    state.manifest["bearer"] = "SYNTHETIC_PRIVATE_TOKEN"
    result = state.task.run_owned(state.manifest, mcp_loader_factory=lambda *_: loader)
    assert state.model_calls == []
    assert result["category"] == "refused" and state.stops == 1
    assert result["failure_category"] == "mcp_loader_required"
    output = _owned_runner_diagnostic(capsys, reason="mcp_loader_required")
    assert state.manifest["output"] not in output and "SYNTHETIC_PRIVATE" not in output


def _owned_runner_diagnostic(capsys, *, reason, receipt_saved=True):
    output = capsys.readouterr().out
    prefix = "[lookup-owned] "
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1 and output == lines[0] + "\n"
    assert len(lines[0].encode()) <= 512
    assert json.loads(lines[0][len(prefix):]) == {
        "category": "refused", "failure_category": reason, "cleanup_failures": 0,
        "owned_listeners_gone": True, "owned_processes_settled": True,
        "user_listener_unchanged": True, "receipt_saved": receipt_saved,
    }
    return output


def test_owned_runner_diagnostic_survives_private_receipt_write_failure(
    owned_runner_before_gateway, monkeypatch, capsys,
):
    state = owned_runner_before_gateway
    original = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.name == "owner-receipt.private.json":
            raise OSError("SYNTHETIC_PRIVATE_RECEIPT_ERROR")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(OSError, match="SYNTHETIC_PRIVATE_RECEIPT_ERROR"):
        state.task.run_owned(state.manifest, mcp_loader_factory=lambda *_: None)
    assert state.model_calls == [] and state.stops == 1
    _owned_runner_diagnostic(capsys, reason="mcp_loader_required", receipt_saved=False)


def test_owned_runner_diagnostic_does_not_disclose_other_value_error(
    owned_runner_before_gateway, capsys,
):
    state = owned_runner_before_gateway

    def factory(*_):
        raise ValueError("SYNTHETIC_PRIVATE_ERROR https://private.invalid/token")

    state.task.run_owned(state.manifest, mcp_loader_factory=factory)
    _owned_runner_diagnostic(capsys, reason="validation_refused")


def test_owned_runner_factory_failure_uses_existing_cleanup(owned_runner_before_gateway, capsys):
    state = owned_runner_before_gateway

    def factory(*_):
        raise RuntimeError("test_factory_failure")

    result = state.task.run_owned(state.manifest, mcp_loader_factory=factory)
    assert state.model_calls == []
    assert result["category"] == "refused" and state.stops == 1
    _owned_runner_diagnostic(capsys, reason="unclassified_failure")


def test_owned_runner_default_retains_native_startup_selection(owned_runner_before_gateway):
    state = owned_runner_before_gateway
    state.task.run_owned(state.manifest)
    args, kwargs = state.model_calls[0]
    assert args[4] == 1060
    assert kwargs["mcp_loader"] is None and kwargs["startup_timeout_seconds"] is None
    assert state.stops == 1


def test_owned_runner_capture_mode_refuses_before_model_start(
    owned_runner_before_gateway, capsys,
):
    state = owned_runner_before_gateway
    result = state.task.run_owned(state.manifest, model_response_capture="PRIVATE_INVALID_MODE")
    assert result["category"] == "refused"
    assert state.model_calls == [] and state.stops == 1
    marker = _owned_runner_diagnostic(capsys, reason="validation_refused")
    assert "PRIVATE_INVALID_MODE" not in marker


@pytest.mark.parametrize("mode", ["off", "metadata", "controlled-public"])
def test_owned_runner_passes_explicit_capture_mode_to_gateway(
    owned_runner_before_gateway, monkeypatch, tmp_path, mode,
):
    import scripts
    from tests.test_lookup_model_capture import config

    state = owned_runner_before_gateway
    state.manifest.update(profile=str(tmp_path), hermes_source=str(tmp_path))
    monkeypatch.setattr(state.task.os, "devnull", __import__("os").devnull, raising=False)
    (tmp_path / "config.yaml").write_text(json.dumps(config()), encoding="utf-8")
    monkeypatch.setenv("HERMES_MODEL_RESPONSE_CAPTURE", "controlled-public")
    monkeypatch.setattr(scripts.rehearse_desktop_mvp, "_isolated", lambda _: {}, raising=False)
    monkeypatch.setattr("scripts.lookup_backend_model.start_model", lambda *a, **k: None)
    seen = []

    def spawn(name, command, **kwargs):
        seen.append(command)
        raise RuntimeError("stop before any gateway process")

    monkeypatch.setattr(state.processes, "spawn", spawn, raising=False)
    state.task.run_owned(state.manifest, model_response_capture=mode)
    assert seen[0][-2:] == ["--model-response-capture", mode]


def test_owned_runner_capture_route_refuses_before_model_start(
    owned_runner_before_gateway, tmp_path, capsys,
):
    from tests.test_lookup_model_capture import config

    state = owned_runner_before_gateway
    state.manifest["profile"] = str(tmp_path)
    (tmp_path / "config.yaml").write_text(json.dumps(config(11434)), encoding="utf-8")
    result = state.task.run_owned(state.manifest, model_response_capture="controlled-public")
    assert result["category"] == "refused"
    assert state.model_calls == [] and state.stops == 1
    _owned_runner_diagnostic(capsys, reason="validation_refused")


def test_owned_runner_existing_capture_artifact_refuses_before_model_start(
    owned_runner_before_gateway, tmp_path, capsys,
):
    from tests.test_lookup_model_capture import config

    state = owned_runner_before_gateway
    state.manifest["profile"] = str(tmp_path)
    (tmp_path / "config.yaml").write_text(json.dumps(config()), encoding="utf-8")
    (tmp_path / "model-response-capture.private.json").write_text("PRIVATE_OLD_CAPTURE")
    result = state.task.run_owned(state.manifest, model_response_capture="controlled-public")
    assert result["category"] == "refused" and state.model_calls == [] and state.stops == 1
    marker = _owned_runner_diagnostic(capsys, reason="validation_refused")
    assert "PRIVATE_OLD_CAPTURE" not in marker


def test_owned_runner_frozen_fixture_refuses_before_loader(owned_runner_before_gateway, capsys):
    state = owned_runner_before_gateway
    called = []
    result = state.task.run_owned(
        state.manifest, frozen_search_fixture="PRIVATE_BAD_PATH", frozen_search_sha256="bad",
        mcp_loader_factory=lambda *_: called.append(True),
    )
    assert result["category"] == "refused" and state.model_calls == [] and called == []
    output = capsys.readouterr().out
    assert "PRIVATE_BAD_PATH" not in output
    assert '"category":"frozen_search_config"' in output


def test_owned_runner_forwards_frozen_fixture_pair(
    owned_runner_before_gateway, monkeypatch, tmp_path,
):
    import hashlib

    import scripts
    from tests.test_lookup_frozen_search import snapshot

    state = owned_runner_before_gateway
    path, digest, _ = snapshot(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    state.manifest.update(objective="question", profile=str(tmp_path), hermes_source=str(tmp_path))
    monkeypatch.setattr(state.task.os, "devnull", __import__("os").devnull, raising=False)
    monkeypatch.setattr(scripts.rehearse_desktop_mvp, "_isolated", lambda _: {}, raising=False)
    monkeypatch.setattr("scripts.lookup_backend_model.start_model", lambda *a, **k: None)
    seen = []

    def spawn(_name, command, **_kwargs):
        seen.append(command)
        raise RuntimeError("stop before gateway")

    monkeypatch.setattr(state.processes, "spawn", spawn, raising=False)
    state.task.run_owned(state.manifest, frozen_search_fixture=path, frozen_search_sha256=digest)
    assert seen[0][-4:] == ["--frozen-search-fixture", str(path), "--frozen-search-sha256", digest]


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows-only composition")
def test_fresh_isolated_parent_imports_owner_before_any_process(tmp_path):
    import subprocess
    import sys

    from scripts.setup_lookup_backend import ROOT

    code = f"""
import ctypes, sys
sys.path.insert(0, {str(ROOT)!r})
def audit(event, args):
    if event in {{"subprocess.Popen", "socket.bind", "socket.connect", "ctypes.dlopen"}}:
        raise AssertionError("effect before import-only stopping point")
sys.addaudithook(audit)
from scripts.lookup_backend_task import run_owned
try:
    run_owned({{"packageRoot": {str(ROOT / "src/hermes_realtime")!r},
               "hermes_python": sys.executable, "output": {str(tmp_path / "absent")!r}}})
except FileNotFoundError:
    pass
else:
    raise AssertionError("import-only stopping point not reached")
assert sys.path[0] == {str(ROOT / "src")!r}
assert sys.modules["local_livekit"].__file__ == {str(ROOT / "scripts/local_livekit.py")!r}
assert sys.modules["real_gate_support"].__file__ == {str(ROOT / "scripts/real_gate_support.py")!r}
print("fresh-parent-import-passed")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fresh-parent-import-passed"
