"""Consequential selection controls for the explicit disposable setup command."""

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
