"""Native credential preflight observes resolution; it never invokes a token source."""
import json
import subprocess
from pathlib import Path

import pytest

from hermes_realtime.integration import native_credential_probe as probe

EXPECTED = {"version": 1, "accepted": True, "selected_command": True, "route": True,
            "no_io": True, "homes": True, "source": True}


def arguments(tmp_path):
    home = tmp_path / "staged"
    home.mkdir()
    env = home / ".env"
    env.write_text("SYNTHETIC_SETTING=value\n")
    return dict(executable=Path("C:/synthetic/python.exe").resolve(), hermes_source=tmp_path,
                native_site_packages=tmp_path / "packages",
                home=home, env_file=env, provider="custom:synthetic", base_url="https://example.invalid",
                model="synthetic", api_mode="codex_responses", command="synthetic-command",
                environment={"HERMES_HOME": str(home)})


def test_probe_runs_only_fixed_isolated_child_with_exact_expected_binding(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []
    def run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, json.dumps(EXPECTED), "")
    monkeypatch.setattr(probe.subprocess, "run", run)
    probe.verify_native_credential_resolution(**args)
    argv, kw = calls[0]
    assert argv == [str(args["executable"]), "-I", "-B", "-S", "-c", probe.NATIVE_PROBE]
    assert kw["env"] == args["environment"]
    assert kw["timeout"] == 15 and kw["capture_output"] and kw["text"]
    payload = json.loads(kw["input"])
    assert payload == {key: str(value) if isinstance(value, Path) else value
                       for key, value in args.items() if key not in {"executable", "environment"}}


@pytest.mark.parametrize("fault", ["exit", "stderr", "invalid", "oversized", "extra", "duplicate",
                                     "version_bool", "boolean_int", "timeout", "spawn",
                                     "selected_command", "route", "no_io", "homes", "source"])
def test_probe_refuses_false_or_untrusted_child_evidence_without_echo(
    tmp_path, monkeypatch, capsys, fault,
):
    def run(argv, **kw):
        value = dict(EXPECTED)
        code, stderr = 0, ""
        if fault == "timeout":
            raise subprocess.TimeoutExpired(argv, 15, output="SYNTHETIC_SECRET")
        if fault == "spawn":
            raise OSError("SYNTHETIC_SECRET")
        if fault in {"selected_command", "route", "no_io", "homes", "source"}:
            value[fault] = False
        if fault == "extra":
            value["private"] = "SYNTHETIC_SECRET"
        if fault == "version_bool":
            value["version"] = True
        if fault == "boolean_int":
            value["accepted"] = 1
        stdout = json.dumps(value)
        if fault == "exit":
            code = 1
        if fault == "stderr":
            stderr = "SYNTHETIC_SECRET"
        if fault == "invalid":
            stdout = "SYNTHETIC_SECRET"
        if fault == "oversized":
            stdout += " " * 4096
        if fault == "duplicate":
            stdout = stdout.replace('"version": 1', '"version": 1, "version": 1')
        return subprocess.CompletedProcess(argv, code, stdout, stderr)
    monkeypatch.setattr(probe.subprocess, "run", run)
    with pytest.raises(ValueError, match="^native_credential_resolution$"):
        probe.verify_native_credential_resolution(**arguments(tmp_path))
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ('[native-credential-probe] '
                          '{"version":1,"category":"resolution_refused"}\n')
    assert "SYNTHETIC_SECRET" not in output.err


def test_inherited_environment_mode_passes_no_dotenv_path(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args["env_file"] = None
    calls = []
    def run(argv, **kw):
        calls.append(kw)
        return subprocess.CompletedProcess(argv, 0, json.dumps(EXPECTED), "")
    monkeypatch.setattr(probe.subprocess, "run", run)
    probe.verify_native_credential_resolution(**args)
    assert json.loads(calls[0]["input"])["env_file"] is None
    assert calls[0]["env"] == args["environment"]


def child_probe(tmp_path, monkeypatch, fault=None, *, env_file=True):
    """Execute exact embedded child source with isolated fake native modules."""
    import contextlib
    import importlib
    import io
    import os
    import sys
    from types import SimpleNamespace
    source = tmp_path / "native-source"
    home = tmp_path / "profile"
    home.mkdir()
    dotenv = home / ".env"
    dotenv.write_text("SYNTHETIC_ONLY=value\n")
    expected = dict(hermes_source=str(source), native_site_packages=str(tmp_path / "packages"),
                    home=str(home),
                    env_file=str(dotenv) if env_file else None, provider="custom:synthetic",
                    base_url="https://example.invalid", model="synthetic",
                    api_mode="codex_responses", command="synthetic-fixed-command")
    hooks, loaded = [], []
    if fault == "provider":
        expected["provider"] = "codex"
    for key in ("HOME", "HERMES_HOME", "CODEX_HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(key, str(home))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    if fault == "real_home":
        monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path / "unrelated"))

    class Source:
        _command = expected["command"]
        def __call__(self):
            raise AssertionError("preflight must never mint a key")

    def load(path, *, override):
        assert path == dotenv and override is True
        loaded.append("load")
        if fault == "dotenv_home":
            os.environ["HOME"] = str(tmp_path / "unrelated")

    def clear(path):
        assert path == dotenv
        loaded.append("clear")

    def resolve(**kwargs):
        assert kwargs == {"requested": expected["provider"], "target_model": expected["model"]}
        if fault == "swallowed_io":
            # Native pool failure can fall back to the command source.
            with contextlib.suppress(RuntimeError):
                hooks[0]("subprocess.Popen", ("SYNTHETIC_PRIVATE_ARGUMENT",))
        return dict(provider="custom", requested_provider=expected["provider"],
                    api_key=(SimpleNamespace(_command=expected["command"])
                             if fault == "pool" else Source()),
                    base_url="https://wrong.invalid" if fault == "route" else expected["base_url"],
                    model=expected["model"], api_mode=expected["api_mode"])

    modules = {
        "hermes_constants": SimpleNamespace(
            __file__=str(source / "hermes_constants.py"),
            get_hermes_home=lambda: tmp_path if fault == "native_home" else home,
            get_default_hermes_root=lambda: tmp_path if fault == "native_root" else home),
        "hermes_cli.env_loader": SimpleNamespace(
            __file__=str(source / "hermes_cli/env_loader.py"),
            _load_dotenv_with_fallback=load, _clear_known_keys_missing_from_dotenv=clear),
        "agent.command_token_source": SimpleNamespace(
            __file__=str(source / "agent/command_token_source.py"), CommandTokenSource=Source),
        "hermes_cli.runtime_provider": SimpleNamespace(
            __file__=str(source / "hermes_cli/runtime_provider.py"),
            resolve_runtime_provider=resolve),
    }
    if fault == "source":
        modules["hermes_cli.runtime_provider"].__file__ = str(tmp_path / "ambient.py")
    if fault == "command":
        Source._command = "wrong-command"
    def native_import(name):
        if fault == "import_network":
            hooks[0]("socket.connect", ("SYNTHETIC_PRIVATE_ARGUMENT",))
        return modules[name]
    monkeypatch.setattr(importlib, "import_module", native_import)
    monkeypatch.setattr(sys, "addaudithook", hooks.append)
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(expected).encode())),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(SystemExit) as exit_status:
        exec(compile(probe.NATIVE_PROBE, "fixed-native-probe", "exec"), {})
    return json.loads(output.getvalue()), exit_status.value.code, loaded


@pytest.mark.parametrize("fault,failed", [
    (None, None), ("pool", "selected_command"), ("command", "selected_command"),
    ("route", "route"), ("source", "source"), ("dotenv_home", "homes"),
    ("real_home", "homes"), ("swallowed_io", "no_io"), ("import_network", "accepted"),
    ("provider", "accepted"), ("native_root", "homes"), ("native_home", "homes"),
])
def test_embedded_probe_retains_exact_authority_and_refuses_hidden_io(
    tmp_path, monkeypatch, fault, failed,
):
    value, code, loaded = child_probe(tmp_path, monkeypatch, fault)
    assert set(value) == set(EXPECTED)
    assert all(type(item) is bool for key, item in value.items() if key != "version")
    if fault is None:
        assert value == EXPECTED and code == 0 and loaded == ["load", "clear"]
    else:
        assert value[failed] is False and value["accepted"] is False and code == 1
    assert "SYNTHETIC_PRIVATE_ARGUMENT" not in json.dumps(value)


def test_embedded_inherited_mode_does_not_reload_dotenv(tmp_path, monkeypatch):
    value, code, loaded = child_probe(tmp_path, monkeypatch, env_file=False)
    assert value == EXPECTED and code == 0 and loaded == []
