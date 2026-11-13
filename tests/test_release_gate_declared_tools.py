"""Issue #279: the local declared-tool PATH mode of the release gate.

Explicit injection in tests stays the primary guarantee. This mode is the
secondary local check that reproduces hosted executable discovery.
"""

from __future__ import annotations

import ctypes
import json
import os
import stat
import subprocess
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _release_gate() -> dict[str, Any]:
    return run_path(str(ROOT / "scripts" / "release_gate.py"))


def _tool(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and "." not in name:
        name += ".exe"
    path = directory / name
    path.write_bytes(b"")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return Path(os.path.normcase(os.path.abspath(path)))


def _python_stage(tmp_path: Path) -> dict[str, Path]:
    return {
        "uv": _tool(tmp_path / "uv-bin", "uv"),
        "git": _tool(tmp_path / "git-cmd", "git"),
        "python": _tool(tmp_path / "python311", "python"),
    }


def _web_stage(tmp_path: Path, gate: dict[str, Any]) -> dict[str, Path]:
    node = tmp_path / "nodejs"
    return {
        "node": _tool(node, "node"),
        gate["NPM"]: _tool(node, gate["NPM"]),
        gate["NPX"]: _tool(node, gate["NPX"]),
    }


def _resolved(gate: dict[str, Any], environment: dict[str, str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name in gate["GUARDED_TOOLS"]:
        found = gate["resolve_on_path"](name, environment["PATH"])
        if found is not None:
            resolved[name] = found
    return resolved


def test_declared_tools_only_pass_and_resolve_to_exactly_the_declared_set(
    tmp_path: Path,
) -> None:
    gate = _release_gate()
    system = tmp_path / "system32"
    system.mkdir()
    base = {"PATH": "ambient", "KEEP": "retained"}
    for tools in (_python_stage(tmp_path), _web_stage(tmp_path, gate)):
        environment = gate["declared_tool_environment"](
            base, tools, system_directories=(str(system),)
        )

        assert environment == base | {
            "PATH": os.pathsep.join(
                (*dict.fromkeys(str(path.parent) for path in tools.values()), str(system))
            )
        }
        assert _resolved(gate, environment) == tools


@pytest.mark.parametrize("undeclared", ["codex", "ollama", "node", "npx", "npm"])
def test_python_stage_refuses_an_undeclared_tool_beside_a_declared_one(
    tmp_path: Path, undeclared: str
) -> None:
    gate = _release_gate()
    tools = _python_stage(tmp_path)
    name = {"npx": gate["NPX"], "npm": gate["NPM"]}.get(undeclared, undeclared)
    _tool(tools["uv"].parent, name)

    with pytest.raises(RuntimeError, match=f"exposes undeclared tool: {name}"):
        gate["declared_tool_environment"]({}, tools, system_directories=())


@pytest.mark.parametrize("undeclared", ["codex", "ollama", "uv", "git", "python"])
def test_web_stage_refuses_an_undeclared_tool_beside_npm(tmp_path: Path, undeclared: str) -> None:
    gate = _release_gate()
    tools = _web_stage(tmp_path, gate)
    _tool(tools[gate["NPM"]].parent, undeclared)

    with pytest.raises(RuntimeError, match=f"exposes undeclared tool: {undeclared}"):
        gate["declared_tool_environment"]({}, tools, system_directories=())


def test_trusted_system_directory_is_held_to_the_same_rule(tmp_path: Path) -> None:
    gate = _release_gate()
    system = tmp_path / "system32"
    _tool(system, "ollama")

    with pytest.raises(RuntimeError, match="exposes undeclared tool: ollama"):
        gate["declared_tool_environment"](
            {}, _python_stage(tmp_path), system_directories=(str(system),)
        )


def test_shadowed_declared_tool_is_refused(tmp_path: Path) -> None:
    gate = _release_gate()
    tools = _python_stage(tmp_path)
    # The uv directory precedes git's, so its own git would win resolution.
    _tool(tools["uv"].parent, "git")

    with pytest.raises(RuntimeError, match="does not re-resolve to its exact executable: git"):
        gate["declared_tool_environment"]({}, tools, system_directories=())


def test_unguarded_declared_name_is_refused(tmp_path: Path) -> None:
    gate = _release_gate()
    tools = _python_stage(tmp_path) | {"bash": _tool(tmp_path / "bash-bin", "bash")}

    with pytest.raises(RuntimeError, match="declared tool is not guarded: bash"):
        gate["declared_tool_environment"]({}, tools, system_directories=())


def test_resolution_ignores_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _release_gate()
    _tool(tmp_path / "cwd", "codex")
    monkeypatch.chdir(tmp_path / "cwd")

    assert gate["resolve_on_path"]("codex", str(tmp_path / "empty")) is None


def test_declared_stage_environments_bind_each_stage_to_its_own_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _release_gate()
    python_tools = _python_stage(tmp_path)
    web_tools = _web_stage(tmp_path, gate)
    # Ambient host PATH: every declared tool plus hosted-absent ones in their
    # own directories, as on a developer machine.
    _tool(tmp_path / "codex-bin", "codex")
    _tool(tmp_path / "ollama-bin", "ollama")
    ambient = os.pathsep.join(
        str(directory)
        for directory in (
            tmp_path / "codex-bin",
            *dict.fromkeys(path.parent for path in (*python_tools.values(), *web_tools.values())),
            tmp_path / "ollama-bin",
        )
    )
    system = tmp_path / "system32"
    system.mkdir()
    selections: list[dict[str, str]] = []

    def uv_python(uv: Path, environment: dict[str, str], cwd: Path) -> Path:
        assert uv == python_tools["uv"] and cwd == tmp_path
        selections.append(environment)
        return python_tools["python"]

    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "uv_selected_python", uv_python)
    monkeypatch.setitem(namespace, "trusted_system_directories", lambda: (str(system),))
    python_environment, web_environment = gate["declared_stage_environments"](
        {"PATH": ambient}, tmp_path
    )

    assert _resolved(gate, python_environment) == python_tools
    assert _resolved(gate, web_environment) == web_tools
    # Hosted runners have no user Git config; both stages exclude the developer's.
    for stage_environment in (python_environment, web_environment):
        assert stage_environment["GIT_CONFIG_GLOBAL"] == os.devnull
        assert "GIT_CONFIG_NOSYSTEM" not in stage_environment
    # uv's interpreter choice is re-resolved under the rebuilt environment.
    assert [selection["PATH"] for selection in selections] == [
        ambient,
        python_environment["PATH"],
    ]


def test_declared_stage_environments_refuse_a_changed_uv_python_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _release_gate()
    python_tools = _python_stage(tmp_path)
    web_tools = _web_stage(tmp_path, gate)
    ambient = os.pathsep.join(
        str(path.parent) for path in (*python_tools.values(), *web_tools.values())
    )
    other = _tool(tmp_path / "downloaded", "python")
    answers = iter((python_tools["python"], other))
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "uv_selected_python", lambda *_args: next(answers))
    monkeypatch.setitem(namespace, "trusted_system_directories", lambda: ())

    with pytest.raises(RuntimeError, match="no longer selects the declared Python"):
        gate["declared_stage_environments"]({"PATH": ambient}, tmp_path)


def test_missing_declared_tool_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = _release_gate()
    python_tools = _python_stage(tmp_path)
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "trusted_system_directories", lambda: ())

    with pytest.raises(RuntimeError, match="declared tool is not installed: node"):
        gate["declared_stage_environments"](
            {"PATH": os.pathsep.join(str(path.parent) for path in python_tools.values())},
            tmp_path,
        )


def _observed_gate(
    monkeypatch: pytest.MonkeyPatch, gate: dict[str, Any], *, declared_tool_path: bool
) -> tuple[list[tuple[tuple[str, ...], dict[str, str]]], dict[str, str]]:
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    environment = {"PATH": "ambient", "BASE": "retained"}
    python_environment = {"PATH": "python-stage", "BASE": "retained"}
    web_environment = {"PATH": "web-stage", "BASE": "retained"}

    class ReachedArtifacts(Exception):
        pass

    def observe_run(*command: str, cwd: Path, env: dict[str, str]) -> None:
        calls.append((command, dict(env)))
        if command[:2] == ("uv", "build"):
            raise ReachedArtifacts

    def stages(base: dict[str, str], root: Path) -> tuple[dict[str, str], dict[str, str]]:
        assert base == environment and root == ROOT
        return python_environment, web_environment

    namespace = gate["gate_materialized_candidate"].__globals__
    for name in (
        "reject_ambient_paths",
        "scan_for_secrets",
        "snapshot_packaged_static",
        "validate_disclosure_manifest",
        "check_static_parity",
    ):
        monkeypatch.setitem(namespace, name, lambda *_args: None)
    # The observed `run` writes no JUnit report, so the report check has nothing to read.
    monkeypatch.setitem(
        namespace, "check_default_suite_outcomes", lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(namespace, "clean_environment", lambda: dict(environment))
    monkeypatch.setitem(namespace, "declared_stage_environments", stages)
    monkeypatch.setitem(namespace, "run", observe_run)
    with pytest.raises(ReachedArtifacts):
        gate["gate_materialized_candidate"](
            ROOT, livekit=False, declared_tool_path=declared_tool_path
        )
    return calls, environment


def test_declared_mode_runs_python_and_web_stages_with_their_own_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _release_gate()
    calls, _ = _observed_gate(monkeypatch, gate, declared_tool_path=True)

    paths = {(command[0], env["PATH"]) for command, env in calls}
    assert paths == {("uv", "python-stage"), (gate["NPM"], "web-stage")}
    assert all(env["BASE"] == "retained" for _, env in calls)


@pytest.mark.parametrize("flag", [False, True])
def test_cli_forwards_the_declared_tool_path_mode(
    monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    gate = _release_gate()
    forwarded: list[dict[str, object]] = []
    namespace = gate["main"].__globals__
    monkeypatch.setitem(namespace, "gate", lambda *_args, **arguments: forwarded.append(arguments))
    monkeypatch.setattr(
        "sys.argv",
        ["release_gate.py", "--candidate", str(ROOT), *(["--declared-tool-path"] * flag)],
    )
    gate["main"]()

    assert [arguments["declared_tool_path"] for arguments in forwarded] == [flag]


def test_default_mode_keeps_the_clean_environment_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _release_gate()
    calls, environment = _observed_gate(monkeypatch, gate, declared_tool_path=False)

    assert {env["PATH"] for _, env in calls} == {environment["PATH"]}
    assert {command[0] for command, _ in calls} == {"uv", gate["NPM"]}
    assert all("GIT_CONFIG_GLOBAL" not in env for _, env in calls)


# Refusal evidence (AGENTS.md): one bounded JSON line per guard rejection,
# categories only. Every fixture path carries SENTINEL so a leak is visible.
SENTINEL = "private-sentinel-7f3a"


def _sentinel_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: dict[str, Any]
) -> tuple[dict[str, Path], str]:
    host = tmp_path / SENTINEL
    python_tools = _python_stage(host)
    web_tools = _web_stage(host, gate)
    ambient = os.pathsep.join(
        str(directory)
        for directory in dict.fromkeys(
            path.parent for path in (*python_tools.values(), *web_tools.values())
        )
    )
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "trusted_system_directories", lambda: ())
    monkeypatch.setitem(namespace, "uv_selected_python", lambda *_args: python_tools["python"])
    return python_tools, ambient


def _evidence(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err
    lines = captured.err.splitlines()
    assert all(line.startswith("[declared-tool-path] {") for line in lines)
    return [json.loads(line.removeprefix("[declared-tool-path] ")) for line in lines]


def _refuse(
    gate: dict[str, Any], ambient: str, root: Path, capsys: pytest.CaptureFixture[str]
) -> list[dict[str, object]]:
    with pytest.raises((RuntimeError, OSError)) as raised:
        gate["declared_stage_environments"]({"PATH": ambient}, root)
    assert SENTINEL not in str(raised.value) or type(raised.value) is OSError
    return _evidence(capsys)


def test_success_emits_no_refusal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    gate["declared_stage_environments"]({"PATH": ambient}, tmp_path)

    assert _evidence(capsys) == []


def test_undeclared_tool_beside_uv_emits_one_bounded_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    python_tools, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    _tool(python_tools["uv"].parent, "codex")

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "undeclared_tool", "tool": "codex"}
    ]


def test_reresolve_mismatch_emits_one_bounded_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    python_tools, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    # Ambient git wins from its own directory; the rebuilt PATH puts uv's
    # directory, which also holds a git, first.
    _tool(python_tools["uv"].parent, "git")
    ambient = os.pathsep.join((str(python_tools["git"].parent), ambient))

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "reresolve_mismatch", "tool": "git"}
    ]


def test_web_stage_refusal_names_the_web_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    _tool(tmp_path / SENTINEL / "nodejs", "ollama")

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "web", "refusal": "undeclared_tool", "tool": "ollama"}
    ]


def test_unguarded_declared_refusal_withholds_the_unguarded_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    python_tools, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    _tool(python_tools["uv"].parent, "bash")
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "PYTHON_STAGE_TOOLS", ("uv", "git", "bash"))

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "unguarded_declared", "tool": None}
    ]


def test_not_installed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    python_tools, _ = _sentinel_host(tmp_path, monkeypatch, gate)
    ambient = os.pathsep.join(str(path.parent) for path in python_tools.values())

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "web", "refusal": "not_installed", "tool": "node"}
    ]


def test_python_selection_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "uv_selected_python", gate["uv_selected_python"])
    failed = SimpleNamespace(
        run=lambda *args, **_kwargs: subprocess.CompletedProcess(args, 2, SENTINEL, SENTINEL),
        DEVNULL=subprocess.DEVNULL,
    )
    monkeypatch.setitem(namespace, "subprocess", failed)

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "python_selection", "tool": "python"}
    ]


def test_python_reselection_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    python_tools, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    answers = iter((python_tools["python"], _tool(tmp_path / SENTINEL / "other", "python")))
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "uv_selected_python", lambda *_args: next(answers))

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "python_reselection", "tool": "python"}
    ]


def test_unexpected_error_reports_only_its_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)

    def unavailable(*_args: object) -> Path:
        raise OSError(f"cannot start {SENTINEL}")

    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "uv_selected_python", unavailable)

    assert _refuse(gate, ambient, tmp_path, capsys) == [
        {"stage": "uv", "refusal": "OSError", "tool": None}
    ]


def test_unsupported_platform_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    namespace = gate["declared_stage_environments"].__globals__
    monkeypatch.setitem(namespace, "os", SimpleNamespace(name="posix", devnull=os.devnull))

    assert _refuse(gate, "", tmp_path, capsys) == [
        {"stage": "setup", "refusal": "unsupported_platform", "tool": None}
    ]


@pytest.mark.skipif(os.name != "nt", reason="GetSystemDirectoryW is Windows-only")
def test_system_directory_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _release_gate()
    real = ctypes.create_unicode_buffer
    # A one-character buffer cannot hold the system directory, so the real
    # GetSystemDirectoryW call reports a size at or above the buffer length.
    monkeypatch.setattr(ctypes, "create_unicode_buffer", lambda _size: real(1))

    assert _refuse(gate, "", tmp_path, capsys) == [
        {"stage": "setup", "refusal": "system_directory", "tool": None}
    ]


def test_refusal_categories_are_a_closed_set() -> None:
    gate = _release_gate()

    with pytest.raises(ValueError, match="unknown declared-tool refusal category"):
        gate["DeclaredToolRefusal"](SENTINEL, None, "message")


def _git_for_windows(root: Path, *, launcher: bool) -> Path:
    if launcher:
        _tool(root / "cmd", "git.exe")
    return root


@pytest.mark.skipif(os.name != "nt", reason="Git for Windows layout")
@pytest.mark.parametrize("internal", ["mingw64", "usr"])
def test_shell_internal_git_beside_its_launcher_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    internal: str,
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    git_root = _git_for_windows(tmp_path / SENTINEL / "Git", launcher=True)
    _tool(git_root / internal / "bin", "git.exe")
    ambient = os.pathsep.join((str(git_root / internal / "bin"), ambient))

    with pytest.raises(RuntimeError, match="launch the gate from PowerShell or cmd"):
        gate["declared_stage_environments"]({"PATH": ambient}, tmp_path)
    assert _evidence(capsys) == [{"stage": "uv", "refusal": "shell_internal_git", "tool": "git"}]


@pytest.mark.skipif(os.name != "nt", reason="Git for Windows layout")
@pytest.mark.parametrize(
    ("directory", "launcher"),
    [("cmd", True), ("mingw64/bin", False), ("usr/bin", False)],
    ids=["launcher", "mingw64-without-launcher", "usr-without-launcher"],
)
def test_launcher_git_or_lone_bin_git_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    directory: str,
    launcher: bool,
) -> None:
    gate = _release_gate()
    _, ambient = _sentinel_host(tmp_path, monkeypatch, gate)
    git_root = _git_for_windows(tmp_path / SENTINEL / "Git", launcher=launcher)
    git = _tool(git_root / directory, "git.exe")
    ambient = os.pathsep.join((str(git.parent), ambient))

    python_environment, _ = gate["declared_stage_environments"]({"PATH": ambient}, tmp_path)

    assert gate["resolve_on_path"]("git", python_environment["PATH"]) == git
    assert _evidence(capsys) == []
