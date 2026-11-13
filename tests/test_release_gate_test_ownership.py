"""The native gate stage and the default suite partition the tests by outcome.

The hermetic job runs the default suite once; the native job runs only the tests that
need the pinned LiveKit server or the ``local`` extra. Both stages collect from the same
verified archive, so the partition is checked on what each stage actually ran: a test the
default suite skips because only the native environment enables it must be one the native
stage selects, and every test the native stage selects must run there.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from runpy import run_path
from typing import Any
from xml.sax.saxutils import quoteattr

import pytest

ROOT = Path(__file__).resolve().parents[1]
Case = tuple[str, str, str | None]
LIVEKIT_REASON = "set HERMES_REALTIME_LIVEKIT_LOCAL=1 to run against local LiveKit"
IMPORT_REASON = "could not import 'requests': No module named 'requests'"
NATIVE_FILE = "tests.integration.test_local_livekit"
NATIVE_NODE = (
    "tests.test_lookup_backend_gateway",
    "test_exa_event_stream_preserves_punctuation_through_real_requests_response",
)


def _gate() -> dict[str, Any]:
    return run_path(str(ROOT / "scripts" / "release_gate.py"))


def _report(path: Path, cases: list[Case]) -> Path:
    body = "".join(
        f"<testcase classname={quoteattr(classname)} name={quoteattr(name)}>"
        + (
            ""
            if reason is None
            else f'<skipped type="pytest.skip" message={quoteattr(reason)}>detail</skipped>'
        )
        + "</testcase>"
        for classname, name, reason in cases
    )
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">{body}'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


def _every_native_selector(gate: dict[str, Any], reason: str | None) -> list[Case]:
    cases: list[Case] = []
    for selector in gate["NATIVE_STAGE_TESTS"]:
        path, _, test = selector.partition("::")
        cases.append((path.removesuffix(".py").replace("/", "."), test or "test_one", reason))
    return cases


def test_each_pytest_stage_is_launched_through_its_outcome_check() -> None:
    source = (ROOT / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    launchers = sorted(
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and any(
            isinstance(argument, ast.Constant) and argument.value == "pytest"
            for argument in node.args
        )
    )

    # The remaining plain `run` is the focused provider subset, whose tests the default
    # suite collects again; the two stages that own tests go through their checks.
    assert launchers == ["run", "run_default_suite", "run_native_stage"]


def test_native_stage_installs_what_its_import_gated_tests_need_on_purpose() -> None:
    # These two tests once ran only because an earlier command left the `local` extra's
    # transitive packages installed. The stage now names their packages itself.
    source = (ROOT / "scripts" / "release_gate.py").read_text(encoding="utf-8")
    stage = source.split("    run_native_stage(\n", maxsplit=1)[1].split("cwd=root", maxsplit=1)[0]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    imported = {
        "requests": "tests/test_lookup_backend_gateway.py",
        "yaml": "tests/test_preview_cutover.py",
    }

    assert stage.count('"--group",\n        "native-stage",\n') == 1
    assert project["dependency-groups"]["native-stage"] == ["pyyaml>=6,<7", "requests>=2.32,<3"]
    assert "native-stage" not in project["tool"]["uv"].get("default-groups", ["dev"])
    selected = [selector for selector in _gate()["NATIVE_STAGE_TESTS"] if "::" in selector]
    assert sorted(selector.partition("::")[0] for selector in selected) == sorted(
        imported.values()
    )
    for module, path in imported.items():
        assert (ROOT / path).read_text(encoding="utf-8").count(f'importorskip("{module}")') == 1


def test_native_stage_selection_names_existing_tests_exactly_once() -> None:
    selectors = _gate()["NATIVE_STAGE_TESTS"]

    assert len(set(selectors)) == len(selectors)
    for selector in selectors:
        path, _, test = selector.partition("::")
        source = (ROOT / path).read_text(encoding="utf-8")
        if test:
            assert source.count(f"\ndef {test}(") + source.count(f"\nasync def {test}(") == 1


def test_native_stage_owns_exactly_the_livekit_gated_test_files() -> None:
    gate = _gate()
    gated = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if path != Path(__file__).resolve()
        and gate["NATIVE_STAGE_SKIP_REASON"] in path.read_text(encoding="utf-8")
    }
    selected_files = {
        selector for selector in gate["NATIVE_STAGE_TESTS"] if "::" not in selector
    }

    assert gate["NATIVE_STAGE_SKIP_REASON"] == LIVEKIT_REASON
    assert gated == selected_files


def test_native_stage_passes_when_every_selected_test_ran(tmp_path: Path) -> None:
    gate = _gate()
    report = _report(tmp_path / "native.xml", _every_native_selector(gate, None))

    gate["check_native_stage_outcomes"](report)


@pytest.mark.parametrize("reason", [LIVEKIT_REASON, IMPORT_REASON, "any other reason"])
def test_native_stage_refuses_a_selected_test_that_was_skipped(
    tmp_path: Path, reason: str
) -> None:
    gate = _gate()
    cases = _every_native_selector(gate, None)
    cases[0] = (cases[0][0], cases[0][1], reason)

    with pytest.raises(RuntimeError, match="native stage test did not run"):
        gate["check_native_stage_outcomes"](_report(tmp_path / "native.xml", cases))


def test_native_stage_refuses_a_selector_that_collected_nothing(tmp_path: Path) -> None:
    gate = _gate()
    cases = _every_native_selector(gate, None)[1:]

    with pytest.raises(RuntimeError, match="native stage selector collected no test"):
        gate["check_native_stage_outcomes"](_report(tmp_path / "native.xml", cases))


def test_native_stage_refuses_a_test_outside_its_selection(tmp_path: Path) -> None:
    gate = _gate()
    cases = [*_every_native_selector(gate, None), ("tests.test_other", "test_one", None)]

    with pytest.raises(RuntimeError, match="native stage ran a test outside its selection"):
        gate["check_native_stage_outcomes"](_report(tmp_path / "native.xml", cases))


def test_default_suite_passes_when_native_tests_skip_and_other_skips_are_unrelated(
    tmp_path: Path,
) -> None:
    gate = _gate()
    cases = [
        *_every_native_selector(gate, LIVEKIT_REASON),
        ("tests.test_other", "test_ran", None),
        ("tests.integration.test_installed_kokoro", "test_smoke", "set SOMETHING=1"),
    ]

    gate["check_default_suite_outcomes"](
        _report(tmp_path / "suite.xml", cases), native_stage_ran=False
    )


@pytest.mark.parametrize("reason", [LIVEKIT_REASON, IMPORT_REASON])
def test_default_suite_refuses_a_native_enabled_skip_that_no_stage_owns(
    tmp_path: Path, reason: str
) -> None:
    gate = _gate()
    cases = [
        *_every_native_selector(gate, LIVEKIT_REASON),
        ("tests.test_new_feature", "test_needs_native_environment", reason),
    ]

    with pytest.raises(RuntimeError, match="skipped test runs in no gate stage"):
        gate["check_default_suite_outcomes"](
            _report(tmp_path / "suite.xml", cases), native_stage_ran=False
        )


def test_default_suite_refuses_a_native_stage_test_that_also_ran_here(tmp_path: Path) -> None:
    gate = _gate()
    cases = _every_native_selector(gate, LIVEKIT_REASON)
    cases[0] = (cases[0][0], cases[0][1], None)
    report = _report(tmp_path / "suite.xml", cases)

    with pytest.raises(RuntimeError, match="native stage test also ran in the default suite"):
        gate["check_default_suite_outcomes"](report, native_stage_ran=False)
    # One invocation that ran the native stage first leaves its extra installed, so the
    # same tests may run again; that overlap is expected there and only there.
    gate["check_default_suite_outcomes"](report, native_stage_ran=True)


def test_default_suite_refuses_a_selector_absent_from_the_collection(tmp_path: Path) -> None:
    gate = _gate()
    cases = _every_native_selector(gate, LIVEKIT_REASON)[:-1]

    with pytest.raises(RuntimeError, match="native stage selector collected no test"):
        gate["check_default_suite_outcomes"](
            _report(tmp_path / "suite.xml", cases), native_stage_ran=False
        )


def test_selector_ownership_is_exact_for_files_and_single_tests() -> None:
    owner = _gate()["native_stage_selector"]

    assert owner(NATIVE_FILE, "test_anything") == "tests/integration/test_local_livekit.py"
    assert owner(NATIVE_FILE + ".TestClass", "test_anything") is not None
    assert owner(NATIVE_FILE + "_extra", "test_anything") is None
    assert owner(*NATIVE_NODE) is not None
    assert owner(NATIVE_NODE[0], "test_another_test_in_that_file") is None
    assert owner(NATIVE_NODE[0], NATIVE_NODE[1] + "[case]") is None


@pytest.mark.parametrize(
    "body",
    [
        '<testcase classname="tests.a" name="t"><failure message="x">d</failure></testcase>',
        '<testcase classname="tests.a" name="t"><error message="x">d</error></testcase>',
        '<testcase classname="tests.a" name="t"><skipped type="pytest.xfail" message="x"/>'
        "</testcase>",
        '<testcase classname="tests.a" name="t"/><testcase classname="tests.a" name="t"/>',
        '<testcase name="t"/>',
        "",
    ],
)
def test_outcome_report_fails_closed_on_anything_but_ran_or_skipped(
    tmp_path: Path, body: str
) -> None:
    report = tmp_path / "report.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")

    with pytest.raises(RuntimeError, match="test outcome report"):
        _gate()["read_test_outcomes"](report)


def test_outcome_report_fails_closed_when_absent_or_malformed(tmp_path: Path) -> None:
    gate = _gate()

    with pytest.raises(RuntimeError, match="test outcome report"):
        gate["read_test_outcomes"](tmp_path / "absent.xml")
    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<testsuites><testsuite>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="test outcome report"):
        gate["read_test_outcomes"](malformed)


def test_real_pytest_report_carries_the_reasons_the_ownership_check_keys_on(
    tmp_path: Path,
) -> None:
    # The check recognises a native-enabled skip by pytest's own reason text. This runs the
    # locked pytest, so a pytest upgrade that rewords an import skip fails here instead of
    # letting that class of skip pass unowned.
    gate = _gate()
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n\n\n"
        "def test_import():\n"
        "    pytest.importorskip('hermes_realtime_absent_module_for_gate_probe')\n\n\n"
        f"@pytest.mark.skipif(True, reason={LIVEKIT_REASON!r})\n"
        "def test_gated():\n"
        "    pass\n\n\n"
        "def test_ran():\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    report = tmp_path / "report.xml"
    completed = subprocess.run(
        (
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--rootdir", str(tmp_path), "-c", str(tmp_path / "pytest.ini"),
            f"--junitxml={report}", str(tmp_path / "test_probe.py"),
        ),
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0

    outcomes = gate["read_test_outcomes"](report)

    assert outcomes[("test_probe", "test_ran")] is None
    assert gate["native_enabled_skip"](outcomes[("test_probe", "test_import")])
    assert gate["native_enabled_skip"](outcomes[("test_probe", "test_gated")])
    assert not gate["native_enabled_skip"]("linux-null-capture gate")


class _Stop(Exception):
    pass


def _observed_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[tuple[str, tuple[str, ...]]]]:
    gate = _gate()["gate_materialized_candidate"]
    namespace = gate.__globals__
    calls: list[tuple[str, tuple[str, ...]]] = []

    def observe(kind: str) -> Any:
        def record(*command: str, **_: object) -> None:
            calls.append((kind, command))
            if kind == "run" and command[:2] == ("npm.cmd" if os.name == "nt" else "npm", "ci"):
                raise _Stop

        return record

    monkeypatch.setitem(namespace, "clean_environment", lambda: {})
    for name in (
        "reject_ambient_paths",
        "scan_for_secrets",
        "snapshot_packaged_static",
        "validate_disclosure_manifest",
    ):
        monkeypatch.setitem(namespace, name, lambda *_args: None)
    monkeypatch.setitem(namespace, "run", observe("run"))
    monkeypatch.setitem(namespace, "run_default_suite", observe("default_suite"))
    monkeypatch.setitem(
        namespace, "check_livekit", lambda *_args, **_kwargs: calls.append(("native", ()))
    )
    return gate, calls


def test_native_only_gate_stops_after_the_native_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    gate, calls = _observed_gate(monkeypatch)

    gate(
        ROOT,
        livekit=True,
        native_only=True,
        livekit_executable=Path("server"),
        livekit_executable_sha256="0" * 64,
        livekit_pid=1,
    )

    kinds = [kind for kind, _ in calls]
    assert kinds.count("native") == 1
    assert kinds[-1] == "native"
    assert "default_suite" not in kinds
    commands = [command for kind, command in calls if kind == "run"]
    # The identity and static stages that precede the native stage are unchanged.
    assert any(command[:3] == ("uv", "lock", "--check") for command in commands)
    assert any(command[:3] == ("uv", "sync", "--frozen") for command in commands)
    assert any("ruff" in command for command in commands)
    assert not any("build" in command for command in commands)


@pytest.mark.parametrize("livekit", [False, True])
def test_gate_without_native_only_still_runs_the_default_suite_once(
    monkeypatch: pytest.MonkeyPatch, livekit: bool
) -> None:
    gate, calls = _observed_gate(monkeypatch)

    with pytest.raises(_Stop):
        gate(
            ROOT,
            livekit=livekit,
            livekit_executable=Path("server") if livekit else None,
            livekit_executable_sha256="0" * 64 if livekit else None,
            livekit_pid=1 if livekit else None,
        )

    kinds = [kind for kind, _ in calls]
    assert kinds.count("default_suite") == 1
    assert kinds.count("native") == (1 if livekit else 0)
    if livekit:
        assert kinds.index("native") < kinds.index("default_suite")


def test_native_only_gate_refuses_to_run_without_the_native_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, calls = _observed_gate(monkeypatch)

    with pytest.raises(RuntimeError, match="native-only gate requires the native LiveKit stage"):
        gate(ROOT, livekit=False, native_only=True)
    assert calls == []


@pytest.mark.parametrize("script", ["release_gate.py", "candidate_e2e_fast_track.py"])
def test_native_only_option_requires_the_native_stage_on_both_entry_points(script: str) -> None:
    completed = subprocess.run(
        (sys.executable, str(ROOT / "scripts" / script), "--native-only"),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 2
    assert "--native-only requires --require-livekit" in completed.stderr
