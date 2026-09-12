"""Installed build authority must come from a live source-bound installation."""

from pathlib import Path
from runpy import run_path

import pytest

_inputs = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))
source = _inputs["source"]
tools = _inputs["tools"]


def test_supplied_installed_build_capability_cannot_be_accepted():
    from scripts.qualification_build_environment import (
        InstalledBuildEnvironmentV1,
        installed_build_metadata,
    )

    with pytest.raises(TypeError):
        InstalledBuildEnvironmentV1()
    with pytest.raises(TypeError):
        installed_build_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        installed_build_metadata(object.__new__(InstalledBuildEnvironmentV1))


def test_unbound_build_inputs_refuse_before_installer_or_workspace(monkeypatch):
    from scripts import qualification_build_environment as environment
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    def forbidden(*args, **kwargs):
        pytest.fail("Unbound inputs created an installation workspace")

    monkeypatch.setattr(environment, "_workspace", forbidden)
    with OwnedQualificationWorkV1() as work, pytest.raises(TypeError):
        environment.install_build_environment(work, {"passed": True})


@pytest.mark.parametrize("fault", ["pid", "extra", "type", "fallback", "count", "duplicates"])
def test_import_observation_refuses_forged_or_ambiguous_result(fault):
    import json

    from scripts.qualification_build_environment import _import_observation

    value = {
        "version": 1,
        "kind": "imports",
        "artifact": None,
        "pid": 51,
        "imports": 5,
        "file_origins": 101,
        "source_fallback": False,
    }
    if fault == "pid":
        value["pid"] = 52
    elif fault == "extra":
        value["unbound"] = True
    elif fault == "type":
        value["imports"] = True
    elif fault == "fallback":
        value["source_fallback"] = True
    elif fault == "count":
        value["file_origins"] = 0
    raw = json.dumps(value).encode()
    if fault == "duplicates":
        raw = raw[:-1] + b',"pid":51}'
    with pytest.raises(ValueError):
        _import_observation(raw, 51)


def test_import_observation_retains_only_counts_after_real_identity_comparison():
    from scripts.qualification_build_environment import _import_observation

    assert _import_observation(
        b'{"version":1,"kind":"imports","artifact":null,'
        b'"pid":51,"imports":5,"file_origins":101,"source_fallback":false}',
        51,
    ) == (5, 101)


def test_invalid_installed_bytes_end_dispatch_for_the_work_owner(source, tools, monkeypatch):
    from contextlib import nullcontext

    from scripts.qualification_build_environment import install_build_environment
    from scripts.qualification_build_inputs import bind_build_inputs
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    _, archive, identity, _ = source
    helpers, wheels = _inputs["values"]()
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=helpers["pins"](wheels),
        constraints=b"",
    )
    calls = []

    def invalid_install(self, tools, role, arguments, workspace, **kwargs):
        assert role == "uv"
        calls.append(role)
        target = Path(arguments[arguments.index("--target") + 1])
        target.mkdir()
        (target / "unowned.py").write_bytes(b"# synthetic unowned installer output\n")
        return object()

    monkeypatch.setattr(OwnedQualificationWorkV1, "run_tool", invalid_install)
    with OwnedQualificationWorkV1() as work:
        with pytest.raises(ValueError):
            install_build_environment(work, inputs)
        assert calls == ["uv"]
        with pytest.raises(RuntimeError, match="closing"):
            work.enter(nullcontext())


@pytest.mark.parametrize("change", [None, "missing", "different_source"])
def test_worker_bytes_come_from_the_genuine_candidate_archive(tmp_path_factory, tools, change):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.qualification_build_environment import _build_worker_for_inputs
    from scripts.qualification_build_inputs import bind_build_inputs

    repository, archive, identity, _ = source.__wrapped__(tmp_path_factory)
    path = repository / "scripts/qualification_build_worker.py"
    expected = path.read_bytes()
    if change is not None:
        helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
        if change == "missing":
            path.unlink()
        else:
            expected = b"raise SystemExit(7)\n"
            path.write_bytes(expected)
        helpers["_git"]("add", "scripts/qualification_build_worker.py", cwd=repository)
        helpers["_git"]("commit", "-qm", "synthetic worker source change", cwd=repository)
        identity = helpers["_identity"](repository, identity.canonical_baseline_oid)
        archive = capture_candidate_source_archive(repository, identity, helpers["_pin"]())
    wheels_helper, wheels = _inputs["values"]()
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=wheels_helper["pins"](wheels),
        constraints=b"",
    )
    if change == "missing":
        with pytest.raises(ValueError, match="worker"):
            _build_worker_for_inputs(inputs)
    else:
        assert _build_worker_for_inputs(inputs) == expected
