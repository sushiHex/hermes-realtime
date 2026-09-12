"""Owned-tool source capture requires a live owner; ordinary pins keep their policy."""

import hashlib
import os
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows source/tool ownership")

source = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))["source"]


@pytest.mark.parametrize("kind", ["mapping", "forged", "closed"])
def test_missing_live_tool_owner_refuses_before_git_dispatch(tmp_path, monkeypatch, kind):
    from scripts import candidate_source_archive_oracle as source
    from scripts.qualification_tool_environment import (
        ImmutableToolEnvironmentV1,
        owned_tool_environment,
    )
    from scripts.task13_artifact_orchestrator import CandidateIdentityV1

    identity = CandidateIdentityV1("a" * 40, "b" * 40, "c" * 40, "d" * 64)
    if kind == "mapping":
        owner = {"passed": True}
    elif kind == "forged":
        owner = object.__new__(ImmutableToolEnvironmentV1)
    else:
        helpers = run_path(str(Path(__file__).with_name("test_qualification_tool_environment.py")))
        with owned_tool_environment(helpers["distributions"](monkeypatch)) as owner:
            pass

    def forbidden(*args, **kwargs):
        pytest.fail("invalid tool owner dispatched Git")

    monkeypatch.setattr(source, "_NativeGitChildOwner", forbidden)
    with pytest.raises((TypeError, ValueError)):
        source.capture_candidate_source_archive_from_tools(tmp_path, identity, owner)


def test_ordinary_git_pin_still_rejects_a_caller_owned_image(tmp_path):
    from scripts.candidate_source_archive_oracle import GitExecutablePinV1

    image = tmp_path / "git.exe"
    image.write_bytes(b"synthetic image")
    with pytest.raises(ValueError, match="Program Files"):
        GitExecutablePinV1(image, hashlib.sha256(image.read_bytes()).hexdigest(), "synthetic", 1)


def test_legacy_archive_cannot_substitute_for_complete_tool_capture(source, monkeypatch):
    from scripts import qualification_build_inputs as build
    from scripts.candidate_source_archive_oracle import CandidateSourceArchiveError
    from scripts.qualification_tool_environment import owned_tool_environment

    helper = run_path(str(Path(__file__).with_name("test_qualification_tool_environment.py")))
    distributions = helper["distributions"](monkeypatch, python_version="3.11.16")
    wheel_helper = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))
    pins, wheels = wheel_helper["values"]()
    _, archive, identity, _ = source

    def forbidden(**arguments):
        pytest.fail("Legacy source capture reached build package parsing")

    monkeypatch.setattr(build, "inspect_wheelhouse_files", forbidden)
    with (
        owned_tool_environment(distributions) as tools,
        pytest.raises(CandidateSourceArchiveError, match="capture"),
    ):
        build.bind_build_inputs(
            archive,
            identity,
            tools,
            wheels=wheels,
            requirements=pins["pins"](wheels),
            constraints=b"",
        )
