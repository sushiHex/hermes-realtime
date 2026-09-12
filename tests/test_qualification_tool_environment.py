"""Tool tree tests prove filesystem ownership, never candidate execution."""

import hashlib
import io
import os
import zipfile
from contextlib import contextmanager

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows immutable tool namespace")


def distributions(monkeypatch, *, python_version="synthetic"):
    from scripts import qualification_tool_distributions as tools

    payloads = {}
    policies = {}
    for role in ("git", "uv", "build_python"):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr(role + ".exe", b"synthetic image")
            archive.writestr("Lib/resource file.txt", b"synthetic resource")
        raw = stream.getvalue()
        payloads[role] = raw
        policies[role] = tools._ToolPolicy(
            python_version if role == "build_python" else "synthetic",
            "synthetic.zip",
            role + ".exe",
            "zip",
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        )
    monkeypatch.setattr(tools, "_TOOLS", policies)
    return {role: tools.admit_tool_distribution(role, payloads[role]) for role in policies}


def test_complete_admitted_trees_remain_sealed_until_owner_cleanup(monkeypatch):
    from scripts.qualification_tool_environment import (
        _tool_image_for_consumer,
        owned_tool_environment,
        tool_environment_metadata,
    )

    selected = distributions(monkeypatch)
    with owned_tool_environment(selected) as environment:
        metadata = tool_environment_metadata(environment)
        assert tuple(item.role for item in metadata) == ("build_python", "git", "uv")
        assert all(item.file_count == 2 for item in metadata)
        paths = []
        for role in selected:
            image, digest, version = _tool_image_for_consumer(environment, role)
            paths.append(image)
            assert digest == hashlib.sha256(b"synthetic image").hexdigest()
            assert version == "synthetic"
            assert (image.parent / "Lib/resource file.txt").read_bytes() == b"synthetic resource"
            with pytest.raises(OSError):
                image.write_bytes(b"replacement")
            with pytest.raises(OSError):
                (image.parent / "Lib/shadow.py").write_bytes(b"unbound")
    assert all(not image.exists() for image in paths)
    with pytest.raises(ValueError, match="closed"):
        tool_environment_metadata(environment)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "swapped", "forged"])
def test_incomplete_or_substituted_tool_authority_refuses_before_filesystem_work(
    monkeypatch, fault
):
    from scripts import qualification_tool_environment as owner

    selected = distributions(monkeypatch)
    if fault == "missing":
        del selected["git"]
    elif fault == "duplicate":
        selected["git"] = selected["uv"]
    elif fault == "swapped":
        selected["git"], selected["uv"] = selected["uv"], selected["git"]
    else:
        selected["git"] = {"passed": True}

    def forbidden(*args, **kwargs):
        pytest.fail("invalid tools allocated an execution tree")

    monkeypatch.setattr(owner, "owned_execution_files", forbidden)
    with pytest.raises((TypeError, ValueError)), owner.owned_tool_environment(selected):
        pytest.fail("invalid tool selection accepted")


def test_tool_environment_capability_is_not_caller_constructible():
    from scripts.qualification_tool_environment import (
        ImmutableToolEnvironmentV1,
        tool_environment_metadata,
    )

    with pytest.raises(TypeError):
        ImmutableToolEnvironmentV1()
    with pytest.raises(ValueError):
        tool_environment_metadata(object.__new__(ImmutableToolEnvironmentV1))


def test_completed_environment_is_historical_only_after_successful_snapshot_cleanup(monkeypatch):
    from scripts.qualification_tool_environment import (
        _completed_tool_environment_for_consumer,
        complete_tool_environment,
        completed_tool_environment_metadata,
        owned_tool_environment,
        tool_environment_metadata,
    )

    selected = distributions(monkeypatch)
    with owned_tool_environment(selected) as environment:
        with pytest.raises(ValueError, match="cleanup is incomplete"):
            complete_tool_environment(environment)
        active = tool_environment_metadata(environment)
    completed = complete_tool_environment(environment)
    assert completed_tool_environment_metadata(completed) == active
    assert _completed_tool_environment_for_consumer(completed).tools is environment
    with pytest.raises(ValueError, match="closed"):
        tool_environment_metadata(environment)


def test_failed_snapshot_cleanup_cannot_mint_a_completed_environment(monkeypatch):
    from scripts import qualification_tool_environment as owner

    selected = distributions(monkeypatch)
    original = owner.owned_execution_files

    @contextmanager
    def cleanup_failure(contents):
        with original(contents) as files:
            yield files
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(owner, "owned_execution_files", cleanup_failure)
    with (
        pytest.raises(RuntimeError, match="cleanup failure"),
        owner.owned_tool_environment(selected) as environment,
    ):
        captured = environment
    with pytest.raises(ValueError, match="cleanup is incomplete"):
        owner.complete_tool_environment(captured)


def test_completed_environment_capability_cannot_be_constructed_or_forged():
    from scripts.qualification_tool_environment import (
        CompletedToolEnvironmentV1,
        completed_tool_environment_metadata,
    )

    with pytest.raises(TypeError):
        CompletedToolEnvironmentV1()
    with pytest.raises(TypeError):
        completed_tool_environment_metadata({"completed": True})
    with pytest.raises(ValueError, match="unregistered"):
        completed_tool_environment_metadata(object.__new__(CompletedToolEnvironmentV1))
