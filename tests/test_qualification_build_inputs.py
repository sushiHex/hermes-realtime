"""Build resources must precede, and cannot borrow authority from, final outputs."""

import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import pytest

_candidate_helpers = run_path(
    str(Path(__file__).with_name("test_qualification_candidate_files.py"))
)
source = _candidate_helpers["source"]
bound_graph = _candidate_helpers["bound_graph"]
dependency_graph = _candidate_helpers["dependency_graph"]

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained build inputs")


@pytest.fixture
def tools(monkeypatch):
    if os.name != "nt":
        pytest.skip("Windows immutable tool namespace")
    from scripts import candidate_source_archive_oracle as archives
    from scripts.qualification_tool_environment import (
        owned_tool_environment,
        tool_environment_metadata,
    )

    helpers = run_path(str(Path(__file__).with_name("test_qualification_tool_environment.py")))
    selected = helpers["distributions"](monkeypatch, python_version="3.11.16")
    with owned_tool_environment(selected) as owner:
        # These byte-binding fixtures already substitute synthetic tool releases.
        # Supply capture metadata at the same test seam; this is not native Git
        # execution proof. The dedicated source tests reject legacy capture, and
        # native recipe probes use the genuine complete-tool entry point.
        captured = tool_environment_metadata(owner)
        monkeypatch.setattr(
            archives,
            "_archive_tool_capture_for_consumer",
            lambda archive, identity: captured,
        )
        yield owner


def values():
    helpers = run_path(str(Path(__file__).with_name("test_qualification_wheelhouse.py")))
    wheels = dict([helpers["wheel"]("hatchling", "1.27.0")])
    return helpers, wheels


def test_build_bytes_bind_without_any_final_output_and_expire(source, tools):
    from scripts.qualification_build_inputs import bind_build_inputs, build_input_metadata

    _, archive, identity, _ = source
    helpers, wheels = values()
    receipt = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=helpers["pins"](wheels),
        constraints=b"",
    )
    observation = build_input_metadata(receipt)
    assert observation.source_commit == identity.candidate_head_oid
    assert observation.python_version == "3.11.16"
    assert observation.distributions[0].name == "hatchling"
    expected = hashlib.sha256(next(iter(wheels.values()))).hexdigest()
    assert observation.distributions[0].sha256 == expected
    wheels.clear()
    assert build_input_metadata(receipt) == observation


@pytest.mark.parametrize(
    "fault",
    ["foreign", "unused", "missing", "indirect", "identity", "archive", "tools", "source_tools"],
)
def test_unadmitted_build_resources_refuse_before_installation(source, tools, monkeypatch, fault):
    from scripts import qualification_build_inputs as build
    from scripts.candidate_source_archive_oracle import CandidateSourceArchiveError

    _, archive, identity, _ = source
    helpers, wheels = values()
    if fault == "foreign":
        name, raw = helpers["wheel"]("hatchling", "1.27.0", python=">=3.11.0")
        wheels[name] = raw
        original = build.inspect_wheelhouse_files

        def admitted_only(**arguments):
            assert raw not in arguments["wheels"].values(), "unadmitted bytes reached ZIP parsing"
            return original(**arguments)

        monkeypatch.setattr(build, "inspect_wheelhouse_files", admitted_only)
    elif fault == "unused":
        wheels.update([helpers["wheel"]("aiohttp", "3.11.0")])
    elif fault == "missing":
        wheels = dict([helpers["wheel"]("aiohttp", "3.11.0")])
    elif fault == "identity":
        identity = replace(identity, candidate_head_oid="f" * 40)
    elif fault == "archive":
        archive = {"passed": True}
    elif fault == "tools":
        tools = {"passed": True}
    elif fault == "source_tools":
        from scripts import candidate_source_archive_oracle as archives
        from scripts.qualification_tool_environment import tool_environment_metadata

        actual = tool_environment_metadata(tools)
        mismatched = (replace(actual[0], distribution_sha256="f" * 64), *actual[1:])
        monkeypatch.setattr(
            archives, "_archive_tool_capture_for_consumer", lambda archive, identity: mismatched
        )
    requirements = b"-r another.txt\n" if fault == "indirect" else helpers["pins"](wheels)
    with pytest.raises((TypeError, ValueError, CandidateSourceArchiveError)):
        build.bind_build_inputs(
            archive, identity, tools, wheels=wheels, requirements=requirements, constraints=b""
        )


def test_build_input_capability_expires_after_tool_cleanup(source, monkeypatch):
    from scripts.qualification_build_inputs import (
        BoundBuildInputsV1,
        bind_build_inputs,
        build_input_metadata,
    )

    with pytest.raises(TypeError):
        BoundBuildInputsV1()
    with pytest.raises(ValueError, match="unregistered"):
        build_input_metadata(object.__new__(BoundBuildInputsV1))
    fixture = tools.__wrapped__(monkeypatch)
    owner = next(fixture)
    _, archive, identity, _ = source
    helpers, wheels = values()
    receipt = bind_build_inputs(
        archive,
        identity,
        owner,
        wheels=wheels,
        requirements=helpers["pins"](wheels),
        constraints=b"",
    )
    with pytest.raises(StopIteration):
        next(fixture)
    with pytest.raises(ValueError, match="closed"):
        build_input_metadata(receipt)


@pytest.mark.parametrize("fault", ["backend", "backend_path"])
def test_source_selected_backend_cannot_change_or_import_from_source(
    tmp_path_factory,
    tools,
    fault,
):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.qualification_build_inputs import bind_build_inputs

    repository, _, original, _ = source.__wrapped__(tmp_path_factory)
    path = repository / "pyproject.toml"
    raw = path.read_bytes()
    raw = raw.replace(
        b'build-backend = "hatchling.build"',
        b'build-backend = "unbound.build"'
        if fault == "backend"
        else b'build-backend = "hatchling.build"\nbackend-path = ["src"]',
    )
    path.write_bytes(raw)
    helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    helpers["_git"]("add", "pyproject.toml", cwd=repository)
    helpers["_git"]("commit", "-qm", "synthetic altered backend", cwd=repository)
    identity = helpers["_identity"](repository, original.canonical_baseline_oid)
    archive = capture_candidate_source_archive(repository, identity, helpers["_pin"]())
    wheel_helpers, wheels = values()
    with pytest.raises(ValueError, match="build backend"):
        bind_build_inputs(
            archive,
            identity,
            tools,
            wheels=wheels,
            requirements=wheel_helpers["pins"](wheels),
            constraints=b"",
        )


def _prepare_final_build_inputs(dependency_graph, tools):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_build_inputs import bind_build_inputs
    from scripts.qualification_tool_environment import _tool_image_for_consumer

    root, _, document, archive, identity = dependency_graph

    def write(reference, raw):
        (root / reference["relativePath"]).write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    for tool in document["toolIdentities"]:
        if tool["role"] == "hatchling":
            continue
        image, _, version = _tool_image_for_consumer(tools, tool["role"])
        write(tool["artifact"], image.read_bytes())
        tool["version"] = version
    document["expected"]["pythonFullVersion"] = "3.11.16"
    build = None
    build_reference = None
    for reference in document["files"]:
        if reference["role"] in core._WHEELHOUSE_ROLES.values():
            value = json.loads((root / reference["relativePath"]).read_bytes())
            value["pythonVersion"] = "3.11.16"
            write(reference, core.canonical_json_bytes(value))
            if reference["role"] == "build_wheelhouse_manifest":
                build, build_reference = value, reference
    assert build is not None and build_reference is not None
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels={
            item["basename"]: (root / item["relativePath"]).read_bytes() for item in build["wheels"]
        },
        requirements=(root / build["requirements"]["relativePath"]).read_bytes(),
        constraints=(root / build["constraints"]["relativePath"]).read_bytes(),
    )
    return inputs, build, build_reference, write


@pytest.mark.parametrize(
    "fault", [None, "git", "uv", "build_python", "requirements", "constraints", "source"]
)
def test_final_input_roles_must_match_the_earlier_build_resources(dependency_graph, tools, fault):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_build_inputs import bind_build_input_files, build_file_metadata

    root, _, document, _, identity = dependency_graph
    inputs, build, build_reference, write = _prepare_final_build_inputs(dependency_graph, tools)
    if fault in {"git", "uv", "build_python"}:
        tool = next(item for item in document["toolIdentities"] if item["role"] == fault)
        write(tool["artifact"], b"synthetic replacement image")
    elif fault in {"requirements", "constraints"}:
        reference = build[fault]
        write(reference, (root / reference["relativePath"]).read_bytes() + b"# changed\n")
        write(build_reference, core.canonical_json_bytes(build))
    elif fault == "source":
        document["candidate"]["tree"] = "f" * 40
    with _candidate_helpers["freeze"](dependency_graph) as files:
        if fault is not None:
            with pytest.raises(ValueError, match="build"):
                bind_build_input_files(inputs, files)
        else:
            receipt = bind_build_input_files(inputs, files)
            metadata = build_file_metadata(receipt)
            assert (
                metadata.qualification_input_sha256
                == hashlib.sha256(core.canonical_json_bytes(document)).hexdigest()
            )
            assert metadata.source_commit == identity.candidate_head_oid
    if fault is None:
        with pytest.raises(ValueError, match="closed"):
            build_file_metadata(receipt)


@pytest.mark.parametrize("bind_before_cleanup", [False, True])
def test_final_build_facts_survive_tool_cleanup_but_not_final_seal_release(
    dependency_graph, monkeypatch, bind_before_cleanup
):
    from scripts.qualification_build_inputs import (
        bind_build_input_files,
        build_file_metadata,
        build_input_metadata,
    )
    from scripts.qualification_tool_environment import _tool_image_for_consumer

    # Final inputs have an independent lifetime from disposable execution trees.
    with ExitStack() as final_owner:
        with contextmanager(tools.__wrapped__)(monkeypatch) as owner:
            inputs, _, _, _ = _prepare_final_build_inputs(dependency_graph, owner)
            image = _tool_image_for_consumer(owner, "git")[0]
            files = final_owner.enter_context(_candidate_helpers["freeze"](dependency_graph))
            if bind_before_cleanup:
                receipt = bind_build_input_files(inputs, files)
        assert not image.exists()
        with pytest.raises(ValueError, match="closed"):
            build_input_metadata(inputs)
        if not bind_before_cleanup:
            with pytest.raises(ValueError, match="closed"):
                bind_build_input_files(inputs, files)
            return
        assert build_file_metadata(receipt).source_commit == dependency_graph[4].candidate_head_oid
    with pytest.raises(ValueError, match="closed"):
        build_file_metadata(receipt)
