"""Build authority requires owned invocations, verified outputs and cleanup."""

import hashlib
import json
import os
from pathlib import Path
from runpy import run_path

import pytest

_helpers = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))
source = _helpers["source"]
tools = _helpers["tools"]
bound_graph = _helpers["bound_graph"]


def test_supplied_build_results_cannot_mint_execution_authority():
    from scripts.qualification_builds import CandidateBuildsV1, candidate_build_metadata

    with pytest.raises(TypeError):
        CandidateBuildsV1()
    with pytest.raises(TypeError):
        candidate_build_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        candidate_build_metadata(object.__new__(CandidateBuildsV1))


def test_supplied_final_output_binding_cannot_mint_authority():
    from scripts.qualification_builds import BoundBuildOutputsV1, build_output_metadata

    with pytest.raises(TypeError):
        BoundBuildOutputsV1()
    with pytest.raises(TypeError):
        build_output_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        build_output_metadata(object.__new__(BoundBuildOutputsV1))


@pytest.mark.parametrize("fault", [None, "candidate", "artifact"])
def test_final_output_binding_retains_final_seals_and_completed_facts(
    bound_graph, monkeypatch, fault
):
    # Mock upstream completion only to isolate final file comparison. Native
    # recipe probes separately exercise actual builds; this is not build proof.
    from types import SimpleNamespace as Row

    from scripts import qualification_builds as builds

    root, _, document, _, identity = bound_graph
    contents = {
        item["role"]: (root / item["relativePath"]).read_bytes()
        for item in document["files"]
        if item["role"] in {row[0] for row in builds._BUILD_ORDER}
    }
    completed = Row(identity=identity, metadata=Row(source="synthetic comparison fixture"))
    monkeypatch.setattr(builds, "_candidate_build_bytes", lambda receipt: contents)
    monkeypatch.setattr(builds, "_completed_builds", lambda receipt: completed)
    if fault == "candidate":
        document["candidate"]["candidateCommit"] = "f" * 40
    elif fault == "artifact":
        contents["sdist_built_wheel_repeat"] = b"foreign fixture bytes"
    with _helpers["_candidate_helpers"]["freeze"](bound_graph) as files:
        if fault is not None:
            with pytest.raises(ValueError):
                builds.bind_build_output_files(object(), files)
            return
        receipt = builds.bind_build_output_files(object(), files)
        observed = builds.build_output_metadata(receipt)
        assert observed.builds == completed.metadata

        # Once transferred, final acceptance must not reopen removed intermediates.
        def released(_):
            pytest.fail("Final acceptance reopened disposable build files")

        monkeypatch.setattr(builds, "_candidate_build_bytes", released)
        assert builds.build_output_metadata(receipt) == observed
    with pytest.raises(ValueError, match="closed"):
        builds.build_output_metadata(receipt)


def test_invalid_build_inputs_stop_before_any_installation(monkeypatch):
    from scripts import qualification_builds as builds
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    def forbidden(*args, **kwargs):
        pytest.fail("Unbound input reached an installer")

    monkeypatch.setattr(builds, "install_build_environment", forbidden)
    with OwnedQualificationWorkV1() as work:
        with pytest.raises(TypeError):
            builds.build_candidate_artifacts(work, {"passed": True})
        with pytest.raises(RuntimeError, match="closing"):
            work._accepting()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("fault", [None, "kind", "artifact", "pid", "version", "extra"])
def test_build_observation_binds_kind_artifact_and_actual_process(kind, fault):
    from scripts.qualification_build_environment import _worker_observation

    value = dict(
        version=1,
        kind=kind,
        pid=73,
        imports=5,
        file_origins=142,
        source_fallback=False,
        artifact="hermes_realtime-0.0.3-py3-none-any.whl"
        if kind == "wheel"
        else "hermes_realtime-0.0.3.tar.gz",
    )
    if fault == "kind":
        value["kind"] = "imports"
    elif fault == "artifact":
        value["artifact"] = "../unbound.fixture"
    elif fault == "pid":
        value["pid"] = 74
    elif fault == "version":
        value["version"] = True
    elif fault == "extra":
        value["output"] = "private-fixture"
    if fault is None:
        assert _worker_observation(json.dumps(value).encode(), 73, kind) == (5, 142)
    else:
        with pytest.raises(ValueError):
            _worker_observation(json.dumps(value).encode(), 73, kind)


@pytest.mark.skipif(os.name != "nt", reason="genuine Windows source authority")
def test_build_source_is_read_from_archive_even_after_checkout_changes(source, tools):
    from scripts.qualification_build_inputs import bind_build_inputs
    from scripts.qualification_builds import _archive_source_files

    repository, archive, identity, _ = source
    helper, wheels = _helpers["values"]()
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=helper["pins"](wheels),
        constraints=b"",
    )
    path = repository / "README.md"
    original = path.read_bytes()
    try:
        path.write_bytes(b"uncommitted fixture cannot become build source\n")
        files = _archive_source_files(inputs)
        assert files["README.md"] == original
        assert "scripts/qualification_build_worker.py" in files
        assert not any(name.startswith(".git/") for name in files)
    finally:
        path.write_bytes(original)


@pytest.mark.skipif(os.name != "nt", reason="genuine Windows source authority")
def test_sdist_materialization_reuses_the_source_and_wheel_comparison(source):
    from scripts.candidate_wheel import _verify_candidate_wheel_bytes_v1
    from scripts.qualification_sdist import _candidate_sdist_files, inspect_candidate_sdist

    helper = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))
    _, archive, identity, wheel = source
    verified = _verify_candidate_wheel_bytes_v1(
        archive, identity, wheel, hashlib.sha256(wheel).hexdigest()
    )
    raw = helper["_sdist"](source)
    metadata, contents = _candidate_sdist_files(archive, identity, verified, raw)
    assert metadata == inspect_candidate_sdist(archive, identity, verified, raw)
    assert contents["README.md"] == b"# Synthetic description\n"
    assert "PKG-INFO" in contents and len(contents) == metadata.files
    with pytest.raises(ValueError):
        _candidate_sdist_files(archive, identity, verified, helper["_sdist"](source, "changed"))


@pytest.fixture
def completion_rows(monkeypatch):
    """Fault-injection rows for the private comparison, never minted build receipts."""
    from types import SimpleNamespace as Row

    from scripts import qualification_builds as builds

    selected = Row(source_archive_sha256="a" * 64)
    installed = Row(inputs=selected, worker_sha256="b" * 64)
    rows = []
    for index, (role, kind, upstream) in enumerate(builds._BUILD_ORDER):
        raw = b"fixture wheel" if kind == "wheel" else b"fixture sdist"
        invocations = tuple(
            Row(
                process=Row(pid=index * 3 + offset, creation_filetime=100),
                working_directory=f"owned-workspace-{index}",
                metadata=Row(role="uv" if offset == 0 else "build_python"),
            )
            for offset in range(3)
        )
        rows.append(
            Row(
                metadata=Row(
                    role=role,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    size=len(raw),
                    installed=installed,
                    input_sha256=hashlib.sha256(b"fixture sdist").hexdigest()
                    if upstream
                    else selected.source_archive_sha256,
                ),
                contents=raw,
                invocations=invocations,
                work=Row(_closed=True, _unrecoverable=None),
            )
        )
    monkeypatch.setattr(builds, "_tool_invocation_for_consumer", lambda value: value)
    monkeypatch.setattr(builds, "tool_invocation_metadata", lambda value: value.metadata)
    return rows


@pytest.mark.parametrize(
    "fault",
    [None, "replay", "workspace", "cleanup", "output", "lineage", "candidate", "worker", "role"],
)
def test_reproducibility_requires_independent_complete_same_source_builds(completion_rows, fault):
    from types import SimpleNamespace as Row

    from scripts.qualification_builds import _validate_produced

    rows = completion_rows
    if fault == "replay":
        rows[1].invocations = rows[0].invocations
    elif fault == "workspace":
        rows[1].invocations[-1].working_directory = rows[0].invocations[-1].working_directory
    elif fault == "cleanup":
        rows[1].work._closed = False
    elif fault == "output":
        rows[1].contents = b"different fixture wheel"
        rows[1].metadata.sha256 = hashlib.sha256(rows[1].contents).hexdigest()
        rows[1].metadata.size = len(rows[1].contents)
    elif fault == "lineage":
        rows[-1].metadata.input_sha256 = "c" * 64
    elif fault == "candidate":
        rows[1].metadata.installed = Row(
            inputs=Row(source_archive_sha256="c" * 64), worker_sha256="b" * 64
        )
    elif fault == "worker":
        rows[1].metadata.installed = Row(
            inputs=rows[0].metadata.installed.inputs, worker_sha256="c" * 64
        )
    elif fault == "role":
        rows[1].invocations[0].metadata.role = "build_python"
    if fault is None:
        _validate_produced(tuple(rows))
    else:
        with pytest.raises(ValueError):
            _validate_produced(tuple(rows))
