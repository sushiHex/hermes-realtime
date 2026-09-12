"""Runtime installation cannot borrow authority from supplied reports or builds."""

import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest

_helpers = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))
source = _helpers["source"]
bound_graph = _helpers["bound_graph"]
dependency_graph = _helpers["dependency_graph"]
tools = _helpers["tools"]


def test_dependency_purpose_capabilities_cannot_be_supplied_or_relabelled():
    from scripts.qualification_dependency_files import (
        BoundDependencyPurposeV1,
        bind_dependency_purpose,
        dependency_purpose_metadata,
    )

    with pytest.raises(TypeError):
        BoundDependencyPurposeV1()
    with pytest.raises(TypeError):
        dependency_purpose_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        dependency_purpose_metadata(object.__new__(BoundDependencyPurposeV1))
    with pytest.raises(ValueError, match="purpose"):
        bind_dependency_purpose({}, {}, purpose="unavailable")


@pytest.mark.parametrize("fault", [None, "linux_wheel"])
def test_one_runtime_purpose_has_its_own_authority_without_accepting_the_whole_closure(
    dependency_graph, fault
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import (
        _dependency_wheels_for_consumer,
        bind_dependency_files,
        bind_dependency_purpose,
        dependency_file_metadata,
        dependency_purpose_metadata,
    )
    from scripts.qualify_evidence_slice_zero import canonical_json_bytes

    root, _, document, archive, identity = dependency_graph
    if fault == "linux_wheel":
        reference = next(
            row for row in document["files"] if row["role"] == "linux_runtime_wheelhouse_manifest"
        )
        manifest = json.loads((root / reference["relativePath"]).read_bytes())
        wheel = manifest["wheels"][0]
        original = wheel["sha256"]
        raw = b"not a source-locked wheel"
        (root / wheel["relativePath"]).write_bytes(raw)
        wheel.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        requirements = manifest["requirements"]
        path = root / requirements["relativePath"]
        raw = path.read_bytes().replace(original.encode(), wheel["sha256"].encode())
        path.write_bytes(raw)
        requirements.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        raw = canonical_json_bytes(manifest)
        (root / reference["relativePath"]).write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    purpose = "realtime_windows_direct_runtime"
    with _helpers["_candidate_helpers"]["freeze"](dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        selected = bind_dependency_purpose(files, candidate, purpose=purpose)
        metadata = dependency_purpose_metadata(selected)
        assert tuple(row.purpose for row in metadata.wheelhouses) == (purpose,)
        assert _dependency_wheels_for_consumer(selected, purpose)[0]
        with pytest.raises(ValueError, match="purpose"):
            _dependency_wheels_for_consumer(selected, "realtime_linux_runtime")
        with pytest.raises(TypeError):
            dependency_file_metadata(selected)
        if fault is not None:
            with pytest.raises(ValueError):
                bind_dependency_files(files, candidate)
    with pytest.raises(ValueError, match="closed"):
        dependency_purpose_metadata(selected)


def test_runtime_capabilities_are_recipe_minted_only():
    from scripts.qualification_runtime_environment import (
        CompletedRuntimeEnvironmentV1,
        InstalledRuntimeEnvironmentV1,
        completed_runtime_metadata,
        installed_runtime_metadata,
    )

    for cls, reader in (
        (InstalledRuntimeEnvironmentV1, installed_runtime_metadata),
        (CompletedRuntimeEnvironmentV1, completed_runtime_metadata),
    ):
        with pytest.raises(TypeError):
            cls()
        with pytest.raises(TypeError):
            reader({"passed": True})
        with pytest.raises(ValueError, match="unregistered"):
            reader(object.__new__(cls))


@pytest.mark.parametrize(
    "purpose", ["realtime_linux_runtime", "realtime_windows_direct_runtime", "build", "unavailable"]
)
def test_unavailable_or_unbound_runtime_inputs_refuse_before_installation(monkeypatch, purpose):
    from scripts import qualification_runtime_environment as runtime
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    def forbidden(*args, **kwargs):
        pytest.fail("Unbound runtime input reached an installer")

    monkeypatch.setattr(runtime, "_install_packages", forbidden)
    with OwnedQualificationWorkV1() as work:
        with pytest.raises((TypeError, ValueError)):
            runtime.install_runtime_environment(work, {}, {}, {}, purpose=purpose)
        with pytest.raises(RuntimeError, match="closing"):
            work._accepting()


@pytest.mark.parametrize("fault", [None, "kind", "count", "artifact"])
def test_runtime_observation_is_bound_to_its_exact_purpose(fault):
    from scripts.qualification_build_environment import _worker_observation

    purpose = "realtime_windows_direct_runtime"
    value = dict(
        version=1,
        kind=purpose,
        pid=81,
        imports=4,
        file_origins=501,
        source_fallback=False,
        artifact=None,
    )
    if fault == "kind":
        value["kind"] = "realtime_windows_sdist_built_runtime"
    elif fault == "count":
        value["imports"] = 5
    elif fault == "artifact":
        value["artifact"] = "unrequested.fixture"
    if fault is None:
        assert _worker_observation(json.dumps(value).encode(), 81, purpose) == (4, 501)
    else:
        with pytest.raises(ValueError):
            _worker_observation(json.dumps(value).encode(), 81, purpose)


@pytest.mark.parametrize("fault", [None, "input_digest", "source", "tool_bytes", "tool_version"])
@pytest.mark.parametrize("coverage", ["all", "purpose"])
def test_runtime_selects_the_verified_candidate_and_purpose_closure(
    dependency_graph, tools, monkeypatch, fault, coverage
):
    from types import SimpleNamespace as Row

    from scripts import qualification_runtime_environment as runtime
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import (
        bind_dependency_files,
        bind_dependency_purpose,
    )
    from scripts.qualification_tool_environment import _tool_image_for_consumer

    root, _, document, archive, identity = dependency_graph

    def write(reference, raw):
        (root / reference["relativePath"]).write_bytes(raw)
        reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    for declared in document["toolIdentities"]:
        if declared["role"] == "hatchling":
            continue
        image, _, version = _tool_image_for_consumer(tools, declared["role"])
        write(declared["artifact"], image.read_bytes())
        declared["version"] = version
    document["expected"]["pythonFullVersion"] = "3.11.16"
    for reference in document["files"]:
        if reference["role"] in core._WHEELHOUSE_ROLES.values():
            value = json.loads((root / reference["relativePath"]).read_bytes())
            value["pythonVersion"] = "3.11.16"
            write(reference, core.canonical_json_bytes(value))
    declared = next(item for item in document["toolIdentities"] if item["role"] == "uv")
    if fault == "tool_bytes":
        write(declared["artifact"], b"different synthetic installer")
    elif fault == "tool_version":
        declared["version"] = "unadmitted-version"
    with _helpers["_candidate_helpers"]["freeze"](dependency_graph) as files:
        candidate = bind_candidate_files(files, archive, identity)
        dependencies = (
            bind_dependency_files(files, candidate)
            if coverage == "all"
            else bind_dependency_purpose(
                files, candidate, purpose="realtime_windows_direct_runtime"
            )
        )
        # The upstream build completion is a comparison fixture, not execution proof.
        digest = hashlib.sha256(core.canonical_json_bytes(document)).hexdigest()
        output = Row(
            qualification_input_sha256="f" * 64 if fault == "input_digest" else digest,
            builds=Row(
                inputs=Row(
                    source_commit="f" * 40 if fault == "source" else identity.candidate_head_oid,
                    source_tree=identity.candidate_tree_oid,
                )
            ),
        )
        monkeypatch.setattr(runtime, "build_output_metadata", lambda receipt: output)
        if fault is not None:
            with pytest.raises(ValueError):
                runtime._runtime_inputs(
                    dependencies, object(), tools, "realtime_windows_direct_runtime"
                )
            return
        for purpose in runtime._PURPOSES:
            if coverage == "purpose" and purpose != "realtime_windows_direct_runtime":
                with pytest.raises(ValueError, match="purpose"):
                    runtime._runtime_inputs(dependencies, object(), tools, purpose)
                continue
            wheels, requirements, _, worker = runtime._runtime_inputs(
                dependencies, object(), tools, purpose
            )
            assert "hermes_realtime-0.0.3-py3-none-any.whl" in wheels
            assert b"hermes-realtime==0.0.3 --hash=sha256:" in requirements
            assert worker == (Path.cwd() / "scripts/qualification_build_worker.py").read_bytes()
