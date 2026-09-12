import hashlib
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _runtime_observation() -> SimpleNamespace:
    return SimpleNamespace(
        python_version="3.11.16",
        soabi="cpython-311-x86_64-linux-gnu",
        ext_suffix=".cpython-311-x86_64-linux-gnu.so",
        multiarch="x86_64-linux-gnu",
        glibc="2.36",
    )


def _ref(role: str, path: str, raw: bytes) -> dict[str, object]:
    return {
        "role": role,
        "relativePath": path,
        "basename": Path(path).name,
        "sha256": _sha(raw),
        "bytes": len(raw),
    }


def _prepare(work, tmp_path, monkeypatch, *, dependency_tag="py3-none-any", locked=True):
    from scripts import qualification_linux_prerequisite as linux
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_build_inputs import BoundBuildInputsV1, BuildInputMetadataV1
    from scripts.qualification_builds import CandidateBuildsV1
    from scripts.qualification_linux_image import (
        AdmittedLinuxRuntimeImageV1,
        LinuxRuntimeImageMetadataV1,
    )

    wheel_helpers = run_path(str(Path.cwd() / "tests/test_qualification_wheelhouse.py"))
    dependency_name, dependency = wheel_helpers["wheel"]("aiohttp", "1.0", tag=dependency_tag)
    direct_name, direct = wheel_helpers["wheel"](
        "hermes_realtime", "0.0.3", requires=("aiohttp==1.0",)
    )
    assert direct_name == "hermes_realtime-0.0.3-py3-none-any.whl"
    requirements = (f"aiohttp==1.0 --hash=sha256:{_sha(dependency)}\n").encode("ascii")
    constraints = b"# No additional constraints\n"
    paths = {
        "requirements": "inputs/linux/requirements.txt",
        "constraints": "inputs/linux/constraints.txt",
        "wheel": "inputs/linux/wheels/" + dependency_name,
    }
    closure = {
        paths["requirements"]: requirements,
        paths["constraints"]: constraints,
        paths["wheel"]: dependency,
    }
    document = {
        "schemaVersion": 1,
        "purpose": "realtime_linux_runtime",
        "pythonVersion": "3.11.16",
        "platform": "linux_x86_64",
        "requirements": _ref("requirements", paths["requirements"], requirements),
        "constraints": _ref("constraints", paths["constraints"], constraints),
        "wheels": [_ref("wheel", paths["wheel"], dependency)],
    }
    manifest = core.canonical_json_bytes(document)
    source = BuildInputMetadataV1(
        "a" * 40,
        "b" * 40,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "3.11.16",
        "f" * 64,
        "0" * 64,
        (),
    )
    image_metadata = LinuxRuntimeImageMetadataV1(
        "docker.io/library/python@sha256:" + "1" * 64,
        "3.11.16",
        "2" * 64,
        ("3" * 64,),
        ("4" * 64,),
        123,
    )
    inputs = object.__new__(BoundBuildInputsV1)
    builds = object.__new__(CandidateBuildsV1)
    image = object.__new__(AdmittedLinuxRuntimeImageV1)
    bound = SimpleNamespace(archive=object(), identity=object())
    monkeypatch.setattr(linux, "_build_inputs_for_consumer", lambda value: bound)
    monkeypatch.setattr(linux, "build_input_metadata", lambda value: source)
    monkeypatch.setattr(
        linux, "candidate_build_metadata", lambda value: SimpleNamespace(inputs=source)
    )
    monkeypatch.setattr(linux, "_candidate_build_bytes", lambda value: {"direct_wheel": direct})
    monkeypatch.setattr(
        linux,
        "_archive_locked_wheels",
        lambda *args: (
            source.source_lock_sha256,
            frozenset(
                {("aiohttp", "1.0", dependency_name, _sha(dependency), len(dependency))}
                if locked
                else set()
            ),
        ),
    )
    monkeypatch.setattr(linux, "linux_image_metadata", lambda value: image_metadata)
    monkeypatch.setattr(linux.service, "_workflow_from_source", lambda *args: b"name: gates\n")
    prepared = linux.prepare_linux_receipt_inputs(
        work,
        inputs,
        builds,
        image,
        manifest_path="inputs/linux/manifest.json",
        manifest=manifest,
        closure=closure,
    )
    return linux, prepared, image, source


@pytest.mark.skipif(__import__("os").name != "nt", reason="immutable snapshot owner needs Windows")
def test_preparation_binds_source_build_image_and_owned_linux_recipe(tmp_path, monkeypatch):
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    with OwnedQualificationWorkV1() as work:
        linux, prepared, _, source = _prepare(work, tmp_path, monkeypatch)
        metadata = linux.linux_prerequisite_metadata(prepared)
        assert metadata.source_commit == source.source_commit
        assert metadata.python_version == "3.11.16"
        assert metadata.wheel_count == 2

    with pytest.raises(ValueError, match="unregistered"):
        linux.linux_prerequisite_metadata(prepared)


@pytest.mark.skipif(__import__("os").name != "nt", reason="immutable snapshot owner needs Windows")
def test_authenticated_prefinal_transfers_once_then_retains_only_final_authority(
    tmp_path, monkeypatch
):
    from scripts.github_actions_linux_receipt import (
        LinuxReceiptMetadataV1,
        _AuthenticatedObservation,
    )
    from scripts.qualification_dependency_files import BoundDependencyPurposeV1
    from scripts.qualification_owned_work import OwnedQualificationWorkV1
    from scripts.retained_qualification_inputs import RetainedQualificationInputFilesV1

    with OwnedQualificationWorkV1() as work:
        linux, prepared, image, _ = _prepare(work, tmp_path, monkeypatch)
        expected = linux._prepared(prepared).expected
        observation = _AuthenticatedObservation(
            expected,
            LinuxReceiptMetadataV1(
                42, 1, "5" * 64, expected.candidate_commit, expected.direct_wheel_sha256, True
            ),
            _runtime_observation(),
        )
        monkeypatch.setattr(
            linux.service, "_authenticate_observation", lambda *args, **kwargs: observation
        )
        prefinal = linux.authenticate_prefinal_linux_receipt(42, prepared)
        final_files = object.__new__(RetainedQualificationInputFilesV1)
        runtime = object.__new__(BoundDependencyPurposeV1)
        final_binding = SimpleNamespace(
            files=final_files,
            metadata=SimpleNamespace(qualification_input_sha256="6" * 64),
            linux_target=linux.authenticated_linux_wheel_target(prefinal),
        )
        wrong_runtime = object.__new__(BoundDependencyPurposeV1)
        wrong_binding = SimpleNamespace(
            files=final_files,
            metadata=SimpleNamespace(qualification_input_sha256="6" * 64),
            linux_target=object(),
        )
        monkeypatch.setattr(linux.service, "_facts_from_capabilities", lambda *args: expected)
        monkeypatch.setattr(linux.service, "_match_payload", lambda *args: None)
        monkeypatch.setattr(
            linux,
            "_dependency_binding_for_consumer",
            lambda value: wrong_binding if value is wrong_runtime else final_binding,
        )
        with pytest.raises(ValueError, match="target differs"):
            linux.bind_prefinal_linux_receipt(prefinal, final_files, wrong_runtime, image)
        bound = linux.bind_prefinal_linux_receipt(prefinal, final_files, runtime, image)
        with pytest.raises(ValueError, match="already transferred"):
            linux.bind_prefinal_linux_receipt(prefinal, final_files, runtime, image)

    with pytest.raises(ValueError, match="unregistered"):
        linux.prefinal_linux_receipt_metadata(prefinal)
    assert linux.bound_prefinal_linux_receipt_metadata(bound).qualification_input_sha256 == "6" * 64


def test_capabilities_are_opaque_and_wrong_final_lease_refuses(monkeypatch):
    from scripts import qualification_linux_prerequisite as linux
    from scripts.qualification_dependency_files import BoundDependencyPurposeV1
    from scripts.retained_qualification_inputs import RetainedQualificationInputFilesV1

    for capability in (
        linux.PreparedLinuxReceiptInputsV1,
        linux.AuthenticatedPrefinalLinuxReceiptV1,
        linux.BoundPrefinalLinuxReceiptV1,
    ):
        with pytest.raises(TypeError):
            capability()
    expected = SimpleNamespace(
        image_reference="image",
        image_config_sha256="config",
        image_layers=("layer",),
        image_diff_ids=("diff",),
    )
    prefinal = object.__new__(linux.AuthenticatedPrefinalLinuxReceiptV1)
    target = object()
    linux._PREFINAL[prefinal] = SimpleNamespace(
        prepared=object(), observation=SimpleNamespace(facts=expected), target=target
    )
    monkeypatch.setattr(linux, "_prepared", lambda value: SimpleNamespace(expected=expected))
    monkeypatch.setattr(linux.service, "_facts_from_capabilities", lambda *args: expected)
    monkeypatch.setattr(
        linux,
        "_authenticated_linux_binding_for_consumer",
        lambda value: (
            object(),
            (
                expected.image_reference,
                expected.image_config_sha256,
                expected.image_layers,
                expected.image_diff_ids,
            ),
        ),
    )
    runtime = object.__new__(BoundDependencyPurposeV1)
    expected_files = object.__new__(RetainedQualificationInputFilesV1)
    supplied_files = object.__new__(RetainedQualificationInputFilesV1)
    monkeypatch.setattr(
        linux,
        "_dependency_binding_for_consumer",
        lambda value: SimpleNamespace(
            files=expected_files,
            metadata=SimpleNamespace(qualification_input_sha256="7" * 64),
            linux_target=target,
        ),
    )
    with pytest.raises(ValueError, match="lease or target differs"):
        linux.bind_prefinal_linux_receipt(
            prefinal,
            supplied_files,
            runtime,
            object.__new__(linux.AdmittedLinuxRuntimeImageV1),
        )


@pytest.mark.skipif(__import__("os").name != "nt", reason="immutable snapshot owner needs Windows")
def test_preparation_refuses_unlocked_wheel(tmp_path, monkeypatch):
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    with pytest.raises(ValueError, match="source lock"), OwnedQualificationWorkV1() as work:
        _prepare(work, tmp_path, monkeypatch, locked=False)


@pytest.mark.skipif(__import__("os").name != "nt", reason="immutable snapshot owner needs Windows")
def test_manylinux_compatibility_comes_only_from_authenticated_runtime(monkeypatch, tmp_path):
    from scripts.github_actions_linux_receipt import (
        LinuxReceiptMetadataV1,
        _AuthenticatedObservation,
    )
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    with OwnedQualificationWorkV1() as work:
        linux, prepared, _, _ = _prepare(
            work,
            tmp_path,
            monkeypatch,
            dependency_tag="cp311-cp311-manylinux_2_17_x86_64",
        )
        expected = linux._prepared(prepared).expected
        observation = _AuthenticatedObservation(
            expected,
            LinuxReceiptMetadataV1(
                42, 1, "5" * 64, expected.candidate_commit, expected.direct_wheel_sha256, True
            ),
            _runtime_observation(),
        )
        monkeypatch.setattr(
            linux.service, "_authenticate_observation", lambda *args, **kwargs: observation
        )
        assert (
            linux.prefinal_linux_receipt_metadata(
                linux.authenticate_prefinal_linux_receipt(42, prepared)
            ).run_id
            == 42
        )
