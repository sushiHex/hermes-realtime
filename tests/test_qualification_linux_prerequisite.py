import hashlib
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
    dependency_name, dependency = wheel_helpers["wheel"](
        "aiohttp", "1.0", tag=dependency_tag
    )
    direct_name, direct = wheel_helpers["wheel"](
        "hermes_realtime", "0.0.3", requires=("aiohttp==1.0",)
    )
    assert direct_name == "hermes_realtime-0.0.3-py3-none-any.whl"
    requirements = (
        f"aiohttp==1.0 --hash=sha256:{_sha(dependency)}\n"
    ).encode("ascii")
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
            object(),
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
        )
        monkeypatch.setattr(linux.service, "_facts_from_capabilities", lambda *args: expected)
        monkeypatch.setattr(linux.service, "_match_payload", lambda *args: None)
        monkeypatch.setattr(linux, "_dependency_binding_for_consumer", lambda value: final_binding)
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
    expected = object()
    prefinal = object.__new__(linux.AuthenticatedPrefinalLinuxReceiptV1)
    linux._PREFINAL[prefinal] = SimpleNamespace(
        prepared=object(), observation=SimpleNamespace(facts=expected)
    )
    monkeypatch.setattr(linux, "_prepared", lambda value: SimpleNamespace(expected=expected))
    monkeypatch.setattr(linux.service, "_facts_from_capabilities", lambda *args: expected)
    runtime = object.__new__(BoundDependencyPurposeV1)
    expected_files = object.__new__(RetainedQualificationInputFilesV1)
    supplied_files = object.__new__(RetainedQualificationInputFilesV1)
    monkeypatch.setattr(
        linux,
        "_dependency_binding_for_consumer",
        lambda value: SimpleNamespace(
            files=expected_files, metadata=SimpleNamespace(qualification_input_sha256="7" * 64)
        ),
    )
    with pytest.raises(ValueError, match="lease differs"):
        linux.bind_prefinal_linux_receipt(
            prefinal,
            supplied_files,
            runtime,
            object.__new__(linux.AdmittedLinuxRuntimeImageV1),
        )


@pytest.mark.skipif(__import__("os").name != "nt", reason="immutable snapshot owner needs Windows")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"locked": False}, "source lock"),
        ({"dependency_tag": "cp311-cp311-manylinux_2_17_x86_64"}, "compatible"),
    ],
)
def test_preparation_refuses_unlocked_and_unobserved_manylinux_wheels(
    tmp_path, monkeypatch, options, message
):
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    with pytest.raises(ValueError, match=message), OwnedQualificationWorkV1() as work:
        _prepare(work, tmp_path, monkeypatch, **options)
