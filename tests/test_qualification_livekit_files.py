import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained file authority")


def _zip(
    *, executable: bytes = b"livekit synthetic executable\n", sidecar: bytes = b"sidecar\n"
) -> tuple[bytes, bytes]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("livekit-server.exe", executable)
        archive.writestr("sidecars/media.dll", sidecar)
    return stream.getvalue(), executable


def _inputs_for_policy(
    tmp_path_factory,
    *,
    archive_sha256: str,
    executable_sha256: str,
):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.qualification_build_inputs import bind_build_inputs

    candidate_helpers = run_path(str(Path.cwd() / "tests/test_qualification_candidate_files.py"))
    archive_helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    build_helpers = run_path(str(Path.cwd() / "tests/test_qualification_build_inputs.py"))
    repository, _, prior, _ = candidate_helpers["_make_source"](tmp_path_factory)
    workflow, browser, gate = (
        repository / ".github/workflows/release-gates.yml",
        repository / "tests/integration/test_browser_self_acceptance.py",
        repository / "scripts/real_natural_work_gate.py",
    )
    workflow.parent.mkdir(parents=True)
    browser.parent.mkdir(parents=True)
    gate.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        "\n".join(
            (
                "name: release-gates",
                "- name: Install pinned local LiveKit server",
                "  run: |",
                "    $archive = Join-Path $env:RUNNER_TEMP 'livekit_1.13.4_windows_amd64.zip'",
                "    Invoke-WebRequest 'https://github.com/livekit/livekit/releases/download/"
                "v1.13.4/livekit_1.13.4_windows_amd64.zip' -OutFile $archive",
                "    $actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()",
                f"    $expected = '{archive_sha256}'",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    browser.write_text(
        f'_PINNED_LIVEKIT_SHA256 = "{executable_sha256}"\n',
        encoding="utf-8",
        newline="\n",
    )
    gate.write_text(
        "_CODEX_PROTOCOL_FIXTURE = 'tests/fixtures/codex_app_server_dynamic_tools.json'\n"
    )
    archive_helpers["_git"]("add", ".", cwd=repository)
    archive_helpers["_git"]("commit", "-qm", "synthetic LiveKit policy", cwd=repository)
    identity = archive_helpers["_identity"](repository, prior.canonical_baseline_oid)
    archive = capture_candidate_source_archive(repository, identity, archive_helpers["_pin"]())
    patch = pytest.MonkeyPatch()
    fixture = build_helpers["tools"].__wrapped__(patch)
    tools = next(fixture)
    wheel_helpers, wheels = build_helpers["values"]()
    inputs = bind_build_inputs(
        archive,
        identity,
        tools,
        wheels=wheels,
        requirements=wheel_helpers["pins"](wheels),
        constraints=b"",
    )
    return inputs, identity, fixture, patch


def _close_tools(fixture, patch) -> None:
    try:
        with pytest.raises(StopIteration):
            next(fixture)
        patch.undo()
    except BaseException:
        patch.undo()
        raise


@pytest.fixture
def genuine_inputs(tmp_path_factory):
    archive, executable = _zip()
    inputs, identity, fixture, patch = _inputs_for_policy(
        tmp_path_factory,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        executable_sha256=hashlib.sha256(executable).hexdigest(),
    )
    try:
        yield inputs, identity, archive, executable
    finally:
        _close_tools(fixture, patch)


def test_prefinal_admission_uses_genuine_build_source_and_retains_full_archive(
    genuine_inputs,
):
    from scripts.qualification_livekit_files import (
        admit_livekit_archive,
        livekit_archive_metadata,
    )
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    inputs, identity, archive, executable = genuine_inputs
    with OwnedQualificationWorkV1() as work:
        receipt = admit_livekit_archive(inputs, work, archive)
        metadata = livekit_archive_metadata(receipt)
        assert metadata.source_commit == identity.candidate_head_oid
        assert metadata.executable_sha256 == hashlib.sha256(executable).hexdigest()
        assert tuple(name for name, _, _ in metadata.members) == (
            "livekit-server.exe",
            "sidecars/media.dll",
        )


def test_prefinal_admission_refuses_actual_source_pin_drift(genuine_inputs, tmp_path_factory):
    from scripts import qualification_livekit_files as livekit
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    inputs, _, archive, _ = genuine_inputs
    changed_inputs, _, fixture, patch = _inputs_for_policy(
        tmp_path_factory,
        archive_sha256="0" * 64,
        executable_sha256=hashlib.sha256(genuine_inputs[3]).hexdigest(),
    )
    try:
        with OwnedQualificationWorkV1() as work, pytest.raises(ValueError, match="source pin"):
            livekit.admit_livekit_archive(changed_inputs, work, archive)
    finally:
        _close_tools(fixture, patch)


def test_prefinal_admission_refuses_actual_archive_and_member_digest_drift(
    genuine_inputs, tmp_path_factory
):
    from scripts import qualification_livekit_files as livekit
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    _, _, _, executable = genuine_inputs
    changed_archive, _ = _zip(sidecar=b"changed sidecar\n")
    with OwnedQualificationWorkV1() as work, pytest.raises(ValueError, match="archive differs"):
        livekit.admit_livekit_archive(genuine_inputs[0], work, changed_archive)
    member_archive, _ = _zip(executable=b"changed exe\n")
    member_inputs, _, fixture, patch = _inputs_for_policy(
        tmp_path_factory,
        archive_sha256=hashlib.sha256(member_archive).hexdigest(),
        executable_sha256=hashlib.sha256(executable).hexdigest(),
    )
    try:
        with (
            OwnedQualificationWorkV1() as work,
            pytest.raises(ValueError, match="executable differs"),
        ):
            livekit.admit_livekit_archive(member_inputs, work, member_archive)
    finally:
        _close_tools(fixture, patch)


def test_final_binder_requires_exact_archive_and_executable_and_expires(genuine_inputs, tmp_path):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.qualification_livekit_files import (
        admit_livekit_archive,
        bind_livekit_files,
        livekit_file_metadata,
    )
    from scripts.qualification_owned_work import OwnedQualificationWorkV1
    from scripts.retained_qualification_inputs import retain_qualification_input_files

    inputs, identity, archive, executable = genuine_inputs
    helpers = run_path(str(Path.cwd() / "tests/test_qualify_evidence_slice_zero.py"))
    root, manifest, _, _, _ = helpers["_make_input_root"](tmp_path)
    document = json.loads(manifest.read_bytes())
    document["candidate"] = {
        "baselineCommit": identity.canonical_baseline_oid,
        "candidateCommit": identity.candidate_head_oid,
        "tree": identity.candidate_tree_oid,
        "canonicalDiffSha256": identity.canonical_diff_sha256,
        "version": "0.0.3",
    }
    refs = {item["role"]: item for item in document["files"]}
    for role, raw in (("livekit_archive", archive), ("livekit_executable", executable)):
        path = root / refs[role]["relativePath"]
        path.write_bytes(raw)
        refs[role].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    raw = core.canonical_json_bytes(document)
    manifest.write_bytes(raw)
    work = OwnedQualificationWorkV1()
    try:
        with retain_qualification_input_files(
            root, manifest.name, hashlib.sha256(raw).hexdigest()
        ) as final:
            prefinal = admit_livekit_archive(inputs, work, archive)
            bound = bind_livekit_files(prefinal, final)
            assert livekit_file_metadata(bound).candidate_commit == identity.candidate_head_oid
            altered_parent = tmp_path / "altered"
            altered_parent.mkdir()
            altered_root, altered_manifest, _, _, _ = helpers["_make_input_root"](altered_parent)
            altered_document = json.loads(altered_manifest.read_bytes())
            altered_document["candidate"] = document["candidate"]
            altered_refs = {item["role"]: item for item in altered_document["files"]}
            for role, value in (
                ("livekit_archive", archive + b"final alteration"),
                ("livekit_executable", executable),
            ):
                path = altered_root / altered_refs[role]["relativePath"]
                path.write_bytes(value)
                altered_refs[role].update(
                    sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)
                )
            altered_raw = core.canonical_json_bytes(altered_document)
            altered_manifest.write_bytes(altered_raw)
            with (
                retain_qualification_input_files(
                    altered_root,
                    altered_manifest.name,
                    hashlib.sha256(altered_raw).hexdigest(),
                ) as altered_final,
                pytest.raises(ValueError, match="final files differ"),
            ):
                bind_livekit_files(prefinal, altered_final)
            work.close()
            assert livekit_file_metadata(bound).candidate_commit == identity.candidate_head_oid
            with pytest.raises(RuntimeError, match="closing"):
                bind_livekit_files(prefinal, final)
        with pytest.raises(ValueError, match="closed"):
            livekit_file_metadata(bound)
    finally:
        if not work._closed:
            work.close()
