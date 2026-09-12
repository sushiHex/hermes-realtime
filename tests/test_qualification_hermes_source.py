import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained file authority")


def _codeload() -> bytes:
    from scripts import qualification_hermes_source as owner

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:") as archive:
        for name, payload, mode in (
            (owner._CODELOAD_PREFIX + "/", None, 0o755),
            (owner._CODELOAD_PREFIX + "/hermes_cli/", None, 0o755),
            (owner._CODELOAD_PREFIX + "/hermes_cli/plugins.py", b"plugins\n", 0o644),
            (owner._CODELOAD_PREFIX + "/utils.py", b"utils\n", 0o755),
        ):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE if payload is None else tarfile.REGTYPE
            info.mode = mode
            info.size = 0 if payload is None else len(payload)
            info.mtime = 0
            archive.addfile(info, None if payload is None else io.BytesIO(payload))
    return gzip.compress(raw.getvalue(), mtime=0)


def _admit_policy(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    from scripts import qualification_hermes_source as owner

    source_tar = gzip.decompress(raw)
    normalized, rows = owner._normalize_source_tar(source_tar)
    monkeypatch.setattr(owner, "_CODELOAD_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(owner, "_CODELOAD_BYTES", len(raw))
    monkeypatch.setattr(owner, "_SOURCE_TAR_SHA256", hashlib.sha256(source_tar).hexdigest())
    monkeypatch.setattr(owner, "_SOURCE_TAR_BYTES", len(source_tar))
    monkeypatch.setattr(owner, "_FILE_COUNT", len(rows))
    monkeypatch.setattr(owner, "_TREE_BYTES", sum(len(payload) for _, payload, _ in rows))


def _inputs(tmp_path_factory):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
    from scripts.qualification_build_inputs import bind_build_inputs

    candidate_helpers = run_path(str(Path.cwd() / "tests/test_qualification_candidate_files.py"))
    archive_helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    build_helpers = run_path(str(Path.cwd() / "tests/test_qualification_build_inputs.py"))
    repository, _, prior, _ = candidate_helpers["_make_source"](tmp_path_factory)
    harness = repository / "scripts/qualify_hermes_v020_pluginmanager.py"
    harness.parent.mkdir(parents=True, exist_ok=True)
    harness.write_bytes(b"# candidate Hermes harness\n")
    archive_helpers["_git"]("add", ".", cwd=repository)
    archive_helpers["_git"]("commit", "-qm", "add Hermes harness", cwd=repository)
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
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def genuine_inputs(tmp_path_factory):
    inputs, identity, fixture, patch = _inputs(tmp_path_factory)
    try:
        yield inputs, identity
    finally:
        _close_tools(fixture, patch)


def _final_inputs(tmp_path: Path, identity, source: bytes, harness: bytes):
    from scripts import qualify_evidence_slice_zero as core
    from scripts.retained_qualification_inputs import retain_qualification_input_files

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
    for role, value in (
        ("hermes_source_archive", source),
        ("hermes_pluginmanager_runner", harness),
    ):
        path = root / refs[role]["relativePath"]
        path.write_bytes(value)
        refs[role].update(sha256=hashlib.sha256(value).hexdigest(), bytes=len(value))
    raw = core.canonical_json_bytes(document)
    manifest.write_bytes(raw)
    return retain_qualification_input_files(root, manifest.name, hashlib.sha256(raw).hexdigest())


def test_hermes_source_admission_and_final_transfer(genuine_inputs, monkeypatch, tmp_path):
    from scripts import qualification_hermes_source as owner
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    raw = _codeload()
    _admit_policy(monkeypatch, raw)
    inputs, identity = genuine_inputs
    harness = owner._source_member(inputs, owner._HARNESS, live_tools=True)
    work = OwnedQualificationWorkV1()
    try:
        source = owner.admit_hermes_publisher_source(inputs, work, raw)
        metadata = owner.hermes_publisher_source_metadata(source)
        assert metadata.candidate_commit == identity.candidate_head_oid
        normalized, _ = owner._normalize_source_tar(gzip.decompress(raw))
        with _final_inputs(tmp_path, identity, normalized, harness) as final:
            bound = owner.bind_hermes_publisher_source(source, final)
            work.close()
            assert owner.hermes_publisher_file_metadata(bound).source == metadata
            with pytest.raises(RuntimeError, match="closing"):
                owner.bind_hermes_publisher_source(source, final)
        with pytest.raises(ValueError, match="closed"):
            owner.hermes_publisher_file_metadata(bound)
    finally:
        if not work._closed:
            work.close()


def test_hermes_source_refuses_changed_publisher_bytes_before_parse(genuine_inputs, monkeypatch):
    from scripts import qualification_hermes_source as owner
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    raw = _codeload()
    _admit_policy(monkeypatch, raw)
    with OwnedQualificationWorkV1() as work, pytest.raises(ValueError, match="reviewed publisher"):
        owner.admit_hermes_publisher_source(genuine_inputs[0], work, raw + b"foreign")


def test_hermes_source_tar_identity_refuses_before_archive_parsing(genuine_inputs, monkeypatch):
    from scripts import qualification_hermes_source as owner
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    raw = _codeload()
    _admit_policy(monkeypatch, raw)
    altered = gzip.compress(b"not a Hermes tar archive", mtime=0)
    monkeypatch.setattr(owner, "_CODELOAD_SHA256", hashlib.sha256(altered).hexdigest())
    monkeypatch.setattr(owner, "_CODELOAD_BYTES", len(altered))
    with OwnedQualificationWorkV1() as work, pytest.raises(ValueError, match="source tar differs"):
        owner.admit_hermes_publisher_source(genuine_inputs[0], work, altered)


def test_hermes_source_refuses_changed_final_source_or_harness(
    genuine_inputs, monkeypatch, tmp_path
):
    from scripts import qualification_hermes_source as owner
    from scripts.qualification_owned_work import OwnedQualificationWorkV1

    raw = _codeload()
    _admit_policy(monkeypatch, raw)
    inputs, identity = genuine_inputs
    harness = owner._source_member(inputs, owner._HARNESS, live_tools=True)
    with OwnedQualificationWorkV1() as work:
        source = owner.admit_hermes_publisher_source(inputs, work, raw)
        normalized, _ = owner._normalize_source_tar(gzip.decompress(raw))
        for label, source_bytes, harness_bytes in (
            ("source", normalized + b"changed", harness),
            ("harness", normalized, harness + b"changed"),
        ):
            case = tmp_path / label
            case.mkdir()
            with (
                _final_inputs(case, identity, source_bytes, harness_bytes) as final,
                pytest.raises(ValueError, match="final source or harness"),
            ):
                owner.bind_hermes_publisher_source(source, final)
