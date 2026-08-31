from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest


def _archive() -> tuple[bytes, object]:
    from scripts import candidate_e2e_fast_track as fast_track

    CandidateArchiveMemberV1 = (
        fast_track.candidate_source_archive_oracle.CandidateArchiveMemberV1
    )
    CandidateSourceArchiveMetadataV1 = fast_track.CandidateSourceArchiveMetadataV1

    prefix = "hermes-realtime-0.0.3"
    payload = b"[project]\nname = \"fixture\"\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        root = tarfile.TarInfo(prefix)
        root.type = tarfile.DIRTYPE
        root.mode = 0o775
        archive.addfile(root)
        directory = tarfile.TarInfo(f"{prefix}/src")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o775
        archive.addfile(directory)
        source = tarfile.TarInfo(f"{prefix}/src/pyproject.toml")
        source.size = len(payload)
        source.mode = 0o664
        archive.addfile(source, io.BytesIO(payload))
    archive_bytes = buffer.getvalue()
    metadata = CandidateSourceArchiveMetadataV1(
        candidate_head_oid="a" * 40,
        candidate_tree_oid="b" * 40,
        canonical_baseline_oid="c" * 40,
        canonical_diff_sha256="d" * 64,
        prefix=prefix,
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        archive_bytes=len(archive_bytes),
        manifest=(
            CandidateArchiveMemberV1("", "dir", "40000", 0o775, 0, None),
            CandidateArchiveMemberV1("src", "dir", "40000", 0o775, 0, None),
            CandidateArchiveMemberV1(
                "src/pyproject.toml",
                "file",
                "100644",
                0o664,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )
    return archive_bytes, metadata


def test_materialize_archive_creates_exact_disposable_source_tree(tmp_path: Path) -> None:
    from scripts.candidate_e2e_fast_track import _materialize_archive

    archive, metadata = _archive()

    root = _materialize_archive(tmp_path, archive, metadata)

    assert root == tmp_path / "candidate"
    assert root.is_dir()
    assert (root / "src" / "pyproject.toml").read_bytes() == b'[project]\nname = "fixture"\n'
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == [
        "src",
        "src/pyproject.toml",
    ]


def test_materialize_archive_refuses_an_existing_destination(tmp_path: Path) -> None:
    from scripts.candidate_e2e_fast_track import _materialize_archive

    archive, metadata = _archive()
    (tmp_path / "candidate").mkdir()

    try:
        _materialize_archive(tmp_path, archive, metadata)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing candidate destination was overwritten")


def test_materialize_archive_refuses_metadata_drift_before_output(tmp_path: Path) -> None:
    from scripts.candidate_e2e_fast_track import _materialize_archive

    archive, metadata = _archive()

    with pytest.raises(ValueError, match="verified metadata"):
        _materialize_archive(tmp_path, archive + b"drift", metadata)

    assert not (tmp_path / "candidate").exists()


def test_materialize_archive_refuses_unsafe_members_before_output(tmp_path: Path) -> None:
    from scripts.candidate_e2e_fast_track import _materialize_archive

    _, metadata = _archive()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        root = tarfile.TarInfo(metadata.prefix)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        escape = tarfile.TarInfo(f"{metadata.prefix}/../escape.txt")
        escape.size = 1
        archive.addfile(escape, io.BytesIO(b"x"))
    unsafe = buffer.getvalue()
    matching_envelope = replace(
        metadata,
        archive_sha256=hashlib.sha256(unsafe).hexdigest(),
        archive_bytes=len(unsafe),
    )

    with pytest.raises(ValueError, match="unsafe member name"):
        _materialize_archive(tmp_path, unsafe, matching_envelope)

    assert not (tmp_path / "candidate").exists()


def test_run_fast_track_consumes_verified_token_into_downstream_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    candidate = tmp_path / "checkout"
    candidate.mkdir()
    executable = tmp_path / "git.exe"
    executable.write_bytes(b"git")
    identity = object()
    pin = object()
    token = object()
    metadata = object()
    archive = b"verified archive"
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(fast_track, "_measure_git_pin", lambda value: pin, raising=False)
    monkeypatch.setattr(
        fast_track.release_gate,
        "scan_git_blobs",
        lambda root: events.append(("scan-git-blobs", root)),
    )
    monkeypatch.setattr(
        fast_track.task13_artifact_orchestrator,
        "verify_clean_tracked_candidate",
        lambda value: identity,
        raising=False,
    )
    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle,
        "capture_candidate_source_archive",
        lambda root, observed_identity, observed_pin: (
            events.append(("capture", root, observed_identity, observed_pin)) or token
        ),
    )
    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle,
        "verified_candidate_source_archive_metadata",
        lambda value: metadata,
    )
    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle,
        "_archive_bytes_for_consumer",
        lambda observed_token, observed_identity: (
            events.append(("consume", observed_token, observed_identity)) or archive
        ),
    )

    def materialize(workspace: Path, payload: bytes, observed_metadata: object) -> Path:
        assert workspace.is_dir()
        events.append(("materialize", payload, observed_metadata))
        root = workspace / "candidate"
        root.mkdir()
        return root

    monkeypatch.setattr(fast_track, "_materialize_archive", materialize)

    def gate(root: Path, **arguments: object) -> None:
        assert root.is_dir()
        events.append(("gate", root.name, arguments))

    monkeypatch.setattr(fast_track.release_gate, "gate_materialized_candidate", gate, raising=False)

    fast_track.run_fast_track(
        candidate,
        git_executable=executable,
        require_livekit=True,
        livekit_executable=tmp_path / "livekit.exe",
        livekit_executable_sha256="e" * 64,
        livekit_pid=123,
    )

    assert events == [
        ("scan-git-blobs", candidate.resolve()),
        ("capture", candidate.resolve(), identity, pin),
        ("consume", token, identity),
        ("materialize", archive, metadata),
        (
            "gate",
            "candidate",
            {
                "livekit": True,
                "livekit_executable": tmp_path / "livekit.exe",
                "livekit_executable_sha256": "e" * 64,
                "livekit_pid": 123,
            },
        ),
    ]


def test_main_forwards_candidate_and_livekit_ownership_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from scripts import candidate_e2e_fast_track as fast_track

    candidate = tmp_path / "candidate"
    git = tmp_path / "git.exe"
    livekit = tmp_path / "livekit.exe"
    observed: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        fast_track,
        "run_fast_track",
        lambda root, **arguments: observed.append((root, arguments)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "candidate_e2e_fast_track.py",
            "--candidate",
            str(candidate),
            "--git-executable",
            str(git),
            "--require-livekit",
            "--livekit-executable",
            str(livekit),
            "--livekit-executable-sha256",
            "f" * 64,
            "--livekit-pid",
            "456",
        ],
    )

    fast_track.main()

    assert observed == [
        (
            candidate,
            {
                "git_executable": git,
                "require_livekit": True,
                "livekit_executable": livekit,
                "livekit_executable_sha256": "f" * 64,
                "livekit_pid": 456,
            },
        )
    ]
