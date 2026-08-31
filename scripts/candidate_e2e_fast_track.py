"""Fast-track verified candidate materialization and release qualification."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

try:
    import candidate_source_archive_oracle
    import release_gate
    import task13_artifact_orchestrator
    from candidate_source_archive_oracle import (
        CandidateSourceArchiveMetadataV1,
        GitExecutablePinV1,
    )
    from source_archive_authority import SourceArchivePolicyV1, validated_archive_members
except ModuleNotFoundError:
    from scripts import (
        candidate_source_archive_oracle,
        release_gate,
        task13_artifact_orchestrator,
    )
    from scripts.candidate_source_archive_oracle import (
        CandidateSourceArchiveMetadataV1,
        GitExecutablePinV1,
    )
    from scripts.source_archive_authority import (
        SourceArchivePolicyV1,
        validated_archive_members,
    )

_MAX_MEMBER_COUNT = 4096
_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_TREE_BYTES = 12 * 1024 * 1024


def _measure_git_pin(executable: Path) -> GitExecutablePinV1:
    """Measure one Git executable; the oracle retains and rechecks the exact file."""

    executable = executable.resolve(strict=True)
    completed = subprocess.run(
        (str(executable), "--version"),
        capture_output=True,
        check=True,
    )
    try:
        version = completed.stdout.removesuffix(b"\n").decode("ascii", "strict")
    except UnicodeDecodeError as error:
        raise ValueError("Git version output is not strict ASCII") from error
    if not version or completed.stdout != (version + "\n").encode("ascii"):
        raise ValueError("Git version output is not one canonical line")
    return GitExecutablePinV1(
        executable,
        hashlib.sha256(executable.read_bytes()).hexdigest(),
        version,
        executable.stat().st_nlink,
    )


def run_fast_track(
    candidate: Path,
    *,
    git_executable: Path,
    require_livekit: bool = False,
    livekit_executable: Path | None = None,
    livekit_executable_sha256: str | None = None,
    livekit_pid: int | None = None,
) -> None:
    """Capture, materialize, build, install, and optionally exercise one candidate."""

    candidate = candidate.resolve(strict=True)
    identity = task13_artifact_orchestrator.verify_clean_tracked_candidate(candidate)
    release_gate.scan_git_blobs(candidate)
    pin = _measure_git_pin(git_executable)
    token = candidate_source_archive_oracle.capture_candidate_source_archive(
        candidate, identity, pin
    )
    metadata = candidate_source_archive_oracle.verified_candidate_source_archive_metadata(token)
    archive = candidate_source_archive_oracle._archive_bytes_for_consumer(token, identity)
    with tempfile.TemporaryDirectory(prefix="hermes-realtime-fast-track-") as temporary:
        root = _materialize_archive(Path(temporary), archive, metadata)
        release_gate.gate_materialized_candidate(
            root,
            livekit=require_livekit,
            livekit_executable=livekit_executable,
            livekit_executable_sha256=livekit_executable_sha256,
            livekit_pid=livekit_pid,
        )


def _materialize_archive(
    workspace: Path,
    archive_bytes: bytes,
    metadata: CandidateSourceArchiveMetadataV1,
) -> Path:
    """Materialize one already-verified archive into a disposable mutable tree."""

    if type(workspace) is not type(Path()) or type(archive_bytes) is not bytes:
        raise TypeError("fast-track materialization inputs have invalid types")
    if type(metadata) is not CandidateSourceArchiveMetadataV1:
        raise TypeError("fast-track archive metadata has an invalid type")
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("fast-track workspace must be an existing ordinary directory")
    if len(archive_bytes) != metadata.archive_bytes or (
        hashlib.sha256(archive_bytes).hexdigest() != metadata.archive_sha256
    ):
        raise ValueError("fast-track archive bytes differ from verified metadata")

    policy = SourceArchivePolicyV1(
        prefix=metadata.prefix,
        workspace_child="candidate",
        owner_marker=".fast-track-owner",
        max_members=_MAX_MEMBER_COUNT,
        max_file_bytes=_MAX_FILE_BYTES,
        max_tree_bytes=_MAX_TREE_BYTES,
        error_label="candidate source archive",
    )
    expected_files = {
        member.path: member for member in metadata.manifest if member.kind == "file"
    }
    extracted: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        members = validated_archive_members(archive, policy)
        for info in members:
            relative = info.name.removeprefix(metadata.prefix + "/")
            expected = expected_files.get(relative)
            if expected is None or expected.size != info.size:
                raise ValueError("candidate source archive differs from verified manifest")
            stream = archive.extractfile(info)
            if stream is None:
                raise ValueError("candidate source archive member is unreadable")
            with stream:
                payload = stream.read(info.size + 1)
            if len(payload) != info.size or hashlib.sha256(payload).hexdigest() != expected.sha256:
                raise ValueError("candidate source archive payload differs from verified manifest")
            extracted.append((relative, payload))
    if {relative for relative, _ in extracted} != set(expected_files):
        raise ValueError("candidate source archive files differ from verified manifest")

    destination = workspace / "candidate"
    destination.mkdir()
    try:
        for relative, payload in extracted:
            target = destination.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(payload)
        return destination
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path.cwd())
    parser.add_argument(
        "--git-executable",
        type=Path,
        default=Path(r"C:\Program Files\Git\mingw64\bin\git.exe"),
    )
    parser.add_argument("--require-livekit", action="store_true")
    parser.add_argument("--livekit-executable", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--livekit-executable-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--livekit-pid", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    ownership = (
        arguments.livekit_executable,
        arguments.livekit_executable_sha256,
        arguments.livekit_pid,
    )
    if arguments.require_livekit:
        if any(value is None for value in ownership):
            parser.error("--require-livekit requires verified executable, SHA-256, and PID")
        assert arguments.livekit_executable_sha256 is not None
        assert arguments.livekit_pid is not None
        if re.fullmatch(r"[0-9a-f]{64}", arguments.livekit_executable_sha256) is None:
            parser.error("--livekit-executable-sha256 must be lowercase hexadecimal SHA-256")
        if arguments.livekit_pid < 1:
            parser.error("--livekit-pid must be positive")
    elif any(value is not None for value in ownership):
        parser.error("LiveKit ownership arguments require --require-livekit")
    run_fast_track(
        arguments.candidate,
        git_executable=arguments.git_executable,
        require_livekit=arguments.require_livekit,
        livekit_executable=arguments.livekit_executable,
        livekit_executable_sha256=arguments.livekit_executable_sha256,
        livekit_pid=arguments.livekit_pid,
    )
    print("candidate E2E fast track passed")


if __name__ == "__main__":
    main()
