from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_git_installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts import candidate_e2e_fast_track as fast_track

    program_files = tmp_path / "native-program-files"
    program_files.mkdir()
    validations, invocations = [], []
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle,
        "_known_program_files",
        lambda: program_files,
    )
    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle,
        "_validate_public_git_location",
        lambda path: validations.append(path),
    )

    def version(command, **kwargs):
        invocations.append(command)
        assert validations == [Path(command[0])]
        return SimpleNamespace(stdout=b"git version 2.56.0.windows.2\n")

    monkeypatch.setattr(fast_track.subprocess, "run", version)
    monkeypatch.setattr(
        fast_track,
        "GitExecutablePinV1",
        lambda path, sha256, version, links: SimpleNamespace(
            path=path, sha256=sha256, version=version, link_count=links
        ),
    )
    return program_files, validations, invocations


@pytest.mark.parametrize("layout", ["mingw64", "ucrt64"])
def test_default_git_measures_the_unique_native_image(native_git_installation, layout: str) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    program_files, validations, invocations = native_git_installation
    executable = program_files / "Git" / layout / "bin/git.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic native Git image")

    pin = fast_track._measure_git_pin(None)

    assert pin.path == executable
    assert pin.sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert pin.version == "git version 2.56.0.windows.2"
    assert pin.link_count == executable.stat().st_nlink
    assert validations == [executable]
    assert invocations == [(str(executable), "--version")]


@pytest.mark.parametrize("layouts", [(), ("mingw64", "ucrt64")], ids=["missing", "ambiguous"])
def test_default_git_refuses_missing_or_ambiguous_images(
    native_git_installation, layouts: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    program_files, validations, invocations = native_git_installation
    for layout in layouts:
        executable = program_files / "Git" / layout / "bin/git.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"synthetic image")

    with pytest.raises(ValueError, match="--git-executable"):
        fast_track._measure_git_pin(None)

    assert validations == invocations == []
    output = capsys.readouterr().out
    assert output.startswith("[native-git-discovery] ") and output.count("\n") == 1
    marker = json.loads(output.removeprefix("[native-git-discovery] "))
    assert marker == {
        "version": 1,
        "candidates": len(layouts),
        "category": "ambiguous" if layouts else "missing",
    }
    assert str(program_files) not in output


def test_default_git_ignores_environment_and_launcher_decoys(
    native_git_installation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    program_files, _, invocations = native_git_installation
    for executable in (
        tmp_path / "path/git.exe",
        tmp_path / "environment-program-files/Git/ucrt64/bin/git.exe",
        program_files / "Git/cmd/git.exe",
        program_files / "Git/bin/git.exe",
    ):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"synthetic decoy")
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "environment-program-files"))
    with pytest.raises(ValueError, match="--git-executable"):
        fast_track._measure_git_pin(None)
    assert invocations == []


def test_default_git_refuses_native_location_validation_before_execution(
    native_git_installation, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    program_files, _, invocations = native_git_installation
    executable = program_files / "Git/ucrt64/bin/git.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic image")

    def refused(path):
        raise ValueError("synthetic native authority refusal")

    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle, "_validate_public_git_location", refused
    )
    with pytest.raises(ValueError, match="synthetic native authority refusal"):
        fast_track._measure_git_pin(None)
    assert invocations == []
    marker = json.loads(capsys.readouterr().out.removeprefix("[native-git-discovery] "))
    assert marker == {"version": 1, "candidates": 1, "category": "location_refused"}


def test_default_git_preserves_native_authority_refusal(
    native_git_installation, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    _, validations, invocations = native_git_installation

    def refused():
        raise fast_track.candidate_source_archive_oracle.CandidateSourceArchiveError(
            "synthetic native authority refusal"
        )

    monkeypatch.setattr(fast_track.candidate_source_archive_oracle, "_known_program_files", refused)
    with pytest.raises(
        fast_track.candidate_source_archive_oracle.CandidateSourceArchiveError,
        match="synthetic native authority refusal",
    ):
        fast_track._measure_git_pin(None)
    assert validations == invocations == []
    marker = json.loads(capsys.readouterr().out.removeprefix("[native-git-discovery] "))
    assert marker == {"version": 1, "candidates": 0, "category": "authority_refused"}


def test_explicit_git_path_is_never_replaced_by_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    def no_discovery():
        raise AssertionError("explicit image must not use discovery")

    monkeypatch.setattr(
        fast_track.candidate_source_archive_oracle, "_known_program_files", no_discovery
    )
    missing = tmp_path / "missing/git.exe"
    with pytest.raises(FileNotFoundError):
        fast_track._measure_git_pin(missing)
    executable = tmp_path / "selected/git.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"explicit image")
    observed = []
    monkeypatch.setattr(
        fast_track.subprocess,
        "run",
        lambda command, **kwargs: (
            observed.append(command) or SimpleNamespace(stdout=b"git version 2.55.0.windows.5\n")
        ),
    )
    monkeypatch.setattr(fast_track, "GitExecutablePinV1", lambda *fields: fields)
    pin = fast_track._measure_git_pin(executable)
    assert pin == (
        executable,
        hashlib.sha256(b"explicit image").hexdigest(),
        "git version 2.55.0.windows.5",
        executable.stat().st_nlink,
    )
    assert observed == [(str(executable), "--version")]


def test_git_version_probe_is_bounded_noninteractive_and_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    executable = tmp_path / "git.exe"
    executable.write_bytes(b"synthetic native image")
    observed = []
    monkeypatch.setattr(
        fast_track.subprocess,
        "run",
        lambda command, **kwargs: (
            observed.append((command, kwargs))
            or SimpleNamespace(stdout=b"git version 2.56.0.windows.2\n")
        ),
    )
    monkeypatch.setattr(fast_track, "GitExecutablePinV1", lambda *fields: fields)

    fast_track._measure_git_pin(executable)

    assert observed == [
        (
            (str(executable), "--version"),
            {
                "capture_output": True,
                "check": True,
                "stdin": fast_track.subprocess.DEVNULL,
                "timeout": 30.0,
                "creationflags": fast_track.subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            },
        )
    ]


def test_git_version_probe_preserves_timeout_with_bounded_final_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    executable = tmp_path / "git.exe"
    executable.write_bytes(b"synthetic native image")
    failure = fast_track.subprocess.TimeoutExpired((str(executable), "--version"), 30.0)

    def expired(*args, **kwargs):
        raise failure

    monkeypatch.setattr(fast_track.subprocess, "run", expired)
    with pytest.raises(fast_track.subprocess.TimeoutExpired) as observed:
        fast_track._measure_git_pin(executable)
    assert observed.value is failure
    output = capsys.readouterr().out
    assert output.startswith("[native-git-probe] ") and output.count("\n") == 1
    assert json.loads(output.removeprefix("[native-git-probe] ")) == {
        "version": 1,
        "calls": 1,
        "category": "timeout",
    }
    assert str(executable) not in output


_GIT_CONSUMERS = (
    "candidate_e2e_fast_track",
    "qualify_capacity_rollover",
    "qualify_deterministic_equivalence",
    "qualify_full_purge_cleanup",
    "qualify_over_budget_turn",
    "qualify_owned_close_faults",
    "qualify_revoke_race",
    "qualify_spool_crash_matrix",
    "qualify_synthetic_fault_matrix",
)


@pytest.mark.parametrize("consumer", _GIT_CONSUMERS)
@pytest.mark.parametrize("explicit", [False, True], ids=["discover", "explicit"])
def test_git_cli_consumers_deliver_the_measured_image_to_the_archive_oracle(
    native_git_installation,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    explicit: bool,
) -> None:
    from scripts import candidate_e2e_fast_track as fast_track

    module = importlib.import_module("scripts." + consumer)
    program_files, validations, invocations = native_git_installation
    executable = program_files / "Git/ucrt64/bin/git.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic selected image")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    arguments = [consumer, "--candidate", str(candidate)]
    if explicit:
        arguments += ["--git-executable", str(executable)]
        # The existing explicit-path pin validates authority in its constructor.
        validations.append(executable)
    if consumer not in {
        "candidate_e2e_fast_track",
        "qualify_full_purge_cleanup",
        "qualify_spool_crash_matrix",
        "qualify_synthetic_fault_matrix",
    }:
        arguments += ["--livekit-executable", "livekit-server.exe", "--livekit-sha256", "a" * 64]
    if consumer not in {"candidate_e2e_fast_track", "qualify_deterministic_equivalence"}:
        arguments += ["--candidate-wheel", "candidate.whl", "--wheel-sha256", "b" * 64]
    monkeypatch.setattr(sys, "argv", arguments)
    identity, captured = object(), []

    class ArchiveReached(Exception):
        pass

    def capture(root, observed_identity, pin):
        assert root == candidate and observed_identity is identity
        captured.append(pin)
        raise ArchiveReached

    if consumer == "candidate_e2e_fast_track":
        monkeypatch.setattr(
            fast_track.task13_artifact_orchestrator,
            "verify_clean_tracked_candidate",
            lambda *args: identity,
        )
        monkeypatch.setattr(fast_track.release_gate, "scan_git_blobs", lambda *args: None)
        monkeypatch.setattr(
            fast_track.candidate_source_archive_oracle, "capture_candidate_source_archive", capture
        )
    else:
        monkeypatch.setattr(module, "verify_clean_tracked_candidate", lambda *args: identity)
        monkeypatch.setattr(module, "capture_candidate_source_archive", capture)
    with pytest.raises(ArchiveReached):
        module.main()
    assert len(captured) == 1
    assert captured[0].path == executable
    assert captured[0].sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert invocations == [(str(executable), "--version")]


def _archive() -> tuple[bytes, object]:
    from scripts import candidate_e2e_fast_track as fast_track

    CandidateArchiveMemberV1 = fast_track.candidate_source_archive_oracle.CandidateArchiveMemberV1
    CandidateSourceArchiveMetadataV1 = fast_track.CandidateSourceArchiveMetadataV1

    prefix = "hermes-realtime-0.0.3"
    payload = b'[project]\nname = "fixture"\n'
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
    from types import SimpleNamespace

    pin = SimpleNamespace(path=executable)
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
                "git_executable": executable,
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


@pytest.mark.parametrize(
    ("layout", "digest", "version"),
    [
        (
            "mingw64",
            "d1b62b94aa15e5c3bbcdd6440d5f716f78daa2736a951b0f1fad11d38c5f16da",
            "git version 2.55.0.windows.5",
        ),
        (
            "ucrt64",
            "b824434dfe3b5f9cf648af6c9cc8cb1f6a982cc6e13b6c6ebbe6533c7e51cc2e",
            "git version 2.56.0.windows.2",
        ),
    ],
    ids=["legacy", "current"],
)
def test_oracle_fixture_uses_shared_native_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    digest: str,
    version: str,
) -> None:
    from runpy import run_path
    from types import SimpleNamespace

    from scripts import candidate_e2e_fast_track as fast_track

    executable = tmp_path / layout / "bin" / "git.exe"
    observed: list[object] = []

    def measure(path: object) -> object:
        observed.append(path)
        return SimpleNamespace(path=executable, sha256=digest, version=version, link_count=4)

    def refuse_unshared_probe(*args: object, **kwargs: object) -> object:
        raise AssertionError("fixture bypassed shared native discovery")

    monkeypatch.setattr(fast_track, "_measure_git_pin", measure)
    monkeypatch.setattr(fast_track.subprocess, "run", refuse_unshared_probe)
    namespace = run_path(str(Path(__file__).with_name("test_candidate_source_archive_oracle.py")))
    pin = namespace["_initialize_git_fixture"](None)

    assert observed == [None]
    assert pin.path == executable
    assert (pin.sha256, pin.version, pin.link_count) == (digest, version, 4)


@pytest.mark.parametrize("changed", ["sha256", "version", "link_count"])
def test_oracle_fixture_retains_exact_image_identity_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    from runpy import run_path
    from types import SimpleNamespace

    from scripts import candidate_e2e_fast_track as fast_track

    pin = SimpleNamespace(
        path=tmp_path / "ucrt64" / "bin" / "git.exe",
        sha256="b824434dfe3b5f9cf648af6c9cc8cb1f6a982cc6e13b6c6ebbe6533c7e51cc2e",
        version="git version 2.56.0.windows.2",
        link_count=4,
    )
    setattr(
        pin,
        changed,
        {"sha256": "0" * 64, "version": "git version 2.56.0.windows.3", "link_count": 5}[changed],
    )
    monkeypatch.setattr(fast_track, "_measure_git_pin", lambda path: pin)
    namespace = run_path(str(Path(__file__).with_name("test_candidate_source_archive_oracle.py")))
    with pytest.raises(RuntimeError, match="unapproved Git fixture identity"):
        namespace["_initialize_git_fixture"](None)


def test_oracle_fixture_explicit_selection_survives_ambiguous_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from runpy import run_path
    from types import SimpleNamespace

    from scripts import candidate_e2e_fast_track as fast_track

    selected = tmp_path / "ucrt64" / "bin" / "git.exe"
    calls: list[object] = []

    def measure(path: object) -> object:
        calls.append(path)
        if path is None:
            raise ValueError("ambiguous native images")
        assert path == selected
        return SimpleNamespace(
            path=selected,
            sha256="b824434dfe3b5f9cf648af6c9cc8cb1f6a982cc6e13b6c6ebbe6533c7e51cc2e",
            version="git version 2.56.0.windows.2",
            link_count=4,
        )

    monkeypatch.setattr(fast_track, "_measure_git_pin", measure)
    namespace = run_path(str(Path(__file__).with_name("test_candidate_source_archive_oracle.py")))

    def selected_option(name: str) -> Path:
        assert name == "native_git_executable"
        return selected

    fixture = namespace["_selected_git_fixture"].__wrapped__
    fixture(SimpleNamespace(getoption=selected_option))
    assert fixture.__globals__["GIT"] == selected
    assert calls == [selected]
