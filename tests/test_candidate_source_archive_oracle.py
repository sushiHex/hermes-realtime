# ruff: noqa: E501
"""Contract tests for the in-memory Task-13 source archive oracle."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
import threading
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = Path(r"C:\Program Files\Git\mingw64\bin\git.exe")
_MEASURED_GIT_IDENTITY = (
    hashlib.sha256(GIT.read_bytes()).hexdigest(),
    subprocess.run((str(GIT), "--version"), check=True, capture_output=True)
    .stdout.rstrip(b"\n")
    .decode("ascii"),
    GIT.stat().st_nlink,
)
_AUTHORIZED_GIT_IDENTITIES = frozenset(
    {
        (
            "d09a1324132aa9da4b4c2b14242dcac893539a735fab288136b101156434a55a",
            "git version 2.53.0.windows.1",
            4,
        ),
        (
            "d1b62b94aa15e5c3bbcdd6440d5f716f78daa2736a951b0f1fad11d38c5f16da",
            "git version 2.55.0.windows.5",
            4,
        ),
        (
            "f83cb7f94f7c714f71ed4976c2c55ab35d80f392881ee586e5a0933424c135e6",
            "git version 2.55.0.windows.4",
            4,
        ),
    }
)
if _MEASURED_GIT_IDENTITY not in _AUTHORIZED_GIT_IDENTITIES:
    raise RuntimeError(f"unapproved Git fixture identity: {_MEASURED_GIT_IDENTITY!r}")
GIT_SHA256, GIT_VERSION, GIT_LINK_COUNT = _MEASURED_GIT_IDENTITY


def _git(*arguments: str, cwd: Path = ROOT) -> bytes:
    return subprocess.run((str(GIT), *arguments), cwd=cwd, check=True, capture_output=True).stdout


def _identity(root: Path, baseline: str) -> object:
    from scripts import candidate_source_archive_oracle as oracle

    head = _git("rev-parse", "HEAD^{commit}", cwd=root).strip().decode("ascii")
    tree = _git("rev-parse", "HEAD^{tree}", cwd=root).strip().decode("ascii")
    base = _git("rev-parse", f"{baseline}^{{commit}}", cwd=root).strip().decode("ascii")
    diff = _git(
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        f"{base}..{head}",
        cwd=root,
    )
    return oracle.CandidateIdentityV1(head, tree, base, hashlib.sha256(diff).hexdigest())


def _pin() -> object:
    from scripts.candidate_source_archive_oracle import GitExecutablePinV1

    assert hashlib.sha256(GIT.read_bytes()).hexdigest() == GIT_SHA256
    assert GIT.stat().st_nlink == GIT_LINK_COUNT
    assert _git("--version") == (GIT_VERSION + "\n").encode("ascii")
    return GitExecutablePinV1(GIT, GIT_SHA256, GIT_VERSION, GIT_LINK_COUNT)


def _repository(tmp_path: Path, attributes: str = "") -> tuple[Path, str]:
    repository = tmp_path / "candidate"
    repository.mkdir(parents=True)
    for command in (
        ("init", "-q"),
        ("config", "user.email", "task13@example.invalid"),
        ("config", "user.name", "task13"),
        ("config", "core.autocrlf", "false"),
    ):
        _git(*command, cwd=repository)
    (repository / "visible.txt").write_text("visible\n", encoding="utf-8")
    (repository / "ignored").mkdir()
    (repository / "ignored" / "hidden.txt").write_text("hidden\n", encoding="utf-8")
    (repository / ".gitattributes").write_text(attributes, encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git("commit", "-qm", "candidate", cwd=repository)
    baseline = _git("rev-parse", "HEAD", cwd=repository).strip().decode("ascii")
    return repository.resolve(), baseline


def test_manifest_order_matches_git_when_file_name_precedes_directory(
    tmp_path: Path,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    evidence = repository / "docs" / "evidence"
    evidence.mkdir(parents=True)
    (repository / "docs" / "evidence-capture.md").write_text("capture\n", encoding="utf-8")
    (evidence / "member.md").write_text("member\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git("commit", "-qm", "add sibling file and directory", cwd=repository)

    identity = _identity(repository, baseline)
    token = oracle.capture_candidate_source_archive(repository, identity, _pin())
    paths = [
        member.path
        for member in oracle.verified_candidate_source_archive_metadata(token).manifest
    ]

    assert paths.index("docs/evidence-capture.md") < paths.index("docs/evidence")
    assert paths.index("docs/evidence") < paths.index("docs/evidence/member.md")


def _archive_inputs(
    tmp_path: Path,
) -> tuple[object, object, tuple[object, ...], dict[str, bytes], int, bytes]:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    identity = _identity(repository, baseline)
    token = oracle.capture_candidate_source_archive(repository, identity, _pin())
    metadata = oracle.verified_candidate_source_archive_metadata(token)
    payloads = {
        member.path: _git("cat-file", "blob", f"HEAD:{member.path}", cwd=repository)
        for member in metadata.manifest
        if member.kind == "file"
    }
    timestamp = int(_git("show", "-s", "--format=%ct", "HEAD", cwd=repository))
    return (
        oracle,
        identity,
        metadata.manifest,
        payloads,
        timestamp,
        oracle._archive_bytes_for_consumer(token, identity),
    )


def _raw_size(header: bytes) -> int:
    return int(header[124:136].rstrip(b"\0 ") or b"0", 8)


def _raw_member_offsets(archive: bytes) -> list[int]:
    offsets: list[int] = []
    offset = 0
    while archive[offset : offset + 512] != b"\0" * 512:
        header = archive[offset : offset + 512]
        if header[156:157] not in {b"g", b"x", b"L", b"K", b"S"}:
            offsets.append(offset)
        offset += 512 + ((_raw_size(header) + 511) // 512) * 512
    return offsets


def _member_span(archive: bytes, offset: int) -> bytes:
    return archive[offset : offset + 512 + ((_raw_size(archive[offset : offset + 512]) + 511) // 512) * 512]


def _termination_offset(archive: bytes) -> int:
    offset = 0
    while archive[offset : offset + 512] != b"\0" * 512:
        offset += 512 + ((_raw_size(archive[offset : offset + 512]) + 511) // 512) * 512
    return offset


def _rewrite_header(archive: bytes, offset: int, mutate: object) -> bytes:
    header = bytearray(archive[offset : offset + 512])
    assert callable(mutate)
    mutate(header)
    header[148:156] = b" " * 8
    checksum = sum(header)
    header[148:156] = f"{checksum:07o}".encode("ascii") + b"\0"
    return archive[:offset] + bytes(header) + archive[offset + 512 :]


def test_oracle_module_exposes_closed_public_api() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    assert oracle.GitExecutablePinV1.__name__ == "GitExecutablePinV1"
    assert oracle.VerifiedCandidateSourceArchiveV1.__name__ == "VerifiedCandidateSourceArchiveV1"
    assert callable(oracle.capture_candidate_source_archive)


def test_release_gate_registers_candidate_oracle_once_by_observing_mypy_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    registered = "scripts/candidate_source_archive_oracle.py"
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def observe_run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
        assert env is not None
        calls.append((command, cwd, env))

    monkeypatch.setitem(release_gate["run_script_mypy"].__globals__, "run", observe_run)
    release_gate["run_script_mypy"](ROOT, {"BASE": "retained"})

    assert tuple(release_gate["required_sdist_paths"]()).count(registered) == 1
    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert cwd == ROOT
    assert command.count(registered) == 1
    assert environment["BASE"] == "retained"
    assert environment["MYPYPATH"] == os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))


def test_pin_requires_exact_positive_link_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    GitExecutablePinV1 = oracle.GitExecutablePinV1
    for link_count in (0, -1, True, 4.0):
        with pytest.raises(ValueError):
            GitExecutablePinV1(GIT, GIT_SHA256, GIT_VERSION, link_count)
    with pytest.raises(ValueError):
        GitExecutablePinV1(Path("git.exe"), GIT_SHA256, GIT_VERSION, 4)
    with pytest.raises(ValueError):
        GitExecutablePinV1(GIT, GIT_SHA256.upper(), GIT_VERSION, 4)
    with pytest.raises(ValueError):
        GitExecutablePinV1(GIT, GIT_SHA256, "", 4)

    monkeypatch.setattr(oracle, "_known_program_files", lambda: Path(r"C:\Program Files"))
    monkeypatch.setattr(oracle, "_require_local_nonreparse_path", lambda _: None)
    with pytest.raises(ValueError, match="canonical|Program Files"):
        GitExecutablePinV1(
            Path(r"C:\Program Files\..\attacker\git.exe"),
            GIT_SHA256,
            GIT_VERSION,
            4,
        )

    assert _pin().link_count == 4


def test_retained_disposable_file_refuses_write_delete_and_replace_via_hardlink(
    tmp_path: Path,
) -> None:
    """Native deny-sharing evidence uses only a disposable copied executable."""
    from scripts import candidate_source_archive_oracle as oracle

    executable = tmp_path / "git.exe"
    executable.write_bytes(GIT.read_bytes())
    alias = tmp_path / "git-hardlink.exe"
    os.link(executable, alias)
    assert os.path.samefile(executable, alias)
    retained = oracle._retained_disposable_file_for_test(executable)
    replacement = tmp_path / "replacement.exe"
    replacement.write_bytes(b"replacement")
    alias_replacement = tmp_path / "alias-replacement.exe"
    alias_replacement.write_bytes(b"replacement")
    try:
        with pytest.raises(PermissionError):
            executable.open("r+b")
        with pytest.raises(PermissionError):
            os.unlink(executable)
        with pytest.raises(PermissionError):
            os.replace(replacement, executable)
        with pytest.raises(PermissionError):
            alias.open("r+b")
        with pytest.raises(PermissionError):
            os.unlink(alias)
        with pytest.raises(PermissionError):
            os.replace(alias_replacement, alias)
    finally:
        retained.close()


def test_git_executor_uses_exact_absolute_child_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    captured: dict[str, object] = {}

    class Retained:
        executable = str(GIT)

        def assert_sealed(self) -> None:
            return None

    class Owner:
        def __init__(
            self, kernel: object, application: str, root: Path, environment: dict[str, str]
        ) -> None:
            captured.update(
                kernel=kernel, application=application, root=root, environment=environment
            )

        def run(self, command: tuple[str, ...], *, stdout_limit: int) -> bytes:
            captured["command"] = command
            captured["stdout_limit"] = stdout_limit
            return b"ok"

    monkeypatch.setattr(oracle, "_NativeGitChildOwner", Owner)
    monkeypatch.setenv("HOME", "host-home")
    monkeypatch.setenv("PYTHONPATH", "host-python")
    monkeypatch.setenv("SYSTEMROOT", r"C:\\hostile-root")
    monkeypatch.setenv("WINDIR", r"C:\\hostile-windir")
    monkeypatch.setenv("GIT_DIR", r"C:\\hostile-git")
    result = oracle._GitExecutor(Retained(), ROOT).run("status", "--porcelain=v1")

    assert result == b"ok"
    command = captured["command"]
    assert command[0] == str(GIT) and command[-2:] == ("status", "--porcelain=v1")
    assert captured["application"] == str(GIT)
    assert captured["root"] == ROOT
    environment = captured["environment"]
    assert isinstance(environment, dict)
    trusted_root, trusted_system32 = oracle._trusted_windows_directories()
    assert environment["PATH"] == os.pathsep.join((str(GIT.parent), trusted_system32))
    assert environment["SystemRoot"] == trusted_root
    assert environment["WINDIR"] == trusted_root
    assert "hostile" not in "\0".join(environment.values())
    assert not {
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "XDG_CONFIG_HOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    } & set(environment)
    assert set(environment) == {
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "PATH",
        "LC_ALL",
        "LANG",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_COUNT",
        "GIT_ATTR_NOSYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "GIT_NO_LAZY_FETCH",
    }


def test_native_git_child_owner_assigns_before_resume_and_finalizes_in_order() -> None:
    """The controller seam proves no Git instruction runs before Job ownership."""
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []

    class Kernel:
        def create_job(self, active_limit: int) -> int:
            assert active_limit == 4
            events.append("job")
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            events.append("stdio")
            return (1, 2, 3, 4, 5, 6)

        def create_suspended(
            self,
            application: str,
            command: tuple[str, ...],
            root: str,
            environment: dict[str, str],
            handles: tuple[int, ...],
        ) -> tuple[int, int]:
            assert application == str(GIT)
            assert command[0] == str(GIT)
            assert root == str(ROOT)
            assert handles == (1, 3, 5)
            assert environment["GIT_CONFIG_GLOBAL"] == "NUL"
            events.append("create-suspended")
            return (7, 8)

        def assign(self, job: int, process: int) -> None:
            assert (job, process) == (10, 7)
            events.append("assign")

        def resume(self, thread: int) -> None:
            assert thread == 8
            events.append("resume")

        def read_bounded(self, read: int, limit: int) -> bytes:
            assert read in {2, 4}
            return b"ok" if read == 2 else b""

        def wait(self, process: int, timeout_ms: int) -> int:
            events.append("wait")
            return 0

        def exit_code(self, process: int) -> int:
            return 0

        def active(self, job: int) -> int:
            events.append("zero-active")
            return 0

        def close(self, handle: int) -> None:
            events.append(f"close-{handle}")

        def terminate(self, job: int) -> None:
            events.append("terminate")

    owner = oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {"GIT_CONFIG_GLOBAL": "NUL"})
    assert owner.run((str(GIT), "--version"), stdout_limit=16) == b"ok"
    assert events.index("create-suspended") < events.index("assign") < events.index("resume")
    assert events.index("wait") < events.index("zero-active") < events.index("close-8")


@pytest.mark.parametrize("result", ["nonzero", "overflow"])
def test_native_git_child_owner_failure_terminates_waits_zeroes_and_closes(result: str) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []

    class Kernel:
        def create_job(self, active_limit: int) -> int:
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            return (1, 2, 3, 4, 5, 6)

        def create_suspended(self, *args: object) -> tuple[int, int]:
            events.append("create-suspended")
            return (7, 8)

        def assign(self, *args: object) -> None:
            events.append("assign")

        def resume(self, *args: object) -> None:
            events.append("resume")

        def read_bounded(self, read: int, limit: int) -> bytes:
            if result == "overflow":
                raise oracle.CandidateSourceArchiveError("overflow")
            return b""

        def wait(self, *args: object) -> int:
            events.append("wait")
            if result == "timeout" and events.count("wait") == 1:
                return 258
            return 0

        def exit_code(self, *args: object) -> int:
            return 1 if result == "nonzero" else 0

        def active(self, *args: object) -> int:
            events.append("zero-active")
            return 0

        def close(self, handle: int) -> None:
            events.append(f"close-{handle}")

        def terminate(self, *args: object) -> None:
            events.append("terminate")

    owner = oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {})
    with pytest.raises(oracle.CandidateSourceArchiveError):
        owner.run((str(GIT), "status"), stdout_limit=1)
    assert (
        events.index("terminate")
        < len(events) - 1 - events[::-1].index("wait")
        < events.index("zero-active")
    )
    assert events.index("zero-active") < events.index("close-8")


def test_native_git_child_owner_preserves_primary_and_cleanup_failure() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    class Kernel:
        def create_job(self, active_limit: int) -> int:
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            return (1, 2, 3, 4, 5, 6)

        def create_suspended(self, *args: object) -> tuple[int, int]:
            return (7, 8)

        def assign(self, *args: object) -> None:
            return None

        def resume(self, *args: object) -> None:
            return None

        def read_bounded(self, *args: object) -> bytes:
            raise oracle.CandidateSourceArchiveError("primary")

        def terminate(self, *args: object) -> None:
            return None

        def wait(self, *args: object) -> int:
            return 0

        def active(self, *args: object) -> int:
            return 0

        def close(self, handle: int) -> None:
            if handle == 10:
                raise oracle.CandidateSourceArchiveError("cleanup")

    with pytest.raises(BaseExceptionGroup) as error:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run(
            (str(GIT), "status"), stdout_limit=1
        )
    assert {str(item) for item in error.value.exceptions} >= {"primary", "cleanup"}


def test_real_clean_candidate_archive_is_opaque_and_candidate_bound(tmp_path: Path) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path, "ignored export-ignore\n")
    identity = _identity(repository, baseline)
    token = oracle.capture_candidate_source_archive(repository, identity, _pin())
    repeated = oracle.capture_candidate_source_archive(repository, identity, _pin())
    metadata = oracle.verified_candidate_source_archive_metadata(token)

    assert type(token) is oracle.VerifiedCandidateSourceArchiveV1
    assert metadata.candidate_head_oid == identity.candidate_head_oid
    assert metadata.prefix == "hermes-realtime-0.0.3"
    assert 0 < metadata.archive_bytes <= 15 * 1024 * 1024
    archive = oracle._archive_bytes_for_consumer(token, identity)
    assert archive == oracle._archive_bytes_for_consumer(repeated, identity)
    assert metadata.archive_sha256 == hashlib.sha256(archive).hexdigest()
    assert metadata.archive_bytes == len(archive)
    assert [member.path for member in metadata.manifest if member.kind == "file"] == [
        ".gitattributes",
        "visible.txt",
    ]
    timestamp = int(_git("show", "-s", "--format=%ct", "HEAD", cwd=repository))
    payloads = {
        path: _git("cat-file", "blob", f"HEAD:{path}", cwd=repository)
        for path in (".gitattributes", "visible.txt")
    }
    oracle._validate_tar(archive, identity, metadata.manifest, payloads, timestamp)
    with pytest.raises(oracle.CandidateSourceArchiveError, match="PAX comment"):
        oracle._parse_raw_tar(archive, "0" * 40, timestamp, metadata.manifest)
    visible = next(member for member in metadata.manifest if member.path == "visible.txt")
    altered = tuple(
        replace(member, tar_mode=0o775) if member == visible else member
        for member in metadata.manifest
    )
    with pytest.raises(oracle.CandidateSourceArchiveError, match="mode header is not canonical"):
        oracle._validate_tar(archive, identity, altered, payloads, timestamp)
    with pytest.raises(oracle.CandidateSourceArchiveError, match="payload differs"):
        oracle._validate_tar(
            archive, identity, metadata.manifest, {**payloads, "visible.txt": b"drift"}, timestamp
        )
    assert not list(repository.rglob("*.tar"))
    assert not hasattr(token, "archive") and not hasattr(token, "path")
    with pytest.raises(TypeError):
        oracle.VerifiedCandidateSourceArchiveV1()
    forged = object.__new__(oracle.VerifiedCandidateSourceArchiveV1)
    with pytest.raises(oracle.CandidateSourceArchiveError):
        oracle._archive_bytes_for_consumer(forged, identity)


def test_candidate_dirty_identity_drift_and_export_subst_fail_closed(tmp_path: Path) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(
        tmp_path, "ignored export-ignore\nignored/hidden.txt export-subst\n"
    )
    identity = _identity(repository, baseline)
    with pytest.raises(oracle.CandidateSourceArchiveError, match="export-subst"):
        oracle.capture_candidate_source_archive(repository, identity, _pin())
    (repository / "visible.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(oracle.CandidateSourceArchiveError, match="tracked index/worktree drift"):
        oracle.capture_candidate_source_archive(repository, identity, _pin())
    clean_repository, clean_baseline = _repository(tmp_path / "clean")
    clean_identity = _identity(clean_repository, clean_baseline)
    drifted = type(clean_identity)(
        clean_identity.candidate_head_oid,
        clean_identity.candidate_tree_oid,
        clean_identity.canonical_baseline_oid,
        "0" * 64,
    )
    with pytest.raises(oracle.CandidateSourceArchiveError, match="identity drifted"):
        oracle.capture_candidate_source_archive(clean_repository, drifted, _pin())


def test_object_hash_and_unsafe_tree_entries_fail_closed() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    class Executor:
        def run(self, *arguments: str, **_: object) -> bytes:
            return b"wrong"

    with pytest.raises(oracle.CandidateSourceArchiveError, match="hash differs"):
        oracle._git_object(Executor(), "blob", "0" * 40)
    symlink = b"120000 link\0" + bytes.fromhex("0" * 40)
    tree_oid = hashlib.sha1(
        b"tree " + str(len(symlink)).encode("ascii") + b"\0" + symlink
    ).hexdigest()

    class TreeExecutor:
        def run(self, *arguments: str, **_: object) -> bytes:
            kind, oid = arguments[1:3]
            assert kind == "tree" and oid == tree_oid
            return symlink

    with pytest.raises(oracle.CandidateSourceArchiveError, match="symlink or gitlink"):
        oracle._walk_tree(TreeExecutor(), tree_oid)


def test_attribute_parser_requires_repeated_path_name_value_triples() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    class Executor:
        def run(self, *arguments: str, **_: object) -> bytes:
            assert arguments[:4] == ("check-attr", "--cached", "-z", "export-ignore")
            return b"file\0export-ignore\0unset\0file\0export-subst\0set\0"

    assert oracle._attribute(Executor(), "file") == ("unset", "set")


def test_raw_tar_rejects_empty_framing() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    with pytest.raises(oracle.CandidateSourceArchiveError, match="invalid raw tar framing"):
        oracle._parse_raw_tar(b"", "0" * 40, 0, ())


def test_o2_member_set_raw_oracle_rejects_missing_and_duplicate_members(tmp_path: Path) -> None:
    oracle, identity, manifest, payloads, timestamp, archive = _archive_inputs(tmp_path)
    offsets = _raw_member_offsets(archive)
    assert len(offsets) == len(manifest)
    missing_offset = offsets[-1]
    missing = archive[:missing_offset] + archive[missing_offset + len(_member_span(archive, missing_offset)) :]
    duplicate = archive[: _termination_offset(archive)] + _member_span(archive, offsets[-1]) + archive[_termination_offset(archive) :]
    for altered in (missing, duplicate):
        with tarfile.open(fileobj=BytesIO(altered), mode="r:") as readable:
            assert readable.getmembers()
        with pytest.raises(oracle.CandidateSourceArchiveError, match="raw member sequence"):
            oracle._validate_tar(altered, identity, manifest, payloads, timestamp)


def test_o2_o4_tracked_drift_after_first_archive_refuses_before_second_archive_or_mint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    identity = _identity(repository, baseline)
    archives: list[tuple[str, ...]] = []
    original_run = oracle._GitExecutor.run
    original_validate = oracle._validate_tar
    monkeypatch.setattr(oracle, "_RECORDS", {})

    def record_archive(self: object, *arguments: str, **kwargs: object) -> bytes:
        if arguments[:1] == ("archive",):
            archives.append(arguments)
        return original_run(self, *arguments, **kwargs)

    def dirty_tracked_file(*arguments: object, **kwargs: object) -> None:
        original_validate(*arguments, **kwargs)
        if len(archives) == 1:
            (repository / "visible.txt").write_text("drifted after archive\n", encoding="utf-8")

    monkeypatch.setattr(oracle._GitExecutor, "run", record_archive)
    monkeypatch.setattr(oracle, "_validate_tar", dirty_tracked_file)
    with pytest.raises(oracle.CandidateSourceArchiveError, match="tracked index/worktree drift"):
        oracle.capture_candidate_source_archive(repository, identity, _pin())
    assert len(archives) == 1
    assert archives[0][-1] == identity.candidate_head_oid
    assert "HEAD" not in archives[0]
    assert oracle._RECORDS == {}


def test_repeated_archive_equality_boundary_refuses_unequal_bytes() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    assert oracle._require_equal_repeated_archives(b"same", b"same") == b"same"
    with pytest.raises(oracle.CandidateSourceArchiveError, match="repeated Git archives differ"):
        oracle._require_equal_repeated_archives(b"first", b"second")


def test_pin_version_and_candidate_identity_fail_before_archive(tmp_path: Path) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    identity = _identity(repository, baseline)
    for forged_pin in (
        replace(_pin(), sha256="0" * 64),
        replace(_pin(), link_count=GIT_LINK_COUNT + 1),
    ):
        with pytest.raises(oracle.CandidateSourceArchiveError, match="seal differs from pin"):
            oracle.capture_candidate_source_archive(repository, identity, forged_pin)
    forged_version = oracle.GitExecutablePinV1(
        GIT, GIT_SHA256, "git version forged", GIT_LINK_COUNT
    )
    with pytest.raises(oracle.CandidateSourceArchiveError, match="version differs"):
        oracle.capture_candidate_source_archive(repository, identity, forged_version)
    for field, malformed in (
        ("candidate_head_oid", "-bad"),
        ("candidate_tree_oid", "A" * 40),
        ("canonical_baseline_oid", "0" * 39),
        ("canonical_diff_sha256", "0" * 63),
    ):
        values = {
            "candidate_head_oid": identity.candidate_head_oid,
            "candidate_tree_oid": identity.candidate_tree_oid,
            "canonical_baseline_oid": identity.canonical_baseline_oid,
            "canonical_diff_sha256": identity.canonical_diff_sha256,
        }
        values[field] = malformed
        malformed_identity = type(identity)(**values)
        with pytest.raises(oracle.CandidateSourceArchiveError, match="identity"):
            oracle.capture_candidate_source_archive(repository, malformed_identity, _pin())


def test_candidate_local_info_attributes_fail_closed(tmp_path: Path) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    identity = _identity(repository, baseline)
    info = repository / ".git" / "info" / "attributes"
    info.parent.mkdir(exist_ok=True)
    info.write_text("visible.txt export-ignore\n", encoding="utf-8")
    with pytest.raises(oracle.CandidateSourceArchiveError, match="info attributes"):
        oracle.capture_candidate_source_archive(repository, identity, _pin())


def test_gitlink_and_primary_child_failure_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    gitlink = b"160000 module\0" + bytes.fromhex("0" * 40)
    tree_oid = hashlib.sha1(
        b"tree " + str(len(gitlink)).encode("ascii") + b"\0" + gitlink
    ).hexdigest()

    class TreeExecutor:
        def run(self, *arguments: str, **_: object) -> bytes:
            assert arguments[1:3] == ("tree", tree_oid)
            return gitlink

    with pytest.raises(oracle.CandidateSourceArchiveError, match="gitlink"):
        oracle._walk_tree(TreeExecutor(), tree_oid)

    class Retained:
        executable = str(GIT)
        calls = 0

        def assert_sealed(self) -> None:
            self.calls += 1
            if self.calls == 2:
                raise oracle.CandidateSourceArchiveError("post-child seal")

    class FailingOwner:
        def __init__(self, *_: object) -> None:
            return None

        def run(self, *_: object, **__: object) -> bytes:
            raise oracle.CandidateSourceArchiveError("primary child failure")

    monkeypatch.setattr(oracle, "_NativeGitChildOwner", FailingOwner)
    with pytest.raises(BaseExceptionGroup) as failure:
        oracle._GitExecutor(Retained(), ROOT).run("status")
    assert {str(error) for error in failure.value.exceptions} >= {
        "primary child failure",
        "post-child seal",
    }


def test_direct_script_import_smoke() -> None:
    completed = subprocess.run(
        (
            os.environ.get("PYTHON", "python"),
            str(ROOT / "scripts" / "candidate_source_archive_oracle.py"),
        ),
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "VIRTUAL_ENV"}
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_assignment_failure_directly_terminates_unassigned_suspended_root() -> None:
    """A failed assignment leaves a suspended root outside the Job."""
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []

    class Kernel:
        def create_job(self, _: int) -> int:
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            return (1, 2, 3, 4, 5, 6)

        def create_suspended(self, *_: object) -> tuple[int, int]:
            return (7, 8)

        def assign(self, *_: object) -> None:
            raise oracle.CandidateSourceArchiveError("assign failed")

        def resume(self, *_: object) -> None:
            events.append("resume")

        def terminate(self, _: int) -> None:
            events.append("terminate-job")

        def terminate_process(self, _: int) -> None:
            events.append("terminate-process")

        def wait(self, _: int, __: int) -> int:
            events.append("wait")
            return oracle._WAIT_OBJECT_0

        def active(self, _: int) -> int:
            events.append("active")
            return 0

        def close(self, _: int) -> None:
            return None

    with pytest.raises(oracle.CandidateSourceArchiveError, match="assign failed") as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run(
            (str(GIT), "status"), stdout_limit=8
        )
    assert type(failure.value) is oracle.CandidateSourceArchiveError
    assert "resume" not in events and "terminate-job" not in events
    assert events[:2] == ["terminate-process", "wait"]


def test_wait_object_zero_with_nonzero_exit_is_not_success() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []

    class Kernel:
        def create_job(self, _: int) -> int:
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            return (1, 2, 3, 4, 5, 6)

        def create_suspended(self, *_: object) -> tuple[int, int]:
            return (7, 8)

        def assign(self, *_: object) -> None:
            return None

        def resume(self, *_: object) -> None:
            return None

        def read_bounded(self, read: int, _: int) -> bytes:
            return b"" if read == 4 else b"ok"

        def wait(self, _: int, __: int) -> int:
            events.append("wait")
            return oracle._WAIT_OBJECT_0

        def exit_code(self, _: int) -> int:
            events.append("exit-code")
            return 1

        def active(self, _: int) -> int:
            return 0

        def terminate(self, _: int) -> None:
            events.append("terminate")

        def terminate_process(self, _: int) -> None:
            return None

        def close(self, _: int) -> None:
            return None

    with pytest.raises(oracle.CandidateSourceArchiveError, match="Git child failed") as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run(
            (str(GIT), "status"), stdout_limit=8
        )
    assert type(failure.value) is oracle.CandidateSourceArchiveError
    assert events[:3] == ["wait", "exit-code", "terminate"]


def test_fixed_drive_requirement_refuses_remote_but_accepts_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    monkeypatch.setattr(oracle, "_attributes", lambda _: 0)
    monkeypatch.setattr(oracle, "_drive_type", lambda _: 4)  # DRIVE_REMOTE
    with pytest.raises(oracle.CandidateSourceArchiveError, match="fixed drive"):
        oracle._require_local_nonreparse_path(Path(r"C:\\fixture\\child"))
    monkeypatch.setattr(oracle, "_drive_type", lambda _: oracle._DRIVE_FIXED)
    oracle._require_local_nonreparse_path(Path(r"C:\\fixture\\child"))


def test_real_standard_linked_worktree_captures_and_routing_file_refuses(tmp_path: Path) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    linked = tmp_path / "linked"
    _git("worktree", "add", "--detach", "-q", str(linked), "HEAD", cwd=repository)
    try:
        identity = _identity(linked, baseline)
        token = oracle.capture_candidate_source_archive(linked.resolve(), identity, _pin())
        assert (
            oracle.verified_candidate_source_archive_metadata(token).candidate_head_oid
            == identity.candidate_head_oid
        )
    finally:
        _git("worktree", "remove", "--force", str(linked), cwd=repository)
    routing = repository / ".git" / "objects" / "info" / "alternates"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text("C:/hostile/objects\n", encoding="ascii")
    with pytest.raises(oracle.CandidateSourceArchiveError, match="alternate, graft, or shallow"):
        oracle.capture_candidate_source_archive(repository, _identity(repository, baseline), _pin())


def test_p5_created_alternates_after_first_archive_refuses_before_second_archive_or_mint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A formerly absent routing control is sealed across both archive captures."""
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    linked = tmp_path / "linked"
    _git("worktree", "add", "--detach", "-q", str(linked), "HEAD", cwd=repository)
    identity = _identity(linked, baseline)
    archive_calls: list[tuple[str, ...]] = []
    original_run = oracle._GitExecutor.run
    original_validate = oracle._validate_tar
    monkeypatch.setattr(oracle, "_RECORDS", {})

    def record_archive(self: object, *arguments: str, **kwargs: object) -> bytes:
        if arguments[:1] == ("archive",):
            archive_calls.append(arguments)
        return original_run(self, *arguments, **kwargs)

    def create_alternates(*arguments: object, **kwargs: object) -> None:
        original_validate(*arguments, **kwargs)
        if len(archive_calls) == 1:
            common = Path(
                _git("rev-parse", "--git-common-dir", cwd=linked).strip().decode("utf-8")
            )
            routing = common / "objects" / "info" / "alternates"
            routing.parent.mkdir(parents=True, exist_ok=True)
            routing.write_text("C:/hostile/objects\n", encoding="ascii")

    monkeypatch.setattr(oracle._GitExecutor, "run", record_archive)
    monkeypatch.setattr(oracle, "_validate_tar", create_alternates)
    try:
        with pytest.raises(oracle.CandidateSourceArchiveError, match="metadata authority drifted"):
            oracle.capture_candidate_source_archive(linked.resolve(), identity, _pin())
    finally:
        _git("worktree", "remove", "--force", str(linked), cwd=repository)
    assert len(archive_calls) == 1
    assert archive_calls[0][-1] == identity.candidate_head_oid
    assert "HEAD" not in archive_calls[0]
    assert oracle._RECORDS == {}


def test_reader_overflow_interrupts_wait_and_keeps_clean_primary_shape() -> None:
    """Reader failure must cut short polling and remain the only failure when clean."""
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []
    reader_started = threading.Event()
    terminated = False

    class Kernel:
        def create_job(self, _: int) -> int:
            return 10

        def create_stdio(self) -> tuple[int, int, int, int, int, int]:
            return (1, 2, 3, 4, 5, 0)

        def create_suspended(self, *_: object) -> tuple[int, int]:
            return (7, 8)

        def assign(self, *_: object) -> None:
            events.append("assign")

        def resume(self, _: int) -> None:
            events.append("resume")

        def read_bounded(self, read: int, _: int) -> bytes:
            if read == 2:
                events.append("reader-overflow")
                reader_started.set()
                raise oracle.CandidateSourceArchiveError("overflow")
            return b""

        def wait(self, _: int, timeout_ms: int) -> int:
            assert timeout_ms <= oracle._WAIT_POLL_MS
            if terminated:
                events.append("wait-signaled")
                return oracle._WAIT_OBJECT_0
            reader_started.wait(0.25)
            events.append("wait-timeout")
            return oracle._WAIT_TIMEOUT

        def terminate(self, _: int) -> None:
            nonlocal terminated
            terminated = True
            events.append("terminate-job")

        def active(self, _: int) -> int:
            events.append("active-zero")
            return 0

        def close(self, handle: int) -> None:
            events.append(f"close-{handle}")

        def exit_code(self, _: int) -> int:
            raise AssertionError("overflow must interrupt before successful exit observation")

    with pytest.raises(oracle.CandidateSourceArchiveError, match="overflow") as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run(
            (str(GIT), "status"), stdout_limit=1
        )
    assert type(failure.value) is oracle.CandidateSourceArchiveError
    assert events.index("reader-overflow") < events.index("terminate-job")
    assert events.index("terminate-job") < events.index("active-zero") < events.index("close-10")


def test_reader_error_interrupts_wait_and_remains_clean_primary() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []
    reader_started = threading.Event()
    terminated = False

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, read: int, _: int) -> bytes:
            if read == 4:
                events.append("reader-error")
                reader_started.set()
                raise oracle.CandidateSourceArchiveError("reader failed")
            return b"ok"
        def wait(self, _: int, __: int) -> int:
            if terminated:
                events.append("cleanup-signaled")
                return oracle._WAIT_OBJECT_0
            reader_started.wait(0.25)
            events.append("poll-timeout")
            return oracle._WAIT_TIMEOUT
        def terminate(self, _: int) -> None:
            nonlocal terminated
            terminated = True
            events.append("terminate-job")
        def active(self, _: int) -> int: return 0
        def close(self, _: int) -> None: return None

    with pytest.raises(oracle.CandidateSourceArchiveError, match="reader failed") as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
    assert type(failure.value) is oracle.CandidateSourceArchiveError
    assert events.index("reader-error") < events.index("terminate-job") < events.index("cleanup-signaled")


def test_timeout_is_primary_and_cleanup_uses_job_then_exact_root_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    monkeypatch.setattr(oracle, "_CHILD_TIMEOUT_SECONDS", 0.01)
    events: list[str] = []
    terminated = False

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, _: int, __: int) -> bytes: return b""
        def wait(self, _: int, timeout_ms: int) -> int:
            assert timeout_ms <= oracle._WAIT_POLL_MS
            events.append("wait")
            return oracle._WAIT_OBJECT_0 if terminated else oracle._WAIT_TIMEOUT
        def terminate(self, _: int) -> None:
            nonlocal terminated
            terminated = True
            events.append("terminate-job")
        def active(self, _: int) -> int:
            events.append("active-zero")
            return 0
        def close(self, _: int) -> None: return None

    with pytest.raises(oracle.CandidateSourceArchiveError, match="process wait timed out") as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
    assert type(failure.value) is oracle.CandidateSourceArchiveError
    assert events.index("terminate-job") < len(events) - 1 - events[::-1].index("wait") < events.index("active-zero")


def test_persistent_descendant_is_primary_then_job_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    monkeypatch.setattr(oracle, "_CHILD_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(oracle, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    events: list[str] = []

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, _: int, __: int) -> bytes: return b""
        def wait(self, _: int, __: int) -> int:
            events.append("wait")
            return oracle._WAIT_OBJECT_0
        def exit_code(self, _: int) -> int:
            events.append("exit-zero")
            return 0
        def active(self, _: int) -> int:
            events.append("active-nonzero")
            return 1
        def terminate(self, _: int) -> None: events.append("terminate-job")
        def close(self, handle: int) -> None: events.append(f"close-{handle}")

    with pytest.raises(BaseExceptionGroup) as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
    assert str(failure.value.exceptions[0]) == "Git Job active-process count is not zero"
    assert events.index("exit-zero") < events.index("active-nonzero") < events.index("terminate-job")
    # Even when zero-active proof fails, closing the Job is the final
    # kill-on-close containment action; the proof failure remains reported.
    assert events.count("close-10") == 1


def test_cleanup_wait_failure_is_preserved_with_primary() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    events: list[str] = []

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, _: int, __: int) -> bytes: return b""
        def wait(self, _: int, __: int) -> int:
            events.append("wait")
            return 0xFFFFFFFF
        def terminate(self, _: int) -> None: events.append("terminate-job")
        def active(self, _: int) -> int: return 0
        def close(self, _: int) -> None: return None

    with pytest.raises(BaseExceptionGroup) as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
    assert str(failure.value.exceptions[0]).endswith("status 4294967295")
    assert any("cleanup wait returned unexpected status 4294967295" in str(item) for item in failure.value.exceptions[1:])
    assert events.index("terminate-job") < len(events) - 1 - events[::-1].index("wait")


def test_live_reader_is_cancelled_then_its_read_handles_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    monkeypatch.setattr(oracle, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    release_reader = threading.Event()
    events: list[str] = []

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, read: int, _: int) -> bytes:
            if read == 2:
                release_reader.wait(1.0)
            return b""
        def wait(self, _: int, __: int) -> int: return oracle._WAIT_OBJECT_0
        def exit_code(self, _: int) -> int: return 1
        def terminate(self, _: int) -> None: events.append("terminate-job")
        def active(self, _: int) -> int: return 0
        def cancel(self, handle: int) -> None:
            events.append(f"cancel-{handle}")
            release_reader.set()
        def close(self, handle: int) -> None: events.append(f"close-{handle}")

    try:
        with pytest.raises(oracle.CandidateSourceArchiveError) as failure:
            oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
        assert str(failure.value) == "Git child failed"
        assert "cancel-2" in events
        assert "close-2" in events and "close-4" in events
    finally:
        release_reader.set()


def test_reader_cancellation_and_close_failures_aggregate_with_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    monkeypatch.setattr(oracle, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    release_reader = threading.Event()

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, read: int, _: int) -> bytes:
            if read == 2:
                release_reader.wait(1.0)
            return b""
        def wait(self, _: int, __: int) -> int: return oracle._WAIT_OBJECT_0
        def exit_code(self, _: int) -> int: return 1
        def terminate(self, _: int) -> None: return None
        def active(self, _: int) -> int: return 0
        def cancel(self, handle: int) -> None:
            if handle == 2:
                raise oracle.CandidateSourceArchiveError("cancel failed")
            release_reader.set()
        def close(self, handle: int) -> None:
            if handle == 2:
                raise oracle.CandidateSourceArchiveError("close failed")

    try:
        with pytest.raises(BaseExceptionGroup) as failure:
            oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run(
                (str(GIT), "status"), stdout_limit=8
            )
        assert [str(error) for error in failure.value.exceptions] == [
            "Git child failed", "cancel failed", "close failed"
        ]
    finally:
        release_reader.set()


def test_primary_and_multiple_cleanup_failures_preserve_flat_order() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    class Kernel:
        def create_job(self, _: int) -> int: return 10
        def create_stdio(self) -> tuple[int, int, int, int, int, int]: return (1, 2, 3, 4, 5, 0)
        def create_suspended(self, *_: object) -> tuple[int, int]: return (7, 8)
        def assign(self, *_: object) -> None: return None
        def resume(self, _: int) -> None: return None
        def read_bounded(self, read: int, _: int) -> bytes:
            if read == 2:
                raise oracle.CandidateSourceArchiveError("primary reader")
            return b""
        def wait(self, _: int, __: int) -> int: return 0xDEAD
        def terminate(self, _: int) -> None: raise oracle.CandidateSourceArchiveError("terminate failed")
        def active(self, _: int) -> int: raise oracle.CandidateSourceArchiveError("active failed")
        def close(self, handle: int) -> None:
            if handle in {2, 4}:
                raise oracle.CandidateSourceArchiveError(f"close failed {handle}")

    with pytest.raises(BaseExceptionGroup) as failure:
        oracle._NativeGitChildOwner(Kernel(), str(GIT), ROOT, {}).run((str(GIT), "status"), stdout_limit=8)
    assert [str(item) for item in failure.value.exceptions] == [
        "primary reader",
        "terminate failed",
        "Git child cleanup wait returned unexpected status 57005",
        "active failed",
        "close failed 2",
        "close failed 4",
    ]


def test_o3_unrepresentable_committed_path_refuses_before_git_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    (repository / "café.txt").write_text("non-ascii\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git("commit", "-qm", "non-ascii", cwd=repository)
    identity = _identity(repository, baseline)
    archives: list[tuple[str, ...]] = []
    original_run = oracle._GitExecutor.run

    def record_archive(self: object, *arguments: str, **kwargs: object) -> bytes:
        if arguments[:1] == ("archive",):
            archives.append(arguments)
        return original_run(self, *arguments, **kwargs)

    monkeypatch.setattr(oracle._GitExecutor, "run", record_archive)
    with pytest.raises(oracle.CandidateSourceArchiveError, match="ASCII USTAR-representable"):
        oracle.capture_candidate_source_archive(repository, identity, _pin())
    assert archives == []


def test_o3_raw_oracle_refuses_canonical_header_padding_extension_and_termination_tampering(
    tmp_path: Path,
) -> None:
    oracle, identity, manifest, payloads, timestamp, archive = _archive_inputs(tmp_path)
    offsets = _raw_member_offsets(archive)
    file_offset = next(offset for offset in offsets if archive[offset + 156 : offset + 157] == b"0")
    canonical = archive[file_offset + 148 : file_offset + 156]
    noncanonical_checksum = bytearray(archive)
    noncanonical_checksum[file_offset + 148 : file_offset + 156] = b" " + canonical[1:]
    name_mismatch = _rewrite_header(archive, file_offset, lambda header: header.__setitem__(0, ord("X")))
    linkname = _rewrite_header(archive, file_offset, lambda header: header.__setitem__(157, ord("x")))
    devmajor = _rewrite_header(archive, file_offset, lambda header: header.__setitem__(329, ord("1")))
    payload_padding = bytearray(archive)
    payload_padding[file_offset + 512 + _raw_size(archive[file_offset : file_offset + 512])] = 1
    later_extension = _rewrite_header(archive, file_offset, lambda header: header.__setitem__(156, ord("x")))
    malformed_termination = bytearray(archive)
    malformed_termination[_termination_offset(archive) + 512] = 1
    for altered in (
        bytes(noncanonical_checksum),
        name_mismatch,
        linkname,
        devmajor,
        bytes(payload_padding),
        later_extension,
        bytes(malformed_termination),
    ):
        with pytest.raises(oracle.CandidateSourceArchiveError):
            oracle._validate_tar(altered, identity, manifest, payloads, timestamp)


def test_o3_canonical_ustar_split_is_deterministic_for_long_paths() -> None:
    from scripts import candidate_source_archive_oracle as oracle

    member = oracle.CandidateArchiveMemberV1(
        "directory/" * 15 + "leaf.txt", "file", "100644", 0o664, 0, hashlib.sha256(b"").hexdigest()
    )
    name, prefix = oracle._canonical_ustar_name_prefix(member)
    assert 1 <= len(name.rstrip(b"\0")) <= 100
    assert 1 <= len(prefix.rstrip(b"\0")) <= 155
    assert prefix.rstrip(b"\0") + b"/" + name.rstrip(b"\0") == (
        oracle._PREFIX.encode("ascii") + b"/" + member.path.encode("ascii")
    )


def test_raw_tar_rejects_nonzero_leading_pax_payload_padding(tmp_path: Path) -> None:
    """PAX framing stays readable, but its exact payload padding is authority."""
    oracle, identity, manifest, payloads, timestamp, archive = _archive_inputs(tmp_path)
    pax_end = 512 + len(oracle._pax_comment(identity.candidate_head_oid))
    altered = bytearray(archive)
    altered[pax_end] = 1
    with tarfile.open(fileobj=BytesIO(altered), mode="r:", ignore_zeros=True) as readable:
        assert readable.getmembers()
    with pytest.raises(oracle.CandidateSourceArchiveError, match="PAX payload padding"):
        oracle._validate_tar(bytes(altered), identity, manifest, payloads, timestamp)


def test_private_retained_handle_seal_rejects_each_frozen_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controlled disposable handle seam never touches the installed Git binary."""
    from scripts import candidate_source_archive_oracle as oracle

    executable = tmp_path / "git.exe"
    executable.write_bytes(b"retained handle authority")
    retained = oracle._retained_git_executable_for_test(executable)
    original = retained._seal_from_handle
    try:
        baseline = original(retained._handle)
        for field, value in (
            ("file_id", baseline.file_id + 1),
            ("final_path", baseline.final_path + "-wrong"),
            ("digest", "0" * 64),
            ("links", baseline.links + 1),
        ):
            monkeypatch.setattr(
                retained,
                "_seal_from_handle",
                lambda _, field=field, value=value: replace(baseline, **{field: value}),
            )
            with pytest.raises(oracle.CandidateSourceArchiveError, match="retained Git executable"):
                retained.assert_sealed()
        monkeypatch.setattr(
            retained,
            "_seal_from_handle",
            lambda handle: baseline if handle == retained._handle else replace(baseline, file_id=baseline.file_id + 1),
        )
        with pytest.raises(oracle.CandidateSourceArchiveError, match="fresh Git executable"):
            retained.assert_sealed()
    finally:
        retained.close()


def test_capture_preserves_primary_and_retained_handle_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer retained authority close is cleanup evidence, never discarded."""
    from scripts import candidate_source_archive_oracle as oracle

    class Retained:
        executable = str(GIT)

        def __init__(self, _: object) -> None:
            pass

        def assert_sealed(self) -> None:
            raise oracle.CandidateSourceArchiveError("capture primary")

        def close(self) -> None:
            raise oracle.CandidateSourceArchiveError("retained close")

    monkeypatch.setattr(oracle, "_RetainedGitExecutableV1", Retained)
    identity = oracle.CandidateIdentityV1("0" * 40, "1" * 40, "2" * 40, "3" * 64)

    with pytest.raises(BaseExceptionGroup) as failure:
        oracle.capture_candidate_source_archive(ROOT, identity, _pin())

    assert [str(error) for error in failure.value.exceptions] == [
        "capture primary",
        "retained close",
    ]


def test_native_job_setup_preserves_configuration_and_close_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially created Job retains both setup and handle-close evidence."""
    import ctypes

    from scripts import candidate_source_archive_oracle as oracle

    class Api:
        @staticmethod
        def CreateJobObjectW(_: object, __: object) -> int:
            return 10

        @staticmethod
        def SetInformationJobObject(*_: object) -> bool:
            return False

    kernel = object.__new__(oracle._NativeWin32GitKernel)
    kernel.api = Api()
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)
    monkeypatch.setattr(
        kernel,
        "close",
        lambda handle: (_ for _ in ()).throw(
            oracle.CandidateSourceArchiveError(f"close job {handle}")
        ),
    )

    with pytest.raises(BaseExceptionGroup) as failure:
        kernel.create_job(4)

    assert [str(error) for error in failure.value.exceptions] == [
        "could not configure Git Job Object (Win32 error 5)",
        "close job 10",
    ]


def test_partial_stdio_setup_preserves_primary_and_every_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-invisible partial stdio handles cannot lose cleanup evidence."""
    import ctypes
    from ctypes import wintypes

    from scripts import candidate_source_archive_oracle as oracle

    class Api:
        @staticmethod
        def CreateFileW(*_: object) -> int:
            return 1

        @staticmethod
        def SetHandleInformation(*_: object) -> bool:
            return True

        @staticmethod
        def CreatePipe(read: object, write: object, *_: object) -> bool:
            ctypes.cast(read, ctypes.POINTER(wintypes.HANDLE)).contents.value = 2
            ctypes.cast(write, ctypes.POINTER(wintypes.HANDLE)).contents.value = 3
            return False

    kernel = object.__new__(oracle._NativeWin32GitKernel)
    kernel.api = Api()
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6)
    monkeypatch.setattr(
        kernel,
        "close",
        lambda handle: (_ for _ in ()).throw(
            oracle.CandidateSourceArchiveError(f"close stdio {handle}")
        ),
    )

    with pytest.raises(BaseExceptionGroup) as failure:
        kernel.create_stdio()

    assert [str(error) for error in failure.value.exceptions] == [
        "could not create Git stdout pipe (Win32 error 6)",
        "close stdio 1",
        "close stdio 2",
        "close stdio 3",
    ]


def test_candidate_revalidation_uses_exact_no_ext_diff_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate identity uses the same canonical raw diff argv as capture."""
    from scripts import candidate_source_archive_oracle as oracle

    repository, baseline = _repository(tmp_path)
    identity = _identity(repository, baseline)
    observed: list[tuple[str, ...]] = []
    original = oracle._GitExecutor.run

    def record(
        self: object, *arguments: str, stdout_limit: int = oracle._MAX_ARCHIVE_BYTES
    ) -> bytes:
        observed.append(arguments)
        return original(self, *arguments, stdout_limit=stdout_limit)

    monkeypatch.setattr(oracle._GitExecutor, "run", record)
    oracle.capture_candidate_source_archive(repository, identity, _pin())

    assert (
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        f"{identity.canonical_baseline_oid}..{identity.candidate_head_oid}",
    ) in observed
