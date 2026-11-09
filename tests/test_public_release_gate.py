from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_FAST_PROVIDER_TARGET = (
    "tests/providers/test_codex_app_server.py::"
    "test_codex_bound_work_tools_supply_exact_narrow_dynamic_schemas"
)


@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "explicit"])
def test_materialized_gate_forwards_native_git_selection_only_to_full_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    gate = run_path(str(ROOT / "scripts" / "release_gate.py"))["gate_materialized_candidate"]
    selected = tmp_path / "ucrt64" / "bin" / "git.exe" if explicit else None
    commands: list[tuple[str, ...]] = []

    class FullPytestReached(Exception):
        pass

    def observe(*command: str, **kwargs: object) -> None:
        commands.append(command)
        if (
            "pytest" in command
            and not any("tests/providers/" in value for value in command)
        ):
            raise FullPytestReached

    for name in ("reject_ambient_paths", "scan_for_secrets", "validate_disclosure_manifest"):
        monkeypatch.setitem(gate.__globals__, name, lambda root: None)
    monkeypatch.setitem(gate.__globals__, "snapshot_packaged_static", lambda root, target: None)
    monkeypatch.setitem(gate.__globals__, "run", observe)
    with pytest.raises(FullPytestReached):
        gate(tmp_path, livekit=False, git_executable=selected)
    expected = (f"--native-git-executable={selected}",) if explicit else ()
    assert commands[-1] == (
        "uv",
        "run",
        "--frozen",
        "--group",
        "dev",
        "pytest",
        "-q",
        "--durations=0",
        "--durations-min=0.005",
        *expected,
    )
    assert all("--native-git-executable" not in command for command in commands[:-1])
    provider_command = next(
        command for command in commands
        if _FAST_PROVIDER_TARGET in command
    )
    assert commands.index(provider_command) < commands.index(commands[-1])
    assert "tests/providers/test_speech_presence.py" in provider_command
    assert "tests/providers/test_status_authority_boundary.py" in provider_command
    assert "tests/providers/test_status_policy_selection.py" in provider_command
    parsed = pytest.Config.fromdictargs({}, list(commands[-1][6:]))
    try:
        assert parsed.getoption("native_git_executable") == selected
        assert parsed.rootpath == ROOT
    finally:
        parsed._ensure_unconfigure()


def test_native_and_provider_contract_checks_precede_full_pytest_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = run_path(str(ROOT / "scripts" / "release_gate.py"))["gate_materialized_candidate"]
    events: list[tuple[str, object]] = []

    class FullPytestReached(Exception):
        pass

    def observe(*command: str, **kwargs: object) -> None:
        events.append(("command", command))
        if (
            "pytest" in command
            and not any("tests/providers/" in value for value in command)
        ):
            raise FullPytestReached

    def observe_livekit(root, environment, **ownership) -> None:
        events.append(("native", ownership))

    for name in ("reject_ambient_paths", "scan_for_secrets", "validate_disclosure_manifest"):
        monkeypatch.setitem(gate.__globals__, name, lambda root: None)
    monkeypatch.setitem(gate.__globals__, "snapshot_packaged_static", lambda root, target: None)
    monkeypatch.setitem(gate.__globals__, "run", observe)
    monkeypatch.setitem(gate.__globals__, "check_livekit", observe_livekit)
    executable = tmp_path / "livekit-server.exe"

    with pytest.raises(FullPytestReached):
        gate(
            tmp_path,
            livekit=True,
            livekit_executable=executable,
            livekit_executable_sha256="a" * 64,
            livekit_pid=321,
        )

    full_matrix_index = next(
        index
        for index, (kind, value) in enumerate(events)
        if kind == "command"
        and "pytest" in value
        and not any("tests/providers/" in item for item in value)
    )
    provider_index = next(
        index
        for index, (kind, value) in enumerate(events)
        if kind == "command" and _FAST_PROVIDER_TARGET in value
    )
    native_index = next(index for index, (kind, _) in enumerate(events) if kind == "native")
    assert provider_index < full_matrix_index
    provider_command = events[provider_index][1]
    assert "tests/providers/test_speech_presence.py" in provider_command
    assert "tests/providers/test_status_authority_boundary.py" in provider_command
    assert "tests/providers/test_status_policy_selection.py" in provider_command
    assert native_index < full_matrix_index
    assert events[native_index] == (
        "native",
        {
            "executable": executable,
            "expected_sha256": "a" * 64,
            "process_id": 321,
        },
    )


_DISCLOSURE_RESOURCES = (
    "hermes_realtime/evidence/disclosure_v1.txt",
    "hermes_realtime/evidence/disclosure_manifest_v1.json",
    "hermes_realtime/client/static/index.html",
    "hermes_realtime/client/static/assets/app.js",
    "hermes_realtime/client/static/assets/app.js.LEGAL.txt",
    "hermes_realtime/client/static/assets/styles.css",
)


@pytest.mark.parametrize("stale", [False, True], ids=["committed", "stale-asset-hash"])
def test_materialized_gate_validates_committed_disclosure_manifest_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    """Issue #235: ``web/build.mjs`` rewrites the manifest's asset hashes.

    A stale committed manifest must be rejected against the committed packaged
    bytes before ``uv sync`` or any npm step can run, not repaired by the build
    and then validated.
    """

    gate = run_path(str(ROOT / "scripts" / "release_gate.py"))["gate_materialized_candidate"]
    candidate = tmp_path / "candidate"
    for relative in _DISCLOSURE_RESOURCES:
        destination = candidate / "src" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / "src" / relative).read_bytes())
    manifest = candidate / "src" / _DISCLOSURE_RESOURCES[1]
    if stale:
        manifest.write_text(
            re.sub(
                r'("styles\.css":")[0-9a-f]{64}"',
                lambda match: match.group(1) + "0" * 64 + '"',
                manifest.read_text(encoding="utf-8"),
                count=1,
            ),
            encoding="utf-8",
        )
        assert manifest.read_bytes() != (ROOT / "src" / _DISCLOSURE_RESOURCES[1]).read_bytes()
    commands: list[tuple[str, ...]] = []

    class DependencySyncReached(Exception):
        pass

    def observe(*command: str, **kwargs: object) -> None:
        commands.append(command)
        if command[:2] == ("uv", "sync"):
            raise DependencySyncReached

    for name in ("reject_ambient_paths", "scan_for_secrets"):
        monkeypatch.setitem(gate.__globals__, name, lambda root: None)
    monkeypatch.setitem(gate.__globals__, "run", observe)

    if stale:
        with pytest.raises(
            RuntimeError, match="evidence disclosure manifest does not match packaged resources"
        ):
            gate(candidate, livekit=False)
        assert [command[:3] for command in commands] == [("uv", "lock", "--check")]
    else:
        with pytest.raises(DependencySyncReached):
            gate(candidate, livekit=False)
        assert [command[:2] for command in commands] == [("uv", "lock"), ("uv", "sync")]
    assert all(command[0] != gate.__globals__["NPM"] for command in commands)
    assert (tmp_path / "packaged-static-before-web-build").is_dir()


def _run(*command: str, cwd: Path) -> bytes:
    from scripts.release_gate import clean_git_environment

    return subprocess.run(
        command, cwd=cwd, env=clean_git_environment(), check=True, capture_output=True
    ).stdout


def test_public_release_gate_hashes_root_commit_against_empty_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for command in (
        ("git", "init", "-q", "-b", "main"),
        ("git", "config", "user.email", "public-gate@example.invalid"),
        ("git", "config", "user.name", "public-gate"),
    ):
        _run(*command, cwd=repository)
    (repository / "tracked.txt").write_text("public root\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=repository)
    _run("git", "commit", "-qm", "initial public commit", cwd=repository)

    expected_diff = _run(
        "git",
        "diff-tree",
        "--root",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        "--no-commit-id",
        "-r",
        "HEAD",
        cwd=repository,
    )
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))

    assert release_gate["canonical_candidate_diff_sha256"](repository, None) == (
        hashlib.sha256(expected_diff).hexdigest()
    )


def test_public_release_gate_defaults_to_single_parent(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for command in (
        ("git", "init", "-q", "-b", "main"),
        ("git", "config", "user.email", "public-gate@example.invalid"),
        ("git", "config", "user.name", "public-gate"),
    ):
        _run(*command, cwd=repository)
    tracked = repository / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=repository)
    _run("git", "commit", "-qm", "first", cwd=repository)
    baseline = _run("git", "rev-parse", "HEAD", cwd=repository).decode("ascii").strip()
    tracked.write_text("second\n", encoding="utf-8")
    _run("git", "commit", "-qam", "second", cwd=repository)

    expected_diff = _run(
        "git",
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        f"{baseline}..HEAD",
        cwd=repository,
    )
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))

    assert release_gate["canonical_candidate_diff_sha256"](repository, None) == (
        hashlib.sha256(expected_diff).hexdigest()
    )


def test_public_release_gate_rejects_private_deployment_markers() -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    findings = release_gate["public_disclosure_findings"]
    private_user_path = b"C:" + b"/Users/" + b"release-operator/repository"
    private_hardware = b"ACME" + b" USB" + b" DAC"
    private_audio_model = b"X" + b"42 MQA"
    private_reference_host = b"AMD Ryzen 7 " + b"9999X"
    private_gpu_model = b"NVIDIA GeForce RTX " + b"9999"

    assert findings("notes.txt", private_user_path) == ["private machine path: notes.txt"]
    assert findings("notes.txt", private_hardware) == ["private hardware marker: notes.txt"]
    assert findings("notes.txt", private_audio_model) == ["private hardware marker: notes.txt"]
    assert findings("notes.txt", private_reference_host) == [
        "private reference-host marker: notes.txt"
    ]
    assert findings("notes.txt", private_gpu_model) == [
        "private reference-host marker: notes.txt"
    ]
    assert findings(".hermes/plans/internal.md", b"generic") == [
        "private governance path: .hermes/plans/internal.md"
    ]
    assert findings("docs/public.md", b"generic") == []


def test_public_release_gate_allows_documented_path_placeholders() -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    findings = release_gate["public_disclosure_findings"]

    for placeholder in (b"owner", b"me", b"name", b"gate", b"gate-user", b"private"):
        assert findings("tests/example.py", b"C:\\Users\\" + placeholder + b"\\notes.txt") == []


def test_public_release_gate_ignores_hostile_git_repository_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    hostile = tmp_path / "hostile"
    for root in (candidate, hostile):
        root.mkdir()
        _run("git", "init", "-q", "-b", "main", cwd=root)
        _run("git", "config", "user.email", "gate@example.invalid", cwd=root)
        _run("git", "config", "user.name", "gate", cwd=root)
    (candidate / "public.txt").write_text("public\n", encoding="utf-8")
    (hostile / ".env").write_text("TOKEN=redacted\n", encoding="utf-8")
    for root in (candidate, hostile):
        _run("git", "add", ".", cwd=root)
        _run("git", "commit", "-qm", "fixture", cwd=root)
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    monkeypatch.setenv("GIT_DIR", str(hostile / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(hostile))
    release_gate["scan_git_blobs"](candidate)
    archived = tmp_path / "archived"
    archived.mkdir()
    release_gate["git_archive"](
        candidate,
        archived,
        revision=release_gate["_resolved_commit_oid"](candidate, "HEAD", label="candidate HEAD"),
    )

    assert (archived / "public.txt").read_text(encoding="utf-8") == "public\n"
    assert not (archived / ".env").exists()


def test_public_gate_has_no_private_history_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=candidate)
    _run("git", "config", "user.email", "gate@example.invalid", cwd=candidate)
    _run("git", "config", "user.name", "gate", cwd=candidate)
    (candidate / "public.txt").write_text("public\n", encoding="utf-8")
    _run("git", "add", "public.txt", cwd=candidate)
    _run("git", "commit", "-qm", "fixture", cwd=candidate)
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    gate = release_gate["gate"]
    events: list[tuple[object, ...]] = []
    monkeypatch.setitem(
        gate.__globals__,
        "canonical_candidate_diff_sha256",
        lambda source, baseline, **kwargs: events.append(("diff", source, baseline)) or "0" * 64,
    )
    monkeypatch.setitem(
        gate.__globals__,
        "scan_git_blobs",
        lambda source, **kwargs: events.append(("scan", source)),
    )
    monkeypatch.setitem(
        gate.__globals__,
        "git_archive",
        lambda source, destination, *, revision: events.append(
            ("archive", source, destination.name)
        ),
    )
    monkeypatch.setitem(
        gate.__globals__,
        "gate_materialized_candidate",
        lambda root, **arguments: events.append(("materialized", root.name, arguments)),
    )

    gate(candidate, False)

    assert events[0] == ("diff", candidate, None)
    assert events[1] == ("scan", candidate)
    assert events[2][0:2] == ("archive", candidate)
    assert events[3] == (
        "materialized",
        "candidate",
        {
            "livekit": False,
            "livekit_executable": None,
            "livekit_executable_sha256": None,
            "livekit_pid": None,
        },
    )


@pytest.mark.parametrize("stale", [False, True], ids=["current", "stale"])
@pytest.mark.parametrize("override", [None, "UV_PROJECT", "UV_WORKING_DIR", "UV_CONFIG_FILE"])
def test_materialized_gate_requires_current_lock_before_build_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale: bool,
    override: str | None,
) -> None:
    for name in ("UV_PROJECT", "UV_WORKING_DIR", "UV_CONFIG_FILE", "UV_PROJECT_ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
    candidate = tmp_path / "candidate"
    decoy = tmp_path / "decoy"
    for root in (candidate, decoy):
        root.mkdir()
        (root / "pyproject.toml").write_text(
            '[project]\nname = "lock-freshness-fixture"\n'
            'version = "0.1.0"\nrequires-python = ">=3.11,<3.12"\n',
            encoding="utf-8",
        )
        _run("uv", "lock", "--offline", "--no-config", "--python", "3.11", cwd=root)
    project = candidate / "pyproject.toml"
    lock = candidate / "uv.lock"
    committed_lock = lock.read_bytes()
    decoy_lock = (decoy / "uv.lock").read_bytes()
    if stale:
        project.write_text(
            project.read_text(encoding="utf-8").replace('"0.1.0"', '"0.2.0"'),
            encoding="utf-8",
        )
    if override == "UV_CONFIG_FILE":
        config = tmp_path / "ambient-uv.toml"
        config.write_text("this is not valid TOML", encoding="utf-8")
        monkeypatch.setenv(override, str(config))
    elif override is not None:
        monkeypatch.setenv(override, str(decoy))
    gate = run_path(str(ROOT / "scripts" / "release_gate.py"))["gate_materialized_candidate"]
    monkeypatch.setenv("UV_OFFLINE", "1")

    class BuildPreparationReached(Exception):
        pass

    def stop_before_build(root: Path, snapshot: Path) -> None:
        assert root == candidate
        raise BuildPreparationReached

    monkeypatch.setitem(gate.__globals__, "snapshot_packaged_static", stop_before_build)
    expected_error = subprocess.CalledProcessError if stale else BuildPreparationReached
    with pytest.raises(expected_error):
        gate(candidate, livekit=False)

    assert lock.read_bytes() == committed_lock
    assert (decoy / "uv.lock").read_bytes() == decoy_lock
    assert not (candidate / ".venv").exists()
    assert not (decoy / ".venv").exists()


def test_release_gate_prints_the_one_revision_it_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The named revision is the archived revision, and the working tree cannot move it.

    ``docs/contributor-setup-validation.md`` established that the gate validates
    committed bytes and silently excludes the working tree.  Naming the revision
    makes that property observable instead of merely documented.
    """

    repository = tmp_path / "repository"
    repository.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=repository)
    _run("git", "config", "user.email", "gate@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "gate", cwd=repository)
    (repository / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=repository)
    _run("git", "commit", "-qm", "candidate fixture", cwd=repository)
    head = _run("git", "rev-parse", "HEAD", cwd=repository).decode("ascii").strip()
    digest = "0" * 64

    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    gate = release_gate["gate"]
    archived: list[str] = []
    resolutions: list[str] = []
    resolve = release_gate["_resolved_commit_oid"]

    def observed_resolve(source: Path, rev: str, *, label: str) -> str:
        resolutions.append(rev)
        return str(resolve(source, rev, label=label))

    monkeypatch.setitem(gate.__globals__, "_resolved_commit_oid", observed_resolve)
    monkeypatch.setitem(
        gate.__globals__,
        "canonical_candidate_diff_sha256",
        lambda source, baseline, **kwargs: digest,
    )
    monkeypatch.setitem(gate.__globals__, "scan_git_blobs", lambda source, **kwargs: None)
    monkeypatch.setitem(
        gate.__globals__,
        "git_archive",
        lambda source, destination, *, revision: archived.append(revision),
    )
    monkeypatch.setitem(
        gate.__globals__, "gate_materialized_candidate", lambda root, **arguments: None
    )

    gate(repository, False)
    clean = capsys.readouterr().out.splitlines()
    (repository / "tracked.txt").write_text("uncommitted edit\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("uncommitted addition\n", encoding="utf-8")
    gate(repository, False)
    dirty = capsys.readouterr().out.splitlines()

    named = [line for line in clean if head in line]
    assert len(named) == 1, clean
    revision = named[0].split()[-1]
    assert revision == head
    assert re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    # One resolution used twice: the printed value is the archived value.
    assert resolutions == ["HEAD", "HEAD"]
    assert archived == [revision, revision]
    # No explanatory prose beside the OID, and ahead of the digest a contributor
    # cannot look up.
    assert len(named[0].split()) <= 3
    assert clean.index(named[0]) < clean.index(next(line for line in clean if digest in line))
    assert dirty == clean
