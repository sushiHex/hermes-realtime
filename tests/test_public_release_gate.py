from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*command: str, cwd: Path) -> bytes:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True).stdout


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
    release_gate["git_archive"](candidate, archived)

    assert (archived / "public.txt").read_text(encoding="utf-8") == "public\n"
    assert not (archived / ".env").exists()


def test_public_gate_has_no_private_history_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_gate = run_path(str(ROOT / "scripts" / "release_gate.py"))
    gate = release_gate["gate"]
    events: list[tuple[object, ...]] = []
    monkeypatch.setitem(
        gate.__globals__,
        "canonical_candidate_diff_sha256",
        lambda source, baseline: events.append(("diff", source, baseline)) or "0" * 64,
    )
    monkeypatch.setitem(
        gate.__globals__,
        "scan_git_blobs",
        lambda source: events.append(("scan", source)),
    )
    monkeypatch.setitem(
        gate.__globals__,
        "git_archive",
        lambda source, destination: events.append(("archive", source, destination.name)),
    )
    monkeypatch.setitem(
        gate.__globals__,
        "gate_materialized_candidate",
        lambda root, **arguments: events.append(("materialized", root.name, arguments)),
    )

    gate(tmp_path, False)

    assert events[0] == ("diff", tmp_path, None)
    assert events[1] == ("scan", tmp_path)
    assert events[2][0:2] == ("archive", tmp_path)
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
