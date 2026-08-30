from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", message)
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "task13@example.invalid")
    _git(repository, "config", "user.name", "Task 13")
    (repository / "payload.bin").write_bytes(b"before\x00payload\r\n")
    _commit(repository, "baseline")
    (repository / "payload.bin").write_bytes(b"after\x00payload\n")
    _commit(repository, "candidate")
    return repository


def _orchestrator() -> dict[str, object]:
    return run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "task13_artifact_orchestrator.py")
    )


def test_orchestrator_script_imports_release_gate_when_executed_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "task13_artifact_orchestrator.py"

    completed = subprocess.run(
        (sys.executable, str(script)),
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_verify_clean_tracked_candidate_captures_immutable_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = _orchestrator()
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    expected_head = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    expected_tree = _git(repository, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    expected_diff_sha256 = hashlib.sha256(
        _git(
            repository,
            "diff",
            "--binary",
            "--full-index",
            "--no-renames",
            f"{baseline}..{expected_head}",
        )
    ).hexdigest()

    identity = orchestrator["verify_clean_tracked_candidate"](repository, baseline)

    assert type(identity) is orchestrator["CandidateIdentityV1"]
    assert tuple(identity.__dataclass_fields__) == (
        "candidate_head_oid",
        "candidate_tree_oid",
        "canonical_baseline_oid",
        "canonical_diff_sha256",
    )
    assert identity.candidate_head_oid == expected_head
    assert identity.candidate_tree_oid == expected_tree
    assert identity.canonical_baseline_oid == baseline
    assert identity.canonical_diff_sha256 == expected_diff_sha256


def test_verify_clean_tracked_candidate_defaults_to_parent(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    orchestrator = _orchestrator()
    expected_baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()

    identity = orchestrator["verify_clean_tracked_candidate"](repository)

    assert identity.canonical_baseline_oid == expected_baseline


def test_candidate_identity_ignores_hostile_git_selection_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    candidate = _repository(first_parent)
    hostile = _repository(second_parent)
    baseline = _git(candidate, "rev-parse", "HEAD~1").decode("ascii").strip()
    expected_head = _git(candidate, "rev-parse", "HEAD").decode("ascii").strip()
    monkeypatch.setenv("GIT_DIR", str(hostile / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(hostile))
    monkeypatch.setenv("GIT_INDEX_FILE", str(hostile / ".git" / "index"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(hostile / ".git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(hostile / ".git" / "objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(hostile / ".git" / "objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryformatversion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "1")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/hostile/")
    monkeypatch.setenv("GIT_GRAFT_FILE", str(hostile / ".git" / "info" / "grafts"))
    monkeypatch.setenv("GIT_SHALLOW_FILE", str(hostile / ".git" / "shallow"))
    monkeypatch.setenv("GIT_NAMESPACE", "hostile")
    orchestrator = _orchestrator()

    identity = orchestrator["verify_clean_tracked_candidate"](candidate, baseline)

    assert identity.candidate_head_oid == expected_head


def test_candidate_identity_requires_exact_checkout_root(tmp_path: Path) -> None:
    candidate = _repository(tmp_path)
    nested = candidate / "nested"
    nested.mkdir()
    baseline = _git(candidate, "rev-parse", "HEAD~1").decode("ascii").strip()
    orchestrator = _orchestrator()

    with pytest.raises(RuntimeError, match="checkout root"):
        orchestrator["verify_clean_tracked_candidate"](nested, baseline)


def test_verify_clean_tracked_candidate_rejects_staged_tracked_drift(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    (repository / "payload.bin").write_bytes(b"staged\x00payload\n")
    _git(repository, "add", "payload.bin")
    orchestrator = _orchestrator()

    with pytest.raises(RuntimeError, match="tracked index/worktree drift"):
        orchestrator["verify_clean_tracked_candidate"](repository, baseline)


def test_verify_clean_tracked_candidate_rejects_unstaged_tracked_drift(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    (repository / "payload.bin").write_bytes(b"unstaged\x00payload\n")
    orchestrator = _orchestrator()

    with pytest.raises(RuntimeError, match="tracked index/worktree drift"):
        orchestrator["verify_clean_tracked_candidate"](repository, baseline)


def test_verify_clean_tracked_candidate_ignores_untracked_artifact_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    artifact = repository / "artifacts" / "candidate.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    orchestrator = _orchestrator()

    identity = orchestrator["verify_clean_tracked_candidate"](repository, baseline)

    expected_head = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    assert identity.candidate_head_oid == expected_head


def test_verify_clean_tracked_candidate_rechecks_clean_state_after_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    orchestrator = _orchestrator()
    release_gate = orchestrator["release_gate"]
    canonical_diff = release_gate.canonical_baseline_diff_sha256

    def drift_after_diff(candidate: Path, baseline_oid: str) -> str:
        digest = canonical_diff(candidate, baseline_oid)
        (candidate / "payload.bin").write_bytes(b"drifted\x00payload\n")
        return digest

    monkeypatch.setattr(release_gate, "canonical_baseline_diff_sha256", drift_after_diff)

    with pytest.raises(RuntimeError, match="tracked index/worktree drift"):
        orchestrator["verify_clean_tracked_candidate"](repository, baseline)


def test_verify_clean_tracked_candidate_rechecks_head_after_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    baseline = _git(repository, "rev-parse", "HEAD~1").decode("ascii").strip()
    orchestrator = _orchestrator()
    release_gate = orchestrator["release_gate"]
    canonical_diff = release_gate.canonical_baseline_diff_sha256

    def advance_head_after_diff(candidate: Path, baseline_oid: str) -> str:
        digest = canonical_diff(candidate, baseline_oid)
        (candidate / "payload.bin").write_bytes(b"advanced\x00payload\n")
        _commit(candidate, "advance candidate during identity capture")
        return digest

    monkeypatch.setattr(release_gate, "canonical_baseline_diff_sha256", advance_head_after_diff)

    with pytest.raises(RuntimeError, match="candidate HEAD changed during identity capture"):
        orchestrator["verify_clean_tracked_candidate"](repository, baseline)
