"""Capture a clean, immutable Git candidate identity for Task 13 artifacts."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import release_gate
except ModuleNotFoundError as error:
    if error.name != "release_gate":
        raise
    from scripts import release_gate


@dataclass(frozen=True)
class CandidateIdentityV1:
    candidate_head_oid: str
    candidate_tree_oid: str
    canonical_baseline_oid: str
    canonical_diff_sha256: str


class ArtifactOrchestrationError(RuntimeError):
    """The requested candidate cannot safely identify release artifacts."""


_FULL_OID = re.compile(rb"[0-9a-f]{40}\n")


def _git(candidate: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=candidate,
        env=release_gate.clean_git_environment(),
        capture_output=True,
        check=False,
    )


def _require_checkout(candidate: Path) -> None:
    probe = _git(candidate, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout != b"true\n":
        raise ArtifactOrchestrationError(f"candidate must be a Git checkout: {candidate}")
    top_level = _git(candidate, "rev-parse", "--show-toplevel")
    try:
        resolved_top_level = Path(top_level.stdout[:-1].decode("utf-8", "strict")).resolve(
            strict=True
        )
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactOrchestrationError("could not resolve candidate checkout root") from error
    if top_level.returncode != 0 or resolved_top_level != resolved_candidate:
        raise ArtifactOrchestrationError("candidate must be the exact checkout root")


def _full_oid(candidate: Path, revision: str, *, label: str) -> str:
    resolved = _git(candidate, "rev-parse", "--verify", revision)
    if resolved.returncode != 0 or _FULL_OID.fullmatch(resolved.stdout) is None:
        raise ArtifactOrchestrationError(f"{label} must resolve to a full lowercase Git OID")
    return resolved.stdout[:-1].decode("ascii")


def _require_clean_tracked_state(candidate: Path) -> None:
    status = _git(candidate, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    if status.returncode != 0:
        raise ArtifactOrchestrationError("could not inspect candidate tracked state")
    if status.stdout:
        raise ArtifactOrchestrationError("candidate has tracked index/worktree drift")


def verify_clean_tracked_candidate(
    candidate: Path,
    baseline_commit: str = "HEAD^",
) -> CandidateIdentityV1:
    """Return a stable identity only for a clean candidate Git checkout."""

    _require_checkout(candidate)
    _require_clean_tracked_state(candidate)
    candidate_head_oid = _full_oid(candidate, "HEAD^{commit}", label="candidate HEAD")
    candidate_tree_oid = _full_oid(candidate, "HEAD^{tree}", label="candidate tree")
    canonical_baseline_oid = _full_oid(
        candidate, f"{baseline_commit}^{{commit}}", label="baseline commit"
    )
    try:
        canonical_diff_sha256 = release_gate.canonical_baseline_diff_sha256(
            candidate, canonical_baseline_oid
        )
    except RuntimeError as error:
        raise ArtifactOrchestrationError(str(error)) from error

    _require_clean_tracked_state(candidate)
    if _full_oid(candidate, "HEAD^{commit}", label="candidate HEAD") != candidate_head_oid:
        raise ArtifactOrchestrationError("candidate HEAD changed during identity capture")
    if _full_oid(candidate, "HEAD^{tree}", label="candidate tree") != candidate_tree_oid:
        raise ArtifactOrchestrationError("candidate tree changed during identity capture")
    return CandidateIdentityV1(
        candidate_head_oid=candidate_head_oid,
        candidate_tree_oid=candidate_tree_oid,
        canonical_baseline_oid=canonical_baseline_oid,
        canonical_diff_sha256=canonical_diff_sha256,
    )
