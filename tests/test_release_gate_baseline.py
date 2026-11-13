"""Explicit baseline authority for the committed-candidate release gate (issue #324)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import release_gate as gate

_MARKER = "[release-gate-baseline] "
_DIGEST_LINE = "canonical candidate diff SHA-256: "


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()


def _commit(root: Path, name: str, payload: bytes, message: str) -> str:
    (root / name).write_bytes(payload)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


class _MergeFixture:
    """A branch refreshed by an ordinary merge of its target, as in issue #324."""

    def __init__(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "fixture@example.invalid")
        _git(root, "config", "user.name", "Fixture")
        self.root = root
        self.fork = _commit(root, "shared.py", b"shared\n", "fork point")
        target = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        _git(root, "checkout", "-q", "-b", "topic")
        self.topic = _commit(root, "topic.py", b"topic\n", "topic change")
        _git(root, "checkout", "-q", target)
        self.target = _commit(root, "target.py", b"target\n", "target change")
        _git(root, "checkout", "-q", "topic")
        _git(root, "merge", "-q", "--no-ff", "--no-edit", target)
        self.merge = _git(root, "rev-parse", "HEAD")
        assert _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split() == [
            self.merge,
            self.topic,
            self.target,
        ]
        assert _git(root, "status", "--porcelain") == ""

    def squash_onto(self, parent: str) -> str:
        """A single-parent commit carrying the merge candidate's exact tree."""
        tree = _git(self.root, "rev-parse", f"{self.merge}^{{tree}}")
        return _git(self.root, "commit-tree", tree, "-p", parent, "-m", "linear equivalent")


@pytest.fixture
def merged(tmp_path: Path) -> _MergeFixture:
    return _MergeFixture(tmp_path)


@pytest.fixture
def later_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every gate stage that follows the diff binding, without running it."""
    reached: list[str] = []
    monkeypatch.setattr(
        gate, "scan_git_blobs", lambda source, *, revision: reached.append("scan")
    )
    monkeypatch.setattr(
        gate,
        "git_archive",
        lambda source, destination, *, revision: reached.append("archive"),
    )
    monkeypatch.setattr(
        gate, "gate_materialized_candidate", lambda *args, **kwargs: reached.append("gate")
    )
    return reached


def _markers(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(_MARKER))
        for line in output.splitlines()
        if line.startswith(_MARKER)
    ]


def _printed_digest(output: str) -> str:
    (line,) = [line for line in output.splitlines() if line.startswith(_DIGEST_LINE)]
    return line.removeprefix(_DIGEST_LINE)


def test_gate_refuses_merge_candidate_without_baseline_before_any_stage(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(RuntimeError, match="a merge candidate requires an explicit baseline"):
        gate.gate(merged.root, False)
    output = capsys.readouterr().out
    assert later_stages == []
    assert _DIGEST_LINE not in output
    assert _markers(output) == [
        {"version": 1, "category": "merge_requires_baseline", "explicit": False}
    ]


def test_gate_binds_merge_candidate_to_its_explicit_baseline(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    gate.gate(merged.root, False, baseline=merged.target)
    output = capsys.readouterr().out
    assert later_stages == ["scan", "archive", "gate"]
    assert _markers(output) == []
    # The binding is exactly the one a single-parent commit with the same tree on
    # the same baseline already receives, and is not the other parent's binding.
    linear = gate.canonical_candidate_diff_sha256(
        merged.root, None, revision=merged.squash_onto(merged.target)
    )
    other = gate.canonical_candidate_diff_sha256(
        merged.root, None, revision=merged.squash_onto(merged.topic)
    )
    assert linear != other
    assert _printed_digest(output) == linear


def test_gate_binds_each_merge_parent_only_when_named(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    gate.gate(merged.root, False, baseline=merged.topic)
    assert _printed_digest(capsys.readouterr().out) == gate.canonical_candidate_diff_sha256(
        merged.root, None, revision=merged.squash_onto(merged.topic)
    )


def _abbreviated(fixture: _MergeFixture) -> object:
    return fixture.target[:12]


def _uppercase(fixture: _MergeFixture) -> object:
    return fixture.target.upper()


def _symbolic_parent(fixture: _MergeFixture) -> object:
    return "HEAD^2"


def _trailing_newline(fixture: _MergeFixture) -> object:
    return fixture.target + "\n"


def _peel_suffix(fixture: _MergeFixture) -> object:
    return fixture.target + "^{commit}"


def _bytes(fixture: _MergeFixture) -> object:
    return fixture.target.encode("ascii")


def _absent_object(fixture: _MergeFixture) -> object:
    return "0123456789abcdef0123456789abcdef01234567"


def _tree_object(fixture: _MergeFixture) -> object:
    return _git(fixture.root, "rev-parse", f"{fixture.target}^{{tree}}")


def _annotated_tag_object(fixture: _MergeFixture) -> object:
    _git(fixture.root, "tag", "-a", "annotated", "-m", "annotated", fixture.target)
    tag = _git(fixture.root, "rev-parse", "refs/tags/annotated")
    assert tag != fixture.target
    return tag


def _candidate_itself(fixture: _MergeFixture) -> object:
    return fixture.merge


def _unrelated_commit(fixture: _MergeFixture) -> object:
    # Same tree, valid commit, present in the object store, but not in the
    # candidate's history.
    return fixture.squash_onto(fixture.target)


@pytest.mark.parametrize(
    ("baseline", "category", "message"),
    [
        (_abbreviated, "malformed", "full lowercase Git commit OID"),
        (_uppercase, "malformed", "full lowercase Git commit OID"),
        (_symbolic_parent, "malformed", "full lowercase Git commit OID"),
        (_trailing_newline, "malformed", "full lowercase Git commit OID"),
        (_peel_suffix, "malformed", "full lowercase Git commit OID"),
        (_bytes, "malformed", "full lowercase Git commit OID"),
        (_absent_object, "unresolved", "exact OID of an existing commit"),
        (_tree_object, "unresolved", "exact OID of an existing commit"),
        (_annotated_tag_object, "unresolved", "exact OID of an existing commit"),
        (_candidate_itself, "is_candidate", "must differ from candidate HEAD"),
        (_unrelated_commit, "not_ancestor", "must be an ancestor of candidate HEAD"),
    ],
)
def test_gate_refuses_inexact_or_unrelated_baseline_before_any_stage(
    merged: _MergeFixture,
    later_stages: list[str],
    capsys: pytest.CaptureFixture[str],
    baseline,
    category: str,
    message: str,
) -> None:
    supplied = baseline(merged)
    capsys.readouterr()
    with pytest.raises(RuntimeError, match=message):
        gate.gate(merged.root, False, baseline=supplied)
    output = capsys.readouterr().out
    assert later_stages == []
    assert _DIGEST_LINE not in output
    assert _markers(output) == [{"version": 1, "category": category, "explicit": True}]
    marker_lines = [line for line in output.splitlines() if line.startswith(_MARKER)]
    assert len(marker_lines) == 1 and len(marker_lines[0]) < 128
    assert str(supplied)[:12] not in marker_lines[0]


def test_single_parent_candidate_keeps_its_implicit_parent_binding(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    _git(merged.root, "checkout", "-q", "--detach", merged.topic)
    gate.gate(merged.root, False)
    output = capsys.readouterr().out
    assert later_stages == ["scan", "archive", "gate"]
    assert _markers(output) == []
    implicit = _printed_digest(output)
    # Naming the only parent states the same binding; it never changes it.
    gate.gate(merged.root, False, baseline=merged.fork)
    assert _printed_digest(capsys.readouterr().out) == implicit


def test_single_parent_candidate_accepts_an_earlier_explicit_baseline(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    _git(merged.root, "checkout", "-q", "--detach", merged.topic)
    tip = _commit(merged.root, "more.py", b"more\n", "second topic change")
    gate.gate(merged.root, False, baseline=merged.fork)
    explicit = _printed_digest(capsys.readouterr().out)
    assert explicit == gate.canonical_candidate_diff_sha256(merged.root, merged.fork)
    assert explicit != gate.canonical_candidate_diff_sha256(merged.root, None, revision=tip)


def test_root_candidate_stays_valid_and_admits_no_baseline(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    _git(merged.root, "checkout", "-q", "--detach", merged.fork)
    gate.gate(merged.root, False)
    output = capsys.readouterr().out
    assert later_stages == ["scan", "archive", "gate"]
    assert _printed_digest(output) == gate.canonical_candidate_diff_sha256(merged.root, None)
    later_stages.clear()
    with pytest.raises(RuntimeError, match="must be an ancestor of candidate HEAD"):
        gate.gate(merged.root, False, baseline=merged.target)
    assert later_stages == []
    assert _markers(capsys.readouterr().out) == [
        {"version": 1, "category": "not_ancestor", "explicit": True}
    ]


def test_unrelated_failure_emits_no_baseline_marker(
    merged: _MergeFixture, later_stages: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    (merged.root / ".git" / "info" / "attributes").write_text("*.py -diff\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="local info attributes"):
        gate.gate(merged.root, False, baseline=merged.target)
    assert _markers(capsys.readouterr().out) == []


def test_baseline_refusal_category_is_closed() -> None:
    with pytest.raises(ValueError, match="unknown baseline refusal category"):
        gate.BaselineRefusal("anything_else", "message")
    assert issubclass(gate.BaselineRefusal, RuntimeError)


@pytest.mark.parametrize("supplied", [True, False])
def test_cli_propagates_the_baseline_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supplied: bool
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(gate, "gate", lambda *args, **kwargs: calls.append((args, kwargs)))
    baseline = "89abcdef0123456789abcdef0123456789abcdef"
    arguments = ["release_gate.py", "--candidate", str(tmp_path)]
    if supplied:
        arguments += ["--baseline", baseline]
    monkeypatch.setattr(sys, "argv", arguments)
    gate.main()
    ((positional, keywords),) = calls
    assert positional == (tmp_path.resolve(), False)
    assert keywords["baseline"] == (baseline if supplied else None)


def test_cli_does_not_normalize_a_malformed_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(gate, "gate", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["release_gate.py", "--candidate", str(tmp_path), "--baseline", " HEAD^2 "]
    )
    gate.main()
    assert calls[0]["baseline"] == " HEAD^2 "


@pytest.mark.parametrize(
    "broken",
    ["first_parent", "ignores_baseline", "accepts_anything"],
)
def test_self_test_rejects_a_weakened_baseline_binding(
    monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    real = gate.canonical_candidate_diff_sha256

    def weakened(source: Path, baseline: str | None, *, revision: str) -> str:
        if broken == "first_parent" and baseline is None:
            # Silently choose a merge parent instead of refusing.
            parent = _git(source, "rev-parse", f"{revision}^1")
            return real(source, parent, revision=revision)
        if broken == "ignores_baseline":
            parent = _git(source, "rev-parse", f"{revision}^1")
            return real(source, parent, revision=revision)
        if broken == "accepts_anything" and baseline is not None:
            return "0" * 64
        return real(source, baseline, revision=revision)

    monkeypatch.setattr(gate, "candidate_diff_binding", weakened, raising=False)
    with pytest.raises(RuntimeError, match="self-test"):
        gate.self_test()


def test_self_test_keeps_baseline_refusal_markers_out_of_its_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate.self_test()
    assert _MARKER not in capsys.readouterr().out
