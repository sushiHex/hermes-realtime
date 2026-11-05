from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import release_gate as gate


def test_clean_git_environment_isolates_user_system_and_attribute_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "ambient-selector")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "0")
    monkeypatch.setenv("GIT_ATTR_NOSYSTEM", "0")
    environment = gate.clean_git_environment()
    assert environment["GIT_CONFIG_GLOBAL"] == gate.os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    (root / "keep.py").write_bytes(b"a\n\nb\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


@pytest.mark.parametrize("scope", ["global", "local"])
def test_release_digest_ignores_ambient_config(tmp_path: Path, monkeypatch, scope) -> None:
    root = _repo(tmp_path)
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "keep.py").write_bytes(b"a\n\nc\n")
    _git(root, "commit", "-qam", "change")
    expected = gate.canonical_candidate_diff_sha256(root, baseline)
    config = tmp_path / "ambient.config"
    config.write_text(
        "[diff]\n noprefix = true\n context = 1\n suppressBlankEmpty = true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / ".gitconfig").write_bytes(config.read_bytes())
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    if scope == "local":
        _git(root, "config", "diff.noprefix", "true")
        _git(root, "config", "diff.context", "1")
        _git(root, "config", "diff.suppressBlankEmpty", "true")
    assert gate.canonical_candidate_diff_sha256(root, baseline) == expected


def test_release_root_digest_ignores_local_config(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    expected = gate.canonical_candidate_diff_sha256(root, None)
    _git(root, "config", "diff.noprefix", "true")
    _git(root, "config", "diff.context", "1")
    assert gate.canonical_candidate_diff_sha256(root, None) == expected


@pytest.mark.parametrize("root_commit", [False, True])
def test_release_digest_ignores_external_attributes(tmp_path: Path, root_commit) -> None:
    root = _repo(tmp_path)
    baseline = None
    if not root_commit:
        baseline = _git(root, "rev-parse", "HEAD")
        (root / "keep.py").write_bytes(b"a\n\nc\n")
        _git(root, "commit", "-qam", "change")
    expected = gate.canonical_candidate_diff_sha256(root, baseline)
    attributes = tmp_path / "ambient.attributes"
    attributes.write_text("*.py -diff\n", encoding="utf-8")
    _git(root, "config", "core.attributesFile", str(attributes))
    assert gate.canonical_candidate_diff_sha256(root, baseline) == expected


def test_release_digest_binds_committed_attributes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / ".gitattributes").write_text("*.py diff\n", encoding="utf-8")
    (root / "keep.py").write_bytes(b"candidate\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "candidate attributes")
    expected = gate.canonical_candidate_diff_sha256(root, None)
    (root / ".gitattributes").write_text("*.py -diff\n", encoding="utf-8")
    assert gate.canonical_candidate_diff_sha256(root, None) == expected


def test_release_digest_normalizes_interhunk_context(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    lines = [f"line {index}\n" for index in range(50)]
    (root / "keep.py").write_text("".join(lines), encoding="utf-8")
    _git(root, "commit", "-qam", "baseline")
    baseline = _git(root, "rev-parse", "HEAD")
    lines[2] = "first change\n"
    lines[45] = "second change\n"
    (root / "keep.py").write_text("".join(lines), encoding="utf-8")
    _git(root, "commit", "-qam", "candidate")
    expected = gate.canonical_candidate_diff_sha256(root, baseline)
    _git(root, "config", "diff.interHunkContext", "50")
    assert gate.canonical_candidate_diff_sha256(root, baseline) == expected


@pytest.mark.parametrize("root_commit", [False, True])
def test_release_digest_ignores_local_order_file(tmp_path: Path, root_commit) -> None:
    root = _repo(tmp_path)
    for name in ("a.py", "b.py"):
        (root / name).write_bytes(b"baseline\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    baseline = _git(root, "rev-parse", "HEAD")
    for name in ("a.py", "b.py"):
        (root / name).write_bytes(b"candidate\n")
    _git(root, "commit", "-qam", "candidate")
    if root_commit:
        # Use a fresh single root with both files, so diff-tree is exercised.
        root = tmp_path / "single-root"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Fixture")
        _git(root, "config", "user.email", "fixture@example.invalid")
        for name in ("a.py", "b.py"):
            (root / name).write_bytes(b"candidate\n")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "root")
        baseline = None
    expected = gate.canonical_candidate_diff_sha256(root, baseline)
    order = tmp_path / "order.txt"
    order.write_bytes(b"b.py\na.py\n")
    _git(root, "config", "diff.orderFile", str(order))
    assert gate.canonical_candidate_diff_sha256(root, baseline) == expected


def test_release_digest_refuses_info_attributes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / ".git" / "info" / "attributes").write_text("*.py -diff\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="attributes"):
        gate.canonical_candidate_diff_sha256(root, None)


def test_release_archive_refuses_local_info_attributes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "retained.py").write_bytes(b"retained\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "retained archive member")
    info = root / ".git" / "info" / "attributes"
    info.write_text("keep.py export-ignore\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="attributes"):
        gate.git_archive(root, output, revision=_git(root, "rev-parse", "HEAD"))


def test_release_archive_ignores_external_attributes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "retained.py").write_bytes(b"retained\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "retained archive member")
    attributes = tmp_path / "external.attributes"
    attributes.write_text("keep.py export-ignore\n", encoding="utf-8")
    _git(root, "config", "core.attributesFile", str(attributes))
    output = tmp_path / "output"
    output.mkdir()
    gate.git_archive(root, output, revision=_git(root, "rev-parse", "HEAD"))
    assert (output / "keep.py").read_bytes() == (root / "keep.py").read_bytes()


@pytest.mark.parametrize("guard", ["hash", "pid"])
def test_native_livekit_rejects_foreign_identity_before_network(
    tmp_path: Path, monkeypatch, guard
) -> None:
    executable = tmp_path / "livekit.exe"
    executable.write_bytes(b"fixture")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setattr(gate, "_verified_livekit_listener_pid", lambda: 42)
    monkeypatch.setattr(
        gate.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("identity guard reached network"),
    )
    with pytest.raises(
        RuntimeError, match="hash" if guard == "hash" else "not owned by the supplied process"
    ):
        gate.check_livekit(
            tmp_path,
            {},
            executable=executable,
            expected_sha256="0" * 64 if guard == "hash" else digest,
            process_id=43,
        )


def test_release_self_test_is_automated() -> None:
    gate.self_test()


def test_release_rejects_nested_forbidden_archive_path(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(tmp_path / "one.whl", "w") as archive:
        for name in {
            *gate.REQUIRED_STATIC,
            *gate.REQUIRED_MODEL_RESOURCES,
            "hermes_realtime/evidence/disclosure_v1.txt",
            "hermes_realtime/evidence/disclosure_manifest_v1.json",
        }:
            archive.writestr(name, (root / "src" / name).read_bytes())
        archive.writestr("web/tmp/trace.json", b"{}")
    with tarfile.open(tmp_path / "one.tar.gz", "w:gz") as archive:
        for name in gate.required_sdist_paths():
            member = tarfile.TarInfo("fixture/" + name)
            archive.addfile(member, io.BytesIO(b""))
    with pytest.raises(RuntimeError, match="forbidden path"):
        gate.check_artifacts(root, tmp_path)


def test_gate_binds_scan_and_digest_to_resolved_revision(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path)
    revision = _git(root, "rev-parse", "HEAD")
    calls = []

    def digest(source, baseline, *, revision):
        calls.append(("digest", revision))
        return "0" * 64

    monkeypatch.setattr(gate, "canonical_candidate_diff_sha256", digest)
    monkeypatch.setattr(
        gate, "scan_git_blobs", lambda source, *, revision: calls.append(("scan", revision))
    )
    monkeypatch.setattr(
        gate,
        "git_archive",
        lambda source, destination, *, revision: calls.append(("archive", revision)),
    )
    monkeypatch.setattr(gate, "gate_materialized_candidate", lambda *args, **kwargs: None)
    gate.gate(root, False)
    assert calls == [("digest", revision), ("scan", revision), ("archive", revision)]


def test_release_rejects_ambiguous_artifact_count(tmp_path: Path) -> None:
    for name in ("one.whl", "two.whl", "one.tar.gz"):
        (tmp_path / name).write_bytes(b"fixture")
    with pytest.raises(RuntimeError, match="exactly one wheel"):
        gate.check_artifacts(tmp_path, tmp_path)


@pytest.mark.parametrize("change", ["bytes", "missing", "extra"])
def test_rebuilt_wheel_requires_all_package_bytes(tmp_path: Path, change) -> None:
    for name in gate.REQUIRED_STATIC:
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"static")
    package = tmp_path / "src" / "hermes_realtime"
    for name in (
        "evidence/disclosure_v1.txt",
        "evidence/disclosure_manifest_v1.json",
        "runtime.py",
    ):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"canonical")
    wheel = tmp_path / "rebuilt.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in package.rglob("*"):
            if path.is_file():
                if change == "missing" and path.name == "runtime.py":
                    continue
                archive.writestr(
                    path.relative_to(tmp_path / "src").as_posix(),
                    b"drift"
                    if change == "bytes" and path.name == "runtime.py"
                    else path.read_bytes(),
                )
        if change == "extra":
            archive.writestr("hermes_realtime/foreign.py", b"foreign")
    with pytest.raises(RuntimeError, match="differ.*from source"):
        gate.check_sdist_built_wheel_disclosure_parity(tmp_path, wheel)
