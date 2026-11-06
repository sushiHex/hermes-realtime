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


def test_release_digest_avoids_git_243_implicit_no_index(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path)
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "keep.py").write_bytes(b"changed\n")
    _git(root, "commit", "-qam", "candidate")
    run = subprocess.run
    diff_commands = []

    def git_243_boundary(command, *args, **kwargs):
        if command[0] == "git" and "diff" in command:
            options = command[command.index("diff") + 1:]
            diff_commands.append(options)
            # https://github.com/git/git/blob/v2.43.0/builtin/diff.c#L413-L441
            # stops at the first non-option before parsing option arguments.
            # Two remaining operands, one outside the repo, imply --no-index.
            # Model /dev/null using the platform's null-device argument here;
            # all actual diff execution still uses the installed native Git.
            first = next(i for i, item in enumerate(options) if not item.startswith("-"))
            if len(options) - first == 2 and options[first] == gate.os.devnull:
                raise subprocess.CalledProcessError(129, command)
        return run(command, *args, **kwargs)

    monkeypatch.setattr(gate.subprocess, "run", git_243_boundary)
    digest = gate.canonical_candidate_diff_sha256(root, baseline)
    assert len(digest) == 64 and len(diff_commands) == 1


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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("core.bigFileThreshold", "1"),
        ("core.quotePath", "false"),
        ("diff.outputIndicatorNew", "@"),
        ("diff.outputIndicatorOld", "%"),
        ("diff.outputIndicatorContext", "!"),
    ],
)
def test_release_digest_normalizes_binary_path_and_patch_rendering(tmp_path, key, value) -> None:
    root = _repo(tmp_path)
    unicode_path = root / "\u03bb.py"
    unicode_path.write_bytes(b"a\n\nb\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    baseline = _git(root, "rev-parse", "HEAD")
    for path in (unicode_path, root / "keep.py"):
        path.write_bytes(b"a\n\nc\n")
    _git(root, "commit", "-qam", "candidate")
    expected = gate.canonical_candidate_diff_sha256(root, baseline)
    _git(root, "config", key, value)
    assert gate.canonical_candidate_diff_sha256(root, baseline) == expected


def test_release_root_digest_normalizes_path_spelling(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _git(root, "checkout", "--orphan", "single-root")
    (root / "\u03bb.py").write_bytes(b"candidate\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "single root")
    expected = gate.canonical_candidate_diff_sha256(root, None)
    _git(root, "config", "core.quotePath", "false")
    assert gate.canonical_candidate_diff_sha256(root, None) == expected


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


@pytest.mark.parametrize("consumer", ["archive", "digest", "scan"])
def test_release_source_readers_ignore_local_replacement_objects(tmp_path: Path, consumer) -> None:
    root = _repo(tmp_path)
    original = _git(root, "rev-parse", "HEAD")
    original_digest = gate.canonical_candidate_diff_sha256(root, None, revision=original)
    (root / "keep.py").write_bytes(b"replacement\n")
    _git(root, "commit", "-qam", "replacement")
    replacement = _git(root, "rev-parse", "HEAD")
    _git(root, "replace", original, replacement)
    if consumer == "archive":
        output = tmp_path / "output"
        output.mkdir()
        gate.git_archive(root, output, revision=original)
        assert (output / "keep.py").read_bytes() == b"a\n\nb\n"
    elif consumer == "digest":
        actual_digest = gate.canonical_candidate_diff_sha256(root, None, revision=original)
        assert actual_digest == original_digest
    else:
        # A replacement must not hide a forbidden path in the selected commit.
        _git(root, "replace", "-d", original)
        (root / "research").mkdir()
        (root / "research" / "secret.py").write_bytes(b"ghp_" + b"x" * 40)
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "forbidden")
        forbidden = _git(root, "rev-parse", "HEAD")
        _git(root, "replace", forbidden, original)
        with pytest.raises(RuntimeError, match="research"):
            gate.scan_git_blobs(root, revision=forbidden)


def test_linux_archive_refuses_local_attributes(tmp_path: Path) -> None:
    from scripts import qualification_linux_producer as producer
    root = _repo(tmp_path)
    (root / ".git" / "info" / "attributes").write_text("keep.py export-ignore\n")
    with pytest.raises(RuntimeError, match="local info attributes"):
        producer._git_archive(root)


def test_linux_archive_ignores_ambient_archive_configuration(tmp_path: Path, monkeypatch) -> None:
    from scripts import qualification_linux_producer as producer
    root = _repo(tmp_path)
    original = producer._git_archive(root)
    attrs = tmp_path / "attrs"
    attrs.write_text("keep.py export-ignore\n")
    _git(root, "config", "core.attributesFile", str(attrs))
    _git(root, "config", "tar.umask", "0077")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "foreign"))
    assert producer._git_archive(root) == original


@pytest.mark.parametrize("setting", ["local_filter", "local_attributes"])
def test_worktree_identity_refuses_untrusted_cleaning_rules(tmp_path: Path, setting) -> None:
    from scripts import real_gate_support as support
    root = _repo(tmp_path)
    if setting == "local_filter":
        _git(root, "config", "filter.mask.clean", "echo stable")
    else:
        (root / ".git" / "info" / "attributes").write_text("*.py -text\n")
    with pytest.raises(RuntimeError, match="(clean filters|local info attributes)"):
        support.installed_hermes_identity("fixture", root)


def test_worktree_identity_ignores_global_status_configuration(tmp_path: Path, monkeypatch) -> None:
    from scripts import real_gate_support as support
    root = _repo(tmp_path)
    configuration = tmp_path / "global-config"
    configuration.write_text("[core]\n fsmonitor = false\n[status]\n showUntrackedFiles = no\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(configuration))
    environment = gate.worktree_git_environment(root)
    assert environment["GIT_CONFIG_GLOBAL"] == gate.os.devnull
    assert support.installed_hermes_identity("fixture", root)["commit"]



def test_worktree_identity_ignores_external_attributes(tmp_path: Path) -> None:
    from scripts import real_gate_support as support
    root = _repo(tmp_path)
    attrs = tmp_path / "attrs"
    attrs.write_text("keep.py -text\n")
    _git(root, "config", "core.attributesFile", str(attrs))
    environment = gate.worktree_git_environment(root)
    result = subprocess.run(
        ("git", "check-attr", "text", "--", "keep.py"), cwd=root,
        env=environment, capture_output=True, check=True, text=True,
    )
    assert result.stdout.strip() == "keep.py: text: unspecified"
    assert support.installed_hermes_identity("fixture", root)["commit"]


@pytest.mark.parametrize("mode", ["yes", "1", "on", "no", "0", "off", "input"])
def test_worktree_identity_accepts_git_line_ending_aliases(tmp_path: Path, mode) -> None:
    from scripts import real_gate_support as support
    root = _repo(tmp_path)
    _git(root, "config", "core.autocrlf", mode)
    assert support.installed_hermes_identity("fixture", root)["commit"]



def test_worktree_identity_refuses_unknown_line_ending_mode(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _git(root, "config", "core.autocrlf", "unsupported-value")
    with pytest.raises(RuntimeError, match="unsupported worktree line-ending mode"):
        gate.worktree_git_environment(root)
