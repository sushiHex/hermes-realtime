"""Focused clean-source export controls; no provider or runtime is executed."""

import contextlib
import hashlib
import importlib.util
import json
import math
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generation_source", Path(__file__).with_name("generation_source.py")
)
source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source)


def git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout


@pytest.fixture
def committed(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    git(repository, "config", "user.name", "Synthetic Fixture")
    git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "module.py").write_bytes(b"PINNED = True\n")
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repository / ".gitattributes").write_text("module.py export-ignore\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "synthetic source")
    pin = git(repository, "rev-parse", "HEAD").decode().strip()
    return repository, pin


def test_export_uses_exact_blobs_and_omits_every_worktree_overlay(committed, tmp_path):
    repository, pin = committed
    (repository / "module.py").write_bytes(b"UNCOMMITTED = True\n")
    ignored = repository / "ignored"
    ignored.mkdir()
    for name in ("overlay.py", "overlay.pyc", "overlay.pyd", "overlay.so", "startup.pth"):
        (ignored / name).write_bytes(b"not a committed Git blob")
    (repository / "untracked.py").write_bytes(b"not committed")
    destination = tmp_path / "export"
    manifest = source.export_tree(repository, pin, destination)
    assert {
        p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
    } == {"module.py", ".gitignore", ".gitattributes"}
    assert (destination / "module.py").read_bytes() == b"PINNED = True\n"
    assert manifest["source_export_files"] == 3
    assert (
        manifest["source_export_tree"]
        == git(repository, "rev-parse", f"{pin}^{{tree}}").decode().strip()
    )
    assert manifest["source_export_bytes"] == sum(p.stat().st_size for p in destination.iterdir())
    assert len(manifest["source_export_manifest_sha256"]) == 64
    assert all(key.startswith("source_export_") for key in manifest)
    assert str(repository) not in json.dumps(manifest)


def test_local_replace_refs_cannot_override_pinned_objects(committed, tmp_path):
    repository, pin = committed
    (repository / "module.py").write_bytes(b"REPLACEMENT = True\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "replacement source")
    replacement = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "replace", pin, replacement)
    destination = tmp_path / "export"
    source.export_tree(repository, pin, destination)
    assert (destination / "module.py").read_bytes() == b"PINNED = True\n"


def test_export_timings_are_fixed_content_free_and_preserve_payload(committed, tmp_path):
    # Mutation: omit the exporter timing fields while retaining a successful export.
    repository, pin = committed
    first = source.export_tree(repository, pin, tmp_path / "first")
    second = source.export_tree(repository, pin, tmp_path / "second")
    timings = {key: value for key, value in first.items() if key.endswith("_seconds")}
    assert set(timings) == {
        "source_export_metadata_seconds",
        "source_export_blob_read_seconds",
        "source_export_validation_seconds",
        "source_export_filesystem_work_seconds",
        "source_export_manifest_work_seconds",
        "source_export_materialization_wall_seconds",
        "source_export_total_seconds",
    }
    assert all(
        type(value) is float and math.isfinite(value) and value >= 0 for value in timings.values()
    )
    assert {key: value for key, value in first.items() if key not in timings} == {
        key: value for key, value in second.items() if key not in timings
    }
    assert (tmp_path / "first/module.py").read_bytes() == (
        tmp_path / "second/module.py"
    ).read_bytes()


def test_export_creates_each_parent_once_without_changing_payload(committed, tmp_path):
    # Mutation: perform the parent mkdir for every blob instead of once per parent.
    repository, _ = committed
    nested = repository / "package"
    nested.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (nested / name).write_bytes(b"SYNTHETIC = True\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "nested synthetic source")
    pin = git(repository, "rev-parse", "HEAD").decode().strip()
    destination = tmp_path / "export"
    original = Path.mkdir
    created = []

    def mkdir(path, *args, **kwargs):
        created.append(path)
        return original(path, *args, **kwargs)

    with patch.object(Path, "mkdir", mkdir):
        manifest = source.export_tree(repository, pin, destination)
    assert created == [destination, destination / "package"]
    assert manifest["source_export_files"] == 6
    assert [(destination / "package" / name).read_bytes() for name in ("a.py", "b.py", "c.py")] == [
        b"SYNTHETIC = True\n"
    ] * 3


def test_existing_destination_refused_before_any_write(committed, tmp_path):
    repository, pin = committed
    destination = tmp_path / "export"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"preserved")
    with pytest.raises(source.SourceExportError, match="destination"):
        source.export_tree(repository, pin, destination)
    assert (destination / "sentinel").read_bytes() == b"preserved"


def test_revision_must_be_an_exact_full_commit(committed, tmp_path):
    repository, _ = committed
    with (
        patch.object(source, "_git") as query,
        pytest.raises(source.SourceExportError, match="revision"),
    ):
        source.export_tree(repository, "HEAD", tmp_path / "export")
    query.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/absolute.py",
        "a\\b.py",
        "C:drive.py",
        ".git/config",
        "x/../escape",
        "a//b",
        "a./b",
        "NUL.py",
        "a/CON",
        "bad\nname",
    ],
)
def test_unsafe_paths_refused_before_destination_creation(tmp_path, path):
    repository, pin = tmp_path, "1" * 40
    listing = f"100644 blob {'a' * 40}\t{path}\0".encode()
    destination = tmp_path / "export"
    with (
        patch.object(source, "_git", side_effect=[pin.encode(), b"b" * 40, listing]),
        pytest.raises(source.SourceExportError, match="path"),
    ):
        source.export_tree(repository, pin, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mode,kind", [("120000", "blob"), ("160000", "commit")])
def test_symlink_and_submodule_entries_refused(tmp_path, mode, kind):
    repository, pin = tmp_path, "1" * 40
    listing = f"{mode} {kind} {'a' * 40}\tlinked\0".encode()
    with (
        patch.object(source, "_git", side_effect=[pin.encode(), b"b" * 40, listing]),
        pytest.raises(source.SourceExportError, match="entry"),
    ):
        source.export_tree(repository, pin, tmp_path / "export")


@pytest.mark.parametrize("paths", [("A/file", "a/other"), ("file", "file/nested")])
def test_case_aliases_and_file_directory_collisions_refused(tmp_path, paths):
    repository, pin = tmp_path, "1" * 40
    listing = b"".join(f"100644 blob {'a' * 40}\t{name}\0".encode() for name in paths)
    with (
        patch.object(source, "_git", side_effect=[pin.encode(), b"b" * 40, listing]),
        pytest.raises(source.SourceExportError, match="collision"),
    ):
        source.export_tree(repository, pin, tmp_path / "export")


def test_corrupt_blob_payload_refused(tmp_path):
    repository, pin = tmp_path, "1" * 40
    payload = b"changed"
    listing = f"100644 blob {'a' * 40}\tmodule.py\0".encode()
    batch = f"{'a' * 40} blob {len(payload)}\n".encode() + payload + b"\n"
    with (
        patch.object(source, "_git", side_effect=[pin.encode(), b"b" * 40, listing, batch]),
        pytest.raises(source.SourceExportError, match="blob"),
    ):
        source.export_tree(repository, pin, tmp_path / "export")


def test_git_failure_is_content_free_and_fresh(tmp_path):
    repository, pin = tmp_path, "1" * 40
    with (
        patch.object(
            source.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-path", 30)
        ),
        pytest.raises(source.SourceExportError, match="^source_export_git_failed$"),
    ):
        source.export_tree(repository, pin, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    "responses,category",
    [
        ([b"2" * 40], "revision_mismatch"),
        ([b"1" * 40, b"not-a-tree"], "tree_invalid"),
        ([b"1" * 40, b"2" * 40, b"unterminated"], "entry_invalid"),
        ([b"1" * 40, b"2" * 40, (f"100644 blob {'a' * 40}\tfile\0" * 2).encode()], "entry_invalid"),
    ],
)
def test_invalid_object_tree_or_listing_refuses(tmp_path, responses, category):
    with (
        patch.object(source, "_git", side_effect=responses),
        pytest.raises(source.SourceExportError, match=category),
    ):
        source.export_tree(tmp_path, "1" * 40, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("suffix", [b"trailing", b"missing"])
def test_batch_framing_refused_before_write(tmp_path, suffix):
    payload = b"pinned"
    import hashlib

    oid = hashlib.sha1(b"blob 6\0" + payload).hexdigest()
    listing = f"100644 blob {oid}\tmodule.py\0".encode()
    batch = f"{oid} blob 6\n".encode() + payload
    batch += b"\ntrailing" if suffix == b"trailing" else b""
    with (
        patch.object(source, "_git", side_effect=[b"1" * 40, b"2" * 40, listing, batch]),
        pytest.raises(source.SourceExportError, match="blob_invalid"),
    ):
        source.export_tree(tmp_path, "1" * 40, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_healthy_export_without_overlays(committed, tmp_path):
    repository, pin = committed
    result = source.export_tree(repository, pin, tmp_path / "healthy")
    assert result["source_export_files"] == 3
    assert (tmp_path / "healthy" / "module.py").read_bytes() == b"PINNED = True\n"


def test_git_invocation_is_bounded_and_hidden(tmp_path):
    from types import SimpleNamespace

    with patch.object(source.subprocess, "run", return_value=SimpleNamespace(stdout=b"ok")) as run:
        assert source._git(tmp_path, "rev-parse", "HEAD") == b"ok"
    assert run.call_args.kwargs["timeout"] == 30
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert run.call_args.kwargs["stderr"] == subprocess.DEVNULL
    assert run.call_args.kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert run.call_args.args[0][1] == "--no-replace-objects"


@pytest.mark.parametrize(
    "paths", [["../escape.py"], ["A/file", "a/other"], ["file", "file/nested"]]
)
def test_path_validator_refusals_without_filesystem_writes(paths):
    with pytest.raises(source.SourceExportError, match="path"):
        source._paths([(path, "100644", "a" * 40) for path in paths])


def projected_metadata_case(count=935, prefix="contributors/emails/"):
    import hashlib

    content = b"pinned"
    oid = hashlib.sha1(b"blob 6\0" + content).hexdigest()
    names = ["Alias", "alias", *(f"metadata-{index}" for index in range(max(0, count - 2)))]
    metadata = b"".join(f"100644 blob {oid}\t{prefix}{name}\0".encode() for name in names)
    listing = f"100644 blob {oid}\tmodule.py\0".encode() + metadata
    batch = f"{oid} blob 6\n".encode() + content + b"\n"
    return listing, batch


def test_exact_metadata_projection_records_full_tree_exclusion_and_payload(tmp_path):
    listing, batch = projected_metadata_case()
    responses = [b"1" * 40, b"2" * 40, listing, b"04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d", batch]
    destination = tmp_path / "export"
    with patch.object(source, "_git", side_effect=responses):
        result = source.export_tree(tmp_path, "1" * 40, destination)
    assert result["source_export_tree"] == "2" * 40
    assert result["source_export_full_tree_files"] == 936
    assert (
        result["source_export_excluded_metadata_tree"] == "04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d"
    )
    assert result["source_export_excluded_metadata_files"] == 935
    assert result["source_export_projection"] == "committed_without_pinned_attribution_metadata"
    assert result["source_export_files"] == 1
    assert (destination / "module.py").read_bytes() == b"pinned"
    assert not (destination / "contributors").exists()


@pytest.mark.parametrize(
    "count,tree",
    [
        (935, b"f" * 40),
        (934, b"04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d"),
        (936, b"04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d"),
    ],
)
def test_metadata_projection_refuses_changed_tree_or_count(tmp_path, count, tree):
    listing, batch = projected_metadata_case(count)
    with (
        patch.object(source, "_git", side_effect=[b"1" * 40, b"2" * 40, listing, tree, batch]),
        pytest.raises(source.SourceExportError, match="projection"),
    ):
        source.export_tree(tmp_path, "1" * 40, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_metadata_projection_does_not_ignore_another_case_collision(tmp_path):
    listing, _ = projected_metadata_case(prefix="other/emails/")
    with (
        patch.object(source, "_git", side_effect=[b"1" * 40, b"2" * 40, listing]),
        pytest.raises(source.SourceExportError, match="collision"),
    ):
        source.export_tree(tmp_path, "1" * 40, tmp_path / "export")


def test_source_export_exception_exposes_only_fixed_category():
    assert (
        source.SourceExportError("source_export_path_collision").category
        == "source_export_path_collision"
    )
    assert source.SourceExportError("not-an-approved-category").category == "source_export_refused"


def test_projection_preserves_sibling_contributor_module(tmp_path):
    import hashlib

    listing, batch = projected_metadata_case()
    oid = hashlib.sha1(b"blob 6\0pinned").hexdigest()
    listing += f"100644 blob {oid}\tcontributors/runtime.py\0".encode()
    responses = [
        b"1" * 40,
        b"2" * 40,
        listing,
        b"04e3b76fdc46dc1c6b36fd6f4b94701bf975f51d",
        batch + batch,
    ]
    destination = tmp_path / "export"
    with patch.object(source, "_git", side_effect=responses):
        result = source.export_tree(tmp_path, "1" * 40, destination)
    assert result["source_export_files"] == 2
    assert result["source_export_excluded_metadata_files"] == 935
    assert (destination / "contributors" / "runtime.py").read_bytes() == b"pinned"


def test_parallel_export_overlaps_two_workers_and_preserves_git_order(committed, tmp_path):
    # RED on the serial exporter; mutations: worker limit or completion-order results.
    repository, pin = committed
    payloads = {
        p.name: git(repository, "show", f"{pin}:{p.name}")
        for p in repository.iterdir()
        if p.is_file()
    }
    manifest = [
        [name, "100644", hashlib.sha256(payloads[name]).hexdigest()] for name in sorted(payloads)
    ]
    expected = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    destination = tmp_path / "export"
    original_open = Path.open
    caller = threading.current_thread()
    barrier = threading.Barrier(2)
    second_finished = threading.Event()
    lock = threading.Lock()
    active = peak = entered = 0
    limits = []

    names = sorted(payloads)

    class ObservedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            limits.append(kwargs["max_workers"])
            super().__init__(*args, **kwargs)

        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            if args[0][1] == names[1]:
                future.add_done_callback(lambda _future: second_finished.set())
            return future

    @contextlib.contextmanager
    def observed_open(path, *args, **kwargs):
        nonlocal active, peak, entered
        with original_open(path, *args, **kwargs) as stream:
            with lock:
                index = names.index(path.relative_to(destination).as_posix())
                entered += 1
                active += 1
                peak = max(peak, active)
            try:
                if threading.current_thread() is not caller and index < 2:
                    barrier.wait(timeout=5)
                    if index == 0:
                        assert second_finished.wait(5)
                yield stream
            finally:
                with lock:
                    active -= 1

    with (
        patch.object(source, "ThreadPoolExecutor", ObservedExecutor, create=True),
        patch.object(Path, "open", observed_open),
    ):
        result = source.export_tree(repository, pin, destination)
    assert (limits, peak, active, entered) == ([2], 2, 0, 3)
    assert result["source_export_manifest_sha256"] == expected
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == payloads


def test_worker_failure_settles_peer_before_export_error(committed, tmp_path):
    # Mutation: executor context shuts down with wait=False after a worker error.
    repository, pin = committed
    original_worker = source._materialize_blob
    barrier = threading.Barrier(2)
    release = threading.Event()
    peer_finished = threading.Event()
    shutdown_entered = threading.Event()
    finished = threading.Event()
    names = sorted(p.name for p in repository.iterdir() if p.is_file())
    waits = []
    outcomes = []

    class ObservedExecutor(ThreadPoolExecutor):
        def shutdown(self, wait=True, **kwargs):
            waits.append(wait)
            shutdown_entered.set()
            return super().shutdown(wait=wait, **kwargs)

    def worker(payload):
        index = names.index(payload[1])
        if index < 2:
            barrier.wait(timeout=5)
        if index == 0:
            raise OSError("synthetic worker failure")
        if index == 1:
            assert release.wait(5)
            try:
                return original_worker(payload)
            finally:
                peer_finished.set()
        return original_worker(payload)

    def export():
        try:
            source.export_tree(repository, pin, tmp_path / "export")
        except OSError:
            outcomes.append(peer_finished.is_set())
        finally:
            finished.set()

    thread = threading.Thread(target=export, daemon=True)
    with (
        patch.object(source, "ThreadPoolExecutor", ObservedExecutor),
        patch.object(source, "_materialize_blob", worker),
    ):
        try:
            thread.start()
            assert shutdown_entered.wait(5)
            assert waits == [True] and not finished.is_set()
        finally:
            release.set()
            thread.join(5)
    assert not thread.is_alive() and outcomes == [True] and peer_finished.is_set()


def _missing_promisor_blob(committed, tmp_path):
    import sys

    repository, pin = committed
    remote = tmp_path / "promisor.git"
    git(tmp_path, "clone", "--bare", "--no-hardlinks", str(repository), str(remote))
    marker = tmp_path / "upload-pack-called"
    helper = tmp_path / "upload-pack.py"
    helper.write_text(
        "import subprocess, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text('called', encoding='utf-8')\n"
        "raise SystemExit(subprocess.run(['git', 'upload-pack', *sys.argv[2:]], "
        "timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).returncode)\n",
        encoding="utf-8",
    )
    git(repository, "config", "remote.origin.url", str(remote))
    git(repository, "config", "remote.origin.promisor", "true")
    git(repository, "config", "extensions.partialClone", "origin")
    git(repository, "config", "protocol.file.allow", "always")
    git(
        repository,
        "config",
        "remote.origin.uploadpack",
        f'"{Path(sys.executable).as_posix()}" "{helper.as_posix()}" "{marker.as_posix()}"',
    )
    oid = git(repository, "rev-parse", f"{pin}:module.py").decode().strip()
    blob = repository / ".git" / "objects" / oid[:2] / oid[2:]
    assert blob.is_file()
    blob.chmod(0o600)
    blob.unlink()
    return repository, pin, oid, marker


def _trace_git_children(monkeypatch, trace):
    import os

    original = subprocess.run

    def traced(*args, **kwargs):
        kwargs["env"] = dict(kwargs.get("env", os.environ), GIT_TRACE2_EVENT=str(trace))
        return original(*args, **kwargs)

    monkeypatch.setattr(source.subprocess, "run", traced)


def test_promisor_control_really_fetches_missing_blob_with_local_helper(committed, tmp_path):
    repository, _pin, oid, marker = _missing_promisor_blob(committed, tmp_path)
    assert git(repository, "cat-file", "blob", oid) == b"PINNED = True\n"
    assert marker.read_text(encoding="utf-8") == "called"


def test_export_missing_promisor_blob_refuses_before_any_git_child(
    committed,
    tmp_path,
    monkeypatch,
):
    repository, pin, _oid, marker = _missing_promisor_blob(committed, tmp_path)
    trace = tmp_path / "git-trace.jsonl"
    _trace_git_children(monkeypatch, trace)
    refused = None
    try:
        source.export_tree(repository, pin, tmp_path / "export")
    except source.SourceExportError as error:
        refused = error.category
    children = sum(
        json.loads(line).get("event") == "child_start"
        for line in trace.read_text(encoding="utf-8").splitlines()
    )
    assert children == 0, "source-export-started-unobserved-git-child"
    assert not marker.exists(), "source-export-invoked-promisor-upload-pack"
    assert refused in {"source_export_git_failed", "source_export_blob_invalid"}
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("setting", ["global", "parameters", "objects"])
def test_export_ignores_inherited_git_config_and_object_redirects(
    committed,
    tmp_path,
    monkeypatch,
    setting,
):
    repository, pin = committed
    malformed = tmp_path / "invalid.gitconfig"
    malformed.write_text("[invalid config\n", encoding="utf-8")
    if setting == "global":
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(malformed))
    elif setting == "parameters":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "include.path")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(malformed))
    else:
        objects = tmp_path / "empty-objects"
        objects.mkdir()
        monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(objects))
    destination = tmp_path / "export"
    source.export_tree(repository, pin, destination)
    assert (destination / "module.py").read_bytes() == b"PINNED = True\n"


def test_git_transport_denial_cannot_run_local_remote_helper(committed, tmp_path):
    repository, _pin, oid, marker = _missing_promisor_blob(committed, tmp_path)
    with pytest.raises(source.SourceExportError, match="^source_export_git_failed$"):
        source._git(repository, "fetch", "origin", oid)
    assert not marker.exists()


def test_export_ignores_default_home_config(committed, tmp_path, monkeypatch):
    repository, pin = committed
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()
    (isolated_home / ".gitconfig").write_text("[invalid config\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated_home / "xdg"))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    destination = tmp_path / "export"
    source.export_tree(repository, pin, destination)
    assert (destination / "module.py").read_bytes() == b"PINNED = True\n"
