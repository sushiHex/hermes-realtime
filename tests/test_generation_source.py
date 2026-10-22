"""Focused clean-source export controls; no provider or runtime is executed."""

import importlib.util
import json
import subprocess
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
