"""Retained file authority must reject aliasing and expire with its owner."""

import os
from hashlib import sha256

import pytest


@pytest.mark.parametrize("unsupported", ["remote", "unknown_filesystem"])
def test_file_seals_refuse_unsupported_volume_authority(tmp_path, monkeypatch, unsupported):
    from scripts import qualification_file_seals as owner

    (tmp_path / "artifact.bin").write_bytes(b"synthetic artifact\n")
    kernel, _ = owner._api()
    if unsupported == "remote":
        monkeypatch.setattr(kernel, "GetDriveTypeW", lambda *_: 4)
    else:
        monkeypatch.setattr(kernel, "GetVolumeInformationByHandleW", lambda *_: 0)
    with (
        pytest.raises(ValueError, match="fixed|NTFS"),
        owner.retain_file_seals(tmp_path, ("artifact.bin",)),
    ):
        pytest.fail("unsupported volume was admitted")

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained file sharing semantics")


def test_seals_deny_transient_write_replace_delete_and_ancestor_rename(tmp_path):
    from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes

    root = tmp_path / "inputs"
    (root / "artifacts").mkdir(parents=True)
    artifact = root / "artifacts/tool.bin"
    artifact.write_bytes(b"synthetic immutable input")
    with retain_file_seals(root, ("artifacts/tool.bin",)) as seal:
        assert sealed_file_bytes(seal, "artifacts/tool.bin", 1024) == b"synthetic immutable input"
        with pytest.raises(OSError):
            artifact.write_bytes(b"changed")
        with pytest.raises(OSError):
            artifact.unlink()
        with pytest.raises(OSError):
            artifact.rename(root / "artifacts/changed.bin")
        with pytest.raises(OSError):
            (root / "artifacts").rename(root / "changed")
        with pytest.raises(OSError):
            root.rename(tmp_path / "moved")
    with pytest.raises(ValueError, match="closed"):
        sealed_file_bytes(seal, "artifacts/tool.bin", 1024)
    artifact.write_bytes(b"released")


def test_hard_link_input_is_rejected_and_failed_seals_release_handles(tmp_path):
    from scripts.qualification_file_seals import retain_file_seals

    root = tmp_path / "inputs"
    root.mkdir()
    original, alias = root / "first.bin", root / "second.bin"
    original.write_bytes(b"synthetic")
    os.link(original, alias)
    with (
        pytest.raises(ValueError, match="hard link"),
        retain_file_seals(root, ("first.bin", "second.bin")),
    ):
        pytest.fail("aliased input accepted")
    original.unlink()
    alias.unlink()


def test_read_and_metadata_require_the_original_live_capability(tmp_path):
    from scripts.qualification_file_seals import (
        RetainedFileSealsV1,
        retain_file_seals,
        sealed_file_bytes,
        sealed_file_metadata,
    )

    (tmp_path / "input.bin").write_bytes(b"synthetic")
    with pytest.raises(TypeError):
        RetainedFileSealsV1()
    with pytest.raises(ValueError, match="closed"):
        sealed_file_metadata(object.__new__(RetainedFileSealsV1))
    with retain_file_seals(tmp_path, ("input.bin",)) as seal:
        assert sealed_file_metadata(seal) == (("input.bin", sha256(b"synthetic").hexdigest(), 9),)
        with pytest.raises(ValueError, match="member"):
            sealed_file_bytes(seal, "unknown.bin", 1024)
        with pytest.raises(ValueError, match="bound"):
            sealed_file_bytes(seal, "input.bin", 2)


@pytest.mark.parametrize(
    "relative", ["../escape", "C:/outside", "input.bin:stream", "a//b", "a/../b"]
)
def test_indirect_or_noncanonical_member_paths_are_rejected(tmp_path, relative):
    from scripts.qualification_file_seals import retain_file_seals

    with pytest.raises(ValueError), retain_file_seals(tmp_path, (relative,)):
        pytest.fail("noncanonical member accepted")


@pytest.mark.parametrize("member", ["launcher manifest.xml", "GMT+0", "script (dev).tmpl"])
def test_file_seals_preserve_ordinary_windows_resource_names(tmp_path, member):
    from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes

    (tmp_path / member).write_bytes(b"synthetic resource")
    with retain_file_seals(tmp_path, (member,)) as receipt:
        assert sealed_file_bytes(receipt, member, 1024) == b"synthetic resource"
        with pytest.raises(OSError):
            (tmp_path / member).write_bytes(b"changed")
