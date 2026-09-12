"""Synthetic parser fixtures do not establish publisher or execution evidence."""

import hashlib
import io
import tarfile
import zipfile

import pytest


def fixture_archive(format, members):
    stream = io.BytesIO()
    if format == "zip":
        with zipfile.ZipFile(stream, "w") as archive:
            for name, raw in members:
                archive.writestr(name, raw)
    else:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, raw in members:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


def admit_fixture(monkeypatch, raw, format):
    from scripts import qualification_tool_distributions as tools

    # Test-only trust substitution exercises the parser, never real tool admission.
    policy = tools._ToolPolicy(
        "synthetic", "synthetic", "tool.exe", format, hashlib.sha256(raw).hexdigest(), len(raw)
    )
    monkeypatch.setattr(tools, "_TOOLS", {"synthetic": policy})
    return tools.admit_tool_distribution("synthetic", raw)


@pytest.mark.parametrize("role", ["git", "uv", "build_python", "unknown", True])
def test_caller_bytes_cannot_choose_their_own_tool_authority(role):
    from scripts.qualification_tool_distributions import admit_tool_distribution

    with pytest.raises((ValueError, TypeError)):
        admit_tool_distribution(role, b"synthetic replacement")


@pytest.mark.parametrize("format", ["zip", "tar.gz"])
def test_complete_distribution_is_preserved_as_immutable_bytes(monkeypatch, format):
    from scripts.qualification_tool_distributions import (
        _tool_distribution_files,
        tool_distribution_metadata,
    )

    raw = fixture_archive(format, [("tool.exe", b"image"), ("lib/resource", b"resource")])
    receipt = admit_fixture(monkeypatch, raw, format)
    metadata = tool_distribution_metadata(receipt)
    assert metadata.distribution_sha256 == hashlib.sha256(raw).hexdigest()
    assert metadata.file_count == 2 and metadata.expanded_bytes == 13
    files = _tool_distribution_files(receipt)
    assert dict(files) == {"lib/resource": b"resource", "tool.exe": b"image"}
    with pytest.raises(TypeError):
        files[0] = ("tool.exe", b"replacement")


@pytest.mark.parametrize("format", ["zip", "tar.gz"])
@pytest.mark.parametrize("member", ["../escape", "/absolute", "AUX.txt", "file.", "x:y"])
def test_unsafe_distribution_members_refuse_before_materialization(monkeypatch, format, member):
    raw = fixture_archive(format, [("tool.exe", b"image"), (member, b"foreign")])
    with pytest.raises(ValueError):
        admit_fixture(monkeypatch, raw, format)


@pytest.mark.parametrize(
    "members",
    [
        [("tool.exe", b"a"), ("TOOL.EXE", b"b")],
        [("tool.exe", b"a"), ("tool.exe/child", b"b")],
        [("lib/a", b"a"), ("Lib/b", b"b"), ("tool.exe", b"image")],
        [("resource", b"missing image")],
    ],
)
def test_aliases_and_missing_selected_image_refuse(monkeypatch, members):
    raw = fixture_archive("zip", members)
    with pytest.raises(ValueError):
        admit_fixture(monkeypatch, raw, "zip")


def test_tar_link_is_never_a_tool_file(monkeypatch):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("tool.exe")
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        archive.addfile(member)
    with pytest.raises(ValueError):
        admit_fixture(monkeypatch, stream.getvalue(), "tar.gz")


def test_distribution_member_bound_is_enforced(monkeypatch):
    from scripts import qualification_tool_distributions as tools

    monkeypatch.setattr(tools, "_MAX_MEMBERS", 1)
    raw = fixture_archive("zip", [("tool.exe", b"image"), ("resource", b"resource")])
    with pytest.raises(ValueError):
        admit_fixture(monkeypatch, raw, "zip")


@pytest.mark.parametrize("format", ["zip", "tar.gz"])
def test_private_archive_consumer_can_select_a_narrow_larger_member_bound(monkeypatch, format):
    from scripts import qualification_tool_distributions as tools

    monkeypatch.setattr(tools, "_MAX_MEMBER", 4)
    raw = fixture_archive(format, [("tool.exe", b"image bytes")])
    with pytest.raises(ValueError, match="bounded ordinary file"):
        tools._inspect_archive_members(raw, format)
    assert tools._inspect_archive_members(raw, format, max_member_bytes=11) == (
        ("tool.exe", b"image bytes"),
    )
    with pytest.raises(ValueError, match="bounded ordinary file"):
        tools._inspect_archive_members(raw, format, max_member_bytes=10)


def test_distribution_capability_cannot_be_constructed_or_replaced_with_metadata():
    from scripts.qualification_tool_distributions import (
        AdmittedToolDistributionV1,
        tool_distribution_metadata,
    )

    with pytest.raises(TypeError):
        AdmittedToolDistributionV1()
    with pytest.raises(ValueError):
        tool_distribution_metadata(object.__new__(AdmittedToolDistributionV1))
    with pytest.raises(TypeError):
        tool_distribution_metadata({"passed": True})


@pytest.mark.parametrize("format", ["zip", "tar.gz"])
def test_distribution_keeps_ordinary_windows_resource_names(monkeypatch, format):
    from scripts.qualification_tool_distributions import _tool_distribution_files

    members = [
        ("tool.exe", b"image"),
        ("Lib/launcher manifest.xml", b"manifest"),
        ("tcl/Etc/GMT+0", b"timezone"),
        ("Lib/script (dev).tmpl", b"template"),
    ]
    receipt = admit_fixture(monkeypatch, fixture_archive(format, members), format)
    assert dict(_tool_distribution_files(receipt)) == dict(members)
