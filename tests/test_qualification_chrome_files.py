"""Synthetic publisher policy tests do not establish browser execution evidence."""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from runpy import run_path

import pytest


def _archive(members=None):
    members = members or {
        "First Run": b"publisher first-run metadata",
        "chrome.exe": b"synthetic image",
        "locales/en-US.pak": b"resource",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, raw in members.items():
            archive.writestr("chrome-win64/" + name, raw)
    return stream.getvalue()


def _admit(monkeypatch, raw):
    from scripts import qualification_chrome_files as chrome

    # Substitute only publisher policy in a deterministic parser fixture.
    monkeypatch.setattr(
        chrome,
        "_POLICY",
        chrome._ChromePolicy("153.0.8010.36", hashlib.sha256(raw).hexdigest(), len(raw)),
    )
    return chrome.admit_chrome_distribution(raw)


def test_publisher_digest_is_checked_before_archive_parsing():
    from scripts.qualification_chrome_files import admit_chrome_distribution

    with pytest.raises(ValueError, match="publisher distribution"):
        admit_chrome_distribution(_archive())


def test_admission_retains_every_publisher_member_and_produces_the_version_manifest(monkeypatch):
    from scripts.qualification_chrome_files import (
        _chrome_distribution_files,
        chrome_distribution_metadata,
        chrome_version_manifest,
    )

    admitted = _admit(monkeypatch, _archive())
    metadata = chrome_distribution_metadata(admitted)
    assert metadata.file_count == 3
    assert dict(_chrome_distribution_files(admitted)) == {
        "First Run": b"publisher first-run metadata",
        "chrome.exe": b"synthetic image",
        "locales/en-US.pak": b"resource",
    }
    manifest = json.loads(chrome_version_manifest(admitted))
    assert manifest["chromeVersion"] == metadata.version
    assert [row["name"] for row in manifest["files"]] == [
        "First Run",
        "chrome.exe",
        "locales/en-US.pak",
    ]
    assert manifest["files"][1]["sha256"] == hashlib.sha256(b"synthetic image").hexdigest()


@pytest.mark.parametrize(
    "members",
    [
        {"resource.dll": b"missing image"},
        {"chrome.exe": b"image", "CHROME.EXE": b"alias"},
        {"chrome.exe": b"image", "../outside": b"escape"},
        {"chrome.exe": b"image", "resource": b"file", "resource/child": b"collision"},
        {"chrome.exe": b"image", "resource": b""},
    ],
)
def test_incomplete_or_unsafe_publisher_namespace_refuses(monkeypatch, members):
    with pytest.raises(ValueError):
        _admit(monkeypatch, _archive(members))


def test_distribution_cannot_be_forged_or_replaced_with_metadata():
    from scripts.qualification_chrome_files import (
        AdmittedChromeDistributionV1,
        chrome_distribution_metadata,
    )

    with pytest.raises(TypeError):
        AdmittedChromeDistributionV1()
    with pytest.raises(ValueError, match="unregistered"):
        chrome_distribution_metadata(object.__new__(AdmittedChromeDistributionV1))
    with pytest.raises(TypeError):
        chrome_distribution_metadata({"passed": True})


@pytest.mark.parametrize("name", ["../outside", "AUX", "name.", "name ", "a:b", "a\\b", "\nname"])
def test_chrome_resource_names_keep_windows_path_refusals(name):
    from scripts.qualify_evidence_slice_zero import _chrome_resource_name

    with pytest.raises(ValueError):
        _chrome_resource_name(name)


def test_chrome_publisher_names_do_not_relax_direct_artifact_paths():
    from scripts.qualify_evidence_slice_zero import _chrome_resource_name, _safe_posix_path

    assert _chrome_resource_name("First Run") == "First Run"
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_posix_path("First Run", label="direct artifact")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    if os.name != "nt":
        pytest.skip("genuine Windows source and retained input authority")
    helpers = run_path(str(Path.cwd() / "tests/test_qualification_candidate_files.py"))
    return helpers, helpers["_make_source"](tmp_path_factory)


def _graph(tmp_path, source, distribution, *, change=None):
    from scripts.qualification_chrome_files import (
        _chrome_distribution_files,
        chrome_version_manifest,
    )
    from scripts.qualify_evidence_slice_zero import canonical_json_bytes

    helpers, candidate = source
    graph = helpers["bound_graph"].__wrapped__(tmp_path, candidate)
    root, _, document, _, _ = graph
    version_document = json.loads(chrome_version_manifest(distribution))
    contents = dict(_chrome_distribution_files(distribution))
    if change == "changed_resource":
        contents["locales/en-US.pak"] = b"rehashed replacement"
    if change == "missing_resource":
        del contents["locales/en-US.pak"]
    version_document["files"] = [
        {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(contents.items())
    ]
    version = version_document["chromeVersion"]
    if change == "version":
        version = "153.0.8010.37"
        version_document["chromeVersion"] = version
    base = f"publisher-chrome/{version}"
    for name, raw in contents.items():
        target = root / base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    version_raw = canonical_json_bytes(version_document)
    version_path = root / base / "manifest.json"
    version_path.write_bytes(version_raw)
    refs = {row["role"]: row for row in document["files"]}
    for role, name, raw in (
        ("chrome_executable", "chrome.exe", contents["chrome.exe"]),
        ("chrome_version_directory_manifest", "manifest.json", version_raw),
    ):
        refs[role].update(
            relativePath=base + "/" + name,
            basename=name,
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )
    document["expected"]["chromeVersion"] = version
    return graph


def test_final_binding_requires_same_candidate_and_expires_with_input_seals(
    monkeypatch, tmp_path, source
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_chrome_files import bind_chrome_files, chrome_file_metadata

    admitted = _admit(monkeypatch, _archive())
    graph = _graph(tmp_path, source, admitted)
    with source[0]["freeze"](graph) as files:
        candidate = bind_candidate_files(files, graph[3], graph[4])
        bound = bind_chrome_files(admitted, files, candidate)
        metadata = chrome_file_metadata(bound)
        assert metadata.candidate_commit == graph[4].candidate_head_oid
        assert metadata.distribution.file_count == 3
        other_parent = tmp_path / "another-input"
        other_parent.mkdir()
        other = _graph(other_parent, source, admitted, change="changed_resource")
        with (
            source[0]["freeze"](other) as other_files,
            pytest.raises(ValueError, match="same final input"),
        ):
            bind_chrome_files(admitted, other_files, candidate)
    with pytest.raises(ValueError, match="closed"):
        chrome_file_metadata(bound)


@pytest.mark.parametrize("change", ["changed_resource", "missing_resource", "version"])
def test_rehashed_final_manifest_cannot_authorize_a_changed_publisher_directory(
    monkeypatch, tmp_path, source, change
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_chrome_files import bind_chrome_files

    admitted = _admit(monkeypatch, _archive())
    graph = _graph(tmp_path, source, admitted, change=change)
    with source[0]["freeze"](graph) as files:
        candidate = bind_candidate_files(files, graph[3], graph[4])
        with pytest.raises(ValueError, match="Chrome final"):
            bind_chrome_files(admitted, files, candidate)
