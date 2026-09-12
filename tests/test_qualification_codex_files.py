"""Synthetic package tests do not establish Codex execution or installation."""

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained file authority")
_MATCHING_EXECUTABLE = b"codex executable\n"


def _archive(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


def _distribution(monkeypatch, *, executable: bytes = _MATCHING_EXECUTABLE):
    from scripts import qualification_codex_files as codex

    files = {
        "bin/codex-code-mode-host.exe": b"code mode host\n",
        "bin/codex.exe": executable,
        "codex-package.json": json.dumps(
            {
                "layoutVersion": 1,
                "version": "0.145.0",
                "target": "x86_64-pc-windows-msvc",
                "variant": "codex",
                "entrypoint": "bin/codex.exe",
                "resourcesDir": "codex-resources",
                "pathDir": "codex-path",
            },
            separators=(",", ":"),
        ).encode(),
        "codex-path/rg.exe": b"rg\n",
        "codex-resources/codex-command-runner.exe": b"command runner\n",
        "codex-resources/codex-windows-sandbox-setup.exe": b"sandbox setup\n",
    }
    raw = _archive(files)
    members = tuple(
        codex._Member(name, len(value), hashlib.sha256(value).hexdigest())
        for name, value in sorted(files.items())
    )
    monkeypatch.setattr(codex, "_ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(codex, "_ARCHIVE_BYTES", len(raw))
    monkeypatch.setattr(codex, "_MEMBERS", members)
    return codex.admit_codex_distribution(raw), files, raw


def _candidate(tmp_path_factory, executable_sha256: str):
    from scripts.candidate_source_archive_oracle import capture_candidate_source_archive

    helpers = run_path(str(Path.cwd() / "tests/test_qualification_candidate_files.py"))
    archive_helpers = run_path(str(Path.cwd() / "tests/test_candidate_source_archive_oracle.py"))
    repository, _, prior, wheel = helpers["_make_source"](tmp_path_factory)
    fixture = repository / "tests/fixtures/codex_app_server_dynamic_tools.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(
        json.dumps(
            {
                "codexVersion": "codex-cli 0.145.0",
                "codexBinarySha256": executable_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    archive_helpers["_git"]("add", ".", cwd=repository)
    archive_helpers["_git"]("commit", "-qm", "Codex fixture", cwd=repository)
    identity = archive_helpers["_identity"](repository, prior.canonical_baseline_oid)
    archive = capture_candidate_source_archive(repository, identity, archive_helpers["_pin"]())
    return helpers, (repository, archive, identity, wheel)


def _graph(tmp_path, source, executable: bytes, *, change: str | None = None):
    helpers, candidate = source
    graph = helpers["bound_graph"].__wrapped__(tmp_path, candidate)
    root, _, document, _, _ = graph
    reference = next(item for item in document["files"] if item["role"] == "codex_executable")
    raw = executable if change != "executable" else executable + b"altered"
    path = root / reference["relativePath"]
    path.write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    document["expected"].update(
        codexVersion="0.145.0" if change != "version" else "0.145.1",
        codexModel="gpt-5.6-terra" if change != "model" else "other",
        codexEffort="low" if change != "effort" else "high",
    )
    return graph


@pytest.fixture(scope="module")
def matching_source(tmp_path_factory):
    """One genuine candidate archive shared by all matching final-binding cases."""
    return _candidate(tmp_path_factory, hashlib.sha256(_MATCHING_EXECUTABLE).hexdigest())


def test_publisher_digest_is_checked_before_archive_parsing():
    from scripts.qualification_codex_files import admit_codex_distribution

    with pytest.raises(ValueError, match="publisher distribution"):
        admit_codex_distribution(b"not the governed release")


def test_admission_retains_complete_publisher_namespace_and_layout(monkeypatch):
    from scripts.qualification_codex_files import (
        _codex_distribution_files,
        codex_distribution_metadata,
    )

    admitted, files, _ = _distribution(monkeypatch)
    metadata = codex_distribution_metadata(admitted)
    assert metadata.version == "0.145.0"
    assert metadata.file_count == 6
    assert metadata.executable_sha256 == hashlib.sha256(files["bin/codex.exe"]).hexdigest()
    assert dict(_codex_distribution_files(admitted)) == files


@pytest.mark.parametrize(
    "mutate",
    [
        lambda files: files.pop("codex-path/rg.exe"),
        lambda files: files.__setitem__("bin/CODEX.exe", b"ambiguous"),
        lambda files: files.__setitem__("codex-package.json", b'{"layoutVersion":true}'),
    ],
)
def test_incomplete_or_changed_publisher_package_refuses(monkeypatch, mutate):
    from scripts import qualification_codex_files as codex

    _, files, _ = _distribution(monkeypatch)
    mutate(files)
    raw = _archive(files)
    monkeypatch.setattr(codex, "_ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(codex, "_ARCHIVE_BYTES", len(raw))
    with pytest.raises(ValueError):
        codex.admit_codex_distribution(raw)


def test_distribution_cannot_be_forged():
    from scripts.qualification_codex_files import (
        AdmittedCodexDistributionV1,
        codex_distribution_metadata,
    )

    with pytest.raises(TypeError):
        AdmittedCodexDistributionV1()
    with pytest.raises(ValueError, match="unregistered"):
        codex_distribution_metadata(object.__new__(AdmittedCodexDistributionV1))


def test_final_binder_requires_candidate_fixture_and_sealed_executable(
    monkeypatch, matching_source, tmp_path
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_codex_files import bind_codex_files, codex_file_metadata

    admitted, files, _ = _distribution(monkeypatch)
    graph = _graph(tmp_path, matching_source, files["bin/codex.exe"])
    with matching_source[0]["freeze"](graph) as final:
        candidate = bind_candidate_files(final, graph[3], graph[4])
        bound = bind_codex_files(admitted, final, candidate)
        metadata = codex_file_metadata(bound)
        assert metadata.candidate_commit == graph[4].candidate_head_oid
        assert (metadata.version, metadata.model, metadata.effort) == (
            "0.145.0",
            "gpt-5.6-terra",
            "low",
        )
    with pytest.raises(ValueError, match="closed"):
        codex_file_metadata(bound)


@pytest.mark.parametrize("change", ["executable", "version", "model", "effort"])
def test_final_binder_refuses_changed_final_bytes_or_expected_provider_selection(
    monkeypatch, matching_source, tmp_path, change
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_codex_files import bind_codex_files

    admitted, files, _ = _distribution(monkeypatch)
    graph = _graph(tmp_path, matching_source, files["bin/codex.exe"], change=change)
    with matching_source[0]["freeze"](graph) as final:
        candidate = bind_candidate_files(final, graph[3], graph[4])
        with pytest.raises(ValueError, match="Codex final"):
            bind_codex_files(admitted, final, candidate)


def test_final_binder_refuses_actual_candidate_fixture_drift(
    monkeypatch, tmp_path_factory, tmp_path
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_codex_files import bind_codex_files

    admitted, files, _ = _distribution(monkeypatch)
    source = _candidate(tmp_path_factory, "0" * 64)
    graph = _graph(tmp_path, source, files["bin/codex.exe"])
    with source[0]["freeze"](graph) as final:
        candidate = bind_candidate_files(final, graph[3], graph[4])
        with pytest.raises(ValueError, match="Codex final"):
            bind_codex_files(admitted, final, candidate)
