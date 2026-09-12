"""Final Moonshine resources bind pre-final acquisition to final sealed bytes."""

import base64
import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

from scripts.qualify_evidence_slice_zero import canonical_json_bytes

_helpers = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))
bound_graph = _helpers["bound_graph"]
dependency_graph = _helpers["dependency_graph"]
_PURPOSES = ("realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _helpers["_make_source"](tmp_path_factory)


def _resources():
    from scripts.qualification_moonshine_catalog import _MODEL_BASE, _SPELLING_BASE, _Resource
    from scripts.qualification_moonshine_preparation import PreparedMoonshineResourceV1
    from scripts.qualification_moonshine_worker import _crc32c

    rows = []
    payloads = {}
    for group, base, names in (
        ("primary", _MODEL_BASE, ("a.ort", "b.ort", "c.ort", "d.ort", "e.ort", "f.json", "g.bin")),
        ("spelling", _SPELLING_BASE, ("a.ort", "b.json")),
    ):
        for name in names:
            url = f"{base}/{name}"
            cache_path = url.removeprefix("https://")
            payload = f"publisher:{cache_path}".encode()
            crc = base64.b64encode(_crc32c(payload).to_bytes(4, "big")).decode("ascii")
            sha256 = hashlib.sha256(payload).hexdigest()
            rows.append(
                (
                    _Resource(group, cache_path, url, len(payload), crc),
                    PreparedMoonshineResourceV1(group, cache_path, url, len(payload), crc, sha256),
                )
            )
            payloads[cache_path] = payload
    return tuple(rows), payloads


def _replace_final_resources(graph, payloads):
    root, _, document, _, _ = graph
    reference = next(row for row in document["files"] if row["role"] == "moonshine_model_manifest")
    manifest_path = root / reference["relativePath"]
    if manifest_path.parent.exists():
        for path in tuple(manifest_path.parent.rglob("*")):
            if path.is_file():
                path.unlink()
    resources = []
    for name, raw in sorted(payloads.items()):
        target = manifest_path.parent.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        resources.append(
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    model_identity = hashlib.sha256(
        canonical_json_bytes({row["name"]: row["sha256"] for row in resources}, terminal_lf=False)
    ).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "provider": "moonshine",
        "modelIdentitySha256": model_identity,
        "resources": resources,
    }
    raw = canonical_json_bytes(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return model_identity


def test_final_resource_capability_cannot_be_supplied():
    from scripts.qualification_moonshine_resources import (
        BoundMoonshineResourcesV1,
        moonshine_resource_metadata,
    )

    with pytest.raises(TypeError):
        BoundMoonshineResourcesV1()
    with pytest.raises(TypeError):
        moonshine_resource_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        moonshine_resource_metadata(object.__new__(BoundMoonshineResourcesV1))


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
@pytest.mark.parametrize("purpose", _PURPOSES)
def test_final_binding_survives_clean_original_cleanup_and_expires_with_final_seals(
    dependency_graph, tmp_path, monkeypatch, purpose
):
    from scripts.qualification_file_seals import retain_file_seals
    from scripts.qualification_moonshine_resources import (
        _moonshine_resources_for_consumer,
        bind_moonshine_resources,
        moonshine_resource_metadata,
    )

    pairs, payloads = _resources()
    model_identity = _replace_final_resources(dependency_graph, payloads)
    original = tmp_path / "prepared"
    for name, raw in payloads.items():
        target = original.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    state = SimpleNamespace(_closing=False, _closed=False, _unrecoverable=None)
    with _helpers["freeze"](dependency_graph) as files:
        with retain_file_seals(original, tuple(sorted(payloads))) as original_seals:
            receipt, prepared = _authorities(
                monkeypatch,
                files,
                pairs,
                original,
                original_seals,
                state,
                purpose,
            )
            bound = bind_moonshine_resources(*receipt, purpose=purpose)
            with pytest.raises(ValueError, match="cleanup is incomplete"):
                moonshine_resource_metadata(bound)
            with pytest.raises(ValueError, match="cleanup is incomplete"):
                _moonshine_resources_for_consumer(bound)
        monkeypatch.setattr(
            "scripts.qualification_moonshine_resources._moonshine_preparation_for_consumer",
            lambda selected: pytest.fail("closed preparation receipt was reused"),
        )
        shutil.rmtree(original)
        state._closing = state._closed = True
        metadata = moonshine_resource_metadata(bound)
        assert metadata.resource_count == 9
        assert metadata.model_identity_sha256 == model_identity
        assert dict(_moonshine_resources_for_consumer(bound)) == payloads
    with pytest.raises(ValueError, match="closed"):
        moonshine_resource_metadata(bound)


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
def test_final_binding_rejects_mismatched_source_wheel_catalog_and_graph_authorities(
    dependency_graph, tmp_path, monkeypatch
):
    from scripts.qualification_file_seals import retain_file_seals
    from scripts.qualification_moonshine_resources import bind_moonshine_resources

    pairs, payloads = _resources()
    _replace_final_resources(dependency_graph, payloads)
    original = tmp_path / "prepared"
    for name, raw in payloads.items():
        target = original.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    state = SimpleNamespace(_closing=False, _closed=False, _unrecoverable=None)
    with (
        _helpers["freeze"](dependency_graph) as files,
        retain_file_seals(original, tuple(sorted(payloads))) as original_seals,
    ):
        for mutation in (
            "preparation_source",
            "catalog_source",
            "distribution",
            "catalog_purpose",
            "catalog_identity",
            "catalog_resource",
            "catalog_candidate",
            "catalog_dependencies",
        ):
            receipt, _ = _authorities(
                monkeypatch,
                files,
                pairs,
                original,
                original_seals,
                state,
                _PURPOSES[0],
                mutation,
            )
            with pytest.raises(ValueError):
                bind_moonshine_resources(*receipt, purpose=_PURPOSES[0])


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
def test_final_binding_rejects_a_valid_but_different_final_resource_closure(
    dependency_graph, tmp_path, monkeypatch
):
    from scripts.qualification_file_seals import retain_file_seals
    from scripts.qualification_moonshine_resources import bind_moonshine_resources

    pairs, payloads = _resources()
    changed = dict(payloads)
    selected = next(iter(changed))
    changed[selected] += b"changed"
    _replace_final_resources(dependency_graph, changed)
    original = tmp_path / "prepared"
    for name, raw in payloads.items():
        target = original.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    state = SimpleNamespace(_closing=False, _closed=False, _unrecoverable=None)
    with (
        _helpers["freeze"](dependency_graph) as files,
        retain_file_seals(original, tuple(sorted(payloads))) as original_seals,
    ):
        receipt, _ = _authorities(
            monkeypatch,
            files,
            pairs,
            original,
            original_seals,
            state,
            _PURPOSES[0],
        )
        with pytest.raises(ValueError, match="final resource manifest"):
            bind_moonshine_resources(*receipt, purpose=_PURPOSES[0])


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
def test_bound_resources_refuse_pending_or_failed_preparation_cleanup(
    dependency_graph, tmp_path, monkeypatch
):
    from scripts.qualification_file_seals import retain_file_seals
    from scripts.qualification_moonshine_resources import (
        bind_moonshine_resources,
        moonshine_resource_metadata,
    )

    pairs, payloads = _resources()
    _replace_final_resources(dependency_graph, payloads)
    original = tmp_path / "prepared"
    for name, raw in payloads.items():
        target = original.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    state = SimpleNamespace(_closing=False, _closed=False, _unrecoverable=None)
    with (
        _helpers["freeze"](dependency_graph) as files,
        retain_file_seals(original, tuple(sorted(payloads))) as original_seals,
    ):
        receipt, _ = _authorities(
            monkeypatch,
            files,
            pairs,
            original,
            original_seals,
            state,
            _PURPOSES[0],
        )
        bound = bind_moonshine_resources(*receipt, purpose=_PURPOSES[0])
        with pytest.raises(ValueError, match="cleanup is incomplete"):
            moonshine_resource_metadata(bound)
        state._closing = True
        with pytest.raises(ValueError, match="cleanup is incomplete"):
            moonshine_resource_metadata(bound)
        state._closed = True
        state._unrecoverable = RuntimeError("cleanup failed")
        with pytest.raises(ValueError, match="cleanup is incomplete"):
            moonshine_resource_metadata(bound)
        state._unrecoverable = None
        assert moonshine_resource_metadata(bound).resource_count == 9


def _authorities(
    monkeypatch,
    files,
    pairs,
    original,
    original_seals,
    state,
    purpose,
    mutation=None,
):
    from scripts import qualification_moonshine_resources as binding
    from scripts.retained_qualification_inputs import retained_input_metadata

    prepared_receipt, catalog_receipt = object(), object()
    candidate, dependencies = object(), object()
    final = retained_input_metadata(files)
    source_commit, source_tree = "a" * 40, "b" * 40
    source_archive_sha256 = "f" * 64
    distribution_sha256 = "c" * 64
    catalog_identity = "d" * 64
    prepared_rows = tuple(row[1] for row in pairs)
    catalog_rows = tuple(row[0] for row in pairs)
    preparation_metadata = SimpleNamespace(
        source_commit=source_commit,
        source_tree=source_tree,
        source_archive_sha256=source_archive_sha256,
        worker_sha256="e" * 64,
        distribution_sha256=distribution_sha256,
        python_api_sha256="1" * 64,
        native_library_sha256="2" * 64,
        catalog_identity_sha256=catalog_identity,
        resource_count=9,
        resource_bytes=sum(row.size for row in prepared_rows),
    )
    preparation_value = SimpleNamespace(
        resource_work=state,
        resource_root=original,
        resource_seals=original_seals,
        resources=prepared_rows,
        metadata=preparation_metadata,
    )
    catalog_metadata = SimpleNamespace(
        qualification_input_sha256=final.qualification_input_sha256,
        source_commit=source_commit,
        source_tree=source_tree,
        purpose=purpose,
        worker_sha256="e" * 64,
        distribution_sha256=distribution_sha256,
        python_api_sha256="1" * 64,
        native_library_sha256="2" * 64,
        catalog_identity_sha256=catalog_identity,
        primary_resource_count=7,
        spelling_resource_count=2,
        resource_bytes=sum(row.size for row in prepared_rows),
    )
    if mutation == "preparation_source":
        preparation_metadata.source_tree = "9" * 40
    elif mutation == "catalog_source":
        catalog_metadata.source_commit = "9" * 40
    elif mutation == "catalog_purpose":
        catalog_metadata.purpose = next(item for item in _PURPOSES if item != purpose)
    elif mutation == "catalog_identity":
        preparation_metadata.catalog_identity_sha256 = "9" * 64
    elif mutation == "catalog_resource":
        catalog_rows = (replace(catalog_rows[0], url="https://example.invalid/a.ort"),) + (
            catalog_rows[1:]
        )
    closure = SimpleNamespace(
        files=files,
        candidate=candidate,
        metadata=SimpleNamespace(
            qualification_input_sha256=final.qualification_input_sha256,
            wheelhouses=(
                SimpleNamespace(
                    purpose=purpose,
                    distributions=(
                        SimpleNamespace(
                            name="moonshine-voice",
                            version="0.1.0",
                            sha256=distribution_sha256,
                        ),
                    ),
                ),
            ),
        ),
    )
    if mutation == "distribution":
        closure.metadata.wheelhouses[0].distributions[0].sha256 = "9" * 64
    catalog_candidate = object() if mutation == "catalog_candidate" else candidate
    catalog_dependencies = object() if mutation == "catalog_dependencies" else dependencies
    monkeypatch.setattr(
        binding, "_candidate_files_for_consumer", lambda selected: SimpleNamespace(files=files)
    )
    monkeypatch.setattr(binding, "_dependency_binding_for_consumer", lambda selected: closure)
    monkeypatch.setattr(
        binding,
        "candidate_file_metadata",
        lambda selected: SimpleNamespace(
            qualification_input_sha256=final.qualification_input_sha256,
            source_commit=source_commit,
            source_tree=source_tree,
            source_archive_sha256=source_archive_sha256,
        ),
    )
    monkeypatch.setattr(
        binding,
        "_moonshine_preparation_for_consumer",
        lambda selected: preparation_value,
    )
    monkeypatch.setattr(
        binding,
        "_catalog_binding",
        lambda selected: SimpleNamespace(
            candidate=catalog_candidate, dependencies=catalog_dependencies
        ),
    )
    monkeypatch.setattr(binding, "moonshine_catalog_metadata", lambda selected: catalog_metadata)
    monkeypatch.setattr(
        binding,
        "_moonshine_catalog_for_consumer",
        lambda selected, include_spelling: catalog_rows,
    )
    return (
        (files, candidate, dependencies, prepared_receipt, catalog_receipt),
        preparation_value,
    )
