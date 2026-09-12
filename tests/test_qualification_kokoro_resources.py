"""Provider resources need candidate authority beyond their own manifest hashes."""

import hashlib
import json
import os
from pathlib import Path
from runpy import run_path

import pytest

from scripts.qualify_evidence_slice_zero import canonical_json_bytes

_helpers = run_path(str(Path(__file__).with_name("test_qualification_candidate_files.py")))
bound_graph = _helpers["bound_graph"]
dependency_graph = _helpers["dependency_graph"]
_MODEL = b"synthetic model bytes"
_VOICES = b"synthetic voices bytes"
_RESOURCES = {"kokoro-v1.0.onnx": _MODEL, "voices-v1.0.bin": _VOICES}
_PURPOSES = ("realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime")


def _provider_source():
    # This module would fail if imported. The verifier must only read its literals.
    raw = 'raise RuntimeError("provider construction is forbidden")\n'
    for variable, name in (
        ("_MODEL_ASSET", "kokoro-v1.0.onnx"),
        ("_VOICES_ASSET", "voices-v1.0.bin"),
    ):
        payload = _RESOURCES[name]
        digest = hashlib.sha256(payload).hexdigest()
        raw += (
            f"{variable} = _PinnedAsset(filename={name!r}, "
            f"size={len(payload)}, sha256={digest!r})\n"
        )
    return raw.encode()


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _helpers["_make_source"](
        tmp_path_factory, {"hermes_realtime/providers/kokoro.py": _provider_source()}
    )


def _replace_resources(graph, resources):
    root, _, document, _, _ = graph
    reference = next(row for row in document["files"] if row["role"] == "kokoro_model_manifest")
    path = root / reference["relativePath"]
    manifest = json.loads(path.read_bytes())
    manifest["resources"] = []
    for name, raw in sorted(resources.items()):
        target = path.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest["resources"].append(
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest["modelIdentitySha256"] = hashlib.sha256(
        canonical_json_bytes(
            {row["name"]: row["sha256"] for row in manifest["resources"]}, terminal_lf=False
        )
    ).hexdigest()
    raw = canonical_json_bytes(manifest)
    path.write_bytes(raw)
    reference.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def test_resource_capability_cannot_be_supplied():
    from scripts.qualification_kokoro_resources import (
        BoundKokoroResourcesV1,
        kokoro_resource_metadata,
    )

    with pytest.raises(TypeError):
        BoundKokoroResourcesV1()
    with pytest.raises(TypeError):
        kokoro_resource_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        kokoro_resource_metadata(object.__new__(BoundKokoroResourcesV1))


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
@pytest.mark.parametrize("purpose", _PURPOSES)
def test_candidate_selected_resources_bind_before_any_provider_import_and_expire(
    dependency_graph, purpose
):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_purpose
    from scripts.qualification_kokoro_resources import (
        _kokoro_resources_for_consumer,
        bind_kokoro_resources,
        kokoro_resource_metadata,
    )

    _replace_resources(dependency_graph, _RESOURCES)
    with _helpers["freeze"](dependency_graph) as files:
        candidate = bind_candidate_files(files, dependency_graph[3], dependency_graph[4])
        dependencies = bind_dependency_purpose(files, candidate, purpose=purpose)
        receipt = bind_kokoro_resources(files, candidate, dependencies, purpose=purpose)
        metadata = kokoro_resource_metadata(receipt)
        assert metadata.purpose == purpose
        assert metadata.resource_count == 2
        assert metadata.resource_bytes == len(_MODEL) + len(_VOICES)
        assert dict(_kokoro_resources_for_consumer(receipt)) == _RESOURCES
        # A second source binding is not this dependency capability's authority.
        replacement = bind_candidate_files(files, dependency_graph[3], dependency_graph[4])
        with pytest.raises(ValueError, match="same candidate"):
            bind_kokoro_resources(files, replacement, dependencies, purpose=purpose)
    with pytest.raises(ValueError, match="closed"):
        kokoro_resource_metadata(receipt)
    with pytest.raises(ValueError, match="closed"):
        _kokoro_resources_for_consumer(receipt)


@pytest.mark.skipif(os.name != "nt", reason="genuine retained Windows inputs")
@pytest.mark.parametrize("fault", ["model", "voices", "missing", "extra", "renamed", "purpose"])
def test_rehashed_model_manifest_cannot_replace_candidate_selected_bytes(dependency_graph, fault):
    from scripts.qualification_candidate_files import bind_candidate_files
    from scripts.qualification_dependency_files import bind_dependency_purpose
    from scripts.qualification_kokoro_resources import bind_kokoro_resources

    resources = dict(_RESOURCES)
    if fault in {"model", "voices"}:
        name = "kokoro-v1.0.onnx" if fault == "model" else "voices-v1.0.bin"
        resources[name] += b"replacement"
    elif fault == "missing":
        del resources["voices-v1.0.bin"]
    elif fault == "extra":
        resources["extra.bin"] = b"extra"
    elif fault == "renamed":
        resources["renamed.bin"] = resources.pop("voices-v1.0.bin")
    _replace_resources(dependency_graph, resources)
    with _helpers["freeze"](dependency_graph) as files:
        candidate = bind_candidate_files(files, dependency_graph[3], dependency_graph[4])
        dependencies = bind_dependency_purpose(files, candidate, purpose=_PURPOSES[0])
        with pytest.raises(ValueError, match="resource|coverage"):
            bind_kokoro_resources(
                files,
                candidate,
                dependencies,
                purpose=_PURPOSES[1] if fault == "purpose" else _PURPOSES[0],
            )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "computed", "bool_size", "traversal", "duplicate_name"]
)
def test_unavailable_or_ambiguous_source_asset_selection_refuses(mutation):
    from scripts.qualification_kokoro_resources import _source_assets

    raw = _provider_source()
    if mutation == "missing":
        raw = raw.replace(b"_MODEL_ASSET =", b"OTHER =")
    elif mutation == "duplicate":
        raw += raw
    elif mutation == "computed":
        raw = raw.replace(b"size=21", b"size=int('21')")
    elif mutation == "bool_size":
        raw = raw.replace(b"size=21", b"size=True")
    elif mutation == "traversal":
        raw = raw.replace(b"kokoro-v1.0.onnx", b"../model.onnx")
    else:
        raw = raw.replace(b"voices-v1.0.bin", b"kokoro-v1.0.onnx")
    with pytest.raises(ValueError, match="asset"):
        _source_assets(raw)


def test_current_candidate_kokoro_assets_use_supported_literal_selection():
    from scripts.qualification_kokoro_resources import _source_assets

    source = Path("src/hermes_realtime/providers/kokoro.py").read_bytes()
    assets = _source_assets(source)
    assert tuple(row.name for row in assets) == ("kokoro-v1.0.onnx", "voices-v1.0.bin")
    assert all(row.size > 1024**2 for row in assets)
