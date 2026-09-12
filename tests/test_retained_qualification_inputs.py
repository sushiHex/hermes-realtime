"""Input graph leases retain bytes; they never imply build or installed execution."""

import hashlib
import json
import os
from pathlib import Path
from runpy import run_path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows retained input graph")


@pytest.fixture
def graph(tmp_path):
    helpers = run_path(str(Path.cwd() / "tests/test_qualify_evidence_slice_zero.py"))
    root, manifest, _, _, _ = helpers["_make_input_root"](tmp_path)
    return root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_direct_and_transitive_graph_is_retained_and_expires(graph):
    from scripts.retained_qualification_inputs import (
        retain_qualification_input_files,
        retained_input_metadata,
    )

    root, manifest, digest = graph
    with retain_qualification_input_files(root, manifest.name, digest) as receipt:
        metadata = retained_input_metadata(receipt)
        assert metadata.qualification_input_sha256 == digest
        assert any(
            item.logical_id.startswith("provider:moonshine:resource:")
            for item in metadata.verified_artifacts
        )
        assert any(
            item.logical_id.startswith("chrome:file:") for item in metadata.verified_artifacts
        )
        document = json.loads(manifest.read_bytes())
        roles = {item["role"]: root / item["relativePath"] for item in document["files"]}
        selected = [manifest, *roles.values()]
        selected.extend(
            root / tool["artifact"]["relativePath"] for tool in document["toolIdentities"]
        )
        for role in ("moonshine_model_manifest", "kokoro_model_manifest"):
            selected.extend(
                roles[role].parent / item["name"]
                for item in json.loads(roles[role].read_bytes())["resources"]
            )
        for role in roles:
            if role.endswith("wheelhouse_manifest"):
                wheelhouse = json.loads(roles[role].read_bytes())
                selected.extend(root / item["relativePath"] for item in wheelhouse["wheels"])
                selected.extend(
                    root / wheelhouse[key]["relativePath"]
                    for key in ("requirements", "constraints")
                )
        chrome = roles["chrome_version_directory_manifest"]
        selected.extend(
            chrome.parent / item["name"] for item in json.loads(chrome.read_bytes())["files"]
        )
        for file in selected:
            with pytest.raises(OSError), file.open("r+b"):
                pytest.fail("declared input could be opened for writing")
    with pytest.raises(ValueError, match="closed"):
        retained_input_metadata(receipt)


def test_changed_transitive_resource_or_manifest_is_rejected(graph):
    from scripts.retained_qualification_inputs import retain_qualification_input_files

    root, manifest, digest = graph
    document = json.loads(manifest.read_bytes())
    provider = next(
        item for item in document["files"] if item["role"] == "moonshine_model_manifest"
    )
    provider_path = root / provider["relativePath"]
    resources = json.loads(provider_path.read_bytes())["resources"]
    (provider_path.parent / resources[0]["name"]).write_bytes(b"changed synthetic resource")
    with (
        pytest.raises(ValueError, match="resource bytes do not match"),
        retain_qualification_input_files(root, manifest.name, digest),
    ):
        pytest.fail("changed provider accepted")


def test_role_hard_links_cannot_substitute_for_distinct_outputs(graph):
    from scripts.retained_qualification_inputs import retain_qualification_input_files

    root, manifest, digest = graph
    document = json.loads(manifest.read_bytes())
    first = next(item for item in document["files"] if item["role"] == "direct_wheel")
    second = next(item for item in document["files"] if item["role"] == "direct_wheel_repeat")
    target = root / second["relativePath"]
    target.unlink()
    os.link(root / first["relativePath"], target)
    with (
        pytest.raises(ValueError, match="hard link"),
        retain_qualification_input_files(root, manifest.name, digest),
    ):
        pytest.fail("aliased repeated build accepted")


def test_supplied_metadata_cannot_be_used_as_a_live_lease(graph):
    from scripts.qualify_evidence_slice_zero import QualificationInputClosure
    from scripts.retained_qualification_inputs import retained_input_metadata

    with pytest.raises(TypeError):
        retained_input_metadata(QualificationInputClosure(graph[2], ()))
