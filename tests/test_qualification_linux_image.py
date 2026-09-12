"""An admitted OCI descriptor is not a running Linux environment or receipt."""

import hashlib
import json
from dataclasses import replace

import pytest


@pytest.fixture
def image_bytes(monkeypatch):
    from scripts import qualification_linux_image as images

    config = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {"Env": ["PYTHON_VERSION=3.11.16"]},
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "d" * 64]},
        }
    ).encode()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:" + hashlib.sha256(config).hexdigest(),
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": "sha256:" + "c" * 64,
                    "size": 10,
                }
            ],
        }
    ).encode()
    # Substitute trust only for parser tests, never as publisher or Linux proof.
    monkeypatch.setattr(
        images,
        "_POLICY",
        replace(
            images._POLICY,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            manifest_bytes=len(manifest),
        ),
    )
    return manifest, config


def test_image_descriptor_capabilities_are_admission_minted_only():
    from scripts.qualification_linux_image import AdmittedLinuxRuntimeImageV1, linux_image_metadata

    with pytest.raises(TypeError):
        AdmittedLinuxRuntimeImageV1()
    with pytest.raises(TypeError):
        linux_image_metadata({"passed": True})
    with pytest.raises(ValueError, match="unregistered"):
        linux_image_metadata(object.__new__(AdmittedLinuxRuntimeImageV1))


def test_complete_descriptor_graph_retains_only_its_admitted_image_identity(image_bytes):
    from scripts.qualification_linux_image import admit_linux_runtime_image, linux_image_metadata

    manifest, config = image_bytes
    receipt = admit_linux_runtime_image(manifest, config)
    value = linux_image_metadata(receipt)
    assert (
        value.image_reference
        == "docker.io/library/python@sha256:" + hashlib.sha256(manifest).hexdigest()
    )
    assert value.config_sha256 == hashlib.sha256(config).hexdigest()
    assert value.python_version == "3.11.16"
    assert value.layer_sha256s == ("c" * 64,)
    assert value.layer_diff_sha256s == ("d" * 64,)


@pytest.mark.parametrize("target", ["manifest", "config"])
def test_changed_image_descriptors_refuse_before_their_json_is_parsed(
    image_bytes, monkeypatch, target
):
    from scripts import qualification_linux_image as images

    manifest, config = image_bytes
    original = images.json.loads
    changed = manifest + b" " if target == "manifest" else config + b" "

    def admitted_only(raw, **kwargs):
        assert raw != changed, "Unadmitted descriptor reached parsing"
        return original(raw, **kwargs)

    monkeypatch.setattr(images.json, "loads", admitted_only)
    with pytest.raises(ValueError):
        images.admit_linux_runtime_image(
            changed if target == "manifest" else manifest, changed if target == "config" else config
        )


@pytest.mark.parametrize("fault", ["architecture", "python", "rootfs", "layers"])
def test_publisher_descriptor_still_has_to_match_the_selected_execution_profile(
    image_bytes, monkeypatch, fault
):
    from scripts import qualification_linux_image as images

    manifest, config = image_bytes
    value = json.loads(config)
    if fault == "architecture":
        value["architecture"] = "arm64"
    elif fault == "python":
        value["config"]["Env"] = ["PYTHON_VERSION=3.12.0"]
    elif fault == "rootfs":
        value["rootfs"]["type"] = "unavailable"
    else:
        value["rootfs"]["diff_ids"] = []
    config = json.dumps(value).encode()
    value = json.loads(manifest)
    value["config"].update(digest="sha256:" + hashlib.sha256(config).hexdigest(), size=len(config))
    manifest = json.dumps(value).encode()
    monkeypatch.setattr(
        images,
        "_POLICY",
        replace(
            images._POLICY,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            manifest_bytes=len(manifest),
        ),
    )
    with pytest.raises(ValueError):
        images.admit_linux_runtime_image(manifest, config)
