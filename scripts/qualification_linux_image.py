"""Admit a complete Linux runtime image descriptor from reviewed publisher bytes.

This authenticates the selected OCI graph, not materialized layers, an immutable
container, actual Linux execution or a service-authenticated qualification receipt.
Those separate authorities must verify the graph before using its runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from weakref import WeakKeyDictionary


@dataclass(frozen=True, slots=True)
class _ImagePolicy:
    manifest_sha256: str
    manifest_bytes: int
    python_version: str


# Official Docker Python 3.11.16 slim-bookworm, linux/amd64 manifest (not its
# multi-platform index). Publisher metadata and the complete config were read
# through default Windows HTTPS certificate validation; their byte digests agree.
# https://hub.docker.com/v2/repositories/library/python/tags/3.11.16-slim-bookworm
# https://github.com/docker-library/python/tree/688a0b86bb44289df16a363e9f41d90514c1a5f9/3.11/slim-bookworm
# Admission changes require source review, never a caller-selected image or tag.
_POLICY = _ImagePolicy(
    "b1add8a6f2aca6bcfcf0b9c9b522352f7ce0d62a3d556a2f2f32511aa0cca250",
    1752,
    "3.11.16",
)
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: object) -> str:
    _require(type(value) is str, "Linux image digest type differs")
    assert isinstance(value, str)
    match = _DIGEST.fullmatch(value)
    _require(match is not None, "Linux image digest differs")
    assert match is not None
    return match.group(1)


@dataclass(frozen=True, slots=True)
class LinuxRuntimeImageMetadataV1:
    image_reference: str
    python_version: str
    config_sha256: str
    layer_sha256s: tuple[str, ...]
    layer_diff_sha256s: tuple[str, ...]
    compressed_bytes: int


class AdmittedLinuxRuntimeImageV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Linux runtime image descriptors are admission-minted only")


_ADMITTED: WeakKeyDictionary[AdmittedLinuxRuntimeImageV1, LinuxRuntimeImageMetadataV1] = (
    WeakKeyDictionary()
)


def admit_linux_runtime_image(manifest: bytes, config: bytes) -> AdmittedLinuxRuntimeImageV1:
    """Authenticate the approved manifest before following its config reference."""
    _require(
        type(manifest) is bytes
        and len(manifest) == _POLICY.manifest_bytes
        and hashlib.sha256(manifest).hexdigest() == _POLICY.manifest_sha256,
        "Linux image manifest differs from the admitted publisher distribution",
    )
    document = json.loads(manifest)
    _require(
        document["schemaVersion"] == 2
        and document["mediaType"] == "application/vnd.oci.image.manifest.v1+json",
        "Linux image manifest profile differs",
    )
    reference = document["config"]
    digest = _sha256(reference["digest"])
    _require(
        reference["mediaType"] == "application/vnd.oci.image.config.v1+json"
        and type(config) is bytes
        and 0 < len(config) == reference["size"] <= 64 * 1024
        and hashlib.sha256(config).hexdigest() == digest,
        "Linux image config differs from its admitted manifest",
    )
    configuration = json.loads(config)
    _require(
        configuration["architecture"] == "amd64"
        and configuration["os"] == "linux"
        and "PYTHON_VERSION=" + _POLICY.python_version in configuration["config"]["Env"]
        and configuration["rootfs"]["type"] == "layers",
        "Linux image runtime profile differs",
    )
    layers = document["layers"]
    diff_ids = configuration["rootfs"]["diff_ids"]
    _require(
        type(layers) is list and type(diff_ids) is list and 0 < len(layers) == len(diff_ids) <= 16,
        "Linux image layer graph differs",
    )
    _require(
        all(
            layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar+gzip"
            and type(layer["size"]) is int
            and 0 < layer["size"] <= 256 * 1024**2
            for layer in layers
        ),
        "Linux image layer profile differs",
    )
    total = sum(layer["size"] for layer in layers)
    _require(total <= 512 * 1024**2, "Linux image layers exceed their bound")
    metadata = LinuxRuntimeImageMetadataV1(
        "docker.io/library/python@sha256:" + _POLICY.manifest_sha256,
        _POLICY.python_version,
        digest,
        tuple(_sha256(layer["digest"]) for layer in layers),
        tuple(_sha256(value) for value in diff_ids),
        total,
    )
    receipt = object.__new__(AdmittedLinuxRuntimeImageV1)
    _ADMITTED[receipt] = metadata
    return receipt


def linux_image_metadata(receipt: AdmittedLinuxRuntimeImageV1) -> LinuxRuntimeImageMetadataV1:
    if type(receipt) is not AdmittedLinuxRuntimeImageV1:
        raise TypeError("Linux runtime image capability type differs")
    _require(receipt in _ADMITTED, "Linux runtime image capability is unregistered")
    return _ADMITTED[receipt]
