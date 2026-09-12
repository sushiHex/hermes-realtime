"""Strict private payload grammar for the Linux installed/null-capture receipt.

Parsing a receipt never authenticates its producer.  The GitHub Actions consumer
performs that separate service observation and mints the only usable capability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

_OID = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_PLATFORM = "linux_x86_64"
_REPOSITORY_ID = 1351392603
_WORKFLOW_ID = 345893071
_ATTEMPT = 1
_JOB = "Linux null capture"
_ARTIFACT = "hermes-realtime-linux-null-capture-receipt-v1"
_MEMBER = "linux-null-capture-receipt-v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(rows)
    _require(len(result) == len(rows), "Linux receipt has duplicate fields")
    return result


def _exact(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == keys, label + " fields differ")
    return cast(dict[str, Any], value)


def _sha(value: object, label: str) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, label + " differs")
    return value  # type: ignore[return-value]


def _positive(value: object, label: str) -> int:
    _require(type(value) is int and not isinstance(value, bool) and 0 < value <= 2**63 - 1, label)
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class LinuxReceiptPayloadV1:
    repository_id: int
    workflow_id: int
    run_id: int
    attempt: int
    candidate_commit: str
    candidate_tree: str
    source_archive_sha256: str
    workflow_sha256: str
    direct_wheel_sha256: str
    direct_wheel_bytes: int
    direct_wheel_basename: str
    wheelhouse_manifest_sha256: str
    source_lock_sha256: str
    requirements_sha256: str
    constraints_sha256: str
    wheels: tuple[tuple[str, str, int], ...]
    image_reference: str
    image_config_sha256: str
    image_layers: tuple[str, ...]
    image_diff_ids: tuple[str, ...]
    python_version: str
    soabi: str
    ext_suffix: str
    multiarch: str
    glibc: str
    interpreter_sha256: str
    stdlib_inventory_sha256: str
    installed_inventory_sha256: str
    installed_file_count: int
    import_origin_count: int
    null_capture_passed: bool
    cleanup_removed: bool


def parse_linux_receipt_payload(raw: bytes) -> LinuxReceiptPayloadV1:
    """Return immutable validated data; this does not authenticate a producer."""
    _require(type(raw) is bytes and 0 < len(raw) <= 128 * 1024, "Linux receipt payload is bounded")
    _require(raw.endswith(b"\n") and b"\r" not in raw, "Linux receipt payload is not canonical")
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Linux receipt payload is invalid JSON") from error
    canonical = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    _require(raw == canonical, "Linux receipt payload is not canonical")
    document = _exact(
        value,
        frozenset(
            {
                "version",
                "service",
                "candidate",
                "directWheel",
                "linuxWheelhouse",
                "image",
                "runtime",
                "installation",
            }
        ),
        "Linux receipt",
    )
    _require(
        type(document["version"]) is int and document["version"] == 1,
        "Linux receipt version differs",
    )
    service = _exact(
        document["service"],
        frozenset({"repositoryId", "workflowId", "runId", "attempt"}),
        "Linux receipt service",
    )
    service_values = tuple(
        _positive(service[key], "Linux receipt service identity differs")
        for key in ("repositoryId", "workflowId", "runId", "attempt")
    )
    _require(service_values[3] == 1, "Linux receipt attempt differs")
    candidate = _exact(
        document["candidate"],
        frozenset({"commit", "tree", "sourceArchiveSha256", "workflowSha256"}),
        "Linux receipt candidate",
    )
    _require(
        type(candidate["commit"]) is str and _OID.fullmatch(candidate["commit"]) is not None,
        "Linux receipt candidate commit differs",
    )
    _require(
        type(candidate["tree"]) is str and _OID.fullmatch(candidate["tree"]) is not None,
        "Linux receipt candidate tree differs",
    )
    direct = _exact(
        document["directWheel"],
        frozenset({"basename", "sha256", "bytes"}),
        "Linux receipt direct wheel",
    )
    wheelhouse = _exact(
        document["linuxWheelhouse"],
        frozenset(
            {
                "manifestSha256",
                "sourceLockSha256",
                "requirementsSha256",
                "constraintsSha256",
                "wheels",
            }
        ),
        "Linux receipt wheelhouse",
    )
    wheels = wheelhouse["wheels"]
    _require(type(wheels) is list and 0 < len(wheels) <= 65536, "Linux receipt wheel list differs")
    parsed_wheels: list[tuple[str, str, int]] = []
    for item in wheels:
        row = _exact(item, frozenset({"basename", "sha256", "bytes"}), "Linux receipt wheel")
        _require(
            type(row["basename"]) is str and _BASENAME.fullmatch(row["basename"]) is not None,
            "Linux receipt wheel basename differs",
        )
        parsed_wheels.append(
            (
                row["basename"],
                _sha(row["sha256"], "Linux receipt wheel digest"),
                _positive(row["bytes"], "Linux receipt wheel bytes differ"),
            )
        )
    _require(
        type(direct["basename"]) is str and _BASENAME.fullmatch(direct["basename"]) is not None,
        "Linux receipt direct wheel basename differs",
    )
    _require(
        parsed_wheels == sorted(parsed_wheels)
        and len({row[0] for row in parsed_wheels}) == len(parsed_wheels),
        "Linux receipt wheels are not sorted unique",
    )
    image = _exact(
        document["image"],
        frozenset({"reference", "configSha256", "layerSha256s", "layerDiffSha256s"}),
        "Linux receipt image",
    )
    _require(
        type(image["reference"]) is str
        and re.fullmatch(r"docker\.io/library/python@sha256:[0-9a-f]{64}", image["reference"])
        is not None,
        "Linux receipt image reference differs",
    )
    layers, diff_ids = image["layerSha256s"], image["layerDiffSha256s"]
    _require(
        type(layers) is list and type(diff_ids) is list and 0 < len(layers) == len(diff_ids) <= 16,
        "Linux receipt image graph differs",
    )
    parsed_layers = tuple(_sha(item, "Linux receipt image layer differs") for item in layers)
    parsed_diff_ids = tuple(_sha(item, "Linux receipt image diff ID differs") for item in diff_ids)
    runtime = _exact(
        document["runtime"],
        frozenset(
            {
                "platform",
                "pythonVersion",
                "soabi",
                "extSuffix",
                "multiarch",
                "glibc",
                "interpreterSha256",
                "stdlibInventorySha256",
            }
        ),
        "Linux receipt runtime",
    )
    _require(
        runtime["platform"] == _PLATFORM
        and type(runtime["pythonVersion"]) is str
        and re.fullmatch(r"3\.11\.[0-9]{1,3}", runtime["pythonVersion"]) is not None,
        "Linux receipt runtime identity differs",
    )
    for key in ("soabi", "extSuffix", "multiarch", "glibc"):
        _require(
            type(runtime[key]) is str
            and 0 < len(runtime[key]) <= 128
            and all(0x21 <= ord(char) <= 0x7E for char in runtime[key]),
            "Linux receipt ABI identity differs",
        )
    interpreter_sha256 = _sha(runtime["interpreterSha256"], "Linux receipt interpreter differs")
    stdlib_inventory_sha256 = _sha(
        runtime["stdlibInventorySha256"], "Linux receipt standard library differs"
    )
    installation = _exact(
        document["installation"],
        frozenset(
            {
                "installedInventorySha256",
                "installedFileCount",
                "importOriginCount",
                "nullCapturePassed",
                "cleanupRemoved",
            }
        ),
        "Linux receipt installation",
    )
    installed_inventory_sha256 = _sha(
        installation["installedInventorySha256"], "Linux receipt installed inventory differs"
    )
    installed_file_count = _positive(
        installation["installedFileCount"], "Linux receipt installed count differs"
    )
    import_origin_count = _positive(
        installation["importOriginCount"], "Linux receipt import count differs"
    )
    _require(
        installation["nullCapturePassed"] is True and installation["cleanupRemoved"] is True,
        "Linux receipt execution or cleanup is incomplete",
    )
    return LinuxReceiptPayloadV1(
        repository_id=service_values[0],
        workflow_id=service_values[1],
        run_id=service_values[2],
        attempt=service_values[3],
        candidate_commit=candidate["commit"],
        candidate_tree=candidate["tree"],
        source_archive_sha256=_sha(
            candidate["sourceArchiveSha256"], "Linux receipt source archive differs"
        ),
        workflow_sha256=_sha(candidate["workflowSha256"], "Linux receipt workflow differs"),
        direct_wheel_sha256=_sha(direct["sha256"], "Linux receipt direct wheel differs"),
        direct_wheel_bytes=_positive(direct["bytes"], "Linux receipt direct wheel bytes differ"),
        direct_wheel_basename=direct["basename"],
        wheelhouse_manifest_sha256=_sha(
            wheelhouse["manifestSha256"], "Linux receipt manifest differs"
        ),
        source_lock_sha256=_sha(
            wheelhouse["sourceLockSha256"], "Linux receipt source lock differs"
        ),
        requirements_sha256=_sha(
            wheelhouse["requirementsSha256"], "Linux receipt requirements differ"
        ),
        constraints_sha256=_sha(
            wheelhouse["constraintsSha256"], "Linux receipt constraints differ"
        ),
        wheels=tuple(parsed_wheels),
        image_reference=image["reference"],
        image_config_sha256=_sha(image["configSha256"], "Linux receipt image config differs"),
        image_layers=parsed_layers,
        image_diff_ids=parsed_diff_ids,
        python_version=runtime["pythonVersion"],
        soabi=runtime["soabi"],
        ext_suffix=runtime["extSuffix"],
        multiarch=runtime["multiarch"],
        glibc=runtime["glibc"],
        interpreter_sha256=interpreter_sha256,
        stdlib_inventory_sha256=stdlib_inventory_sha256,
        installed_inventory_sha256=installed_inventory_sha256,
        installed_file_count=installed_file_count,
        import_origin_count=import_origin_count,
        null_capture_passed=installation["nullCapturePassed"],
        cleanup_removed=installation["cleanupRemoved"],
    )
