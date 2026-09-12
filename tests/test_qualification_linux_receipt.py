import hashlib
import json

import pytest


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload() -> bytes:
    document = {
        "version": 1,
        "service": {"repositoryId": 1351392603, "workflowId": 345893071, "runId": 42, "attempt": 1},
        "candidate": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "sourceArchiveSha256": "c" * 64,
            "workflowSha256": "9" * 64,
        },
        "directWheel": {
            "basename": "hermes_realtime-0.0.3-py3-none-any.whl",
            "sha256": "d" * 64,
            "bytes": 123,
        },
        "linuxWheelhouse": {
            "manifestSha256": "e" * 64,
            "sourceLockSha256": "a" * 64,
            "requirementsSha256": "f" * 64,
            "constraintsSha256": "0" * 64,
            "wheels": [
                {"basename": "aiohttp-3.11.0-py3-none-any.whl", "sha256": "1" * 64, "bytes": 456}
            ],
        },
        "image": {
            "reference": "docker.io/library/python@sha256:" + "2" * 64,
            "configSha256": "3" * 64,
            "layerSha256s": ["4" * 64],
            "layerDiffSha256s": ["5" * 64],
        },
        "runtime": {
            "platform": "linux_x86_64",
            "pythonVersion": "3.11.16",
            "soabi": "cpython-311-x86_64-linux-gnu",
            "extSuffix": ".cpython-311-x86_64-linux-gnu.so",
            "multiarch": "x86_64-linux-gnu",
            "glibc": "2.36",
            "interpreterSha256": "6" * 64,
            "stdlibInventorySha256": "7" * 64,
        },
        "installation": {
            "installedInventorySha256": "8" * 64,
            "installedFileCount": 9,
            "importOriginCount": 10,
            "nullCapturePassed": True,
            "cleanupRemoved": True,
        },
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def test_payload_is_canonical_closed_and_metadata_contains_no_paths():
    from scripts.qualification_linux_receipt import (
        parse_linux_receipt_payload,
    )

    parsed = parse_linux_receipt_payload(_payload())
    metadata = parsed

    assert metadata.candidate_commit == "a" * 40
    assert metadata.direct_wheel_sha256 == "d" * 64
    assert metadata.direct_wheel_basename == "hermes_realtime-0.0.3-py3-none-any.whl"
    assert metadata.source_lock_sha256 == "a" * 64
    assert metadata.interpreter_sha256 == "6" * 64
    assert metadata.installed_inventory_sha256 == "8" * 64
    assert metadata.cleanup_removed is True


@pytest.mark.parametrize(
    "mutation",
    ["extra", "newline", "cleanup", "wheel_order", "path", "version_bool", "version_float"],
)
def test_payload_refuses_ambiguous_or_incomplete_claims_before_any_receipt_is_minted(mutation):
    from scripts.qualification_linux_receipt import parse_linux_receipt_payload

    raw = _payload()
    if mutation == "newline":
        raw = raw[:-1]
    else:
        value = json.loads(raw)
        if mutation == "extra":
            value["extra"] = True
        elif mutation == "cleanup":
            value["installation"]["cleanupRemoved"] = False
        elif mutation == "wheel_order":
            value["linuxWheelhouse"]["wheels"].append(
                {"basename": "aaa-1-py3-none-any.whl", "sha256": "9" * 64, "bytes": 1}
            )
        elif mutation == "version_bool":
            value["version"] = True
        elif mutation == "version_float":
            value["version"] = 1.0
        else:
            value["runtime"]["path"] = "/private/path"
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError):
        parse_linux_receipt_payload(raw)


def test_payload_is_immutable_data_not_a_success_authority():
    from dataclasses import FrozenInstanceError

    from scripts.qualification_linux_receipt import parse_linux_receipt_payload

    with pytest.raises(FrozenInstanceError):
        parse_linux_receipt_payload(_payload()).run_id = 9
