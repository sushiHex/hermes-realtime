"""Small native-web presentation attestation contract; no tool authority.

``no_web`` and ``mixed_ungrounded`` preserve ordinary native output. Neither
disposition attests factual grounding; only ``presented`` identifies the bounded
passage-backed presentation path. No disposition proves semantic correctness.
"""

from __future__ import annotations

import hashlib
import json
import re

CONTRACT = "hermes.web-evidence/v1"
READY_PATH = "/hermes-realtime/v1/web-evidence/ready"
CONTRACT_HEADER = "X-Web-Evidence-Contract"
GENERATION_HEADER = "X-Web-Evidence-Generation"
RUN_HEADER = "X-Web-Evidence-Run"
ATTESTATION_HEADER = "X-Web-Evidence-Attestation"
REPORT_MAX_BYTES = 2048
INPUT_MAX_BYTES = 16384
CATALOG_MAX = 256
PASSAGE_MAX = 2048
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"version", "contract", "generation", "run_id", "disposition", "output_sha256",
           "input_sha256", "input_bytes", "catalog_sha256", "catalog_count", "passage_count",
           "truncated_count", "missing_count"}


def valid_generation(value: object) -> bool:
    return type(value) is str and _GENERATION.fullmatch(value) is not None


def valid_selection(contract: object, generation: object) -> bool:
    return (contract is None and generation is None) or (
        type(contract) is str and contract == CONTRACT and valid_generation(generation)
    )


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate attestation field")
    return value


def attestation_valid(raw: object, generation: str, run_id: str, output: object) -> bool:
    """Validate metadata against exact native output before any text projection."""
    if type(raw) is not str or type(output) is not str or not valid_generation(generation):
        return False
    try:
        if len(raw.encode("utf-8")) > REPORT_MAX_BYTES:
            return False
        value = json.loads(raw, object_pairs_hook=_unique)
        output_sha = hashlib.sha256(output.encode("utf-8")).hexdigest()
    except (ValueError, UnicodeError, RecursionError):
        return False
    if (type(value) is not dict or set(value) != _FIELDS
            or type(value["version"]) is not int or value["version"] != 1
            or value["contract"] != CONTRACT or value["generation"] != generation
            or value["run_id"] != run_id or value["output_sha256"] != output_sha
            or type(value["disposition"]) is not str):
        return False
    for name, maximum in (("input_bytes", INPUT_MAX_BYTES), ("catalog_count", CATALOG_MAX),
                          ("passage_count", PASSAGE_MAX), ("missing_count", CATALOG_MAX),
                          ("truncated_count", PASSAGE_MAX)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            return False
    if (value["truncated_count"] > value["passage_count"]
            or value["missing_count"] > value["catalog_count"]):
        return False
    if value["disposition"] == "presented":
        return (value["input_bytes"] > 0 and value["catalog_count"] > 0
                and value["passage_count"] > 0
                and value["missing_count"] < value["catalog_count"] and all(
            type(value[name]) is str and _SHA256.fullmatch(value[name]) is not None
            for name in ("input_sha256", "catalog_sha256")
        ))
    if value["disposition"] not in {"no_web", "mixed_ungrounded"}:
        return False
    return (value["input_sha256"] is None and value["catalog_sha256"] is None
            and all(value[name] == 0 for name in
                    ("input_bytes", "catalog_count", "passage_count", "missing_count",
                     "truncated_count")))
