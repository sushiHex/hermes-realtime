"""Small native-web presentation attestation contract; no tool authority.

``no_web``, ``mixed_ungrounded`` and ``ungrounded`` preserve ordinary native output.
None of them attests factual grounding; only ``presented`` identifies the bounded
passage-backed presentation path. No disposition proves semantic correctness.
``ungrounded`` is the terminal decision for an answer the observer never saw (for
example Hermes's turn-budget summary, which bypasses LLM middleware) in a no-web or
mixed run; a web-only run's unobserved answer is refused. It is bound to the exact
output like every other disposition. ``refused`` withholds the output and
names why with one category from ``REFUSED_REASONS``.
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
ATTESTATION_VERSION = 2
REPORT_MAX_BYTES = 2048
INPUT_MAX_BYTES = 16384
CATALOG_MAX = 256
PASSAGE_MAX = 2048
# Why an answer was delivered without a grounding claim, or withheld. Closed sets.
UNGROUNDED_REASONS = frozenset({"unobserved"})
REFUSED_REASONS = frozenset({
    "binding", "capture", "closed", "deadline", "final_response", "input_bound",
    "insufficient", "internal", "presentation_failed", "route", "source_changed",
    "unobserved",
})
_UNCLAIMED = frozenset({"no_web", "mixed_ungrounded"})
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"version", "contract", "generation", "run_id", "disposition", "reason",
           "output_sha256", "input_sha256", "input_bytes", "catalog_sha256", "catalog_count",
           "passage_count", "truncated_count", "missing_count"}


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


def attestation_outcome(
    raw: object, generation: str, run_id: str, output: object
) -> tuple[str, str | None] | None:
    """Return ``(disposition, reason)`` for an attestation bound to exact native output.

    ``None`` means the attestation is missing, malformed or bound to something else.
    A valid ``refused`` outcome is returned so its reason can be reported; it never
    permits publication.
    """
    if type(raw) is not str or type(output) is not str or not valid_generation(generation):
        return None
    try:
        if len(raw.encode("utf-8")) > REPORT_MAX_BYTES:
            return None
        value = json.loads(raw, object_pairs_hook=_unique)
        output_sha = hashlib.sha256(output.encode("utf-8")).hexdigest()
    except (ValueError, UnicodeError, RecursionError):
        return None
    if (type(value) is not dict or set(value) != _FIELDS
            or type(value["version"]) is not int or value["version"] != ATTESTATION_VERSION
            or value["contract"] != CONTRACT or value["generation"] != generation
            or value["run_id"] != run_id or value["output_sha256"] != output_sha
            or type(value["disposition"]) is not str):
        return None
    for name, maximum in (("input_bytes", INPUT_MAX_BYTES), ("catalog_count", CATALOG_MAX),
                          ("passage_count", PASSAGE_MAX), ("missing_count", CATALOG_MAX),
                          ("truncated_count", PASSAGE_MAX)):
        if type(value[name]) is not int or not 0 <= value[name] <= maximum:
            return None
    if (value["truncated_count"] > value["passage_count"]
            or value["missing_count"] > value["catalog_count"]):
        return None
    disposition, reason = value["disposition"], value["reason"]
    if disposition == "presented":
        if (reason is None and value["input_bytes"] > 0 and value["catalog_count"] > 0
                and value["passage_count"] > 0
                and value["missing_count"] < value["catalog_count"] and all(
                    type(value[name]) is str and _SHA256.fullmatch(value[name]) is not None
                    for name in ("input_sha256", "catalog_sha256"))):
            return "presented", None
        return None
    if disposition in _UNCLAIMED:
        if reason is not None:
            return None
    elif disposition == "ungrounded":
        if type(reason) is not str or reason not in UNGROUNDED_REASONS:
            return None
    elif disposition == "refused":
        if type(reason) is not str or reason not in REFUSED_REASONS:
            return None
    else:
        return None
    if (value["input_sha256"] is None and value["catalog_sha256"] is None
            and all(value[name] == 0 for name in
                    ("input_bytes", "catalog_count", "passage_count", "missing_count",
                     "truncated_count"))):
        return disposition, reason
    return None


def attestation_valid(raw: object, generation: str, run_id: str, output: object) -> bool:
    """Whether the attestation permits publishing exactly ``output``."""
    outcome = attestation_outcome(raw, generation, run_id, output)
    return outcome is not None and outcome[0] != "refused"
