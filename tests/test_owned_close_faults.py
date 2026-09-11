"""Owned-close acceptance preserves attempts and derives conversation equality."""

from __future__ import annotations

from copy import deepcopy

import pytest


def _observations():
    from scripts.deterministic_equivalence import _CONVERSATION_KINDS, _expected_close

    arms = (
        "ordinary_disabled",
        "ordinary_consented",
        "retry_disabled",
        "retry_consented",
        "cancelled_disabled",
        "cancelled_consented",
        "perturbed",
    )
    cases = []
    for name in arms:
        consented = name.endswith("consented")
        outcome = "failed" if name.startswith("retry") else (
            "cancelled" if name.startswith("cancelled") else "returned"
        )
        records = [
            {"kind": kind, "value": f"{index + 1:064x}"}
            for index, kind in enumerate(_CONVERSATION_KINDS[:-1])
        ] + [{"kind": "host_return", "value": outcome}]
        close = _expected_close("consented" if consented else "disabled")
        if name.startswith("retry"):
            close.insert(-1, {"stage": "launcher", "result": "failed"})
            events = [
                "provider_entered", "provider_failed", "caller_failed",
                "provider_entered", "provider_returned", "retry_returned",
            ]
        elif name.startswith("cancelled"):
            events = [
                "provider_entered", "caller_cancelled", "provider_released",
                "provider_returned", "join_returned",
            ]
        else:
            events = ["provider_entered", "provider_returned", "caller_returned"]
        if name == "perturbed":
            records[0]["value"] = "f" * 64
        cases.append({
            "arm": name,
            "complete": True,
            "records": records,
            "terminals": [
                {"disposition": "completed", "reason": "authoritative_close_completed",
                 "contextCommitted": True},
            ] * 2 if consented else [],
            "close": close,
            "owner_events": events,
        })
    return {"arm": "owned_close_faults", "cases": cases}


def test_registration_is_canonical_and_supplied_rows_cannot_mint_receipts() -> None:
    from scripts.owned_close_faults import (
        ObservedOwnedCloseFaultsV1,
        validate_owned_close_faults_v1,
    )
    from scripts.qualify_evidence_slice_zero import (
        OWNED_CLOSE_FAULTS_REGISTRATION_V1,
        SCENARIO_REGISTRY_V1,
        OwnedCloseFaultsRegistrationV1,
    )

    assert SCENARIO_REGISTRY_V1[19] is OWNED_CLOSE_FAULTS_REGISTRATION_V1
    with pytest.raises(ValueError, match="canonical"):
        OwnedCloseFaultsRegistrationV1().produce(
            None, None, None, livekit_executable=None, livekit_sha256="0" * 64
        )
    with pytest.raises(TypeError):
        validate_owned_close_faults_v1(_observations())
    with pytest.raises(ValueError, match="unregistered"):
        validate_owned_close_faults_v1(object.__new__(ObservedOwnedCloseFaultsV1))


@pytest.mark.parametrize(
    "mutation",
    [None, "missing", "duplicate", "order", "incomplete", "context", "kind", "digest",
     "normalized_failure", "normalized_cancel", "missing_terminal", "terminal_bool",
     "missing_failed_attempt", "duplicate_runtime", "provider_cancel", "retry_omitted",
     "changed_pair", "changed_fault_baseline", "negative", "extra"],
)
def test_acceptance_requires_exact_attempts_and_real_comparisons(mutation) -> None:
    from scripts.owned_close_faults import _validate_observations

    row = deepcopy(_observations())
    cases = row["cases"]
    if mutation == "missing":
        cases.pop()
    elif mutation == "duplicate":
        cases[1] = deepcopy(cases[0])
    elif mutation == "order":
        cases[0], cases[1] = cases[1], cases[0]
    elif mutation == "incomplete":
        cases[1]["complete"] = False
    elif mutation == "context":
        cases[0]["records"].pop(0)
    elif mutation == "kind":
        cases[0]["records"][0]["kind"] = "generated_text"
    elif mutation == "digest":
        cases[0]["records"][0]["value"] = "short"
    elif mutation in {"normalized_failure", "normalized_cancel"}:
        cases[2 if mutation == "normalized_failure" else 4]["records"][-1]["value"] = "returned"
    elif mutation == "missing_terminal":
        cases[1]["terminals"].pop()
    elif mutation == "terminal_bool":
        cases[1]["terminals"][0]["contextCommitted"] = 1
    elif mutation == "missing_failed_attempt":
        cases[2]["close"].pop(-2)
    elif mutation == "duplicate_runtime":
        cases[3]["close"].insert(0, deepcopy(cases[3]["close"][0]))
    elif mutation == "provider_cancel":
        cases[4]["owner_events"].insert(1, "provider_cancelled")
    elif mutation == "retry_omitted":
        cases[2]["owner_events"].pop()
    elif mutation == "changed_pair":
        cases[3]["records"][0]["value"] = "e" * 64
    elif mutation == "changed_fault_baseline":
        cases[2]["records"][0]["value"] = cases[3]["records"][0]["value"] = "e" * 64
    elif mutation == "negative":
        cases[-1]["records"] = deepcopy(cases[0]["records"])
    elif mutation == "extra":
        cases[1]["injected"] = True
    if mutation is None:
        _validate_observations(row)
    else:
        with pytest.raises(ValueError):
            _validate_observations(row)
