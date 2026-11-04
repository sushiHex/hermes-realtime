"""Call-local extraction identity gaps preserve immutable prior evidence."""

from types import MappingProxyType

import pytest

from hermes_realtime.integration.evidence_presentation import captured_passages

A = "https://example.invalid/a"
B = "https://example.invalid/b"
FOREIGN = "https://example.invalid/foreign"
GAP = ("extract-unavailable", "")


def snapshot():
    return MappingProxyType({A: (("search", "A context"),), B: (("search", "B context"),)})


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_extraction_marks_only_missing_call_members(reverse):
    current = snapshot()
    items = [{"url": A, "content": "Bound A text"},
             {"url": FOREIGN, "content": "UNBOUND_TEXT"}]
    updated = captured_passages(current, list(reversed(items)) if reverse else items,
                                "extract", {A, B}, {A, B})
    assert updated[A] == current[A] + (("extract", "Bound A text"),)
    assert updated[B] == current[B] + (GAP,)
    assert dict(current) == dict(snapshot())
    assert FOREIGN not in updated and "UNBOUND_TEXT" not in repr(dict(updated))
    assert isinstance(updated, MappingProxyType) and updated is not current


def test_successful_call_does_not_erase_a_previous_call_gap():
    current = MappingProxyType({A: (("search", "A context"), GAP), B: (("search", "B context"),)})
    updated = captured_passages(current, [{"url": FOREIGN, "content": "UNBOUND_TEXT"},
                                         {"url": A, "content": "Bound A text"}],
                                "extract", {A, B}, {A, B})
    assert updated[A] == current[A] + (("extract", "Bound A text"),)
    assert updated[A].count(GAP) == 1 and updated[B] == current[B] + (GAP,)


@pytest.mark.parametrize("membership", ["admitted", "catalog"])
def test_gap_and_bound_text_respect_request_and_catalog_membership(membership):
    current = snapshot()
    admitted = {B} if membership == "admitted" else {A, B}
    catalog = {B} if membership == "catalog" else {A, B}
    updated = captured_passages(current, [{"url": A, "content": "NOT_ADMITTED_TEXT"},
                                         {"url": FOREIGN, "content": "UNBOUND_TEXT"}],
                                "extract", admitted, catalog)
    assert updated[A] == current[A]
    assert updated[B] == current[B] + (GAP,)
    assert "NOT_ADMITTED_TEXT" not in repr(dict(updated))


@pytest.mark.parametrize("failure", [{"error": "Synthetic failure"}, {"blocked_by_policy": True}])
def test_failed_bound_item_does_not_suppress_the_call_gap(failure):
    current = snapshot()
    updated = captured_passages(current, [{"url": A, "content": "REJECTED_TEXT", **failure},
                                         {"url": FOREIGN, "content": "UNBOUND_TEXT"}],
                                "extract", {A, B}, {A, B})
    assert updated[A] == current[A] + (GAP,)
    assert updated[B] == current[B] + (GAP,)
    assert "REJECTED_TEXT" not in repr(dict(updated))


def test_empty_validated_extraction_still_establishes_returned_identity():
    current = snapshot()
    updated = captured_passages(current, [{"url": A, "content": ""},
                                         {"url": FOREIGN, "content": "UNBOUND_TEXT"}],
                                "extract", {A, B}, {A, B})
    assert updated[A] == current[A] + (("extract", ""),)
    assert updated[B] == current[B] + (GAP,)


def test_mixed_extraction_failure_keeps_original_snapshot_unchanged():
    current = snapshot()
    original_a, original_b = current[A], current[B]
    with pytest.raises(ValueError, match="evidence-shape"):
        captured_passages(current, [{"url": A, "content": "Bound A text"},
                                    {"url": FOREIGN, "content": "UNBOUND_TEXT"},
                                    {"url": B, "content": 1}], "extract", {A, B}, {A, B})
    assert current[A] is original_a and current[B] is original_b
    assert dict(current) == dict(snapshot())
