"""Negative controls for the M3 installed-boundary evidence gate."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qualify_voice_delete as m3  # noqa: E402


def _good() -> dict[str, object]:
    return copy.deepcopy(m3._EXPECTED)


def test_complete_pinned_witness_passes() -> None:
    assert m3._passed(_good(), m3.HERMES_BASELINE | {"baseline": True})


@pytest.mark.parametrize("section", tuple(m3._EXPECTED))
def test_missing_or_extra_section_fails(section: str) -> None:
    missing = _good()
    missing.pop(section)
    assert not m3._passed(missing, m3.HERMES_BASELINE | {"baseline": True})
    extra = _good()
    extra[section + "_extra"] = 0
    assert not m3._passed(extra, m3.HERMES_BASELINE | {"baseline": True})


@pytest.mark.parametrize(
    ("section", "field"),
    tuple(
        (section, field)
        for section, expected in m3._EXPECTED.items()
        if type(expected) is dict
        for field in expected
    ),
)
def test_each_weakened_witness_fails(section: str, field: str) -> None:
    observed = _good()
    item = observed[section]
    assert type(item) is dict
    item[field] = 1 - item[field] if item[field] in (0, 1) else 0
    assert not m3._passed(observed, m3.HERMES_BASELINE | {"baseline": True})
    item[field] = True
    assert not m3._passed(observed, m3.HERMES_BASELINE | {"baseline": True})
    item[field] = m3._EXPECTED[section][field]
    item["unexpected"] = 0
    assert not m3._passed(observed, m3.HERMES_BASELINE | {"baseline": True})
    item.pop("unexpected")
    item.pop(field)
    assert not m3._passed(observed, m3.HERMES_BASELINE | {"baseline": True})


def test_wrong_pin_and_unattributed_fail() -> None:
    assert not m3._passed(_good(), {"version": "0.21.0", "commit": "different", "baseline": True})
    observed = _good()
    observed["unattributed"] = 1
    assert not m3._passed(observed, m3.HERMES_BASELINE | {"baseline": True})
