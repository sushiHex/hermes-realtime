"""Freeze development inputs separately from the unchanged routing holdout."""

from hashlib import sha256
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"


def test_development_fixture_matches_the_approved_labels() -> None:
    assert sha256((_FIXTURES / "natural_work_development_cases.json").read_text(
        encoding="utf-8").encode()).hexdigest() == (
        "1fb100ca8cd749fceda243242d33b81be3f4391ceed6b673c23238ae9f2c6c88"
    )


def test_development_wave_preserves_the_holdout_content() -> None:
    assert sha256((_FIXTURES / "natural_work_intent_cases.json").read_text(
        encoding="utf-8").encode()).hexdigest() == (
        "21d96578dfa8b3f833d0c940edf7cfae64ab4efd355290ab7d80bcf0d8e97ee9"
    )
