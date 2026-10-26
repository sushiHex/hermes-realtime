from __future__ import annotations

import pytest

from hermes_realtime.providers._work_claims import has_background_work_claim


@pytest.mark.parametrize("text", [
    "Could you send the error log? Share it here, and I'll inspect it once it arrives.",
    "I will research the timeout.",
    "Let me investigate the timeout.",
])
def test_future_offers_are_not_reports_of_started_work(text: str) -> None:
    assert not has_background_work_claim(text)


@pytest.mark.parametrize("text", [
    "I've started the investigation.",
    "I’ve launched the investigation.",
    "I started a background task.",
    "I launched the analysis.",
])
def test_recognized_actual_start_claims_remain_guarded(text: str) -> None:
    assert has_background_work_claim(text)
