import subprocess
import sys
from datetime import datetime, timedelta, timezone

from hermes_realtime.conversation import result_projection
from hermes_realtime.providers import _speech_text


def test_task_module_starts_without_provider_import_cycle():
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import hermes_realtime.conversation.tasks"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_complete_result_localizes_voice_without_rewriting_source_detail(monkeypatch):
    class LocalClock(datetime):
        def astimezone(self, tz=None):
            assert tz is None
            return super().astimezone(timezone(timedelta(hours=-7), "Pacific Daylight Time"))

    monkeypatch.setattr(_speech_text, "datetime", LocalClock)
    voice = "NWS reported 70 degrees at 2026-10-30T21:53:00Z."
    detail = "Original observation: 2026-10-30T21:53:00Z. Source: NWS."
    projected, retained = result_projection._completion_projections(
        f"<voice>{voice}</voice><detail>{detail}</detail>"
    )
    assert projected == (
        "NWS reported 70 degrees at 2:53 PM Pacific Daylight Time on October 30, 2026."
    )
    assert retained == detail


def test_expanded_voice_does_not_weaken_existing_projection_bound(monkeypatch):
    voice = "A valid short summary."
    monkeypatch.setattr(_speech_text, "project_speech_text", lambda _: "x" * 241)
    projected, retained = result_projection._completion_projections(
        f"<voice>{voice}</voice><detail>Original source detail.</detail>"
    )
    assert projected == voice
    assert retained == "Original source detail."
