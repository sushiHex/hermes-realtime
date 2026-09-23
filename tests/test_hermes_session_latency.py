from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hermes_session_latency.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))  # The script imports its sibling gate support.
_SCRIPT_SPEC = importlib.util.spec_from_file_location("hermes_session_latency", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

_STARTED = (5.0, "run.started", "{}")
_DELTA = (40.0, "assistant.delta", json.dumps({"delta": "Hello."}))
_COMPLETED = (90.0, "run.completed", json.dumps({"usage": {"input_tokens": 993}}))
_ERROR = (60.0, "error", json.dumps({"message": "provider failed"}))
_DONE = (91.0, "done", "{}")


def test_a_complete_stream_is_admitted_with_its_deltas_and_usage() -> None:
    sample = _SCRIPT._turn_sample([_STARTED, _DELTA, _COMPLETED, _DONE])

    assert sample == {"deltas": [(40.0, "Hello.")], "completed_ms": 90.0, "input_tokens": 993}


@pytest.mark.parametrize(
    ("frames", "reason"),
    [
        ([_STARTED, _COMPLETED, _DONE], "without speech and completion"),
        ([_STARTED, _DELTA, _DONE], "without speech and completion"),
        ([_STARTED, _DELTA, _ERROR, _COMPLETED, _DONE], "error event"),
    ],
    ids=["no-speech", "no-completion", "error-event"],
)
def test_an_incomplete_or_failed_stream_is_refused(
    frames: list[tuple[float, str, str]], reason: str
) -> None:
    with pytest.raises(RuntimeError, match=reason):
        _SCRIPT._turn_sample(frames)


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ([(10.0, "Hel"), (20.0, "lo there."), (30.0, " More")], 20.0),
        ([(10.0, "No sentence end"), (20.0, " yet")], 99.0),
        ([(10.0, "word " * 100), (20.0, "word " * 110)], 20.0),
    ],
    ids=["sentence-end", "spoken-at-completion", "segment-cap"],
)
def test_first_speech_is_the_voice_paths_first_speakable_segment(
    deltas: list[tuple[float, str]], expected: float
) -> None:
    assert _SCRIPT._first_speech_ms(deltas, 99.0) == expected


def test_the_worker_inherits_no_credentials_but_the_promised_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setenv("CODEX_HOME", "real-codex-login")

    environment = _SCRIPT._worker_environment(str(tmp_path), "access-token")

    assert "OPENAI_API_KEY" not in environment
    assert environment["HERMES_LATENCY_ACCESS_TOKEN"] == "access-token"
    assert environment["HERMES_HOME"] == str(tmp_path)
    assert environment["CODEX_HOME"] == str(tmp_path / "no-codex-login")
    assert environment["PATH"]


def test_a_stopped_measurement_records_one_content_free_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = [{"first_speech_ms": 1.0}]

    with pytest.raises(TimeoutError), _SCRIPT._refusal_evidence("hermes", samples, 21):
        raise TimeoutError("provider text that must not be recorded")

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("[hermes-latency] ")
    assert json.loads(lines[0].removeprefix("[hermes-latency] ")) == {
        "path": "hermes",
        "completed": 1,
        "planned": 21,
        "failure": "TimeoutError",
    }


def test_a_finished_measurement_records_no_marker(capsys: pytest.CaptureFixture[str]) -> None:
    with _SCRIPT._refusal_evidence("hermes", [], 21):
        pass

    assert capsys.readouterr().err == ""
