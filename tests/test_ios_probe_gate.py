from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gate(tmp_path: Path):
    probe = tmp_path / "probe"
    shutil.copytree(ROOT / "ios-probe", probe)
    module = runpy.run_path(str(probe / "scripts" / "validate_probe.py"))
    return probe, module["main"]


def test_committed_probe_passes_structural_gate(tmp_path: Path) -> None:
    _, gate = _gate(tmp_path)
    assert gate() == 0


@pytest.mark.parametrize(
    "change", ["extra_source", "comment_version", "comment_target", "comment_api"]
)
def test_probe_gate_rejects_uncompiled_evidence(tmp_path: Path, change: str) -> None:
    probe, gate = _gate(tmp_path)
    if change == "extra_source":
        (probe / "Sources" / "Extra.swift").write_text("AVAudioEngine()", encoding="utf-8")
    elif change == "comment_api":
        path = probe / "Sources" / "ProbeModel.swift"
        text = path.read_text(encoding="utf-8")
        text = text.replace("AudioManager.shared.platformVoiceProcessingState", "unprovenState")
        path.write_text(
            text + "\n// AudioManager.shared.platformVoiceProcessingState\n", encoding="utf-8"
        )
    else:
        path = probe / "project.yml"
        text = path.read_text(encoding="utf-8")
        old, new = (
            ('exactVersion: "2.15.3"', 'exactVersion: "0.0.0"')
            if change == "comment_version"
            else ("IPHONEOS_DEPLOYMENT_TARGET: 17.0", "IPHONEOS_DEPLOYMENT_TARGET: 16.0")
        )
        path.write_text(text.replace(old, new) + "\n# " + old + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        gate()


@pytest.mark.parametrize("comment", ['/* " */', '/* // " */', '/* outer /* " */ inner */'])
def test_probe_gate_comments_cannot_hide_live_forbidden_api(tmp_path: Path, comment: str) -> None:
    probe, gate = _gate(tmp_path)
    (probe / "Sources" / "Extra.swift").write_text(
        comment + '\nAVAudioEngine()\n/* " */', encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="forbidden parallel/session-owner API"):
        gate()


def test_probe_comment_removal_preserves_token_boundaries_and_urls() -> None:
    strip = runpy.run_path(str(ROOT / "ios-probe/scripts/validate_probe.py"))[
        "_swift_without_comments"
    ]
    assert "AVAudioEngine" not in strip("AVAudio/* comment */Engine()")
    assert "https://example.invalid" in strip('let url = "https://example.invalid" // note')
    assert "live" in strip('/* outer /* " */ inner */ live')
