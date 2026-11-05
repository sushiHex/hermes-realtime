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
