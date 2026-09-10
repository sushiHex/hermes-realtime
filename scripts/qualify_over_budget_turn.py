"""Qualify capture admission overflow from a source-bound pure wheel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from scripts.candidate_e2e_fast_track import _measure_git_pin
from scripts.candidate_source_archive_oracle import capture_candidate_source_archive
from scripts.candidate_wheel import verify_candidate_wheel_v1
from scripts.over_budget_turn import validate_over_budget_turn_v1
from scripts.qualify_evidence_slice_zero import (
    OVER_BUDGET_TURN_REGISTRATION_V1,
    canonical_json_bytes,
)
from scripts.task13_artifact_orchestrator import verify_clean_tracked_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", default="HEAD^")
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--livekit-executable", type=Path, required=True)
    parser.add_argument("--livekit-sha256", required=True)
    parser.add_argument(
        "--git-executable",
        type=Path,
        default=Path("C:/Program Files/Git/mingw64/bin/git.exe"),
    )
    arguments = parser.parse_args()
    candidate = arguments.candidate.resolve(strict=True)
    identity = verify_clean_tracked_candidate(candidate, arguments.baseline)
    archive = capture_candidate_source_archive(
        candidate,
        identity,
        _measure_git_pin(arguments.git_executable),
    )
    wheel = verify_candidate_wheel_v1(
        archive, identity, arguments.candidate_wheel, arguments.wheel_sha256
    )
    receipt = OVER_BUDGET_TURN_REGISTRATION_V1.produce(
        archive,
        identity,
        wheel,
        livekit_executable=arguments.livekit_executable,
        livekit_sha256=arguments.livekit_sha256,
    )
    evidence = validate_over_budget_turn_v1(receipt)
    summary = {
        "schemaVersion": "over-budget-turn-summary-v1",
        "proofClass": "packaged_process",
        **asdict(evidence),
    }
    print(canonical_json_bytes(summary).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
