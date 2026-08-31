"""Package boundary checks for the test-only raw qualification owner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).parents[1]


def test_wheel_excludes_test_support_and_source_has_no_raw_qualification_trace_owner(
    tmp_path: Path,
) -> None:
    """The only raw trace owner is test support, never an installed package member."""

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    wheels = tuple(tmp_path.glob("hermes_realtime-*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel:
        members = wheel.namelist()
        assert not any(member.startswith("tests/support/") for member in members)
        assert "hermes_realtime/_qualification.py" in members
        qualification_source = wheel.read("hermes_realtime/_qualification.py").decode("utf-8")
        metadata_members = [
            member
            for member in members
            if member.endswith(".dist-info/METADATA") and member.count("/") == 1
        ]
        assert len(metadata_members) == 1, metadata_members
        metadata = wheel.read(metadata_members[0]).decode("utf-8")

    # The built distribution is what a package index and `pip show` consume, so
    # the qualifier is asserted on the artifact and not only on pyproject.toml.
    # The literal is duplicated from tests/test_repository_text_policy.py on
    # purpose: sharing it would let one edit move both assertions together.
    summary_lines = [
        line[len("Summary: ") :]
        for line in metadata.splitlines()
        if line.startswith("Summary: ")
    ]
    assert summary_lines == [
        "Unofficial realtime LiveKit conversation runtime for Hermes Agent"
    ]

    assert "_QualificationTraceState" not in qualification_source
    assert "_QualificationTraceV1" not in qualification_source
    assert "_QualificationCompositionBundleV1" not in qualification_source
    assert "_QualificationHostRegistrationV1" not in qualification_source
    assert "committed_conversation_context_snapshot_bytes" not in qualification_source
    assert "compose_host" not in qualification_source
    assert "tests.support" in (ROOT / "src" / "hermes_realtime" / "_qualification.py").read_text(
        encoding="utf-8"
    )
