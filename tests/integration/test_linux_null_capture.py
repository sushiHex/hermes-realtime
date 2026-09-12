"""Linux candidate-wheel proof that unsupported capture leaves no local residue."""

from __future__ import annotations

import sys

import pytest

from scripts.qualification_linux_worker import prove_linux_null_capture

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="linux-null-capture gate")


def test_linux_candidate_has_no_capture_artifacts_threads_or_local_cli_surface() -> None:
    assert prove_linux_null_capture() > 0
