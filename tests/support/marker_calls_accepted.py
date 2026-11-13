"""Call sites a strict type check must accept. Checked by tests/markers/test_marker_core.py."""

from __future__ import annotations

from tests.support.marker_lines import (
    ForegroundFailure,
    LocalModelPrepare,
    OllamaProfile,
    PrepareCategory,
    PreviewStatus,
    ProducerFailure,
    ReadinessRefusal,
    StagingRefusal,
)


def prepare(ready: bool) -> None:
    category: PrepareCategory = "prepared" if ready else "readiness_refused"
    with LocalModelPrepare(category="choice_refused").scope() as line:
        line.category = category


def refuse_profile() -> None:
    raise OllamaProfile(category="provider_refused", count=1).raising(ValueError("refused"))


def report(kind: ProducerFailure, after_text: bool) -> None:
    ForegroundFailure(
        stage="producer",
        failure_kind=kind,
        text_phase="after_text" if after_text else "before_text",
    ).emit()


def status(attempts: int) -> None:
    with PreviewStatus(category="ready", attempts=attempts).scope() as line:
        if attempts > 3:
            raise StagingRefusal("run_directory")
        line.report()
    refusal = ReadinessRefusal("tcp_port_invalid")
    line.category = refusal.category
