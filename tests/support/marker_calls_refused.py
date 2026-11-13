"""Call sites a strict type check must refuse, one per line that ends in ``# refused``.

This file is deliberately ill-typed. tests/markers/test_marker_core.py checks that the type
checker reports an error on every marked line and on no other line.
"""

from __future__ import annotations

from tests.support.marker_lines import (
    ForegroundFailure,
    LocalModelPrepare,
    OllamaProfile,
    PreviewStatus,
    StagingRefusal,
)


def free_strings(text: str, error: Exception, model: object) -> None:
    LocalModelPrepare(category=text)  # refused
    LocalModelPrepare(category=str(error))  # refused
    LocalModelPrepare(category=type(error).__name__)  # refused
    LocalModelPrepare(category="not_a_member")  # refused
    OllamaProfile(category="provider_refused", count="1")  # refused
    OllamaProfile(category="provider_refused", count=1, model="name")  # refused
    OllamaProfile(category="provider_refused")  # refused
    ForegroundFailure(stage="producer", failure_kind=text, text_phase="after_text")  # refused
    seen = ForegroundFailure(stage="producer", failure_kind="other", text_phase="after_text")
    seen.scope()  # refused
    LocalModelPrepare(category="prepared").emit()  # refused
    with LocalModelPrepare(category="choice_refused").scope() as line:
        line.category = text  # refused
        line.detail = "free text"  # refused
    inferred = "prepared" if text else "deadline"
    LocalModelPrepare(category=inferred)  # refused
    PreviewStatus(category="ready", failure_kind=type(error).__name__)  # refused


def refusals(text: str) -> None:
    StagingRefusal(text)  # refused
    StagingRefusal("tcp_port_invalid")  # refused
    StagingRefusal("run_directory", "and a message")  # refused
    StagingRefusal()  # refused
