"""Evidence lines declared for the mechanism's own tests. No product code uses them.

Each declaration copies the name, the members and the byte layout of a marker that exists
today, so the tests can show that a declaration reproduces the bytes its marker writes.
They live in their own catalogue: declaring the same name in product code later does not
collide with them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from hermes_realtime.markers import (
    SILENT,
    Count,
    ExceptionKind,
    Layout,
    Observation,
    Outcome,
    Refusal,
    isolated_catalogue,
)

PrepareCategory = Literal[
    "choice_refused", "prepared", "readiness_refused", "deadline", "cancelled",
]
ProfileCategory = Literal[
    "provider_refused", "override_refused", "gpu_layers_refused", "thinking_refused",
]
MemoryRefusal = Literal["capacity", "unclassified"]
ProducerFailure = Literal["value", "type", "runtime", "timeout", "group", "refused", "other"]
StartupRefusal = Literal[
    "schema_rejected", "authentication_rejected", "assets_rejected",
    "operation_catalog_rejected", "startup_deadline_expired", "policy_rejected",
    "cleanup_unknown",
]
StagingCategory = Literal["run_directory", "output_receipt_refused"]
ReadinessCategory = Literal["host_ready_receipt", "tcp_port_invalid"]
StatusCategory = StagingCategory | ReadinessCategory | Literal["failed", "interrupted", "ready"]


class StagingRefusal(Refusal[StagingCategory]):
    """A refusal the status line admits."""


class ReadinessRefusal(Refusal[ReadinessCategory]):
    """A second refusal the status line admits."""


class ForeignRefusal(Refusal[Literal["elsewhere"]]):
    """A refusal no line here admits."""


with isolated_catalogue() as CATALOGUE:

    class LocalModelPrepare(
        Outcome,
        name="local-model-prepare",
        version=1,
        layout=Layout(order=("version", "category")),
        unclassified="choice_refused",
        on_cancel="cancelled",
        when="always",
    ):
        category: PrepareCategory

    class OllamaProfile(
        Outcome, name="ollama-profile", version=1, unclassified="provider_refused",
        on_cancel=SILENT,
    ):
        category: ProfileCategory
        count: Annotated[int, Count(1)]

    class VoiceMemory(
        Outcome,
        name="voice-memory",
        version=1,
        layout=Layout(compact=False),
        category="refusal",
        unclassified="unclassified",
        on_cancel=SILENT,
    ):
        refusal: MemoryRefusal

    class ForegroundFailure(
        Observation,
        name="foreground-failure",
        version=1,
        layout=Layout(order=("version", "stage", "failure_kind", "text_phase")),
    ):
        stage: Literal["producer"]
        failure_kind: ProducerFailure
        text_phase: Literal["before_text", "after_text"]

    class KokoroStartupWitnessRefused(
        Outcome,
        name="kokoro-startup-witness-refused",
        version=1,
        stream="stderr",
        detach_on_failure=True,
        unclassified="cleanup_unknown",
        on_cancel=SILENT,
    ):
        category: StartupRefusal

    class PreviewStatus(
        Outcome,
        name="preview-status",
        version=1,
        unclassified="failed",
        on_cancel="interrupted",
        admits=(StagingRefusal, ReadinessRefusal),
        kind="failure_kind",
    ):
        category: StatusCategory
        failure_kind: ExceptionKind | None = None
        attempts: Annotated[int, Count(9)] = 0
