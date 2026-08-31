"""Bounded immutable context exposed to latency-critical model adapters."""

import re
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from hermes_realtime.speech import (
    DeliveredSpeechConfirmation,
    DeliveredSpeechLedger,
    SpeechDeliveryAdmission,
    Transcript,
)

_MAX_MESSAGES_LIMIT = 1024
_MAX_ACTIVE_TASKS_LIMIT = 256
_MAX_ITEM_CHARS_LIMIT = 65_536
_MAX_TASK_INCARNATIONS_LIMIT = 4096
_MAX_PENDING_ASSISTANT_ADMISSIONS_LIMIT = 256
_MAX_REVISION_LIMIT = 2**63 - 1
_TASK_ID_PATTERN = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_RUN_ID_PATTERN = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_PRIVATE_RUN_TOKEN_PATTERN = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _validate_model_visible_value(value: str) -> None:
    if _PRIVATE_RUN_TOKEN_PATTERN.search(value) is not None:
        raise PrivateRunDisclosureError(
            "private run identity must not appear in model-visible text"
        )


class ConversationRole(StrEnum):
    """Roles admitted to compact foreground prompt context."""

    USER = "user"
    ASSISTANT = "assistant"


class ActiveTaskCapacityError(RuntimeError):
    """Raised instead of silently evicting authoritative live task state."""


class ActiveTaskIdentityError(RuntimeError):
    """Raised when a task or run handle conflicts with active authority."""


class PrivateRunDisclosureError(ActiveTaskIdentityError):
    """Raised before a reserved private run token enters model-visible state."""


class AssistantTextCapacityError(RuntimeError):
    """Raised instead of retaining unbounded pre-playback admissions."""


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One compact model-visible conversation item."""

    role: str
    text: str

    def __post_init__(self) -> None:
        if type(self.role) is not str:
            raise TypeError("conversation role must be an exact built-in string")
        if type(self.text) is not str:
            raise TypeError("conversation text must be an exact built-in string")
        if self.role not in (ConversationRole.USER.value, ConversationRole.ASSISTANT.value):
            raise ValueError("conversation role is not supported")
        if not self.text.strip():
            raise ValueError("conversation text must not be blank")
        if len(self.text) > _MAX_ITEM_CHARS_LIMIT:
            raise ValueError("conversation text exceeds supported maximum")
        _validate_model_visible_value(self.text)


@dataclass(frozen=True, slots=True)
class ActiveTaskSummary:
    """Model-visible summary without the private Hermes run handle."""

    task_id: str
    objective: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str:
            raise TypeError("task_id must be an exact built-in string")
        if type(self.objective) is not str:
            raise TypeError("objective must be an exact built-in string")
        if not self.task_id.strip() or not self.objective.strip():
            raise ValueError("task summary fields must not be blank")
        if len(self.task_id) > 128 or _TASK_ID_PATTERN.fullmatch(self.task_id) is None:
            raise ValueError("task_id is not a valid model-visible identifier")
        if len(self.objective) > _MAX_ITEM_CHARS_LIMIT:
            raise ValueError("task objective exceeds supported maximum")
        _validate_model_visible_value(self.task_id)
        _validate_model_visible_value(self.objective)


AssistantTextAdmission = SpeechDeliveryAdmission


@dataclass(frozen=True, slots=True, eq=False)
class TaskAdmission:
    """Opaque authority reserving one task before transport exposure."""


@dataclass(frozen=True, slots=True)
class _ActiveTaskRecord:
    task_id: str
    run_id: str
    objective: str


@dataclass(frozen=True, slots=True)
class ConversationContextSnapshot:
    """Immutable point-in-time prompt context."""

    revision: int
    messages: tuple[ConversationMessage, ...]
    active_tasks: tuple[ActiveTaskSummary, ...]
    terminal_task_count: int = 0

    def __post_init__(self) -> None:
        if type(self.revision) is not int:
            raise TypeError("snapshot revision must be an exact integer")
        if self.revision < 0:
            raise ValueError("snapshot revision must be non-negative")
        if self.revision > _MAX_REVISION_LIMIT:
            raise ValueError("snapshot revision exceeds supported maximum")
        if type(self.messages) is not tuple:
            raise TypeError("snapshot messages must be an exact tuple")
        if type(self.active_tasks) is not tuple:
            raise TypeError("snapshot active_tasks must be an exact tuple")
        if type(self.terminal_task_count) is not int:
            raise TypeError("snapshot terminal_task_count must be an exact integer")
        if not 0 <= self.terminal_task_count <= _MAX_TASK_INCARNATIONS_LIMIT:
            raise ValueError("snapshot terminal_task_count is outside the supported range")
        if any(type(message) is not ConversationMessage for message in self.messages):
            raise TypeError("snapshot messages must contain exact ConversationMessage values")
        if any(type(task) is not ActiveTaskSummary for task in self.active_tasks):
            raise TypeError("snapshot active_tasks must contain exact ActiveTaskSummary values")
        if len(self.messages) > _MAX_MESSAGES_LIMIT:
            raise ValueError("snapshot message capacity exceeds supported maximum")
        if len(self.active_tasks) > _MAX_ACTIVE_TASKS_LIMIT:
            raise ValueError("snapshot task capacity exceeds supported maximum")
        messages = tuple(
            ConversationMessage(role=message.role, text=message.text) for message in self.messages
        )
        active_tasks = tuple(
            ActiveTaskSummary(task_id=task.task_id, objective=task.objective)
            for task in self.active_tasks
        )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "active_tasks", active_tasks)


class ConversationContextStore:
    """Retain a compact event-loop-local prompt context."""

    def __init__(
        self,
        *,
        max_messages: int = 16,
        max_active_tasks: int = 8,
        max_item_chars: int = 1024,
        max_task_incarnations: int = 1024,
        max_pending_assistant_admissions: int = 16,
    ) -> None:
        self._max_messages = self._bounded_positive_integer(
            max_messages,
            "max_messages",
            _MAX_MESSAGES_LIMIT,
        )
        self._max_active_tasks = self._bounded_positive_integer(
            max_active_tasks,
            "max_active_tasks",
            _MAX_ACTIVE_TASKS_LIMIT,
        )
        self._max_item_chars = self._bounded_positive_integer(
            max_item_chars,
            "max_item_chars",
            _MAX_ITEM_CHARS_LIMIT,
        )
        self._max_task_incarnations = self._bounded_positive_integer(
            max_task_incarnations,
            "max_task_incarnations",
            _MAX_TASK_INCARNATIONS_LIMIT,
        )
        self._max_pending_assistant_admissions = self._bounded_positive_integer(
            max_pending_assistant_admissions,
            "max_pending_assistant_admissions",
            _MAX_PENDING_ASSISTANT_ADMISSIONS_LIMIT,
        )
        self._messages: deque[ConversationMessage] = deque(maxlen=self._max_messages)
        self._active_tasks: dict[str, _ActiveTaskRecord] = {}
        self._task_ids_by_run_id: dict[str, str] = {}
        self._seen_task_ids: set[str] = set()
        self._seen_run_ids: set[str] = set()
        self._terminal_task_count = 0
        self._pending_task_admissions_by_object_id: dict[
            int,
            tuple[TaskAdmission, str, str],
        ] = {}
        self._assistant_admissions_by_object_id: dict[
            int,
            tuple[AssistantTextAdmission, str, bool],
        ] = {}
        self._revision = 0

    @property
    def max_active_tasks(self) -> int:
        """Return the configured concurrent active-task limit."""

        return self._max_active_tasks

    @property
    def max_item_chars(self) -> int:
        """Return the configured per-item public text limit."""

        return self._max_item_chars

    def record_user_transcript(self, transcript: Transcript) -> None:
        if type(transcript) is not Transcript:
            raise TypeError("transcript must be an exact Transcript value")
        if type(transcript.final) is not bool:
            raise TypeError("transcript final marker must be an exact boolean")
        if not transcript.final:
            raise ValueError("only final user transcripts enter context")
        self._append(ConversationRole.USER, transcript.text)

    def prepare_assistant_text(
        self,
        text: str,
        *,
        record_delivery: bool = True,
    ) -> AssistantTextAdmission:
        """Validate and bind text before admitting it to playback."""

        self._validate_text(text)
        self._validate_model_visible_text(text)
        if type(record_delivery) is not bool:
            raise TypeError("record_delivery must be an exact boolean")
        if len(self._assistant_admissions_by_object_id) >= self._max_pending_assistant_admissions:
            raise AssistantTextCapacityError("assistant admission capacity exhausted")
        admission = AssistantTextAdmission()
        self._assistant_admissions_by_object_id[id(admission)] = (
            admission,
            text,
            record_delivery,
        )
        return admission

    def record_assistant_generation(self, text: str) -> None:
        """Commit validated model output independently from speech delivery."""

        self.validate_assistant_generation(text)
        self._messages.append(ConversationMessage(role=ConversationRole.ASSISTANT.value, text=text))
        self._revision += 1

    def validate_assistant_generation(self, text: str) -> None:
        """Validate generated assistant text without mutating conversation history."""

        self._validate_text(text)
        self._validate_model_visible_text(text)

    def discard_assistant_text(self, admission: AssistantTextAdmission) -> None:
        self._resolve_assistant_admission(admission)
        del self._assistant_admissions_by_object_id[id(admission)]

    def record_assistant_delivery(
        self,
        *,
        admission: AssistantTextAdmission,
        ledger: DeliveredSpeechLedger,
        confirmation: DeliveredSpeechConfirmation,
    ) -> str:
        """Project only text bound before playback and confirmed by its ledger."""

        if type(ledger) is not DeliveredSpeechLedger:
            raise TypeError("ledger must be an exact DeliveredSpeechLedger")
        expected_text, record_delivery = self._resolve_assistant_admission(admission)
        delivered_text = ledger.confirmed_text(confirmation, admission)
        if delivered_text != expected_text:
            raise ValueError("delivery confirmation does not match assistant admission")
        self._validate_text(expected_text)
        delivered_text = ledger.consume_delivery_confirmation(confirmation, admission)
        del self._assistant_admissions_by_object_id[id(admission)]
        if record_delivery:
            self._messages.append(
                ConversationMessage(role=ConversationRole.ASSISTANT.value, text=expected_text)
            )
            self._revision += 1
        return delivered_text

    def prepare_task(self, task_id: str, objective: str) -> TaskAdmission:
        """Reserve bounded task capacity before an asynchronous dispatch."""

        self._validate_task_identifier(task_id)
        self._validate_text(objective)
        self._validate_model_visible_text(task_id)
        self._validate_model_visible_text(objective)
        if task_id in self._seen_task_ids:
            raise ActiveTaskIdentityError("task identity is active or retired")
        if len(self._seen_task_ids) >= self._max_task_incarnations:
            raise ActiveTaskCapacityError("task incarnation capacity exhausted")
        if (
            len(self._active_tasks) + len(self._pending_task_admissions_by_object_id)
            >= self._max_active_tasks
        ):
            raise ActiveTaskCapacityError("active task summary capacity exhausted")
        admission = TaskAdmission()
        self._pending_task_admissions_by_object_id[id(admission)] = (
            admission,
            task_id,
            objective,
        )
        self._seen_task_ids.add(task_id)
        return admission

    def discard_task(self, admission: TaskAdmission) -> None:
        """Release pending capacity while keeping its exposed task identity retired."""

        self._resolve_task_admission(admission)
        del self._pending_task_admissions_by_object_id[id(admission)]

    def record_reserved_task_accepted(
        self,
        *,
        admission: TaskAdmission,
        run_id: str,
    ) -> None:
        """Commit one reserved task only after authoritative acceptance."""

        self._validate_run_identifier(run_id)
        binding = self._resolve_task_admission(admission)
        if run_id in self._seen_run_ids:
            raise ActiveTaskIdentityError("run identity is active or retired")
        _, task_id, objective = binding
        del self._pending_task_admissions_by_object_id[id(admission)]
        self._active_tasks[task_id] = _ActiveTaskRecord(
            task_id=task_id,
            run_id=run_id,
            objective=objective,
        )
        self._task_ids_by_run_id[run_id] = task_id
        self._seen_run_ids.add(run_id)
        self._revision += 1

    def record_task_accepted(self, *, task_id: str, run_id: str, objective: str) -> None:
        self._validate_task_identifier(task_id)
        self._validate_run_identifier(run_id)
        self._validate_text(objective)
        self._validate_model_visible_text(task_id)
        self._validate_model_visible_text(objective)
        if task_id in self._seen_task_ids or run_id in self._seen_run_ids:
            raise ActiveTaskIdentityError("task or run identity is active or retired")
        if len(self._seen_task_ids) >= self._max_task_incarnations:
            raise ActiveTaskCapacityError("task incarnation capacity exhausted")
        if len(self._active_tasks) >= self._max_active_tasks:
            raise ActiveTaskCapacityError("active task summary capacity exhausted")
        self._active_tasks[task_id] = _ActiveTaskRecord(
            task_id=task_id,
            run_id=run_id,
            objective=objective,
        )
        self._task_ids_by_run_id[run_id] = task_id
        self._seen_task_ids.add(task_id)
        self._seen_run_ids.add(run_id)
        self._revision += 1

    def require_active_task(self, task_id: str) -> None:
        """Fail before transport unless the public task identity is currently active."""

        self._validate_task_identifier(task_id)
        if task_id not in self._active_tasks:
            raise ActiveTaskIdentityError("task identity is not active")

    def validate_task_cancellation(
        self,
        *,
        task_id: str,
        signaled_run_ids: tuple[str, ...],
    ) -> None:
        """Validate cancellation evidence without changing terminal authority."""

        self._validate_task_identifier(task_id)
        if type(signaled_run_ids) is not tuple:
            raise TypeError("signaled_run_ids must be an exact tuple")
        if len(signaled_run_ids) != 1:
            raise ActiveTaskIdentityError("cancellation must signal exactly one active run")
        run_id = signaled_run_ids[0]
        self._validate_run_identifier(run_id)
        active = self._active_tasks.get(task_id)
        if (
            active is None
            or active.run_id != run_id
            or self._task_ids_by_run_id.get(run_id) != task_id
        ):
            raise ActiveTaskIdentityError("cancellation does not match active authority")

    def record_task_completed(self, *, task_id: str, run_id: str) -> None:
        self._validate_task_identifier(task_id)
        self._validate_run_identifier(run_id)
        active = self._active_tasks.get(task_id)
        if (
            active is None
            or active.run_id != run_id
            or self._task_ids_by_run_id.get(run_id) != task_id
        ):
            raise ActiveTaskIdentityError("task completion does not match active authority")
        del self._active_tasks[task_id]
        del self._task_ids_by_run_id[run_id]
        self._terminal_task_count += 1
        self._revision += 1

    def snapshot(self) -> ConversationContextSnapshot:
        return ConversationContextSnapshot(
            revision=self._revision,
            messages=tuple(
                ConversationMessage(role=message.role, text=message.text)
                for message in self._messages
            ),
            active_tasks=tuple(
                ActiveTaskSummary(task_id=task.task_id, objective=task.objective)
                for task in self._active_tasks.values()
            ),
            terminal_task_count=self._terminal_task_count,
        )

    def _append(self, role: ConversationRole, text: str) -> None:
        self._validate_text(text)
        self._validate_model_visible_text(text)
        self._messages.append(ConversationMessage(role=role.value, text=text))
        self._revision += 1

    def _resolve_assistant_admission(
        self,
        admission: AssistantTextAdmission,
    ) -> tuple[str, bool]:
        if type(admission) is not AssistantTextAdmission:
            raise TypeError("admission must be an exact AssistantTextAdmission")
        binding = self._assistant_admissions_by_object_id.get(id(admission))
        if binding is None or binding[0] is not admission:
            raise KeyError("unknown assistant text admission")
        return binding[1], binding[2]

    def _resolve_task_admission(
        self,
        admission: TaskAdmission,
    ) -> tuple[TaskAdmission, str, str]:
        if type(admission) is not TaskAdmission:
            raise TypeError("admission must be an exact TaskAdmission")
        binding = self._pending_task_admissions_by_object_id.get(id(admission))
        if binding is None or binding[0] is not admission:
            raise KeyError("unknown task admission")
        return binding

    def _validate_text(self, text: str) -> None:
        if type(text) is not str:
            raise TypeError("conversation text must be an exact built-in string")
        if not text.strip():
            raise ValueError("conversation text must not be blank")
        if len(text) > self._max_item_chars:
            raise ValueError("conversation text exceeds max_item_chars")

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if type(value) is not str:
            raise TypeError(f"{name} must be an exact built-in string")
        if len(value) > 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None:
            raise ValueError(f"{name} is not a valid identifier")

    @staticmethod
    def _validate_task_identifier(value: str) -> None:
        ConversationContextStore._validate_identifier(value, "task_id")
        if _TASK_ID_PATTERN.fullmatch(value) is None:
            raise ActiveTaskIdentityError("task identity namespace is reserved")

    @staticmethod
    def _validate_run_identifier(value: str) -> None:
        ConversationContextStore._validate_identifier(value, "run_id")
        if _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ActiveTaskIdentityError(
                "private run identity must use the reserved deleg_ namespace"
            )

    @staticmethod
    def _validate_model_visible_text(value: str) -> None:
        _validate_model_visible_value(value)

    @staticmethod
    def _bounded_positive_integer(value: int, name: str, maximum: int) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value > maximum:
            raise ValueError(f"{name} exceeds supported maximum {maximum}")
        return value
