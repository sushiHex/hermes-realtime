"""Keep file dependencies alive until their existing process owners finalize.

This is in-process resource composition, not a qualification receipt, new process
authority or durable recovery journal. File-owner cleanup failure remains a
terminal refusal; a complete controller must retain its durable recovery record.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from types import TracebackType
from typing import TypeVar

from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_tool_environment import ImmutableToolEnvironmentV1
from scripts.qualification_tool_process import CompletedToolInvocationV1, _run_tool

T = TypeVar("T")


class OwnedQualificationCleanupError(RuntimeError):
    def __init__(self, owner: OwnedQualificationWorkV1) -> None:
        self.owner = owner
        super().__init__("qualification work retains resources pending cleanup")


class OwnedQualificationWorkV1:
    """Retain resources in dependency order and retry only existing Job cleanup."""

    def __init__(self) -> None:
        self._resources = ExitStack()
        self._pending: list[core._WindowsScenarioJobV1] = []
        self._children: list[OwnedQualificationWorkV1] = []
        self._closing = False
        self._closed = False
        self._unrecoverable: BaseException | None = None

    def _accepting(self) -> None:
        if self._closing:
            raise RuntimeError("qualification work is closing")

    def _retain_failure(self, error: BaseException) -> None:
        self._closing = True
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                self._retain_failure(nested)
        elif isinstance(error, OwnedQualificationCleanupError):
            if error.owner is self:
                return
            if all(owner is not error.owner for owner in self._children):
                self._children.append(error.owner)
        elif isinstance(error, core._WindowsFinalizationError):
            if error.owner is None:
                self._unrecoverable = error
            elif all(owner is not error.owner for owner in self._pending):
                self._pending.append(error.owner)

    def enter(self, context: AbstractContextManager[T]) -> T:
        self._accepting()
        try:
            return self._resources.enter_context(context)
        except BaseException as error:
            self._retain_failure(error)
            raise

    def run_tool(
        self,
        tools: ImmutableToolEnvironmentV1,
        role: str,
        arguments: tuple[str, ...],
        workspace: Path,
        *,
        timeout_milliseconds: int = 60_000,
    ) -> CompletedToolInvocationV1:
        self._accepting()
        try:
            return _run_tool(
                tools,
                role,
                arguments,
                workspace,
                timeout_milliseconds=timeout_milliseconds,
            )
        except BaseException as error:
            self._retain_failure(error)
            raise

    def close(self) -> None:
        self._closing = True
        if self._closed:
            return
        if self._unrecoverable is not None:
            raise OwnedQualificationCleanupError(self) from self._unrecoverable
        errors: list[BaseException] = []
        for child in tuple(self._children):
            try:
                child.close()
                self._children.remove(child)
            except BaseException as error:
                errors.append(error)
        for owner in tuple(self._pending):
            try:
                result = owner.finalize()
                if (
                    type(result) is not core._WindowsFinalizationResultV1
                    or not result.closed
                    or not result.zero_active_observed
                    or result.failures
                    or result.failed_handles
                ):
                    raise RuntimeError("process cleanup completion is unproven")
                self._pending.remove(owner)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("owned process cleanup remains incomplete", errors)
        try:
            self._resources.close()
        except BaseException as error:
            self._unrecoverable = error
            raise OwnedQualificationCleanupError(self) from error
        self._closed = True

    def __enter__(self) -> OwnedQualificationWorkV1:
        self._accepting()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is not None:
            self._retain_failure(error)
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise OwnedQualificationCleanupError(self) from cleanup
            raise OwnedQualificationCleanupError(self) from BaseExceptionGroup(
                "qualification work and cleanup failed",
                [error, cleanup],
            )
