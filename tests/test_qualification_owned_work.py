"""A retryable process failure must retain its complete resource dependency tree."""

from contextlib import contextmanager

import pytest


@pytest.mark.parametrize("grouped", [False, True])
def test_failed_process_cleanup_retains_resources_until_success(monkeypatch, grouped):
    from scripts import qualification_owned_work as work
    from scripts import qualify_evidence_slice_zero as core

    events = []

    @contextmanager
    def resource(name):
        events.append("enter:" + name)
        try:
            yield name
        finally:
            events.append("close:" + name)

    result = core._WindowsFinalizationResultV1(None, True, (), False, (), (), (), False)

    class Process:
        fail = True

        def finalize(self):
            events.append("cleanup")
            if self.fail:
                raise core._WindowsFinalizationError(result, self)
            return core._WindowsFinalizationResultV1(None, True, (), True, (), (), (), True)

    process = Process()

    def dispatch(*args, **kwargs):
        error = core._WindowsFinalizationError(result, process)
        raise (
            ExceptionGroup("primary and cleanup", [ValueError("primary"), error])
            if grouped
            else error
        )

    monkeypatch.setattr(work, "_run_tool", dispatch)
    owner = work.OwnedQualificationWorkV1()
    assert owner.enter(resource("tools")) == "tools"
    assert owner.enter(resource("inputs")) == "inputs"
    with pytest.raises((core._WindowsFinalizationError, ExceptionGroup)):
        owner.run_tool(None, "uv", ("--version",), None)
    with pytest.raises((core._WindowsFinalizationError, ExceptionGroup)):
        owner.close()
    assert events == ["enter:tools", "enter:inputs", "cleanup"]
    with pytest.raises(RuntimeError, match="closing"):
        owner.enter(resource("new"))
    with pytest.raises(RuntimeError, match="closing"):
        owner.run_tool(None, "uv", ("--version",), None)
    process.fail = False
    owner.close()
    assert events == [
        "enter:tools",
        "enter:inputs",
        "cleanup",
        "cleanup",
        "close:inputs",
        "close:tools",
    ]
    owner.close()
    assert events.count("cleanup") == 2


def test_exception_unwinding_retains_the_owner_for_cleanup_retry(monkeypatch):
    from scripts import qualification_owned_work as work
    from scripts import qualify_evidence_slice_zero as core

    released = []

    @contextmanager
    def resource():
        try:
            yield
        finally:
            released.append(True)

    result = core._WindowsFinalizationResultV1(None, True, (), False, (), (), (), False)

    class Process:
        def finalize(self):
            raise core._WindowsFinalizationError(result, self)

    def dispatch(*args, **kwargs):
        raise core._WindowsFinalizationError(result, Process())

    monkeypatch.setattr(work, "_run_tool", dispatch)
    with (
        pytest.raises(work.OwnedQualificationCleanupError) as error,
        work.OwnedQualificationWorkV1() as owner,
    ):
        owner.enter(resource())
        owner.run_tool(None, "uv", ("--version",), None)
    assert error.value.owner is owner
    assert not released
    # Deliberately clear the synthetic fault so test resources can be reclaimed.
    for process in owner._pending:
        process.finalize = lambda: core._WindowsFinalizationResultV1(
            None, True, (), True, (), (), (), True
        )
    owner.close()
    assert released == [True]


def test_plain_invocation_failure_releases_resources_without_claiming_success(monkeypatch):
    from scripts import qualification_owned_work as work

    released = []

    @contextmanager
    def resource():
        try:
            yield
        finally:
            released.append(True)

    def dispatch(*args, **kwargs):
        raise ValueError("normal exit failed after completed cleanup")

    monkeypatch.setattr(work, "_run_tool", dispatch)
    with pytest.raises(ValueError, match="normal exit"), work.OwnedQualificationWorkV1() as owner:
        owner.enter(resource())
        owner.run_tool(None, "uv", ("--version",), None)
    assert released == [True]


def test_work_owner_returns_only_the_actual_invocation_receipt(monkeypatch):
    from scripts import qualification_owned_work as work

    receipt = object()
    monkeypatch.setattr(work, "_run_tool", lambda *args, **kwargs: receipt)
    with work.OwnedQualificationWorkV1() as owner:
        assert owner.run_tool(None, "uv", ("--version",), None) is receipt
    with pytest.raises(RuntimeError, match="closing"):
        owner.run_tool(None, "uv", ("--version",), None)


def test_nested_failure_keeps_outer_tool_seals_until_inner_consumer_cleanup(monkeypatch):
    from scripts import qualification_owned_work as work
    from scripts import qualify_evidence_slice_zero as core

    released = []

    @contextmanager
    def resource(name):
        try:
            yield
        finally:
            released.append(name)

    failure = core._WindowsFinalizationResultV1(None, True, (), False, (), (), (), False)
    complete = core._WindowsFinalizationResultV1(None, True, (), True, (), (), (), True)

    class Process:
        fails = True

        def finalize(self):
            if self.fails:
                raise core._WindowsFinalizationError(failure, self)
            return complete

    process = Process()

    def dispatch(*args, **kwargs):
        raise core._WindowsFinalizationError(failure, process)

    monkeypatch.setattr(work, "_run_tool", dispatch)
    with (
        pytest.raises(work.OwnedQualificationCleanupError) as error,
        work.OwnedQualificationWorkV1() as outer,
    ):
        outer.enter(resource("tools"))
        with work.OwnedQualificationWorkV1() as inner:
            inner.enter(resource("packages"))
            inner.run_tool(None, "uv", ("--version",), None)
    try:
        assert error.value.owner is outer
        assert released == []
    finally:
        process.fails = False
        inner.close()
        outer.close()
    assert released == ["packages", "tools"]
