"""The evidence-line mechanism of ADR 0007: grammar, declaration, refusal, scope, faults."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace, new_class
from typing import Annotated, Any, Literal

import pytest

import hermes_realtime.markers as markers
from hermes_realtime.markers import (
    EXCEPTION_KINDS,
    SILENT,
    Count,
    Digest,
    ExceptionKind,
    ExceptionKinds,
    Layout,
    LineFault,
    LineRefused,
    MaxItems,
    Observation,
    Outcome,
    Refusal,
    Signed,
    Tally,
    catalogue,
    dropped,
    isolated_catalogue,
    parse_line,
    strict,
)
from tests.support import marker_lines as lines

ROOT = Path(__file__).resolve().parents[2]

Colour = Literal["red", "green"]
Phase = Literal["first", "second"]
Outcome3 = Literal["done", "failed", "stopped"]
Narrow = Literal["value", "other"]


class _Interrupt(BaseException):
    """An interrupt that is not an ``Exception``, and that the test runner treats as a failure."""


class _Stream:
    """A text stream that records each write and can be told to fail."""

    def __init__(self, fail: BaseException | None = None, *, at: str = "write") -> None:
        self.writes: list[str] = []
        self.flushes = 0
        self._fail = fail
        self._at = at

    def write(self, text: str) -> int:
        if self._fail is not None and self._at == "write":
            raise self._fail
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        if self._fail is not None and self._at == "flush":
            raise self._fail
        self.flushes += 1


@pytest.fixture(autouse=True)
def _own_catalogue() -> Iterator[None]:
    with isolated_catalogue():
        yield


@pytest.fixture(autouse=True)
def _own_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test runner re-binds sys.stdout between a fixture and its test, so the
    # mechanism's view of ``sys`` is replaced instead. It still reads the stream per write.
    monkeypatch.setattr(
        markers._core, "sys", SimpleNamespace(stdout=_Stream(), stderr=_Stream()),
    )


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> _Stream:
    stream = _Stream()
    monkeypatch.setattr(markers._core.sys, "stdout", stream)
    return stream


@pytest.fixture
def err(monkeypatch: pytest.MonkeyPatch) -> _Stream:
    stream = _Stream()
    monkeypatch.setattr(markers._core.sys, "stderr", stream)
    return stream


def _declare(annotation: Any, **options: Any) -> type[Observation]:
    """Declare one observation whose only field has ``annotation``."""
    options = {"name": "probe-line", "version": 1} | options
    return type("Probe", (Observation,), {"__annotations__": {"value": annotation}}, **options)


# -- the value grammar ------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        int,
        float,
        object,
        Any,
        bytes,
        list[str],
        dict[str, str],
        str | None,
        Colour | str,
        Literal[1, 2],
        Literal["ok", 3],
        Literal["has space"],
        Literal["trailing\n"],
        Literal[""],
        Annotated[str, Count(3)],
        Annotated[int, "no bound"],
        Annotated[int, Digest(64, of="file")],
        Annotated[float, Count(3)],
        Annotated[tuple[str, ...], MaxItems(3)],
        Annotated[tuple[Colour, ...], Count(3)],
        tuple[Colour, ...],
        Annotated[dict[str, int], Tally(9)],
        Annotated[dict[Colour, str], Tally(9)],
    ],
)
def test_a_field_that_is_not_an_evidence_kind_refuses_the_declaration(annotation: Any) -> None:
    with pytest.raises(TypeError, match="value"):
        _declare(annotation)
    assert "probe-line" not in catalogue()


def test_a_vocabulary_that_cannot_be_resolved_refuses_the_declaration() -> None:
    with pytest.raises(TypeError, match="module level"):
        _declare("NotDefinedAnywhere")
    assert catalogue() == {}


@pytest.mark.parametrize(
    ("annotation", "value", "encoded"),
    [
        (Colour, "red", '"red"'),
        (bool, True, "true"),
        (Annotated[int, Count(9)], 4, "4"),
        (Annotated[int, Signed(-3, 3)], -2, "-2"),
        (Annotated[str, Digest(16, of="file")], "0123456789abcdef", '"0123456789abcdef"'),
        (Colour | None, None, "null"),
        (Colour | Phase, "second", '"second"'),
        (Annotated[int, Count(9)] | None, None, "null"),
        (Annotated[tuple[Colour, ...], MaxItems(2)], ("green", "red"), '["green","red"]'),
        (Annotated[dict[Colour, int], Tally(9)], {"red": 2}, '{"red":2}'),
    ],
)
def test_each_kind_is_written_and_read_back(annotation: Any, value: Any, encoded: str) -> None:
    probe = _declare(annotation)
    line = probe(value=value)
    assert line.encode() == '[probe-line] {"value":' + encoded + ',"version":1}'
    assert probe.parse(line.encode()) == line
    assert line.fault == 0


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (Colour, "blue"),
        (Colour, "RED"),
        (Colour, None),
        (Colour, 1),
        (bool, 1),
        (bool, "true"),
        (Annotated[int, Count(9)], -1),
        (Annotated[int, Count(9)], True),
        (Annotated[int, Count(9)], 1.0),
        (Annotated[int, Count(9)], "4"),
        (Annotated[int, Signed(-3, 3)], 4),
        (Annotated[int, Signed(-3, 3)], -4),
        (Annotated[str, Digest(16, of="file")], "0123456789ABCDEF"),
        (Annotated[str, Digest(16, of="file")], "0123"),
        (Annotated[str, Digest(16, of="file")], "a private sentence"),
        (Annotated[tuple[Colour, ...], MaxItems(2)], ("red", "blue")),
        (Annotated[tuple[Colour, ...], MaxItems(2)], ("red", "red", "red")),
        (Annotated[tuple[Colour, ...], MaxItems(2)], "red"),
        (Annotated[dict[Colour, int], Tally(9)], {"blue": 1}),
        (Annotated[dict[Colour, int], Tally(9)], {"red": -1}),
        (Annotated[dict[Colour, int], Tally(9)], {"red": "many"}),
    ],
)
def test_a_value_outside_its_kind_is_never_written(annotation: Any, value: Any) -> None:
    probe = _declare(annotation)
    line = probe(value=value)
    assert line.encode() == '[probe-line] {"fault":1,"value":null,"version":1}'
    assert line.fault == 1 and line.value is None
    with strict(), pytest.raises(LineFault, match=r"probe-line\.value") as raised:
        probe(value=value)
    assert repr(value) not in str(raised.value) and str(value) not in str(raised.value)


def test_a_count_saturates_at_its_declared_maximum() -> None:
    probe = _declare(Annotated[int, Count(9)])
    assert probe(value=10**9).encode() == '[probe-line] {"value":9,"version":1}'
    with pytest.raises(LineRefused, match="value"):
        probe.parse('[probe-line] {"value":10,"version":1}')


@pytest.mark.parametrize("build", [lambda: Count(-1), lambda: Count(True), lambda: Signed(3, -3),
                                   lambda: Digest(20, of="file"), lambda: Digest(16, of="a b"),
                                   lambda: MaxItems(0), lambda: Tally(-1)])
def test_a_bound_must_be_a_real_bound(build: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        build()


# -- the declaration --------------------------------------------------------------------


def test_a_declaration_is_registered_under_its_name_once() -> None:
    first = _declare(Colour)
    assert catalogue() == {"probe-line": first}
    with pytest.raises(TypeError, match="probe-line"):
        _declare(Colour)
    assert catalogue() == {"probe-line": first}


@pytest.mark.parametrize(
    "options",
    [
        {"name": None},
        {"name": "Has_Capitals"},
        {"name": "has space"},
        {"name": "x" * 49},
        {"version": "1"},
        {"version": True},
        {"version": 0},
        {"stream": "stdlog"},
        {"detach_on_failure": True},
        {"layout": Layout(order=("version",))},
        {"layout": Layout(order=("value", "version", "extra"))},
        {"layout": Layout(order=("value", "value", "version"))},
        {"unknown_option": 1},
    ],
)
def test_a_malformed_declaration_is_refused(options: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        _declare(Colour, **options)


def test_the_version_must_be_stated_even_when_a_legacy_line_has_none() -> None:
    with pytest.raises(TypeError, match="version"):
        type("Probe", (Observation,), {"__annotations__": {"value": Colour}}, name="probe-line")
    legacy = _declare(Colour, version=None)
    assert legacy(value="red").encode() == '[probe-line] {"value":"red"}'


@pytest.mark.parametrize("field", ["version", "fault", "_hidden", "emit", "encode"])
def test_a_reserved_field_name_is_refused(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        type("Probe", (Observation,), {"__annotations__": {field: Colour}},
             name="probe-line", version=1)


def test_a_default_must_itself_be_a_valid_value() -> None:
    with pytest.raises(TypeError, match="value"):
        type("Probe", (Observation,), {"__annotations__": {"value": Colour}, "value": "blue"},
             name="probe-line", version=1)
    probe = type("Probe", (Observation,), {"__annotations__": {"value": Colour}, "value": "red"},
                 name="other-line", version=1)
    assert probe().encode() == '[other-line] {"value":"red","version":1}'


@pytest.mark.parametrize(
    ("line", "legacy"),
    [
        (
            lines.LocalModelPrepare(category="deadline"),
            "[local-model-prepare] "
            + json.dumps({"version": 1, "category": "deadline"}, separators=(",", ":")),
        ),
        (
            lines.OllamaProfile(category="override_refused", count=1),
            '[ollama-profile] {"category":"override_refused","count":1,"version":1}',
        ),
        (
            lines.VoiceMemory(refusal="capacity"),
            '[voice-memory] {"refusal": "capacity", "version": 1}',
        ),
        (
            lines.ForegroundFailure(stage="producer", failure_kind="timeout",
                                    text_phase="before_text"),
            "[foreground-failure] " + json.dumps({
                "version": 1, "stage": "producer", "failure_kind": "timeout",
                "text_phase": "before_text",
            }, separators=(",", ":")),
        ),
        (
            lines.KokoroStartupWitnessRefused(category="policy_rejected"),
            "[kokoro-startup-witness-refused] " + json.dumps(
                {"version": 1, "category": "policy_rejected"},
                separators=(",", ":"), sort_keys=True,
            ),
        ),
    ],
)
def test_a_declared_layout_reproduces_the_bytes_written_today(line: Any, legacy: str) -> None:
    assert line.encode() == legacy
    assert type(line).parse(legacy) == line
    assert parse_line(legacy, lines.CATALOGUE) == line


def test_a_layout_can_omit_the_space_after_the_prefix_and_the_flush(out: _Stream) -> None:
    probe = _declare(Colour, layout=Layout(space=False, flush=False))
    probe(value="red").emit()
    assert out.writes == ['[probe-line]{"value":"red","version":1}\n'] and out.flushes == 0
    assert probe.parse('[probe-line]{"value":"red","version":1}').value == "red"
    with pytest.raises(LineRefused, match="prefix"):
        probe.parse('[probe-line] {"value":"red","version":1}')


# -- parsing ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ('[other-line] {"value":"red","version":1}', "prefix"),
        ('[probe-line]{"value":"red","version":1}', "prefix"),
        ('[probe-line]  {"value":"red","version":1}', "json"),
        ('x[probe-line] {"value":"red","version":1}', "prefix"),
        ('[probe-line] {"value":"red","version":1}\n', "json"),
        ('[probe-line] {"value":"red","version":1} trailing', "json"),
        ("[probe-line] not json", "json"),
        ('[probe-line] ["red"]', "json"),
        ('[probe-line] {"value":"red","value":"red","version":1}', "duplicate_key"),
        ('[probe-line] {"value":"red"}', "keys"),
        ('[probe-line] {"value":"red","version":1,"extra":1}', "keys"),
        ('[probe-line] {"value":"blue","version":1}', "value"),
        ('[probe-line] {"value":null,"version":1}', "value"),
        ('[probe-line] {"value":"red","version":2}', "version"),
        ('[probe-line] {"value":"red","version":true}', "version"),
        ('[probe-line] {"value":"red","version":1.0}', "json"),
        ('[probe-line] {"value":NaN,"version":1}', "json"),
        ('[probe-line] {"fault":0,"value":"red","version":1}', "fault"),
        ('[probe-line] {"fault":true,"value":null,"version":1}', "fault"),
    ],
)
def test_parse_refuses_the_whole_line(text: str, category: str) -> None:
    probe = _declare(Colour)
    with pytest.raises(LineRefused) as refused:
        probe.parse(text)
    assert refused.value.category == category


def test_parse_reads_a_faulted_line_as_faulted() -> None:
    probe = _declare(Colour)
    line = probe.parse('[probe-line] {"fault":1,"value":null,"version":1}')
    assert line.fault == 1 and line.value is None
    assert line == probe(value="not a member")


def test_parse_line_finds_the_declaration_by_name() -> None:
    probe = _declare(Colour)
    assert parse_line('[probe-line] {"value":"red","version":1}') == probe(value="red")
    for text, category in (
        ('[unknown-line] {"value":"red"}', "unknown_name"),
        ("ordinary output", "not_a_line"),
        ("[Probe-Line] {}", "not_a_line"),
    ):
        with pytest.raises(LineRefused) as refused:
            parse_line(text)
        assert refused.value.category == category


# -- refusals ---------------------------------------------------------------------------


def test_a_refusal_is_its_category_and_nothing_else() -> None:
    refusal = lines.StagingRefusal("run_directory")
    assert refusal.category == "run_directory" and str(refusal) == "run_directory"
    assert refusal.args == ("run_directory",)
    assert lines.StagingRefusal.categories == {"run_directory", "output_receipt_refused"}


@pytest.mark.parametrize("category", ["tcp_port_invalid", "a private sentence", 3, None])
def test_a_refusal_takes_only_a_category_it_declares(category: Any) -> None:
    with pytest.raises(TypeError) as raised:
        lines.StagingRefusal(category)
    assert str(category) not in str(raised.value)
    with pytest.raises(TypeError):
        lines.StagingRefusal("run_directory", "a message")  # type: ignore[call-arg]


def test_a_refusal_type_cannot_take_a_message_or_a_free_vocabulary() -> None:
    with pytest.raises(TypeError, match="category"):
        Refusal("anything")
    with pytest.raises(TypeError, match="Literal"):
        new_class("Loose", (Refusal[str],))
    with pytest.raises(TypeError, match="Literal"):
        new_class("Mixed", (Refusal[Literal["ok", "has space"]],))
    with pytest.raises(TypeError, match="Literal"):
        new_class("Bare", (Refusal,))
    for member in ("__init__", "__str__", "__repr__"):
        with pytest.raises(TypeError, match=member):
            new_class(
                "Chatty", (Refusal[Colour],),
                exec_body=lambda body, member=member: body.update({member: lambda *_: None}),
            )
    narrower = new_class("Narrower", (lines.StagingRefusal,))
    assert narrower("run_directory").category == "run_directory"


def test_a_parse_refusal_is_itself_a_typed_refusal() -> None:
    assert issubclass(LineRefused, Refusal)
    assert "duplicate_key" in LineRefused.categories


# -- writing ----------------------------------------------------------------------------


def test_an_observation_is_one_write_of_one_line(out: _Stream, err: _Stream) -> None:
    lines.ForegroundFailure(stage="producer", failure_kind="value", text_phase="after_text").emit()
    assert out.writes == [
        '[foreground-failure] {"version":1,"stage":"producer","failure_kind":"value",'
        '"text_phase":"after_text"}\n'
    ]
    assert out.flushes == 1 and err.writes == []


def test_a_line_is_written_at_most_once(out: _Stream) -> None:
    line = lines.ForegroundFailure(stage="producer", failure_kind="value", text_phase="after_text")
    before = dropped()
    line.emit()
    line.emit()
    assert len(out.writes) == 1 and dropped() == before + 1
    again = lines.ForegroundFailure(stage="producer", failure_kind="x", text_phase="after_text")
    with strict(), pytest.raises(LineFault):
        again.emit()
        again.emit()


def test_an_outcome_line_has_no_free_standing_emit() -> None:
    assert not hasattr(lines.OllamaProfile(category="provider_refused", count=1), "emit")
    assert not hasattr(Outcome, "emit")
    assert not hasattr(lines.ForegroundFailure, "scope")
    assert not hasattr(lines.ForegroundFailure, "raising")


def test_raising_writes_the_line_and_returns_the_same_exception(out: _Stream) -> None:
    error = ValueError("local model profile is invalid")
    returned = lines.OllamaProfile(category="thinking_refused", count=1).raising(error)
    assert returned is error
    assert out.writes == [
        '[ollama-profile] {"category":"thinking_refused","count":1,"version":1}\n'
    ]


def test_raising_takes_the_category_from_an_admitted_refusal(out: _Stream) -> None:
    refusal = lines.ReadinessRefusal("tcp_port_invalid")
    assert lines.PreviewStatus(category="ready").raising(refusal) is refusal
    assert json.loads(out.writes[0].split("] ", 1)[1]) == {
        "attempts": 0, "category": "tcp_port_invalid", "failure_kind": "other", "version": 1,
    }


def test_a_line_goes_to_the_stream_it_declares(out: _Stream, err: _Stream) -> None:
    lines.KokoroStartupWitnessRefused(category="schema_rejected").raising(RuntimeError())
    assert err.writes == [
        '[kokoro-startup-witness-refused] {"category":"schema_rejected","version":1}\n'
    ]
    assert out.writes == []


def test_the_real_stream_is_looked_up_when_the_line_is_written(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(markers._core, "sys", sys)
    line = lines.OllamaProfile(category="provider_refused", count=1)
    capsys.readouterr()
    line.raising(ValueError())
    assert capsys.readouterr().out == (
        '[ollama-profile] {"category":"provider_refused","count":1,"version":1}\n'
    )


# -- the scope --------------------------------------------------------------------------


def _status(out: _Stream) -> dict[str, Any]:
    assert len(out.writes) == 1 and out.writes[0].endswith("\n")
    parsed = lines.PreviewStatus.parse(out.writes[0][:-1])
    return json.loads(parsed.encode().split("] ", 1)[1])


def test_a_scope_writes_one_line_and_lets_the_exception_through(out: _Stream) -> None:
    error = lines.StagingRefusal("output_receipt_refused")
    with pytest.raises(lines.StagingRefusal) as raised, lines.PreviewStatus(
        category="ready", attempts=2,
    ).scope():
        raise error
    assert raised.value is error and raised.value.__context__ is None
    assert _status(out) == {
        "attempts": 2, "category": "output_receipt_refused", "failure_kind": "other",
        "version": 1,
    }


def test_a_scope_records_an_unclassified_failure_with_its_kind(out: _Stream) -> None:
    with pytest.raises(TimeoutError), lines.PreviewStatus(category="ready").scope():
        raise TimeoutError("a message that must not be written")
    assert _status(out) == {
        "attempts": 0, "category": "failed", "failure_kind": "TimeoutError", "version": 1,
    }
    assert "message" not in out.writes[0]


def test_a_category_the_body_recorded_survives_an_ordinary_exception(out: _Stream) -> None:
    with pytest.raises(OSError), lines.PreviewStatus(category="ready").scope() as line:
        line.category = "host_ready_receipt"
        raise OSError
    assert _status(out)["category"] == "host_ready_receipt"


def test_a_refusal_overrides_what_the_body_recorded(out: _Stream) -> None:
    with pytest.raises(lines.StagingRefusal), lines.PreviewStatus(category="ready").scope() as line:
        line.category = "host_ready_receipt"
        raise lines.StagingRefusal("run_directory")
    assert _status(out)["category"] == "run_directory"


def test_a_scope_that_ends_normally_writes_only_when_asked(out: _Stream) -> None:
    with lines.PreviewStatus(category="ready").scope():
        pass
    assert out.writes == []
    with lines.PreviewStatus(category="ready").scope() as line:
        line.attempts = 3
        line.report()
    assert _status(out) == {
        "attempts": 3, "category": "ready", "failure_kind": None, "version": 1,
    }


def test_a_line_declared_always_writes_on_a_normal_exit(out: _Stream) -> None:
    with lines.LocalModelPrepare(category="choice_refused").scope() as line:
        line.category = "prepared"
    assert out.writes == ['[local-model-prepare] {"version":1,"category":"prepared"}\n']


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cancellation_records_what_the_line_declares(
    out: _Stream, interrupt: type[BaseException],
) -> None:
    with pytest.raises(interrupt), lines.PreviewStatus(category="ready").scope():
        raise interrupt
    assert _status(out)["category"] == "interrupted"
    out.writes.clear()
    with pytest.raises(interrupt), lines.OllamaProfile(
        category="provider_refused", count=1,
    ).scope():
        raise interrupt
    assert out.writes == []


def test_a_task_cancellation_is_cancellation(out: _Stream) -> None:
    import asyncio

    with pytest.raises(asyncio.CancelledError), lines.PreviewStatus(category="ready").scope():
        raise asyncio.CancelledError
    assert _status(out)["category"] == "interrupted"


def test_a_scope_exits_once(out: _Stream) -> None:
    scope = lines.LocalModelPrepare(category="prepared").scope()
    before = dropped()
    with scope as line:
        pass
    assert scope.__exit__(None, None, None) is False
    with pytest.raises(ValueError), scope:
        raise ValueError
    assert len(out.writes) == 1 and dropped() == before
    assert line.category == "prepared"


@pytest.mark.parametrize(
    "options",
    [
        {"unclassified": "failed"},
        {"on_cancel": SILENT},
        {"unclassified": "not_a_member", "on_cancel": SILENT},
        {"unclassified": "failed", "on_cancel": "not_a_member"},
        {"unclassified": "failed", "on_cancel": SILENT, "when": "sometimes"},
        {"unclassified": "failed", "on_cancel": SILENT, "category": "missing"},
        {"unclassified": "failed", "on_cancel": SILENT, "admits": (lines.StagingRefusal,)},
        {"unclassified": "failed", "on_cancel": SILENT, "admits": (ValueError,)},
        {"unclassified": "failed", "on_cancel": SILENT, "kind": "missing"},
        {"unclassified": "failed", "on_cancel": SILENT, "kind": "category"},
    ],
)
def test_an_outcome_must_declare_how_it_settles(options: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        type("Probe", (Outcome,), {"__annotations__": {"category": Outcome3}},
             name="probe-line", version=1, **options)


def test_the_kind_field_must_hold_exactly_the_kinds_of_its_table() -> None:
    fields = {"category": Outcome3, "kind": Narrow | None}
    settle = {"unclassified": "failed", "on_cancel": SILENT, "kind": "kind"}
    with pytest.raises(TypeError, match="kind"):
        type("Probe", (Outcome,), {"__annotations__": fields, "kind": None},
             name="probe-line", version=1, **settle)
    table = ExceptionKinds({ValueError: "value"}, other="other")
    probe = type("Probe", (Outcome,), {"__annotations__": fields, "kind": None},
                 name="probe-line", version=1, kinds=table, **settle)
    assert probe(category="done").raising(ValueError()) is not None


# -- failures of the write --------------------------------------------------------------


@pytest.mark.parametrize("failure", [OSError("broken pipe"), ValueError("closed file")])
@pytest.mark.parametrize("at", ["write", "flush"])
def test_a_failed_write_is_dropped_and_counted_and_never_replaces_the_refusal(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, at: str,
) -> None:
    monkeypatch.setattr(markers._core.sys, "stdout", _Stream(failure, at=at))
    before = dropped()
    error = lines.StagingRefusal("run_directory")
    with pytest.raises(lines.StagingRefusal) as raised, lines.PreviewStatus(
        category="ready",
    ).scope():
        raise error
    assert raised.value is error and raised.value.__context__ is None
    assert dropped() == before + 1
    assert lines.OllamaProfile(category="provider_refused", count=1).raising(error) is error
    lines.ForegroundFailure(stage="producer", failure_kind="value", text_phase="after_text").emit()
    assert dropped() == before + 3


def test_a_missing_stream_is_dropped_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(markers._core.sys, "stdout", None)
    before = dropped()
    with lines.LocalModelPrepare(category="prepared").scope():
        pass
    assert dropped() == before + 1


def test_an_interrupt_during_the_write_does_not_replace_an_exception_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(markers._core.sys, "stdout", _Stream(_Interrupt()))
    error = ValueError("the refusal")
    with pytest.raises(ValueError) as raised, lines.PreviewStatus(category="ready").scope():
        raise error
    assert raised.value is error


def test_an_interrupt_during_the_write_propagates_when_nothing_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(markers._core.sys, "stdout", _Stream(_Interrupt()))
    with pytest.raises(_Interrupt):
        lines.ForegroundFailure(
            stage="producer", failure_kind="value", text_phase="after_text",
        ).emit()
    with pytest.raises(_Interrupt), lines.LocalModelPrepare(category="prepared").scope():
        pass


def test_a_failed_standard_error_is_detached_only_where_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _Stream(OSError())
    monkeypatch.setattr(markers._core.sys, "stderr", broken)
    lines.KokoroStartupWitnessRefused(category="schema_rejected").raising(RuntimeError())
    assert markers._core.sys.stderr is None
    monkeypatch.setattr(markers._core.sys, "stderr", broken)
    probe = _declare(Colour, stream="stderr")
    probe(value="red").emit()
    assert markers._core.sys.stderr is broken


# -- faults -----------------------------------------------------------------------------


def test_a_refusal_the_line_does_not_admit_is_a_fault_and_not_a_generic_label(
    out: _Stream,
) -> None:
    with pytest.raises(lines.ForeignRefusal), lines.PreviewStatus(category="ready").scope():
        raise lines.ForeignRefusal("elsewhere")
    assert out.writes == [
        '[preview-status] {"attempts":0,"category":null,"failure_kind":"other","fault":1,'
        '"version":1}\n'
    ]
    with strict(), pytest.raises(LineFault, match=r"preview-status\.category") as raised, (
        lines.PreviewStatus(category="ready").scope()
    ):
        raise lines.ForeignRefusal("elsewhere")
    assert type(raised.value.__context__) is lines.ForeignRefusal


def test_a_fault_never_changes_control_flow_outside_strict_mode(out: _Stream) -> None:
    text: Any = "a private sentence"
    with lines.PreviewStatus(category="ready").scope() as line:
        line.category = text
        line.attempts = -5
        line.undeclared = text
        line.report()
    written = out.writes[0]
    assert "private" not in written
    assert json.loads(written.split("] ", 1)[1]) == {
        "attempts": None, "category": None, "failure_kind": None, "fault": 3, "version": 1,
    }
    line = lines.OllamaProfile(category="provider_refused", count=1, model=text)  # type: ignore[call-arg]
    assert line.fault == 1 and "private" not in line.encode()
    assert lines.OllamaProfile(category="provider_refused").fault == 1  # type: ignore[call-arg]


def test_strict_mode_raises_at_the_faulting_statement() -> None:
    text: Any = "a private sentence"
    with strict():
        line = lines.PreviewStatus(category="ready")
        with pytest.raises(LineFault, match=r"preview-status\.category") as raised:
            line.category = text
        assert "private" not in str(raised.value)
        with pytest.raises(LineFault, match=r"preview-status\.undeclared"):
            line.undeclared = 1
        with pytest.raises(LineFault, match=r"ollama-profile\.count"):
            lines.OllamaProfile(category="provider_refused")  # type: ignore[call-arg]
    assert lines.PreviewStatus(category=text).fault == 1


def test_a_valid_value_clears_the_fault_it_replaces() -> None:
    line = lines.PreviewStatus(category="nonsense")  # type: ignore[arg-type]
    assert line.fault == 1
    line.category = "ready"
    assert line.fault == 0 and line.category == "ready"


# -- exception kinds --------------------------------------------------------------------


def test_exception_kinds_are_read_by_exact_type() -> None:
    class Narrower(ValueError):
        pass

    assert EXCEPTION_KINDS.of(ValueError("text")) == "ValueError"
    assert EXCEPTION_KINDS.of(Narrower("text")) == "other"
    assert EXCEPTION_KINDS.of(KeyError("text")) == "other"
    assert EXCEPTION_KINDS.labels == set(ExceptionKind.__args__)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ExceptionKinds({ValueError: "has space"}),
        lambda: ExceptionKinds({ValueError: "value"}, other="not ok"),
        lambda: ExceptionKinds({"ValueError": "value"}),  # type: ignore[dict-item]
        lambda: ExceptionKinds({int: "value"}),  # type: ignore[dict-item]
    ],
)
def test_an_exception_kind_table_holds_only_types_and_labels(build: Any) -> None:
    with pytest.raises(TypeError):
        build()


# -- the unit ---------------------------------------------------------------------------


def test_the_mechanism_imports_only_the_standard_library() -> None:
    # A fresh interpreter: the isolated worker and the credential helper import nothing else.
    completed = subprocess.run(
        [sys.executable, "-I", "-c", """
import sys

sys.path.insert(0, sys.argv[1])
before = set(sys.modules)
import hermes_realtime.markers as markers

loaded = {name.partition(".")[0] for name in set(sys.modules) - before}
foreign = sorted(loaded - set(sys.stdlib_module_names) - {"hermes_realtime"})
assert not foreign, foreign
package = sorted(name for name in sys.modules if name.startswith("hermes_realtime"))
assert package == ["hermes_realtime", "hermes_realtime.markers", "hermes_realtime.markers._core"]
assert markers.catalogue() == {}
""", str(ROOT / "src")],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_no_product_code_uses_the_mechanism_yet() -> None:
    users = sorted(
        path.relative_to(ROOT).as_posix()
        for base in ("src", "scripts")
        for path in (ROOT / base).rglob("*.py")
        if "markers" not in path.relative_to(ROOT / base).parts[:2]
        and "hermes_realtime.markers" in path.read_text(encoding="utf-8")
    )
    assert users == []


def test_the_public_names_are_the_declared_ones() -> None:
    assert sorted(markers.__all__) == sorted(
        name for name in vars(markers) if not name.startswith("_") and name != "annotations"
    )


def test_a_strict_type_check_refuses_free_strings_at_call_sites(tmp_path: Path) -> None:
    def check(name: str) -> tuple[set[int], str]:
        # A fresh interpreter: the type checker raises the recursion limit of the process
        # it runs in, which would change what later tests in this run observe.
        completed = subprocess.run(
            [
                sys.executable, "-m", "mypy", "--strict", "--python-version", "3.11",
                "--no-error-summary", "--cache-dir", str(tmp_path / "cache"),
                "--show-error-codes", str(Path("tests") / "support" / name),
            ],
            cwd=ROOT,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            text=True,
            timeout=240,
            check=False,
        )
        report = completed.stdout
        assert completed.stderr == "", completed.stderr
        found = re.findall(r"^(.*?):(\d+): error:", report, re.MULTILINE)
        elsewhere = sorted({path for path, _ in found if not path.endswith(name)})
        assert not elsewhere, report
        return {int(number) for _, number in found}, report

    accepted, report = check("marker_calls_accepted.py")
    assert accepted == set(), report
    source = (ROOT / "tests" / "support" / "marker_calls_refused.py").read_text(encoding="utf-8")
    marked = {
        number for number, line in enumerate(source.splitlines(), 1) if line.endswith("# refused")
    }
    refused, report = check("marker_calls_refused.py")
    assert len(marked) == 18
    assert refused == marked, report
