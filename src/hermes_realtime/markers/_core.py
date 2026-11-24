"""One contract for a content-free evidence line (ADR 0007).

A line is ``[name] {json}``. Its declaration is its emitter, its parser and its test oracle,
and its fields can hold only what this module's kinds describe: a label from a declared set,
a flag, a bounded number, a fixed-length digest, or a bounded list or tally of those. There
is no kind for a string. A declaration that asks for one fails when its module is imported.

This module imports only the standard library, because emitters that must stay free of
every other dependency use it.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType, NoneType, TracebackType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Self,
    TypeVar,
    Union,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
)

_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_NAME_LIMIT = 48
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}\Z")
_HEX = frozenset("0123456789abcdef")
_DIGEST_LENGTHS = (16, 32, 64)
_LINE = re.compile(r"\[([a-z][a-z0-9]*(?:-[a-z0-9]+)*)\]")
_RESERVED = frozenset({"version", "fault"})
_MISSING: Final = object()


def _exact_int(value: object) -> bool:
    return type(value) is int


def _is_label(value: object) -> bool:
    return type(value) is str and _LABEL.match(value) is not None


# -- bounds a declaration attaches to a field ---------------------------------------------


class Count:
    """A count from zero to ``maximum``; a larger value is written as the maximum."""

    def __init__(self, maximum: int) -> None:
        if not _exact_int(maximum) or maximum < 0:
            raise TypeError("a count needs a non-negative integer maximum")
        self.maximum = maximum


class Signed:
    """An integer between ``minimum`` and ``maximum`` inclusive."""

    def __init__(self, minimum: int, maximum: int) -> None:
        if not _exact_int(minimum) or not _exact_int(maximum) or minimum > maximum:
            raise TypeError("a signed number needs integer bounds in order")
        self.minimum = minimum
        self.maximum = maximum


class Digest:
    """A lowercase hexadecimal digest of a fixed length; ``of`` labels what was hashed."""

    def __init__(self, length: int, *, of: str) -> None:
        if not _exact_int(length) or length not in _DIGEST_LENGTHS:
            raise TypeError("a digest is 16, 32 or 64 hexadecimal characters")
        if not _is_label(of):
            raise TypeError("a digest names what was hashed with a label")
        self.length = length
        self.of = of


class MaxItems:
    """The most labels a list field may hold."""

    def __init__(self, maximum: int) -> None:
        if not _exact_int(maximum) or maximum < 1:
            raise TypeError("a list needs a positive integer maximum")
        self.maximum = maximum


class Tally:
    """A tally by label; each count saturates at ``maximum``."""

    def __init__(self, maximum: int) -> None:
        if not _exact_int(maximum) or maximum < 0:
            raise TypeError("a tally needs a non-negative integer maximum")
        self.maximum = maximum


class _Invalid(Exception):
    """A value is outside its kind. It carries nothing, so the value cannot leak."""


@dataclass(frozen=True)
class _Kind:
    """What one field may hold. ``labels`` is set for a label field only."""

    shape: str
    optional: bool = False
    labels: frozenset[str] = frozenset()
    low: int = 0
    high: int = 0
    length: int = 0

    def admit(self, value: object, *, saturate: bool) -> object:
        """Return ``value`` as it is stored, or raise when it is outside this kind."""
        if value is None:
            if self.optional:
                return None
            raise _Invalid
        if self.shape == "label":
            if type(value) is str and value in self.labels:
                return value
        elif self.shape == "flag":
            if type(value) is bool:
                return value
        elif self.shape == "number":
            if type(value) is int and self.low <= value:
                if value <= self.high:
                    return value
                if saturate and self.low == 0:
                    return self.high
        elif self.shape == "digest":
            if type(value) is str and len(value) == self.length and set(value) <= _HEX:
                return value
        elif self.shape == "labels":
            if type(value) is tuple or type(value) is list:
                items = tuple(value)
                if len(items) <= self.high and all(
                    type(item) is str and item in self.labels for item in items
                ):
                    return items
        elif self.shape == "tally" and type(value) is dict:
            stored: dict[str, int] = {}
            for key, count in value.items():
                if type(key) is not str or key not in self.labels or type(count) is not int:
                    raise _Invalid
                if count < 0 or (count > self.high and not saturate):
                    raise _Invalid
                stored[key] = min(count, self.high)
            return stored
        raise _Invalid


def _labels(annotation: object) -> frozenset[str] | None:
    """The members of a ``Literal`` of labels, or of a union of them; None for anything else."""
    origin = get_origin(annotation)
    if origin is Literal:
        members = get_args(annotation)
        return frozenset(members) if members and all(map(_is_label, members)) else None
    if origin in (Union, UnionType):
        parts = [_labels(part) for part in get_args(annotation)]
        if parts and all(part is not None for part in parts):
            return frozenset().union(*(part for part in parts if part is not None))
    return None


def _kind(annotation: object) -> _Kind | None:
    """The kind an annotation declares, or None when it declares none of them."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        parts = get_args(annotation)
        rest = [part for part in parts if part is not NoneType]
        if len(rest) != len(parts):
            if not rest:
                return None
            inner = _kind(rest[0] if len(rest) == 1 else Union[tuple(rest)])  # noqa: UP007
            if inner is None:
                return None
            return _Kind(inner.shape, True, inner.labels, inner.low, inner.high, inner.length)
    members = _labels(annotation)
    if members is not None:
        return _Kind("label", labels=members)
    if annotation is bool:
        return _Kind("flag")
    if origin is not Annotated:
        return None
    base, *bounds = get_args(annotation)
    if len(bounds) != 1:
        return None
    bound = bounds[0]
    if base is int and type(bound) is Count:
        return _Kind("number", high=bound.maximum)
    if base is int and type(bound) is Signed:
        return _Kind("number", low=bound.minimum, high=bound.maximum)
    if base is str and type(bound) is Digest:
        return _Kind("digest", length=bound.length)
    if get_origin(base) is tuple and type(bound) is MaxItems:
        item, *more = get_args(base)
        members = _labels(item)
        if more == [Ellipsis] and members is not None:
            return _Kind("labels", labels=members, high=bound.maximum)
    if get_origin(base) is dict and type(bound) is Tally:
        key, value = get_args(base)
        members = _labels(key)
        if value is int and members is not None:
            return _Kind("tally", labels=members, high=bound.maximum)
    return None


# -- refusals -----------------------------------------------------------------------------

_C = TypeVar("_C", bound=str)
_E = TypeVar("_E", bound=BaseException)


class Refusal(Exception, Generic[_C]):
    """An exception that is its category and nothing else.

    A subclass names its categories as a ``Literal``: ``class Stopped(Refusal[StopCategory])``.
    There is no message, so a refusal cannot carry text past the line that reports it.
    """

    categories: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **options: Any) -> None:
        super().__init_subclass__(**options)
        for member in ("__init__", "__str__", "__repr__"):
            if member in cls.__dict__:
                raise TypeError(f"{cls.__name__} defines {member}: a refusal is only its category")
        declared = [
            get_args(base)[0]
            for base in getattr(cls, "__orig_bases__", ())
            if get_origin(base) is not None and issubclass(get_origin(base), Refusal)
        ]
        if not declared:
            if not cls.categories:
                raise TypeError(f"{cls.__name__} must name its categories as a Literal")
            return
        if type(declared[0]) is TypeVar:
            return
        members = _labels(declared[0])
        if members is None:
            raise TypeError(f"{cls.__name__} must name its categories as a Literal of labels")
        cls.categories = members

    def __init__(self, category: _C) -> None:
        if type(category) is not str or category not in type(self).categories:
            raise TypeError(f"{type(self).__name__} was given a category it does not declare")
        super().__init__(category)
        self.category: _C = category

    def __str__(self) -> str:
        return self.category


ParseCategory = Literal[
    "not_a_line", "unknown_name", "prefix", "json", "duplicate_key", "keys", "value", "version",
    "fault",
]


class LineRefused(Refusal[ParseCategory]):
    """A line was not read: it is not exactly what its declaration writes."""


class LineFault(Exception):
    """Strict mode only: a line was given something its declaration does not allow."""


# -- exception identity -------------------------------------------------------------------


class ExceptionKinds:
    """A fixed table from exact exception type to label; any other type is ``other``."""

    def __init__(
        self, table: Mapping[type[BaseException], str], *, other: str = "other",
    ) -> None:
        for kind, label in table.items():
            if not isinstance(kind, type) or not issubclass(kind, BaseException):
                raise TypeError("an exception kind table is keyed by exception types")
            if not _is_label(label):
                raise TypeError("an exception kind is named with a label")
        if not _is_label(other):
            raise TypeError("an exception kind is named with a label")
        self._table = dict(table)
        self._other = other
        self.labels = frozenset(self._table.values()) | {other}

    def of(self, error: BaseException) -> str:
        return self._table.get(type(error), self._other)


ExceptionKind = Literal[
    "ValueError", "TypeError", "RuntimeError", "OSError", "TimeoutError", "ImportError",
    "PermissionError", "other",
]
EXCEPTION_KINDS: Final = ExceptionKinds({
    ValueError: "ValueError",
    TypeError: "TypeError",
    RuntimeError: "RuntimeError",
    OSError: "OSError",
    TimeoutError: "TimeoutError",
    ImportError: "ImportError",
    PermissionError: "PermissionError",
})


# -- the sink -----------------------------------------------------------------------------


class _State:
    lock = threading.Lock()
    dropped = 0
    strict = False
    catalogue: dict[str, type[Line]] = {}


def dropped() -> int:
    """How many lines this process could not write, or refused to write twice."""
    return _State.dropped


def _drop() -> None:
    with _State.lock:
        _State.dropped += 1


@contextmanager
def strict(enabled: bool = True) -> Iterator[None]:
    """Raise ``LineFault`` where a fault would otherwise be written as null. For tests."""
    previous = _State.strict
    _State.strict = enabled
    try:
        yield
    finally:
        _State.strict = previous


def catalogue() -> Mapping[str, type[Line]]:
    """Every declared line, by name."""
    return MappingProxyType(dict(_State.catalogue))


@contextmanager
def isolated_catalogue() -> Iterator[dict[str, type[Line]]]:
    """Declare lines into a catalogue of their own. For tests of the mechanism."""
    previous = _State.catalogue
    own: dict[str, type[Line]] = {}
    _State.catalogue = own
    try:
        yield own
    finally:
        _State.catalogue = previous


def _write(spec: _Spec, text: str, *, in_flight: bool) -> None:
    """Send one line in one call. A failure is counted and never raised over a refusal."""
    stream = getattr(sys, spec.stream)
    if stream is None:
        _drop()
        return
    try:
        stream.write(text + "\n")
        if spec.layout.flush:
            stream.flush()
    except Exception:
        _drop()
        if spec.detach and sys.stderr is stream:
            # The interpreter flushes this stream again at exit and would change the status.
            sys.stderr = None
    except BaseException:
        if not in_flight:
            raise
        _drop()


# -- lines --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """How a line is laid out in bytes, so a declaration can reproduce an existing marker.

    ``order`` lists every member in the order written, ``version`` included when the line
    has one; without it the members are sorted.
    """

    order: tuple[str, ...] | None = None
    compact: bool = True
    space: bool = True
    flush: bool = True


@dataclass(frozen=True)
class _Spec:
    name: str
    version: int | None
    stream: str
    layout: Layout
    detach: bool
    fields: Mapping[str, _Kind]
    defaults: Mapping[str, object]


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = dict(pairs)
    if len(document) != len(pairs):
        raise LineRefused("duplicate_key")
    return document


def _no_constant(_: str) -> object:
    raise ValueError


@dataclass_transform(kw_only_default=True)
class Line:
    """The base of every declared line. Declare an ``Observation`` or an ``Outcome``."""

    _spec: ClassVar[_Spec | None] = None

    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        version: Any = _MISSING,
        stream: str = "stdout",
        layout: Layout | None = None,
        detach_on_failure: bool = False,
        **options: Any,
    ) -> None:
        super().__init_subclass__(**options)
        cls._spec = None
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            raise TypeError(
                f"{cls.__name__}: a field annotation cannot be resolved. Declare each "
                "vocabulary at module level."
            ) from None
        declared = {
            field: hint for field, hint in hints.items() if get_origin(hint) is not ClassVar
        }
        if name is None:
            if declared or version is not _MISSING or layout is not None:
                raise TypeError(f"{cls.__name__} declares a line and must be given a name")
            return
        if type(name) is not str or _NAME.match(name) is None or len(name) > _NAME_LIMIT:
            raise TypeError(f"{cls.__name__}: {name!r} is not a marker name")
        if version is _MISSING:
            raise TypeError(f"{name}: state its version, or None for a line that has none")
        if version is not None and (not _exact_int(version) or version < 1):
            raise TypeError(f"{name}: a version is a positive integer or None")
        if stream not in ("stdout", "stderr"):
            raise TypeError(f"{name}: a line is written to stdout or stderr")
        if type(detach_on_failure) is not bool or (detach_on_failure and stream != "stderr"):
            raise TypeError(f"{name}: only a stderr line can detach a failed stream")
        layout = Layout() if layout is None else layout
        if type(layout) is not Layout:
            raise TypeError(f"{name}: layout must be a Layout")
        fields: dict[str, _Kind] = {}
        defaults: dict[str, object] = {}
        for field, hint in declared.items():
            kind = _kind(hint)
            reserved = field in _RESERVED or hasattr(Line, field) or hasattr(Outcome, field)
            if _FIELD.match(field) is None or reserved or field in vars(Observation):
                raise TypeError(f"{name}: {field} cannot be a field name")
            if kind is None:
                raise TypeError(
                    f"{name}: field {field} is not an evidence kind. A field is a Literal of "
                    "labels, a bool, a bounded int, a digest, or a bounded list or tally of labels."
                )
            fields[field] = kind
            if field in cls.__dict__:
                try:
                    defaults[field] = kind.admit(cls.__dict__[field], saturate=False)
                except _Invalid:
                    raise TypeError(f"{name}: the default of {field} is not a value") from None
        members = set(fields) | ({"version"} if version is not None else set())
        if layout.order is not None and (
            len(layout.order) != len(members) or set(layout.order) != members
        ):
            raise TypeError(f"{name}: layout order must list every member exactly once")
        cls._spec = _Spec(
            name, version, stream, layout, detach_on_failure,
            MappingProxyType(fields), MappingProxyType(defaults),
        )
        cls._settle_declaration()
        if name in _State.catalogue:
            cls._spec = None
            raise TypeError(f"{name} is already declared by {_State.catalogue[name].__name__}")
        _State.catalogue[name] = cls

    @classmethod
    def _settle_declaration(cls) -> None:
        """Checked by ``Outcome``; an observation has nothing more to declare."""

    @classmethod
    def _declared(cls) -> _Spec:
        if cls._spec is None:
            raise TypeError(f"{cls.__name__} does not declare a line")
        return cls._spec

    def __init__(self, **values: object) -> None:
        spec = type(self)._declared()
        state = self.__dict__
        state["_faults"] = set()
        state["_extra"] = 0
        state["_written"] = False
        state["_recorded"] = set()
        state["_report"] = False
        for field in spec.fields:
            if field in values:
                self._assign(field, values[field])
            elif field in spec.defaults:
                state[field] = spec.defaults[field]
            else:
                self._fault(field)
        for field in values:
            if field not in spec.fields:
                self._stray(field)
        state["_recorded"] = set()

    # -- values -----------------------------------------------------------------------

    def _fault(self, field: str) -> None:
        spec = type(self)._declared()
        if _State.strict:
            raise LineFault(f"{spec.name}.{field} was given a value outside its kind")
        self.__dict__[field] = None
        self.__dict__["_faults"].add(field)

    def _stray(self, field: str) -> None:
        if _State.strict:
            raise LineFault(f"{type(self)._declared().name}.{field} is not a declared field")
        self.__dict__["_extra"] += 1

    def _assign(self, field: str, value: object) -> None:
        kind = type(self)._declared().fields[field]
        try:
            stored = kind.admit(value, saturate=True)
        except _Invalid:
            self._fault(field)
            return
        self.__dict__[field] = stored
        self.__dict__["_faults"].discard(field)
        self.__dict__["_recorded"].add(field)

    if not TYPE_CHECKING:
        # Hidden from the type checker: a visible ``__setattr__`` would make it accept an
        # assignment to any name, and an undeclared field must be a type error.
        def __setattr__(self, field: str, value: object) -> None:
            if field in type(self)._declared().fields:
                self._assign(field, value)
            else:
                self._stray(field)

    @property
    def fault(self) -> int:
        """How many values this line was given that it could not hold."""
        return len(self.__dict__["_faults"]) + int(self.__dict__["_extra"])

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        fields = type(self)._declared().fields
        return self.fault == other.fault and all(
            self.__dict__[field] == other.__dict__[field] for field in fields
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.encode()}>"

    # -- bytes ------------------------------------------------------------------------

    def encode(self) -> str:
        """The line as written, without its newline."""
        spec = type(self)._declared()
        members: dict[str, object] = {}
        for field in spec.fields:
            value = self.__dict__[field]
            members[field] = list(value) if type(value) is tuple else value
        if spec.version is not None:
            members["version"] = spec.version
        if spec.layout.order is None:
            if self.fault:
                members["fault"] = self.fault
            ordered = {key: members[key] for key in sorted(members)}
        else:
            ordered = {key: members[key] for key in spec.layout.order}
            if self.fault:
                ordered["fault"] = self.fault
        separators = (",", ":") if spec.layout.compact else (", ", ": ")
        body = json.dumps(ordered, separators=separators, allow_nan=False)
        return f"[{spec.name}]" + (" " if spec.layout.space else "") + body

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read exactly what ``encode`` writes; anything else refuses the whole line."""
        spec = cls._declared()
        prefix = f"[{spec.name}]" + (" " if spec.layout.space else "")
        if type(text) is not str or not text.startswith(prefix):
            raise LineRefused("prefix")
        body = text[len(prefix):]
        if not spec.layout.space and body.startswith(" "):
            raise LineRefused("prefix")
        if not body.startswith("{") or not body.endswith("}"):
            raise LineRefused("json")
        try:
            document = json.loads(
                body, object_pairs_hook=_unique, parse_constant=_no_constant,
                parse_float=_no_constant,
            )
        except LineRefused:
            raise
        except (ValueError, RecursionError):
            raise LineRefused("json") from None
        if type(document) is not dict:
            raise LineRefused("json")
        expected = set(spec.fields) | ({"version"} if spec.version is not None else set())
        if set(document) - {"fault"} != expected:
            raise LineRefused("keys")
        if spec.version is not None and (
            type(document["version"]) is not int or document["version"] != spec.version
        ):
            raise LineRefused("version")
        line = cls.__new__(cls)
        state = line.__dict__
        state.update(_faults=set(), _extra=0, _written=False, _recorded=set(), _report=False)
        nulls: list[str] = []
        for field, kind in spec.fields.items():
            value = document[field]
            if value is None and not kind.optional:
                nulls.append(field)
                state[field] = None
                continue
            try:
                state[field] = kind.admit(value, saturate=False)
            except _Invalid:
                raise LineRefused("value") from None
        if "fault" not in document:
            if nulls:
                raise LineRefused("value")
            return line
        # A fault is one per value the line could not hold: at least one per null that its
        # kind does not allow, and more when undeclared fields were refused as well.
        count = document["fault"]
        if type(count) is not int or count < max(1, len(nulls)):
            raise LineRefused("fault")
        state["_faults"] = set(nulls)
        state["_extra"] = count - len(nulls)
        return line

    def _send(self, *, in_flight: bool) -> None:
        if self.__dict__["_written"]:
            if _State.strict:
                raise LineFault(f"{type(self)._declared().name} was written twice")
            _drop()
            return
        self.__dict__["_written"] = True
        _write(type(self)._declared(), self.encode(), in_flight=in_flight)


class Observation(Line):
    """A line that reports something seen. It is written where it is made."""

    def emit(self) -> None:
        self._send(in_flight=False)


class _Silent:
    def __repr__(self) -> str:
        return "SILENT"


SILENT: Final = _Silent()

_O = TypeVar("_O", bound="Outcome")


@dataclass(frozen=True)
class _Settlement:
    category: str
    unclassified: str
    on_cancel: str | None
    always: bool
    admits: tuple[type[Refusal[Any]], ...]
    kind: str | None
    kinds: ExceptionKinds


class Outcome(Line):
    """A line that reports how a guard or a bounded region ended.

    It has no free-standing emit. ``raising`` writes it at a guard and hands back the
    exception to raise; ``scope`` writes it once when a region exits.
    """

    _settlement: ClassVar[_Settlement | None] = None
    _pending: ClassVar[dict[str, Any] | None] = None

    def __init_subclass__(
        cls,
        *,
        category: str = "category",
        unclassified: Any = _MISSING,
        on_cancel: Any = _MISSING,
        when: str = "refusal",
        admits: tuple[type[Refusal[Any]], ...] = (),
        kind: str | None = None,
        kinds: ExceptionKinds = EXCEPTION_KINDS,
        **options: Any,
    ) -> None:
        cls._pending = {
            "category": category, "unclassified": unclassified, "on_cancel": on_cancel,
            "when": when, "admits": admits, "kind": kind, "kinds": kinds,
        }
        try:
            super().__init_subclass__(**options)
        finally:
            cls._pending = None

    @classmethod
    def _settle_declaration(cls) -> None:
        spec, given = cls._declared(), cls._pending or {}
        name = spec.name
        field = given["category"]
        chosen = spec.fields.get(field)
        if chosen is None or chosen.shape != "label":
            raise TypeError(f"{name}: its category field {field!r} must be a field of labels")
        unclassified, on_cancel = given["unclassified"], given["on_cancel"]
        if unclassified is _MISSING or unclassified not in chosen.labels:
            raise TypeError(f"{name}: declare the label an unclassified failure records")
        if on_cancel is _MISSING or (on_cancel is not SILENT and on_cancel not in chosen.labels):
            raise TypeError(f"{name}: declare what cancellation records, a label or SILENT")
        if given["when"] not in ("always", "refusal"):
            raise TypeError(f"{name}: when is 'always' or 'refusal'")
        admits = given["admits"]
        if type(admits) is not tuple:
            raise TypeError(f"{name}: admits is a tuple of refusal types")
        for refusal in admits:
            if not isinstance(refusal, type) or not issubclass(refusal, Refusal):
                raise TypeError(f"{name}: admits holds refusal types only")
            if not refusal.categories or not refusal.categories <= chosen.labels:
                raise TypeError(
                    f"{name}: {refusal.__name__} has a category that {field} does not list"
                )
        kind, kinds = given["kind"], given["kinds"]
        if type(kinds) is not ExceptionKinds:
            raise TypeError(f"{name}: kinds is an ExceptionKinds table")
        if kind is not None:
            holder = spec.fields.get(kind)
            if kind == field or holder is None or holder.shape != "label":
                raise TypeError(f"{name}: its kind field {kind!r} must be another label field")
            if holder.labels != kinds.labels:
                raise TypeError(f"{name}: {kind} must list exactly the kinds of its table")
            if not holder.optional and kind not in spec.defaults:
                raise TypeError(f"{name}: {kind} is filled at exit, so it needs a default")
        cls._settlement = _Settlement(
            field, unclassified, None if on_cancel is SILENT else on_cancel,
            given["when"] == "always", admits, kind, kinds,
        )

    @classmethod
    def _rules(cls) -> _Settlement:
        if cls._settlement is None:
            raise TypeError(f"{cls.__name__} does not declare a line")
        return cls._settlement

    def report(self) -> None:
        """Ask for this line to be written even though its region ends normally."""
        self.__dict__["_report"] = True

    def _settle(self, error: BaseException | None, *, guard: bool) -> bool:
        """Fill what the exit decides and say whether the line is written. No I/O."""
        rules = type(self)._rules()
        if error is None:
            return rules.always or bool(self.__dict__["_report"])
        if isinstance(error, Refusal):
            if type(error) in rules.admits:
                self._assign(rules.category, error.category)
            else:
                self._fault(rules.category)
        elif not isinstance(error, Exception):
            if rules.on_cancel is None:
                return False
            self._assign(rules.category, rules.on_cancel)
        elif not guard and rules.category not in self.__dict__["_recorded"]:
            self._assign(rules.category, rules.unclassified)
        if rules.kind is not None:
            self._assign(rules.kind, rules.kinds.of(error))
        return True

    def raising(self, error: _E) -> _E:
        """Write this line for a guard that rejects, and return ``error`` to raise."""
        try:
            write = self._settle(error, guard=True)
        except LineFault:
            raise
        except Exception:
            _drop()
            return error
        if write:
            self._send(in_flight=False)
        return error

    def scope(self) -> _Scope[Self]:
        """A region that writes this line exactly once, when it exits."""
        return _Scope(self)


class _Scope(Generic[_O]):
    def __init__(self, line: _O) -> None:
        self._line = line
        self._done = False

    def __enter__(self) -> _O:
        self._line.__dict__["_recorded"] = set()
        return self._line

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> Literal[False]:
        if self._done:
            return False
        self._done = True
        try:
            write = self._line._settle(error, guard=False)
        except LineFault:
            raise
        except Exception:
            _drop()
            return False
        if write:
            self._line._send(in_flight=error is not None)
        return False


def parse_line(text: str, known: Mapping[str, type[Line]] | None = None) -> Line:
    """Read any declared line. An undeclared name is refused, never read loosely."""
    match = _LINE.match(text) if type(text) is str else None
    if match is None:
        raise LineRefused("not_a_line")
    declaration = (_State.catalogue if known is None else known).get(match.group(1))
    if declaration is None:
        raise LineRefused("unknown_name")
    return declaration.parse(text)
